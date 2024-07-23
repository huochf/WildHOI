"""Image demo script."""
import argparse
import os
import sys
sys.path.append('/public/home/huochf/projects/3D_HOI/hoiYouTube/05_smpler_x/SMPLer-X/main/')
sys.path.append('/public/home/huochf/projects/3D_HOI/hoiYouTube/05_smpler_x/SMPLer-X/data/')
sys.path.append('/public/home/huochf/projects/3D_HOI/hoiYouTube/05_smpler_x/SMPLer-X/common/')
import json
import pickle
import cv2
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from easydict import EasyDict as edict
from torchvision import transforms as T
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
from scipy.spatial.transform import Rotation

pretrained_model = 'smpler_x_h32'
testset = 'EHF'
agora_benchmark = 'agora_model'
num_gpus = 1
exp_name = 'inference'

from config import cfg
config_path = os.path.join('./SMPLer-X/main/config', f'config_{pretrained_model}.py')
ckpt_path = os.path.join('./SMPLer/pretrained_models', f'{pretrained_model}.pth.tar')

cfg.get_config_fromfile(config_path)
cfg.update_test_config(testset, agora_benchmark, shapy_eval_split=None, 
                        pretrained_model_path=ckpt_path, use_cache=False)
cfg.update_config(num_gpus, exp_name)
cfg.encoder_config_file = '/public/home/huochf/projects/3D_HOI/hoiYouTube/05_smpler_x/SMPLer-X/main/transformer_utils/configs/smpler_x/encoder/body_encoder_huge.py'
cfg.pretrained_model_path = '/public/home/huochf/projects/3D_HOI/hoiYouTube/05_smpler_x/SMPLer-X/pretrained_models/smpler_x_h32.pth.tar'
cudnn.benchmark = True

from utils.preprocessing import load_img, process_bbox, generate_patch_image


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


class ImageDataset():

    def __init__(self, image_dir, sequences):
        self.image_dir = image_dir
        self.sequences = sequences
        print('loaded {} frames.'.format(len(self.sequences)))

        person_bboxes = np.zeros((len(self.sequences), 5))
        for idx, item in enumerate(self.sequences):
            if item['person_bbox'] is not None:
                person_bboxes[idx, :4] = item['person_bbox']
                person_bboxes[idx, 4] = 1

        self.person_bboxes = self.smooth_bboxes(person_bboxes)
        self.input_img_shape = (512, 384)
        self.transform = transforms.ToTensor()


    def smooth_bboxes(self, boxes, windows=5):
        boxes = np.array(boxes) # [n, 5]
        n, d = boxes.shape

        sequence = np.concatenate([np.zeros((windows // 2, d)), boxes, np.zeros((windows // 2, d))], axis=0)
        confidence = sequence[:, 4:]
        smooth_boxes = np.stack([
            sequence[i:n+i] for i in range(windows)], axis=0)
        confidence = np.stack([
            confidence[i:n+i] for i in range(windows)], axis=0)

        smooth_boxes = (confidence * smooth_boxes).sum(0) / (confidence.sum(0) + 1e-8)
        return smooth_boxes


    def __len__(self, ):
        return len(self.sequences)


    def __getitem__(self, idx):
        item = self.sequences[idx]
        frame_id = item['frame_id']
        person_bbox = self.person_bboxes[idx]

        image_path = os.path.join(self.image_dir, '{}.jpg'.format(frame_id))
        original_img = load_img(image_path)
        original_img_height, original_img_width = original_img.shape[:2]
        box_xywh = np.zeros((4))
        box_xywh[0] = person_bbox[0]
        box_xywh[1] = person_bbox[1]
        box_xywh[2] = person_bbox[2] - person_bbox[0]
        box_xywh[3] = person_bbox[3] - person_bbox[1]

        start_point = (int(box_xywh[0]), int(box_xywh[1]))
        end_point = (int(box_xywh[2]), int(box_xywh[3]))
        bbox = process_bbox(box_xywh, original_img_width, original_img_height)
        if bbox is None:
            ratio = 1.25
            w = box_xywh[2]
            h = box_xywh[3]
            c_x = box_xywh[0] + w / 2.
            c_y = box_xywh[1] + h / 2.
            aspect_ratio = self.input_img_shape[1] / self.input_img_shape[0]
            if w > aspect_ratio * h:
                h = w / aspect_ratio
            elif w < aspect_ratio * h:
                w = h * aspect_ratio
            bbox = np.zeros((4))
            bbox[2] = w * ratio
            bbox[3] = h * ratio
            bbox[0] = c_x - box_xywh[2] / 2.
            bbox[1] = c_y - box_xywh[3] / 2.
            bbox = bbox.astype(np.float32)

        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, self.input_img_shape)
        img = self.transform(img.astype(np.float32)) / 255

        img_orig_shape = np.array([original_img_height, original_img_width], dtype=np.float32)

        return frame_id, img, bbox, img_orig_shape


def inference_all(args):

    device = torch.device('cuda')
    batch_size = 8
    f = (5000, 5000)
    input_body_shape = (256, 192)
    princpt = (input_body_shape[1] / 2, input_body_shape[0] / 2)

    from base import Demoer
    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()

    tracking_results_dir = os.path.join(args.root_dir, 'hoi_tracking')
    image_dir = os.path.join(args.root_dir, 'images_temp')

    for video_idx in range(args.begin_idx, args.end_idx):
        video_id = '{:04d}'.format(video_idx)
        try:
            tracking_results = load_pickle(os.path.join(tracking_results_dir, '{}_tracking.pkl'.format(video_id)))
        except:
            print('skip video {}.'.format(video_id))
            continue

        smpl_out_dir = os.path.join(args.root_dir, 'smpler_x', video_id)
        os.makedirs(smpl_out_dir, exist_ok=True)
        print('Found {} instances.'.format(len(tracking_results['hoi_instances'])))

        for hoi_instance in tracking_results['hoi_instances']:
            hoi_id = hoi_instance['hoi_id']
            # if os.path.exists(os.path.join(smpl_out_dir, '{}_smplx.pkl'.format(hoi_id))):
            #     continue

            smpl_params_all = []

            dataset = ImageDataset(os.path.join(image_dir, video_id), hoi_instance['sequences'])
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False)
            for items in tqdm(dataloader):
                frame_ids, images, box_xywh, img_orig_shapes = items
                b = images.shape[0]
                inputs = {'img': images.to(device)}
                targets = {}
                meta_info = {}

                with torch.no_grad():
                    output = demoer.model(inputs, targets, meta_info, 'test')

                for idx, frame_id in enumerate(frame_ids):
                    smplx_params = {}
                    focal = [f[0] / input_body_shape[1] * box_xywh[idx, 2], f[1] / input_body_shape[0] * box_xywh[idx, 3]]
                    _princpt = [princpt[0] / input_body_shape[1] * box_xywh[idx, 2] + box_xywh[idx, 0], princpt[1] / input_body_shape[0] * box_xywh[idx, 3] + box_xywh[idx, 1]]
                    smplx_params['frame_id'] = frame_id
                    smplx_params['focal'] = focal
                    smplx_params['princpt'] = _princpt
                    smplx_params['bbox'] = box_xywh[idx].numpy()
                    smplx_params['global_orient'] = output['smplx_root_pose'].reshape(b, -1, 3)[idx].detach().cpu().numpy()
                    smplx_params['body_pose'] = output['smplx_body_pose'].reshape(b, -1, 3)[idx].detach().cpu().numpy()
                    smplx_params['left_hand_pose'] = output['smplx_lhand_pose'].reshape(b, -1, 3)[idx].detach().cpu().numpy()
                    smplx_params['right_hand_pose'] = output['smplx_rhand_pose'].reshape(b, -1, 3)[idx].detach().cpu().numpy()
                    smplx_params['jaw_pose'] = output['smplx_jaw_pose'].reshape(b, -1, 3)[idx].detach().cpu().numpy()
                    smplx_params['betas'] = output['smplx_shape'].reshape(b, -1, 10)[idx].detach().cpu().numpy()
                    smplx_params['expression'] = output['smplx_expr'].reshape(b, -1, 10)[idx].detach().cpu().numpy()
                    smplx_params['transl'] = output['cam_trans'].reshape(b, -1, 3)[idx].detach().cpu().numpy()

                    smpl_params_all.append(smplx_params)

                save_pickle(os.path.join(smpl_out_dir, '{}_smplx.pkl'.format(hoi_id)), smpl_params_all)

        print('Video {:04d} done!'.format(video_idx))


if __name__ == '__main__':
    # envs: smplerx
    parser = argparse.ArgumentParser(description='SMPLer-X')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    parser.add_argument('--begin_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    args = parser.parse_args()

    inference_all(args)
