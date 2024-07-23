import os
import sys
sys.path.append('./bigdetection')
import math
import argparse
import pickle
from tqdm import tqdm

import mmcv
import torch

from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.apis import (async_inference_detector, inference_detector, init_detector, show_result_pyplot)

PERSON_BIGDET_IDX = 383
SKATEBOARD_IDX = 464
TENNIS_IDX = 535
BASEBALL_IDX = 32
BASKETBALL_IDX = 22
CELLO_IDX = 104
SURFBOARD_IDX = 515
VIOLIN_IDX = 574
YOGABALL_IDX = 22
FLUTE_IDX = 207
BICYCLE_IDX = 47
GUITAR_IDX = 244
BARBELL_IDX = 175
GOLF_IDX = 233

PERSON_COCO_IDX = 0
SKATEBOARD_COCO_IDX = 36
TENNIS_COCO_IDX = 38
BASEBALL_COCO_IDX = 34
SURFBOARD_COCO_IDX = 37
BASKETBALL_COCO_IDX = 32
BICYCLE_COCO_IDX = 1
YOGABALL_COCO_IDX = 32
UMBRELLA_COCO_IDX = 25

device = torch.device('cuda')


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


class ImageDataset():

    def __init__(self, image_list, transforms):
        self.image_list = image_list
        self.transforms = transforms


    def __len__(self, ):
        return len(self.image_list)


    def __getitem__(self, idx):
        data = self.transforms(
            dict(img_info=dict(filename=self.image_list[idx]), img_prefix=None))
        # print(data)
        # exit(0)
        image = data['img'][0].data
        image_metas = {k: v[0] for k, v in data.items() if k != 'img'}
        
        return image, image_metas

def collate_fn(items):
    images = torch.stack([item[0] for item in items], dim=0)
    image_metas = [item[1] for item in items]

    return images, image_metas


def begin_detection(args, model, dataloader):

    root_dir = args.root_dir
    object_name = root_dir.split('/')[-1]
    if object_name == 'skateboard':
        OBJECT_IDX = SKATEBOARD_COCO_IDX
        PERSON_IDX = PERSON_COCO_IDX
    elif object_name == 'tennis':
        OBJECT_IDX = TENNIS_COCO_IDX
        PERSON_IDX = PERSON_COCO_IDX
    elif object_name == 'baseball':
        OBJECT_IDX = BASEBALL_COCO_IDX
        PERSON_IDX = PERSON_COCO_IDX
    elif object_name == 'basketball':
        OBJECT_IDX = BASKETBALL_COCO_IDX
        PERSON_IDX = PERSON_COCO_IDX
    elif object_name == 'cello':
        OBJECT_IDX = CELLO_IDX
        PERSON_IDX = PERSON_BIGDET_IDX
    elif object_name == 'surfboard':
        OBJECT_IDX = SURFBOARD_COCO_IDX
        PERSON_IDX = PERSON_COCO_IDX
    elif object_name == 'violin':
        OBJECT_IDX = VIOLIN_IDX
        PERSON_IDX = PERSON_BIGDET_IDX
    elif object_name == 'yogaball':
        OBJECT_IDX = YOGABALL_COCO_IDX
        PERSON_IDX = PERSON_COCO_IDX
    elif object_name == 'flute':
        OBJECT_IDX = FLUTE_IDX
        PERSON_IDX = PERSON_BIGDET_IDX
    elif object_name == 'bicycle':
        OBJECT_IDX = BICYCLE_COCO_IDX
        PERSON_IDX = PERSON_COCO_IDX
    elif object_name == 'guitar':
        OBJECT_IDX = GUITAR_IDX
        PERSON_IDX = PERSON_BIGDET_IDX
    elif object_name == 'barbell':
        OBJECT_IDX = BARBELL_IDX
        PERSON_IDX = PERSON_BIGDET_IDX
    elif object_name == 'golf':
        OBJECT_IDX = GOLF_IDX
        PERSON_IDX = PERSON_BIGDET_IDX
    elif object_name == 'umbrella':
        OBJECT_IDX = UMBRELLA_COCO_IDX
        PERSON_IDX = PERSON_COCO_IDX

    results_all = {}

    for data in tqdm(dataloader):
        images, image_metas = data
        images = images.to(device)
        torch.set_grad_enabled(False)
        results = model(return_loss=False, rescale=True, img=[images,], img_metas=[image_metas,])

        for image_meta, result in zip(image_metas, results):
            frame_id = image_meta['filename'].split('/')[-1].split('.')[0]
            object_dict = {}
            if len(result[PERSON_IDX]) == 0:
                object_dict['person'] = None
            else:
                object_dict['person'] = result[PERSON_IDX]

            if len(result[OBJECT_IDX]) == 0:
                object_dict[object_name] = None
            else:
                object_dict[object_name] = result[OBJECT_IDX]
            results_all[frame_id] = object_dict
    return results_all


def inference_all(args):

    root_dir = args.root_dir
    object_name = root_dir.split('/')[-1]
    image_dir = os.path.join(root_dir, 'images_temp')

    output_dir = os.path.join(root_dir, 'bigdetection_temp')
    os.makedirs(output_dir, exist_ok=True)

    if object_name in ['tennis', 'skateboard', 'baseball', 'bicycle', 'surfboard', 'basketball', 'yogaball', 'umbrella']:
        model = init_detector('./bigdetection/configs/BigDetection/cbnetv2/htc_cbv2_swin_base_giou_4conv1f_adamw_bigdet_coco.py', 
            './bigdetection/checkpoints/htc_cbv2_swin_base_giou_4conv1f_bigdet_coco-ft_20e.pth', device=device)
    else:
        model = init_detector('./bigdetection/configs/BigDetection/cbnetv2/htc_cbv2_swin_base_giou_4conv1f_adamw_bigdet.py', 
            './bigdetection/checkpoints/htc_cbv2_swin_base_giou_4conv1f_bigdet.pth', device=device)
    cfg = model.cfg
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    for video_idx in range(args.begin_idx, args.end_idx):
        interval_files = os.path.join(output_dir, '{:04d}_{}_intervals.pkl'.format(video_idx, args.interval))
        if os.path.exists(interval_files):
            if os.path.exists(os.path.join(output_dir, '{:04d}_all.pkl'.format(video_idx))):
                print('skip video {}.'.format(video_idx))
                continue
            intervals = load_pickle(interval_files)
            print('Found {} intervals.'.format(len(intervals)))
            results_all = []
            for interval in intervals:
                begin_frame_idx = int(interval[0])
                end_frame_idx = int(interval[-1])
                image_files = []
                for frame_idx in range(begin_frame_idx, end_frame_idx):
                    image_files.append(os.path.join(image_dir, '{:04d}'.format(video_idx), '{:06d}.jpg'.format(frame_idx)))

                dataset = ImageDataset(image_files, test_pipeline)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.batch_size // 2, shuffle=False, collate_fn=collate_fn)

                results = begin_detection(args, model, dataloader)
                results_all.append(results)
            save_pickle(os.path.join(output_dir, '{:04d}_all.pkl'.format(video_idx)), results_all)

        else:
            image_files = sorted(os.listdir(os.path.join(image_dir, '{:04d}'.format(video_idx))))[::args.interval]
            image_files = [os.path.join(image_dir, '{:04d}'.format(video_idx), file) for file in image_files]


            dataset = ImageDataset(image_files, test_pipeline)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.batch_size // 2, shuffle=False, collate_fn=collate_fn)

            results_all = begin_detection(args, model, dataloader)
            save_pickle(os.path.join(output_dir, '{:04d}_{}.pkl'.format(video_idx, args.interval)), results_all)

        print('Video {:04d} done !'.format(video_idx))


if __name__ == '__main__':
# envs: bigdet
    parser = argparse.ArgumentParser(description='BigDetection Inference.')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    parser.add_argument('--begin_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    parser.add_argument('--interval', type=int)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    inference_all(args)
