import os
import cv2
import pickle
import json
from glob import glob
import numpy as np
import random
from torchvision import transforms


def rotate_2d(pt_2d, rot_rad):
    x, y = pt_2d
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(box_center_x, box_center_y, box_size, out_size, rot):
    src_w = src_h = box_size
    rot_rad = np.pi * rot / 180
    src_center = np.array([box_center_x, box_center_y], dtype=np.float32)
    src_rightdir = src_center + rotate_2d(np.array([0, src_w * 0.5], dtype=np.float32), rot_rad)
    src_downdir = src_center + rotate_2d(np.array([src_h * 0.5, 0], dtype=np.float32), rot_rad)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_rightdir
    src[2, :] = src_downdir

    dst = np.array([[out_size / 2, out_size / 2], [out_size / 2, out_size], [out_size, out_size / 2]], dtype=np.float32)
    trans = cv2.getAffineTransform(src, dst)
    return trans


def generate_image_patch(image, box_center_x, box_center_y, box_size, out_size, rot, color_scale):

    img_trans = gen_trans_from_patch_cv(box_center_x, box_center_y, box_size, out_size, rot)
    img_patch = cv2.warpAffine(image, img_trans, (int(out_size), int(out_size)), flags=cv2.INTER_LINEAR)
    return img_patch, img_trans


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


class ObjectImageDataset:

    def __init__(self, root_dir, video_ids, out_res=256, coor_res=64):
        self.dataset_root_dir = root_dir
        self.out_res = out_res
        self.coor_res = coor_res

        self.items = []
        for file in os.listdir(os.path.join(self.dataset_root_dir, 'hoi_tracking')):

            video_id = file.split('_')[0]
            # if int(video_id) != 426:
            #     continue
            if video_id not in video_ids:
                continue
            tracking_results = load_pickle(os.path.join(self.dataset_root_dir, 'hoi_tracking', file))
            hoi_instances = tracking_results['hoi_instances']
            random.shuffle(hoi_instances)
            for hoi_instance in hoi_instances: #[:3]:
                hoi_id = hoi_instance['hoi_id']

                # if int(hoi_id) != 29:
                #     continue

                sequences = hoi_instance['sequences']
                random.shuffle(sequences)
                for item in sequences: # [:256]:
                    frame_id = item['frame_id']

                    # if frame_id != '011407':
                    #     continue

                    rgb_path = os.path.join(self.dataset_root_dir, 'images_temp', video_id, '{}.jpg'.format(frame_id))
                    mask_path = os.path.join(self.dataset_root_dir, 'hoi_mask', video_id, hoi_id, '{}.pkl'.format(frame_id))

                    self.items.append((video_id, hoi_id, frame_id, rgb_path, mask_path))
        # self.items = self.items[:8]
        random.shuffle(self.items)
        print('Find {} instances.'.format(len(self.items)))

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


    def __len__(self, ):
        return len(self.items)


    def __getitem__(self, idx):
        video_id, hoi_id, frame_id, rgb_path, mask_path = self.items[idx]
        item_id = '_'.join([video_id, hoi_id, frame_id])

        image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_h, img_w, _ = image.shape

        mask_load = load_pickle(mask_path)
        if mask_load['object']['mask'] is None:
            cx, cy = img_w / 2, img_h / 2
            s = min(img_h, img_w)
        else:
            mask_h, mask_w = mask_load['object']['mask_shape']
            x1, y1, x2, y2 = mask_load['object']['mask_box']
            cx, cy = (x1 + x2) / 2, (y1  + y2) / 2
            s = max((y2 - y1), (x2 - x1)) * 1.5

        mask = np.zeros((img_h, img_w))
        if mask_load['object']['mask'] is not None:
            mask_h, mask_w = mask_load['object']['mask_shape']
            x1, y1, x2, y2 = mask_load['object']['mask_box']
            mask[y1:y2+1, x1:x2+1] = np.unpackbits(mask_load['object']['mask'])[:mask_h * mask_w].reshape(mask_h, mask_w)

        rot, color_scale = 0., [1., 1., 1.]

        img_patch, img_trans = generate_image_patch(image, cx, cy, s, self.out_res, rot, color_scale)
        mask_patch, img_trans = generate_image_patch(mask, cx, cy, s, self.out_res, rot, color_scale)

        img_patch = img_patch.astype(np.float32).transpose((2, 0, 1))
        for n_c in range(3):
            img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255) / 255.
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        mask_patch = mask_patch.astype(np.float32)

        cx /= 4
        cy /= 4
        s /= 4

        return img_patch, mask_patch, np.array([cx, cy]), s, item_id
