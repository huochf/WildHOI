import os
import cv2
import pickle
import json
from glob import glob
import numpy as np
import random
from torchvision import transforms


def get_augmentation_params():
    tx = np.random.randn() / 3 * 0.1
    ty = np.random.randn() / 3 * 0.1
    rot = np.random.randn() / 3 * 30 if np.random.random() < 0.5 else 0
    scale = 1. + np.random.randn() / 3 * 0.3
    c_up = 1. + 0.2
    c_low = 1. - 0.2
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    return tx, ty, rot, scale, color_scale


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


def trans_keypoints(keypoint_2d, trans):
    src_pt = np.concatenate([keypoint_2d, np.ones((len(keypoint_2d), 1))], axis=1).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2].T


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_keypoints(file):
    with open(file, 'r') as f:
        all_lines = f.readlines()
    if int(all_lines[0]) == 0 or len(all_lines) != 13:
        return np.zeros((12, 3))
    keypoints = []
    for i in range(12):
        line = all_lines[i + 1]
        x, y = line.split(' ')
        x, y = float(x), float(y)
        if x > 1 or y > 1:
            keypoints.append([0, 0, 0])
        else:
            keypoints.append([x, y, 1])

    keypoints = np.array(keypoints)

    item_id = file.split('/')[-1].split('.')[0]
    video_id = item_id[:5]
    seg_id = item_id[6:-6]
    frame_id = item_id[-5:]

    mask_path = os.path.join(os.path.dirname(file).replace('object_keypoints', 'hoi_mask'), video_id, seg_id, '{}.pkl'.format(frame_id))
    mask_load = load_pickle(mask_path)
    x1, y1, x2, y2 = mask_load['object']['mask_box']
    cx, cy = (x1 + x2) / 2, (y1  + y2) / 2
    s = max((y2 - y1), (x2 - x1))
    _x1 = int(cx - s)
    _x2 = int(cx + s)
    _y1 = int(cy - s)
    _y2 = int(cy + s)

    x = _x1 + keypoints[:, 0] * 2 * s
    y = _y1 + keypoints[:, 1] * 2 * s
    keypoints = np.stack([x, y, keypoints[:, 2]], axis=1)

    return keypoints

CORR_NORM = {
    'skateboard': [0.19568036, 0.10523215, 0.77087334],
    'tennis': [0.27490701, 0.68580002, 0.03922762],
    'cello': [0.47329763, 0.25910739, 1.40221876],
    'basketball': [0.24651748, 0.24605956, 0.24669009],
    'baseball': [0.07836291, 0.07836725, 1.0668    ],
    'barbell': [2.19961905, 0.4503098,  0.45047669],
    'yogaball': [0.74948615, 0.74948987, 0.75054681],
    'bicycle': [0.76331246, 1.025965,   1.882407],
    'violin': [0.2292318, 0.1302547, 0.61199997],
}


def extract_bbox_from_mask(mask):
    try:
        indices = np.array(np.nonzero(np.array(mask)))
        y1 = np.min(indices[0, :])
        y2 = np.max(indices[0, :])
        x1 = np.min(indices[1, :])
        x2 = np.max(indices[1, :])

        return np.array([x1, y1, x2, y2])
    except:
        return None


class ObjCorrDataset:

    def __init__(self, root_dir, corr_dir, out_res=256, coor_res=64):
        object_name = root_dir.split('/')[-1]
        self.object_name = object_name
        self.corr_norm = CORR_NORM[object_name]
        self.dataset_root_dir = root_dir
        self.corr_dir = corr_dir

        self.out_res = out_res
        self.coor_res = coor_res

        all_frames = os.path.join(self.corr_dir, '*-box.txt')
        self.items = []
        for frame in glob(all_frames):
            item_id = frame.split('/')[-1].split('-')[0]

            video_id, frame_id, instance_id = item_id.split('_')

            box_path = frame
            coor_path = box_path.replace('box.txt', 'corr.pkl')
            rgb_path = os.path.join(self.dataset_root_dir, 'images_temp', video_id, '{}.jpg'.format(frame_id))

            self.items.append((item_id, box_path, coor_path, rgb_path))
        # self.items = self.items[:8]
        print('Find {} frames.'.format(len(self.items)))

        self.boxes_bigdet = self.load_bigdet_boxes(self.items)

        # with open('datasets/{}_negative_list.json'.format(object_name), 'r') as f:
        #     self.negative_img_list = json.load(f)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


    def load_bigdet_boxes(self, items):
        bigdet_boxes = {}
        for item in items:
            item_id = item[0]
            video_id, frame_id, hoi_id = item_id.split('_')
            mask_path = os.path.join(self.dataset_root_dir, 'hoi_mask', video_id, hoi_id, '{}.pkl'.format(frame_id))
            mask_load = load_pickle(mask_path)
            if mask_load['object']['mask'] is None:
                cx, cy = img_w / 2, img_h / 2
                s = min(img_h, img_w)
            else:
                mask_h, mask_w = mask_load['object']['mask_shape']
                x1, y1, x2, y2 = mask_load['object']['mask_box']
                cx, cy = (x1 + x2) / 2, (y1  + y2) / 2
                s = max((y2 - y1), (x2 - x1)) * 1.5
            bigdet_boxes[item_id] = (cx, cy, s)
        return bigdet_boxes


    def __len__(self, ):
        return len(self.items)


    def __getitem__(self, idx):
        item_id, box_path, coor_path, rgb_path = self.items[idx]

        image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_h, img_w, _ = image.shape

        mask_path = coor_path.replace('corr.pkl', 'label.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        with open(coor_path, 'rb') as f:
            coor_load = pickle.load(f)
        u = coor_load['u']
        l = coor_load['l']
        h = coor_load['h']
        w = coor_load['w']
        coor = np.zeros((img_h, img_w, 3)).astype(np.float32)
        coor[u:(u+h), l:(l+w), :] = coor_load['coor']

        coor_sym = coor.copy()
        if self.object_name == 'skateboard':
            coor_sym[:, :, 0] = - coor_sym[:, :, 0]
            coor_sym[:, :, 2] = - coor_sym[:, :, 2]
        elif self.object_name == 'tennis':
            coor_sym[:, :, 0] = - coor_sym[:, :, 0]
            coor_sym[:, :, 2] = - coor_sym[:, :, 2]
        elif self.object_name == 'baseball':
            coor_sym[:, :, 0] = - coor_sym[:, :, 0]
            coor_sym[:, :, 1] = - coor_sym[:, :, 1]
        elif self.object_name == 'barbell':
            coor_sym[:, :, 0] = - coor_sym[:, :, 0]
            coor_sym[:, :, 2] = - coor_sym[:, :, 2]


        box = np.loadtxt(box_path) # xywh
        x, y, w, h = box
        c = np.array([x + w * 0.5, y + h * 0.5])
        s = max(1, max(w, h) * 1.5)

        tx, ty, rot, scale, color_scale = get_augmentation_params()
        if self.object_name in ['tennis', 'baseball']:
            rot = np.random.randn() / 3 * 145 if np.random.random() < 0.5 else 0
            s = max(1, max(w, h))
            scale = max(0.1, 1. + np.random.randn() / 3 * 0.75)
        if self.object_name == 'violin':
            rot = np.random.randn() / 3 * 145 if np.random.random() < 0.5 else 0
            s = max(1, max(w, h))
            scale = max(0.7, 1. + np.random.randn() / 3 * 1.75)

        c[0] += tx * s
        c[1] += ty * s
        s = s * scale
        out_size = self.out_res

        if random.random() < 0.5:
            cx, cy, s = self.boxes_bigdet[item_id]
            s = s * 1.5

        # rot = 0 # basketball, barbell

        if random.random() < 0.3:

            if random.random() < 1: # 0.5:
                s = min(img_w, img_h)
                # cx, cy = np.random.random() * img_w / 2 + img_w / 4, np.random.random() * img_h / 2 + img_h / 4
                cx, cy = np.random.random() * img_w, np.random.random() * img_h
                c = np.array([cx, cy])
                # s = (0.02 + np.random.random() / 2) * s
                s = (0.02 + np.random.random() / 10) * s
                # s = (0.2 + np.random.random() / 5) * s
                img_patch, img_trans = generate_image_patch(image, c[0], c[1], s, self.out_res, rot, color_scale)
                img_patch = img_patch.astype(np.float32).transpose((2, 0, 1))
            else:
                img_id = self.negative_img_list[np.random.randint(len(self.negative_img_list))]
                video_id, seq_id, frame_id = img_id.split('_')
                image_path = os.path.join('/storage/data/huochf/HOIYouTube/{}/images_temp/{}/{}.jpg'.format(self.object_name, video_id, frame_id))
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                h, w, _ = image.shape
                mask_dir = os.path.join('/storage/data/huochf/HOIYouTube/{}/hoi_mask/{}/{}/{}.pkl'.format(self.object_name, video_id, seq_id, frame_id))
                mask_load = load_pickle(mask_dir)
                object_mask = np.zeros((h, w))
                if mask_load['object']['mask'] is not None:
                    mask_h, mask_w = mask_load['object']['mask_shape']
                    x1, y1, x2, y2 = mask_load['object']['mask_box']
                    object_mask[y1:y2+1, x1:x2+1] = np.unpackbits(mask_load['object']['mask'])[:mask_h * mask_w].reshape(mask_h, mask_w)

                object_bbox = extract_bbox_from_mask(object_mask)
                if object_bbox is None:
                    object_bbox = np.array([0, 0, 256, 256])

                x1, y1, x2, y2 = object_bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                s = max(x2 - x1, y2 - y1) * 1.5

                img_patch, img_trans = generate_image_patch(image, cx, cy, s, self.out_res, rot, color_scale)
                img_patch = img_patch.astype(np.float32).transpose((2, 0, 1))

            for n_c in range(3):
                img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255) / 255.
                img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]
            mask_patch = np.zeros((self.coor_res, self.coor_res), dtype=np.float32)
            coor_patch = coor_sym_patch = np.zeros((3, self.coor_res, self.coor_res), dtype=np.float32)

            return img_patch, mask_patch, coor_patch, coor_sym_patch

        img_patch, img_trans = generate_image_patch(image, c[0], c[1], s, self.out_res, rot, color_scale)
        mask_patch, _ = generate_image_patch(mask, c[0], c[1], s, self.coor_res, rot, color_scale)
        coor_patch, _ = generate_image_patch(coor, c[0], c[1], s, self.coor_res, rot, color_scale)
        coor_sym_patch, _ = generate_image_patch(coor_sym, c[0], c[1], s, self.coor_res, rot, color_scale)

        img_patch = img_patch.astype(np.float32).transpose((2, 0, 1))
        for n_c in range(3):
            img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255) / 255.
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        coor_patch = coor_patch.astype(np.float32).transpose((2, 0, 1))
        coor_patch = coor_patch / np.array(self.corr_norm).reshape(3, 1, 1)
        coor_sym_patch = coor_sym_patch.astype(np.float32).transpose((2, 0, 1))
        coor_sym_patch = coor_sym_patch / np.array(self.corr_norm).reshape(3, 1, 1)

        return img_patch, mask_patch, coor_patch, coor_sym_patch
