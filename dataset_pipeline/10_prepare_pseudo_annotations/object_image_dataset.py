import os
import cv2
import numpy as np


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


class ObjectImageDataset:

    def __init__(self, root_dir, frame_ids, bboxes, out_res=256, coor_res=64):
        self.root_dir = root_dir
        self.frame_ids = frame_ids
        self.bboxes = bboxes
        assert len(self.frame_ids) == len(self.bboxes)

        self.out_res = out_res
        self.coor_res = coor_res
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


    def __len__(self, ):
        return len(self.frame_ids)


    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        bbox = self.bboxes[idx]

        image = cv2.imread(os.path.join(self.root_dir, '{}.jpg'.format(frame_id)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_h, img_w, _ = image.shape

        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        s = max((y2 - y1), (x2 - x1)) * 1.5

        rot, color_scale = 0., [1., 1., 1.]

        img_patch, img_trans = generate_image_patch(image, cx, cy, s, self.out_res, rot, color_scale)
        img_patch = img_patch.astype(np.float32).transpose((2, 0, 1))
        for n_c in range(3):
            img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255) / 255.
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        return img_patch, np.array([cx, cy]), s
