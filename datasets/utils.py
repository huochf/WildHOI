import random
import json
import pickle
import numpy as np
import cv2
import torch


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

        
def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except:
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='iso-8859-1')

    return data


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_J_regressor(path):
    data = np.loadtxt(path)

    with open(path, 'r') as f:
        shape = f.readline().split()[1:]
    J_regressor = np.zeros((int(shape[0]), int(shape[1])), dtype=np.float32)
    for i, j, v in data:
        J_regressor[int(i), int(j)] = v
    return J_regressor



def get_augmentation_params():
    tx = np.random.randn() / 3 * 0.1
    ty = np.random.randn() / 3 * 0.1
    rot = np.random.randn() / 3 * 30 if np.random.random() < 0.5 else 0
    scale = 1. + np.random.randn() / 3 * 0.3
    c_up = 1. + 0.2
    c_low = 1. - 0.2
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    return tx, ty, rot, scale, color_scale


################################### copy from prohmr.datasets.utils.py ##############################################
def get_augmentation_params_from_cfg(cfg):
    tx = np.random.randn() / 3 * cfg.dataset.aug_trans_factor
    ty = np.random.randn() / 3 * cfg.dataset.aug_trans_factor
    rot = np.random.randn() / 3 * cfg.dataset.aug_rot_factor if np.random.random() < cfg.dataset.aug_ratio else 0
    scale = 1. + np.random.randn() / 3 * cfg.dataset.aug_scale_factor
    c_up = 1. + cfg.dataset.aug_color_scale
    c_low = 1. - cfg.dataset.aug_color_scale
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


def get_rotmat_from_angle(rot):
    rot_mat = np.eye(3)
    if not rot == 0:
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
    return rot_mat


def rot_keypoints(keypoint_3d, rot):
    rot_mat = get_rotmat_from_angle(rot)
    keypoint_3d = np.einsum('ij,kj->ki', rot_mat, keypoint_3d)
    keypoint_3d = keypoint_3d.astype(np.float32)
    return keypoint_3d


def trans_keypoints(keypoint_2d, trans):
    src_pt = np.concatenate([keypoint_2d, np.ones((len(keypoint_2d), 1))], axis=1).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2].T


def perspective_projection(points, trans=None, rotmat=None, focal_length=None, optical_center=None):
    # points: [b, n, 3], trans: [b, 3], rotmat: [b, 3, 3], focal_length: [b, 2], optical_center: [b, 2]
    if rotmat is not None:
        points = torch.matmul(points, rotmat.permute(0, 2, 1))
    if trans is not None:
        points = points + trans[:, None]

    if focal_length is not None:
        u = points[:, :, 0] / points[:, :, 2] * focal_length[:, 0:1]
        v = points[:, :, 1] / points[:, :, 2] * focal_length[:, 1:]
        points_2d = torch.stack([u, v], dim=2)
    else:
        u = points[:, :, 0] / points[:, :, 2]
        v = points[:, :, 1] / points[:, :, 2]
        points_2d = torch.stack([u, v], dim=2)

    if optical_center is not None:
        points_2d = points_2d + optical_center[:, None]

    return points_2d
################################################### end of copy ######################################################
