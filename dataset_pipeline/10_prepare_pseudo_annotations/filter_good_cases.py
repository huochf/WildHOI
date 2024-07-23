import os
import argparse
import trimesh
import pickle
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import torch
from smplx import SMPLX


def load_keypoints(file):
    with open(file, 'r') as f:
        all_lines = f.readlines()
    if int(all_lines[0]) == 0:
        return np.empty((0, 3))
    keypoints = []
    for i in range(len(all_lines) - 1):
        line = all_lines[i + 1]
        x, y = line.split(' ')
        x, y = float(x), float(y)
        keypoints.append([x, y, 1])
    return np.array(keypoints)


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


object_name = 'baseball'
hoi_recon_results = load_pickle(os.path.join('hoi_recon_with_contact', '{}_test.pkl'.format(object_name)))
hoi_recon_results = {item['image_id']: item for item in hoi_recon_results}

good_cases_count = 0
_output_dir = os.path.join('annotation_hoi', object_name, 'test')
os.makedirs(_output_dir, exist_ok=True)

for file in os.listdir('./__debug__/{}_recon_vis_good_cases'.format(object_name)):

    if 'novel_views' not in file:
        continue

    keypoints = load_keypoints('./__debug__/{}_recon_vis_good_cases/'.format(object_name) + file)
    if len(keypoints) == 1:
        good_cases_count += 1
        video_id, hoi_id, frame_id, _, _ = file.split('_')
        image_id = '_'.join([video_id, hoi_id, frame_id])
        save_dict = {
            'smplx_betas': hoi_recon_results[image_id]['smplx_betas'],
            'smplx_body_pose': hoi_recon_results[image_id]['smplx_body_pose'],
            'smplx_lhand_pose': hoi_recon_results[image_id]['smplx_lhand_pose'],
            'smplx_rhand_pose': hoi_recon_results[image_id]['smplx_rhand_pose'],
            'obj_rel_rotmat': hoi_recon_results[image_id]['obj_rel_rotmat'],
            'obj_rel_trans': hoi_recon_results[image_id]['obj_rel_trans'],
            'hoi_rotmat': hoi_recon_results[image_id]['hoi_rotmat'],
            'hoi_trans': hoi_recon_results[image_id]['hoi_trans'],
            'object_scale': hoi_recon_results[image_id]['object_scale'],
            'crop_bboxes': hoi_recon_results[image_id]['crop_bboxes'],
            'focal': hoi_recon_results[image_id]['focal'],
            'princpt': hoi_recon_results[image_id]['princpt'],
        }
        if 'object_rot_angle' in hoi_recon_results[image_id]:
            save_dict['object_rot_angle'] = hoi_recon_results[image_id]['object_rot_angle']
        save_pickle(os.path.join(_output_dir, '{}.pkl'.format(image_id)), save_dict)

print(good_cases_count)
