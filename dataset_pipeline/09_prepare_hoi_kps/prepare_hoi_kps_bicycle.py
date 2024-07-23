import os
import pickle
import json
import cv2
from tqdm import tqdm
import argparse
import trimesh
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from smplx import SMPLX


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_object_kps_indices(object_name):
    if object_name == 'barbell':
        object_file = '../data/objects/barbell_keypoints_12.json'
    elif object_name == 'cello':
        object_file = '../data/objects/cello_keypoints_14.json'
    elif object_name == 'baseball':
        object_file = '../data/objects/baseball_keypoints.json'
    elif object_name == 'tennis':
        object_file = '../data/objects/tennis_keypoints_7.json'
    elif object_name == 'skateboard':
        object_file = '../data/objects/skateboard_keypoints_8.json'
    elif object_name == 'basketball':
        object_file = '../data/objects/basketball_keypoints.json'
    elif object_name == 'yogaball':
        object_file = '../data/objects/yogaball_keypoints.json'
    with open(object_file, 'r') as f:
        indices = json.load(f)

    if object_name == 'baseball':
        indices = {'1': indices['1'], '5': indices['5']}

    return indices


def generate_hoi_kps(args):
    device = torch.device('cuda')

    smpl_dir = os.path.join(args.root_dir, 'smplx_tuned')
    object_pose_dir = os.path.join(args.root_dir, 'object_pose')
    tracking_results_dir = os.path.join(args.root_dir, 'hoi_tracking')

    object_name = args.root_dir.split('/')[-1]
    assert object_name == 'bicycle'

    bicycle_front = trimesh.load('../data/objects/bicycle_front.ply', process=False)
    bicycle_back = trimesh.load('../data/objects/bicycle_back.ply', process=False)
    bicycle_front_v = np.array(bicycle_front.vertices)
    bicycle_back_v = np.array(bicycle_back.vertices)

    with open('../data/objects/bicycle_front_keypoints.json', 'r') as f:
        bicycle_front_kps_indices = json.load(f)
    with open('../data/objects/bicycle_back_keypoints.json', 'r') as f:
        bicycle_back_kps_indices = json.load(f)
    bicycle_front_kps = []
    bicycle_back_kps = []
    for k, v in bicycle_front_kps_indices.items():
        bicycle_front_kps.append(bicycle_front_v[v].mean(0))
    for k, v in bicycle_back_kps_indices.items():
        bicycle_back_kps.append(bicycle_back_v[v].mean(0))
    bicycle_front_kps = np.stack(bicycle_front_kps) # [n, 3]
    bicycle_back_kps = np.stack(bicycle_back_kps) # [n, 3]

    rot_axis_begin = bicycle_front_v[bicycle_front_kps_indices['5']].mean(0)
    rot_axis_end = bicycle_front_v[bicycle_front_kps_indices['6']].mean(0)
    rot_axis = rot_axis_end - rot_axis_begin
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    smplx = SMPLX('/public/home/huochf/projects/3D_HOI/hoiYouTube/data/smpl/smplx/', gender='neutral', use_pca=False).to(device)

    for tracking_file in tqdm(sorted(os.listdir(tracking_results_dir))):
        video_id = tracking_file.split('_')[0]

        tracking_results = load_pickle(os.path.join(tracking_results_dir, tracking_file))
        output_dir = os.path.join(args.root_dir, 'hoi_kps', video_id)
        os.makedirs(output_dir, exist_ok=True)

        for hoi_instance in tracking_results['hoi_instances']:
            hoi_id = hoi_instance['hoi_id']

            smplx_params_file = os.path.join(smpl_dir, video_id, '{}_smplx.pkl'.format(hoi_id))
            if not os.path.exists(smplx_params_file):
                continue
            smpl_params = load_pickle(smplx_params_file)
            object_pose_file = os.path.join(object_pose_dir, video_id, '{}_obj_RT.pkl'.format(hoi_id))
            if not os.path.exists(object_pose_file):
                continue
            object_RT = load_pickle(object_pose_file)

            assert len(smpl_params) == len(object_RT)
            n_seq = len(object_RT)

            frame_ids = []
            smpl_kps_seq = []
            object_kps_seq = []
            cam_R_seq = []
            cam_T_seq = []
            focal_seq = []
            princpt_seq = []
            for i in range(n_seq):
                smplx_body_pose = torch.tensor(smpl_params[i]['body_pose']).reshape(1, 63).float().to(device)
                smplx_lhand_pose = torch.tensor(smpl_params[i]['left_hand_pose']).reshape(1, 45).float().to(device)
                smplx_rhand_pose = torch.tensor(smpl_params[i]['right_hand_pose']).reshape(1, 45).float().to(device)
                smplx_shape = torch.tensor(smpl_params[i]['betas']).reshape(1, 10).float().to(device)

                cam_R = torch.tensor(smpl_params[i]['cam_R']).reshape(3, 3).float().to(device)
                cam_T = torch.tensor(smpl_params[i]['cam_T']).reshape(1, 3).float().to(device)

                smplx_out = smplx(betas=smplx_shape, body_pose=smplx_body_pose, left_hand_pose=smplx_lhand_pose, right_hand_pose=smplx_rhand_pose)
                smpl_J = smplx_out.joints.detach()[0]
                smpl_J = smpl_J - smpl_J[:1]
                smpl_J = smpl_J @ cam_R.transpose(1, 0) + cam_T
                smpl_J = smpl_J.cpu().numpy()

                object_rotmat = object_RT[i]['rotmat']
                object_trans = object_RT[i]['trans']
                rot_angle = object_RT[i]['rot_angle']

                _front_kps = bicycle_front_kps - rot_axis_begin.reshape(1, 3)
                _front_kps = _front_kps @ Rotation.from_rotvec(rot_angle * rot_axis).as_matrix().T
                _front_kps = _front_kps + rot_axis_begin.reshape(1, 3)
                object_kps = np.concatenate([_front_kps, bicycle_back_kps], axis=0)
                object_kps = object_kps @ object_rotmat.T + object_trans.reshape(1, 3)

                fx, fy = smpl_params[i]['focal']
                cx, cy = smpl_params[i]['princpt']

                def project(kps, fx, fy, cx, cy):
                    u = kps[:, 0] / kps[:, 2] * fx + cx
                    v = kps[:, 1] / kps[:, 2] * fy + cy
                    return np.stack([u, v], axis=-1)

                smpl_kp2d = project(smpl_J, fx.item(), fy.item(), cx.item(), cy.item())
                object_kps2d = project(object_kps, fx.item(), fy.item(), cx.item(), cy.item())

                frame_ids.append(smpl_params[i]['frame_id'])
                smpl_kps_seq.append(smpl_kp2d[:22])
                object_kps_seq.append(object_kps2d)
                cam_R_seq.append(smpl_params[i]['cam_R'])
                cam_T_seq.append(smpl_params[i]['cam_T'])
                focal_seq.append(smpl_params[i]['focal'])
                princpt_seq.append(smpl_params[i]['princpt'])

            save_pickle(os.path.join(output_dir, '{}_hoi_kps.pkl'.format(hoi_id)), 
                {
                    'frame_ids': frame_ids,
                    'smpl_kps_seq': np.stack(smpl_kps_seq, axis=0),
                    'object_kps_seq': np.stack(object_kps_seq, axis=0),
                    'cam_R_seq': np.stack(cam_R_seq, axis=0),
                    'cam_T_seq': np.stack(cam_T_seq, axis=0),
                    'focal_seq': np.stack(focal_seq, axis=0),
                    'princpt_seq': np.stack(princpt_seq, axis=0),
                })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare hoi kps')
    parser.add_argument('--root_dir', type=str, help='The dataset directory')
    args = parser.parse_args()

    generate_hoi_kps(args)
