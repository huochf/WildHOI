import os
import pickle
import json
import cv2
from tqdm import tqdm
import argparse
import trimesh
import numpy as np
import torch

from sklearn.neighbors import KernelDensity
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle
from smplx import SMPLLayer

from visualize_hoi_kps import visualize_sparse_keypoints


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


def smooth_sequence(sequence, windows=5):
    sequence = np.stack(sequence, axis=0)
    n, d = sequence.shape

    sequence = np.concatenate([np.zeros((windows // 2, d)), sequence, np.zeros((windows // 2, d))], axis=0) # [seq_n + windows - 1, n, 3]
    confidence_score = np.ones([n + windows // 2 * 2, 1])
    confidence_score[:windows // 2] = 0
    confidence_score[- windows // 2:] = 0
    smooth_kps = np.stack([
        sequence[i: n + i, :] for i in range(windows)
    ], axis=0)    
    confidence_score = np.stack([
        confidence_score[i: n + i, :] for i in range(windows)
    ], axis=0)
    smooth_kps = (smooth_kps * confidence_score).sum(0) / (confidence_score.sum(0) + 1e-8)
    return smooth_kps


def filling_empty(object_kps_seq, valid):
    object_kps_seq_new = []
    n_seq = len(object_kps_seq)

    for i in range(n_seq):
        kps = object_kps_seq[i]
        if not valid[i]:
            prev_i = i
            while not valid[prev_i] and prev_i > 0:
                prev_i -= 1
            next_i = i
            while not valid[next_i] and next_i < n_seq - 1:
                next_i += 1

            if valid[prev_i] and valid[next_i]:
                kps = (object_kps_seq[prev_i] * (i - prev_i) + object_kps_seq[next_i] * (next_i - i)) / (next_i - prev_i)
            elif valid[prev_i] and not valid[next_i]:
                kps = object_kps_seq[prev_i]
            elif not valid[prev_i] and valid[next_i]:
                kps = object_kps_seq[next_i]

        object_kps_seq_new.append(kps)

    begin_idx = 0
    while begin_idx < n_seq and not valid[begin_idx]:
        begin_idx += 1
    end_idx = n_seq - 1
    while end_idx > 0 and not valid[end_idx]:
        end_idx -= 1
    return np.array(object_kps_seq), begin_idx, end_idx


def generate_hoi_kps(args):
    device = torch.device('cuda')

    smpl_dir = os.path.join(args.root_dir, 'potter_smpl_tuned')
    object_pose_dir = os.path.join(args.root_dir, 'object_pose_framewise')
    tracking_results_dir = os.path.join(args.root_dir, 'hoi_tracking')

    object_name = args.root_dir.split('/')[-1]
    if object_name == 'cello':
        object_mesh = trimesh.load('../data/objects/{}_body.ply'.format(object_name), process=False)
    else:
        object_mesh = trimesh.load('../data/objects/{}.ply'.format(object_name), process=False)

    object_v = np.array(object_mesh.vertices)
    object_kps_indices = load_object_kps_indices(object_name)
    object_kps_org = []
    for k, v in object_kps_indices.items():
        object_kps_org.append(object_v[v].mean(0))
    object_kps_org = np.stack(object_kps_org) # [n, 3]

    smpl = SMPLLayer('/public/home/huochf/projects/3D_HOI/hoiYouTube/data/smpl/smpl/', gender='neutral', use_pca=False).to(device)


    for tracking_file in tqdm(os.listdir(tracking_results_dir)):
        video_id = tracking_file.split('_')[0]

        # if int(video_id) > 150:
        #     continue

        tracking_results = load_pickle(os.path.join(tracking_results_dir, tracking_file))

        output_dir = os.path.join(args.root_dir, 'hoi_kps_framewise', video_id)
        os.makedirs(output_dir, exist_ok=True)
        for hoi_instance in tracking_results['hoi_instances']:
            hoi_id = hoi_instance['hoi_id']

            if os.path.exists(os.path.join(output_dir, '{}_hoi_kps.pkl'.format(hoi_id))):
                continue

            smpl_params_file = os.path.join(smpl_dir, video_id, '{}_smpl.pkl'.format(hoi_id))
            if not os.path.exists(smpl_params_file):
                continue
            smpl_params = load_pickle(smpl_params_file)
            object_pose_file = os.path.join(object_pose_dir, video_id, '{}_obj_RT.pkl'.format(hoi_id))
            if not os.path.exists(object_pose_file):
                continue
            try:
                object_RT = load_pickle(object_pose_file)
            except:
                continue

            assert len(smpl_params['frame_ids']) == len(object_RT)
            n_seq = len(object_RT)

            smpl_kps_seq = []
            smpl_orient_seq = []
            object_kps_seq = []
            object_kps_valid = []
            smpl_body_pose_seq = []
            for i in range(n_seq):
                smpl_orient = torch.tensor(smpl_params['global_orient_rotmat'][i]).reshape(1, 3, 3).float().to(device)
                smpl_body_pose = torch.tensor(smpl_params['body_pose_rotmat'][i]).reshape(1, 23, 3, 3).float().to(device)
                smpl_shape = torch.tensor(smpl_params['smpl_betas'][i]).reshape(1, 10).float().to(device)
                cam_trans = torch.tensor(smpl_params['cam_transl'][i]).reshape(1, 1, 3).float().to(device)

                smpl_out = smpl(betas=smpl_shape, 
                                  global_orient=smpl_orient, 
                                  body_pose=smpl_body_pose,)
                smpl_J = smpl_out.joints.detach()
                smpl_J = smpl_J - smpl_J[:, :1] + cam_trans
                smpl_J = smpl_J[0].cpu().numpy()

                rotmat = object_RT[i]['rotmat']
                trans = object_RT[i]['trans']
                success = object_RT[i]['success']
                object_kps = np.matmul(object_kps_org, rotmat.transpose(1, 0)) + trans.reshape(1, 3)

                f = 1000
                s = 256
                d = smpl_params['cam_transl'][i][2]

                def project(x, f):
                    u = x[..., 0] / (x[..., 2] + 1e-8) * f
                    v = x[..., 1] / (x[..., 2] + 1e-8) * f
                    return np.stack([u, v], axis=-1)
                smpl_kps = project(smpl_J, f)
                object_kps = project(object_kps, f)

                center = smpl_kps[:1]
                smpl_kps = smpl_kps - center
                object_kps = object_kps - center

                scale = d / (f / s * 2)
                smpl_kps = smpl_kps * scale
                object_kps = object_kps * scale

                #################################################################################

                # smpl_kps = smpl_kps[:22]
                # smpl_kps = smpl_kps / 128
                # object_kps = object_kps / 128
                # image_kps = visualize_sparse_keypoints(smpl_kps, object_kps, 'barbell', res=256)

                # smpl_out = smpl(betas=smpl_shape, 
                #                   global_orient=smpl_orient, 
                #                   body_pose=smpl_body_pose,)
                # smpl_J = smpl_out.joints.detach()
                # smpl_J = smpl_J - smpl_J[:, :1]
                # smpl_J = smpl_J[0].cpu().numpy()

                # def project(kps, d, f):
                #     kps[..., 2] += d
                #     u = kps[..., 0] / kps[..., 2] * f
                #     v = kps[..., 1] / kps[..., 2] * f
                #     return np.stack([u, v], axis=-1)

                # f = 1000
                # s = 128
                # d = f / s
                # smpl_kps = project(smpl_J, d=d, f=f / s)

                # image_kps = image_kps * 0.5
                # smpl_kps = smpl_kps[:22]
                # center = smpl_kps[:1]
                # smpl_kps = smpl_kps - center
                # image_kps = visualize_sparse_keypoints(smpl_kps, object_kps, 'barbell', res=256, image=image_kps)

                # cv2.imwrite('./__debug__/{}_{}_{}.jpg'.format(video_id, hoi_id, i), image_kps.astype(np.uint8))

                #################################################################################

                smpl_kps_seq.append(smpl_kps)
                object_kps_seq.append(object_kps)
                smpl_orient_seq.append(smpl_orient[0].detach().cpu().numpy())
                smpl_body_pose_seq.append(matrix_to_axis_angle(smpl_body_pose[0]).reshape(69).detach().cpu().numpy())

                if np.abs(trans[2] - smpl_params['cam_transl'][i][2]) > 1.5:
                    object_kps_valid.append(False)
                else:
                    object_kps_valid.append(success)

            smpl_kps_seq = np.array(smpl_kps_seq).reshape(n_seq, -1)
            object_kps_seq = np.array(object_kps_seq).reshape(n_seq, -1)
            smpl_orient_seq = matrix_to_rotation_6d(torch.tensor(np.array(smpl_orient_seq)))
            smpl_orient_seq = smpl_orient_seq.numpy()
            smpl_orient_seq = rotation_6d_to_matrix(torch.tensor(smpl_orient_seq))
            smpl_orient_seq = smpl_orient_seq.numpy()
            smpl_body_pose_seq = np.array(smpl_body_pose_seq)
            n_seq = object_kps_seq.shape[0]
            save_pickle(os.path.join(output_dir, '{}_hoi_kps.pkl'.format(hoi_id)),
                {
                    'frame_ids': smpl_params['frame_ids'],
                    'smpl_kps_seq': smpl_kps_seq.reshape(n_seq, -1, 2),
                    'object_kps_seq': object_kps_seq.reshape(n_seq, -1, 2),
                    'smpl_orient_seq': smpl_orient_seq.reshape(n_seq, 3, 3),
                    'smpl_body_pose': smpl_body_pose_seq.reshape(n_seq, -1),
                })
            # exit(0)


def generate_importance_score(args):
    X = []
    frame_id_to_idx = {}
    count = 0
    for video_id in os.listdir(os.path.join(args.root_dir, 'hoi_kps', )):
        for file in os.listdir(os.path.join(args.root_dir, 'hoi_kps', video_id)):
            hoi_id = file.split('_')[0]
            kps_load = load_pickle(os.path.join(args.root_dir, 'hoi_kps', video_id, file))

            n_seq = kps_load['smpl_kps_seq'].shape[0]
            for i in range(n_seq):
                frame_id = '{}_{}_{}'.format(video_id, hoi_id, i)
                kps = np.concatenate([kps_load['smpl_kps_seq'][i], kps_load['object_kps_seq'][i]], axis=0).reshape(-1)
                X.append(kps)

                frame_id_to_idx[frame_id] = count
                count += 1
    print('found {} frames'.format(len(X)))
    print('running kernel density estimation...')
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
    density_score_all = kde.score_samples(X)

    print('updating density scores...')
    for video_id in os.listdir(os.path.join(args.root_dir, 'hoi_kps', )):
        for file in os.listdir(os.path.join(args.root_dir, 'hoi_kps', video_id)):
            hoi_id = file.split('_')[0]
            kps_load = load_pickle(os.path.join(args.root_dir, 'hoi_kps', video_id, file))

            density_scores = []
            n_seq = kps_load['smpl_kps_seq'].shape[0]
            for i in range(n_seq):
                frame_id = '{}_{}_{}'.format(video_id, hoi_id, i)
                density_scores.append(density_score_all[frame_id_to_idx[frame_id]])
            kps_load['density_scores'] = np.array(density_scores).reshape(-1)
            save_pickle(os.path.join(args.root_dir, 'hoi_kps', video_id, file), kps_load)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BigDetection Inference.')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    args = parser.parse_args()

    generate_hoi_kps(args)
    # generate_importance_score(args)
