import os
import argparse
import numpy as np
import cv2
import trimesh
import json
import pickle
from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F

from smplx import SMPLX, SMPLXLayer
from pytorch3d.transforms import matrix_to_rotation_6d, matrix_to_axis_angle, axis_angle_to_matrix, rotation_6d_to_matrix


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def smooth_wholebody_kps_sequence(wholebody_kps_seq, windows=5):

    seq_n, kps_n, _ = wholebody_kps_seq.shape
    smooth_kps = np.concatenate([np.zeros((windows // 2, kps_n, 3)), wholebody_kps_seq, np.zeros((windows // 2, kps_n, 3))], axis=0) # [seq_n + windows - 1, n, 3]
    confidence_score = np.stack([
        smooth_kps[i: seq_n + i, :, 2:] for i in range(windows)
    ], axis=0)
    smooth_kps = np.stack([
        smooth_kps[i: seq_n + i, :, :2] for i in range(windows)
    ], axis=0)
    smooth_kps = (smooth_kps * confidence_score).sum(0) / (confidence_score.sum(0) + 1e-8)
    smooth_kps = np.concatenate([smooth_kps, wholebody_kps_seq[:, :, 2:]], axis=2)

    return smooth_kps


class Keypoint2DLoss(nn.Module):

    def __init__(self, wholebody_kps, ):
        super().__init__()
        self.register_buffer('wholebody_kps', wholebody_kps)


    def project(self, points3d, batch_idx, batch_size):
        u = points3d[..., 0] / points3d[..., 2]
        v = points3d[..., 1] / points3d[..., 2]
        return torch.stack([u, v], dim=-1)


    def forward(self, smplx_out, batch_idx, batch_size):
        wholebody_kps = smplx_out['wholebody_kps']
        wholebody_kps = self.project(wholebody_kps, batch_idx, batch_size) 

        loss_body_kps2d = ((wholebody_kps[:, :23] - self.wholebody_kps[batch_idx:batch_idx + batch_size, :23, :2]) * 1000) ** 2
        loss_body_kps2d = (loss_body_kps2d * self.wholebody_kps[batch_idx:batch_idx + batch_size, :23, 2:]).mean()

        # loss_lhand_kps2d = ((wholebody_kps[:, 23:44] - self.wholebody_kps[batch_idx:batch_idx + batch_size, 91:112, :2]) / 
        #     self.person_bboxes[batch_idx:batch_idx + batch_size, :, 2:]) ** 2
        # loss_lhand_kps2d = (loss_lhand_kps2d * self.wholebody_kps[batch_idx:batch_idx + batch_size, 91:112, 2:]).mean()

        # loss_rhand_kps2d = ((wholebody_kps[:, 44:65] - self.wholebody_kps[batch_idx:batch_idx + batch_size, 112:133, :2]) / 
        #     self.person_bboxes[batch_idx:batch_idx + batch_size, :, 2:]) ** 2
        # loss_rhand_kps2d = (loss_rhand_kps2d * self.wholebody_kps[batch_idx:batch_idx + batch_size, 112:133, 2:]).mean()

        return {
            'loss_body_kps2d': loss_body_kps2d,
            # 'loss_lhand_kps2d': loss_lhand_kps2d,
            # 'loss_rhand_kps2d': loss_rhand_kps2d,
        }


class SMPLXDecayLoss(nn.Module):

    def __init__(self, betas_init, body_pose_init, lhand_pose_init, rhand_pose_init):
        super().__init__()
        self.register_buffer('betas_init', betas_init)
        self.register_buffer('body_pose_init', body_pose_init)
        self.register_buffer('lhand_pose_init', lhand_pose_init)
        self.register_buffer('rhand_pose_init', rhand_pose_init)


    def forward(self, smplx_out, batch_idx, batch_size):
        betas = smplx_out['betas']
        body_pose = smplx_out['body_pose']
        lhand_pose = smplx_out['lhand_pose']
        rhand_pose = smplx_out['rhand_pose']

        betas_decay_loss = ((betas - self.betas_init[batch_idx:batch_idx + batch_size]) ** 2).mean()
        body_pose_decay_loss = ((body_pose - self.body_pose_init[batch_idx:batch_idx + batch_size]) ** 2).mean()
        # lhand_pose_decay_loss = ((lhand_pose - self.lhand_pose_init[batch_idx:batch_idx + batch_size]) ** 2).mean()
        # rhand_pose_decay_loss = ((rhand_pose - self.rhand_pose_init[batch_idx:batch_idx + batch_size]) ** 2).mean()

        body_pose_norm_loss = (body_pose ** 2).mean()
        lhand_pose_norm_loss = (lhand_pose ** 2).mean()
        rhand_pose_norm_loss = (rhand_pose ** 2).mean()

        return {
            'betas_decay_loss': betas_decay_loss,
            'bop_decay_loss': body_pose_decay_loss,
            # 'lhp_decay_loss': lhand_pose_decay_loss,
            # 'rhp_decay_loss': rhand_pose_decay_loss,
            'bop_norm_loss': body_pose_norm_loss,
            'lhp_norm_loss': lhand_pose_norm_loss,
            'rhp_norm_loss': rhand_pose_norm_loss,
        }


class SmoothLoss(nn.Module):

    def __init__(self, ):
        super().__init__()


    def forward(self, smplx_out, batch_idx, batch_size):
        smplx_v = smplx_out['smplx_v']

        if smplx_v.shape[0] > 1: 
            loss_smo_v = ((smplx_v[1:] - smplx_v[:-1]) ** 2).sum(-1).mean()
        else:
            loss_smo_v = torch.zeros(1).to(smplx_v.device).mean()

        smplx_J = smplx_out['smplx_J']

        if smplx_J.shape[0] > 1: 
            loss_smo_J = ((smplx_J[1:] - smplx_J[:-1]) ** 2).sum(-1).mean()
        else:
            loss_smo_J = torch.zeros(1).to(smplx_J.device).mean()

        cam_Ts = smplx_out['cam_Ts']

        if cam_Ts.shape[0] > 1: 
            loss_cam_trans = ((cam_Ts[1:] - cam_Ts[:-1]) ** 2).sum(-1).mean()
        else:
            loss_cam_trans = torch.zeros(1).to(cam_Ts.device).mean()

        cam_R6d = smplx_out['cam_R6d']

        if cam_R6d.shape[0] > 1: 
            loss_cam_rot = ((cam_R6d[1:] - cam_R6d[:-1]) ** 2).sum(-1).mean()
        else:
            loss_cam_rot = torch.zeros(1).to(cam_R6d.device).mean()

        return {
            'loss_smo_v': loss_smo_v,
            'loss_smo_J': loss_smo_J,
            'loss_smo_trans': loss_cam_trans,
            'loss_smo_rot': loss_cam_rot,
        }


class CameraDecayLoss(nn.Module):

    def __init__(self, cam_Rs, cam_Ts):
        super().__init__()
        cam_R6d_init = matrix_to_rotation_6d(axis_angle_to_matrix(cam_Rs.reshape(-1, 3)))
        self.register_buffer('cam_R6d_init', cam_R6d_init)
        self.register_buffer('cam_T_init', cam_Ts)


    def forward(self, smplx_out, batch_idx, batch_size):
        cam_R6d = smplx_out['cam_R6d']
        cam_Ts = smplx_out['cam_Ts']
        cam_R_decay_loss = ((cam_R6d - self.cam_R6d_init[batch_idx:batch_idx + batch_size]) ** 2).mean()
        cam_T_decay_loss = ((cam_Ts - self.cam_T_init[batch_idx:batch_idx + batch_size]) ** 2).mean()

        return {
            'loss_cam_R_decay': cam_R_decay_loss,
            'loss_cam_T_decay': cam_T_decay_loss,
        }



class SMPLXInstance(nn.Module):

    def __init__(self, betas, body_pose, lhand_pose, rhand_pose, cam_Rs, cam_Ts, batch_size, shared_shape=False):
        super().__init__()

        n_seq = betas.shape[0]
        self.smplx = SMPLX('/public/home/huochf/projects/3D_HOI/hoiYouTube/data/smpl/smplx/', gender='neutral', use_pca=False)
        self.shared_shape = shared_shape
        if shared_shape:
            self.betas = nn.Parameter(betas.reshape(n_seq, 10).mean(0, keepdim=True))
        else:
            self.betas = nn.Parameter(betas.reshape(n_seq, 10))
        self.body_pose = nn.Parameter(body_pose.reshape(n_seq, 21, 3))
        self.lhand_pose = nn.Parameter(lhand_pose.reshape(n_seq, 15, 3))
        self.rhand_pose = nn.Parameter(rhand_pose.reshape(n_seq, 15, 3))

        self.cam_R6d = nn.Parameter(matrix_to_rotation_6d(axis_angle_to_matrix(cam_Rs.reshape(n_seq, 3))))
        self.cam_Ts = nn.Parameter(cam_Ts.reshape(n_seq, 3))

        wholebody_regressor = np.load('../data/smpl/smplx_wholebody_regressor.npz')
        self.register_buffer('wholebody_regressor', torch.tensor(wholebody_regressor['wholebody_regressor']).float())


    def forward(self, batch_idx, batch_size):
        b = self.body_pose[batch_idx:batch_idx + batch_size].shape[0]
        if self.shared_shape:
            betas = self.betas.repeat(b, 1)
        else:
            betas = self.betas[batch_idx:batch_idx + batch_size]

        global_orient = torch.zeros((b, 3), dtype=self.betas.dtype, device=self.betas.device)
        jaw_pose = torch.zeros((b, 3), dtype=self.betas.dtype, device=self.betas.device)
        leye_pose = torch.zeros((b, 3), dtype=self.betas.dtype, device=self.betas.device)
        reye_pose = torch.zeros((b, 3), dtype=self.betas.dtype, device=self.betas.device)
        expression = torch.zeros((b, 10), dtype=self.betas.dtype, device=self.betas.device)

        smplx_out = self.smplx(betas=betas,
                               body_pose=self.body_pose[batch_idx:batch_idx + batch_size],
                               left_hand_pose=self.lhand_pose[batch_idx:batch_idx + batch_size],
                               right_hand_pose=self.rhand_pose[batch_idx:batch_idx + batch_size],
                               global_orient=global_orient,
                               leye_pose=leye_pose,
                               reye_pose=reye_pose,
                               jaw_pose=jaw_pose,
                               expression=expression)
        smplx_v = smplx_out.vertices
        smplx_J = smplx_out.joints
        smplx_v_centered = smplx_v - smplx_J[:, :1]

        cam_Rs = rotation_6d_to_matrix(self.cam_R6d[batch_idx:batch_idx + batch_size])
        cam_Ts = self.cam_Ts[batch_idx:batch_idx + batch_size].reshape(-1, 1, 3)

        smplx_v = smplx_v_centered @ cam_Rs.permute(0, 2, 1) + cam_Ts
        wholebody_kps = self.wholebody_regressor.unsqueeze(0) @ smplx_v # [bs, 65, 3]

        results = {
            'betas': betas,
            'body_pose': self.body_pose[batch_idx:batch_idx + batch_size],
            'lhand_pose': self.lhand_pose[batch_idx:batch_idx + batch_size],
            'rhand_pose': self.rhand_pose[batch_idx:batch_idx + batch_size],
            'cam_R6d': self.cam_R6d[batch_idx:batch_idx + batch_size],
            'cam_Rs': cam_Rs,
            'cam_Ts': cam_Ts,
            'wholebody_kps': wholebody_kps,
            'smplx_v': smplx_v,
            'smplx_J': smplx_J,
        }
        return results


def optimize_all(args):
    root_dir = args.root_dir
    device = torch.device('cuda')
    tracking_results_dir = os.path.join(args.root_dir, 'hoi_tracking')

    batch_size = 256

    for video_idx in range(args.begin_idx, args.end_idx):
        video_id = '{:04d}'.format(video_idx)
        if os.path.exists(os.path.join(tracking_results_dir, '{}_tracking.pkl'.format(video_id))):
            tracking_results = load_pickle(os.path.join(tracking_results_dir, '{}_tracking.pkl'.format(video_id)))
        else:
            continue

        smpl_out_dir = os.path.join(args.root_dir, 'smplx_tuned', video_id)
        os.makedirs(smpl_out_dir, exist_ok=True)
        print('Found {} instances.'.format(len(tracking_results['hoi_instances'])))

        for hoi_instance in tracking_results['hoi_instances']:
            hoi_id = hoi_instance['hoi_id']
            # if os.path.exists(os.path.join(smpl_out_dir, '{}_smplx.pkl'.format(hoi_id))):
            #     continue

            # if not os.path.exists(os.path.join(root_dir, 'wholebody_kps_refined', video_id, '{}_wholebody_kps.pkl'.format(hoi_id))):
            #     continue
            if not os.path.exists(os.path.join(root_dir, 'wholebody_kps', video_id, '{}_wholebody_kps.pkl'.format(hoi_id))):
                continue
            if not os.path.exists(os.path.join(root_dir, 'smpler_x', video_id, '{}_smplx.pkl'.format(hoi_id))):
                continue

            wholebody_kps_list = load_pickle(os.path.join(root_dir, 'wholebody_kps', video_id, '{}_wholebody_kps.pkl'.format(hoi_id)))
            smplx_params_all = load_pickle(os.path.join(root_dir, 'smpler_x', video_id, '{}_smplx.pkl'.format(hoi_id)))

            assert len(wholebody_kps_list) == len(smplx_params_all)
            n_seq = len(smplx_params_all)

            if n_seq < 10:
                continue

            frame_ids = []
            wholebody_kps = []
            smplx_betas = []
            smplx_body_poses = []
            smplx_lhand_poses = []
            smplx_rhand_poses = []
            cam_Rs = []
            cam_Ts = []
            upper_body_count = 0
            for wholebody_kps_item, smplx_params_item in zip(wholebody_kps_list, smplx_params_all):
                assert wholebody_kps_item['frame_id'] == smplx_params_item['frame_id']
                frame_ids.append(smplx_params_item['frame_id'])
                smplx_betas.append(smplx_params_item['betas'])
                smplx_body_poses.append(smplx_params_item['body_pose'])
                smplx_lhand_poses.append(smplx_params_item['left_hand_pose'])
                smplx_rhand_poses.append(smplx_params_item['right_hand_pose'])
                cam_Rs.append(smplx_params_item['global_orient'])
                cam_Ts.append(smplx_params_item['transl'])
                princpt = np.array(smplx_params_item['princpt'])
                focal = np.array(smplx_params_item['focal'])
                keypoints = wholebody_kps_item['keypoints']
                keypoints[:, :2] = (keypoints[:, :2] - princpt.reshape(1, 2)) / focal.reshape(1, 2)
                keypoints[keypoints[:, 2] < 0.5] = 0
                if (keypoints[:23, 2] == 0).sum() > 5:
                    upper_body_count += 1
                wholebody_kps.append(keypoints)

            if upper_body_count < 1:
                continue

            wholebody_kps = smooth_wholebody_kps_sequence(np.array(wholebody_kps))
            wholebody_kps = torch.from_numpy(wholebody_kps).float()
            smplx_betas = torch.from_numpy(np.array(smplx_betas)).float()
            smplx_body_poses = torch.from_numpy(np.array(smplx_body_poses)).float()
            smplx_lhand_poses = torch.from_numpy(np.array(smplx_lhand_poses)).float()
            smplx_rhand_poses = torch.from_numpy(np.array(smplx_rhand_poses)).float()
            cam_Rs = torch.from_numpy(np.array(cam_Rs)).float()
            cam_Ts = torch.from_numpy(np.array(cam_Ts)).float()

            smplx_params_tuned = []

            smplx_instance = SMPLXInstance(smplx_betas, smplx_body_poses, smplx_lhand_poses, smplx_rhand_poses, cam_Rs, cam_Ts, batch_size, shared_shape=True).to(device)
            loss_functions = [
                Keypoint2DLoss(wholebody_kps).to(device),
                SmoothLoss().to(device),
                CameraDecayLoss(cam_Rs, cam_Ts).to(device),
            ]

            loss_weights_stage1 = {
                'loss_body_kps2d': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
                'loss_smo_v': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
                'loss_smo_J': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
                'loss_smo_trans': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
                'loss_smo_rot': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
                'loss_cam_R_decay': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
                'loss_cam_T_decay': lambda cst, it: 0 * 10. ** 0 * cst / (1 + 10 * it),
            }
            optimizer_stage1 = torch.optim.Adam([smplx_instance.cam_Ts, smplx_instance.cam_R6d], lr=0.01, betas=(0.9, 0.999))
            iterations = 1
            steps_per_iter = 500
            for it in range(iterations):
                loop = tqdm(range(steps_per_iter))
                for i in loop:
                    optimizer_stage1.zero_grad()
                    losses_seq = {}
                    for idx in range(i % 2, n_seq, batch_size):

                        losses = {}
                        smplx_out = smplx_instance.forward(idx, batch_size)
                        for f in loss_functions:
                            losses.update(f(smplx_out, idx, batch_size))

                        for k, v in losses.items():
                            if k not in losses_seq:
                                losses_seq[k] = []
                            losses_seq[k].append(v)

                    for k in losses_seq:
                        losses_seq[k] = torch.stack(losses_seq[k]).mean()

                    loss_list = [loss_weights_stage1[k](v, it) for k, v in losses_seq.items()]

                    loss = torch.stack(loss_list).sum()
                    loss.backward()
                    optimizer_stage1.step()

                    l_str = 'Optim. Stage 1, Step {}: Iter: {}, loss: {:.4f}'.format(it, i, loss.item())
                    for k, v in losses_seq.items():
                        l_str += ', {}: {:.4f}'.format(k, v.item())
                    # loop.set_description(l_str)

            loss_weights_stage2 = {
                'loss_body_kps2d': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
                'betas_decay_loss': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
                'bop_decay_loss': lambda cst, it: 10. ** 1 * cst / (1 + 10 * it),
                'bop_norm_loss': lambda cst, it: 10. ** -5 * cst / (1 + 10 * it),
                'lhp_norm_loss': lambda cst, it: 10. ** -2 * cst / (1 + 10 * it),
                'rhp_norm_loss': lambda cst, it: 10. ** -2 * cst / (1 + 10 * it),
                'loss_smo_v': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
                'loss_smo_J': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
                'loss_smo_trans': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
                'loss_smo_rot': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
                'loss_cam_R_decay': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
                'loss_cam_T_decay': lambda cst, it: 0 * 10. ** 0 * cst / (1 + 10 * it),
            }
            loss_functions.append(SMPLXDecayLoss(smplx_betas, smplx_body_poses, smplx_lhand_poses, smplx_rhand_poses).to(device))
            optimizer_stage2 = torch.optim.Adam([smplx_instance.betas, smplx_instance.cam_Ts, smplx_instance.body_pose, smplx_instance.lhand_pose, smplx_instance.rhand_pose], 
                lr=0.005, betas=(0.9, 0.999))
            iterations = 1
            steps_per_iter = 500
            for it in range(iterations):
                loop = tqdm(range(steps_per_iter))
                for i in loop:
                    optimizer_stage2.zero_grad()
                    losses_seq = {}
                    for idx in range(i % 2, n_seq, batch_size):

                        losses = {}
                        smplx_out = smplx_instance.forward(idx, batch_size)
                        for f in loss_functions:
                            losses.update(f(smplx_out, idx, batch_size))

                        for k, v in losses.items():
                            if k not in losses_seq:
                                losses_seq[k] = []
                            losses_seq[k].append(v)

                    for k in losses_seq:
                        losses_seq[k] = torch.stack(losses_seq[k]).mean()

                    loss_list = [loss_weights_stage2[k](v, it) for k, v in losses_seq.items()]

                    loss = torch.stack(loss_list).sum()
                    loss.backward()
                    optimizer_stage2.step()

                    l_str = 'Optim. Stage 2, Step {}: Iter: {}, loss: {:.4f}'.format(it, i, loss.item())
                    for k, v in losses_seq.items():
                        l_str += ', {}: {:.4f}'.format(k, v.item())
                    # loop.set_description(l_str)

            smplx_out = smplx_instance.forward(0, n_seq)
            for batch_idx, frame_id in enumerate(frame_ids):
                smplx_params_tuned.append({
                    'frame_id': frame_id,
                    'focal': smplx_params_all[batch_idx]['focal'],
                    'princpt': smplx_params_all[batch_idx]['princpt'],
                    'bbox': smplx_params_all[batch_idx]['bbox'],
                    'jaw_pose': smplx_params_all[batch_idx]['jaw_pose'],
                    'expression': smplx_params_all[batch_idx]['expression'],
                    'body_pose': smplx_out['body_pose'][batch_idx].detach().cpu().numpy().reshape(21, 3),
                    'left_hand_pose': smplx_out['lhand_pose'][batch_idx].detach().cpu().numpy().reshape(15, 3),
                    'right_hand_pose': smplx_out['rhand_pose'][batch_idx].detach().cpu().numpy().reshape(15, 3),
                    'betas': smplx_out['betas'][batch_idx].detach().cpu().numpy(),
                    'cam_R': smplx_out['cam_Rs'][batch_idx].detach().cpu().numpy(),
                    'cam_T': smplx_out['cam_Ts'][batch_idx].detach().cpu().numpy(),
                })


            save_pickle(os.path.join(smpl_out_dir, '{}_smplx.pkl'.format(hoi_id)), smplx_params_tuned)

        print('Video {:04d} done!'.format(video_idx))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize SMPL framewise.')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    parser.add_argument('--begin_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    args = parser.parse_args()

    optimize_all(args)
