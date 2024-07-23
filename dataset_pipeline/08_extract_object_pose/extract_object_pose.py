import os
import cv2
import pickle
import argparse
from tqdm import tqdm
import json
from glob import glob
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import random
from scipy.spatial.transform import Rotation
from torchvision import transforms
from pytorch3d.transforms import matrix_to_rotation_6d, axis_angle_to_matrix, rotation_6d_to_matrix

from datasets.object_corr_dataset import CORR_NORM
from models import Model
from object_instance import ObjectInstance


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


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


def smooth_sequence(sequence, scores, windows=5):
    n, d = sequence.shape
    scores = scores.reshape(n, 1)

    sequence = np.concatenate([np.zeros((windows // 2, d)), sequence, np.zeros((windows // 2, d))], axis=0) # [seq_n + windows - 1, n]
    confidence_score = np.concatenate([np.zeros((windows // 2, 1)), scores, np.zeros((windows // 2, 1))], axis=0) # [seq_n + windows - 1, 1]
    smooth_kps = np.stack([
        sequence[i: n + i, :] for i in range(windows)
    ], axis=0)    
    confidence_score = np.stack([
        confidence_score[i: n + i, :] for i in range(windows)
    ], axis=0)
    smooth_kps = (smooth_kps * confidence_score).sum(0) / (confidence_score.sum(0) + 1e-8)

    return smooth_kps


class SmoothLoss(nn.Module):

    def __init__(self, smpl_princpt, smpl_bboxes, scores):
        super().__init__()
        self.register_buffer('smpl_princpt', torch.tensor(smpl_princpt).float())
        self.register_buffer('smpl_bboxes', torch.tensor(smpl_bboxes).float())
        self.register_buffer('scores', torch.tensor(scores).float())


    def forward(self, object_dict, batch_idx, batch_size):
        object_v = object_dict['object_v']
        f = 5000
        u = object_v[:, :, 0] / (object_v[:, :, 2] + 1e-8) * f
        v = object_v[:, :, 1] / (object_v[:, :, 2] + 1e-8) * f
        object_v_reproj = torch.stack([u, v], dim=-1)

        b = object_v.shape[0]
        object_v_reproj = object_v_reproj / 256 * self.smpl_bboxes[batch_idx:batch_idx+batch_size, 2].reshape(b, 1, 1)
        object_v_reproj = object_v_reproj + self.smpl_princpt[batch_idx:batch_idx+batch_size, :2].reshape(b, 1, 2)

        scores = self.scores[batch_idx:batch_idx+batch_size]
        scores = scores[1:] * scores[:-1]
        if b > 1:
            # loss_object_v = (((object_v_reproj[:-1] - object_v_reproj[1:]) ** 2).sum(-1).mean(-1))
            # loss_object_v = loss_object_v * scores
            # loss_object_v = loss_object_v / (scores.sum() + 1e-8)
            loss_object_v = ((object_v_reproj[:-1] - object_v_reproj[1:]) ** 2).mean()
        else:
            loss_object_v = torch.zeros(1).to(object_v.device)

        R6d = object_dict['R6d']
        trans = object_dict['trans']
        if R6d.shape[0] > 1:
            # loss_r6d = (((R6d[:-1] - R6d[1:]) ** 2).sum(-1).reshape(b-1, -1) * scores) / (scores.sum() + 1e-8)
            # loss_trans = (((trans[:-1] - trans[1:]) ** 2).sum(-1).reshape(b-1, -1) * scores) / (scores.sum() + 1e-8)
            loss_r6d = ((R6d[:-1] - R6d[1:]) ** 2).mean()
            loss_trans = ((trans[:-1] - trans[1:]) ** 2).mean()
        else:
            loss_r6d = loss_trans = torch.zeros(1).to(R6d.device)
        return {
            'loss_smooth_r6d': loss_r6d,
            'loss_smooth_trans': loss_trans,
            'loss_smooth_obj_v': loss_object_v,
        }


OBJECT_KPS_SYM = {
    'barbell': [3, 2, 1, 0],
    'cello': np.arange(14).tolist(),
    'baseball': [0, 1, ],
    # 'tennis': [0, 1, 2, 3, 4, 5, 6],
    'skateboard': [4, 5, 6, 7, 0, 1, 2, 3],
    'basketball': [0],
    'yogaball': [0],
}


class SmoothKpsLoss(nn.Module):

    def __init__(self, object_name, smpl_bboxes, scores):
        super().__init__()
        self.object_name = object_name
        self.register_buffer('smpl_bboxes', torch.tensor(smpl_bboxes).float())
        self.register_buffer('scores', torch.tensor(scores).float())
        self.sym_indices = OBJECT_KPS_SYM[object_name]


    def forward(self, object_dict, batch_idx, batch_size):
        object_kps = object_dict['object_kps']
        f = 5000
        u = object_kps[:, :, 0] / (object_kps[:, :, 2] + 1e-8) * f
        v = object_kps[:, :, 1] / (object_kps[:, :, 2] + 1e-8) * f
        object_kps_reproj = torch.stack([u, v], dim=-1)

        b = object_kps.shape[0]
        object_kps_reproj = object_kps_reproj / 256 * self.smpl_bboxes[batch_idx:batch_idx+batch_size, 2].reshape(b, 1, 1)
        object_kps_reproj = object_kps_reproj + self.smpl_bboxes[batch_idx:batch_idx+batch_size, :2].reshape(b, 1, 2)
        object_kps_reproj_sym = object_kps_reproj[:, self.sym_indices]
        object_kps_reproj = torch.stack([object_kps_reproj, object_kps_reproj_sym], dim=1) # [b, 2, n, 2]

        if b > 1:
            loss_obj_kps = ((object_kps_reproj[:-1] - object_kps_reproj[1:]) ** 2).min(1)[0].mean()
        else:
            loss_obj_kps = torch.zeros(1).to(object_v.device)

        return {
            'loss_smooth_obj_kps': loss_obj_kps,
        }


class CorrLoss(nn.Module):

    def __init__(self, object_name, coor_x2d, coor_x3d, coor_mask):

        super().__init__()
        self.object_name = object_name
        coor_x2d = torch.tensor(coor_x2d).float() # [b, n, 2]
        coor_x3d = torch.tensor(coor_x3d).float() # [b, n, 3]
        coor_mask = torch.tensor(coor_mask).float() # [b, n, 1]
        coor_x3d_sym = coor_x3d.clone()
        if self.object_name == 'skateboard':
            coor_x3d_sym[:, :, 0] = - coor_x3d_sym[:, :, 0]
            coor_x3d_sym[:, :, 2] = - coor_x3d_sym[:, :, 2]
        elif self.object_name == 'tennis':
            coor_x3d_sym[:, :, 0] = - coor_x3d_sym[:, :, 0]
            coor_x3d_sym[:, :, 2] = - coor_x3d_sym[:, :, 2]
        elif self.object_name == 'baseball':
            coor_x3d_sym[:, :, 0] = - coor_x3d_sym[:, :, 0]
            coor_x3d_sym[:, :, 1] = - coor_x3d_sym[:, :, 1]
        elif self.object_name == 'barbell':
            coor_x3d_sym[:, :, 0] = - coor_x3d_sym[:, :, 0]
            coor_x3d_sym[:, :, 2] = - coor_x3d_sym[:, :, 2]
        coor_x3d = torch.stack([coor_x3d, coor_x3d_sym], dim=1) # [b, 2, n, 3]

        self.register_buffer('coor_x2d', coor_x2d)
        self.register_buffer('coor_x3d', coor_x3d)
        self.register_buffer('coor_mask', coor_mask)


    def forward(self, object_dict, batch_idx, batch_size):
        rotmat = object_dict['rotmat']
        trans = object_dict['trans']
        f = 5000
        b = trans.shape[0]

        coor_x3d = self.coor_x3d[batch_idx:batch_idx+batch_size]
        coor_x2d = self.coor_x2d[batch_idx:batch_idx+batch_size]
        coor_mask = self.coor_mask[batch_idx:batch_idx+batch_size]

        coor_x3d = coor_x3d @ rotmat.unsqueeze(1).transpose(-1, -2) + trans.reshape(b, 1, 1, 3)
        u = coor_x3d[:, :, :, 0] / (coor_x3d[:, :, :, 2] + 1e-8) * f
        v = coor_x3d[:, :, :, 1] / (coor_x3d[:, :, :, 2] + 1e-8) * f
        coor_x2d_reproj = torch.stack([u, v], dim=-1)
        loss_coor = ((coor_x2d_reproj - coor_x2d.unsqueeze(1)) ** 2).sum(-1)
        loss_coor, indices = torch.sort(loss_coor, dim=2)

        loss_coor = (loss_coor * coor_mask.reshape(b, 1, -1)).mean(-1)
        loss_coor, _ = loss_coor.min(1)

        return {
            'loss_corr': loss_coor.mean(),
        }


class ObjectImageSeqDataset:

    def __init__(self, root_dir, image_ids, bboxes, out_res=256, coor_res=64):
        self.root_dir = root_dir
        self.image_ids = image_ids
        self.bboxes = bboxes
        assert len(self.image_ids) == len(self.bboxes)

        self.out_res = out_res
        self.coor_res = coor_res
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


    def __len__(self, ):
        return len(self.image_ids)


    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        bbox = self.bboxes[idx]

        image = cv2.imread(os.path.join(self.root_dir, '{}.jpg'.format(img_id)))
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



def extract_object_pose(args):
    device = torch.device('cuda')
    object_name = args.root_dir.split('/')[-1]
    tracking_results_dir = os.path.join(args.root_dir, 'hoi_tracking')

    model = Model(num_kps=12).to(device)
    model.load_checkpoint('./weights/model_{}_stage1.pth'.format(object_name))
    model.eval()

    for video_idx in range(args.begin_idx, args.end_idx):
        video_id = '{:04d}'.format(video_idx)
        try:
            tracking_results = load_pickle(os.path.join(tracking_results_dir, '{}_tracking.pkl'.format(video_id)))
        except:
            continue

        object_pose_dir = os.path.join(args.root_dir, 'object_pose', video_id)
        os.makedirs(object_pose_dir, exist_ok=True)
        print('Found {} instances.'.format(len(tracking_results['hoi_instances'])))

        for hoi_instance in tracking_results['hoi_instances']:
            hoi_id = hoi_instance['hoi_id']
            # if os.path.exists(os.path.join(object_pose_dir, '{}_obj_RT.pkl'.format(hoi_id))):
            #     continue

            image_ids, obj_bboxes = [], []
            for item in hoi_instance['sequences']:
                image_ids.append(item['frame_id'])
                if item['object_bbox'] is not None:
                    obj_bboxes.append(item['object_bbox'])
                else:
                    obj_bboxes.append(np.array([0, 0, 256, 256]))
            ##########################
            image_ids = image_ids
            ##########################

            obj_bboxes = np.stack(obj_bboxes)
            smpl_bboxes, smpl_focal, smpl_princpt = [], [], []
            try:
                smpl_params = load_pickle(os.path.join(args.root_dir, 'smpler_x', video_id, '{}_smplx.pkl'.format(hoi_id)))
            except:
                continue
            for item in smpl_params:
                smpl_bboxes.append(item['bbox'])
                smpl_focal.append(item['focal'])
                smpl_princpt.append(item['princpt'])
            smpl_bboxes = np.stack(smpl_bboxes)
            smpl_focal = np.stack(smpl_focal)
            smpl_princpt = np.stack(smpl_princpt)
            smpl_s = smpl_bboxes[:, 2]

            n_box = smpl_bboxes.shape[0]

            assert len(obj_bboxes) == len(smpl_bboxes)

            dataset = ObjectImageSeqDataset(root_dir=os.path.join(args.root_dir, 'images_temp', video_id), image_ids=image_ids, bboxes=obj_bboxes, out_res=224, coor_res=64)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=8, shuffle=False, drop_last=False)
            all_corr_maps = []
            obj_box_centers, obj_box_sizes = [], []

            print('Extracting corresponding maps.')
            for item in tqdm(dataloader):
                images, obj_box_c, obj_box_s = item
                images = images.to(device)
                corr_maps = model.inference_step(images) # numpy, [b, 4, h, w]
                all_corr_maps.append(corr_maps)
                obj_box_centers.append(obj_box_c)
                obj_box_sizes.append(obj_box_s)
            all_corr_maps = np.concatenate(all_corr_maps)
            obj_box_centers = torch.cat(obj_box_centers)
            obj_box_sizes = torch.cat(obj_box_sizes)

            b, _, corr_h, corr_w = all_corr_maps.shape
            all_corr_maps = all_corr_maps.transpose(0, 2, 3, 1)
            corr_norm = CORR_NORM[object_name]
            coor_x3d = all_corr_maps[:, :, :, :3] * np.array(corr_norm).reshape(1, 1, 1, 3)

            coor_mask = all_corr_maps[:, :, :, 3:]
            grid_2d = torch.arange(corr_h).float()
            ys, xs = torch.meshgrid(grid_2d, grid_2d) # (h, w)
            grid_2d = torch.stack([xs, ys], dim=2).unsqueeze(0).repeat(b, 1, 1, 1).reshape(b, -1, 2) # (b, h * w, 2)
            stride = obj_box_sizes / corr_h
            stride = stride.reshape(b, 1, 1).float()
            x1 = obj_box_centers[:, 0] - obj_box_sizes / 2
            y1 = obj_box_centers[:, 1] - obj_box_sizes / 2
            begin_point = torch.stack([x1, y1], dim=1)
            begin_point = begin_point.reshape(b, 1, 2).float()
            coor_x2d = grid_2d * stride + begin_point # [b, h*w, 2]
            coor_x2d = coor_x2d.numpy()
            coor_x2d = (coor_x2d - smpl_princpt.reshape(n_box, 1, 2)) / smpl_bboxes[:, 2].reshape(n_box, 1, 1) * 192

            coor_x3d = coor_x3d.reshape(b, -1, 3) # [b, n_init, h*w, 3]
            coor_mask = coor_mask.reshape(b, -1).clip(0, 1)

            coor_x3d_sym = coor_x3d.copy()
            if object_name == 'skateboard':
                coor_x3d_sym[:, :, 0] = - coor_x3d_sym[:, :, 0]
                coor_x3d_sym[:, :, 2] = - coor_x3d_sym[:, :, 2]
            elif object_name == 'tennis':
                coor_x3d_sym[:, :, 0] = - coor_x3d_sym[:, :, 0]
                coor_x3d_sym[:, :, 2] = - coor_x3d_sym[:, :, 2]
            elif object_name == 'baseball':
                coor_x3d_sym[:, :, 0] = - coor_x3d_sym[:, :, 0]
                coor_x3d_sym[:, :, 1] = - coor_x3d_sym[:, :, 1]
            elif object_name == 'barbell':
                coor_x3d_sym[:, :, 0] = - coor_x3d_sym[:, :, 0]
                coor_x3d_sym[:, :, 2] = - coor_x3d_sym[:, :, 2]

            # for i in range(b):
            #     if coor_mask[i].sum() < 100:
            #         coor_mask[i] *= 0
            binary_mask = coor_mask > 0.5
            ns = binary_mask.sum(-1)
            mask_corr_min = np.zeros_like(binary_mask)
            for i in range(b):
                mask_corr_min[i, :int(ns[i] * 0.8)] = 1

            K = np.eye(3)
            K[0, 0] = K[1, 1] = 5000
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)

            rot_aa = []
            trans = []
            R_vector_prev = np.zeros(3)
            T_vector_prev = np.zeros(3)
            for x2d_np_, x3d_np_, x3d_np_sym_, mask_np_ in zip(coor_x2d, coor_x3d, coor_x3d_sym, binary_mask):
                try:
                    _, R_vector, T_vector, _ = cv2.solvePnPRansac(
                        x3d_np_[mask_np_], x2d_np_[mask_np_], K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

                    _, R_vector_sym, T_vector_sym, _ = cv2.solvePnPRansac(
                        x3d_np_sym_[mask_np_], x2d_np_[mask_np_], K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
                    if R_vector_prev.sum() != 0:
                        quat_prev = Rotation.from_rotvec(np.array(rot_aa[-5:]).reshape(-1, 3)).as_quat()
                        quat = Rotation.from_rotvec(R_vector.reshape(1, -1)).as_quat()
                        quat_sym = Rotation.from_rotvec(R_vector_sym.reshape(1, -1)).as_quat()
                        quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
                        quat_sym = quat_sym / np.linalg.norm(quat_sym, axis=-1, keepdims=True)
                        quat_prev = quat_prev / np.linalg.norm(quat_prev, axis=-1, keepdims=True)

                        R_dist = np.abs((quat * quat_prev).sum(axis=-1)).mean()
                        R_dist_sym = np.abs((quat_sym * quat_prev).sum(axis=-1)).mean()
                        if R_dist_sym > R_dist:
                            R_vector = R_vector_sym
                            T_vector = T_vector_sym

                    R_vector_prev = R_vector
                    T_vector_prev = T_vector
                except:
                    R_vector = R_vector_prev
                    T_vector = T_vector_prev
                q = R_vector.reshape(-1)
                rot_aa.append(q)
                trans.append(T_vector.reshape(-1))

            windows = 7
            n = len(trans)
            trans_mean = np.concatenate([np.zeros((windows // 2, 3)), np.stack(trans), np.zeros((windows // 2, 3))], axis=0) # [seq_n + windows - 1, n]
            trans_mean = np.stack([trans_mean[i: n + i, :] for i in range(windows)], axis=0)
            score = np.concatenate([np.zeros((windows // 2, 1)), np.ones((n, 1)), np.zeros((windows // 2, 1))], axis=0) # [seq_n + windows - 1, 1]
            score = np.stack([score[i: n + i, :] for i in range(windows)], axis=0)

            trans_mean = (trans_mean * score).mean(0)

            for idx, t in enumerate(trans):
                if np.abs(t - trans_mean[idx]).sum() > 20: # 10 for barbell and cello
                    trans[idx] = trans_mean[idx]
                    # rot_aa[idx] = rot_mean.reshape(-1)
            trans = np.array(trans)

            rot_aa = torch.tensor(rot_aa)
            trans = torch.tensor(trans)

            ##############################################################################
            # smooth sequence
            scores = np.ones(b,)
            # scores[binary_mask.sum(-1) != 0] = 1
            R6d = matrix_to_rotation_6d(axis_angle_to_matrix(rot_aa)).numpy()
            trans = trans.numpy()
            # R6d = smooth_sequence(R6d, scores, windows=7)
            # trans = smooth_sequence(trans, scores, windows=7)
            # scores = smooth_sequence(scores.reshape(-1, 1), scores, windows=7)

            ##############################################################################

            b = all_corr_maps.shape[0]
            R6d = torch.tensor(R6d).to(torch.float32)
            T = torch.tensor(trans).reshape(b, 3).to(torch.float32)
            object_instance = ObjectInstance(object_name, R6d, T).to(device)
            optimizer = object_instance.get_optimizer(lr=2e-2)

            loss_functions = [
                SmoothLoss(smpl_princpt, smpl_bboxes, scores).to(device),
                CorrLoss(object_name, coor_x2d, coor_x3d, mask_corr_min).to(device),
            ]
            # loss_weights = {
            #     'loss_smooth_trans': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
            #     'loss_smooth_r6d': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
            #     'loss_smooth_obj_v': lambda cst, it: 10. ** -3 * cst / (1 + 10 * it),
            #     'loss_corr': lambda cst, it: 10 ** 0 * cst / (1 + 10 * it),
            # } # basketball, barbell, cello
            loss_weights = {
                'loss_smooth_trans': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
                'loss_smooth_r6d': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
                'loss_smooth_obj_v': lambda cst, it: 10. ** -3 * cst / (1 + 10 * it),
                'loss_corr': lambda cst, it: 10 ** 0 * cst / (1 + 10 * it),
            } # skateboard

            ################ do post optimization here ####################
            iterations = 3
            steps_per_iter_max = 1000
            seq_len = all_corr_maps.shape[0]
            batch_size = 64
            for it in range(iterations):
                if it == 0:
                    object_instance.R6d.requires_grad_(False)
                else:
                    object_instance.R6d.requires_grad_(True)
                loop = tqdm(range(steps_per_iter_max))
                stable_counting = 0
                loss_prev = 0
                for i in loop:
                    optimizer.zero_grad()
                    total_loss = 0
                    for batch_idx in range(i % 2, seq_len, batch_size):
                        object_outputs = object_instance.forward(batch_idx, batch_size)
                        losses = {}
                        for f in loss_functions:
                            losses.update(f(object_outputs, batch_idx, batch_size))
                        loss_list = [loss_weights[k](v.mean(), it) for k, v in losses.items()]
                        total_loss += torch.stack(loss_list).sum()
                    # total_loss = total_loss / seq_len
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_([object_instance.R6d, object_instance.trans], 0.1)
                    optimizer.step()

                    if np.abs(loss_prev - total_loss.item() / (seq_len * 2)) < 0.0001 / 10 ** it:
                        stable_counting += 1
                        loss_prev = total_loss.item() / (seq_len * 2)

                    # if stable_counting > 10:
                    #     break

                    l_str = 'Optim. Step {}: Iter: {}, loss: {:.4f}'.format(it, i, total_loss.item() / (seq_len * 2))
                    for k, v in losses.items():
                        l_str += ', {}: {:.4f}'.format(k, v.mean().detach().item())
                        loop.set_description(l_str)

            object_outputs = object_instance.forward(0, seq_len)
            R = object_outputs['rotmat'].detach().cpu().numpy()
            T = object_outputs['trans'].detach().cpu().numpy()

            ##############################################################################
            # smooth sequence
            # scores = np.zeros(b,)
            # scores[trans.sum(-1) != 0] = 1
            # R6d = matrix_to_rotation_6d(torch.tensor(R)).numpy()
            # trans = T
            # R6d = smooth_sequence(R6d, scores)
            # trans = smooth_sequence(trans, scores)

            # R = rotation_6d_to_matrix(torch.tensor(R6d)).numpy()
            # T = trans
            ##############################################################################

            object_params_all = []
            for idx, frame_id in enumerate(image_ids):
                object_params = {}
                object_params['frame_id'] = frame_id
                object_params['trans'] = T[idx]
                object_params['rotmat'] = R[idx]
                object_params['success'] = scores[idx]
                object_params['smpl_princpt'] = smpl_princpt[idx]
                object_params['smpl_focal'] = smpl_focal[idx]
                object_params['smpl_bboxes'] = smpl_bboxes[idx]

                object_params_all.append(object_params)

            save_pickle(os.path.join(object_pose_dir, '{}_obj_RT.pkl'.format(hoi_id)), object_params_all)
            print(os.path.join(object_pose_dir, '{}_obj_RT.pkl'.format(hoi_id)))
            # exit(0)
    print('Video {:04d} done!'.format(video_idx))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    parser.add_argument('--begin_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    args = parser.parse_args()

    extract_object_pose(args)
