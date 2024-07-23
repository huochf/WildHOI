import os
import sys
sys.path.insert(0, '../08_extract_object_pose')
sys.path.append('/inspurfs/group/wangjingya/huochf/Thesis/')
import argparse
import numpy as np
import cv2
import trimesh
import json
import pickle
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F

from smplx import SMPLX
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix

from hoi_recon.datasets.utils import load_pickle, save_pickle, load_json

from hoi_instance import HOIInstance
from models import Model as ObjectModel
from object_image_dataset import ObjectImageDataset, CORR_NORM
from optim_losses import ObjectScaleLoss, ObjectCoorLoss, SMPLDecayLoss, SMPLKPSLoss, CamDecayLoss, ContactLoss, HOCollisionLoss


def load_J_regressor(path):
    data = np.loadtxt(path)

    with open(path, 'r') as f:
        shape = f.readline().split()[1:]
    J_regressor = np.zeros((int(shape[0]), int(shape[1])), dtype=np.float32)
    for i, j, v in data:
        J_regressor[int(i), int(j)] = v
    return J_regressor


class HOIInstance(nn.Module):

    def __init__(self, smpl, smpl_betas=None, smpl_body_pose=None, lhand_pose=None, rhand_pose=None,
        rot_angle=None, obj_rel_trans=None, obj_rel_rotmat=None, hoi_trans=None, hoi_rot6d=None, object_scale=None):
        super(HOIInstance, self).__init__()

        self.smpl = smpl
        npose = 21 # SMPLX

        if smpl_betas is not None:
            batch_size = smpl_betas.shape[0]
        else:
            batch_size = 1

        bicycle_front = trimesh.load('../data/objects/bicycle_front.ply', process=False)
        bicycle_back = trimesh.load('../data/objects/bicycle_back.ply', process=False)
        bicycle_front_v = np.array(bicycle_front.vertices)
        bicycle_back_v = np.array(bicycle_back.vertices)

        bicycle_front = torch.from_numpy(bicycle_front.vertices).float().unsqueeze(0).repeat(batch_size, 1, 1)
        bicycle_back = torch.from_numpy(bicycle_back.vertices).float().unsqueeze(0).repeat(batch_size, 1, 1)
        self.register_buffer('bicycle_front', bicycle_front)
        self.register_buffer('bicycle_back', bicycle_back)

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
        bicycle_front_kps = torch.from_numpy(bicycle_front_kps).float().unsqueeze(0).repeat(batch_size, 1, 1)
        bicycle_back_kps = torch.from_numpy(bicycle_back_kps).float().unsqueeze(0).repeat(batch_size, 1, 1)
        self.register_buffer('bicycle_front_kps', bicycle_front_kps)
        self.register_buffer('bicycle_back_kps', bicycle_back_kps)

        rot_axis_begin = bicycle_front[0, bicycle_front_kps_indices['5']].mean(0).reshape(1, 3).repeat(batch_size, 1)
        rot_axis_end = bicycle_front[0, bicycle_front_kps_indices['6']].mean(0).reshape(1, 3).repeat(batch_size, 1)
        rot_axis = rot_axis_end - rot_axis_begin
        rot_axis = rot_axis / torch.sqrt((rot_axis ** 2).sum(-1)).reshape(batch_size, 1)
        self.register_buffer('rot_axis_begin', rot_axis_begin)
        self.register_buffer('rot_axis', rot_axis)

        if smpl_betas is not None:
            self.smpl_betas = nn.Parameter(smpl_betas.reshape(batch_size, 10))
        else:
            self.smpl_betas = nn.Parameter(torch.zeros(batch_size, 10, dtype=torch.float32))

        if smpl_body_pose is not None:
            self.smpl_body_pose = nn.Parameter(smpl_body_pose.reshape(batch_size, npose, 3))
        else:
            self.smpl_body_pose = nn.Parameter(torch.zeros(batch_size, npose, 3))

        if lhand_pose is not None:
            self.lhand_pose = nn.Parameter(lhand_pose.reshape(batch_size, 15, 3))
        else:
            self.lhand_pose = nn.Parameter(torch.zeros(batch_size, 15, 3))

        if rhand_pose is not None:
            self.rhand_pose = nn.Parameter(rhand_pose.reshape(batch_size, 15, 3))
        else:
            self.rhand_pose = nn.Parameter(torch.zeros(batch_size, 15, 3))

        if obj_rel_trans is not None:
            self.obj_rel_trans = nn.Parameter(obj_rel_trans.reshape(batch_size, 3))
        else:
            self.obj_rel_trans = nn.Parameter(torch.zeros(batch_size, 3, dtype=torch.float32))

        if obj_rel_rotmat is not None:
            self.obj_rel_rot6d = nn.Parameter(matrix_to_rotation_6d(obj_rel_rotmat.reshape(batch_size, 3, 3)))
        else:
            self.obj_rel_rot6d = nn.Parameter(matrix_to_rotation_6d(torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).repeat(batch_size, 1, 1)))

        if hoi_trans is not None:
            self.hoi_trans = nn.Parameter(hoi_trans.reshape(batch_size, 3))
        else:
            self.hoi_trans = nn.Parameter(torch.zeros(batch_size, 3, dtype=torch.float32))

        if hoi_rot6d is not None:
            self.hoi_rot6d = nn.Parameter(hoi_rot6d.reshape(batch_size, 6))
        else:
            self.hoi_rot6d = nn.Parameter(matrix_to_rotation_6d(torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).repeat(batch_size, 1, 1)))

        if object_scale is not None:
            self.object_scale = nn.Parameter(object_scale.reshape(batch_size, 1, 1).float())
        else:
            self.object_scale = nn.Parameter(torch.ones(1).reshape(1, 1, 1).repeat(batch_size, 1, 1).float())
        if rot_angle is not None:
            self.rot_angle = nn.Parameter(torch.zeros(batch_size, dtype=torch.float32))
        else:
            self.rot_angle = nn.Parameter(rot_angle)

        wholebody_regressor = np.load('../data/smpl/smplx_wholebody_regressor.npz')
        self.register_buffer('wholebody_regressor', torch.tensor(wholebody_regressor['wholebody_regressor']).float())


    def forward(self, ):
        b = self.smpl_betas.shape[0]

        global_orient = torch.zeros((b, 3), dtype=self.smpl_betas.dtype, device=self.smpl_betas.device)
        jaw_pose = torch.zeros((b, 3), dtype=self.smpl_betas.dtype, device=self.smpl_betas.device)
        leye_pose = torch.zeros((b, 3), dtype=self.smpl_betas.dtype, device=self.smpl_betas.device)
        reye_pose = torch.zeros((b, 3), dtype=self.smpl_betas.dtype, device=self.smpl_betas.device)
        expression = torch.zeros((b, 10), dtype=self.smpl_betas.dtype, device=self.smpl_betas.device)

        smplx_out = self.smpl(betas=self.smpl_betas,
                               body_pose=self.smpl_body_pose,
                               left_hand_pose=self.lhand_pose,
                               right_hand_pose=self.rhand_pose,
                               global_orient=global_orient,
                               leye_pose=leye_pose,
                               reye_pose=reye_pose,
                               jaw_pose=jaw_pose,
                               expression=expression)
        smplx_v = smplx_out.vertices
        smplx_J = smplx_out.joints
        smplx_v_centered = smplx_v - smplx_J[:, :1]
        smplx_J_centered = smplx_J - smplx_J[:, :1]

        hoi_rotmat = rotation_6d_to_matrix(self.hoi_rot6d)
        smplx_v = smplx_v_centered @ hoi_rotmat.transpose(2, 1) + self.hoi_trans.reshape(b, 1, 3)
        smplx_J = smplx_J_centered @ hoi_rotmat.transpose(2, 1) + self.hoi_trans.reshape(b, 1, 3)

        front_rotmat = axis_angle_to_matrix(self.rot_axis.reshape(-1, 3) * self.rot_angle.reshape(-1, 1))
        front_v = self.bicycle_front - self.rot_axis_begin.view(-1, 1, 3)
        front_v = front_v @ front_rotmat.transpose(2, 1)
        front_v = front_v + self.rot_axis_begin.view(-1, 1, 3)
        object_v = torch.cat([front_v, self.bicycle_back], dim=1) # [b, n, 3]

        front_kps = self.bicycle_front_kps - self.rot_axis_begin.view(-1, 1, 3)
        front_kps = front_kps @ front_rotmat.transpose(2, 1)
        front_kps = front_kps + self.rot_axis_begin.view(-1, 1, 3)
        object_kps = torch.cat([front_kps, self.bicycle_back_kps], dim=1) # [b, n, 3]

        scale = self.object_scale.reshape(b, 1, 1)
        object_v_org = object_v * scale
        object_kps_org = object_kps * scale

        object_rel_rotmat = rotation_6d_to_matrix(self.obj_rel_rot6d)
        object_rotmat = hoi_rotmat.detach() @ object_rel_rotmat
        object_trans = (hoi_rotmat.detach() @ self.obj_rel_trans.reshape(b, 3, 1)).squeeze(-1) + self.hoi_trans.detach()
        object_v = object_v_org @ object_rotmat.transpose(2, 1) + object_trans.reshape(b, 1, 3)
        object_kps = object_kps_org @ object_rotmat.transpose(2, 1) + object_trans.reshape(b, 1, 3)
        object_v_centered = object_v_org @ object_rel_rotmat.transpose(2, 1) + self.obj_rel_trans.reshape(b, 1, 3)
        object_kps_centered = object_kps_org @ object_rel_rotmat.transpose(2, 1) + self.obj_rel_trans.reshape(b, 1, 3)

        wholebody_kps = self.wholebody_regressor.unsqueeze(0) @ smplx_v # [b, 65, 3]

        results = {
            'betas': self.smpl_betas,
            'body_pose': self.smpl_body_pose,
            'lhand_pose': self.lhand_pose,
            'rhand_pose': self.rhand_pose,
            'smplx_v': smplx_v,
            'wholebody_kps': wholebody_kps,
            'smplx_J': smplx_J,
            'smplx_J_centered': smplx_J_centered,
            'smplx_v_centered': smplx_v_centered,

            'hoi_rot6d': self.hoi_rot6d,
            'hoi_rotmat': hoi_rotmat,
            'hoi_trans': self.hoi_trans,

            'object_scale': self.object_scale,
            'object_rot_angle': self.rot_angle,
            'object_rel_rotmat': object_rel_rotmat,
            'object_rel_trans': self.obj_rel_trans,
            'object_rotmat': object_rotmat,
            'object_trans': object_trans,
            'object_v': object_v,
            'object_kps': object_kps,
            'object_v_centered': object_v_centered,
            'object_kps_centered': object_kps_centered,
        }
        return results


OBJECT_KPS_N = {
    'barbell': 4,
    'cello': 14,
    'baseball': 2,
    'tennis': 7,
    'skateboard': 8,
    'basketball': 1,
    'yogaball': 1,
    'bicycle': 10,
    'violin': 14,
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
    img_patch = cv2.warpAffine(image, img_trans, (int(out_size), int(out_size)), flags=cv2.INTER_NEAREST)
    return img_patch, img_trans


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
    elif object_name == 'violin':
        object_file = '../data/objects/violin_body_keypoints.json'
    with open(object_file, 'r') as f:
        indices = json.load(f)

    if object_name == 'baseball':
        indices = {'1': indices['1'], '5': indices['5']}
    elif object_name == 'barbell':
        indices = {'1': indices['1'], '2': indices['2'], '3': indices['3'], '4': indices['4'],}

    return indices


OBJECT_KPS_PERM = {
    'barbell': [[0, 1, 2, 3], [3, 2, 1, 0]],
    'cello': [np.arange(14).tolist(), ],
    'baseball': [[0, 1, ],],
    'tennis': [[0, 1, 2, 3, 4, 5, 6]],
    'skateboard': [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 0, 1, 2, 3]],
    'basketball': [[0]],
    'yogaball': [[0]],
    'bicycle': [np.arange(10).tolist(), ],
    'violin': [np.arange(14).tolist(), ],
}


def get_gt_object_coor(root_dir, image_ids, object_name, smpl_princpt, smpl_bboxes, device):
    coor_x3d_all, coor_x2d_all, coor_mask_all = [], [], []

    print('loadding object coordinate maps.')
    for image_id in tqdm(image_ids):
        video_id, hoi_id, frame_id = image_id.split('_')
        mask_path = os.path.join(root_dir, object_name, 'object_annotations', 'corr', '{}_{}_{}-label.png'.format(video_id, frame_id, hoi_id))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img_h, img_w = mask.shape

        coor_load = load_pickle(os.path.join(root_dir, object_name, 'object_annotations', 'corr', '{}_{}_{}-corr.pkl'.format(video_id, frame_id, hoi_id)))
        u, l, h, w = coor_load['u'], coor_load['l'], coor_load['h'], coor_load['w']
        coor = np.zeros((img_h, img_w, 3)).astype(np.float32)
        coor[u:(u+h), l:(l+w), :] = coor_load['coor']

        ys, xs = torch.meshgrid(torch.arange(img_h).float(), torch.arange(img_w).float()) # (h, w)
        coor_x2d = torch.stack([xs, ys], dim=2).reshape(img_h, img_w, 2) # (h * w, 2)
        coor_x2d = coor_x2d.numpy()

        coor_x2d = np.concatenate([coor_x2d, mask.reshape(img_h, img_w, 1)], axis=2)

        cx, cy, s = l + w / 2, u + h / 2, max(h, w)
        coor_x3d, _ = generate_image_patch(coor, cx, cy, s, 128, 0., [1., 1., 1.])
        coor_x2d, _ = generate_image_patch(coor_x2d, cx, cy, s, 128, 0., [1., 1., 1.])

        coor_x3d_all.append(coor_x3d.reshape(-1, 3))
        coor_x2d_all.append(coor_x2d[:, :, :2].reshape(-1, 2))
        coor_mask_all.append(coor_x2d[:, :, 2].reshape(-1))

    coor_x3d = np.stack(coor_x3d_all, axis=0)
    coor_x2d = np.stack(coor_x2d_all, axis=0)
    coor_x2d = (coor_x2d - smpl_princpt.reshape(-1, 1, 2)) / smpl_bboxes[:, 2].reshape(-1, 1, 1) * 192
    coor_mask = np.stack(coor_mask_all, axis=0)

    binary_mask = coor_mask > 0.5
    ns = binary_mask.sum(-1)
    mask_corr_min = np.zeros_like(binary_mask)
    for i in range(len(image_ids)):
        mask_corr_min[i, :int(ns[i] * 0.98)] = 1

    return coor_x2d, coor_x3d, mask_corr_min


def optimize_all(args):
    device = torch.device('cuda')
    root_dir = args.root_dir
    object_name = args.object

    assert object_name == 'bicycle'

    tracking_results_dir = os.path.join(args.root_dir, args.object, 'hoi_tracking')

    smplx = SMPLX('/public/home/huochf/projects/3D_HOI/hoiYouTube/data/smpl/smplx/', gender='neutral', use_pca=False)

    smpl_indices = load_json('../data/contact/smplx_mesh_contact_{}.json'.format(object_name))

    bicycle_front = trimesh.load('../data/objects/bicycle_front.ply', process=False)
    bicycle_front_indices = load_json('../data/contact/bicycle_front_contact.json')
    bicycle_back_indices = load_json('../data/contact/bicycle_back_contact.json')
    object_indices = {'1': bicycle_front_indices['1'], '2': bicycle_front_indices['2'], 
                      '3': np.array(bicycle_back_indices['1']) + bicycle_front.vertices.shape[0], 
                      '4': np.array(bicycle_back_indices['2']) + bicycle_front.vertices.shape[0]}

    contact_annotations = load_json('./annotation_contact_v2/{}_contact_annotations.json'.format(object_name))

    with open('/storage/data/huochf/HOIYouTube/train_test_split_v2.json'.format(object_name), 'r') as f:
        train_test_split = json.load(f)

    hoi_out_dir = './hoi_recon_with_contact'
    os.makedirs(hoi_out_dir, exist_ok=True)

    image_ids = []
    wholebody_kps = []
    smplx_betas = []
    smplx_betas_init = []
    smplx_body_poses = []
    smplx_body_poses_init = []
    smplx_lhand_poses = []
    smplx_lhand_poses_init = []
    smplx_rhand_poses = []
    smplx_rhand_poses_init = []
    cam_Rs = []
    cam_Ts = []
    object_bboxes = []
    object_rotmat = []
    object_trans = []
    object_rot_angle = []
    hoi_bboxes = []
    smpl_focal = []
    smpl_princpt = []
    smpl_bboxes = []

    annotated_images = []
    if os.path.exists('./annotation_hoi/{}/test'.format(object_name)):
        for file in os.listdir('./annotation_hoi/{}/test'.format(object_name)):
            annotated_images.append(file.split('.')[0])

    print(len(annotated_images), len(train_test_split[object_name]['test_frames']))
    # exit(0)

    print('collect data (keypoints, coor, ) ...')
    for img_id in tqdm(train_test_split[object_name]['test_frames']):
        if img_id in annotated_images:
            continue

        video_id, hoi_id, frame_id = img_id.split('_')

        mask_path = os.path.join(root_dir, object_name, 'object_annotations', 'corr', '{}_{}_{}-label.png'.format(video_id, frame_id, hoi_id))
        if not os.path.exists(mask_path):
            continue

        tracking_results = load_pickle(os.path.join(tracking_results_dir, '{}_tracking.pkl'.format(video_id)))

        for hoi_instance in tracking_results['hoi_instances']:
            if hoi_instance['hoi_id'] != hoi_id:
                continue

            wholebody_kps_list = load_pickle(os.path.join(root_dir, args.object, 'wholebody_kps_refined', video_id, '{}_wholebody_kps.pkl'.format(hoi_id)))
            smplx_init_params_all = load_pickle(os.path.join(root_dir, args.object, 'smpler_x', video_id, '{}_smplx.pkl'.format(hoi_id)))
            smplx_params_all = load_pickle(os.path.join(root_dir, args.object, 'smplx_tuned', video_id, '{}_smplx.pkl'.format(hoi_id)))
            try:
                object_RT = load_pickle(os.path.join(root_dir, args.object, 'object_pose', video_id, '{}_obj_RT.pkl'.format(hoi_id)))
            except:
                continue

            for tracking_item, object_RT_item, wholebody_kps_item, smplx_params_item, smplx_params_init_items in zip(
                hoi_instance['sequences'], object_RT, wholebody_kps_list, smplx_params_all, smplx_init_params_all):
                if tracking_item['frame_id'] != frame_id:
                    continue

                assert wholebody_kps_item['frame_id'] == smplx_params_item['frame_id']
                assert tracking_item['frame_id'] == wholebody_kps_item['frame_id']

                person_bbox = tracking_item['person_bbox']
                object_bbox = tracking_item['object_bbox']
                if person_bbox is None:
                    person_bbox = [0, 0, 256, 256]
                if object_bbox is None:
                    object_bbox = person_bbox
                x1, y1, x2, y2 = person_bbox
                _x1, _y1, _x2, _y2 = object_bbox
                x1 = min(x1, _x1)
                y1 = min(y1, _y1)
                x2 = max(x2, _x2)
                y2 = max(y2, _y2)
                box_size = max(y2 - y1, x2 - x1) * 1.2
                box_size = max(1, box_size)
                box_cx, box_cy = (x2 + x1) / 2, (y2 + y1) / 2
                hoi_box_cxcys = (box_cx, box_cy, box_size)

                object_bboxes.append(object_bbox)
                hoi_bboxes.append(hoi_box_cxcys)

                object_rotmat.append(object_RT_item['rotmat'])
                object_trans.append(object_RT_item['trans'])
                object_rot_angle.append(object_RT_item['rot_angle'])

                image_ids.append(img_id)
                smplx_betas.append(smplx_params_item['betas'])
                smplx_betas_init.append(smplx_params_init_items['betas'])
                smplx_body_poses.append(smplx_params_item['body_pose'])
                smplx_lhand_poses.append(smplx_params_item['left_hand_pose'])
                smplx_rhand_poses.append(smplx_params_item['right_hand_pose'])
                smplx_body_poses_init.append(smplx_params_init_items['body_pose'])
                smplx_lhand_poses_init.append(smplx_params_init_items['left_hand_pose'])
                smplx_rhand_poses_init.append(smplx_params_init_items['right_hand_pose'])
                cam_Rs.append(smplx_params_item['cam_R'])
                cam_Ts.append(smplx_params_item['cam_T'].reshape(3, ))
                princpt = np.array(smplx_params_item['princpt'])
                focal = np.array(smplx_params_item['focal'])
                smpl_focal.append(focal)
                smpl_princpt.append(princpt)
                smpl_bboxes.append(smplx_params_item['bbox'])

                keypoints = wholebody_kps_item['keypoints']
                keypoints[:, :2] = (keypoints[:, :2] - princpt.reshape(1, 2)) / focal.reshape(1, 2)
                keypoints[keypoints[:, 2] < 0.5] = 0
                wholebody_kps.append(keypoints)

    print('loaded {} items'.format(len(image_ids)))

    object_rotmat = torch.from_numpy(np.array(object_rotmat)).float().to(device)
    object_trans = torch.from_numpy(np.array(object_trans)).reshape(-1, 3).float().to(device)
    object_rot_angle = torch.from_numpy(np.array(object_rot_angle)).reshape(-1).float().to(device)
    smpl_princpt = np.array(smpl_princpt)
    smpl_bboxes = np.array(smpl_bboxes)

    wholebody_kps = torch.from_numpy(np.array(wholebody_kps)).float().to(device)
    smplx_betas = torch.from_numpy(np.array(smplx_betas)).float().to(device)
    smplx_body_poses = torch.from_numpy(np.array(smplx_body_poses)).float().to(device)
    smplx_lhand_poses = torch.from_numpy(np.array(smplx_lhand_poses)).float().to(device)
    smplx_rhand_poses = torch.from_numpy(np.array(smplx_rhand_poses)).float().to(device)
    smplx_betas_init = torch.from_numpy(np.array(smplx_betas_init)).float().to(device)
    smplx_body_poses_init = torch.from_numpy(np.array(smplx_body_poses_init)).float().to(device)
    smplx_lhand_poses_init = torch.from_numpy(np.array(smplx_lhand_poses_init)).float().to(device)
    smplx_rhand_poses_init = torch.from_numpy(np.array(smplx_rhand_poses_init)).float().to(device)
    cam_Rs = torch.from_numpy(np.array(cam_Rs)).float().to(device)
    cam_Ts = torch.from_numpy(np.array(cam_Ts)).float().to(device)

    cam_R6d = matrix_to_rotation_6d(cam_Rs)
    object_rel_rotmat = cam_Rs.transpose(2, 1) @ object_rotmat
    object_rel_trans = (cam_Rs.transpose(2, 1) @ (object_trans - cam_Ts).reshape(-1, 3, 1)).squeeze(-1)

    _object_rel_rotmat = {}
    _object_rel_trans = {}
    _object_scale = {}
    for file in os.listdir('./annotation_RT/{}/params'.format(object_name)):
        image_id = file.split('.')[0]
        params = load_pickle('./annotation_RT/{}/params/{}'.format(object_name, file))
        _object_rel_rotmat[image_id] = params['object_rel_rotmat']
        _object_rel_trans[image_id] = params['object_rel_trans']
        _object_scale[image_id] = params['object_scale']
    object_rel_rotmat = torch.tensor([_object_rel_rotmat[img_id] for img_id in image_ids]).reshape(-1, 3, 3).float().to(device)
    object_rel_trans = torch.tensor([_object_rel_trans[img_id] for img_id in image_ids]).reshape(-1, 3).float().to(device)
    object_scale = torch.tensor([_object_scale[img_id] for img_id in image_ids]).reshape(-1).float().to(device)

    print('extract object coordinate maps ...')
    coor_x2d, coor_x3d, coor_mask = get_gt_object_coor(root_dir, image_ids, object_name, smpl_princpt, smpl_bboxes, device)

    hoi_recon_results = []
    n_items = smplx_betas.shape[0]
    batch_size = 64
    iterations = 2
    steps_per_iter = 500

    for begin_idx in range(0, n_items, batch_size):

        _image_ids = image_ids[begin_idx:begin_idx + batch_size]

        _smplx_betas = smplx_betas[begin_idx:begin_idx+batch_size]
        _smplx_betas_init = smplx_betas_init[begin_idx:begin_idx+batch_size]
        _smplx_body_poses = smplx_body_poses[begin_idx:begin_idx+batch_size]
        _smplx_lhand_poses = smplx_lhand_poses[begin_idx:begin_idx+batch_size]
        _smplx_rhand_poses = smplx_rhand_poses[begin_idx:begin_idx+batch_size]
        _smplx_body_poses_init = smplx_body_poses_init[begin_idx:begin_idx+batch_size]
        _smplx_lhand_poses_init = smplx_lhand_poses_init[begin_idx:begin_idx+batch_size]
        _smplx_rhand_poses_init = smplx_rhand_poses_init[begin_idx:begin_idx+batch_size]
        _object_rel_trans = object_rel_trans[begin_idx:begin_idx+batch_size]
        _object_rel_rotmat = object_rel_rotmat[begin_idx:begin_idx+batch_size]
        _object_rot_angle = object_rot_angle[begin_idx:begin_idx+batch_size]
        _object_scale = object_scale[begin_idx:begin_idx+batch_size]
        _hoi_trans = cam_Ts[begin_idx:begin_idx+batch_size]
        _hoi_rot6d = cam_R6d[begin_idx:begin_idx+batch_size]
        _smpl_bboxes = smpl_bboxes[begin_idx:begin_idx+batch_size]
        _smpl_focal = smpl_focal[begin_idx:begin_idx+batch_size]
        _smpl_princpt = smpl_princpt[begin_idx:begin_idx+batch_size]

        b = _smplx_betas.shape[0]
        hoi_instances = HOIInstance(smplx, _smplx_betas, _smplx_body_poses, _smplx_lhand_poses, _smplx_rhand_poses, _object_rot_angle,
            _object_rel_trans, _object_rel_rotmat, _hoi_trans, _hoi_rot6d, _object_scale).to(device)

        _wholebody_kps = wholebody_kps[begin_idx:begin_idx+batch_size]
        _coor_x2d = coor_x2d[begin_idx:begin_idx+batch_size]
        _coor_x3d = coor_x3d[begin_idx:begin_idx+batch_size]
        _coor_mask = coor_mask[begin_idx:begin_idx+batch_size]

        contact_labels = [contact_annotations[img_id] for img_id in _image_ids]

        loss_functions = [
            ContactLoss(contact_labels, smpl_indices, object_indices).to(device),
            ObjectCoorLoss(object_name, _coor_x2d, _coor_x3d, _coor_mask).to(device),
            ObjectScaleLoss().to(device),
            SMPLKPSLoss(_wholebody_kps).to(device),
            SMPLDecayLoss(_smplx_betas_init.clone(), _smplx_body_poses_init.clone(), _smplx_lhand_poses_init.clone(), _smplx_rhand_poses_init.clone()).to(device),
            CamDecayLoss(_hoi_rot6d.clone(), _hoi_trans.clone()).to(device),
        ]
        loss_weights = {
            'body_kps2d': lambda cst, it: 10. ** -2 * cst / (1 + 10 * it),
            'lhand_kps2d': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
            'rhand_kps2d': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),

            'betas_decay': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
            'body_decay': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
            'lhand_decay': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
            'rhand_decay': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
            'body_norm': lambda cst, it: 10. ** -5 * cst / (1 + 10 * it),
            'lhand_norm': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
            'rhand_norm': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),

            'cam_R_decay': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
            'cam_T_decay': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),

            'obj_corr': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
            'obj_scale': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),

            'contact': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
            'collision': lambda cst, it: 10. ** -3 * cst / (1 + 10 * it),
        }

        param_dicts = [
            {"params": [hoi_instances.smpl_betas, hoi_instances.smpl_body_pose, hoi_instances.lhand_pose, hoi_instances.rhand_pose, hoi_instances.rot_angle,
            hoi_instances.obj_rel_trans, hoi_instances.obj_rel_rot6d, hoi_instances.hoi_trans, hoi_instances.hoi_rot6d, hoi_instances.object_scale]},
        ]
        optimizer = torch.optim.Adam(param_dicts, lr=0.05, betas=(0.9, 0.999))
        for it in range(iterations):

            if it == 0:
                hoi_instances.smpl_betas.requires_grad_(False)
                hoi_instances.smpl_body_pose.requires_grad_(False)
                hoi_instances.lhand_pose.requires_grad_(False)
                hoi_instances.rhand_pose.requires_grad_(False)
                hoi_instances.hoi_trans.requires_grad_(False)
                hoi_instances.hoi_rot6d.requires_grad_(False)
                hoi_instances.rot_angle.requires_grad_(True)
                hoi_instances.obj_rel_trans.requires_grad_(False)
                hoi_instances.obj_rel_rot6d.requires_grad_(False)
                hoi_instances.object_scale.requires_grad_(False)
                steps_per_iter = 1
            else:
                # loss_functions.insert(0, HOCollisionLoss(smplx.faces).to(device))
                hoi_instances.smpl_betas.requires_grad_(True)
                hoi_instances.smpl_body_pose.requires_grad_(True)
                hoi_instances.lhand_pose.requires_grad_(True)
                hoi_instances.rhand_pose.requires_grad_(True)
                hoi_instances.hoi_trans.requires_grad_(False)
                hoi_instances.hoi_rot6d.requires_grad_(False)
                hoi_instances.rot_angle.requires_grad_(False)
                hoi_instances.obj_rel_trans.requires_grad_(False)
                hoi_instances.obj_rel_rot6d.requires_grad_(False)
                hoi_instances.object_scale.requires_grad_(False)
                steps_per_iter = 500
                # loss_weights['contact'] = lambda cst, it: 0 * 10. ** 0 * cst / (1 + 10 * it)

            loop = tqdm(range(steps_per_iter))
            for i in loop:
                optimizer.zero_grad()
                losses_all = {}

                hoi_dict = hoi_instances.forward()

                for f in loss_functions:
                    losses_all.update(f(hoi_dict))

                loss_list = [loss_weights[k](v, it).mean() for k, v in losses_all.items()]
                loss = torch.stack(loss_list).sum()
                loss.backward()
                optimizer.step()

                l_str = 'Optim. Step {}: Iter: {}, loss: {:.4f}'.format(it, i, loss.item())
                for k, v in losses_all.items():
                    l_str += ', {}: {:.4f}'.format(k, v.mean().detach().item())
                loop.set_description(l_str)

        hoi_out = hoi_instances.forward()
        for batch_idx, image_id in enumerate(_image_ids):
            hoi_recon_results.append({
                'image_id': image_id,
                'smplx_betas': hoi_out['betas'][batch_idx].detach().cpu().numpy().reshape(10, ),
                'smplx_body_pose': hoi_out['body_pose'][batch_idx].detach().cpu().numpy().reshape(21, 3),
                'smplx_lhand_pose': hoi_out['lhand_pose'][batch_idx].detach().cpu().numpy().reshape(15, 3),
                'smplx_rhand_pose': hoi_out['rhand_pose'][batch_idx].detach().cpu().numpy().reshape(15, 3),
                'object_rot_angle': hoi_out['object_rot_angle'][batch_idx].detach().cpu().numpy().reshape(1, ),
                'obj_rel_rotmat': hoi_out['object_rel_rotmat'][batch_idx].detach().cpu().numpy().reshape(3, 3),
                'obj_rel_trans': hoi_out['object_rel_trans'][batch_idx].detach().cpu().numpy().reshape(3, ),
                'hoi_rotmat': hoi_out['hoi_rotmat'][batch_idx].detach().cpu().numpy().reshape(3, 3),
                'hoi_trans': hoi_out['hoi_trans'][batch_idx].detach().cpu().numpy().reshape(3, ),
                'object_scale': hoi_out['object_scale'][batch_idx].detach().cpu().numpy().reshape(1, ),
                'crop_bboxes': _smpl_bboxes[batch_idx].reshape(4, ),
                'focal': _smpl_focal[batch_idx].reshape(2, ),
                'princpt': _smpl_princpt[batch_idx].reshape(2, ),
            })

    save_pickle(hoi_recon_results, os.path.join(hoi_out_dir, '{}_test.pkl'.format(args.object)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize Human-Object with contact labels.')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    parser.add_argument('--object', type=str)
    args = parser.parse_args()

    optimize_all(args)
