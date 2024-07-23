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
from pytorch3d.transforms import matrix_to_rotation_6d

from hoi_recon.datasets.utils import load_pickle, save_pickle, load_json

from hoi_instance import HOIInstance
from models import Model as ObjectModel
from object_image_dataset import ObjectImageDataset, CORR_NORM
from optim_losses import ObjectScaleLoss, ObjectCoorLoss, SMPLDecayLoss, SMPLKPSLoss, CamDecayLoss, ContactLoss, HOCollisionLoss


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

    tracking_results_dir = os.path.join(args.root_dir, args.object, 'hoi_tracking')

    smplx = SMPLX('/public/home/huochf/projects/3D_HOI/hoiYouTube/data/smpl/smplx/', gender='neutral', use_pca=False)

    smpl_indices = load_json('../data/contact/smplx_mesh_contact_{}.json'.format(object_name))
    if object_name in ['cello', 'violin']:
        object_indices = load_json('../data/contact/{}_body_contact.json'.format(object_name))
    else:
        object_indices = load_json('../data/contact/{}_contact.json'.format(object_name))
    contact_annotations = load_json('./annotation_contact_v2/{}_contact_annotations.json'.format(object_name))

    if object_name in ['cello', 'violin']:
        object_mesh = trimesh.load('../data/objects/{}_body.ply'.format(object_name), process=False)
    else:
        object_mesh = trimesh.load('../data/objects/{}.ply'.format(object_name), process=False)
    object_v = torch.tensor(np.array(object_mesh.vertices), dtype=torch.float32).to(device)

    object_kps_indices = load_object_kps_indices(object_name)
    object_kps = []
    for k, v in object_kps_indices.items():
        object_kps.append(object_v[v].mean(0))
    object_kps_org = torch.stack(object_kps)[OBJECT_KPS_PERM[object_name][0]]

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

    annotated_RT_images = []
    if os.path.exists('./annotation_RT/{}/params'.format(object_name)):
        for file in os.listdir('./annotation_RT/{}/params'.format(object_name)):
            annotated_RT_images.append(file.split('.')[0])

    print('collect data (keypoints, coor, ) ...')
    for img_id in tqdm(train_test_split[object_name]['test_frames']):
        if img_id in annotated_images:
            continue

        if img_id not in annotated_RT_images:
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
    object_trans = torch.from_numpy(np.array(object_trans)).float().to(device)
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
        _object_scale = object_scale[begin_idx:begin_idx+batch_size]
        _hoi_trans = cam_Ts[begin_idx:begin_idx+batch_size]
        _hoi_rot6d = cam_R6d[begin_idx:begin_idx+batch_size]
        _smpl_bboxes = smpl_bboxes[begin_idx:begin_idx+batch_size]
        _smpl_focal = smpl_focal[begin_idx:begin_idx+batch_size]
        _smpl_princpt = smpl_princpt[begin_idx:begin_idx+batch_size]

        b = _smplx_betas.shape[0]
        _object_v = object_v.unsqueeze(0).repeat(b, 1, 1)
        _object_kps = object_kps_org.unsqueeze(0).repeat(b, 1, 1)
        hoi_instances = HOIInstance(smplx, _object_kps, _object_v, _smplx_betas, _smplx_body_poses, _smplx_lhand_poses, _smplx_rhand_poses,
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
            {"params": [hoi_instances.smpl_betas, hoi_instances.smpl_body_pose, hoi_instances.lhand_pose, hoi_instances.rhand_pose,
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
