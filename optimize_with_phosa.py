import os
import sys
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

from datasets.utils import generate_image_patch, load_pickle, save_pickle, load_json
from datasets.hoi_img_kps_dataset import OBJECT_KPS_N
from utils.hoi_instance import HOIInstance
from utils.optim_losses import ObjectScaleLoss, ObjectCoorLoss, SMPLDecayLoss, SMPLKPSLoss, CamDecayLoss
from utils.optim_losses import PHOSAInteractionLoss, OrdinalDepthLoss

from train_kps_flow_wildhoi import Model


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


def load_object_kps_indices(object_name):
    if object_name == 'barbell':
        object_file = 'data/objects/barbell_keypoints_12.json'
    elif object_name == 'cello':
        object_file = 'data/objects/cello_keypoints_14.json'
    elif object_name == 'baseball':
        object_file = 'data/objects/baseball_keypoints.json'
    elif object_name == 'tennis':
        object_file = 'data/objects/tennis_keypoints_7.json'
    elif object_name == 'skateboard':
        object_file = 'data/objects/skateboard_keypoints_8.json'
    elif object_name == 'basketball':
        object_file = 'data/objects/basketball_keypoints.json'
    elif object_name == 'yogaball':
        object_file = 'data/objects/yogaball_keypoints.json'
    elif object_name == 'violin':
        object_file = 'data/objects/violin_body_keypoints.json'
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


def get_object_coor(dataloader, model, object_name, smpl_princpt, smpl_bboxes, device):
    all_coor_maps = []
    object_box_centers, object_box_sizes = [], []
    for item in tqdm(dataloader):
        images, obj_box_c, obj_box_s = item
        images = images.to(device)
        coor_maps = model.inference_step(images) # numpy, [b, 4, h, w]
        all_coor_maps.append(coor_maps)
        object_box_centers.append(obj_box_c)
        object_box_sizes.append(obj_box_s)
    all_coor_maps = np.concatenate(all_coor_maps)
    object_box_centers = torch.cat(object_box_centers)
    object_box_sizes = torch.cat(object_box_sizes)

    b, _, coor_h, coor_w = all_coor_maps.shape
    all_coor_maps = all_coor_maps.transpose(0, 2, 3, 1)
    coor_norm = CORR_NORM[object_name]
    coor_x3d = all_coor_maps[:, :, :, :3] * np.array(coor_norm).reshape(1, 1, 1, 3)

    coor_mask = all_coor_maps[:, :, :, 3:]
    grid_2d = torch.arange(coor_h).float()
    ys, xs = torch.meshgrid(grid_2d, grid_2d) # (h, w)
    grid_2d = torch.stack([xs, ys], dim=2).unsqueeze(0).repeat(b, 1, 1, 1).reshape(b, -1, 2) # (b, h * w, 2)
    stride = object_box_sizes / coor_h
    stride = stride.reshape(b, 1, 1).float()
    x1 = object_box_centers[:, 0] - object_box_sizes / 2
    y1 = object_box_centers[:, 1] - object_box_sizes / 2
    begin_point = torch.stack([x1, y1], dim=1)
    begin_point = begin_point.reshape(b, 1, 2).float()
    coor_x2d = grid_2d * stride + begin_point # [b, h*w, 2]
    coor_x2d = coor_x2d.numpy()
    coor_x2d = (coor_x2d - smpl_princpt.reshape(b, 1, 2)) / smpl_bboxes[:, 2].reshape(b, 1, 1) * 192

    coor_x3d = coor_x3d.reshape(b, -1, 3) # [b, n_init, h*w, 3]
    coor_mask = coor_mask.reshape(b, -1).clip(0, 1)

    binary_mask = coor_mask > 0.5
    ns = binary_mask.sum(-1)
    mask_corr_min = np.zeros_like(binary_mask)
    for i in range(b):
        mask_corr_min[i, :int(ns[i] * 0.8)] = 1

    return coor_x2d, coor_x3d, mask_corr_min


def get_sam_masks(root_dir, image_ids, object_name, smpl_bboxes, device):
    person_masks = []
    object_masks = []

    print('loadding sam masks ...')
    for idx, image_id in enumerate(image_ids):
        video_id, hoi_id, frame_id = image_id.split('_')
        image_path = os.path.join(root_dir, object_name, 'images_temp', video_id, '{}.jpg'.format(frame_id))
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        person_mask = np.zeros((h, w))
        # object_mask = np.zeros((h, w))

        mask_load = load_pickle(os.path.join(root_dir, object_name, 'hoi_mask', video_id, hoi_id, '{}.pkl'.format(frame_id)))
        if mask_load['human']['mask'] is not None:
            mask_h, mask_w = mask_load['human']['mask_shape']
            x1, y1, x2, y2 = mask_load['human']['mask_box']
            person_mask[y1:y2+1, x1:x2+1] = np.unpackbits(mask_load['human']['mask'])[:mask_h * mask_w].reshape(mask_h, mask_w)

        # if mask_load['object']['mask'] is not None:
        #     mask_h, mask_w = mask_load['object']['mask_shape']
        #     x1, y1, x2, y2 = mask_load['object']['mask_box']
        #     object_mask[y1:y2+1, x1:x2+1] = np.unpackbits(mask_load['object']['mask'])[:mask_h * mask_w].reshape(mask_h, mask_w)

        object_mask = cv2.imread('data/object_masks_test/{}/{}_object_mask.jpg'.format(object_name, image_id), cv2.IMREAD_GRAYSCALE)
        object_mask = object_mask / 255

        bbox_xywh = smpl_bboxes[idx]
        x, y, w, h = bbox_xywh
        cx, cy = x + w / 2, y + h / 2
        s = max(w, h)

        person_mask, _ = generate_image_patch(person_mask, cx, cy, s, 256, 0, [1., 1., 1.])
        object_mask, _ = generate_image_patch(object_mask, cx, cy, s, 256, 0, [1., 1., 1.])

        person_masks.append(person_mask)
        object_masks.append(object_mask)

    person_masks = torch.from_numpy(np.stack(person_masks).astype(np.float32)).to(device)
    object_masks = torch.from_numpy(np.stack(object_masks).astype(np.float32)).to(device)

    return person_masks, object_masks


def optimize_all(args):
    device = torch.device('cuda')
    root_dir = args.root_dir
    object_name = args.object

    tracking_results_dir = os.path.join(args.root_dir, args.object, 'hoi_tracking')

    smplx = SMPLX('data/smpl/smplx/', gender='neutral', use_pca=False)
    smpl_f = torch.from_numpy(np.array(smplx.faces).astype(np.int64)).to(device)

    if object_name in ['cello', 'violin']:
        object_mesh = trimesh.load('data/objects/{}_body.ply'.format(object_name), process=False)
    else:
        object_mesh = trimesh.load('data/objects/{}.ply'.format(object_name), process=False)
    object_v = torch.from_numpy(np.array(object_mesh.vertices).astype(np.float32)).to(device)
    object_f = torch.from_numpy(np.array(object_mesh.faces).astype(np.int64)).to(device)

    object_kps_indices = load_object_kps_indices(object_name)
    object_kps = []
    for k, v in object_kps_indices.items():
        object_kps.append(object_v[v].mean(0))
    object_kps_org = torch.stack(object_kps)[OBJECT_KPS_PERM[object_name][0]]

    results_dir = './outputs/optim_with_phosa'
    os.makedirs(results_dir, exist_ok=True)

    smpl_parts_indices = load_json('data/contact/smplx_mesh_contact_{}.json'.format(object_name))
    if object_name in ['cello', 'violin']:
        object_parts_indices = load_json('data/contact/{}_body_contact.json'.format(object_name))
    else:
        object_parts_indices = load_json('data/contact/{}_contact.json'.format(object_name))


    image_ids = []
    wholebody_kps = []
    smplx_betas = []
    smplx_body_poses = []
    smplx_lhand_poses = []
    smplx_rhand_poses = []
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
    if os.path.exists('data/annotation_hoi/{}/test'.format(object_name)):
        for file in os.listdir('data/annotation_hoi/{}/test'.format(object_name)):
            annotated_images.append(file.split('.')[0])

    print('collect data (keypoints, coor, ) ...')
    for img_id in tqdm(annotated_images):
        if img_id not in annotated_images:
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
                box_cx, box_cy = (x2 + x1) / 2, (y2 + y1) / 2
                hoi_box_cxcys = (box_cx, box_cy, box_size)

                object_bboxes.append(object_bbox)
                hoi_bboxes.append(hoi_box_cxcys)

                object_rotmat.append(object_RT_item['rotmat'])
                object_trans.append(object_RT_item['trans'])

                image_ids.append(img_id)
                smplx_betas.append(smplx_params_item['betas'])
                smplx_body_poses.append(smplx_params_item['body_pose'])
                smplx_lhand_poses.append(smplx_params_item['left_hand_pose'])
                smplx_rhand_poses.append(smplx_params_item['right_hand_pose'])
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

    object_rotmat = torch.from_numpy(np.array(object_rotmat)).float().to(device)
    object_trans = torch.from_numpy(np.array(object_trans)).float().to(device)
    smpl_princpt = np.array(smpl_princpt)
    smpl_bboxes = np.array(smpl_bboxes)

    wholebody_kps = torch.from_numpy(np.array(wholebody_kps)).float().to(device)
    smplx_betas = torch.from_numpy(np.array(smplx_betas)).float().to(device)
    smplx_body_poses = torch.from_numpy(np.array(smplx_body_poses)).float().to(device)
    smplx_lhand_poses = torch.from_numpy(np.array(smplx_lhand_poses)).float().to(device)
    smplx_rhand_poses = torch.from_numpy(np.array(smplx_rhand_poses)).float().to(device)
    cam_Rs = torch.from_numpy(np.array(cam_Rs)).float().to(device)
    cam_Ts = torch.from_numpy(np.array(cam_Ts)).float().to(device)
    cam_R6d = matrix_to_rotation_6d(cam_Rs)
    object_rel_rotmat = cam_Rs.transpose(2, 1) @ object_rotmat
    object_rel_trans = (cam_Rs.transpose(2, 1) @ (object_trans - cam_Ts).reshape(-1, 3, 1)).squeeze(-1)

    print('loadding object coordinate maps ...')
    coor_x2d, coor_x3d, coor_mask = get_gt_object_coor(root_dir, image_ids, object_name, smpl_princpt, smpl_bboxes, device)

    person_masks, object_masks = get_sam_masks(root_dir, image_ids, object_name, smpl_bboxes, device)

    hoi_recon_results = []
    n_seq = smplx_betas.shape[0]
    batch_size = 64
    iterations = 2
    steps_per_iter = 2000
    for begin_idx in range(0, n_seq, batch_size):

        _image_ids = image_ids[begin_idx:begin_idx + batch_size]

        _smplx_betas = smplx_betas[begin_idx:begin_idx+batch_size]
        _smplx_body_poses = smplx_body_poses[begin_idx:begin_idx+batch_size]
        _smplx_lhand_poses = smplx_lhand_poses[begin_idx:begin_idx+batch_size]
        _smplx_rhand_poses = smplx_rhand_poses[begin_idx:begin_idx+batch_size]
        _object_rel_trans = object_rel_trans[begin_idx:begin_idx+batch_size]
        _object_rel_rotmat = object_rel_rotmat[begin_idx:begin_idx+batch_size]
        _hoi_trans = cam_Ts[begin_idx:begin_idx+batch_size]
        _hoi_rot6d = cam_R6d[begin_idx:begin_idx+batch_size]
        _smpl_bboxes = smpl_bboxes[begin_idx:begin_idx+batch_size]
        _smpl_focal = smpl_focal[begin_idx:begin_idx+batch_size]
        _smpl_princpt = smpl_princpt[begin_idx:begin_idx+batch_size]

        b = _smplx_betas.shape[0]
        _object_v = object_v.unsqueeze(0).repeat(b, 1, 1)
        _object_kps = object_kps_org.unsqueeze(0).repeat(b, 1, 1)
        hoi_instances = HOIInstance(smplx, _object_kps, _object_v, _smplx_betas, _smplx_body_poses, _smplx_lhand_poses, _smplx_rhand_poses,
            _object_rel_trans, _object_rel_rotmat, _hoi_trans, _hoi_rot6d).to(device)

        _wholebody_kps = wholebody_kps[begin_idx:begin_idx+batch_size]
        _coor_x2d = coor_x2d[begin_idx:begin_idx+batch_size]
        _coor_x3d = coor_x3d[begin_idx:begin_idx+batch_size]
        _coor_mask = coor_mask[begin_idx:begin_idx+batch_size]

        _person_masks = person_masks[begin_idx:begin_idx + batch_size]
        _object_masks = object_masks[begin_idx:begin_idx + batch_size]

        _smpl_f = smpl_f.unsqueeze(0).repeat(b, 1, 1)
        _object_f = object_f.unsqueeze(0).repeat(b, 1, 1)

        loss_functions = [
            OrdinalDepthLoss(_person_masks, _object_masks, _smpl_f, _object_f, _smpl_bboxes, _smpl_focal).to(device),
            PHOSAInteractionLoss(b, smpl_parts_indices, object_parts_indices, _smpl_bboxes, _smpl_focal).to(device),
            ObjectCoorLoss(object_name, _coor_x2d, _coor_x3d, _coor_mask).to(device),
            ObjectScaleLoss().to(device),
            SMPLKPSLoss(_wholebody_kps).to(device),
            SMPLDecayLoss(_smplx_betas.clone(), _smplx_body_poses.clone(), _smplx_lhand_poses.clone(), _smplx_rhand_poses.clone()).to(device),
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

            'obj_corr': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
            'obj_scale': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),

            'depth': lambda cst, it: 10. ** -2 * cst / (1 + 10 * it),
            'inter': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
            'inter_part': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
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
                hoi_instances.obj_rel_trans.requires_grad_(True)
                hoi_instances.obj_rel_rot6d.requires_grad_(True)
                hoi_instances.object_scale.requires_grad_(True)
                steps_per_iter = 4000
            else:
                hoi_instances.smpl_betas.requires_grad_(True)
                hoi_instances.smpl_body_pose.requires_grad_(True)
                hoi_instances.lhand_pose.requires_grad_(True)
                hoi_instances.rhand_pose.requires_grad_(True)
                hoi_instances.hoi_trans.requires_grad_(False)
                hoi_instances.hoi_rot6d.requires_grad_(False)
                hoi_instances.obj_rel_trans.requires_grad_(True)
                hoi_instances.obj_rel_rot6d.requires_grad_(True)
                hoi_instances.object_scale.requires_grad_(True)
                steps_per_iter = 1000

            loop = tqdm(range(steps_per_iter))
            for i in loop:
                optimizer.zero_grad()
                losses_all = {}

                hoi_dict = hoi_instances.forward()

                for f in loss_functions:
                    losses_all.update(f(hoi_dict))

                loss_list = [loss_weights[k](v, it) for k, v in losses_all.items()]
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

    save_pickle(hoi_recon_results, os.path.join(results_dir, '{}_test.pkl'.format(args.object)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize with PHOSA')
    parser.add_argument('--root_dir', default='/storage/data/huochf/HOIYouTube')
    parser.add_argument('--object', default='barbell')

    args = parser.parse_args()
    optimize_all(args)
