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
import neural_renderer as nr

from smplx import SMPLX
from pytorch3d.transforms import matrix_to_rotation_6d

from datasets.utils import generate_image_patch, load_pickle, save_pickle
from datasets.hoi_img_kps_dataset import OBJECT_KPS_N
from utils.hoi_instance import HOIInstance
from utils.optim_losses import MultiViewHOIKPSLoss, ObjectScaleLoss, ObjectCoorLoss, SMPLDecayLoss, SMPLKPSLoss, CamDecayLoss

from train_kps_flow_wildhoi import Model


object_name = 'cello'
video_id = '0000'
hoi_id = '001'
frame_id = '002414'
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

    'kps_nll': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
}


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


def render_hoi(smpl_v, smpl_f, object_v, object_f, R, T, K, s, novel_views=False):
    device = torch.device('cuda')
    smpl_v = torch.tensor(smpl_v, dtype=torch.float32).reshape(1, -1, 3).to(device)
    object_v = torch.tensor(object_v, dtype=torch.float32).reshape(1, -1, 3).to(device)
    smpl_f = torch.tensor(smpl_f, dtype=torch.int64).reshape(1, -1, 3).to(device)
    object_f = torch.tensor(object_f, dtype=torch.int64).reshape(1, -1, 3).to(device)

    vertices = torch.cat([smpl_v, object_v], dim=1)
    faces = torch.cat([smpl_f, object_f + smpl_v.shape[1]], dim=1)

    colors_list = [
        [251 / 255.0, 128 / 255.0, 114 / 255.0],  # red
        [0.65098039, 0.74117647, 0.85882353],  # blue
        [0.9, 0.7, 0.7],  # pink
    ]
    smpl_t = torch.tensor(colors_list[1], dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).repeat(1, smpl_f.shape[1], 1, 1, 1, 1)
    object_t = torch.tensor(colors_list[0], dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).repeat(1, object_f.shape[1], 1, 1, 1, 1)
    textures = torch.cat([smpl_t, object_t], dim=1).to(device)

    K = K.reshape(1, 3, 3).to(device)
    R = R.reshape(1, 3, 3).to(device)
    t = T.reshape(1, 3).to(device)

    vertices = vertices @ R.transpose(2, 1) + t.reshape(1, 1, 3)

    R = torch.eye(3).reshape(1, 3, 3).to(device)
    t = torch.zeros(3).reshape(1, 3).to(device)

    renderer = nr.renderer.Renderer(image_size=s, K=K, R=R, t=t, orig_size=1)
    
    renderer.background_color = [1, 1, 1]
    # if novel_views:
    #     renderer.light_direction = [1, 0.5, 1]
    # else:
    renderer.light_direction = [1, 0.5, 1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.5

    rend, _, mask = renderer.render(vertices=vertices, faces=faces, textures=textures)
    rend = rend.clip(0, 1)
    rend = rend[0].permute(1, 2, 0).detach().cpu().numpy()
    mask = mask[0].detach().cpu().numpy().reshape((s, s, 1))
    rend = (rend * 255).astype(np.uint8)
    rend = rend[:, :, ::-1]

    return rend, mask


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


def load_gt_coor(smpl_princpt, smpl_bboxes):
    mask = cv2.imread('data/demo/{}_{}_{}_object_mask.png'.format(video_id, hoi_id, frame_id), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_h, img_w = mask.shape

    coor_load = load_pickle('data/demo/{}_{}_{}_object_coor.pkl'.format(video_id, hoi_id, frame_id))
    u, l, h, w = coor_load['u'], coor_load['l'], coor_load['h'], coor_load['w']
    coor = np.zeros((img_h, img_w, 3)).astype(np.float32)
    coor[u:(u+h), l:(l+w), :] = coor_load['coor']

    ys, xs = torch.meshgrid(torch.arange(img_h).float(), torch.arange(img_w).float()) # (h, w)
    coor_x2d = torch.stack([xs, ys], dim=2).reshape(img_h, img_w, 2) # [h * w, 2]
    coor_x2d = coor_x2d.numpy()

    coor_x2d = np.concatenate([coor_x2d, mask.reshape(img_h, img_w, 1)], axis=2)

    cx, cy, s = l + w / 2, u + h / 2, max(h, w)
    coor_x3d, _ = generate_image_patch(coor, cx, cy, s, 128, 0., [1., 1., 1.])
    coor_x2d, _ = generate_image_patch(coor_x2d, cx, cy, s, 128, 0., [1., 1., 1.])

    coor_x3d = coor_x3d.reshape(1, -1, 3)
    coor_mask = coor_x2d[:, :, 2].reshape(1, -1)
    coor_x2d = coor_x2d[:, :, :2].reshape(1, -1, 2)

    coor_x2d = (coor_x2d - smpl_princpt.reshape(1, 1, 2)) / smpl_bboxes[:, 2].reshape(1, 1, 1) * 192

    binary_mask = coor_mask > 0.5
    ns = binary_mask.sum(-1)
    mask_corr_min = np.zeros_like(binary_mask)
    mask_corr_min[0, :int(ns[0] * 0.98)] = 1

    return coor_x2d, coor_x3d, mask_corr_min


def load_visual_feats(model, hoi_bbox, device):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = cv2.imread('data/demo/{}_{}_{}_image.jpg'.format(video_id, hoi_id, frame_id), cv2.IMREAD_COLOR)
    box_cx, box_cy, box_size = hoi_bbox[0]
    out_size = 256

    rot, color_scale = 0, [1., 1., 1.]
    img_patch, _ = generate_image_patch(image, box_cx, box_cy, box_size, out_size, rot, color_scale)
    img_patch = img_patch[:, :, ::-1].astype(np.float32)
    img_patch = img_patch.transpose((2, 0, 1)) / 256

    for n_c in range(3):
        img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]
    images = torch.from_numpy(img_patch).unsqueeze(0).float().to(device)

    visual_feats = model.image_embedding(images).reshape(1, -1).detach()

    return visual_feats


def optimize_human_object(hoi_instances, loss_functions, loss_weights, optimizer):
    iterations = 2
    steps_per_iter = 2000
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
            losses_all =  {}

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


def demo(args):
    device = torch.device('cuda')

    flow_dim = (22 + OBJECT_KPS_N[object_name] + 1) * 3
    model = Model(flow_dim=flow_dim, 
                      flow_width=512, 
                      c_dim=256,
                      num_blocks_per_layers=2, 
                      layers=4,
                      dropout_probability=0).to(device)
    state_dict = torch.load(os.path.join('outputs/cflow_pseudo_kps2d/checkpoint_{}.pth'.format(object_name)))
    model.load_state_dict(state_dict['model'])

    smplx = SMPLX('data/smpl/smplx/', gender='neutral', use_pca=False)
    smpl_f = torch.from_numpy(np.array(smplx.faces).astype(np.int64)).to(device)

    if object_name in ['cello', 'violin']:
        object_mesh = trimesh.load('data/objects/{}_body.ply'.format(object_name), process=False)
    else:
        object_mesh = trimesh.load('data/objects/{}.ply'.format(object_name), process=False)
    object_v = object_v_org = torch.tensor(np.array(object_mesh.vertices), dtype=torch.float32).to(device)
    object_f = torch.from_numpy(np.array(object_mesh.faces).astype(np.int64)).to(device)

    object_kps_indices = load_object_kps_indices(object_name)
    object_kps = []
    for k, v in object_kps_indices.items():
        object_kps.append(object_v[v].mean(0))
    object_kps_org = torch.stack(object_kps)[OBJECT_KPS_PERM[object_name][0]]

    results_dir = './outputs/demo'
    os.makedirs(results_dir, exist_ok=True)

    tracking_item = load_pickle('data/demo/{}_{}_{}_tracking.pkl'.format(video_id, hoi_id, frame_id))
    object_RT_item = load_pickle('data/demo/{}_{}_{}_object_RT.pkl'.format(video_id, hoi_id, frame_id))
    wholebody_kps_item = load_pickle('data/demo/{}_{}_{}_wholebody_kps.pkl'.format(video_id, hoi_id, frame_id))
    smplx_params_item = load_pickle('data/demo/{}_{}_{}_smplx.pkl'.format(video_id, hoi_id, frame_id))

    person_bbox = tracking_item['person_bbox']
    object_bbox = tracking_item['object_bbox']
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
    hoi_bboxes = [hoi_box_cxcys, ]

    object_trans = torch.from_numpy(np.array(object_RT_item['trans'])).unsqueeze(0).float().to(device)
    object_rotmat = torch.from_numpy(np.array(object_RT_item['rotmat'])).unsqueeze(0).float().to(device)
    smplx_betas = torch.from_numpy(np.array(smplx_params_item['betas'])).unsqueeze(0).float().to(device)
    smplx_body_poses = torch.from_numpy(np.array(smplx_params_item['body_pose'])).unsqueeze(0).float().to(device)
    smplx_lhand_poses = torch.from_numpy(np.array(smplx_params_item['left_hand_pose'])).unsqueeze(0).float().to(device)
    smplx_rhand_poses = torch.from_numpy(np.array(smplx_params_item['right_hand_pose'])).unsqueeze(0).float().to(device)
    hoi_rotmat = torch.from_numpy(np.array(smplx_params_item['cam_R'])).unsqueeze(0).float().to(device)
    hoi_trans = torch.from_numpy(np.array(smplx_params_item['cam_T'])).reshape(1, 3).float().to(device)
    hoi_rot6d = matrix_to_rotation_6d(hoi_rotmat)
    smpl_princpt = np.array(smplx_params_item['princpt']).reshape(1, -1)
    smpl_focal = np.array(smplx_params_item['focal']).reshape(1, -1)
    smpl_bboxes = np.array(smplx_params_item['bbox']).reshape(1, -1)

    object_rel_rotmat = hoi_rotmat.transpose(2, 1) @ object_rotmat
    object_rel_trans = (hoi_rotmat.transpose(2, 1) @ (object_trans - hoi_trans).reshape(-1, 3, 1)).squeeze(-1)

    keypoints = wholebody_kps_item['keypoints']
    keypoints[:, :2] = (keypoints[:, :2] - smpl_princpt) / smpl_focal
    keypoints[keypoints[:, 2] < 0.5] = 0
    wholebody_kps = torch.from_numpy(np.array(keypoints)).unsqueeze(0).float().to(device)

    coor_x2d, coor_x3d, coor_mask = load_gt_coor(smpl_princpt, smpl_bboxes)

    visual_feats = load_visual_feats(model, hoi_bboxes, device)

    object_v = object_v.unsqueeze(0)
    object_kps = object_kps_org.unsqueeze(0)
    hoi_instance_no_flow = HOIInstance(smplx, object_kps, object_v, smplx_betas.clone(), smplx_body_poses.clone(), smplx_lhand_poses.clone(), smplx_rhand_poses.clone(), 
        object_rel_trans.clone(), object_rel_rotmat.clone(), hoi_trans.clone(), hoi_rot6d.clone()).to(device)
    loss_functions_no_flow = [
        ObjectCoorLoss(object_name, coor_x2d, coor_x3d, coor_mask).to(device),
        ObjectScaleLoss().to(device),
        SMPLKPSLoss(wholebody_kps).to(device),
        SMPLDecayLoss(smplx_betas.clone(), smplx_body_poses.clone(), smplx_lhand_poses.clone(), smplx_rhand_poses.clone()).to(device),
        CamDecayLoss(hoi_rot6d.clone(), hoi_trans.clone()).to(device),
    ]

    param_dicts = [
        {"params": [hoi_instance_no_flow.smpl_betas, hoi_instance_no_flow.smpl_body_pose, hoi_instance_no_flow.lhand_pose, hoi_instance_no_flow.rhand_pose,
        hoi_instance_no_flow.obj_rel_trans, hoi_instance_no_flow.obj_rel_rot6d, hoi_instance_no_flow.hoi_trans, hoi_instance_no_flow.hoi_rot6d, hoi_instance_no_flow.object_scale]},
    ]
    optimizer_no_flow = torch.optim.Adam(param_dicts, lr=0.05, betas=(0.9, 0.999))
    print('optimize w/o kps flow:')
    optimize_human_object(hoi_instance_no_flow, loss_functions_no_flow, loss_weights, optimizer_no_flow)
    hoi_out_no_flow = hoi_instance_no_flow.forward()
    hoi_recon_no_flow = {
        'image_id': '{}_{}_{}'.format(video_id, hoi_id, frame_id),
        'smplx_betas': hoi_out_no_flow['betas'].detach().cpu().numpy().reshape(10, ),
        'smplx_body_pose': hoi_out_no_flow['body_pose'].detach().cpu().numpy().reshape(21, 3),
        'smplx_lhand_pose': hoi_out_no_flow['lhand_pose'].detach().cpu().numpy().reshape(15, 3),
        'smplx_rhand_pose': hoi_out_no_flow['rhand_pose'].detach().cpu().numpy().reshape(15, 3),
        'obj_rel_rotmat': hoi_out_no_flow['object_rel_rotmat'].detach().cpu().numpy().reshape(3, 3),
        'obj_rel_trans': hoi_out_no_flow['object_rel_trans'].detach().cpu().numpy().reshape(3, ),
        'hoi_rotmat': hoi_out_no_flow['hoi_rotmat'].detach().cpu().numpy().reshape(3, 3),
        'hoi_trans': hoi_out_no_flow['hoi_trans'].detach().cpu().numpy().reshape(3, ),
        'object_scale': hoi_out_no_flow['object_scale'].detach().cpu().numpy().reshape(1, ),
        'crop_bboxes': smpl_bboxes.reshape(4, ),
        'focal': smpl_focal.reshape(2, ),
        'princpt': smpl_princpt.reshape(2, ),
    }
    save_pickle(hoi_recon_no_flow, os.path.join(results_dir, '{}_{}_{}_recon_results_no_flow.pkl'.format(video_id, hoi_id, frame_id)))

    hoi_instances_flow = HOIInstance(smplx, object_kps, object_v, smplx_betas.clone(), smplx_body_poses.clone(), smplx_lhand_poses.clone(), smplx_rhand_poses.clone(), 
        object_rel_trans.clone(), object_rel_rotmat.clone(), hoi_trans.clone(), hoi_rot6d.clone()).to(device)
    loss_functions_flow = [
        MultiViewHOIKPSLoss(model, visual_feats, n_views=16).to(device),
        ObjectCoorLoss(object_name, coor_x2d, coor_x3d, coor_mask).to(device),
        ObjectScaleLoss().to(device),
        SMPLKPSLoss(wholebody_kps).to(device),
        SMPLDecayLoss(smplx_betas.clone(), smplx_body_poses.clone(), smplx_lhand_poses.clone(), smplx_rhand_poses.clone()).to(device),
        CamDecayLoss(hoi_rot6d.clone(), hoi_trans.clone()).to(device),
    ]

    param_dicts = [
        {"params": [loss_functions_flow[0].cam_pos, ], "lr": 1e-3},
        {"params": [hoi_instances_flow.smpl_betas, hoi_instances_flow.smpl_body_pose, hoi_instances_flow.lhand_pose, hoi_instances_flow.rhand_pose,
        hoi_instances_flow.obj_rel_trans, hoi_instances_flow.obj_rel_rot6d, hoi_instances_flow.hoi_trans, hoi_instances_flow.hoi_rot6d, hoi_instances_flow.object_scale]},
    ]
    optimizer_flow = torch.optim.Adam(param_dicts, lr=0.05, betas=(0.9, 0.999))
    print('optimize with kps flow:')
    optimize_human_object(hoi_instances_flow, loss_functions_flow, loss_weights, optimizer_flow)
    hoi_out_flow = hoi_instances_flow.forward()
    hoi_recon_flow = {
        'image_id': '{}_{}_{}'.format(video_id, hoi_id, frame_id),
        'smplx_betas': hoi_out_flow['betas'].detach().cpu().numpy().reshape(10, ),
        'smplx_body_pose': hoi_out_flow['body_pose'].detach().cpu().numpy().reshape(21, 3),
        'smplx_lhand_pose': hoi_out_flow['lhand_pose'].detach().cpu().numpy().reshape(15, 3),
        'smplx_rhand_pose': hoi_out_flow['rhand_pose'].detach().cpu().numpy().reshape(15, 3),
        'obj_rel_rotmat': hoi_out_flow['object_rel_rotmat'].detach().cpu().numpy().reshape(3, 3),
        'obj_rel_trans': hoi_out_flow['object_rel_trans'].detach().cpu().numpy().reshape(3, ),
        'hoi_rotmat': hoi_out_flow['hoi_rotmat'].detach().cpu().numpy().reshape(3, 3),
        'hoi_trans': hoi_out_flow['hoi_trans'].detach().cpu().numpy().reshape(3, ),
        'object_scale': hoi_out_flow['object_scale'].detach().cpu().numpy().reshape(1, ),
        'crop_bboxes': smpl_bboxes.reshape(4, ),
        'focal': smpl_focal.reshape(2, ),
        'princpt': smpl_princpt.reshape(2, ),
    }
    save_pickle(hoi_recon_flow, os.path.join(results_dir, '{}_{}_{}_recon_results_flow.pkl'.format(video_id, hoi_id, frame_id)))

    # Visualization

    def visualize_human_object(hoi_recon):

        smplx_betas = torch.tensor(hoi_recon['smplx_betas']).reshape(1, 10).float().to(device)
        smplx_body_pose = torch.tensor(hoi_recon['smplx_body_pose']).reshape(1, 63).float().to(device)
        smplx_lhand_pose = torch.tensor(hoi_recon['smplx_lhand_pose']).reshape(1, 45).float().to(device)
        smplx_rhand_pose = torch.tensor(hoi_recon['smplx_rhand_pose']).reshape(1, 45).float().to(device)

        smplx_out = smplx(betas=smplx_betas, body_pose=smplx_body_pose, left_hand_pose=smplx_lhand_pose, right_hand_pose=smplx_rhand_pose)
        smpl_v = smplx_out.vertices.detach()[0]
        smpl_J = smplx_out.joints.detach()[0]
        smpl_v = smpl_v - smpl_J[:1]

        object_scale = torch.tensor(hoi_recon['object_scale']).float().to(device).reshape(1, )
        object_v = object_v_org.reshape(-1, 3) * object_scale
        object_rel_rotmat = torch.tensor(hoi_recon['obj_rel_rotmat']).float().to(device).reshape(3, 3)
        object_rel_trans = torch.tensor(hoi_recon['obj_rel_trans']).float().to(device).reshape(1, 3)
        object_v = object_v @ object_rel_rotmat.transpose(1, 0) + object_rel_trans

        hoi_rotmat = torch.tensor(hoi_recon['hoi_rotmat']).reshape(3, 3).float().to(device)
        hoi_trans = torch.tensor(hoi_recon['hoi_trans']).reshape(1, 3).float().to(device)

        image_org = cv2.imread('data/demo/{}_{}_{}_image.jpg'.format(video_id, hoi_id, frame_id))
        h, w, _ = image_org.shape

        fx, fy = smpl_focal.reshape(2, )
        cx, cy = smpl_princpt.reshape(2, )
        s = max(h, w)
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().to(device) / s

        rend, mask = render_hoi(smpl_v, smpl_f, object_v, object_f, hoi_rotmat, hoi_trans, K, s)
        rend = rend[:h, :w]
        mask = mask[:h, :w]
        mask = mask.reshape(h, w, 1)
        image_rend = rend.astype(np.uint8)

        crop_boxes = hoi_recon['crop_bboxes']
        cx, cy = crop_boxes[0] + crop_boxes[2] / 2, crop_boxes[1] + crop_boxes[3] / 2
        s = 1.2 * max(crop_boxes[2], crop_boxes[3])
        image_rend_flow, _ = generate_image_patch(image_rend, cx, cy, s, 512, 0, [1., 1., 1.])

        image_org_flow, _ = generate_image_patch(image_org, cx, cy, s, 512, 0, [1., 1., 1.])
        image_rend_flow[image_rend_flow.sum(-1) == 0] = 255
        image_org_flow[image_org_flow.sum(-1) == 0] = 255

        result_images = []
        for i in range(15):
            result_images.append(image_org_flow.astype(np.uint8))
        for i in range(15):
            result_images.append((image_rend_flow.astype(np.float32) * i / 15 + image_org_flow * (1 - i / 15)).astype(np.uint8))
        for i in range(15):
            result_images.append(image_rend_flow.astype(np.uint8))

        smpl_centered_images_flow = []
        for rot in np.arange(0, 360, 2):
            rotmat_centered = torch.tensor(Rotation.from_euler('y', rot, degrees=True).as_matrix()).float().to(device) \
                @ hoi_rotmat

            s = max(h, w)
            rend, mask = render_hoi(smpl_v, smpl_f, object_v, object_f, rotmat_centered, hoi_trans, K, s, novel_views=False)
            s = 1.2 * max(crop_boxes[2], crop_boxes[3])
            rend, _ = generate_image_patch(rend, cx, cy, s, 512, 0, [1., 1., 1.])
            rend[rend.sum(-1) == 0] = 255

            result_images.append(rend.astype(np.uint8))

        for i in range(15):
            result_images.append((image_org_flow.astype(np.float32) * i / 15 + image_rend_flow.astype(np.float32) * (1 - i / 15)).astype(np.uint8))

        return result_images

    rend_images_no_flow = visualize_human_object(hoi_recon_no_flow)
    rend_images_flow = visualize_human_object(hoi_recon_flow)
    video = cv2.VideoWriter(os.path.join(results_dir, '{}_{}_{}_comparison.mp4'.format(video_id, hoi_id, frame_id)), cv2.VideoWriter_fourcc(*'mp4v'), 30, (512 * 2, 512))
    for image_no_flow, image_flow in zip(rend_images_no_flow, rend_images_flow):
        cv2.putText(image_no_flow, "w/o flow", (206, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (10, 10, 10), 2)
        cv2.putText(image_flow, "with flow", (206, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (10, 10, 10), 2)
        image = np.concatenate([image_no_flow, image_flow], axis=1)
        video.write(image)
    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo")
    args = parser.parse_args()
    demo(args)
