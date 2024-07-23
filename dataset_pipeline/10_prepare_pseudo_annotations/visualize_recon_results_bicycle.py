import os
import argparse
import trimesh
import pickle
import numpy as np
import cv2
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix

import torch
from smplx import SMPLX

import neural_renderer as nr


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


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

    renderer = nr.renderer.Renderer(image_size=s, K=K, R=R, t=t, orig_size=1)
    
    renderer.background_color = [1, 1, 1]
    if novel_views:
        renderer.light_direction = [1, 0.5, 1]
    else:
        renderer.light_direction = [1, 0.5, -1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.5

    rend, _, mask = renderer.render(vertices=vertices, faces=faces, textures=textures)
    rend = rend.clip(0, 1)
    rend = rend[0].permute(1, 2, 0).detach().cpu().numpy()
    mask = mask[0].detach().cpu().numpy().reshape((s, s, 1)) * 0.7
    rend = (rend * 255).astype(np.uint8)
    rend = rend[:, :, ::-1]

    return rend, mask


def visualize_hoi(args):
    device = torch.device('cuda')
    object_name = args.root_dir.split('/')[-1]

    if object_name in ['cello', 'violin']:
        object_mesh = trimesh.load('../data/objects/{}_body.ply'.format(object_name), process=False)
    else:
        object_mesh = trimesh.load('../data/objects/{}.ply'.format(object_name), process=False)
    object_v = torch.tensor(np.array(object_mesh.vertices), dtype=torch.float32).to(device)
    object_f = torch.tensor(np.array(object_mesh.faces, dtype=np.int64)).to(device)
    object_v_org = object_v - object_v.mean(0).reshape(1, -1)

    smplx = SMPLX('/public/home/huochf/projects/3D_HOI/hoiYouTube/data/smpl/smplx/', gender='neutral', use_pca=False).to(device)
    smpl_f = torch.tensor(smplx.faces.astype(np.int64)).to(device)

    video_id = '{:04d}'.format(args.video_idx)
    hoi_id = '{:03d}'.format(args.sequence_idx)
    image_dir = os.path.join(args.root_dir, 'images_temp', video_id)
    hoi_recon_results = load_pickle(os.path.join(args.root_dir, 'hoi_joint_optim_with_contact_labels_v2', video_id, '{}.pkl'.format(hoi_id)))
    save_dir = '__debug__/{}'.format(video_id)
    os.makedirs(save_dir, exist_ok=True)

    for item in tqdm(hoi_recon_results):
        frame_id = item['frame_id']

        image_org = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))
        h, w, _ = image_org.shape

        smplx_betas = torch.tensor(item['smplx_betas']).reshape(1, 10).float().to(device)
        smplx_body_pose = torch.tensor(item['smplx_body_pose']).reshape(1, 63).float().to(device)
        smplx_lhand_pose = torch.tensor(item['smplx_lhand_pose']).reshape(1, 45).float().to(device)
        smplx_rhand_pose = torch.tensor(item['smplx_rhand_pose']).reshape(1, 45).float().to(device)

        smplx_out = smplx(betas=smplx_betas, body_pose=smplx_body_pose, left_hand_pose=smplx_lhand_pose, right_hand_pose=smplx_rhand_pose)
        smpl_v = smplx_out.vertices.detach()[0]
        smpl_J = smplx_out.joints.detach()[0]
        smpl_v = smpl_v - smpl_J[:1]

        object_scale = torch.tensor(item['object_scale']).float().to(device).reshape(1, )
        object_v = object_v_org * object_scale
        object_rel_rotmat = torch.tensor(item['obj_rel_rotmat']).float().to(device).reshape(3, 3)
        object_rel_trans = torch.tensor(item['obj_rel_trans']).float().to(device).reshape(1, 3)
        object_v = object_v @ object_rel_rotmat.transpose(1, 0) + object_rel_trans

        hoi_rotmat = torch.tensor(item['hoi_rotmat']).reshape(3, 3).float().to(device)
        hoi_trans = torch.tensor(item['hoi_trans']).reshape(1, 3).float().to(device)

        fx, fy = item['focal']
        cx, cy = item['princpt']
        s = max(h, w)
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().to(device) / s

        rend, mask = render_hoi(smpl_v, smpl_f, object_v, object_f, hoi_rotmat, hoi_trans, K, s)
        rend = rend[:h, :w]
        mask = mask[:h, :w]
        mask = mask.reshape(h, w, 1)
        image = image_org * (1 - mask) + rend * mask
        image_rend = image.astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, '{}_{}_cam_view.jpg'.format(hoi_id, frame_id)), image_rend)

        s = 512
        cx = cy = 256
        fx = fy = 5 * 256
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().to(device) / s
        z0 = 5 * 1.5
        smpl_centered_images = []
        for rot in range(0, 360, 60):
            rotmat_centered = torch.tensor(Rotation.from_euler('y', rot, degrees=True).as_matrix()).float().to(device) \
                @ torch.tensor(Rotation.from_euler('z', 180, degrees=True).as_matrix()).float().to(device)
            _smpl_v = smpl_v @ rotmat_centered.transpose(1, 0)
            _object_v = object_v @ rotmat_centered.transpose(1, 0)
            _smpl_v[:, 2] += z0
            _object_v[:, 2] += z0

            rend, mask = render_hoi(_smpl_v, smpl_f, _object_v, object_f, 
                torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32), K, s, novel_views=True)
            smpl_centered_images.append(rend)
        smpl_centered_images = np.concatenate(smpl_centered_images).reshape(2, 3, s, s, 3)
        smpl_centered_images = smpl_centered_images.transpose(0, 2, 1, 3, 4).reshape(2 * s, 3 * s, 3)
        smpl_centered_images = smpl_centered_images.astype(np.uint8)

        cv2.imwrite(os.path.join(save_dir, '{}_{}_novel_views.jpg'.format(hoi_id, frame_id)), smpl_centered_images)


def visualize_hoi_all(args):
    device = torch.device('cuda')
    object_name = args.root_dir.split('/')[-1]

    bicycle_front = trimesh.load('../data/objects/bicycle_front.ply', process=False)
    bicycle_back = trimesh.load('../data/objects/bicycle_back.ply', process=False)

    bicycle_front_f = torch.from_numpy(np.array(bicycle_front.faces))
    bicycle_back_f = torch.from_numpy(np.array(bicycle_back.faces))
    bicycle_front = torch.from_numpy(bicycle_front.vertices).float()
    bicycle_back = torch.from_numpy(bicycle_back.vertices).float()
    object_f = torch.cat([bicycle_front_f, bicycle_back_f + bicycle_front.shape[0]], dim=0).float().to(device)

    with open('../data/objects/bicycle_front_keypoints.json', 'r') as f:
        bicycle_front_kps_indices = json.load(f)
    with open('../data/objects/bicycle_back_keypoints.json', 'r') as f:
        bicycle_back_kps_indices = json.load(f)
    rot_axis_begin = bicycle_front[bicycle_front_kps_indices['5']].mean(0)
    rot_axis_end = bicycle_front[bicycle_front_kps_indices['6']].mean(0)
    rot_axis = rot_axis_end - rot_axis_begin
    rot_axis = rot_axis / torch.sqrt((rot_axis ** 2).sum())

    smplx = SMPLX('/public/home/huochf/projects/3D_HOI/hoiYouTube/data/smpl/smplx/', gender='neutral', use_pca=False).to(device)
    smpl_f = torch.tensor(smplx.faces.astype(np.int64)).to(device)

    save_dir = '__debug__/{}_recon_vis'.format(object_name)
    os.makedirs(save_dir, exist_ok=True)

    hoi_recon_results = load_pickle(os.path.join('hoi_recon_with_contact', '{}_test.pkl'.format(object_name)))

    for item in tqdm(hoi_recon_results):
        image_id = item['image_id']
        video_id, hoi_id, frame_id = image_id.split('_')

        image_org = cv2.imread(os.path.join(args.root_dir, 'images_temp', video_id, '{}.jpg'.format(frame_id)))
        h, w, _ = image_org.shape

        smplx_betas = torch.tensor(item['smplx_betas']).reshape(1, 10).float().to(device)
        smplx_body_pose = torch.tensor(item['smplx_body_pose']).reshape(1, 63).float().to(device)
        smplx_lhand_pose = torch.tensor(item['smplx_lhand_pose']).reshape(1, 45).float().to(device)
        smplx_rhand_pose = torch.tensor(item['smplx_rhand_pose']).reshape(1, 45).float().to(device)

        smplx_out = smplx(betas=smplx_betas, body_pose=smplx_body_pose, left_hand_pose=smplx_lhand_pose, right_hand_pose=smplx_rhand_pose)
        smpl_v = smplx_out.vertices.detach()[0]
        smpl_J = smplx_out.joints.detach()[0]
        smpl_v = smpl_v - smpl_J[:1]

        rot_angle = torch.tensor(item['object_rot_angle'])
        _bicycle_front = bicycle_front - rot_axis_begin.reshape(1, 3)
        front_rotmat = axis_angle_to_matrix(rot_axis * rot_angle) # [3, 3]
        _bicycle_front = _bicycle_front @ front_rotmat.transpose(1, 0)
        _bicycle_front = _bicycle_front + rot_axis_begin.reshape(1, 3)
        object_v_org = torch.cat([_bicycle_front, bicycle_back], dim=0)
        object_v_org = object_v_org.float().to(device).reshape(-1, 3)

        object_scale = torch.tensor(item['object_scale']).float().to(device).reshape(1, )
        object_v = object_v_org * object_scale
        object_rel_rotmat = torch.tensor(item['obj_rel_rotmat']).float().to(device).reshape(3, 3)
        object_rel_trans = torch.tensor(item['obj_rel_trans']).float().to(device).reshape(1, 3)
        object_v = object_v @ object_rel_rotmat.transpose(1, 0) + object_rel_trans

        hoi_rotmat = torch.tensor(item['hoi_rotmat']).reshape(3, 3).float().to(device)
        hoi_trans = torch.tensor(item['hoi_trans']).reshape(1, 3).float().to(device)

        fx, fy = item['focal']
        cx, cy = item['princpt']
        s = max(h, w)
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().to(device) / s

        rend, mask = render_hoi(smpl_v, smpl_f, object_v, object_f, hoi_rotmat, hoi_trans, K, s)
        rend = rend[:h, :w]
        mask = mask[:h, :w]
        mask = mask.reshape(h, w, 1)
        image = image_org * (1 - mask) + rend * mask
        image_rend = image.astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, '{}_{}_{}_cam_view.jpg'.format(video_id, hoi_id, frame_id)), image_rend)

        s = 512
        cx = cy = 256
        fx = fy = 5 * 256
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().to(device) / s
        z0 = 5 * 1.5
        smpl_centered_images = []
        for rot in range(0, 360, 60):
            rotmat_centered = torch.tensor(Rotation.from_euler('y', rot, degrees=True).as_matrix()).float().to(device) \
                @ torch.tensor(Rotation.from_euler('z', 180, degrees=True).as_matrix()).float().to(device)
            _smpl_v = smpl_v @ rotmat_centered.transpose(1, 0)
            _object_v = object_v @ rotmat_centered.transpose(1, 0)
            _smpl_v[:, 2] += z0
            _object_v[:, 2] += z0

            rend, mask = render_hoi(_smpl_v, smpl_f, _object_v, object_f, 
                torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32), K, s, novel_views=True)
            smpl_centered_images.append(rend)
        smpl_centered_images = np.concatenate(smpl_centered_images).reshape(2, 3, s, s, 3)
        smpl_centered_images = smpl_centered_images.transpose(0, 2, 1, 3, 4).reshape(2 * s, 3 * s, 3)
        smpl_centered_images = smpl_centered_images.astype(np.uint8)

        cv2.imwrite(os.path.join(save_dir, '{}_{}_{}_novel_views.jpg'.format(video_id, hoi_id, frame_id)), smpl_centered_images)


def visualize_hoi_annotations(args):
    device = torch.device('cuda')
    object_name = args.root_dir.split('/')[-1]

    bicycle_front = trimesh.load('../data/objects/bicycle_front.ply', process=False)
    bicycle_back = trimesh.load('../data/objects/bicycle_back.ply', process=False)

    bicycle_front_f = torch.from_numpy(np.array(bicycle_front.faces))
    bicycle_back_f = torch.from_numpy(np.array(bicycle_back.faces))
    bicycle_front = torch.from_numpy(bicycle_front.vertices).float()
    bicycle_back = torch.from_numpy(bicycle_back.vertices).float()
    object_f = torch.cat([bicycle_front_f, bicycle_back_f + bicycle_front.shape[0]], dim=0).float().to(device)

    with open('../data/objects/bicycle_front_keypoints.json', 'r') as f:
        bicycle_front_kps_indices = json.load(f)
    with open('../data/objects/bicycle_back_keypoints.json', 'r') as f:
        bicycle_back_kps_indices = json.load(f)
    rot_axis_begin = bicycle_front[bicycle_front_kps_indices['5']].mean(0)
    rot_axis_end = bicycle_front[bicycle_front_kps_indices['6']].mean(0)
    rot_axis = rot_axis_end - rot_axis_begin
    rot_axis = rot_axis / torch.sqrt((rot_axis ** 2).sum())

    smplx = SMPLX('/public/home/huochf/projects/3D_HOI/hoiYouTube/data/smpl/smplx/', gender='neutral', use_pca=False).to(device)
    smpl_f = torch.tensor(smplx.faces.astype(np.int64)).to(device)

    save_dir = '__debug__/{}_annotation_vis'.format(object_name)
    os.makedirs(save_dir, exist_ok=True)

    for file in tqdm(os.listdir('./annotation_hoi/{}/test'.format(object_name))):
        image_id = file.split('.')[0]
        video_id, hoi_id, frame_id = image_id.split('_')

        item = load_pickle(os.path.join('./annotation_hoi/{}/test'.format(object_name), file))

        image_org = cv2.imread(os.path.join(args.root_dir, 'images_temp', video_id, '{}.jpg'.format(frame_id)))
        h, w, _ = image_org.shape

        smplx_betas = torch.tensor(item['smplx_betas']).reshape(1, 10).float().to(device)
        smplx_body_pose = torch.tensor(item['smplx_body_pose']).reshape(1, 63).float().to(device)
        smplx_lhand_pose = torch.tensor(item['smplx_lhand_pose']).reshape(1, 45).float().to(device)
        smplx_rhand_pose = torch.tensor(item['smplx_rhand_pose']).reshape(1, 45).float().to(device)

        smplx_out = smplx(betas=smplx_betas, body_pose=smplx_body_pose, left_hand_pose=smplx_lhand_pose, right_hand_pose=smplx_rhand_pose)
        smpl_v = smplx_out.vertices.detach()[0]
        smpl_J = smplx_out.joints.detach()[0]
        smpl_v = smpl_v - smpl_J[:1]

        rot_angle = torch.tensor(item['object_rot_angle'])
        _bicycle_front = bicycle_front - rot_axis_begin.reshape(1, 3)
        front_rotmat = axis_angle_to_matrix(rot_axis * rot_angle) # [3, 3]
        _bicycle_front = _bicycle_front @ front_rotmat.transpose(1, 0)
        _bicycle_front = _bicycle_front + rot_axis_begin.reshape(1, 3)
        object_v_org = torch.cat([_bicycle_front, bicycle_back], dim=0)
        object_v_org = object_v_org.float().to(device).reshape(-1, 3)

        object_scale = torch.tensor(item['object_scale']).float().to(device).reshape(1, )
        object_v = object_v_org * object_scale
        object_rel_rotmat = torch.tensor(item['obj_rel_rotmat']).float().to(device).reshape(3, 3)
        object_rel_trans = torch.tensor(item['obj_rel_trans']).float().to(device).reshape(1, 3)
        object_v = object_v @ object_rel_rotmat.transpose(1, 0) + object_rel_trans

        hoi_rotmat = torch.tensor(item['hoi_rotmat']).reshape(3, 3).float().to(device)
        hoi_trans = torch.tensor(item['hoi_trans']).reshape(1, 3).float().to(device)

        fx, fy = item['focal']
        cx, cy = item['princpt']
        s = max(h, w)
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().to(device) / s

        rend, mask = render_hoi(smpl_v, smpl_f, object_v, object_f, hoi_rotmat, hoi_trans, K, s)
        rend = rend[:h, :w]
        mask = mask[:h, :w]
        mask = mask.reshape(h, w, 1)
        image = image_org * (1 - mask) + rend * mask
        image_rend = image.astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, '{}_{}_{}_cam_view.jpg'.format(video_id, hoi_id, frame_id)), image_rend)

        s = 512
        cx = cy = 256
        fx = fy = 5 * 256
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().to(device) / s
        z0 = 5 * 1.5
        smpl_centered_images = []
        for rot in range(0, 360, 60):
            rotmat_centered = torch.tensor(Rotation.from_euler('y', rot, degrees=True).as_matrix()).float().to(device) \
                @ torch.tensor(Rotation.from_euler('z', 180, degrees=True).as_matrix()).float().to(device)
            _smpl_v = smpl_v @ rotmat_centered.transpose(1, 0)
            _object_v = object_v @ rotmat_centered.transpose(1, 0)
            _smpl_v[:, 2] += z0
            _object_v[:, 2] += z0

            rend, mask = render_hoi(_smpl_v, smpl_f, _object_v, object_f, 
                torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32), K, s, novel_views=True)
            smpl_centered_images.append(rend)
        smpl_centered_images = np.concatenate(smpl_centered_images).reshape(2, 3, s, s, 3)
        smpl_centered_images = smpl_centered_images.transpose(0, 2, 1, 3, 4).reshape(2 * s, 3 * s, 3)
        smpl_centered_images = smpl_centered_images.astype(np.uint8)

        cv2.imwrite(os.path.join(save_dir, '{}_{}_{}_novel_views.jpg'.format(video_id, hoi_id, frame_id)), smpl_centered_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize HOI reconstruction results.')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    parser.add_argument('--video_idx', type=int)
    parser.add_argument('--sequence_idx', type=int, )
    args = parser.parse_args()

    # visualize_hoi(args)
    # visualize_hoi_all(args)
    visualize_hoi_annotations(args)
