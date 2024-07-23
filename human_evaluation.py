import os
import sys
import argparse
import trimesh
import pickle
import numpy as np
import cv2
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import torch
from smplx import SMPLX

import neural_renderer as nr

from datasets.utils import generate_image_patch


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
    mask = mask[0].detach().cpu().numpy().reshape((s, s, 1))
    rend = (rend * 255).astype(np.uint8)
    rend = rend[:, :, ::-1]

    return rend, mask


def prepare_images(args):
    device = torch.device('cuda')
    object_name = args.root_dir.split('/')[-1]

    if object_name in ['cello', 'violin']:
        object_mesh = trimesh.load('data/objects/{}_body.ply'.format(object_name), process=False)
    else:
        object_mesh = trimesh.load('data/objects/{}.ply'.format(object_name), process=False)
    object_v = torch.tensor(np.array(object_mesh.vertices), dtype=torch.float32).to(device)
    object_f = torch.tensor(np.array(object_mesh.faces, dtype=np.int64)).to(device)
    object_v_org = object_v - object_v.mean(0).reshape(1, -1)

    smplx = SMPLX('data/smpl/smplx/', gender='neutral', use_pca=False).to(device)
    smpl_f = torch.tensor(smplx.faces.astype(np.int64)).to(device)

    phosa_hoi_recon_results = load_pickle('./outputs/optim_with_phosa/{}_test.pkl'.format(object_name))
    kps_flow_hoi_recon_results = load_pickle('./outputs/optim_with_contact_kps_flow/{}_test.pkl'.format(object_name))

    image_ids = [item['image_id'] for item in phosa_hoi_recon_results]

    phosa_hoi_recon_results = {item['image_id']: item for item in phosa_hoi_recon_results}
    kps_flow_hoi_recon_results = {item['image_id']: item for item in kps_flow_hoi_recon_results}

    save_dir = '__debug__/{}_human_evaluation'.format(object_name)
    os.makedirs(save_dir, exist_ok=True)

    np.random.seed(7)
    phosa_on_tops = {}

    for image_id in tqdm(image_ids):

        video_id, hoi_id, frame_id = image_id.split('_')
        image_dir = os.path.join(args.root_dir, 'images_temp', video_id)
        image_org = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))
        h, w, _ = image_org.shape

        if image_id not in phosa_hoi_recon_results:
            print('key error {}'.format(image_id))
            continue
        if image_id not in kps_flow_hoi_recon_results:
            print('key error {}'.format(image_id))
            continue

        item = phosa_hoi_recon_results[image_id]
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

        crop_boxes = item['crop_bboxes']
        cx, cy = crop_boxes[0] + crop_boxes[2] / 2, crop_boxes[1] + crop_boxes[3] / 2
        s = 1.2 * max(crop_boxes[2], crop_boxes[3])
        image_rend_phosa, _ = generate_image_patch(image_rend, cx, cy, s, 512, 0, [1., 1., 1.])
        image_org_phosa, _ = generate_image_patch(image_org, cx, cy, s, 512, 0, [1., 1., 1.])

        s = 512
        cx = cy = 256
        fx = fy = 5 * 256
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().to(device) / s
        z0 = 5 * 1.5
        smpl_centered_images_phosa = []
        for rot in range(0, 360, 60):
            rotmat_centered = torch.tensor(Rotation.from_euler('y', rot, degrees=True).as_matrix()).float().to(device) \
                @ torch.tensor(Rotation.from_euler('z', 180, degrees=True).as_matrix()).float().to(device)
            _smpl_v = smpl_v @ rotmat_centered.transpose(1, 0)
            _object_v = object_v @ rotmat_centered.transpose(1, 0)
            _smpl_v[:, 2] += z0
            _object_v[:, 2] += z0

            rend, mask = render_hoi(_smpl_v, smpl_f, _object_v, object_f, 
                torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32), K, s, novel_views=True)
            smpl_centered_images_phosa.append(rend)
        smpl_centered_images_phosa = np.concatenate(smpl_centered_images_phosa).reshape(2, 3, s, s, 3)
        smpl_centered_images_phosa = smpl_centered_images_phosa.transpose(0, 2, 1, 3, 4).reshape(2 * s, 3 * s, 3)
        smpl_centered_images_phosa = smpl_centered_images_phosa.astype(np.uint8)
        image_show_phosa = np.concatenate([image_org_phosa, image_rend_phosa], axis=0) # [2 * s, 3, 3]
        image_show_phosa = np.concatenate([image_show_phosa, smpl_centered_images_phosa], axis=1)


        item = kps_flow_hoi_recon_results[image_id]
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

        crop_boxes = item['crop_bboxes']
        cx, cy = crop_boxes[0] + crop_boxes[2] / 2, crop_boxes[1] + crop_boxes[3] / 2
        s = 1.2 * max(crop_boxes[2], crop_boxes[3])
        image_rend_flow, _ = generate_image_patch(image_rend, cx, cy, s, 512, 0, [1., 1., 1.])

        image_org_flow, _ = generate_image_patch(image_org, cx, cy, s, 512, 0, [1., 1., 1.])

        s = 512
        cx = cy = 256
        fx = fy = 5 * 256
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().to(device) / s
        z0 = 5 * 1.5
        smpl_centered_images_flow = []
        for rot in range(0, 360, 60):
            rotmat_centered = torch.tensor(Rotation.from_euler('y', rot, degrees=True).as_matrix()).float().to(device) \
                @ torch.tensor(Rotation.from_euler('z', 180, degrees=True).as_matrix()).float().to(device)
            _smpl_v = smpl_v @ rotmat_centered.transpose(1, 0)
            _object_v = object_v @ rotmat_centered.transpose(1, 0)
            _smpl_v[:, 2] += z0
            _object_v[:, 2] += z0

            rend, mask = render_hoi(_smpl_v, smpl_f, _object_v, object_f, 
                torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32), K, s, novel_views=True)
            smpl_centered_images_flow.append(rend)
        smpl_centered_images_flow = np.concatenate(smpl_centered_images_flow).reshape(2, 3, s, s, 3)
        smpl_centered_images_flow = smpl_centered_images_flow.transpose(0, 2, 1, 3, 4).reshape(2 * s, 3 * s, 3)
        smpl_centered_images_flow = smpl_centered_images_flow.astype(np.uint8)
        image_show_flow = np.concatenate([image_org_flow, image_rend_flow], axis=0) # [2 * s, 3, 3]
        image_show_flow = np.concatenate([image_show_flow, smpl_centered_images_flow], axis=1)

        phosa_top = np.random.randint(2)
        if phosa_top == 1:
            image_compares = np.concatenate([image_show_phosa, image_show_flow], axis=0)
        else:
            image_compares = np.concatenate([image_show_flow, image_show_phosa], axis=0)

        phosa_on_tops[image_id] = phosa_top
        cv2.imwrite(os.path.join(save_dir, '{}_comparison.jpg'.format(image_id)), image_compares)

    with open(os.path.join(save_dir, 'phosa_on_tops.json'), 'w') as f:
        json.dump(phosa_on_tops, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize HOI reconstruction results.')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    args = parser.parse_args()

    prepare_images(args)
