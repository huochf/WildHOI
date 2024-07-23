import os
import argparse
import pickle
import numpy as np
import cv2
import trimesh
from tqdm import tqdm

import torch
from smplx import SMPLX

import neural_renderer as nr


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def render_hoi(smpl_v, smpl_f, object_v, object_f, R, T, focal, cx, cy, s):
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

    fx, fy = focal
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).reshape(1, 3, 3).to(device)
    R = R.reshape(1, 3, 3).to(device)
    t = T.reshape(1, 3).to(device)

    renderer = nr.renderer.Renderer(image_size=s, K=K / s, R=R, t=t, orig_size=1)
    
    renderer.background_color = [1, 1, 1]
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


def visualize_hoi(args):
    osx_dir = os.path.join(args.root_dir, 'smpler_x', '{:04d}'.format(args.video_idx))
    osx_smpl_data = load_pickle(os.path.join(osx_dir, '{:03d}_smplx.pkl'.format(args.sequence_idx)))

    object_pose_dir = os.path.join(args.root_dir, 'object_pose_framewise', '{:04d}'.format(args.video_idx))
    object_pose_data = load_pickle(os.path.join(object_pose_dir, '{:03d}_obj_RT.pkl'.format(args.sequence_idx)))

    image_dir = os.path.join(args.root_dir, 'images_temp', '{:04d}'.format(args.video_idx))

    device = torch.device('cuda')
    object_name = args.root_dir.split('/')[-1]
    if object_name == 'cello':
        object_mesh = trimesh.load('../data/objects/{}_body.ply'.format(object_name), process=False)
    else:
        object_mesh = trimesh.load('../data/objects/{}.ply'.format(object_name), process=False)
    object_v_org = torch.tensor(np.array(object_mesh.vertices), dtype=torch.float32).to(device)
    object_f = torch.tensor(np.array(object_mesh.faces, dtype=np.int64)).to(device)
    object_v_org = object_v_org - object_v_org.mean(0).reshape(1, -1)

    frame_id = object_pose_data[0]['frame_id']
    image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))
    h, w, _ = image.shape

    smplx = SMPLX('/public/home/huochf/projects/3D_HOI/hoiYouTube/data/smpl/smplx/', gender='neutral', use_pca=False).to(device)

    smpl_f = torch.tensor(smplx.faces.astype(np.int64)).to(device)

    video = cv2.VideoWriter('./__debug__/hoi_render_{:04d}_{:03d}.mp4'.format(args.video_idx, args.sequence_idx), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    for idx, smpl_item in enumerate(tqdm(osx_smpl_data)):
        object_item = object_pose_data[idx]
        assert smpl_item['frame_id'] == object_item['frame_id']
        frame_id = smpl_item['frame_id']
        smplx_orient = torch.tensor(smpl_item['global_orient']).reshape(1, 3).float().to(device)
        smplx_body_pose = torch.tensor(smpl_item['body_pose']).reshape(1, 63).float().to(device)
        smplx_lhand_pose = torch.tensor(smpl_item['left_hand_pose']).reshape(1, 45).float().to(device)
        smplx_rhand_pose = torch.tensor(smpl_item['right_hand_pose']).reshape(1, 45).float().to(device)
        smplx_jaw_pose = torch.tensor(smpl_item['jaw_pose']).reshape(1, 3).float().to(device)
        smplx_shape = torch.tensor(smpl_item['betas']).reshape(1, 10).float().to(device)
        smplx_expr = torch.tensor(smpl_item['expression']).reshape(1, 10).float().to(device)
        cam_trans = torch.tensor(smpl_item['transl']).reshape(1, 1, 3).float().to(device)

        smplx_out = smplx(betas=smplx_shape, 
                          global_orient=smplx_orient, 
                          body_pose=smplx_body_pose, 
                          left_hand_pose=smplx_lhand_pose,
                          right_hand_pose=smplx_rhand_pose,
                          expression=smplx_expr,
                          jaw_pose=smplx_jaw_pose,)
        smpl_v = smplx_out.vertices.detach()

        smpl_v = smpl_v + cam_trans

        rotmat = torch.tensor(object_item['rotmat']).reshape(3, 3).float().to(device)
        trans = torch.tensor(object_item['trans']).reshape(1, 3).float().to(device)
        success = object_item['success']
        if success:
            object_v = object_v_org @ rotmat.transpose(1, 0) + trans

            fx, fy = smpl_item['focal']
            cx, cy = smpl_item['princpt']
            R = torch.eye(3, dtype=torch.float32).to(device)
            T = torch.zeros(3, dtype=torch.float32).to(device)
            s = max(h, w)
            K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).to(device) / w

            image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))
            rend, mask = render_hoi(smpl_v, smpl_f, object_v, object_f, R, T, [fx, fy], cx, cy, s)

            rend_full = rend[:h, :w]
            mask_full = mask[:h, :w]
            image = image * (1 - mask_full) + rend_full * mask_full
        image = image.astype(np.uint8)

        video.write(image)
    video.release()


if __name__ == '__main__':
    # envs: preprocess
    parser = argparse.ArgumentParser(description='BigDetection Inference.')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    parser.add_argument('--video_idx', type=int)
    parser.add_argument('--sequence_idx', type=int, )
    args = parser.parse_args()

    visualize_hoi(args)
