import os
import argparse
import trimesh
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import torch

import neural_renderer as nr


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def render(v, f, K, s, t, R):
    device = torch.device('cuda')
    v = v @ R.transpose(1, 0) + t.reshape(1, 3)
    v = v.reshape(1, -1, 3).to(device)
    f = f.reshape(1, -1, 3).to(device)
    textures = torch.tensor([0.65098039, 0.74117647, 0.85882353], dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).repeat(1, f.shape[1], 1, 1, 1, 1).to(device)

    K = torch.tensor(K).float().reshape(1, 3, 3).to(device)
    R = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).to(device)
    t = torch.zeros(3, dtype=torch.float32).reshape(1, 3).to(device)

    renderer = nr.renderer.Renderer(image_size=s, K=K / s, R=R, t=t, orig_size=1)
    renderer.background_color = [1, 1, 1]
    renderer.light_direction = [1.5, -0.5, 1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.5
    rend, _, mask = renderer.render(vertices=v, faces=f, textures=textures)
    rend = rend.clip(0, 1)

    rend = rend[0].permute(1, 2, 0).detach().cpu().numpy()
    mask = mask[0].detach().cpu().numpy().reshape((s, s, 1))
    rend = (rend * 255).astype(np.uint8)
    rend = rend[:, :, ::-1]

    return rend, mask


def visualize(args):
    object_pose_dir = os.path.join(args.root_dir, 'object_pose_framewise', '{:04d}'.format(args.video_idx))
    object_pose_data = load_pickle(os.path.join(object_pose_dir, '{:03d}_obj_RT.pkl'.format(args.sequence_idx)))

    image_dir = os.path.join(args.root_dir, 'images_temp', '{:04d}'.format(args.video_idx))

    device = torch.device('cuda')
    object_name = args.root_dir.split('/')[-1]
    if object_name == 'cello':
        object_mesh = trimesh.load('../data/objects/{}_body.ply'.format(object_name), process=False)
    else:
        object_mesh = trimesh.load('../data/objects/{}_bar.ply'.format(object_name), process=False)
    object_v = torch.tensor(np.array(object_mesh.vertices), dtype=torch.float32).to(device)
    object_f = torch.tensor(np.array(object_mesh.faces, dtype=np.int64)).to(device)
    object_v = object_v - object_v.mean(0).reshape(1, -1)

    frame_id = object_pose_data[0]['frame_id']
    image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))
    h, w, _ = image.shape

    video = cv2.VideoWriter('./__debug__/obj_render_{:04d}_{:03d}.mp4'.format(args.video_idx, args.sequence_idx), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    for idx, item in enumerate(tqdm(object_pose_data)):
        frame_id = item['frame_id']
        rotmat = item['rotmat']
        trans = item['trans']
        success = item['success']
        smpl_princpt = item['smpl_princpt']
        smpl_focal = item['smpl_focal']
        smpl_bboxes = item['smpl_bboxes']

        image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))

        R = torch.tensor(rotmat, dtype=torch.float32).to(device)
        T = torch.tensor(trans, dtype=torch.float32).to(device)
        fx, fy = smpl_focal
        cx, cy = smpl_princpt
        s = max(h, w)
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).to(device)

        if success:
            rend, mask = render(object_v.clone(), object_f.clone(), K, s, T, R)
        else:
            rend = np.zeros((s, s, 3)).astype(np.uint8)
            mask = np.zeros((s, s, 1)).astype(np.uint8)

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

    visualize(args)
