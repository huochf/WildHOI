import os
import argparse
import pickle
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import torch
from smplx import SMPLX

import neural_renderer as nr


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def render_smpl(smpl_v, smpl_f, color, K, R, t, s, s_orig):
    vertices = smpl_v.reshape(1, -1, 3)
    faces = smpl_f.reshape(1, -1, 3)
    textures = torch.tensor(color, dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).repeat(1, faces.shape[1], 1, 1, 1, 1).to(vertices.device)

    K = K.reshape(1, 3, 3)
    R = R.reshape(1, 3, 3)
    t = t.reshape(1, 3)

    renderer = nr.renderer.Renderer(image_size=s, K=K, R=R, t=t, orig_size=s_orig)
    renderer.background_color = [1, 1, 1]
    renderer.light_direction = [1.5, -0.5, -1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.5
    rend, _, mask = renderer.render(vertices=vertices, faces=faces, textures=textures)
    rend = rend.clip(0, 1)

    return rend[0], mask[0]


def visualize_smpl(args):
    osx_dir = os.path.join(args.root_dir, 'smplx_tuned', '{:04d}'.format(args.video_idx))
    osx_smpl_data = load_pickle(os.path.join(osx_dir, '{:03d}_smplx.pkl'.format(args.sequence_idx)))

    image_dir = os.path.join(args.root_dir, 'images_temp', '{:04d}'.format(args.video_idx))

    device = torch.device('cuda')
    smplx = SMPLX('/public/home/huochf/projects/3D_HOI/hoiYouTube/data/smpl/smplx/', gender='neutral', use_pca=False).to(device)

    smpl_f = torch.tensor(smplx.faces.astype(np.int64)).to(device)

    image_org = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(osx_smpl_data[0]['frame_id'])))
    h, w, _ = image_org.shape

    video = cv2.VideoWriter('./__debug__/smplx_render_{:04d}_{:03d}.mp4'.format(args.video_idx, args.sequence_idx), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w * 2, h))

    for item in tqdm(osx_smpl_data):
        frame_id = item['frame_id']
        smplx_body_pose = torch.tensor(item['body_pose']).reshape(1, 63).float().to(device)
        smplx_lhand_pose = torch.tensor(item['left_hand_pose']).reshape(1, 45).float().to(device)
        smplx_rhand_pose = torch.tensor(item['right_hand_pose']).reshape(1, 45).float().to(device)
        smplx_shape = torch.tensor(item['betas']).reshape(1, 10).float().to(device)

        cam_R = torch.tensor(item['cam_R']).reshape(3, 3).float().to(device)
        cam_T = torch.tensor(item['cam_T']).reshape(1, 3).float().to(device)

        smplx_out = smplx(betas=smplx_shape, 
                          body_pose=smplx_body_pose, 
                          left_hand_pose=smplx_lhand_pose,
                          right_hand_pose=smplx_rhand_pose,)
        smpl_v = smplx_out.vertices.detach()[0]
        smpl_J = smplx_out.joints.detach()[0]
        smpl_v = smpl_v - smpl_J[:1]
        smpl_v = smpl_v @ cam_R.transpose(1, 0)

        # smpl_v = smpl_v + cam_T

        fx, fy = item['focal']
        cx, cy = item['princpt']
        R = torch.eye(3, dtype=torch.float32).to(device)
        T = torch.tensor(item['cam_T'], dtype=torch.float32).to(device)
        s = max(h, w)
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).to(device) / w

        image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))
        rend, mask = render_smpl(smpl_v, smpl_f, np.array([255,127,80]) / 255., K, R, T, s, 1)

        rend = rend.permute(1, 2, 0).detach().cpu().numpy()[:h, :w, ::-1]
        mask = mask.detach().cpu().numpy()[:h, :w]
        mask = mask.reshape(h, w, 1)
        rend = (rend * 255).astype(np.uint8)
        image = image * (1 - mask) + rend * mask
        image = image.astype(np.uint8)

        smpl_v = smpl_v @ torch.tensor(Rotation.from_euler('y', 90, degrees=True).as_matrix()).float().to(device)
        rend_side, _ = render_smpl(smpl_v, smpl_f, np.array([255,127,80]) / 255., K, R, T, s, 1)
        rend_side = rend_side.permute(1, 2, 0).detach().cpu().numpy()[:h, :w, ::-1]
        rend_side = (rend_side * 255).astype(np.uint8)

        image = np.concatenate([image, rend_side], axis=1)

        video.write(image)
    video.release()


if __name__ == '__main__':
    # envs: preprocess
    parser = argparse.ArgumentParser(description='BigDetection Inference.')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    parser.add_argument('--video_idx', type=int)
    parser.add_argument('--sequence_idx', type=int, )
    args = parser.parse_args()

    visualize_smpl(args)
