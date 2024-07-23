import os
import argparse
from tqdm import tqdm
import pickle
import trimesh
import cv2
import numpy as np

from plot_hoi_kps_img import plot_object_keypoints, plot_smpl_keypoints


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def visualize_hoi_kps_all(args):
    object_name = args.root_dir.split('/')[-1]

    for video_id in tqdm(sorted(os.listdir(os.path.join(args.root_dir, 'hoi_kps')))):
        for file in os.listdir(os.path.join(args.root_dir, 'hoi_kps', video_id)):
            hoi_id = file.split('_')[0]
            kps_load = load_pickle(os.path.join(args.root_dir, 'hoi_kps', video_id, file))
            res = 256

            if 'focal_seq' not in kps_load:
                continue
            video = cv2.VideoWriter('./__debug__/hoi_kps_{}_{}.mp4'.format(video_id, hoi_id), cv2.VideoWriter_fourcc(*'mp4v'), 30, (res, res))
            frame_idx = 0
            for smpl_kps, object_kps, focal, cam_R, cam_T in zip(
                kps_load['smpl_kps_seq'], kps_load['object_kps_seq'], kps_load['focal_seq'], kps_load['cam_R_seq'], kps_load['cam_T_seq']):
                focal = np.array(focal).reshape(1, 2)
                smpl_kps = smpl_kps / focal
                object_kps = object_kps / focal
                n_smpl_kps, n_object_kps = smpl_kps.shape[0], object_kps.shape[0]
                smpl_kps = np.concatenate([smpl_kps, np.ones((n_smpl_kps, 1))], axis=1)
                object_kps = np.concatenate([object_kps, np.ones((n_object_kps, 1))], axis=1)
                hoi_kps = np.concatenate([smpl_kps, object_kps, np.zeros((1, 3))], axis=0) # [n_smpl_kps + n_object_kps + 1, 3]

                hoi_kps = (hoi_kps - cam_T.reshape(1, 3)) @ cam_R.transpose(1, 0)
                cam_pos = hoi_kps[-1:]
                kps_directions = hoi_kps[:-1] - hoi_kps[-1:]
                kps_directions = kps_directions / np.linalg.norm(kps_directions, axis=1, keepdims=True)

                kps_cos = (kps_directions * kps_directions[:1]).sum(1)
                kps_vis = kps_directions * 1 / kps_cos.reshape(-1, 1)

                y_axis = np.array([0, 1, 0])
                axis_u = np.cross(kps_directions[0], y_axis)
                axis_u = axis_u / np.linalg.norm(axis_u)
                axis_v = - np.cross(axis_u, kps_directions[0])
                axis_v = axis_v / np.linalg.norm(axis_v)

                axis_u, axis_v = axis_u.reshape(1, 3), axis_v.reshape(1, 3)
                u = (kps_directions * axis_u).sum(1)
                v = (kps_directions * axis_v).sum(1)

                kps2d_vis = np.stack([u, v], axis=1)

                f_fix = 5000 / 256
                kps2d_vis = kps2d_vis * f_fix # [1, -1]
                smpl_kps = kps2d_vis[:22]
                object_kps = kps2d_vis[22:]

                smpl_kps = (smpl_kps + 1) / 2 * res
                object_kps = (object_kps + 1) / 2 * res
                image_vis = (np.ones((res, res, 3)) * 255).astype(np.uint8)
                if not np.isnan(smpl_kps).any():
                    image_vis = plot_smpl_keypoints(image_vis, smpl_kps)
                if not np.isnan(object_kps).any():
                    image_vis = plot_object_keypoints(image_vis, object_kps, object_name)
                video.write(image_vis)
            video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BigDetection Inference.')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    # parser.add_argument('--video_idx', type=int)
    # parser.add_argument('--sequence_idx', type=int, )
    args = parser.parse_args()

    visualize_hoi_kps_all(args)
