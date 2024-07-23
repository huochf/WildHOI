import os
import argparse
import pickle
import numpy as np
import cv2
import torch
from tqdm import tqdm

from pose_model import PoseModel


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_and_smooth_wholebody_kps_sequence(wholebody_kps_list, windows=5):
    wholebody_kps_seq = np.array([item['keypoints'] for item in wholebody_kps_list]) # [n, 133, 3]

    seq_n, kps_n, _ = wholebody_kps_seq.shape
    smooth_kps = np.concatenate([np.zeros((windows // 2, kps_n, 3)), wholebody_kps_seq, np.zeros((windows // 2, kps_n, 3))], axis=0) # [seq_n + windows - 1, n, 3]
    confidence_score = np.stack([
        smooth_kps[i: seq_n + i, :, 2:] for i in range(windows)
    ], axis=0)
    smooth_kps = np.stack([
        smooth_kps[i: seq_n + i, :, :2] for i in range(windows)
    ], axis=0)
    smooth_kps = (smooth_kps * confidence_score).sum(0) / (confidence_score.sum(0) + 1e-8)
    smooth_kps = np.concatenate([smooth_kps, wholebody_kps_seq[:, :, 2:]], axis=2)

    return smooth_kps


def visualize(args):

    device = torch.device('cuda')
    wholebody_model = PoseModel('ViTPose-H-WholeBody', device)

    wholebody_dir = os.path.join(args.root_dir, 'wholebody_kps', '{:04d}'.format(args.video_idx), )
    wholebody_kps = load_pickle(os.path.join(wholebody_dir, '{:03d}_wholebody_kps.pkl'.format(args.sequence_idx)))

    image_dir = os.path.join(args.root_dir, 'images_temp', '{:04d}'.format(args.video_idx))

    image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(wholebody_kps[args.sequence_idx]['frame_id'])))
    h, w, _ = image.shape

    video = cv2.VideoWriter('./__debug__/wholebody_kps_{:04d}_{:03d}.mp4'.format(args.video_idx, args.sequence_idx), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    wholebody_kps_smoothed = load_and_smooth_wholebody_kps_sequence(wholebody_kps)
    for idx, item in enumerate(tqdm(wholebody_kps)):
        frame_id = item['frame_id']
        keypoints = item['keypoints']

        if frame_id != '011407':
            continue

        image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))
        # pose_output = [{'keypoints': keypoints}, ]
        pose_output = [{'keypoints': wholebody_kps_smoothed[idx]}, ]
        image = image * 0 + 255
        image = wholebody_model.visualize_pose_results(image, pose_output, vis_dot_radius=7, vis_line_thickness=3)
        cv2.imwrite('./__debug__/wholebody_kps_{}_{}.jpg'.format(args.video_idx, args.sequence_idx, frame_id), image)

        video.write(image)
    video.release()


if __name__ == '__main__':
    # envs: bigdet
    parser = argparse.ArgumentParser('ViTPose inference.')
    parser.add_argument('--root_dir', type=str, )
    parser.add_argument('--video_idx', type=int, )
    parser.add_argument('--sequence_idx', type=int, )

    args = parser.parse_args()

    visualize(args)
