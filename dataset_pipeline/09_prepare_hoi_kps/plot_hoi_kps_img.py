import os
import sys
sys.path.append('/inspurfs/group/wangjingya/huochf/Thesis/')
import pickle
import cv2
from tqdm import tqdm
import argparse
import numpy as np
from hoi_recon.datasets.utils import generate_image_patch


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


KPS_COLORS = [
    [0.,    255.,  255.],
    [0.,   255.,    170.],
    [0., 170., 255.,],
    [85., 170., 255.],
    [0.,   255.,   85.], # 4
    [0., 85., 255.],
    [170., 85., 255.],
    [0.,   255.,   0.], # 7
    [0., 0., 255.], 
    [255., 0., 255.],
    [0.,    255.,  0.], # 10
    [0., 0., 255.],
    [255., 85., 170.],
    [170., 0, 255.],
    [255., 0., 170.],
    [255., 170., 85.],
    [85., 0., 255.],
    [255., 0., 85],
    [32., 0., 255.],
    [255., 0, 32],
    [0., 0., 255.],
    [255., 0., 0.],
]


OBJECT_BONE_IDX = {
'barbell': [[0, 1], [1, 2], [2, 3], ],
            # [1, 4], [1, 5], [1, 6], [1, 7], [4, 5], [5, 6], [6, 7], [7, 4], 
            # [2, 8], [2, 9], [2, 10], [2, 11], [8, 9], [9, 10], [10, 11], [11, 8]],
'cello': [[0, 1], [1, 2], [2, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 7], 
          [2, 8], [3, 9], [4, 10], [5, 11], [6, 12], [7, 13], 
          [8, 9], [8, 10], [9, 11], [10, 12], [11, 13], [12, 13]],
'violin': [[0, 1], [1, 2], [2, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 7], 
          [2, 8], [3, 9], [4, 10], [5, 11], [6, 12], [7, 13], 
          [8, 9], [8, 10], [9, 11], [10, 12], [11, 13], [12, 13]],
'baseball': [[0, 1]],
'tennis': [[0, 1], [1, 2], [1, 3], [2, 3], [2, 4], [4, 6], [3, 5], [5, 6]],
'basketball': [],
'yogaball': [],
'skateboard': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]],
'bicycle': [[0, 5], [1, 5], [2, 4], [3, 4], [4, 5], [4, 8], [5, 8], [8, 9], [6, 9], [7, 9]]
}


def plot_smpl_keypoints(image, keypoints):
    bone_idx = [[ 0,  1], [ 0,  2], [ 0,  3], 
                [ 1,  4], [ 2,  5], [ 3,  6], 
                [ 4,  7], [ 5,  8], [ 6,  9], 
                [ 7, 10], [ 8, 11], [ 9, 12], 
                [ 9, 13], [ 9, 14], [12, 15],
                [13, 16], [14, 17], [16, 18],
                [17, 19], [18, 20], [19, 21]]
    line_thickness = 6
    thickness = 10
    lineType = 8

    for bone in bone_idx:
        idx1, idx2 = bone
        x1, y1 = keypoints[idx1]
        x2, y2 = keypoints[idx2]
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), tuple(KPS_COLORS[idx1 % len(KPS_COLORS)]), line_thickness, lineType)

    for i, points in enumerate(keypoints):
        x, y = points
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), thickness, KPS_COLORS[i % len(KPS_COLORS)], thickness=-1, lineType=lineType)

    return image


def plot_object_keypoints(image, keypoints, object_name):

    line_thickness = 6
    thickness = 10
    lineType = 8

    for bone in OBJECT_BONE_IDX[object_name]:
        idx1, idx2 = bone
        x1, y1 = keypoints[idx1]
        x2, y2 = keypoints[idx2]
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), tuple(KPS_COLORS[idx1 % len(KPS_COLORS)]), line_thickness, lineType)

    for i, points in enumerate(keypoints):
        x, y = points
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), thickness, KPS_COLORS[i % len(KPS_COLORS)], thickness=-1, lineType=lineType)

    return image


def plot_sparse_keypoints(image, smpl_kps, object_kps, object_name):
    image = plot_smpl_keypoints(image, smpl_kps)
    image = plot_object_keypoints(image, object_kps, object_name)
    return image


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_hoi_kps_img(args):
    object_name = args.root_dir.split('/')[-1]

    for video_idx in range(0, 1000):
        args.video_idx = video_idx
        args.sequence_idx = 0
        hoi_kps_file = os.path.join(args.root_dir, 'hoi_kps', '{:04d}'.format(args.video_idx), '{:03d}_hoi_kps.pkl'.format(args.sequence_idx))
        try:
            hoi_kps = load_pickle(hoi_kps_file)
        except:
            continue

        osx_dir = os.path.join(args.root_dir, 'smplx_tuned', '{:04d}'.format(args.video_idx))
        osx_smpl_data = load_pickle(os.path.join(osx_dir, '{:03d}_smplx.pkl'.format(args.sequence_idx)))

        frame_ids = hoi_kps['frame_ids']
        smpl_kps_seq = hoi_kps['smpl_kps_seq']
        object_kps_seq = hoi_kps['object_kps_seq']

        if len(frame_ids) > 300:
            continue

        image = cv2.imread(os.path.join(args.root_dir, 'images_temp', '{:04d}'.format(args.video_idx), '{}.jpg'.format(frame_ids[0])))
        h, w, _ = image.shape
        video = cv2.VideoWriter('./__debug__/hoi_kps_{:04d}_{:03d}.mp4'.format(args.video_idx, args.sequence_idx), cv2.VideoWriter_fourcc(*'mp4v'), 30, (512, 512))
        for idx, frame_id in enumerate(tqdm(frame_ids)):

            crop_boxes = osx_smpl_data[idx]['bbox']
            cx, cy = crop_boxes[0] + crop_boxes[2] / 2, crop_boxes[1] + crop_boxes[3] / 2
            s = 1.2 * max(crop_boxes[2], crop_boxes[3])

            image = cv2.imread(os.path.join(args.root_dir, 'images_temp', '{:04d}'.format(args.video_idx), '{}.jpg'.format(frame_id)))
            image_kps = plot_sparse_keypoints(image, smpl_kps_seq[idx], object_kps_seq[idx], object_name)
            image_kps, _ = generate_image_patch(image_kps, cx, cy, s, 512, 0, [1., 1., 1.])
            video.write(image_kps.astype(np.uint8))
        video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare hoi kps')
    parser.add_argument('--root_dir', type=str, help='The dataset directory')
    parser.add_argument('--video_idx', type=int, )
    parser.add_argument('--sequence_idx', type=int, )
    args = parser.parse_args()

    plot_hoi_kps_img(args)
