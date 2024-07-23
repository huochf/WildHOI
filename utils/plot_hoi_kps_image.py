import os
import pickle
import cv2
from tqdm import tqdm
import argparse
import numpy as np


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
    line_thickness = 3
    thickness = 5
    lineType = 4

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

    line_thickness = 3
    thickness = 5
    lineType = 4

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
