import os
import json
import random
import argparse
import trimesh
from tqdm import tqdm
import numpy as np
import math
import torch
import torch.nn as nn
import pickle
import cv2
import neural_renderer as nr


def render(image, v, f, focal_length, cx, cy, t, R):
    device = torch.device('cuda')
    v = v @ R.transpose(1, 0) + t.reshape(1, 3)
    v = v.reshape(1, -1, 3).to(device)
    f = f.reshape(1, -1, 3).to(device)
    textures = torch.tensor([0.65098039, 0.74117647, 0.85882353], dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).repeat(1, f.shape[1], 1, 1, 1, 1).to(device)
    h, w, _ = image.shape

    fx = fy = focal_length
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).reshape(1, 3, 3).to(device)
    R = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).to(device)
    t = torch.zeros(3, dtype=torch.float32).reshape(1, 3).to(device)

    s = max(h, w)
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
    rend = rend[:h, :w]
    mask = mask[:h, :w]
    mask = mask * 0.7
    image = image * (1 - mask) + rend * mask

    return image


def load_keypoints(file):
    with open(file, 'r') as f:
        all_lines = f.readlines()
    if int(all_lines[0]) == 0:
        return np.empty((0, 3))
    keypoints = []
    for i in range(len(all_lines) - 1):
        line = all_lines[i + 1]
        x, y = line.split(' ')
        x, y = float(x), float(y)
        if x > 1 or y > 1 or x < 0 or y < 0:
            keypoints.append([0, 0, 0])
        else:
            keypoints.append([x, y, 1])
    return np.array(keypoints)


def load_keypoints_v2(file):
    with open(file, 'r') as f:
        data = json.load(f)
    data = data
    keypoints = []
    res = data['info']['height']
    for item in data['dataList']:
        x, y = item['coordinates']
        if x > res or y > res:
            keypoints.append([0, 0, 0])
        else:
            keypoints.append([x, y, 1])
    return np.array(keypoints) / res


def visualize(args):
    object_name = args.object
    root_dir = args.root_dir

    image_dir = os.path.join(root_dir, object_name, 'images_temp')

    if object_name in ['cello', 'violin']:
        object_mesh = trimesh.load('../../data/objects/{}_body.ply'.format(object_name))
    else:
        object_mesh = trimesh.load('../../data/objects/{}.ply'.format(object_name))
    object_v = torch.tensor(np.array(object_mesh.vertices), dtype=torch.float32)
    object_f = torch.tensor(np.array(object_mesh.faces, dtype=np.int64))

    output_dir = './__debug__/{}'.format(object_name)
    os.makedirs(output_dir, exist_ok=True)

    pose_dir = os.path.join(args.annotation_dir, object_name, 'object_annotations', 'pose')
    pose_files = os.listdir(pose_dir)
    random.shuffle(pose_files)

    corr_dir = os.path.join(args.annotation_dir, object_name, 'object_annotations', 'corr')
    kps_dir = os.path.join(args.annotation_dir, object_name, 'object_annotations', 'keypoints')

    for file in tqdm(pose_files[:32]):
        image_id = file.split('.')[0]
        video_id, frame_id, hoi_id = image_id.split('_')

        object_pose = np.load(os.path.join(pose_dir, file))
        image_org = cv2.imread(os.path.join(image_dir, video_id, '{}.jpg'.format(frame_id)))
        box = np.loadtxt(os.path.join(corr_dir, '{}-box.txt'.format(image_id)))
        box_x, box_y, box_w, box_h = box
        cx, cy = box_x + box_w / 2, box_y + box_h / 2
        s = max(1, max(box_w, box_h) * 0.75)
        box = np.array([cx - s / 2, cy - s / 2, 
            cx + s / 2, cy + s / 2])

        kps_file = os.path.join(kps_dir, '{}.txt'.format(image_id))
        if not os.path.exists(kps_file):
            kps_file = os.path.join(kps_dir, '{}.json'.format(image_id))
        try:
            keypoints = load_keypoints_v2(kps_file)
        except:
            keypoints = load_keypoints(kps_file)

        tracking_file = os.path.join(root_dir, object_name, 'hoi_tracking', '{}_tracking.pkl'.format(video_id))
        with open(tracking_file, 'rb') as f:
            tracking_results = pickle.load(f)
        object_box = None
        for hoi_instance in tracking_results['hoi_instances']:
            if hoi_id != hoi_instance['hoi_id']:
                continue
            for item in hoi_instance['sequences']:
                if frame_id == item['frame_id']:
                    object_bbox = item['object_bbox']
        assert object_bbox is not None
        x1, y1, x2, y2 = object_bbox[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        s = int(max((y2 - y1), (x2 - x1)) * 1.0)
        # s = int(max((y2 - y1), (x2 - x1)) * 1.5) # baseball, surfboard, golf
        _x1 = int(cx - s)
        _y1 = int(cy - s)
        keypoints[:, 0] = keypoints[:, 0] * 2 * s + _x1
        keypoints[:, 1] = keypoints[:, 1] * 2 * s + _y1

        image_kps = plot_kps(image_org.copy(), keypoints)

        cx, cy = object_pose['optical_center']
        t = torch.tensor(object_pose['translation'], dtype=torch.float32)
        R = torch.tensor(object_pose['rotmat'], dtype=torch.float32)
        if 'f' in object_pose:
            f = float(object_pose['f'])
        else:
            f = 1000
        image_pose = render(image_org.copy(), object_v, object_f, f, cx, cy, t, R)

        image_kps_crop = crop_images(image_kps, box)
        image_pose_crop = crop_images(image_pose, box)
        image_vis = np.concatenate([image_kps_crop, image_pose_crop], axis=1)

        image_vis[image_vis.sum(-1) < 3] = 255

        cv2.imwrite(os.path.join(output_dir, '{}.jpg'.format(image_id)), image_vis.astype(np.uint8))


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


def plot_kps(image, kps):
    line_thickness = 2
    thickness = 4
    lineType = 8
    h, w, c = image.shape

    for i, point in enumerate(kps):
        x, y, v = point
        if v == 0:
            continue
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), thickness, KPS_COLORS[i % len(KPS_COLORS)], thickness=-1, lineType=lineType)

    return image


def crop_images(image, object_bbox):
    h, w, _ = image.shape
    x1, y1, x2, y2 = object_bbox[:4]

    cx, cy = (x1 + x2) / 2, (y1  + y2) / 2
    s = int(max((y2 - y1), (x2 - x1)) * 1.0)

    crop_image = np.zeros((2 * s, 2 * s, 3))

    _x1 = int(cx - s)
    _y1 = int(cy - s)

    if _x1 < 0 and _y1 < 0:
        crop_image[-_y1 : min(h - _y1, 2 * s), -_x1 : min(w - _x1, 2 * s)] = image[0:_y1 + 2 * s, 0:_x1 + 2 * s]
    elif _x1 < 0:
        crop_image[:min(h - _y1, 2 * s), -_x1 : min(w - _x1, 2 * s)] = image[_y1:_y1 + 2 * s, 0:_x1 + 2 * s]
    elif _y1 < 0:
        crop_image[-_y1 : min(h - _y1, 2 * s), :min(w - _x1, 2 * s)] = image[0: _y1 + 2 * s, _x1:_x1 + 2 * s]
    else:
        crop_image[:min(h - _y1, 2 * s), :min(w - _x1, 2 * s)] = image[_y1:_y1 + 2 * s, _x1:_x1 + 2 * s]

    crop_image = cv2.resize(crop_image, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)

    return crop_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize 6D Pose Annotations of the Object.')
    parser.add_argument('--root_dir', type=str, help="The dataset directory")
    parser.add_argument('--annotation_dir', type=str, )
    parser.add_argument('--object', type=str, help="The object name")
    args = parser.parse_args()

    visualize(args)
