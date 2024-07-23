import os
import json
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


def render_object():
    object_name = 'barbell'
    image_dir = '/storage/data/huochf/HOIYouTube-test/{}/images_temp/'.format(object_name)

    object_mesh = trimesh.load('../data/objects/{}.ply'.format(object_name), process=False)
    object_v = torch.tensor(np.array(object_mesh.vertices), dtype=torch.float32)
    object_f = torch.tensor(np.array(object_mesh.faces, dtype=np.int64))
    count = 0

    os.makedirs('./object_render/{}'.format(object_name), exist_ok=True)

    object_pose_dir = './object_pose/{}/'.format(object_name)

    for file in os.listdir(object_pose_dir):
        item_id = file.split('.')[0]

        video_id, frame_id, instance_id = item_id.split('_')
        # video_id, frame_id = item_id.split('_')
        object_pose = np.load(os.path.join(object_pose_dir, '{}.npz'.format(item_id)))

        image = cv2.imread(os.path.join(image_dir, video_id, '{}.jpg'.format(frame_id)))
        h, w, _ = image.shape

        cx, cy = object_pose['optical_center']

        t = torch.tensor(object_pose['translation'], dtype=torch.float32)
        R = torch.tensor(object_pose['rotmat'], dtype=torch.float32)
        if 'f' in object_pose:
            f = float(object_pose['f'])
        else:
            f = 1000
        if not np.isnan(R).any() and not np.isnan(t).any():
            image = render(image, object_v, object_f, f, cx, cy, t, R)
        cv2.imwrite('./object_render/{}/{}.jpg'.format(object_name, item_id), image.astype(np.uint8))
        print('saved image {}'.format(item_id))


if __name__ == '__main__':
    # envs: preprocess
    render_object()
