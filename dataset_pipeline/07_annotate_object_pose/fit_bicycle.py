import os
import pickle
import trimesh
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import cv2
import json
import math
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
# import neural_renderer as nr
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix


def compute_random_rotations(B=32):
    x1, x2, x3 = torch.split(torch.rand(3 * B).cuda(), B)
    tau = 2 * math.pi
    R = torch.stack(
        (
            torch.stack(
                (torch.cos(tau * x1), torch.sin(tau * x1), torch.zeros_like(x1)), 1
            ),
            torch.stack(
                (-torch.sin(tau * x1), torch.cos(tau * x1), torch.zeros_like(x1)), 1
            ),
            torch.stack(
                (torch.zeros_like(x1), torch.zeros_like(x1), torch.ones_like(x1)), 1
            ),
        ),
        1,
    )
    v = torch.stack(
        (  # B x 3
            torch.cos(tau * x2) * torch.sqrt(x3),
            torch.sin(tau * x2) * torch.sqrt(x3),
            torch.sqrt(1 - x3),
        ),
        1,
    )
    identity = torch.eye(3).repeat(B, 1, 1).cuda()
    H = identity - 2 * v.unsqueeze(2) * v.unsqueeze(1)
    rotation_matrices = -torch.matmul(H, R)

    return rotation_matrices


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data



class Bicycle(nn.Module):

    def __init__(self, n_init):
        super().__init__()

        bicycle_front = trimesh.load('../data/objects/bicycle_front.ply', process=False)
        bicycle_back = trimesh.load('../data/objects/bicycle_back.ply', process=False)

        bicycle_front = torch.from_numpy(bicycle_front.vertices).float().unsqueeze(0)
        bicycle_back = torch.from_numpy(bicycle_back.vertices).float().unsqueeze(0)
        self.register_buffer('bicycle_front', bicycle_front)
        self.register_buffer('bicycle_back', bicycle_back)

        with open('../data/objects/bicycle_front_keypoints.json', 'r') as f:
            self.bicycle_front_kps_indices = json.load(f)
        with open('../data/objects/bicycle_back_keypoints.json', 'r') as f:
            self.bicycle_back_kps_indices = json.load(f)

        rot_axis_begin = bicycle_front[0, self.bicycle_front_kps_indices['5']].mean(0)
        rot_axis_end = bicycle_front[0, self.bicycle_front_kps_indices['6']].mean(0)
        rot_axis = rot_axis_end - rot_axis_begin
        rot_axis = rot_axis / torch.sqrt((rot_axis ** 2).sum())
        self.register_buffer('rot_axis_begin', rot_axis_begin)
        self.register_buffer('rot_axis', rot_axis)

        rot_angle = torch.rand(n_init) * 2 * np.pi - np.pi
        self.rot_angle = nn.Parameter(rot_angle.float())

        R_init = compute_random_rotations(n_init)
        R6d = matrix_to_rotation_6d(R_init).to(torch.float32)
        R6d = R6d.reshape(n_init, 6)
        self.R6d = nn.Parameter(R6d)

        x = torch.rand(n_init) * 2 - 1
        y = torch.rand(n_init) * 2 - 1
        z = torch.rand(n_init) * 10 + 10
        T = torch.stack([x, y, z], dim=1).to(torch.float32)
        self.T = nn.Parameter(T)


    def forward(self, ):

        front_rotmat = axis_angle_to_matrix(self.rot_axis.reshape(1, 3) * self.rot_angle.reshape(-1, 1))
        front_v = self.bicycle_front - self.rot_axis_begin.view(1, 1, 3)
        front_v = front_v @ front_rotmat.transpose(2, 1)
        front_v = front_v + self.rot_axis_begin.view(1, 1, 3)

        global_rotmat = rotation_6d_to_matrix(self.R6d) # [b, 3, 3]
        global_trans = self.T.unsqueeze(1) # [b, 1, 3]
        front_v = front_v @ global_rotmat.transpose(2, 1) + global_trans
        back_v = self.bicycle_back @ global_rotmat.transpose(2, 1) + global_trans

        keypoints = []
        for front_idx in ['3', '4', '5', '6']:
            indices = self.bicycle_front_kps_indices[front_idx]
            keypoints.append(front_v[:, indices].mean(1))
        for back_idx in ['3', '4']:
            indices = self.bicycle_back_kps_indices[back_idx]
            keypoints.append(back_v[:, indices].mean(1))
        keypoints = torch.stack(keypoints, dim=1) # [b, n, 3]

        edge_points = []
        for front_idx in ['1', '2']:
            indices = self.bicycle_front_kps_indices[front_idx]
            edge_points.append(front_v[:, indices])
        for back_idx in ['1', '2']:
            indices = self.bicycle_back_kps_indices[back_idx]
            edge_points.append(back_v[:, indices])

        return keypoints, edge_points


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
    _file = file.split('/')[-1]
    image = cv2.imread('./images/bicycle/{}'.format(_file.replace('json', 'jpg')))
    # image = cv2.imread('./images/skateboard/{}'.format(_file.replace('txt', 'jpg')))

    for item in data['dataList']:
        x, y = item['coordinates']
        if x >= res or y >= res or x < 0 or y < 0:
            keypoints.append([0, 0, 0])
        else:
            is_black = image[int(y), int(x)].sum() == 0
            if is_black:
                keypoints.append([0, 0, 0])
            else:
                keypoints.append([x, y, res])
    return np.array(keypoints) / res


class KeypointDataset():

    def __init__(self, keypoint_dir):
        self.root_dir = '/storage/data/huochf/HOIYouTube/bicycle'
        self.keypoint_all = []
        for file in os.listdir(keypoint_dir):
            # keypoints = load_keypoints_v2(os.path.join(keypoint_dir, file))
            keypoints = load_keypoints(os.path.join(keypoint_dir, file))
            if keypoints.shape[0] != 22:
                continue
            if np.isnan(keypoints).any():
                continue
            self.keypoint_all.append((file.split('.')[0], keypoints))


    def __len__(self):
        return len(self.keypoint_all)


    def __getitem__(self, idx):
        video_frame_id, keypoints = self.keypoint_all[idx]
        video_id, frame_id, instance_idx = video_frame_id.split('_')

        tracking_results = load_pickle(os.path.join(self.root_dir, 'hoi_tracking', '{}_tracking.pkl'.format(video_id)))
        object_bbox = None
        for hoi_instance in tracking_results['hoi_instances']:
            if instance_idx != hoi_instance['hoi_id']:
                continue
            for item in hoi_instance['sequences']:
                if frame_id == item['frame_id']:
                    object_bbox = item['object_bbox']
        assert object_bbox is not None

        x1, y1, x2, y2 = object_bbox[:4]

        cx, cy = (x1 + x2) / 2, (y1  + y2) / 2
        s = int(max((y2 - y1), (x2 - x1)) * 1.0)

        _x1 = int(cx - s)
        _y1 = int(cy - s)

        x = keypoints[:, 0] * 2 * s - s
        y = keypoints[:, 1] * 2 * s - s
        keypoints = np.stack([x, y, keypoints[:, 2]], axis=1)

        edge_keypoints = keypoints[6:]
        keypoints = keypoints[:6]

        return video_frame_id, keypoints.astype(np.float32), edge_keypoints.astype(np.float32), np.array([cx, cy]).astype(np.float32)


def project(keypoints, focal):
    u = keypoints[..., 0] / (keypoints[..., 2] + 1e-8) * focal.unsqueeze(2).exp()
    v = keypoints[..., 1] / (keypoints[..., 2] + 1e-8) * focal.unsqueeze(2).exp()
    return torch.stack([u, v], dim=-1)


def fit_pose(keypoints, edge_keypoints, optical_centers, focal=1000, device=torch.device('cuda')):
    b = keypoints.shape[0]
    n_init = 128
    keypoints = keypoints.unsqueeze(1).repeat(1, n_init, 1, 1)
    edge_keypoints = edge_keypoints.unsqueeze(1).repeat(1, n_init, 1, 1)

    f = torch.log(torch.ones(b, n_init) * focal).float().cuda()
    f = nn.Parameter(f)
    bicycle = Bicycle(b * n_init).to(device)
    optimizer = torch.optim.Adam([bicycle.rot_angle, bicycle.R6d, bicycle.T, f], 0.05, betas=(0.9, 0.999))
    
    weight_f = lambda cst, it: 1. * cst / (1 + it)
    iteration = 3
    steps_per_iter = 1000
    for it in range(iteration):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Fitting object.')
        for i in loop:
            optimizer.zero_grad()
            keypoints_3d, edge_keypoints_3d = bicycle.forward()
            keypoints_3d = keypoints_3d.reshape(b, n_init, -1, 3)
            edge_keypoints_3d = [k.reshape(b, n_init, -1, 3) for k in edge_keypoints_3d]
            keypoint2_2d = project(keypoints_3d, f)
            edge_keypoints_2d = [project(k, f) for k in edge_keypoints_3d]

            loss_kps = (keypoint2_2d - keypoints[..., :2]) ** 2
            loss_kps = loss_kps * keypoints[..., 2:]
            loss_kps = loss_kps.reshape(b, n_init, -1).mean(-1)

            loss_edge = 0
            for k in range(4):
                _l = ((edge_keypoints_2d[k].unsqueeze(2) - edge_keypoints[:, :, k * 4:(k + 1) * 4].unsqueeze(3)[..., :2]) ** 2).sum(-1)
                _l = _l * edge_keypoints[:, :, k * 4:(k + 1) * 4].unsqueeze(3)[..., 2]
                _l = _l.min(3)[0].mean(-1)
                loss_edge = loss_edge + _l

            loss_angle = (bicycle.rot_angle.reshape(b, n_init) ** 2)

            loss_batch = loss_kps + 0.4 * loss_edge + 1e-3 * loss_angle
            loss = weight_f(loss_batch.mean(), it)
            loss.backward()
            optimizer.step()

            indices = torch.argmin(loss_batch, dim=1)
            min_loss = weight_f(loss_batch[range(b), indices].mean(), it)

            loop.set_description('Iter: {}, Loss: {:0.4f}, min_loss: {:0.4f}, loss_angle: {:0.4f}'.format(i, loss.item(), min_loss.item(), loss_angle.mean().item()))

    indices = torch.argmin(loss_batch, dim=1)
    rot_angle = bicycle.rot_angle.reshape(b, n_init)[range(b), indices].detach().cpu().numpy()
    R = rotation_6d_to_matrix(bicycle.R6d.reshape(b, n_init, -1)[range(b), indices]).detach().cpu().numpy()
    T = bicycle.T.reshape(b, n_init, 3)[range(b), indices].detach().cpu().numpy()
    f = f[range(b), indices].exp().detach().cpu().numpy()
    return rot_angle, R, T, f


def fit_bicycle():
    device = torch.device('cuda')

    object_name = 'bicycle'
    image_dir = '/storage/data/huochf/HOIYouTube/{}/images_temp'.format(object_name)
    keypoint_dir = './obj_keypoints/{}'.format(object_name)

    dataset = KeypointDataset(keypoint_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, drop_last=False)

    save_dir = './object_pose/{}'.format(object_name)
    os.makedirs(save_dir, exist_ok=True)

    for item in tqdm(dataloader):
        item_ids, keypoints, edge_keypoints, optical_centers = item
        keypoints = keypoints.to(device)
        edge_keypoints = edge_keypoints.to(device)
        optical_centers = optical_centers.to(device)

        rot_angle, R, T, f = fit_pose(keypoints, edge_keypoints, optical_centers, focal=1000, device=device)

        for batch_idx, item_id in enumerate(item_ids):
            angle = rot_angle[batch_idx]
            rotmat = R[batch_idx]
            translation = T[batch_idx]
            np.savez(os.path.join(save_dir, '{}.npz'.format(item_id)), rot_angle=angle, rotmat=rotmat, translation=translation, f=f[batch_idx], optical_center=optical_centers[batch_idx].detach().cpu().numpy())


if __name__ == '__main__':
    fit_bicycle()
