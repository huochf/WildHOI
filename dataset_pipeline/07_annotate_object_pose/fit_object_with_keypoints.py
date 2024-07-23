import os
import pickle
import trimesh
import numpy as np
import torch
import torch.nn as nn
import cv2
import json
import math
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
# import neural_renderer as nr
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, matrix_to_axis_angle


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
    image = cv2.imread('./images/surfboard/{}'.format(_file.replace('json', 'jpg')))
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
        self.root_dir = '/storage/data/huochf/HOIYouTube-test/barbell'
        self.keypoint_all = []
        for file in os.listdir(keypoint_dir):
            # try:
            # keypoints = load_keypoints_v2(os.path.join(keypoint_dir, file))
            # except:
            #     continue
            keypoints = load_keypoints(os.path.join(keypoint_dir, file))
            if keypoints.shape[0] != 14:
                continue
            if np.isnan(keypoints).any():
                continue

            self.keypoint_all.append((file.split('.')[0], keypoints))

        self.keypoint_all_revised = []
        for item in tqdm(self.keypoint_all, desc='loading data'):
            video_frame_id, keypoints = item

            instance_idx = -1
            if len(item[0].split('_')) == 3:
                video_id, frame_id, instance_idx = video_frame_id.split('_')
            else:
                video_id, frame_id = video_frame_id.split('_')

            tracking_results = load_pickle(os.path.join(self.root_dir, 'hoi_tracking', '{}_tracking.pkl'.format(video_id)))
            object_bbox = None
            for hoi_instance in tracking_results['hoi_instances']:
                if instance_idx != -1 and instance_idx != hoi_instance['hoi_id']:
                    continue
                for item in hoi_instance['sequences']:
                    if frame_id == item['frame_id']:
                        object_bbox = item['object_bbox']
                        if object_bbox is not None:
                            self.keypoint_all_revised.append(('{}_{}_{}'.format(video_id, frame_id, hoi_instance['hoi_id']), keypoints, object_bbox))
        print(len(self.keypoint_all_revised), len(self.keypoint_all))


    def __len__(self):
        return len(self.keypoint_all_revised)


    def __getitem__(self, idx):
        video_frame_id, keypoints, object_bbox = self.keypoint_all_revised[idx]
        video_id, frame_id, instance_idx = video_frame_id.split('_')

        x1, y1, x2, y2 = object_bbox[:4]

        cx, cy = (x1 + x2) / 2, (y1  + y2) / 2
        s = int(max((y2 - y1), (x2 - x1)) * 1.0)
        # s = int(max((y2 - y1), (x2 - x1)) * 1.5) # baseball, surfboard

        _x1 = int(cx - s)
        _y1 = int(cy - s)

        x = keypoints[:, 0] * 2 * s - s
        y = keypoints[:, 1] * 2 * s - s
        keypoints = np.stack([x, y, keypoints[:, 2]], axis=1)

        edge_keypoints = keypoints[4:]

        keypoints = keypoints[:4]

        return video_frame_id, keypoints.astype(np.float32), edge_keypoints.astype(np.float32), np.array([cx, cy]).astype(np.float32)


def get_object_keypoints(object_v, focal, R6d, T):
    b, n_init, _ = R6d.shape
    keypoints = object_v.unsqueeze(0).repeat(b * n_init, 1, 1).reshape(b, n_init, -1, 3)
    R = rotation_6d_to_matrix(R6d)
    keypoints = keypoints @ R.transpose(3, 2) + T.unsqueeze(2)

    u = keypoints[:, :, :, 0] / (keypoints[:, :, :, 2] + 1e-8) * focal.unsqueeze(2).exp()
    v = keypoints[:, :, :, 1] / (keypoints[:, :, :, 2] + 1e-8) * focal.unsqueeze(2).exp()
    points = torch.stack([u, v], dim=3)
    return points


def fit_pose(object_v, keypoint_indices, keypoints, edge_keypoints, focal_init=1000):
    b = keypoints.shape[0]
    n_init = 128
    keypoints = keypoints.unsqueeze(1).repeat(1, n_init, 1, 1)
    edge_keypoints = edge_keypoints.unsqueeze(1).repeat(1, n_init, 1, 1)
    R_init = compute_random_rotations(b * n_init)
    # R_init = torch.tensor([
    #     [1, 0, 0],
    #     [0, 0, -1],
    #     [0, 1, 0]
    # ]) # basketball prior pose
    # R_init = R_init.unsqueeze(0).repeat(b * n_init, 1, 1)
    R6d = matrix_to_rotation_6d(R_init).to(torch.float32).cuda()
    R6d = R6d.reshape(b, n_init, 6)
    R6d = nn.Parameter(R6d)

    # T = torch.tensor([0, 0, 5], dtype=torch.float32).cuda()
    # T = T.unsqueeze(0).repeat(b * n_init, 1).reshape(b, n_init, 3)
    x = torch.rand(n_init) * 2 - 1
    y = torch.rand(n_init) * 2 - 1
    # z = torch.rand(n_init) * 10 + 2
    z = torch.rand(n_init) * 10 + 10 # barbell
    T = torch.stack([x, y, z], dim=1)
    T = T.unsqueeze(0).repeat(b, 1, 1).cuda()
    T = nn.Parameter(T)

    f = torch.log(torch.ones(b, n_init) * focal_init).float().cuda()
    f = nn.Parameter(f)

    weight_f = lambda cst, it: 1. * cst / (1 + it)
    optimizer = torch.optim.Adam([R6d, T, f], 0.05, betas=(0.9, 0.999))
    # optimizer = torch.optim.Adam([T, f], 0.05, betas=(0.9, 0.999))
    iteration = 3
    steps_per_iter = 1000
    for it in range(iteration):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Fitting object.')
        for i in loop:
            optimizer.zero_grad()
            object_keypoints = get_object_keypoints(object_v, f, R6d, T)
            # object_keypoints: [b, n_init, n, 2]
            reproj_points = []
            for point_i in ['1', '2', '3', '4']:
                reproj_points.append(object_keypoints[:, :, keypoint_indices[point_i]].mean(2))
            reproj_points = torch.stack(reproj_points, dim=2)

            loss_kps = (reproj_points - keypoints[:, :, :, :2]) ** 2
            loss_kps = loss_kps * keypoints[:, :, :, 2:]
            loss_kps = loss_kps.reshape(b, n_init, -1).mean(-1)

            # loss_kps = 0

            edge_reproj_points = object_keypoints[:, :, keypoint_indices['5']]
            # edge_reproj_points: [b, n_init, n, 2], edge_keypoints: [b, n_init, m, 2]
            loss_edge = ((edge_reproj_points.unsqueeze(2) - edge_keypoints[:, :, :5].unsqueeze(3)[..., :2]) ** 2).sum(-1) # [b, n_init, m, n]
            loss_edge = loss_edge * edge_keypoints[:, :, :5].unsqueeze(3)[..., 2]
            loss_edge1 = loss_edge.min(3)[0].mean(-1)

            edge_reproj_points = object_keypoints[:, :, keypoint_indices['6']]
            # edge_reproj_points: [b, n_init, n, 2], edge_keypoints: [b, n_init, m, 2]
            loss_edge = ((edge_reproj_points.unsqueeze(2) - edge_keypoints[:, :, 5:10].unsqueeze(3)[..., :2]) ** 2).sum(-1) # [b, n_init, m, n]
            loss_edge = loss_edge * edge_keypoints[:, :, 5:10].unsqueeze(3)[..., 2]
            loss_edge2 = loss_edge.min(3)[0].mean(-1)

            # edge_reproj_points = object_keypoints[:, :, keypoint_indices['7']]
            # # edge_reproj_points: [b, n_init, n, 2], edge_keypoints: [b, n_init, m, 2]
            # loss_edge = ((edge_reproj_points.unsqueeze(2) - edge_keypoints[:, :, 6:9].unsqueeze(3)[..., :2]) ** 2).sum(-1) # [b, n_init, m, n]
            # loss_edge = loss_edge * edge_keypoints[:, :, 6:9].unsqueeze(3)[..., 2]
            # loss_edge3 = loss_edge.min(3)[0].mean(-1)

            # edge_reproj_points = object_keypoints[:, :, keypoint_indices['8']]
            # # edge_reproj_points: [b, n_init, n, 2], edge_keypoints: [b, n_init, m, 2]
            # loss_edge = ((edge_reproj_points.unsqueeze(2) - edge_keypoints[:, :, 9:12].unsqueeze(3)[..., :2]) ** 2).sum(-1) # [b, n_init, m, n]
            # loss_edge = loss_edge * edge_keypoints[:, :, 9:12].unsqueeze(3)[..., 2]
            # loss_edge4 = loss_edge.min(3)[0].mean(-1)

            # edge_reproj_points = object_keypoints[:, :, keypoint_indices['11']]
            # # edge_reproj_points: [b, n_init, n, 2], edge_keypoints: [b, n_init, m, 2]
            # loss_edge = ((edge_reproj_points.unsqueeze(2) - edge_keypoints[:, :, 12:15].unsqueeze(3)[..., :2]) ** 2).sum(-1) # [b, n_init, m, n]
            # loss_edge = loss_edge * edge_keypoints[:, :, 12:15].unsqueeze(3)[..., 2]
            # loss_edge5 = loss_edge.min(3)[0].mean(-1)

            # edge_reproj_points = object_keypoints[:, :, keypoint_indices['12']]
            # # edge_reproj_points: [b, n_init, n, 2], edge_keypoints: [b, n_init, m, 2]
            # loss_edge = ((edge_reproj_points.unsqueeze(2) - edge_keypoints[:, :, 15:18].unsqueeze(3)[..., :2]) ** 2).sum(-1) # [b, n_init, m, n]
            # loss_edge = loss_edge * edge_keypoints[:, :, 15:18].unsqueeze(3)[..., 2]
            # loss_edge6 = loss_edge.min(3)[0].mean(-1)

            # edge_reproj_points = object_keypoints[:, :, keypoint_indices['13']]
            # # edge_reproj_points: [b, n_init, n, 2], edge_keypoints: [b, n_init, m, 2]
            # loss_edge = ((edge_reproj_points.unsqueeze(2) - edge_keypoints[:, :, 18:21].unsqueeze(3)[..., :2]) ** 2).sum(-1) # [b, n_init, m, n]
            # loss_edge = loss_edge * edge_keypoints[:, :, 18:21].unsqueeze(3)[..., 2]
            # loss_edge7 = loss_edge.min(3)[0].mean(-1)

            # edge_reproj_points = object_keypoints[:, :, keypoint_indices['14']]
            # # edge_reproj_points: [b, n_init, n, 2], edge_keypoints: [b, n_init, m, 2]
            # loss_edge = ((edge_reproj_points.unsqueeze(2) - edge_keypoints[:, :, 21:24].unsqueeze(3)[..., :2]) ** 2).sum(-1) # [b, n_init, m, n]
            # loss_edge = loss_edge * edge_keypoints[:, :, 21:24].unsqueeze(3)[..., 2]
            # loss_edge8 = loss_edge.min(3)[0].mean(-1)

            # edge_reproj_points = object_keypoints[:, :, keypoint_indices['15']]
            # # edge_reproj_points: [b, n_init, n, 2], edge_keypoints: [b, n_init, m, 2]
            # loss_edge = ((edge_reproj_points.unsqueeze(2) - edge_keypoints[:, :, 24:27].unsqueeze(3)[..., :2]) ** 2).sum(-1) # [b, n_init, m, n]
            # loss_edge = loss_edge * edge_keypoints[:, :, 24:27].unsqueeze(3)[..., 2]
            # loss_edge9 = loss_edge.min(3)[0].mean(-1)

            # edge_reproj_points = object_keypoints[:, :, keypoint_indices['16']]
            # # edge_reproj_points: [b, n_init, n, 2], edge_keypoints: [b, n_init, m, 2]
            # loss_edge = ((edge_reproj_points.unsqueeze(2) - edge_keypoints[:, :, 27:30].unsqueeze(3)[..., :2]) ** 2).sum(-1) # [b, n_init, m, n]
            # loss_edge = loss_edge * edge_keypoints[:, :, 27:30].unsqueeze(3)[..., 2]
            # loss_edge10 = loss_edge.min(3)[0].mean(-1)

            # edge_reproj_points = object_keypoints[:, :, keypoint_indices['16']]
            # # edge_reproj_points: [b, n_init, n, 2], edge_keypoints: [b, n_init, m, 2]
            # loss_edge = ((edge_reproj_points.unsqueeze(2) - edge_keypoints[:, :, 50:55].unsqueeze(3)[..., :2]) ** 2).sum(-1) # [b, n_init, m, n]
            # loss_edge = loss_edge * edge_keypoints[:, :, 50:55].unsqueeze(3)[..., 2]
            # loss_edge11 = loss_edge.min(3)[0].mean(-1)

            # edge_reproj_points = object_keypoints[:, :, keypoint_indices['17']]
            # # edge_reproj_points: [b, n_init, n, 2], edge_keypoints: [b, n_init, m, 2]
            # loss_edge = ((edge_reproj_points.unsqueeze(2) - edge_keypoints[:, :, 55:60].unsqueeze(3)[..., :2]) ** 2).sum(-1) # [b, n_init, m, n]
            # loss_edge = loss_edge * edge_keypoints[:, :, 55:60].unsqueeze(3)[..., 2]
            # loss_edge12 = loss_edge.min(3)[0].mean(-1)

            loss_edge = loss_edge1 + loss_edge2 # + loss_edge3 + loss_edge4 # + loss_edge5 + loss_edge6 + loss_edge7 + loss_edge8 + loss_edge9 + loss_edge10 # + loss_edge11 + loss_edge12

            loss_pose = (matrix_to_axis_angle(rotation_6d_to_matrix(R6d))[:, :, 0] ** 2) # for barbell

            loss_batch = loss_kps + loss_edge + 0.1 * ((f - 6) ** 2).mean() + 1000 * loss_pose
            loss = weight_f(loss_batch.mean(), it)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([R6d, T, f], 0.1)
            optimizer.step()

            indices = torch.argmin(loss_batch, dim=1)
            min_loss = weight_f(loss_batch[range(b), indices].mean(), it)

            loop.set_description('Iter: {}, Loss: {:0.4f}, min_loss: {:0.4f}'.format(i, loss.item(), min_loss.item()))

    indices = torch.argmin(loss_batch, dim=1)
    R = rotation_6d_to_matrix(R6d[range(b), indices]).detach().cpu().numpy()
    T = T[range(b), indices].detach().cpu().numpy()
    f = f[range(b), indices].exp().detach().cpu().numpy()
    return R, T, f


def fit_object():
    device = torch.device('cuda')

    object_mesh = trimesh.load('../data/objects/barbell.ply', process=False)
    object_v = np.array(object_mesh.vertices)
    object_v = torch.from_numpy(object_v).float().to(device)

    object_name = 'barbell'
    image_dir = '/storage/data/huochf/HOIYouTube-test/{}/images_temp'.format(object_name)
    keypoint_dir = './obj_keypoints/{}'.format(object_name)

    dataset = KeypointDataset(keypoint_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, drop_last=False)

    save_dir = './object_pose/{}'.format(object_name)
    os.makedirs(save_dir, exist_ok=True)

    with open('../data/objects/{}_keypoints.json'.format(object_name), 'r') as f:
        keypoint_indices = json.load(f)

    for item in tqdm(dataloader):
        item_ids, keypoints, edge_keypoints, optical_centers = item
        keypoints = keypoints.to(device)
        edge_keypoints = edge_keypoints.to(device)
        optical_centers = optical_centers.to(device)

        R, T, f = fit_pose(object_v, keypoint_indices, keypoints, edge_keypoints, focal_init=1000)

        for batch_idx, item_id in enumerate(item_ids):
            rotmat = R[batch_idx]
            translation = T[batch_idx]
            np.savez(os.path.join(save_dir, '{}.npz'.format(item_id)), rotmat=rotmat, translation=translation, 
                f=f[batch_idx], optical_center=optical_centers[batch_idx].detach().cpu().numpy())

        # exit(0)


if __name__ == '__main__':
    # envs: pytorch3d
    fit_object()
