import os
import pickle
import trimesh
import random
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
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix



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


class KeypointDataset():

    def __init__(self, root_dir, video_id, hoi_id, hoi_instance):
        self.root_dir = root_dir
        self.video_id = video_id
        self.hoi_id = hoi_id
        self.hoi_instance = hoi_instance
        self.frame_ids = [item['frame_id'] for item in hoi_instance['sequences']][::10]


    def __len__(self):
        return len(self.frame_ids)


    def __getitem__(self, idx):
        item = self.hoi_instance['sequences'][idx]
        object_bbox = item['object_bbox']

        if object_bbox is not None:
            x1, y1, x2, y2 = object_bbox[:4]
            cx, cy = (x1 + x2) / 2, (y1  + y2) / 2
            optical_centers = np.array([cx, cy], dtype=np.float32)
        else:
            optical_centers = np.zeros(2, dtype=np.float32)
        frame_id = self.frame_ids[idx]
        masks = load_pickle(os.path.join(self.root_dir, 'hoi_mask', self.video_id, self.hoi_id, '{}.pkl'.format(frame_id)))

        edge_points_num = 256
        edge_points = np.zeros((edge_points_num, 3), dtype=np.float32)
        if masks['object']['mask'] is not None:
            image = cv2.imread(os.path.join(self.root_dir, 'images_temp', self.video_id, '{}.jpg'.format(frame_id)))
            h, w, _ = image.shape
            object_mask = np.zeros((h, w))
            mask_h, mask_w = masks['object']['mask_shape']
            x1, y1, x2, y2 = masks['object']['mask_box']
            object_mask[y1:y2+1, x1:x2+1] = np.unpackbits(masks['object']['mask'])[:mask_h * mask_w].reshape(mask_h, mask_w)

            object_mask = object_mask.astype(np.uint8)
            contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            best_contours = contours[-1].reshape((-1, 2))

            indices = np.arange(len(best_contours))
            random.shuffle(indices)
            best_contours = best_contours[indices[:edge_points_num]]
            edge_points[:len(best_contours), :2] = best_contours
            edge_points[:len(best_contours), 2] = 1
        edge_points[:, :2] = edge_points[:, :2] - optical_centers.reshape(1, 2)

        return frame_id, edge_points, optical_centers


def get_object_keypoints(object_v, focal, R6d, T):
    b, n_init, _ = R6d.shape
    keypoints = object_v.unsqueeze(0).repeat(b * n_init, 1, 1).reshape(b, n_init, -1, 3)
    R = rotation_6d_to_matrix(R6d)
    keypoints = keypoints @ R.transpose(3, 2) + T.unsqueeze(2)

    u = keypoints[:, :, :, 0] / (keypoints[:, :, :, 2] + 1e-8) * focal
    v = keypoints[:, :, :, 1] / (keypoints[:, :, :, 2] + 1e-8) * focal
    points = torch.stack([u, v], dim=3)
    return points


def fit_pose(object_v, keypoint_indices, edge_keypoints, focal):
    b = edge_keypoints.shape[0]
    n_init = 128
    edge_keypoints = edge_keypoints.unsqueeze(1).repeat(1, n_init, 1, 1)
    # R_init = compute_random_rotations(b * n_init)
    R_init = torch.tensor([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ]) # basketball prior pose
    R_init = R_init.unsqueeze(0).repeat(b * n_init, 1, 1)
    R6d = matrix_to_rotation_6d(R_init).to(torch.float32).cuda()
    R6d = R6d.reshape(b, n_init, 6)
    R6d = nn.Parameter(R6d)

    x = torch.rand(n_init) * 2 - 1
    y = torch.rand(n_init) * 2 - 1
    z = torch.rand(n_init) * 10 + 2
    T = torch.stack([x, y, z], dim=1)
    T = T.unsqueeze(0).repeat(b, 1, 1).cuda()
    T = nn.Parameter(T)

    weight_f = lambda cst, it: 1. * cst / (1 + it)
    # optimizer = torch.optim.Adam([R6d, T], 0.05, betas=(0.9, 0.999))
    optimizer = torch.optim.Adam([T, ], 0.05, betas=(0.9, 0.999))
    iteration = 3
    steps_per_iter = 200

    for it in range(iteration):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Fitting object.')

        for i in loop:
            optimizer.zero_grad()
            object_keypoints = get_object_keypoints(object_v, focal, R6d, T)

            edge_reproj_points = object_keypoints[:, :, keypoint_indices['1']] # [b, n_init, n, 2]
            loss_edge = ((edge_reproj_points.unsqueeze(2) - edge_keypoints.unsqueeze(3)[..., :2]) ** 2).sum(-1) # [b, n_init, m, n]
            loss_edge = loss_edge * edge_keypoints.unsqueeze(3)[..., 2]
            loss_edge = loss_edge.min(3)[0]
            loss_edge, indices = torch.sort(loss_edge)
            loss_edge = loss_edge[:, :, :loss_edge.shape[2] // 2].mean(-1)

            loss = weight_f(loss_edge.mean(), it)
            loss.backward()
            optimizer.step()

            indices = torch.argmin(loss_edge, dim=1)
            min_loss = weight_f(loss_edge[range(b), indices].mean(), it)

            loop.set_description('Iter: {}, Loss: {:0.4f}, min_loss: {:0.4f}'.format(i, loss.item(), min_loss.item()))

    indices = torch.argmin(loss_edge, dim=1)
    R = rotation_6d_to_matrix(R6d[range(b), indices]).detach().cpu().numpy()
    T = T[range(b), indices].detach().cpu().numpy()

    return R, T


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def fit_object():
    device = torch.device('cuda')

    object_name = 'yogaball'
    object_mesh = trimesh.load('../data/objects/{}.ply'.format(object_name), process=False)
    object_v = np.array(object_mesh.vertices)
    object_v = torch.from_numpy(object_v).float().to(device)

    root_dir = '/storage/data/huochf/HOIYouTube/{}'.format(object_name)

    with open('../data/objects/{}_keypoints.json'.format(object_name), 'r') as f:
        keypoint_indices = json.load(f)

    save_dir = './object_pose/{}'.format(object_name)
    os.makedirs(save_dir, exist_ok=True)

    for file in sorted(os.listdir(os.path.join(root_dir, 'hoi_tracking'))):
        video_id = file.split('_')[0]
        # if int(video_id) < 176:
        #     continue
        tracking_results = load_pickle(os.path.join(root_dir, 'hoi_tracking', file))

        for hoi_instance in tracking_results['hoi_instances'][::10]:
            hoi_id = hoi_instance['hoi_id']

            dataset = KeypointDataset(root_dir, video_id, hoi_id, hoi_instance)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, drop_last=False, shuffle=True)

            for item in tqdm(dataloader):
                frame_ids, edge_keypoints, optical_centers = item
                edge_keypoints = edge_keypoints.to(device)
                optical_centers = optical_centers.to(device)

                R, T = fit_pose(object_v, keypoint_indices, edge_keypoints, focal=1000)

                for batch_idx, frame_id in enumerate(frame_ids):
                    rotmat = R[batch_idx]
                    translation = T[batch_idx]
                    np.savez(os.path.join(save_dir, '{}_{}_{}.npz'.format(video_id, frame_id, hoi_id)), rotmat=rotmat, translation=translation, optical_center=optical_centers[batch_idx].detach().cpu().numpy())

                break
        print('Video {} DONE!'.format(video_id))


if __name__ == '__main__':
    fit_object()
