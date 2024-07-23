import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import cv2
import json
import random
import heapq
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


OBJECT_KPS_N = {
    'barbell': 4,
    'cello': 14,
    'baseball': 2,
    'tennis': 7,
    'skateboard': 8,
    'basketball': 1,
    'yogaball': 1,
    'bicycle': 10,
    'violin': 14,
}


OBJECT_KPS_PERM = {
    'barbell': [[0, 1, 2, 3], [3, 2, 1, 0]],
    'cello': [np.arange(14).tolist(), ],
    'baseball': [[0, 1, ],],
    'tennis': [[0, 1, 2, 3, 4, 5, 6]],
    'skateboard': [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 0, 1, 2, 3]],
    'basketball': [[0]],
    'yogaball': [[0]],
    'bicycle': [np.arange(10).tolist(), ],
    'violin': [np.arange(14).tolist(), ],
}


OBJECT_FLIP_INDICES = {
    'barbell': [3, 2, 1, 0],
    'cello': [0, 1, 2, 4, 3, 5, 6, 7, 8, 10, 9, 12, 11, 13],
    'baseball': [0, 1],
    'tennis': [0, 1, 3, 2, 5, 4, 6],
    'skateboard': [0, 7, 6, 5, 4, 3, 2, 1],
    'basketball': [0,],
    'yogaball': [0,],
    'bicycle': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'violin': [0, 1, 2, 4, 3, 5, 6, 7, 8, 10, 9, 12, 11, 13],
}


OBJECT_CAN_FLIP = {
    'barbell': True,
    'cello': False,
    'baseball': True,
    'tennis': True,
    'skateboard': True,
    'basketball': True,
    'yogaball': True,
    'bicycle': True,
    'violin': False,
}


class KNN:

    def __init__(self, root_dir, object_name, k, k_sampling, fps, sim_threshold=0.05, self_weights=0.5, device=torch.device('cuda')):

        self.root_dir = root_dir
        self.object_name = object_name
        self.k = k
        self.k_sampling = k_sampling
        self.fps = fps
        self.sim_threshold = sim_threshold
        self.device = device
        self.self_weights = self_weights

        self.object_kps_perm = OBJECT_KPS_PERM[object_name]
        self.smpl_flip_indices = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]
        self.object_flip_indices = OBJECT_FLIP_INDICES[object_name]
        self.object_flip = OBJECT_CAN_FLIP[object_name]

        with open(os.path.join(root_dir, 'train_test_split.json'), 'r') as f:
            train_test_split = json.load(f)
        video_ids = train_test_split[object_name]['train_videos']

        # random.seed(7)
        # random.shuffle(video_ids)
        # video_ids = video_ids[:int(len(video_ids) * 0.5)]

        self.hoi_kps_all = self.load_kps(video_ids)
        self.item_ids = sorted(list(self.hoi_kps_all.keys()))
        print('loaded {} frames'.format(len(self.item_ids)))
        self._nn = self.init_nn(self.k)
        self._heaps = self.init_heap()


    def load_kps(self, video_ids):
        hoi_kps_all = {}
        kps_dir = os.path.join(self.root_dir, self.object_name, 'hoi_kps')
        print('loading hoi kps ...')
        for video_id in tqdm(sorted(os.listdir(kps_dir))):
            if video_id not in video_ids:
                continue
            for file in os.listdir(os.path.join(kps_dir, video_id)):
                hoi_id = file.split('_')[0]
                kps_load = load_pickle(os.path.join(kps_dir, video_id, file))

                try:
                    smpl_kps_seq = kps_load['smpl_kps_seq'][:, :22]
                    object_kps_seq = kps_load['object_kps_seq'][:, :OBJECT_KPS_N[self.object_name]]
                    focal_seq = kps_load['focal_seq']
                    princpt_seq = kps_load['princpt_seq']
                    cam_R_seq = kps_load['cam_R_seq']
                    cam_T_seq = kps_load['cam_T_seq']
                except:
                    continue

                frame_ids = kps_load['frame_ids']
                n_seq, n_smpl_kps, _ = smpl_kps_seq.shape
                _, n_object_kps, _ = object_kps_seq.shape

                princpt_seq = princpt_seq.reshape(n_seq, 1, 2)
                focal_seq = focal_seq.reshape(n_seq, 1, 2)
                smpl_kps_seq = (smpl_kps_seq - princpt_seq) / focal_seq
                object_kps_seq = (object_kps_seq - princpt_seq) / focal_seq

                for i in range(0, n_seq, 30 // self.fps):

                    if np.isnan(smpl_kps_seq[i]).any() or np.isnan(object_kps_seq[i]).any():
                        continue

                    for perm_idx in range(len(self.object_kps_perm)):
                        smpl_kps = smpl_kps_seq[i]
                        object_kps = object_kps_seq[i, self.object_kps_perm[perm_idx]]
                        hoi_pseudo_3d = self.kps3dfy(smpl_kps, object_kps, cam_R_seq[i], cam_T_seq[i])
                        hoi_kps_all['{}_{}_{}_{:01d}_0'.format(video_id, hoi_id, frame_ids[i], perm_idx)] = hoi_pseudo_3d

                        if self.object_flip:
                            smpl_kps_flip = smpl_kps.copy()
                            smpl_kps_flip[:, 0] = - smpl_kps_flip[:, 0]
                            object_kps_flip = object_kps.copy()
                            object_kps_flip[:, 0] = - object_kps_flip[:, 0]
                            cam_aa = Rotation.from_matrix(cam_R_seq[i]).as_rotvec()
                            cam_aa[1] *= -1
                            cam_aa[2] *= -1
                            cam_R_flip = Rotation.from_rotvec(cam_aa).as_matrix()
                            cam_T_flip = cam_T_seq[i].copy()
                            cam_T_flip[:, 0] = - cam_T_flip[:, 0]
                            hoi_pseudo_3d = self.kps3dfy(smpl_kps_flip, object_kps_flip, cam_R_flip, cam_T_flip)
                            hoi_kps_all['{}_{}_{}_{:01d}_1'.format(video_id, hoi_id, frame_ids[i], perm_idx)] = hoi_pseudo_3d
        return hoi_kps_all


    def kps3dfy(self, smpl_kps, object_kps, cam_R, cam_T):
        n_smpl_kps = smpl_kps.shape[0]
        n_object_kps = object_kps.shape[0]

        smpl_kps = np.concatenate([smpl_kps, np.ones((n_smpl_kps, 1))], axis=1) # X_{2D} --> X_{2.5D}
        object_kps = np.concatenate([object_kps, np.ones((n_object_kps, 1))], axis=1)
        hoi_kps = np.concatenate([smpl_kps, object_kps, np.zeros((1, 3))], axis=0)

        hoi_kps = (hoi_kps - cam_T.reshape(1, 3)) @ cam_R

        return hoi_kps


    def init_nn(self, k):
        neighbors = {}

        print('initialize k nearest neighbors ...')
        for item_id in self.item_ids:
            neighbors[item_id] = {
                'neighbors': [],
                'neighbors_inverse': [],
                'new': [],
                'new_inverse': [],
            }

        for item_id in tqdm(self.item_ids):
            random_indices = np.random.choice(len(self.item_ids), k, replace=False)
            for idx in random_indices:
                neighbors[item_id]['neighbors'].append(self.item_ids[idx])
                neighbors[item_id]['new'].append(True)
                neighbors[self.item_ids[idx]]['neighbors_inverse'].append(item_id)
                neighbors[self.item_ids[idx]]['new_inverse'].append(True)

        return neighbors


    class DatasetInit:

        def __init__(self, knn):
            self.knn = knn


        def __len__(self, ):
            return len(self.knn.item_ids)


        def __getitem__(self, idx):
            item_id = self.knn.item_ids[idx]

            neighbors = self.knn._nn[item_id]['neighbors']
            kps = self.knn.hoi_kps_all[item_id]
            kps_nn = np.stack([self.knn.hoi_kps_all[item_id] for item_id in neighbors], axis=0) # [m, n, 3]

            return item_id, neighbors, kps, kps_nn


    def init_heap(self, ):
        heaps = {}
        print('building heaps ...')
        batch_size = 128

        dataset = self.DatasetInit(self)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False, drop_last=False)

        for item in tqdm(dataloader):
            item_ids, neighbors, kps, kps_nn = item
            kps = kps.float().to(self.device) # [b, n, 3]
            kps_nn = kps_nn.float().to(self.device) # [b, k, n, 3]
            b, k, n, _ = kps_nn.shape

            distance_pairwise = self.cross_view_distance(kps_nn, kps_nn).mean(-1) # [b, k]
            distance_main = self.cross_view_distance(kps.unsqueeze(1), kps_nn).squeeze(1) # [b, k]
            distance = self.self_weights * distance_main + (1 - self.self_weights) * distance_pairwise

            for b_idx in range(b):
                item_id = item_ids[b_idx]
                if item_id not in heaps:
                    heaps[item_id] = []

                for nn_idx in range(k):
                    nn_id = neighbors[nn_idx][b_idx]

                    heaps[item_id].append([- distance[b_idx, nn_idx].item(), nn_id, self.hoi_kps_all[nn_id]])

        for item_id in heaps.keys():
            heapq.heapify(heaps[item_id])

        return heaps


    def cross_view_distance(self, kps1, kps2):
        # kps1: [b, m1, n, 3], kps2: [b, m2, n, 3]
        b = kps1.shape[0]
        m1 = kps1.shape[1]
        m2 = kps2.shape[1]
        kps1_directions = kps1[:, :, :-1] - kps1[:, :, -1:]
        kps2_directions = kps2[:, :, :-1] - kps2[:, :, -1:]

        center_lines = kps1[:, :, -1].reshape(b, m1, 1, 3) - kps2[:, :, -1].reshape(b, 1, m2, 3)
        cross = torch.cross(kps1_directions.reshape(b, m1, 1, -1, 3), kps2_directions.reshape(b, 1, m2, -1, 3), dim=-1)

        distance = torch.abs((center_lines.reshape(b, m1, m2, 1, 3) * cross).sum(-1)) / (torch.norm(cross, dim=-1) + 1e-8)
        distance = distance.mean(-1)
        return distance


    def kps3d_distance(self, kps1, kps2):
        b = kps1.shape[0]
        m1 = kps1.shape[1]
        m2 = kps2.shape[1]
        kps1 = kps1 / torch.norm(kps1, dim=-1, keepdim=True)
        kps2 = kps2 / torch.norm(kps2, dim=-1, keepdim=True)
        distance = torch.abs(kps1.reshape(b, m1, 1, -1, 3) - kps2.reshape(b, 1, m2, -1, 3)).reshape(b, m1, m2, -1).mean(-1)
        return distance


    class DatasetStep:

        def __init__(self, knn):
            self.knn = knn
            self.comparison_list = self.get_comparison_list()


        def get_comparison_list(self, ):
            comparison_list = []
            for item_id in self.knn.item_ids:
                neighbors = self.knn._nn[item_id]['neighbors']
                new = self.knn._nn[item_id]['new']
                neighbors_inverse = self.knn._nn[item_id]['neighbors_inverse']
                new_inverse = self.knn._nn[item_id]['new_inverse']

                neighbors_all = neighbors + neighbors_inverse
                random_indices = np.random.choice(len(neighbors_all), min(self.knn.k_sampling, len(neighbors_all)), replace=False)

                neighbors_new = []
                neighbors_new_idx, neighbors_new_inverse_idx = [], []
                for i in range(len(new)):
                    if new[i]:
                        neighbors_new.append(neighbors[i])
                        neighbors_new_idx.append(i)
                for i in range(len(new_inverse)):
                    if new_inverse[i]:
                        neighbors_new.append(neighbors_inverse[i])
                        neighbors_new_inverse_idx.append(i)

                random_indices = np.random.choice(len(neighbors_new), min(self.knn.k_sampling, len(neighbors_new)), replace=False)
                neighbors_new = [neighbors_new[idx] for idx in random_indices]
                for idx in random_indices:
                    if idx >= len(neighbors_new_idx):
                        new_inverse[neighbors_new_inverse_idx[idx - len(neighbors_new_idx)]] = False
                    else:
                        new[neighbors_new_idx[idx]] = False

                for id1 in neighbors_new:
                    for id2 in neighbors_all:
                        if id1 != id2:
                            comparison_list.append((id1, id2))

            return comparison_list


        def __len__(self, ):
            return len(self.comparison_list)


        def __getitem__(self, idx):
            id1, id2 = self.comparison_list[idx]

            kps1 = self.knn.hoi_kps_all[id1]
            kps2 = self.knn.hoi_kps_all[id2]
            kps1_nn = np.stack([self.knn.hoi_kps_all[item_id] for item_id in self.knn._nn[id1]['neighbors']], axis=0)
            kps2_nn = np.stack([self.knn.hoi_kps_all[item_id] for item_id in self.knn._nn[id2]['neighbors']], axis=0)

            return id1, id2, kps1, kps2, kps1_nn, kps2_nn


    def step(self, ):
        change_count = 0

        batch_size = 4096
        dataset = self.DatasetStep(self)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False, drop_last=False)

        for item in tqdm(dataloader):

            id1s, id2s, kps1s, kps2s, kps1s_nn, kps2s_nn = item
            kps1s = kps1s.float().to(self.device)
            kps2s = kps2s.float().to(self.device) # [b, n, 3]
            kps1s_nn = kps1s_nn.float().to(self.device)
            kps2s_nn = kps2s_nn.float().to(self.device) # [b, k, n, 3]

            b, k, n, _ = kps2s_nn.shape

            distance_pairwise = self.cross_view_distance(kps1s.unsqueeze(1), kps2s_nn).squeeze(1).mean(-1) # [b, ]
            distance_main = self.cross_view_distance(kps1s.unsqueeze(1), kps2s.unsqueeze(1)).reshape(-1) # [b, ]
            distance = (1 - self.self_weights) * distance_pairwise + self.self_weights * distance_main

            similarity, _ = self.kps3d_distance(kps1s.unsqueeze(1), kps2s_nn).squeeze(1).min(1) # [b, ]

            for b_idx in range(b):

                distance_max = - self._heaps[id2s[b_idx]][0][0]

                if distance[b_idx] < distance_max and similarity[b_idx] > self.sim_threshold and id1s[b_idx] not in self._nn[id2s[b_idx]]['neighbors']:
                    _, item_id_del, _ = heapq.heapreplace(self._heaps[id2s[b_idx]], [-distance[b_idx].item(), id1s[b_idx], self.hoi_kps_all[id1s[b_idx]]])
                    # _, item_id_del = heapq.heapreplace(self._heaps[id2s[b_idx]], [-distance[b_idx].item(), id1s[b_idx], ])

                    nn_idx1 = self._nn[id2s[b_idx]]['neighbors'].index(item_id_del)
                    self._nn[id2s[b_idx]]['neighbors'][nn_idx1] = id1s[b_idx]
                    self._nn[id2s[b_idx]]['new'][nn_idx1] = True
                    nn_idx2 = self._nn[item_id_del]['neighbors_inverse'].index(id2s[b_idx])
                    self._nn[item_id_del]['neighbors_inverse'].pop(nn_idx2)
                    self._nn[item_id_del]['new_inverse'].pop(nn_idx2)
                    self._nn[id1s[b_idx]]['neighbors_inverse'].append(id2s[b_idx])
                    self._nn[id1s[b_idx]]['new_inverse'].append(True)

                    change_count += 1

            distance_pairwise = self.cross_view_distance(kps2s.unsqueeze(1), kps1s_nn).squeeze(1).mean(-1) # [b, ]
            distance_main = self.cross_view_distance(kps2s.unsqueeze(1), kps1s.unsqueeze(1)).reshape(-1) # [b, ]
            distance = (1 - self.self_weights) * distance_pairwise + self.self_weights * distance_main

            similarity, _ = self.kps3d_distance(kps2s.unsqueeze(1), kps1s_nn).squeeze(1).min(1) # [b, ]

            for b_idx in range(b):

                distance_max = - self._heaps[id1s[b_idx]][0][0]

                if distance[b_idx] < distance_max and similarity[b_idx] > self.sim_threshold and id2s[b_idx] not in self._nn[id1s[b_idx]]['neighbors']:
                    _, item_id_del, _ = heapq.heapreplace(self._heaps[id1s[b_idx]], [-distance[b_idx].item(), id2s[b_idx], self.hoi_kps_all[id2s[b_idx]]])
                    # _, item_id_del = heapq.heapreplace(self._heaps[id1s[b_idx]], [-distance[b_idx].item(), id2s[b_idx], ])

                    nn_idx1 = self._nn[id1s[b_idx]]['neighbors'].index(item_id_del)
                    self._nn[id1s[b_idx]]['neighbors'][nn_idx1] = id2s[b_idx]
                    self._nn[id1s[b_idx]]['new'][nn_idx1] = True
                    nn_idx2 = self._nn[item_id_del]['neighbors_inverse'].index(id1s[b_idx])
                    self._nn[item_id_del]['neighbors_inverse'].pop(nn_idx2)
                    self._nn[item_id_del]['new_inverse'].pop(nn_idx2)
                    self._nn[id2s[b_idx]]['neighbors_inverse'].append(id1s[b_idx])
                    self._nn[id2s[b_idx]]['new_inverse'].append(True)

                    change_count += 1

        return change_count


    class DatasetUpdate:

        def __init__(self, knn):
            self.knn = knn
            self.item_ids = self.collect_item_ids()


        def collect_item_ids(self, ):
            item_ids = []
            for item_id in self.knn.item_ids:
                new = self.knn._nn[item_id]['new']
                for i in range(len(new)):
                    if new[i]:
                        item_ids.append(item_id)
                        break  

            return item_ids


        def __len__(self, ):
            return len(self.item_ids)


        def __getitem__(self, idx):
            item_id = self.item_ids[idx]

            neighbors = [items[1] for items in self.knn._heaps[item_id]]
            kps = self.knn.hoi_kps_all[item_id] 
            kps_nn = np.stack([self.knn.hoi_kps_all[item_id] for item_id in neighbors], axis=0) # [m, n, 3]

            return item_id, neighbors, kps, kps_nn


    def update_distance(self, ):
        print('update distances ...')
        batch_size = 128

        dataset = self.DatasetUpdate(self)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False, drop_last=False)

        for item in tqdm(dataloader):
            item_ids, neighbors, kps, kps_nn = item
            kps = kps.float().to(self.device) # [b, n, 3]
            kps_nn = kps_nn.float().to(self.device) # [b, k, n, 3]
            b, k, _, _ = kps_nn.shape

            distance_pairwise = self.cross_view_distance(kps_nn, kps_nn).mean(-1) # [b, k]
            distance_main = self.cross_view_distance(kps.unsqueeze(1), kps_nn).squeeze(1) # [b, k]
            distance = (1 - self.self_weights) * distance_pairwise + self.self_weights * distance_main

            for b_idx in range(b):
                item_id = item_ids[b_idx]

                for nn_idx in range(k):
                    nn_id = neighbors[nn_idx][b_idx]

                    self._heaps[item_id][nn_idx][0] = - distance[b_idx][nn_idx].item()

        for item_id in self._heaps.keys():
            heapq.heapify(self._heaps[item_id])


    def save(self, path):
        save_pickle(self._heaps, path)


def main(args):
    knn = KNN(root_dir=args.root_dir,
              object_name=args.object,
              sim_threshold=0.2,
              k=args.k,
              k_sampling=args.k_sampling,
              fps=args.fps,
              device=torch.device('cuda'))
    n_steps = args.n_steps
    for i in range(n_steps):
        knn.self_weights = 0.5
        n_changes = knn.step()
        print('Iter: {}, changes: {}'.format(i, n_changes))
        knn.update_distance()

    os.makedirs(args.save_dir, exist_ok=True)
    knn.save(os.path.join(args.save_dir, '{}_fps_{:02d}_k_{:03d}.pkl'.format(args.object, args.fps, args.k)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN Grouping')
    parser.add_argument('--root_dir', default='/storage/data/huochf/HOIYouTube/')
    parser.add_argument('--object', default='barbell')
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--k_sampling', type=int, default=8)
    parser.add_argument('--fps', type=int, default=3)
    parser.add_argument('--n_steps', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='./outputs/knn_grouping')

    args = parser.parse_args()
    main(args)
