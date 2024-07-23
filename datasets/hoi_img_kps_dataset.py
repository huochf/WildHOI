import os
import numpy as np
import json
import random
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from datasets.utils import get_augmentation_params, generate_image_patch, load_pickle


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


class WildHOIImageDataset:

    def __init__(self, root_dir, object_name='barbell', fps=3, k=16, split='train'):
        self.root_dir = root_dir
        self.object_name = object_name
        self.fps = fps
        self.split = split
        self.is_train = split == 'train'

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
        
        self.hoi_kps_all, self.hoi_bb_cxcys_all = self.load_kps(video_ids)
        self.item_ids = sorted(list(self.hoi_kps_all.keys()))
        print('loaded {} frames'.format(len(self.item_ids)))
        self.knn_grouping = load_pickle('outputs/knn_grouping/{}_fps_{:02d}_k_{:03d}.pkl'.format(object_name, fps, k))
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


    def load_kps(self, video_ids):
        hoi_kps_all = {}
        hoi_bb_cxcys_all = {}
        kps_dir = os.path.join(self.root_dir, self.object_name, 'hoi_kps')
        print('loading hoi kps ...')
        for video_id in tqdm(sorted(os.listdir(kps_dir))):
            if video_id not in video_ids:
                continue
            _hoi_tracking_results = load_pickle(os.path.join(self.root_dir, self.object_name, 'hoi_tracking', '{}_tracking.pkl'.format(video_id)))
            hoi_tracking_results = {}
            for sequence in _hoi_tracking_results['hoi_instances']:
                hoi_tracking_results[sequence['hoi_id']] = {
                    item['frame_id']: {'person_bbox': item['person_bbox'], 'object_bbox': item['object_bbox']} for item in sequence['sequences']
                }

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

                    frame_id = frame_ids[i]
                    person_bbox = hoi_tracking_results[hoi_id][frame_id]['person_bbox']
                    object_bbox = hoi_tracking_results[hoi_id][frame_id]['object_bbox']
                    if person_bbox is None:
                        person_bbox = [0, 0, 256, 256]
                    if object_bbox is None:
                        object_bbox = person_bbox
                    x1, y1, x2, y2 = person_bbox
                    _x1, _y1, _x2, _y2 = object_bbox
                    x1 = min(x1, _x1)
                    y1 = min(y1, _y1)
                    x2 = max(x2, _x2)
                    y2 = max(y2, _y2)
                    box_size = max(y2 - y1, x2 - x1) * 1.2
                    box_cx, box_cy = (x2 + x1) / 2, (y2 + y1) / 2
                    hoi_box_cxcys = (box_cx, box_cy, box_size)

                    for perm_idx in range(len(self.object_kps_perm)):
                        smpl_kps = smpl_kps_seq[i]
                        object_kps = object_kps_seq[i, self.object_kps_perm[perm_idx]]
                        hoi_pseudo_3d = self.kps3dfy(smpl_kps, object_kps, cam_R_seq[i], cam_T_seq[i])
                        item_id = '{}_{}_{}_{:01d}_0'.format(video_id, hoi_id, frame_ids[i], perm_idx)
                        hoi_kps_all[item_id] = hoi_pseudo_3d
                        hoi_bb_cxcys_all[item_id] = hoi_box_cxcys

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
                            item_id = '{}_{}_{}_{:01d}_1'.format(video_id, hoi_id, frame_ids[i], perm_idx)
                            hoi_kps_all[item_id] = hoi_pseudo_3d
                            hoi_bb_cxcys_all[item_id] = hoi_box_cxcys

        return hoi_kps_all, hoi_bb_cxcys_all


    def kps3dfy(self, smpl_kps, object_kps, cam_R, cam_T):
        n_smpl_kps = smpl_kps.shape[0]
        n_object_kps = object_kps.shape[0]

        smpl_kps = np.concatenate([smpl_kps, np.ones((n_smpl_kps, 1))], axis=1) # X_{2D} --> X_{2.5D}
        object_kps = np.concatenate([object_kps, np.ones((n_object_kps, 1))], axis=1)
        hoi_kps = np.concatenate([smpl_kps, object_kps, np.zeros((1, 3))], axis=0)

        hoi_kps = (hoi_kps - cam_T.reshape(1, 3)) @ cam_R
        hoi_kps[:-1] = hoi_kps[:-1] - hoi_kps[-1:]
        hoi_kps[:-1] = hoi_kps[:-1] / np.linalg.norm(hoi_kps[:-1], axis=-1, keepdims=True)

        return hoi_kps


    def __len__(self, ):
        return len(self.item_ids)


    def __getitem__(self, idx):
        item_id = self.item_ids[idx]
        video_id, hoi_id, frame_id, perm_id, flip_id = item_id.split('_')
        box_cx, box_cy, box_size = self.hoi_bb_cxcys_all[item_id]

        tx, ty, rot, scale, color_scale = get_augmentation_params()

        box_cx += tx * box_size
        box_cy += ty * box_size
        box_size = box_size * scale
        out_size = 256

        image_path = os.path.join(self.root_dir, self.object_name, 'images_temp', video_id, '{}.jpg'.format(frame_id))
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if flip_id == '1':
            image = image[:, ::-1]
        img_patch, _ = generate_image_patch(image, box_cx, box_cy, box_size, out_size, rot, color_scale)
        img_patch = img_patch[:, :, ::-1].astype(np.float32)
        img_patch = img_patch.transpose((2, 0, 1))
        img_patch = img_patch / 256

        for n_c in range(3):
            img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        hoi_kps = self.hoi_kps_all[item_id] # [n, 3]

        if not self.is_train:
            return img_patch.astype(np.float32), hoi_kps.astype(np.float32)

        nn_kps = []
        distances = []
        for item in self.knn_grouping[item_id]:
            distance, nn_id, _ = item
            distances.append(distance)
            if nn_id not in self.hoi_kps_all:
                nn_kps.append(hoi_kps)
                print('key ', nn_id, item_id)
            else:
                nn_kps.append(self.hoi_kps_all[nn_id])
        nn_kps = np.stack(nn_kps)
        distances = np.array(distances)

        return img_patch.astype(np.float32), hoi_kps.astype(np.float32), nn_kps.astype(np.float32), distances.astype(np.float32)
