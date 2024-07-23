import os
import argparse
import pickle
import numpy as np
import json
import cv2
import trimesh
from tqdm import tqdm
import torch
from smplx import SMPLX

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)
import neural_renderer as nr

from KNN_grouping import KNN


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data, path):
    with open(path, 'wb') as f:
        data = pickle.dump(data, f)
    return data


def get_occlusion_maps(args):
    device = torch.device('cuda')
    root_dir = args.root_dir
    object_name = args.object
    smpl_dir = os.path.join(root_dir, object_name, 'smplx_tuned')
    object_pose_dir = os.path.join(root_dir, object_name, 'object_pose')
    tracking_results_dir = os.path.join(root_dir, object_name, 'hoi_tracking')

    if object_name in ['cello', 'violin']:
        object_mesh = trimesh.load('data/objects/{}_body.ply'.format(object_name), process=False)
    else:
        object_mesh = trimesh.load('data/objects/{}.ply'.format(object_name), process=False)

    object_v = np.array(object_mesh.vertices)
    object_v_org = torch.from_numpy(object_v).float().to(device)
    object_f = torch.from_numpy(np.array(object_mesh.faces).astype(np.int64)).to(device)

    smplx = SMPLX('data/smpl/smplx/', gender='neutral', use_pca=False).to(device)

    smpl_f = torch.tensor(smplx.faces.astype(np.int64)).to(device)

    begin_idx = 0
    end_idx = 99999
    tracking_results_dir = os.path.join(root_dir, object_name, 'hoi_tracking')

    fx = fy = 5000
    cx = cy = 128
    w = h = 256
    cameras = PerspectiveCameras(R=torch.FloatTensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]]),
                                 T=torch.FloatTensor([[0, 0, 0]]),
                                 focal_length=[[fx, fy]],
                                 principal_point=[[cx, cy]],
                                 image_size=[[h, w]],
                                 in_ndc=False,
                                 device=device)
    raster_settings = RasterizationSettings(image_size=[h, w], bin_size=0)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    smpl_occlusion_map_mean = None
    object_occlusion_map_mean = None
    count = 0

    fps = 30 // 3
    knn = KNN(root_dir, object_name, k=16, k_sampling=8, fps=3)
    hoi_kps_unique = []

    with open(os.path.join(root_dir, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)
    video_ids = train_test_split[object_name]['train_videos']

    for video_idx in range(begin_idx, end_idx):
        video_id = '{:04d}'.format(video_idx)
        if video_id not in video_ids:
            continue
        mask_dir = os.path.join(root_dir, object_name, 'hoi_mask', video_id)
        image_dir = os.path.join(root_dir, object_name, 'images_temp', video_id)

        if not os.path.exists(os.path.join(tracking_results_dir, '{}_tracking.pkl'.format(video_id))):
            continue

        smplx_dir = os.path.join(root_dir, object_name, 'smplx_tuned', video_id)
        object_pose_dir = os.path.join(root_dir, object_name, 'object_pose', video_id)
        tracking_results = load_pickle(os.path.join(tracking_results_dir, '{}_tracking.pkl'.format(video_id)))

        for hoi_instance in tracking_results['hoi_instances']:
            frame_id = hoi_instance['sequences'][0]['frame_id']
            image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(frame_id)))
            h, w, _ = image.shape

            hoi_id = hoi_instance['hoi_id']
            try:
                smpl_tuned_params = load_pickle(os.path.join(smplx_dir, '{}_smplx.pkl'.format(hoi_id)))[::fps]
                object_pose_params = load_pickle(os.path.join(object_pose_dir, '{}_obj_RT.pkl'.format(hoi_id)))[::fps]
            except:
                continue

            output_dir = os.path.join(root_dir, object_name, 'occlusion_maps', video_id, hoi_id)
            os.makedirs(output_dir, exist_ok=True)

            for item_idx, item in enumerate(tqdm(hoi_instance['sequences'][::fps])):
                frame_id = item['frame_id']

                # if os.path.exists(os.path.join(output_dir, '{}.pkl'.format(frame_id))):
                #     continue

                assert frame_id == smpl_tuned_params[item_idx]['frame_id']
                assert frame_id == object_pose_params[item_idx]['frame_id']

                _fx, _fy = smpl_tuned_params[item_idx]['focal']
                _cx, _cy = smpl_tuned_params[item_idx]['princpt']
                _s_crop = 256 / 5000 * _fy
                x1 = int(_cx - _s_crop / 2)
                y1 = int(_cy - _s_crop / 2)
                box_h = box_w = int(_s_crop)
                person_mask_sam = np.zeros((box_h, box_w))

                _image = cv2.imread(os.path.join(root_dir, object_name, 'images_temp', video_id, '{}.jpg'.format(frame_id)))

                masks = load_pickle(os.path.join(mask_dir, '{}'.format(hoi_id), '{}.pkl'.format(frame_id)))
                _person_mask_sam = np.zeros((h, w))
                if masks['human']['mask'] is not None:
                    mask_h, mask_w = masks['human']['mask_shape']
                    _x1, _y1, _x2, _y2 = masks['human']['mask_box']
                    _person_mask_sam[_y1:_y2+1, _x1:_x2+1] = np.unpackbits(masks['human']['mask'])[:mask_h * mask_w].reshape(mask_h, mask_w)

                image = np.zeros((box_h, box_w, 3))
                if y1 < 0 and x1 >= 0:
                    person_mask_sam[-y1:min(box_h, h - y1),  :min(box_w, w - x1)] = _person_mask_sam[:y1 + box_h, x1 : x1 + box_w]
                    image[-y1:min(box_h, h - y1),  :min(box_w, w - x1)] = _image[:y1 + box_h, x1 : x1 + box_w]
                elif x1 < 0 and y1 >= 0:
                    person_mask_sam[:min(box_h, h - y1),  -x1:min(box_w, w - x1)] = _person_mask_sam[y1 : y1 + box_h, : x1 + box_w]
                    image[:min(box_h, h - y1),  -x1:min(box_w, w - x1)] = _image[y1 : y1 + box_h, : x1 + box_w]
                elif y1 < 0 and x1 < 0:
                    person_mask_sam[-y1:min(box_h, h - y1),  -x1:min(box_w, w - x1)] = _person_mask_sam[: y1 + box_h, : x1 + box_w]
                    image[-y1:min(box_h, h - y1),  -x1:min(box_w, w - x1)] = _image[: y1 + box_h, : x1 + box_w]
                else:
                    person_mask_sam[:min(box_h, h - y1), :min(box_w, w - x1)] = _person_mask_sam[y1 : y1 + box_h, x1 : x1 + box_w]
                    image[:min(box_h, h - y1), :min(box_w, w - x1)] = _image[y1 : y1 + box_h, x1 : x1 + box_w]

                try:
                    person_mask_sam = cv2.resize(person_mask_sam, dsize=[256, 256], interpolation=cv2.INTER_NEAREST)
                    image = cv2.resize(image, dsize=[256, 256], interpolation=cv2.INTER_NEAREST)
                except:
                    continue
                person_mask_sam = torch.from_numpy(person_mask_sam).float().to(device)

                smpl_body_pose = torch.tensor(smpl_tuned_params[item_idx]['body_pose']).reshape(1, 63).float().to(device)
                smpl_shape = torch.tensor(smpl_tuned_params[item_idx]['betas']).reshape(1, 10).float().to(device)
                smplx_lhand_pose = torch.tensor(smpl_tuned_params[item_idx]['left_hand_pose']).reshape(1, 45).float().to(device)
                smplx_rhand_pose = torch.tensor(smpl_tuned_params[item_idx]['right_hand_pose']).reshape(1, 45).float().to(device)
                cam_T = torch.tensor(smpl_tuned_params[item_idx]['cam_T']).reshape(1, 1, 3).float().to(device)
                cam_R = torch.tensor(smpl_tuned_params[item_idx]['cam_R']).reshape(1, 3, 3).float().to(device)

                smpl_out = smplx(betas=smpl_shape, body_pose=smpl_body_pose, left_hand_pose=smplx_lhand_pose, right_hand_pose=smplx_rhand_pose,)
                smpl_v = smpl_out.vertices.detach()
                smpl_J = smpl_out.joints.detach()
                smpl_v = smpl_v - smpl_J[:, :1]
                smpl_v = smpl_v @ cam_R.transpose(2, 1) + cam_T

                rotmat = torch.tensor(object_pose_params[item_idx]['rotmat']).reshape(3, 3).to(device)
                trans = torch.tensor(object_pose_params[item_idx]['trans']).reshape(1, 3).to(device)
                object_v = object_v_org @ rotmat.transpose(-1, -2) + trans

                object_mesh = Meshes(verts=[object_v.cpu().reshape(-1, 3)], faces=[object_f.cpu().reshape(-1, 3)]).to(device)
                smpl_mesh = Meshes(verts=[smpl_v.cpu().reshape(-1, 3)], faces=[smpl_f.cpu().reshape(-1, 3)]).to(device)

                object_fragments = rasterizer(object_mesh)
                object_depth = object_fragments.zbuf
                object_depth = object_depth.reshape(256, 256)
                object_mask = torch.zeros_like(object_depth)
                object_mask[object_depth != -1] = 1

                smpl_fragments = rasterizer(smpl_mesh)
                smpl_depth = smpl_fragments.zbuf
                smpl_depth = smpl_depth.reshape(256, 256)
                smpl_mask = torch.zeros_like(smpl_depth)
                smpl_mask[smpl_depth != -1] = 1

                occlusion_mask = (smpl_mask != 0) & (object_mask != 0)

                def project(points, fx, fy, cx, cy):
                    u = points[:, :, 0] / points[:, :, 2] * fx + cx
                    v = points[:, :, 1] / points[:, :, 2] * fy + cy
                    return torch.stack([u, v], dim=2)

                smpl_proj_uv = project(smpl_v, fx, fy, cx, cy).reshape(-1, 2).to(torch.int64)
                object_proj_uv = project(object_v.unsqueeze(0), fx, fy, cx, cy).reshape(-1, 2).to(torch.int64)

                smpl_proj_uv = smpl_proj_uv.clip(min=0, max=255)
                object_proj_uv = object_proj_uv.clip(min=0, max=255)

                smpl_v_back = smpl_depth[smpl_proj_uv[:, 1], smpl_proj_uv[:, 0]] + 0.01 < smpl_v.reshape(-1, 3)[:, 2]
                object_v_back = object_depth[object_proj_uv[:, 1], object_proj_uv[:, 0]] + 0.01 < object_v.reshape(-1, 3)[:, 2]

                smpl_proj_uv = smpl_proj_uv.clip(0, 255)
                object_proj_uv = object_proj_uv.clip(0, 255)
                # print(smpl_proj_uv.max(), smpl_proj_uv.min(), object_proj_uv.max(), object_proj_uv.min())
                # print(occlusion_mask.shape)
                # exit(0)
                smpl_v_mask_back = smpl_v_back & (occlusion_mask[smpl_proj_uv[:, 1], smpl_proj_uv[:, 0]] != 0) & (person_mask_sam[smpl_proj_uv[:, 1], smpl_proj_uv[:, 0]] != 0)
                smpl_v_mask_front = (~ smpl_v_back) & (occlusion_mask[smpl_proj_uv[:, 1], smpl_proj_uv[:, 0]] != 0) & (person_mask_sam[smpl_proj_uv[:, 1], smpl_proj_uv[:, 0]] == 0)
                smpl_v_mask = smpl_v_mask_back | smpl_v_mask_front
                object_v_mask_back = object_v_back & (occlusion_mask[object_proj_uv[:, 1], object_proj_uv[:, 0]] != 0) & (person_mask_sam[object_proj_uv[:, 1], object_proj_uv[:, 0]] == 0)
                object_v_mask_front = (~ object_v_back) & (occlusion_mask[object_proj_uv[:, 1], object_proj_uv[:, 0]] != 0) & (person_mask_sam[object_proj_uv[:, 1], object_proj_uv[:, 0]] != 0)
                object_v_mask = object_v_mask_front | object_v_mask_back

                smpl_v_mask = (occlusion_mask[smpl_proj_uv[:, 1], smpl_proj_uv[:, 0]] != 0) 
                object_v_mask = (occlusion_mask[object_proj_uv[:, 1], object_proj_uv[:, 0]] != 0) 

                # occlusion_map = {
                #     'smpl_v_mask': smpl_v_mask.detach().cpu().numpy(), 
                #     'object_v_mask': object_v_mask.detach().cpu().numpy()
                # }
                # save_pickle(occlusion_map, os.path.join(output_dir, '{}.pkl'.format(frame_id)))

                if '{}_{}_{}_0_0'.format(video_id, hoi_id, frame_id) not in knn.hoi_kps_all:
                    print('key error {}'.format('{}_{}_{}_0_0'.format(video_id, hoi_id, frame_id)))
                    continue
                current_hoi_kps = knn.hoi_kps_all['{}_{}_{}_0_0'.format(video_id, hoi_id, frame_id)]
                duplicated = False
                if len(hoi_kps_unique) == 0:
                    hoi_kps_unique.append(current_hoi_kps)
                    duplicated = False
                else:
                    _batch_size = 512
                    min_distance = 1e8
                    n_kps = current_hoi_kps.shape[0]
                    for i in range(0, len(hoi_kps_unique), _batch_size):
                        _hoi_kps_unique_batch = torch.tensor(hoi_kps_unique[i:i+_batch_size])
                        _hoi_kps_unique_batch = _hoi_kps_unique_batch.float().to(device).reshape(1, -1, n_kps, 3)
                        _current_hoi_kps = torch.from_numpy(current_hoi_kps).float().to(device).reshape(1, 1, n_kps, 3)
                        distance = knn.kps3d_distance(_hoi_kps_unique_batch, _current_hoi_kps).reshape(-1).min()
                        if distance.item() < min_distance:
                            min_distance = distance.item()
                    if min_distance < 0.1:
                        duplicated = True

                if not duplicated:
                    if smpl_occlusion_map_mean is None:
                        smpl_occlusion_map_mean = smpl_v_mask.detach().cpu().numpy().astype(np.float32)
                        object_occlusion_map_mean = object_v_mask.detach().cpu().numpy().astype(np.float32)
                    else:
                        smpl_occlusion_map_mean += smpl_v_mask.detach().cpu().numpy().astype(np.float32)
                        object_occlusion_map_mean += object_v_mask.detach().cpu().numpy().astype(np.float32)
                    count += 1

        print('video {} done'.format(video_id))

    save_pickle({'smpl_occlusion_map_mean': smpl_occlusion_map_mean / count, 'object_occlusion_map_mean': object_occlusion_map_mean / count}, 
        'outputs/mean_occlusion/occlusion_map_mean_{}.pkl'.format(object_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate KPS (BEHAVE)')
    parser.add_argument('--root_dir', default='/storage/data/huochf/HOIYouTube', type=str)
    parser.add_argument('--object', default='barbell', type=str)
    args = parser.parse_args()

    get_occlusion_maps(args)
