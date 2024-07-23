import os
import torch
import numpy as np
import trimesh
from tqdm import tqdm
import pickle
import cv2
import json

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer import TexturesUV
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
    look_at_view_transform,
    PointsRasterizer,
    PointsRenderer,
    PointsRasterizationSettings,
    NormWeightedCompositor,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    MeshRenderer,
    MeshRasterizer,
)


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def extract_bbox_from_mask(mask):
    try:
        indices = np.array(np.nonzero(np.array(mask)))
        y1 = np.min(indices[0, :])
        y2 = np.max(indices[0, :])
        x1 = np.min(indices[1, :])
        x2 = np.max(indices[1, :])

        return np.array([x1, y1, x2, y2])
    except:
        return None


def load_mask(file, h, w):
    masks = load_pickle(file)
    person_mask = np.zeros((h, w))
    if masks['human']['mask'] is not None:
        mask_h, mask_w = masks['human']['mask_shape']
        x1, y1, x2, y2 = masks['human']['mask_box']
        person_mask[y1:y2+1, x1:x2+1] = np.unpackbits(masks['human']['mask'])[:mask_h * mask_w].reshape(mask_h, mask_w)

    object_mask = np.zeros((h, w))
    if masks['object']['mask'] is not None:
        mask_h, mask_w = masks['object']['mask_shape']
        x1, y1, x2, y2 = masks['object']['mask_box']
        object_mask[y1:y2+1, x1:x2+1] = np.unpackbits(masks['object']['mask'])[:mask_h * mask_w].reshape(mask_h, mask_w)

    return person_mask, object_mask


def render_coor_map():
    object_name = 'bicycle'
    device = torch.device('cuda')
    root_dir = '/storage/data/huochf/HOIYouTube/{}'.format(object_name)
    object_pose_dir = os.path.join(root_dir, 'object_annotations', 'pose')
    object_corr_dir = os.path.join(root_dir, 'object_annotations', 'corr')
    os.makedirs(object_corr_dir, exist_ok=True)

    with open('../data/objects/bicycle_front_keypoints.json', 'r') as f:
        bicycle_front_kps_indices = json.load(f)

    bicycle_front = trimesh.load('../data/objects/bicycle_front.ply', process=False)
    front_v = torch.tensor(np.array(bicycle_front.vertices), dtype=torch.float32)
    front_f = torch.tensor(np.array(bicycle_front.faces), dtype=torch.int64)
    bicycle_back = trimesh.load('../data/objects/bicycle_back.ply', process=False)
    back_v = torch.tensor(np.array(bicycle_back.vertices), dtype=torch.float32)
    back_f = torch.tensor(np.array(bicycle_back.faces), dtype=torch.int64)
    bicycle_f = torch.cat([front_f, back_f + front_v.shape[0]], dim=0)

    rot_axis_begin = front_v[bicycle_front_kps_indices['5']].mean(0)
    rot_axis_end = front_v[bicycle_front_kps_indices['6']].mean(0)
    rot_axis = rot_axis_end - rot_axis_begin
    rot_axis = rot_axis / torch.sqrt((rot_axis ** 2).sum())

    count = 0
    for file in tqdm(os.listdir(object_pose_dir)):
        item_id = file.split('.')[0]

        video_id, frame_id, instance_id = item_id.split('_')

        if os.path.exists(os.path.join(object_corr_dir, f"{item_id}-corr.pkl")):
            continue

        image = cv2.imread(os.path.join(root_dir, 'images_temp', video_id, '{}.jpg'.format(frame_id)))
        h, w, _ = image.shape

        try:
            object_pose = np.load(os.path.join(object_pose_dir, file))
        except:
            continue
        rot_angle = torch.tensor(object_pose['rot_angle'], dtype=torch.float32)
        T = torch.tensor(object_pose['translation'], dtype=torch.float32)
        R = torch.tensor(object_pose['rotmat'], dtype=torch.float32)

        front_rotmat = axis_angle_to_matrix(rot_axis * rot_angle)
        _front_v = front_v - rot_axis_begin
        _front_v = _front_v @ front_rotmat.transpose(1, 0)
        _front_v = _front_v + rot_axis_begin
        bicycle_v = torch.cat([_front_v, back_v], dim=0)

        cx, cy = object_pose['optical_center']
        if 'f' in object_pose:
            focal = fx = fy = float(object_pose['f'])
        else:
            focal = fx = fy = 1000

        cam_R = torch.FloatTensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])
        cam_T = torch.FloatTensor([[0, 0, 0]])

        raster_settings = RasterizationSettings(image_size=[h, w], bin_size=0)
        lights = PointLights(ambient_color=[[0.7, 0.7, 0.7]], 
                             diffuse_color=[[0.2, 0.2, 0.2]], 
                             specular_color=[[0.1, 0.1, 0.1]], device=device)
        cameras = PerspectiveCameras(R=cam_R, T=cam_T,
                                 focal_length=[[fx, fy]], 
                                 principal_point=[[ cx ,  cy ]],
                                 image_size=[[h, w]],
                                 in_ndc=False,
                                     device=device)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        _verts = bicycle_v.to(device).reshape(-1, 3)
        R = R.to(device).reshape(3, 3)
        T = T.to(device).reshape(1, 3)
        _verts = _verts @ R.transpose(0, 1) + T
        input_data = Meshes(verts=[_verts.cpu()], faces=[bicycle_f]).to(device)

        fragments = rasterizer(input_data)
        depth = fragments.zbuf

        depth = depth.reshape(h, w)
        mask = torch.zeros_like(depth)
        mask[depth != -1] = 1

        x = torch.arange(0, w, 1, dtype=depth.dtype, device=depth.device).reshape(1, w, 1).repeat(h, 1, 1)
        y = torch.arange(0, h, 1, dtype=depth.dtype, device=depth.device).reshape(h, 1, 1).repeat(1, w, 1)
        z = depth.reshape(h, w, 1)
        x = x - cx
        y = y - cy
        x = x / fx * z
        y = y / fy * z
        xyz = torch.cat([x, y, z], dim=2)

        xyz = xyz - T.reshape(1, 1, 3)
        xyz = torch.matmul(xyz, R.reshape(1, 3, 3))

        xyz[depth == -1, :] = 0

        try:
            indices = torch.nonzero(mask)
            ul, _ = indices.min(0)
            br, _ = indices.max(0)
            u, l = ul.cpu().numpy()
            b, r = br.cpu().numpy()
        except:
            continue
        box_h, box_w = b - u, r - l

        box = np.array([l, u, box_w, box_h]).reshape((4, 1))
        np.savetxt(os.path.join(object_corr_dir, f"{item_id}-box.txt"), box)

        coor = {'u': int(u),'l': int(l), 'h': int(box_h), 'w': int(box_w),
                'coor': xyz[u:u+box_h, l:l+box_w, :].cpu().numpy().astype('float32')}
        with open(os.path.join(object_corr_dir, f"{item_id}-corr.pkl"), 'wb') as f:
            pickle.dump(coor, f)

        cv2.imwrite(os.path.join(object_corr_dir, f"{item_id}-label.png"), mask.cpu().numpy().astype(np.uint8))

        # count += 1
        # if count > 10:
        #     exit(0)


if __name__ == '__main__':
    render_coor_map()
