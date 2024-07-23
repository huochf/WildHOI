import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import math
import json
import pickle
import trimesh
from tqdm import tqdm
import torch.nn.functional as F
from models import Model
import neural_renderer as nr
from datasets.object_image_dataset import ObjectImageDataset, generate_image_patch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix

from datasets.object_corr_dataset import CORR_NORM


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


def plot_keypoints(image, points, labels):
    points_colors = [
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
    line_thickness = 2
    thickness = 4
    lineType = 8

    for point, label in zip(points, labels):
        x, y = point
        if x == 0 and y == 0:
            continue
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), thickness, points_colors[label], thickness=-1, lineType=lineType)
        # cv2.rectangle(image, (x - 10, y - 10), (x + 10, y + 10), (255, 255, 0), 2)

    return image


def visualize_bicycle(outputs, images, bicycle_front_v, bicycle_back_v, bicycle_front_f, bicycle_back_f, 
    rot_axis, rot_axis_begin, rot_angle, R, T, center, s, item_ids, object_name):

    bicycle_f = torch.cat([bicycle_front_f, bicycle_back_f + bicycle_front_v.shape[0]], dim=0)

    corr_maps_pred = outputs
    bs, _, h, w = images.shape
    images = images.numpy()
    std = np.array((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
    mean = np.array((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
    images = (images * std + mean) * 255
    images = images.clip(0, 255).astype(np.uint8)

    images_save = []
    for b_idx in range(bs):
        image = images[b_idx].transpose(1, 2, 0)

        image_corr_pred = corr_maps_pred[b_idx, :3].transpose(1, 2, 0)
        image_corr_pred = ((image_corr_pred + 1) / 2 * 255.).clip(0, 255).astype(np.uint8)
        image_corr_pred = cv2.resize(image_corr_pred, dsize=(w, h))

        mask = corr_maps_pred[b_idx, 3]
        mask = np.stack([mask, mask, mask], axis=2)
        mask = (mask * 255.).clip(0, 255).astype(np.uint8)
        mask = cv2.resize(mask, dsize=(w, h))

        image_corr_pred[mask < 0.02] = 0

        item_id = item_ids[b_idx]
        video_id, hoi_id, frame_id = item_id.split('_')
        image_org = cv2.imread(os.path.join('/storage/data/huochf/HOIYouTube/{}/images_temp'.format(object_name), video_id, '{}.jpg'.format(frame_id)))
        image_org = cv2.resize(image_org, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

        print(T[b_idx])
        if T[b_idx].sum() != 0 and T[b_idx][2] > 0:
            _bicycle_front = bicycle_front_v - rot_axis_begin.reshape(1, 3)
            front_rotmat = axis_angle_to_matrix(rot_axis * rot_angle[b_idx]) # [3, 3]
            _bicycle_front = _bicycle_front @ front_rotmat.transpose(1, 0)
            _bicycle_front = _bicycle_front + rot_axis_begin.reshape(1, 3)
            bicycle_v = torch.cat([_bicycle_front, bicycle_back_v], dim=0)
            image_org_render_pose = render(image_org.copy(), bicycle_v, bicycle_f, 1000, center[b_idx, 0], center[b_idx, 1], T[b_idx], R[b_idx])
        else:
            # print(T[b_idx])
            image_org_render_pose = image_org.copy()
        image_render_pose, _ = generate_image_patch(image_org_render_pose, center[b_idx, 0], center[b_idx, 1], s[b_idx], h, 0, None)

        image = np.concatenate([image_render_pose[:, :, ::-1], image_corr_pred], axis=1)
        images_save.append(image)
    return images_save


def render(image, v, f, focal, cx, cy, t, R):
    device = torch.device('cuda')
    v = v @ R.transpose(1, 0) + t.reshape(1, 3)
    v = v.reshape(1, -1, 3).to(device)
    f = f.reshape(1, -1, 3).to(device)
    textures = torch.tensor([0.65098039, 0.74117647, 0.85882353], dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).repeat(1, f.shape[1], 1, 1, 1, 1).to(device)
    h, w, _ = image.shape

    fx = fy = focal
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


def fit_object(model_outputs, c, s, item_ids, object_name, bicycle_front, bicycle_back, rot_axis, rot_axis_begin):
    device = torch.device('cuda')

    corr_maps_pred = model_outputs

    b, _, corr_h, corr_w = corr_maps_pred.shape

    corr_maps_pred = torch.tensor(corr_maps_pred)
    coor_maps = corr_maps_pred.permute(0, 2, 3, 1)
    corr_norm = CORR_NORM[object_name]
    coor_x3d = coor_maps[:, :, :, :3] * torch.tensor(corr_norm).reshape(1, 1, 1, 3)
    coor_x3d = coor_x3d.float().to(device)

    coor_mask = coor_maps[:, :, :, 3:]
    coor_mask = coor_mask.float().to(device)

    grid_2d = torch.arange(corr_h).float().to(device)
    ys, xs = torch.meshgrid(grid_2d, grid_2d) # (h, w)
    grid_2d = torch.stack([xs, ys], dim=2).unsqueeze(0).repeat(b, 1, 1, 1).reshape(b, -1, 2) # (b, h * w, 2)
    stride = s / corr_h
    stride = stride.reshape(b, 1, 1).float().to(device)
    x1 = c[:, 0] - s / 2
    y1 = c[:, 1] - s / 2
    begin_point = torch.stack([x1, y1], dim=1)
    begin_point = begin_point.reshape(b, 1, 2).float().to(device)
    coor_x2d = grid_2d * stride + begin_point # [b, h*w, 2]
    coor_x3d = coor_x3d.reshape(b, -1, 3) # [b, n_init, h*w, 3]
    coor_mask = coor_mask.reshape(b, -1).clamp(0, 1)

    cx = c[:, 0].float().to(device).reshape(b, 1, 1)
    cy = c[:, 1].float().to(device).reshape(b, 1, 1)
    f = 1000

    cam_K = torch.eye(3).unsqueeze(0).repeat(b, 1, 1)
    cam_K[:, 0, 0] = cam_K[:, 1, 1] = f
    cam_K[:, 0, 2] = c[:, 0]
    cam_K[:, 1, 2] = c[:, 1]

    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    R_vects = []
    T_vectors = []
    x2d_np = coor_x2d.detach().cpu().numpy()
    x3d_np = coor_x3d.detach().cpu().numpy()
    binary_mask = coor_mask.detach().cpu().numpy()
    binary_mask = binary_mask > 0.5
    for x2d_np_, x3d_np_, mask_np_, K in zip(x2d_np, x3d_np, binary_mask, cam_K.cpu().numpy()):
        try:
            success, R_vector, T_vector, _ = cv2.solvePnPRansac(
                x3d_np_[mask_np_], x2d_np_[mask_np_], K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        except:
            R_vector = np.zeros(3)
            T_vector = np.zeros(3)
        q = R_vector.reshape(-1)
        R_vects.append(q)
        T_vectors.append(T_vector.reshape(-1))
    R_vects = coor_x2d.new_tensor(R_vects)
    T_vectors = coor_x2d.new_tensor(T_vectors)

    b = corr_maps_pred.shape[0]
    R6d = matrix_to_rotation_6d(axis_angle_to_matrix(R_vects)).to(torch.float32).to(device)
    R6d = R6d.reshape(b, 6)
    R6d = nn.Parameter(R6d)

    T = T_vectors.reshape(b, 3).to(torch.float32).to(device)
    T = nn.Parameter(T)

    rot_angle = nn.Parameter(torch.zeros(b,).float().to(device))

    bicycle_front = bicycle_front.to(device).reshape(1, 1, -1, 3)
    bicycle_back = bicycle_back.to(device).reshape(1, 1, -1, 3)
    rot_axis = rot_axis.to(device).unsqueeze(0) # [1, 3]
    rot_axis_begin = rot_axis_begin.to(device).reshape(1, 1, 3)
    coor_x3d_front_dist = ((coor_x3d.reshape(b, -1, 1, 3) - bicycle_front) ** 2).sum(-1).min(-1)[0]
    coor_x3d_back_dist = ((coor_x3d.reshape(b, -1, 1, 3) - bicycle_back) ** 2).sum(-1).min(-1)[0]

    coor_x3d_front_mask = coor_x3d_front_dist < coor_x3d_back_dist
    coor_x3d_back_mask = coor_x3d_front_dist > coor_x3d_back_dist

    weight_f = lambda cst, it: 1. * cst / (1 + it)
    optimizer = torch.optim.Adam([R6d, T, rot_angle], 0.05, betas=(0.9, 0.999))
    iteration = 3
    steps_per_iter = 500
    for it in range(iteration):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Fitting object.')
        for i in loop:
            optimizer.zero_grad()

            front_rotmat = axis_angle_to_matrix(rot_axis * rot_angle.reshape(b, 1)) # [b, 3, 3]
            front_v = coor_x3d - rot_axis_begin
            front_v = front_v @ front_rotmat.transpose(2, 1) + rot_axis_begin

            global_rotmat = rotation_6d_to_matrix(R6d)
            global_trans = T.reshape(b, 1, 3)

            front_v = torch.matmul(front_v, global_rotmat.transpose(2, 1)) + global_trans
            u = front_v[..., 0] / (front_v[..., 2] + 1e-8) * f + cx.reshape(b, 1)
            v = front_v[..., 1] / (front_v[..., 2] + 1e-8) * f + cy.reshape(b, 1)
            front_v_x2d_reproj = torch.stack([u, v], dim=2)

            back_v = torch.matmul(coor_x3d, global_rotmat.transpose(2, 1)) + global_trans
            u = back_v[:, :, 0] / (back_v[:, :, 2] + 1e-8) * f + cx.reshape(b, 1)
            v = back_v[:, :, 1] / (back_v[:, :, 2] + 1e-8) * f + cy.reshape(b, 1)
            back_v_x2d_reproj = torch.stack([u, v], dim=2)

            loss_front = ((front_v_x2d_reproj - coor_x2d) ** 2).sum(-1)
            loss_back = ((back_v_x2d_reproj - coor_x2d) ** 2).sum(-1)

            loss_coor = loss_front * coor_x3d_front_mask * coor_mask + loss_back * coor_x3d_back_mask * coor_mask
            loss_coor = loss_coor.mean(-1) # (b, n_init)

            loss = weight_f(loss_coor.mean(), it)

            loss.backward()
            optimizer.step()

            loop.set_description('Iter: {}, Loss: {:0.4f}, loss_coor: {:0.4f}'.format(
                i, loss.item(), loss_coor.mean().item()))

    R = rotation_6d_to_matrix(R6d).detach().cpu().numpy()
    T = T.detach().cpu().numpy()
    rot_angle = rot_angle.detach().cpu().numpy()

    return rot_angle, R, T


def inference():
    device = torch.device('cuda')
    object_name = 'bicycle'

    with open('/storage/data/huochf/HOIYouTube/train_test_split_{}.json'.format(object_name), 'r') as f:
        train_test_split = json.load(f)
        
    bicycle_front = trimesh.load('../data/objects/bicycle_front.ply', process=False)
    bicycle_back = trimesh.load('../data/objects/bicycle_back.ply', process=False)

    bicycle_front_f = torch.from_numpy(np.array(bicycle_front.faces))
    bicycle_back_f = torch.from_numpy(np.array(bicycle_back.faces))
    bicycle_front = torch.from_numpy(bicycle_front.vertices).float()
    bicycle_back = torch.from_numpy(bicycle_back.vertices).float()


    with open('../data/objects/bicycle_front_keypoints.json', 'r') as f:
        bicycle_front_kps_indices = json.load(f)
    with open('../data/objects/bicycle_back_keypoints.json', 'r') as f:
        bicycle_back_kps_indices = json.load(f)
    rot_axis_begin = bicycle_front[bicycle_front_kps_indices['5']].mean(0)
    rot_axis_end = bicycle_front[bicycle_front_kps_indices['6']].mean(0)
    rot_axis = rot_axis_end - rot_axis_begin
    rot_axis = rot_axis / torch.sqrt((rot_axis ** 2).sum())

    os.makedirs('./inference_vis/{}'.format(object_name), exist_ok=True)

    model = Model(num_kps=12).to(device)
    dataset = ObjectImageDataset(root_dir='/storage/data/huochf/HOIYouTube/{}'.format(object_name), video_ids=train_test_split['train'], out_res=224, coor_res=64)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=8, shuffle=True)

    model.load_checkpoint('./weights/model_{}_stage1.pth'.format(object_name))

    model.eval()
    corr_maps = []
    for idx, data in enumerate(tqdm(dataloader)):
        # if idx > 2000:
        #     break
        images, masks, c, s, item_ids = data
        images = images.to(device)

        outputs = model.inference_step(images)
        rot_angle, R, T = fit_object(outputs, c, s, item_ids, object_name, bicycle_front, bicycle_back, rot_axis, rot_axis_begin)

        corr_maps.append(outputs)

        images = visualize_bicycle(outputs, images.cpu(), bicycle_front, bicycle_back, bicycle_front_f, bicycle_back_f, 
            rot_axis, rot_axis_begin, rot_angle, R, T, c, s, item_ids, object_name)

        for item_id, image, mask in zip(item_ids, images, masks):
            # print(mask.sum())
            if mask.sum() <= 10:
                continue
            cv2.imwrite('./inference_vis/{}/{}.jpg'.format(object_name, item_id), image[:, :, ::-1])
    # with open('./corr_map.pkl', 'wb') as f:
    #     pickle.dump(corr_maps, f)


if __name__ == '__main__':
    inference()
