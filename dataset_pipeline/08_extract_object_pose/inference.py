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


def visualize(outputs, images):
    corr_maps_pred, kpts_pred, kpts_score_pred, kpts_label_pred = outputs
    bs, _, h, w = images.shape
    images = images.numpy()
    std = np.array((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
    mean = np.array((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
    images = (images * std + mean) * 255
    images = images.clip(0, 255).astype(np.uint8)

    image_show = np.ones((h * 4, w * 2, 3), dtype=np.uint8) * 255
    for b_idx in range(min(bs, 4)):
        image = images[b_idx].transpose(1, 2, 0)

        image_kps_pred = plot_keypoints(image.copy(), kpts_pred[b_idx], kpts_label_pred[b_idx])
        image_corr_pred = corr_maps_pred[b_idx].transpose(1, 2, 0)
        image_corr_pred = ((image_corr_pred + 1) / 2 * 255.).clip(0, 255).astype(np.uint8)
        image_corr_pred = cv2.resize(image_corr_pred, dsize=(w, h))

        image_show[h * b_idx: h*b_idx + h, 0:w] = image_kps_pred
        image_show[h * b_idx: h*b_idx + h, w:w*2] = image_corr_pred
    return image_show


def visualize2(outputs, images, object_v, object_f, R, T, center, s, item_ids):
    corr_maps_pred, kpts_pred, kpts_score_pred, kpts_label_pred = outputs
    bs, _, h, w = images.shape
    images = images.numpy()
    std = np.array((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
    mean = np.array((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
    images = (images * std + mean) * 255
    images = images.clip(0, 255).astype(np.uint8)

    image_show = np.ones((h * 4, w * 4, 3), dtype=np.uint8) * 255
    for b_idx in range(min(bs, 4)):
        image = images[b_idx].transpose(1, 2, 0)

        image_kps_pred = plot_keypoints(image.copy(), kpts_pred[b_idx], kpts_label_pred[b_idx])
        image_corr_pred = corr_maps_pred[b_idx, :3].transpose(1, 2, 0)
        image_corr_pred = ((image_corr_pred + 1) / 2 * 255.).clip(0, 255).astype(np.uint8)
        image_corr_pred = cv2.resize(image_corr_pred, dsize=(w, h))

        mask = corr_maps_pred[b_idx, 3]
        mask = np.stack([mask, mask, mask], axis=2)
        mask = (mask * 255.).clip(0, 255).astype(np.uint8)
        mask = cv2.resize(mask, dsize=(w, h))

        item_id = item_ids[b_idx]
        video_id = item_id[:5]
        seg_id = item_id[6:-6]
        frame_id = item_id[-5:]
        image_org = cv2.imread(os.path.join('/data/HOISports/skateboard/images', video_id, seg_id, '{}.jpg'.format(frame_id)))

        image_org_render_pose = render(image_org.copy(), object_v, object_f, 1000, center[b_idx, 0], center[b_idx, 1], T[b_idx], R[b_idx])
        image_render_pose, _ = generate_image_patch(image_org_render_pose, center[b_idx, 0], center[b_idx, 1], s[b_idx], h, 0, None)

        image_show[h * b_idx: h*b_idx + h, 0:w] = image_kps_pred
        image_show[h * b_idx: h*b_idx + h, w:w*2] = image_render_pose[:, :, ::-1]
        image_show[h * b_idx: h*b_idx + h, w*2:w*3] = image_corr_pred
        image_show[h * b_idx: h*b_idx + h, w*3:w*4] = mask
    return image_show


def visualize3(outputs, images, object_v, object_f, R, T, center, s, item_ids, object_name):
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

        if T[b_idx].sum() != 0:
            image_org_render_pose = render(image_org.copy(), object_v, object_f, 1000, center[b_idx, 0], center[b_idx, 1], T[b_idx], R[b_idx])
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


def fit_object(model_outputs, c, s, item_ids, object_name):
    device = torch.device('cuda')
    object_mesh = trimesh.load('../data/objects/{}.ply'.format(object_name), process=False)
    object_v = torch.tensor(np.array(object_mesh.vertices), dtype=torch.float32).to(device)

    corr_maps_pred = model_outputs

    b, _, corr_h, corr_w = corr_maps_pred.shape

    corr_maps_pred = torch.tensor(corr_maps_pred)
    coor_maps = corr_maps_pred.permute(0, 2, 3, 1)
    corr_norm = CORR_NORM[object_name]
    coor_x3d = coor_maps[:, :, :, :3] * torch.tensor(corr_norm).reshape(1, 1, 1, 3)
    coor_x3d = coor_x3d.float().to(device)

    coor_mask = coor_maps[:, :, :, 3:]
    coor_mask = coor_mask.float().to(device)

    n_init = 128
    b = corr_maps_pred.shape[0]
    R_init = compute_random_rotations(b * n_init)
    R6d = matrix_to_rotation_6d(R_init).to(torch.float32).to(device)
    R6d = R6d.reshape(b, n_init, 6)
    R6d = nn.Parameter(R6d)
    # T = torch.tensor([0, 0, 5], dtype=torch.float32).to(device)
    x = torch.rand(n_init) * 2 - 1
    y = torch.rand(n_init) * 2 - 1
    z = torch.rand(n_init) * 10 + 2
    T = torch.stack([x, y, z], dim=1)
    T = T.unsqueeze(0).repeat(b, 1, 1).reshape(b, n_init, 3).to(device)
    T = nn.Parameter(T)

    weight_f = lambda cst, it: 1. * cst / (1 + it)
    optimizer = torch.optim.Adam([R6d, T], 0.05, betas=(0.9, 0.999))
    iteration = 3
    steps_per_iter = 1000

    grid_2d = torch.arange(corr_h).float().to(device)
    ys, xs = torch.meshgrid(grid_2d, grid_2d) # (h, w)
    grid_2d = torch.stack([xs, ys], dim=2).unsqueeze(0).repeat(b, 1, 1, 1).reshape(b, -1, 2) # (b, h * w, 2)
    stride = s / corr_h
    stride = stride.reshape(b, 1, 1).float().to(device)
    x1 = c[:, 0] - s / 2
    y1 = c[:, 1] - s / 2
    begin_point = torch.stack([x1, y1], dim=1)
    begin_point = begin_point.reshape(b, 1, 2).float().to(device)
    coor_x2d = grid_2d * stride + begin_point
    coor_x2d = coor_x2d.unsqueeze(1).repeat(1, n_init, 1, 1) # [b, n_init, h*w, 2]

    coor_x3d = coor_x3d.reshape(b, -1, 3).unsqueeze(1).repeat(1, n_init, 1, 1) # [b, n_init, h*w, 3]
    coor_mask = coor_mask.reshape(b, -1).unsqueeze(1).repeat(1, n_init, 1).clamp(0, 1)

    cx = c[:, 0].float().to(device).reshape(b, 1, 1)
    cy = c[:, 1].float().to(device).reshape(b, 1, 1)
    f = 1000
    for it in range(iteration):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Fitting object.')
        for i in loop:
            optimizer.zero_grad()

            rotmat = rotation_6d_to_matrix(R6d)
            trans = T.reshape(b, n_init, 1, 3)
            coor_x3d_reproj = torch.matmul(coor_x3d, rotmat.transpose(3, 2)) + trans
            u = coor_x3d_reproj[:, :, :, 0] / (coor_x3d_reproj[:, :, :, 2] + 1e-8) * f + cx
            v = coor_x3d_reproj[:, :, :, 1] / (coor_x3d_reproj[:, :, :, 2] + 1e-8) * f + cy
            coor_x2d_reproj = torch.stack([u, v], dim=3)

            loss_coor = ((coor_x2d_reproj - coor_x2d) ** 2).sum(-1)
            loss_coor = loss_coor * coor_mask
            loss_coor = loss_coor.mean(-1) # (b, n_init)

            loss = weight_f(loss_coor.mean(), it)

            loss.backward()
            optimizer.step()

            indices = torch.argmin(loss_coor, dim=1)
            min_loss = weight_f(loss_coor[range(b), indices].mean(), it)

            loop.set_description('Iter: {}, Loss: {:0.4f}, min_loss: {:0.4f}, loss_coor: {:0.4f}'.format(
                i, loss.item(), min_loss.item(), loss_coor.mean().item()))

    indices = torch.argmin(loss_coor, dim=1)
    R = rotation_6d_to_matrix(R6d[range(b), indices]).detach().cpu().numpy()
    T = T[range(b), indices].detach().cpu().numpy()

    return R, T


def fit_object2(model_outputs, c, s, item_ids, object_name):
    device = torch.device('cuda')
    if object_name in ['cello', 'violin']:
        object_mesh = trimesh.load('../data/objects/{}_body.ply'.format(object_name), process=False)
    else:
        object_mesh = trimesh.load('../data/objects/{}.ply'.format(object_name), process=False)
    object_v = torch.tensor(np.array(object_mesh.vertices), dtype=torch.float32).to(device)

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

    weight_f = lambda cst, it: 1. * cst / (1 + it)
    optimizer = torch.optim.Adam([R6d, T], 0.05, betas=(0.9, 0.999))
    iteration = 0
    steps_per_iter = 1000
    for it in range(iteration):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Fitting object.')
        for i in loop:
            optimizer.zero_grad()

            rotmat = rotation_6d_to_matrix(R6d)
            trans = T.reshape(b, 1, 3)
            coor_x3d_reproj = torch.matmul(coor_x3d, rotmat.transpose(2, 1)) + trans
            u = coor_x3d_reproj[:, :, 0] / (coor_x3d_reproj[:, :, 2] + 1e-8) * f + cx
            v = coor_x3d_reproj[:, :, 1] / (coor_x3d_reproj[:, :, 2] + 1e-8) * f + cy
            coor_x2d_reproj = torch.stack([u, v], dim=2)

            loss_coor = ((coor_x2d_reproj - coor_x2d) ** 2).sum(-1)
            loss_coor = loss_coor * coor_mask
            loss_coor = loss_coor.mean(-1) # (b, n_init)

            loss = weight_f(loss_coor.mean(), it)

            loss.backward()
            optimizer.step()

            loop.set_description('Iter: {}, Loss: {:0.4f}, loss_coor: {:0.4f}'.format(
                i, loss.item(), loss_coor.mean().item()))

    R = rotation_6d_to_matrix(R6d).detach().cpu().numpy()
    T = T.detach().cpu().numpy()

    return R, T


def inference():
    device = torch.device('cuda')
    object_name = 'violin'

    os.makedirs('./inference_vis/{}'.format(object_name), exist_ok=True)

    if object_name in ['cello', 'violin']:
        object_mesh = trimesh.load('../data/objects/{}_body.ply'.format(object_name), process=False)
    else:
        object_mesh = trimesh.load('../data/objects/{}.ply'.format(object_name), process=False)
    object_v = torch.tensor(np.array(object_mesh.vertices), dtype=torch.float32)
    object_f = torch.tensor(np.array(object_mesh.faces, dtype=np.int64))
    object_v = object_v - object_v.mean(0).reshape(1, -1)

    with open('/storage/data/huochf/HOIYouTube/train_test_split_{}.json'.format(object_name), 'r') as f:
        train_test_split = json.load(f)

    model = Model(num_kps=12).to(device)
    dataset = ObjectImageDataset(root_dir='/storage/data/huochf/HOIYouTube/{}'.format(object_name), video_ids=train_test_split['train'], out_res=224, coor_res=64)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=8, shuffle=True)

    model.load_checkpoint('./weights/model_{}_stage1.pth'.format(object_name))

    model.eval()
    corr_maps = []
    for idx, data in enumerate(tqdm(dataloader)):
        # if idx > 2000:
        #     break
        images, masks, c, s, item_ids = data
        images = images.to(device)

        outputs = model.inference_step(images)
        R, T = fit_object2(outputs, c, s, item_ids, object_name)

        corr_maps.append(outputs)

        images = visualize3(outputs, images.cpu(), object_v, object_f, R, T, c, s, item_ids, object_name)

        for item_id, image, mask in zip(item_ids, images, masks):
            # print(mask.sum())
            if mask.sum() <= 10:
                continue
            cv2.imwrite('./inference_vis/{}/{}.jpg'.format(object_name, item_id), image[:, :, ::-1])
    # with open('./corr_map.pkl', 'wb') as f:
    #     pickle.dump(corr_maps, f)


if __name__ == '__main__':
    inference()
