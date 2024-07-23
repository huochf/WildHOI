import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from models.condition_flow import ConditionFlow
from datasets.hoi_img_kps_dataset import WildHOIImageDataset, OBJECT_KPS_N
from utils.plot_hoi_kps_image import plot_smpl_keypoints, plot_object_keypoints


class Model(nn.Module):

    def __init__(self, flow_dim, flow_width, c_dim, num_blocks_per_layers, layers, dropout_probability):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        modules.append(nn.Conv2d(resnet.fc.in_features, 2048, kernel_size=1))
        self.backbone = nn.Sequential(*modules)

        self.visual_embedding = nn.Linear(2048, c_dim)

        self.flow = ConditionFlow(dim=flow_dim, hidden_dim=flow_width, c_dim=c_dim, num_blocks_per_layer=num_blocks_per_layers, num_layers=layers, dropout_probability=dropout_probability)


    def image_embedding(self, images):
        return self.backbone(images)


    def log_prob(self, x, visual_feats):
        visual_condition = self.visual_embedding(visual_feats)
        log_prob = self.flow.log_prob(x, condition=visual_condition)
        return log_prob


    def sampling(self, n_samples, visual_feats, z_std=1.):
        visual_condition = self.visual_embedding(visual_feats)
        x, log_prob = self.flow.sampling(n_samples, condition=visual_condition, z_std=z_std)
        return x, log_prob


def get_loss_weights(distances, lambda_=100000, alpha=4):
    distances = lambda_ * distances ** alpha
    return (distances + 1) * torch.exp(- distances)


def sampling_and_visualization(args, images, epoch, idx, model, hoi_kps, visual_feats):
    z_stds = [0, 0.1, 0.2, 0.5, 0.75, 1.0]
    batch_size = visual_feats.shape[0]

    save_dir = os.path.join(args.save_dir, 'visualization_{}'.format(args.object))
    os.makedirs(save_dir, exist_ok=True)
    n_samples = 8
    visual_feats = visual_feats.reshape(batch_size, 1, -1).repeat(1, n_samples, 1).reshape(batch_size * n_samples, -1)
    samples_all = []
    for z_std in z_stds:
        kps_samples, _ = model.sampling(batch_size * n_samples, visual_feats, z_std)

        kps_samples = kps_samples.reshape(batch_size, n_samples, -1, 3)
        kps_samples = torch.cat([hoi_kps.reshape(batch_size, 1, -1, 3), kps_samples], dim=1)
        kps_samples = kps_samples.reshape(batch_size * (n_samples + 1), -1, 3)
        kps_directions = kps_samples[:, :-1]

        y_axis = torch.tensor([0, 1, 0]).reshape(1, 3).float().to(kps_directions.device)
        axis_u = torch.cross(y_axis, kps_directions[:, 0])
        axis_u = axis_u / torch.norm(axis_u, dim=-1, keepdim=True)
        axis_v = torch.cross(axis_u, kps_directions[:, 0])
        axis_v = axis_v / torch.norm(axis_v, dim=-1, keepdim=True)

        u = (kps_directions * axis_u.reshape(batch_size * (n_samples + 1), 1, 3)).sum(2)
        v = (kps_directions * axis_v.reshape(batch_size * (n_samples + 1), 1, 3)).sum(2)
        kps2d_vis = torch.stack([u, v], dim=2)

        f_fix = 5000 / 256
        kps2d_vis = kps2d_vis * f_fix # [1, -1]
        kps2d_vis = kps2d_vis.detach().cpu().numpy()
        res = 256
        for i in range(batch_size * (n_samples + 1)):
            smpl_kps = (kps2d_vis[i, :22] + 1) / 2 * res
            object_kps = (kps2d_vis[i, 22:] + 1) / 2 * res
            image_vis = (np.ones((res, res, 3)) * 255).astype(np.uint8)
            if not np.isnan(smpl_kps).any():
                image_vis = plot_smpl_keypoints(image_vis, smpl_kps)
            if not np.isnan(object_kps).any():
                image_vis = plot_object_keypoints(image_vis, object_kps, args.object)
            samples_all.append(image_vis)

    samples_all = np.array(samples_all).reshape(len(z_stds), batch_size, (n_samples + 1), res, res, 3)
    samples_all = samples_all.transpose(1, 0, 3, 2, 4, 5).reshape(batch_size, len(z_stds) * res, (n_samples + 1) * res, 3)

    images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
    images = (images * std) + mean
    images = (images * 255).astype(np.uint8)[:, :, :, ::-1]
    images = images.reshape(batch_size, 1, res, res, 3).repeat(len(z_stds), axis=1).reshape(batch_size, len(z_stds) * res, res, 3)
    image_vis = np.concatenate([images, samples_all], axis=2)

    for i in range(min(batch_size, 8)):
        cv2.imwrite(os.path.join(save_dir, '{:03d}_{:02d}_{:01d}.jpg'.format(epoch, idx, i)), image_vis[i])


def train(args):
    device = torch.device('cuda')
    batch_size = args.batch_size
    output_dir = args.save_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset_train = WildHOIImageDataset(args.root_dir, args.object, args.fps, args.k, split='train')
    dataset_test = WildHOIImageDataset(args.root_dir, args.object, args.fps, args.k, split='test')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=8, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, num_workers=2, shuffle=True)

    flow_dim = (22 + OBJECT_KPS_N[args.object] + 1) * 3
    model = Model(flow_dim=flow_dim, 
                      flow_width=args.flow_width, 
                      c_dim=args.c_dim,
                      num_blocks_per_layers=args.num_blocks_per_layers, 
                      layers=args.layers,
                      dropout_probability=args.dropout_probability)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if os.path.exists(os.path.join(output_dir, 'checkpoint_{}.pth'.format(args.object))):
        state_dict = torch.load(os.path.join(output_dir, 'checkpoint_{}.pth'.format(args.object)))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        begin_epoch = state_dict['epoch']
    else:
        begin_epoch = 0

    f_log = open(os.path.join(output_dir, 'logs_{}.txt'.format(args.object)), 'a')

    for epoch in range(begin_epoch, args.epoch):
        model.train()

        for idx, item in enumerate(dataloader_train):
            image, hoi_kps, nn_kps, distances = item
            image = image.float().to(device)
            hoi_kps = hoi_kps.float().to(device)
            nn_kps = nn_kps.float().to(device)
            distances = distances.float().to(device)

            batch_size, k, _, _ = nn_kps.shape
            hoi_kps = torch.cat([hoi_kps.unsqueeze(1), nn_kps], dim=1)
            hoi_kps = hoi_kps.reshape(batch_size * (k + 1), -1)

            self_distance = torch.zeros(batch_size, 1).float().to(device)
            distances = torch.cat([self_distance, distances], dim=1)

            hoi_kps = hoi_kps + 0.001 * torch.randn_like(hoi_kps)

            visual_feats = model.image_embedding(image)
            visual_feats = visual_feats.reshape(batch_size, 1, -1).repeat(1, k + 1, 1).reshape(batch_size * (k + 1), -1)

            log_prob = model.log_prob(hoi_kps, visual_feats)
            log_prob = log_prob.reshape(batch_size, k + 1)
            loss_weights = get_loss_weights(distances)
            log_prob = loss_weights * log_prob

            loss_nll = - log_prob.sum() / loss_weights.sum()
            loss = loss_nll

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if idx % 10 == 0:
                log_str = '[{} / {}] Loss: {:.4f}, Loss_nll: {:.4f}'.format(
                    epoch, idx, loss.item(), loss_nll.item())
                print(log_str)
                sys.stdout.flush()
                f_log.write(log_str + '\n')

            if idx % 1000 == 0:
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(output_dir, 'checkpoint_{}.pth'.format(args.object)))

        model.eval()
        for idx, item in enumerate(dataloader_test):
            if idx > 10:
                break
            image, hoi_kps = item
            image = image.float().to(device)
            hoi_kps = hoi_kps.float().to(device)

            batch_size = image.shape[0]
            visual_feats = model.image_embedding(image)
            visual_feats = visual_feats.reshape(batch_size, -1)

            sampling_and_visualization(args, image, epoch, idx, model, hoi_kps, visual_feats)

    f_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate KPS (BEHAVE)')
    parser.add_argument('--root_dir', default='/storage/data/huochf/HOIYouTube', type=str)
    parser.add_argument('--epoch', default=999999, type=int)
    parser.add_argument('--fps', default=3, type=int)
    parser.add_argument('--k', default=16, type=int)
    parser.add_argument('--object', default='barbell', type=str)
    parser.add_argument('--flow_width', default=512, type=int)
    parser.add_argument('--num_blocks_per_layers', default=2, type=int)
    parser.add_argument('--layers', default=4, type=int)
    parser.add_argument('--c_dim', default=256, type=int)
    parser.add_argument('--dropout_probability', default=0., type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--save_dir', default='./outputs/cflow_pseudo_kps2d')

    args = parser.parse_args()

    train(args)
