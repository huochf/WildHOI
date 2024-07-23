import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .backbone import DINOv2
from .dense_corr_header import DenseCorrHeader


class Model(nn.Module):

    def __init__(self, num_kps=16):
        super(Model, self).__init__()
        self.backbone = DINOv2()
        self.dense_corr_header = DenseCorrHeader(in_channels=self.backbone.dinov2.num_features, 
                                                 num_layers=2,
                                                 output_dim=4)
        self.BCEobj = nn.BCEWithLogitsLoss()
        self.BCEcls = nn.BCEWithLogitsLoss()

        param_dicts = [
            {"params": list(self.backbone.parameters()), "lr": 1e-4},
            {"params": list(self.dense_corr_header.parameters())},
        ]
        self.optimizer = torch.optim.AdamW(param_dicts, lr=1e-3,)


    def forward(self, images):
        # images: [b, 224, 224, 3]
        features = self.backbone(images) # [b, c, 16, 16]
        corr_maps = self.dense_corr_header(features) # [b, c, 64, 64]

        return corr_maps


    def compute_loss(self, outputs, targets):
        corr_maps = outputs
        corr_gt, corr_sym_gt, masks = targets
        
        bs, _, corr_h, corr_w = corr_maps.shape
        corr_gt = torch.stack([corr_gt, corr_sym_gt], dim=1) # [b, 2, 3, h, w]

        _corr_maps = corr_maps.unsqueeze(1).repeat(1, 2, 1, 1, 1)
        corr_cost = F.l1_loss(corr_gt, _corr_maps[:, :, :3], reduction='none')
        corr_cost = corr_cost * masks.reshape(bs, 1, 1, corr_h, corr_w)
        corr_cost = corr_cost.reshape(bs, 2, -1).mean(-1)
        _, indices = corr_cost.min(-1)

        corr_gt = corr_gt[range(bs), indices]

        if corr_gt.sum() == 0:
            loss_corr = 0 * corr_gt.sum()
        else:
            loss_corr = F.l1_loss(corr_gt[corr_gt !=0 ], corr_maps[:, :3][corr_gt != 0])
        loss_mask = F.l1_loss(masks, corr_maps[:, 3])

        loss = loss_corr + loss_mask
        loss_dict = {'loss': loss, 'loss_corr': loss_corr, 'loss_mask': loss_mask}
        
        return loss, loss_dict


    def train_step(self, images, gts):
        self.optimizer.zero_grad()
        model_outputs = self.forward(images)
        loss, loss_dict = self.compute_loss(model_outputs, gts)
        loss.backward()
        self.optimizer.step()
        return loss_dict


    def inference_step(self, images):
        corr_maps = self.forward(images)

        return corr_maps.detach().cpu().numpy()


    def save_checkpoint(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'backbone': self.backbone.state_dict(),
            'corr_header': self.dense_corr_header.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print('Save checkpoint to {}.'.format(path))


    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.backbone.load_state_dict(checkpoint['backbone'])
        self.dense_corr_header.load_state_dict(checkpoint['corr_header'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint['epoch']
