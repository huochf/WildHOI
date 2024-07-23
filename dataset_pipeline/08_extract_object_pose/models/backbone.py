import torch
import torch.nn as nn


class DINOv2(nn.Module):

    def __init__(self, ):
        super(DINOv2, self).__init__()
        self.dinov2 = torch.hub.load(repo_or_dir='/public/home/huochf/.cache/torch/hub/facebookresearch_dinov2_main/', 
            model='dinov2_vitl14', trust_repo=True, source='local')


    def forward(self, image):
        b, _, h, w = image.shape
        features = self.dinov2.forward_features(image)['x_norm_patchtokens']
        features = features.reshape(b, h // 14, w // 14, -1).permute(0, 3, 1, 2)
        return features
