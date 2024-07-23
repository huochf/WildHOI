# from CDPN
import torch
import torch.nn as nn


class DenseCorrHeader(nn.Module):

    def __init__(self, in_channels, num_layers, num_filters=256, output_dim=3):
        super(DenseCorrHeader, self).__init__()

        kernel_size = 3
        padding = 1
        output_padding = 1

        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(
                nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                                   output_padding=output_padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

            self.features.append(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

            self.features.append(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

        self.features.append(
            nn.Conv2d(num_filters, output_dim, kernel_size=1, padding=0, bias=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)


    def forward(self, x):
        for i, l in enumerate(self.features):
            x = l(x)
        return x
