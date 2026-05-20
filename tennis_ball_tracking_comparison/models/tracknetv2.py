"""TrackNetV2: 3-encoder VGG-style U-Net, MIMO heatmap output."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _VGGBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, num_convs=2):
        layers = []
        for i in range(num_convs):
            layers += [
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        super().__init__(*layers)


class TrackNetV2(nn.Module):
    """
    Input : (B, in_dim, H, W)
    Output: (B, out_dim, H, W) sigmoid heatmap stack
    """

    def __init__(self, in_dim: int = 9, out_dim: int = 3):
        super().__init__()

        self.enc1 = _VGGBlock(in_dim, 64, 2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = _VGGBlock(64, 128, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = _VGGBlock(128, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck = _VGGBlock(256, 512, 3)

        self.dec3 = _VGGBlock(512 + 256, 256, 3)
        self.dec2 = _VGGBlock(256 + 128, 128, 2)
        self.dec1 = _VGGBlock(128 + 64, 64, 2)

        self.head = nn.Conv2d(64, out_dim, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e1p = self.pool1(e1)

        e2 = self.enc2(e1p)
        e2p = self.pool2(e2)

        e3 = self.enc3(e2p)
        e3p = self.pool3(e3)

        b = self.bottleneck(e3p)

        d3 = F.interpolate(b, scale_factor=2, mode='nearest')
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, scale_factor=2, mode='nearest')
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.head(d1))
