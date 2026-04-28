"""
TrackNet: CNN encoder-decoder that takes 3 stacked RGB frames (9 channels)
and outputs a 256×256 heatmap predicting the ball position.

Architecture follows the original TrackNet paper (Huang et al., 2019):
VGG-like encoder → symmetric decoder with skip connections.
"""

import torch
import torch.nn as nn


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


class TrackNet(nn.Module):
    """
    Input : (B, 9, 256, 256)  — 3 RGB frames concatenated channel-wise
    Output: (B, 1, 256, 256)  — ball probability heatmap (sigmoid applied)
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = _VGGBlock(9, 64, 2)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc2 = _VGGBlock(64, 128, 2)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc3 = _VGGBlock(128, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc4 = _VGGBlock(256, 512, 3)
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        # Bottleneck
        self.bottleneck = _VGGBlock(512, 512, 3)

        # Decoder (with skip connections via concatenation)
        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = _VGGBlock(1024, 256, 3)

        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = _VGGBlock(512, 128, 3)

        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = _VGGBlock(256, 64, 2)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = _VGGBlock(128, 64, 2)

        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e1p, idx1 = self.pool1(e1)

        e2 = self.enc2(e1p)
        e2p, idx2 = self.pool2(e2)

        e3 = self.enc3(e2p)
        e3p, idx3 = self.pool3(e3)

        e4 = self.enc4(e3p)
        e4p, idx4 = self.pool4(e4)

        b = self.bottleneck(e4p)

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.head(d1))


def heatmap_to_coords(heatmap: torch.Tensor, threshold: float = 0.5):
    """
    Convert a (B, 1, H, W) heatmap to (B, 2) pixel coordinates.
    Returns (-1, -1) for frames where max activation < threshold (ball absent).
    """
    B, _, H, W = heatmap.shape
    flat = heatmap.view(B, -1)
    vals, indices = flat.max(dim=1)
    ys = indices // W
    xs = indices % W
    coords = torch.stack([xs.float(), ys.float()], dim=1)
    absent = vals < threshold
    coords[absent] = -1.0
    return coords
