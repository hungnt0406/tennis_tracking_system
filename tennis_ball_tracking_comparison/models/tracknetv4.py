"""
TrackNetV4: TrackNet V1 backbone with a Motion Attention Module (MAM).

The MAM computes inter-frame differences to produce a spatial attention map
that is fused into the two highest-resolution decoder stages (dec1, dec2).
Input/output contract is identical to TrackNet V1 so heatmap_to_coords works
unchanged.

Reference design: motion-aware fusion on top of the VGG U-Net backbone.
"""

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


class MotionAttentionModule(nn.Module):
    """
    Produces a (B, 1, H, W) spatial attention map from consecutive frame diffs.
    Input: (B, 9, H, W) — 3 stacked RGB frames (f1 | f2 | f3).
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        f1, f2, f3 = x[:, :3], x[:, 3:6], x[:, 6:]
        m1 = (f2 - f1).abs()
        m2 = (f3 - f2).abs()
        return self.conv(torch.cat([m1, m2], dim=1))  # (B, 1, H, W)


class TrackNetV4(nn.Module):
    """
    Input : (B, 9, H, W)  — 3 RGB frames concatenated channel-wise (H, W % 16 == 0)
    Output: (B, 1, H, W)  — ball probability heatmap (sigmoid applied)
    """

    def __init__(self):
        super().__init__()

        self.mam = MotionAttentionModule()

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
        # Motion attention at full input resolution (B, 1, H, W)
        attn_full = self.mam(x)

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
        # Apply motion attention at half resolution
        attn_half = F.interpolate(attn_full, size=d2.shape[2:], mode="bilinear",
                                  align_corners=False)
        d2 = d2 * attn_half

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        # Apply motion attention at full resolution
        d1 = d1 * attn_full

        return torch.sigmoid(self.head(d1))
