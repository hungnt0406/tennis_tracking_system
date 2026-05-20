"""
TrackNetCourt: VGG-style encoder-decoder adapted for court keypoint detection.

Derived from TrackNet (Huang et al., 2019). Key differences from the ball-tracking
variant:
- Input is a single RGB frame (3 channels) instead of a 3-frame stack (9 channels).
- Output is 15 channels (14 keypoints + court center) instead of a 1-channel heatmap.
- No sigmoid in forward() — raw logits are returned; the training loop applies
  torch.sigmoid inside the loss.
- Designed for 640×360 input. Because 360 is not divisible by 32 (5 pooling steps),
  F.interpolate is used after each decoder upsample to guarantee the skip-connection
  spatial sizes match, and the final output is interpolated to exactly (H, W).
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


class TrackNetCourt(nn.Module):
    """
    Input : (B, 3, H, W)   — single RGB frame; designed for 640×360
    Output: (B, 15, H, W)  — raw logits for 14 keypoints + court center
                             (no sigmoid applied; caller handles activation)
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = _VGGBlock(3, 64, 2)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc2 = _VGGBlock(64, 128, 2)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc3 = _VGGBlock(128, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc4 = _VGGBlock(256, 512, 3)
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        # Bottleneck (5th pooling step lives implicitly here: pool4 brings stride to 32)
        self.bottleneck = _VGGBlock(512, 512, 3)

        # Decoder (skip connections via concatenation)
        # ConvTranspose2d doubles spatial size; F.interpolate corrects any off-by-one
        # mismatches introduced when the input height/width are not divisible by 32.
        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = _VGGBlock(1024, 256, 3)

        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = _VGGBlock(512, 128, 3)

        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = _VGGBlock(256, 64, 2)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = _VGGBlock(128, 64, 2)

        # 15-channel output head: 14 keypoints + court center
        self.head = nn.Conv2d(64, 15, 1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        e1 = self.enc1(x)
        e1p, idx1 = self.pool1(e1)

        e2 = self.enc2(e1p)
        e2p, idx2 = self.pool2(e2)

        e3 = self.enc3(e2p)
        e3p, idx3 = self.pool3(e3)

        e4 = self.enc4(e3p)
        e4p, idx4 = self.pool4(e4)

        b = self.bottleneck(e4p)

        # Decoder: upsample then align to the matching encoder feature map size
        d4 = self.up4(b)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.head(d1)
        # Final interpolate ensures output matches input resolution exactly,
        # guarding against any accumulated rounding across pooling steps.
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out  # raw logits, shape [B, 15, H, W]
