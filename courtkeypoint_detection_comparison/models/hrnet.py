"""
HRNetPose: HRNet-W32 backbone for court keypoint detection.

timm's features_only=True exposes a standard feature pyramid (not HRNet's internal
high-res branch). Actual channel/stride layout at 640×360 padded to 640×384:
  features[0]: stride 2,  64 ch
  features[1]: stride 4, 128 ch  ← used here (90×160 target after interpolation)
  features[2]: stride 8, 256 ch
  features[3]: stride 16, 512 ch
  features[4]: stride 32, 1024 ch

Input is padded to a multiple of 64 before the backbone to avoid internal size
mismatches in HRNet's multi-scale branch fusion.
"""

import torch.nn as nn
import torch.nn.functional as F
import timm


class HRNetPose(nn.Module):
    def __init__(self, num_channels: int = 15, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            'hrnet_w32', features_only=True, pretrained=pretrained
        )
        # features[1] = stride-4 representation, 128 channels
        self.head = nn.Conv2d(128, num_channels, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        target_h, target_w = H // 4, W // 4
        # Pad to multiple of 64 to avoid size mismatches in HRNet's branch fusion
        pad_h = (64 - H % 64) % 64
        pad_w = (64 - W % 64) % 64
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        features = self.backbone(x)
        # features[1]: stride-4 output, 128 channels
        out = self.head(features[1])
        if out.shape[-2:] != (target_h, target_w):
            out = F.interpolate(out, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return out  # raw logits [B, 15, H//4, W//4]
