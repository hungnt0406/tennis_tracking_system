"""
ResNet50Pose: SimpleBaselines-style decoder on top of a pretrained ResNet-50 backbone.

At 640×360 input:
  stem + layer1-4 reduce to stride 32  →  20×11  (640/32=20, 360/32=11.25 → floor=11)
  decoder (3× ConvTranspose2d stride-2) brings to stride 4  →  160×90
  head outputs [B, 15, 160, 90]

Both 640/4=160 and 360/4=90 are exact integers, so the stride-4 output is clean.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Pose(nn.Module):
    def __init__(self, num_channels: int = 15):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Keep all layers up to layer4, strip avgpool and fc
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # stride 4,  256 ch
        self.layer2 = backbone.layer2  # stride 8,  512 ch
        self.layer3 = backbone.layer3  # stride 16, 1024 ch
        self.layer4 = backbone.layer4  # stride 32, 2048 ch

        # SimpleBaselines-style decoder: 3 transposed conv blocks
        # Each block: ConvTranspose2d(in, 256, 4, stride=2, padding=1) + BN + ReLU
        # stride 32 → 16 → 8 → 4
        self.decoder = nn.Sequential(
            *self._deconv_block(2048, 256),
            *self._deconv_block(256, 256),
            *self._deconv_block(256, 256),
        )
        self.head = nn.Conv2d(256, num_channels, 1)

    def _deconv_block(self, in_ch, out_ch):
        return [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]

    def forward(self, x):
        target_h, target_w = x.shape[2] // 4, x.shape[3] // 4
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.decoder(x)
        x = self.head(x)
        # Guarantee stride-4 output regardless of intermediate rounding
        if x.shape[-2:] != (target_h, target_w):
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return x  # raw logits, shape [B, 15, H//4, W//4]
