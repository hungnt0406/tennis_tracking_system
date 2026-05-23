"""
MobileNetV3SmallPose: MobileNetV3-Small backbone + U-Net-style decoder
producing full-resolution (stride 1) heatmaps.

torchvision's ``mobilenet_v3_small.features`` layout (output stride / channels):

    features[0]      stride  2, 16  ch   (stem ConvBNActivation, s=2)
    features[1]      stride  4, 16  ch   (bneck s=2)
    features[2..3]   stride  8, 24  ch
    features[4..8]   stride 16, 48  ch
    features[9..11]  stride 32, 96  ch
    features[12]     stride 32, 576 ch   (classifier-specific 1×1 expand; dropped)

We tap one feature map per stride level and feed them as skip connections into
a U-Net decoder of ``dec_ch`` channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# (last_layer_index_inclusive, output_stride, channels) per stage.
_STAGE_CONFIG = [
    (0,   2,  16),
    (1,   4,  16),
    (3,   8,  24),
    (8,  16,  48),
    (11, 32,  96),
]


class _UpBlock(nn.Module):
    """Upsample to skip's spatial size, concatenate, then refine with 2× 3x3 conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class _UpBlockNoSkip(nn.Module):
    """Upsample to an explicit target size, refine with a single 3x3 conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.conv(x)


class MobileNetV3SmallPose(nn.Module):
    """MobileNetV3-Small encoder + U-Net decoder, full-resolution heatmap output."""

    def __init__(self, num_channels: int = 15, dec_ch: int = 64, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        features = list(backbone.features)

        # Split features into per-stride-level stages.
        prev = 0
        stages = []
        for end_idx, _, _ in _STAGE_CONFIG:
            stages.append(nn.Sequential(*features[prev:end_idx + 1]))
            prev = end_idx + 1
        self.stages = nn.ModuleList(stages)

        ch_s2, ch_s4, ch_s8, ch_s16, ch_s32 = [c for _, _, c in _STAGE_CONFIG]

        self.up32_to_16 = _UpBlock(ch_s32, ch_s16, dec_ch)
        self.up16_to_8  = _UpBlock(dec_ch, ch_s8,  dec_ch)
        self.up8_to_4   = _UpBlock(dec_ch, ch_s4,  dec_ch)
        self.up4_to_2   = _UpBlock(dec_ch, ch_s2,  dec_ch)
        self.up2_to_1   = _UpBlockNoSkip(dec_ch, dec_ch)
        self.head = nn.Conv2d(dec_ch, num_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]

        skips = []
        cur = x
        for stage in self.stages:
            cur = stage(cur)
            skips.append(cur)
        # skips: [s2, s4, s8, s16, s32]

        d = self.up32_to_16(skips[4], skips[3])
        d = self.up16_to_8(d, skips[2])
        d = self.up8_to_4(d, skips[1])
        d = self.up4_to_2(d, skips[0])
        d = self.up2_to_1(d, (H, W))

        return self.head(d)  # raw logits, [B, num_channels, H, W]
