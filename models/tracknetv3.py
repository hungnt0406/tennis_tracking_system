"""TrackNetV3: tracking module + InpaintNet (1D U-Net) for trajectory rectification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tracknetv2 import TrackNetV2


class TrackNetV3Tracker(TrackNetV2):
    """V3 tracking module: V2 backbone with median background concat."""

    def __init__(self, seq_len: int = 8):
        super().__init__(in_dim=(seq_len + 1) * 3, out_dim=seq_len)
        self.seq_len = seq_len


class _Conv1dBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, num_convs=2):
        layers = []
        for i in range(num_convs):
            layers += [
                nn.Conv1d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
        super().__init__(*layers)


class InpaintNet(nn.Module):
    """
    1D U-Net over time.
    Input : (N, L, 3)  — (x, y, mask) normalized coords + mask channel
    Output: (N, L, 2)  — rectified normalized coords
    """

    def __init__(self):
        super().__init__()
        self.enc1 = _Conv1dBlock(3, 32, 2)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = _Conv1dBlock(32, 64, 2)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = _Conv1dBlock(64, 128, 2)
        self.pool3 = nn.MaxPool1d(2)

        self.bottleneck = _Conv1dBlock(128, 256, 2)

        self.dec3 = _Conv1dBlock(256 + 128, 128, 2)
        self.dec2 = _Conv1dBlock(128 + 64, 64, 2)
        self.dec1 = _Conv1dBlock(64 + 32, 32, 2)

        self.head = nn.Conv1d(32, 2, 1)

    def forward(self, x):
        # x: (N, L, 3) -> (N, 3, L)
        x = x.transpose(1, 2)

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

        out = self.head(d1)  # (N, 2, L)
        return out.transpose(1, 2)
