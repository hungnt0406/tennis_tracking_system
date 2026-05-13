"""
TrackNetV5: Shared-encoder + temporal self-attention at the bottleneck.

Each of the 3 input frames is encoded independently with shared weights. The
three bottleneck feature maps are fused via multi-head self-attention over the
temporal axis (T=3 tokens at each spatial position). The decoder runs once
using the attended middle-frame bottleneck and the middle-frame skip connections.

Input/output contract is identical to TrackNet V1 so heatmap_to_coords works
unchanged.
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


class _TemporalAttention(nn.Module):
    """
    Self-attention over T temporal tokens at each spatial position.

    Input:  list of T tensors, each (B, C, H, W)
    Output: attended middle-frame tensor (B, C, H, W)
    """

    def __init__(self, channels: int = 512, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads,
                                          batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, feats: list, mid_idx: int = 1):
        # feats: T tensors each (B, C, H, W)
        B, C, H, W = feats[0].shape
        T = len(feats)

        # (B, T, C, H*W) → (B*H*W, T, C)
        stacked = torch.stack(feats, dim=1)                    # (B, T, C, H, W)
        stacked = stacked.permute(0, 3, 4, 1, 2)              # (B, H, W, T, C)
        tokens = stacked.reshape(B * H * W, T, C)             # (B*H*W, T, C)

        attended, _ = self.attn(tokens, tokens, tokens)       # (B*H*W, T, C)
        attended = self.norm(attended + tokens)               # residual + norm

        # Extract middle frame and reshape back to spatial
        mid = attended[:, mid_idx, :]                         # (B*H*W, C)
        mid = mid.reshape(B, H, W, C).permute(0, 3, 1, 2)    # (B, C, H, W)
        return mid


class TrackNetV5(nn.Module):
    """
    Input : (B, 9, 256, 256)  — 3 RGB frames concatenated channel-wise
    Output: (B, 1, 256, 256)  — ball probability heatmap (sigmoid applied)
    """

    def __init__(self):
        super().__init__()

        # Shared per-frame encoder (3-channel input per frame)
        self.enc1 = _VGGBlock(3, 64, 2)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc2 = _VGGBlock(64, 128, 2)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc3 = _VGGBlock(128, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc4 = _VGGBlock(256, 512, 3)
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        self.bottleneck = _VGGBlock(512, 512, 3)

        # Temporal fusion at bottleneck
        self.temporal_attn = _TemporalAttention(channels=512, num_heads=8)

        # Decoder (uses middle-frame skip connections)
        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = _VGGBlock(1024, 256, 3)

        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = _VGGBlock(512, 128, 3)

        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = _VGGBlock(256, 64, 2)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = _VGGBlock(128, 64, 2)

        self.head = nn.Conv2d(64, 1, 1)

    def _encode(self, f):
        """Encode a single (B, 3, H, W) frame. Returns skip connections and bottleneck."""
        e1 = self.enc1(f)
        e1p, _ = self.pool1(e1)
        e2 = self.enc2(e1p)
        e2p, _ = self.pool2(e2)
        e3 = self.enc3(e2p)
        e3p, _ = self.pool3(e3)
        e4 = self.enc4(e3p)
        e4p, _ = self.pool4(e4)
        b = self.bottleneck(e4p)
        return (e1, e2, e3, e4), b

    def forward(self, x):
        f1, f2, f3 = x[:, :3], x[:, 3:6], x[:, 6:]

        _, b1 = self._encode(f1)
        (e1m, e2m, e3m, e4m), b2 = self._encode(f2)   # middle frame; keep skips
        _, b3 = self._encode(f3)

        # Temporally attended middle-frame bottleneck
        b_mid = self.temporal_attn([b1, b2, b3], mid_idx=1)

        # Decode with middle-frame skip connections
        d4 = self.up4(b_mid)
        d4 = self.dec4(torch.cat([d4, e4m], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3m], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2m], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1m], dim=1))

        return torch.sigmoid(self.head(d1))
