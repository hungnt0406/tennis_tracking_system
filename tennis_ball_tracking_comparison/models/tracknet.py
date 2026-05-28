"""
TrackNet: CNN encoder-decoder that takes 3 stacked RGB frames (9 channels)
and outputs a heatmap predicting the ball position.

The network is fully convolutional; the only spatial constraint is that the
input H and W must each be divisible by 16 (four 2× max-pool stages).

Architecture follows the original TrackNet paper (Huang et al., 2019):
VGG-like encoder → symmetric decoder with skip connections.
"""

import numpy as np
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
    Input : (B, 9, H, W)  — 3 RGB frames concatenated channel-wise (H, W % 16 == 0)
    Output: (B, 256, H, W)  — per-pixel intensity-class logits (no sigmoid)
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

        self.head = nn.Conv2d(64, 256, 1)

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

        return self.head(d1)


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


def intensity_to_coords(intensity: torch.Tensor, threshold: float = 128,
                        use_hough: bool = False):
    """Convert a (B, H, W) intensity map in [0, 255] to (B, 2) pixel coords.

    Used for the 256-way classification recipe, where ``intensity`` is the
    per-pixel argmax over the model's 256 logit channels (or a GT class-map).
    Returns (-1, -1) for frames with no ball. Coordinates stay in the resized
    (IMG_H × IMG_W) pixel space.

    use_hough=False: take the global argmax peak per frame; (-1, -1) if the peak
        intensity is below ``threshold``.
    use_hough=True: per-frame cv2.threshold@127 + cv2.HoughCircles (reference
        params); take the first/strongest detected circle's centre, else (-1, -1).
    """
    B, H, W = intensity.shape

    if not use_hough:
        flat = intensity.view(B, -1)
        vals, indices = flat.max(dim=1)
        ys = (indices // W).float()
        xs = (indices % W).float()
        coords = torch.stack([xs, ys], dim=1)
        coords[vals < threshold] = -1.0
        return coords

    import cv2
    maps = intensity.detach().cpu().numpy().astype(np.uint8)
    coords = torch.full((B, 2), -1.0)
    for i in range(B):
        _, binary = cv2.threshold(maps[i], 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                                   param1=50, param2=2, minRadius=2, maxRadius=7)
        if circles is not None:
            # Reference takes the first/strongest circle (HoughCircles orders by
            # accumulator strength) — gating on count would reject every multi-vote
            # case, collapsing recall under minDist=1, param2=2.
            cx, cy = circles[0, 0, 0], circles[0, 0, 1]
            coords[i, 0] = float(cx)
            coords[i, 1] = float(cy)
    return coords
