"""Image preprocessing and augmentation utilities."""

import random
import numpy as np
import cv2


IMG_H = 256
IMG_W = 256


def resize_frame(img: np.ndarray, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def normalize(img: np.ndarray) -> np.ndarray:
    """Convert uint8 HWC image to float32 CHW in [0, 1]."""
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)


def make_gaussian_heatmap(x: float, y: float, orig_w: int, orig_h: int,
                           out_h: int = IMG_H, out_w: int = IMG_W,
                           sigma: float = 5.0) -> np.ndarray:
    """Generate a 2-D Gaussian heatmap at the (x, y) ball position."""
    heatmap = np.zeros((out_h, out_w), dtype=np.float32)
    if x < 0 or y < 0:
        return heatmap

    cx = int(x / orig_w * out_w)
    cy = int(y / orig_h * out_h)

    size = int(6 * sigma + 1)
    half = size // 2
    x_range = np.arange(-half, half + 1)
    gaussian_1d = np.exp(-0.5 * (x_range / sigma) ** 2)
    kernel = np.outer(gaussian_1d, gaussian_1d)
    kernel /= kernel.max()

    x0, x1 = cx - half, cx + half + 1
    y0, y1 = cy - half, cy + half + 1
    kx0 = max(0, -x0)
    ky0 = max(0, -y0)
    kx1 = size - max(0, x1 - out_w)
    ky1 = size - max(0, y1 - out_h)
    x0, x1 = max(0, x0), min(out_w, x1)
    y0, y1 = max(0, y0), min(out_h, y1)

    if x0 < x1 and y0 < y1:
        heatmap[y0:y1, x0:x1] = kernel[ky0:ky1, kx0:kx1]
    return heatmap


def coords_to_yolo(x: float, y: float, orig_w: int, orig_h: int,
                   ball_radius: int = 10) -> tuple:
    """Convert ball centre (x, y) to YOLO normalised [cx, cy, bw, bh]."""
    cx = x / orig_w
    cy = y / orig_h
    bw = (2 * ball_radius) / orig_w
    bh = (2 * ball_radius) / orig_h
    return cx, cy, bw, bh


class Augmenter:
    """Random augmentations applied consistently to frame(s) and coordinates."""

    def __init__(self, brightness=0.3, contrast=0.3, hflip_prob=0.5, enabled=True):
        self.brightness = brightness
        self.contrast = contrast
        self.hflip_prob = hflip_prob
        self.enabled = enabled

    def __call__(self, frames: list, x: float, y: float, orig_w: int):
        """
        frames: list of HWC uint8 numpy arrays
        Returns: (augmented_frames, new_x, new_y)
        """
        if not self.enabled:
            return frames, x, y

        do_flip = random.random() < self.hflip_prob
        b_delta = random.uniform(-self.brightness, self.brightness)
        c_factor = random.uniform(1 - self.contrast, 1 + self.contrast)

        augmented = []
        for f in frames:
            f = f.astype(np.float32)
            f = f * c_factor + b_delta * 255
            f = np.clip(f, 0, 255).astype(np.uint8)
            if do_flip:
                f = cv2.flip(f, 1)
            augmented.append(f)

        new_x = x
        if do_flip and x >= 0:
            new_x = orig_w - 1 - x

        return augmented, new_x, y
