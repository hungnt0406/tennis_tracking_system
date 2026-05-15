"""Preprocessing helpers for TrackNetV2/V3 (288x512 resolution)."""

import os
import random
import numpy as np
import cv2


IMG_H_V2 = 288
IMG_W_V2 = 512
SIGMA_V2 = 2.5


def resize_v2(frame_bgr: np.ndarray, h: int = IMG_H_V2, w: int = IMG_W_V2) -> np.ndarray:
    """Resize HWC uint8 frame to (h, w) float32 in [0, 1]."""
    resized = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) / 255.0


def make_gaussian_heatmap_v2(x: float, y: float, w: int = IMG_W_V2, h: int = IMG_H_V2,
                              sigma: float = SIGMA_V2) -> np.ndarray:
    heatmap = np.zeros((h, w), dtype=np.float32)
    if x < 0 or y < 0:
        return heatmap

    cx = int(round(x))
    cy = int(round(y))

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
    kx1 = size - max(0, x1 - w)
    ky1 = size - max(0, y1 - h)
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)

    if x0 < x1 and y0 < y1:
        heatmap[y0:y1, x0:x1] = kernel[ky0:ky1, kx0:kx1]
    return heatmap


def compute_clip_median(frame_paths: list, cache_dir: str = None,
                         clip_key: str = None, max_frames: int = 50) -> np.ndarray:
    """Compute per-pixel median image for a clip; cache to disk if requested."""
    if cache_dir is not None and clip_key is not None:
        cache_path = os.path.join(cache_dir, f"{clip_key}.npz")
        if os.path.exists(cache_path):
            return np.load(cache_path)["median"]

    if len(frame_paths) > max_frames:
        rng = random.Random(0)
        sampled = sorted(rng.sample(range(len(frame_paths)), max_frames))
        paths = [frame_paths[i] for i in sampled]
    else:
        paths = frame_paths

    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    stacked = np.stack(imgs, axis=0)
    median = np.median(stacked, axis=0).astype(np.uint8)
    median_resized = resize_v2(median).transpose(2, 0, 1)

    if cache_dir is not None and clip_key is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_path, median=median_resized)

    return median_resized
