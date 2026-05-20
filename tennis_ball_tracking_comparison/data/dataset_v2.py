"""Dataset classes for TrackNetV2, TrackNetV3, and InpaintNet."""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from data.dataset import _load_splits, _group_by_clip
from data.preprocessing import Augmenter
from data.preprocessing_v2 import (
    IMG_H_V2, IMG_W_V2, SIGMA_V2,
    resize_v2, make_gaussian_heatmap_v2, compute_clip_median,
)


def _scaled_xy(x: float, y: float, vis: int, orig_w: int, orig_h: int):
    if vis <= 0 or x < 0 or y < 0:
        return -1.0, -1.0
    return x / orig_w * IMG_W_V2, y / orig_h * IMG_H_V2


def _read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _normalize_chw(frame_f32: np.ndarray) -> np.ndarray:
    return frame_f32.transpose(2, 0, 1)


class TrackNetV2Dataset(Dataset):
    """Sliding window of 3 frames -> 9-channel input + 3-channel heatmap target."""

    def __init__(self, splits_csv: str, split: str, augment: bool = False,
                 max_samples: int = None):
        self.augment = Augmenter(enabled=augment)
        records = _load_splits(splits_csv, split)
        groups = _group_by_clip(records)

        self.samples = []
        for clip_records in groups.values():
            for i in range(len(clip_records) - 2):
                self.samples.append(clip_records[i:i + 3])
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window = self.samples[idx]
        frames_raw = [_read_rgb(r["frame_path"]) for r in window]
        orig_h, orig_w = frames_raw[0].shape[:2]

        xs = [float(r["x"]) for r in window]
        ys = [float(r["y"]) for r in window]
        vs = [int(r["visibility"]) for r in window]

        do_flip = self.augment.enabled and random.random() < self.augment.hflip_prob
        b_delta = (random.uniform(-self.augment.brightness, self.augment.brightness)
                   if self.augment.enabled else 0.0)
        c_factor = (random.uniform(1 - self.augment.contrast, 1 + self.augment.contrast)
                    if self.augment.enabled else 1.0)

        processed = []
        for f in frames_raw:
            if self.augment.enabled:
                f = f.astype(np.float32) * c_factor + b_delta * 255
                f = np.clip(f, 0, 255).astype(np.uint8)
            if do_flip:
                f = cv2.flip(f, 1)
            processed.append(resize_v2(f))

        if do_flip:
            xs = [(orig_w - 1 - x) if x >= 0 else -1.0 for x in xs]

        input_tensor = np.concatenate([_normalize_chw(f) for f in processed], axis=0)

        target = np.zeros((3, IMG_H_V2, IMG_W_V2), dtype=np.float32)
        for k in range(3):
            sx, sy = _scaled_xy(xs[k], ys[k], vs[k], orig_w, orig_h)
            target[k] = make_gaussian_heatmap_v2(sx, sy)

        return torch.from_numpy(input_tensor), torch.from_numpy(target)


class TrackNetV3Dataset(Dataset):
    """Sliding window of seq_len frames + clip median -> tracker input/target."""

    def __init__(self, splits_csv: str, split: str, seq_len: int = 8,
                 bg_mode: str = 'concat', median_cache_dir: str = 'cache/median',
                 augment: bool = False, max_samples: int = None):
        self.seq_len = seq_len
        self.bg_mode = bg_mode
        self.median_cache_dir = median_cache_dir
        self.augment = Augmenter(enabled=augment)

        os.makedirs(median_cache_dir, exist_ok=True)

        records = _load_splits(splits_csv, split)
        groups = _group_by_clip(records)

        self.samples = []
        self.clip_keys = {}
        self.clip_frame_paths = {}
        for (game, clip), clip_records in groups.items():
            if len(clip_records) < seq_len:
                continue
            key = f"{game}_{clip}"
            self.clip_keys[(game, clip)] = key
            self.clip_frame_paths[key] = [r["frame_path"] for r in clip_records]
            for i in range(len(clip_records) - seq_len + 1):
                self.samples.append((key, clip_records[i:i + seq_len]))
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self._median_cache = {}

    def __len__(self):
        return len(self.samples)

    def _get_median(self, clip_key: str) -> np.ndarray:
        if clip_key in self._median_cache:
            return self._median_cache[clip_key]
        median = compute_clip_median(
            self.clip_frame_paths[clip_key],
            cache_dir=self.median_cache_dir,
            clip_key=clip_key,
        )
        self._median_cache[clip_key] = median
        return median

    def __getitem__(self, idx):
        clip_key, window = self.samples[idx]
        frames_raw = [_read_rgb(r["frame_path"]) for r in window]
        orig_h, orig_w = frames_raw[0].shape[:2]

        xs = [float(r["x"]) for r in window]
        ys = [float(r["y"]) for r in window]
        vs = [int(r["visibility"]) for r in window]

        do_flip = self.augment.enabled and random.random() < self.augment.hflip_prob
        b_delta = (random.uniform(-self.augment.brightness, self.augment.brightness)
                   if self.augment.enabled else 0.0)
        c_factor = (random.uniform(1 - self.augment.contrast, 1 + self.augment.contrast)
                    if self.augment.enabled else 1.0)

        processed = []
        for f in frames_raw:
            if self.augment.enabled:
                f = f.astype(np.float32) * c_factor + b_delta * 255
                f = np.clip(f, 0, 255).astype(np.uint8)
            if do_flip:
                f = cv2.flip(f, 1)
            processed.append(resize_v2(f))

        if do_flip:
            xs = [(orig_w - 1 - x) if x >= 0 else -1.0 for x in xs]

        frame_chans = [_normalize_chw(f) for f in processed]
        median = self._get_median(clip_key)
        if do_flip:
            median = median[:, :, ::-1].copy()

        input_tensor = np.concatenate(frame_chans + [median], axis=0)

        target = np.zeros((self.seq_len, IMG_H_V2, IMG_W_V2), dtype=np.float32)
        for k in range(self.seq_len):
            sx, sy = _scaled_xy(xs[k], ys[k], vs[k], orig_w, orig_h)
            target[k] = make_gaussian_heatmap_v2(sx, sy)

        return torch.from_numpy(input_tensor), torch.from_numpy(target)


class TrajectoryDataset(Dataset):
    """Sequences for InpaintNet training: (coords_with_mask, gt_coords, mask)."""

    def __init__(self, npz_path: str, mask_ratio: float = 0.0):
        data = np.load(npz_path)
        self.coords = data["coords"].astype(np.float32)
        self.gt_coords = data["gt_coords"].astype(np.float32)
        self.mask = data["mask"].astype(np.float32)
        self.mask_ratio = mask_ratio

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        coords = self.coords[idx].copy()
        gt = self.gt_coords[idx].copy()
        mask = self.mask[idx].copy()

        if self.mask_ratio > 0:
            L = mask.shape[0]
            extra = (np.random.rand(L) < self.mask_ratio).astype(np.float32)
            mask = np.maximum(mask, extra)
            coords[extra > 0] = 0.0

        coords_with_mask = np.concatenate(
            [coords, mask[:, None]], axis=1
        ).astype(np.float32)

        return (
            torch.from_numpy(coords_with_mask),
            torch.from_numpy(gt),
            torch.from_numpy(mask),
        )
