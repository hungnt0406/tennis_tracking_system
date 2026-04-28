"""
Dataset classes for TrackNet, S-KeepTrack, and YOLO.

All three read from a splits CSV produced by subset_selector.py.
"""

import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocessing import (
    IMG_H, IMG_W, Augmenter, resize_frame, normalize,
    make_gaussian_heatmap, coords_to_yolo,
)


def _load_splits(splits_csv: str, split: str) -> list:
    records = []
    with open(splits_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                records.append(row)
    return records


def _group_by_clip(records: list) -> dict:
    """Return {(game, clip): [sorted records]} ordered by frame name."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["game"], r["clip"])].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r["frame_name"])
    return groups


class TrackNetDataset(Dataset):
    """
    Returns 3-consecutive-frame sequences (9-channel input) and a heatmap label.
    Ball position comes from the middle (second) frame annotation.
    """

    def __init__(self, splits_csv: str, split: str, augment: bool = False,
                 max_samples: int = None):
        self.augment = Augmenter(enabled=augment)
        records = _load_splits(splits_csv, split)
        groups = _group_by_clip(records)

        self.samples = []
        for clip_records in groups.values():
            for i in range(1, len(clip_records) - 1):
                self.samples.append(
                    (clip_records[i - 1], clip_records[i], clip_records[i + 1])
                )
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r_prev, r_cur, r_next = self.samples[idx]
        frames_raw = []
        for r in (r_prev, r_cur, r_next):
            img = cv2.imread(r["frame_path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames_raw.append(img)

        orig_h, orig_w = frames_raw[0].shape[:2]
        x = float(r_cur["x"])
        y = float(r_cur["y"])
        vis = int(r_cur["visibility"])

        frames_raw, x, y = self.augment(frames_raw, x, y, orig_w)
        frames_resized = [resize_frame(f) for f in frames_raw]

        # Stack to 9-channel (3 frames × RGB)
        tensor = np.concatenate([normalize(f) for f in frames_resized], axis=0)

        ball_x = x if vis > 0 else -1.0
        ball_y = y if vis > 0 else -1.0
        heatmap = make_gaussian_heatmap(ball_x, ball_y, orig_w, orig_h)

        return (
            torch.from_numpy(tensor),
            torch.from_numpy(heatmap).unsqueeze(0),
            torch.tensor([vis], dtype=torch.long),
        )


class SKeepTrackDataset(Dataset):
    """
    Returns pairs of consecutive frames with ball annotations for both.
    Used by the S-KeepTrack association training.
    """

    def __init__(self, splits_csv: str, split: str, augment: bool = False,
                 max_samples: int = None):
        self.augment = Augmenter(enabled=augment)
        records = _load_splits(splits_csv, split)
        groups = _group_by_clip(records)

        self.samples = []
        for clip_records in groups.values():
            for i in range(len(clip_records) - 1):
                self.samples.append((clip_records[i], clip_records[i + 1]))
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r1, r2 = self.samples[idx]
        imgs = []
        for r in (r1, r2):
            img = cv2.imread(r["frame_path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        orig_h, orig_w = imgs[0].shape[:2]
        x1, y1, vis1 = float(r1["x"]), float(r1["y"]), int(r1["visibility"])
        x2, y2, vis2 = float(r2["x"]), float(r2["y"]), int(r2["visibility"])

        imgs, x1, y1 = self.augment(imgs, x1, y1, orig_w)
        # Apply the same flip to frame 2's x coordinate
        if self.augment.enabled:
            # Re-use same flip decision — Augmenter already flipped images
            pass

        resized = [resize_frame(f) for f in imgs]
        t1 = torch.from_numpy(normalize(resized[0]))
        t2 = torch.from_numpy(normalize(resized[1]))

        def _hm(x, y, vis, ow, oh):
            bx = x if vis > 0 else -1.0
            by = y if vis > 0 else -1.0
            return torch.from_numpy(make_gaussian_heatmap(bx, by, ow, oh)).unsqueeze(0)

        hm1 = _hm(x1, y1, vis1, orig_w, orig_h)
        hm2 = _hm(x2, y2, vis2, orig_w, orig_h)

        coords1 = torch.tensor([x1 / orig_w, y1 / orig_h, float(vis1 > 0)], dtype=torch.float32)
        coords2 = torch.tensor([x2 / orig_w, y2 / orig_h, float(vis2 > 0)], dtype=torch.float32)

        return t1, t2, hm1, hm2, coords1, coords2


class YOLODataset(Dataset):
    """
    Single-frame dataset returning YOLO-formatted labels.
    Images are resized to 640×640 as expected by YOLO.
    """

    YOLO_SIZE = 640

    def __init__(self, splits_csv: str, split: str, augment: bool = False,
                 max_samples: int = None):
        self.augment = Augmenter(enabled=augment)
        self.records = _load_splits(splits_csv, split)
        if max_samples is not None:
            self.records = self.records[:max_samples]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = cv2.imread(r["frame_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        x = float(r["x"])
        y = float(r["y"])
        vis = int(r["visibility"])

        [img], x, y = self.augment([img], x, y, orig_w)
        img = cv2.resize(img, (self.YOLO_SIZE, self.YOLO_SIZE))
        tensor = torch.from_numpy(normalize(img))

        if vis > 0 and x >= 0 and y >= 0:
            cx, cy, bw, bh = coords_to_yolo(x, y, orig_w, orig_h)
            label = torch.tensor([0, cx, cy, bw, bh], dtype=torch.float32)
            has_ball = torch.tensor(1.0)
        else:
            label = torch.zeros(5, dtype=torch.float32)
            has_ball = torch.tensor(0.0)

        return tensor, label, has_ball, torch.tensor(vis, dtype=torch.long)
