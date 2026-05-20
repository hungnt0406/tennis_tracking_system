"""
Dataset class for court keypoint detection.

All three model families (TrackNet-court, ResNet50, HRNet) consume a single
``CourtKeypointDataset``; they differ only in the ``stride`` argument which
controls the spatial resolution of the returned heatmap.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.splits import load_records
from data.preprocessing import (
    INPUT_H, INPUT_W, NUM_KEYPOINTS,
    Augmenter, resize_image, normalize, make_target_heatmap,
)

_DEFAULT_DATA_DIR = Path(__file__).parent


class CourtKeypointDataset(Dataset):
    """Single-frame court keypoint dataset.

    Returns
    -------
    image_tensor : torch.Tensor  [3, INPUT_H, INPUT_W]  float32
    heatmap_tensor : torch.Tensor  [15, out_h, out_w]  float32
    kps_orig_tensor : torch.Tensor  [14, 2]  float32
        Keypoints in original (pre-resize) pixel space;
        ``[-1, -1]`` for invisible / missing keypoints.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"``, or ``"test"``.
    augment : bool
        Enable random augmentation (brightness, contrast, hflip).
        Should be True only for the training split.
    max_samples : int or None
        Truncate the dataset to this many samples (useful for smoke tests).
    stride : int
        Heatmap downsampling factor relative to ``INPUT_H × INPUT_W``.
        Use 1 for TrackNet-court (heatmap = 360×640).
        Use 4 for ResNet50/HRNet (heatmap = 90×160).
    gaussian_radius : int
        Gaussian blob radius in *output* (heatmap) space.
    imagenet_norm : bool
        If True, apply ImageNet mean/std normalisation to the image tensor.
    data_dir : str or Path, optional
        Root data directory containing ``data_train.json``, ``data_val.json``,
        and an ``images/`` subdirectory. Defaults to the ``data/`` folder
        next to this file. Override on Kaggle, e.g. ``/kaggle/input/dataset``.
    """

    def __init__(
        self,
        split: str,
        augment: bool = False,
        max_samples: int | None = None,
        stride: int = 1,
        gaussian_radius: int = 15,
        imagenet_norm: bool = False,
        data_dir: str | Path | None = None,
    ):
        data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
        self.images_dir = data_dir / "images"
        self.records = load_records(split, data_dir=data_dir)
        if max_samples is not None:
            self.records = self.records[:max_samples]

        self.augmenter = Augmenter(
            brightness=0.3,
            contrast=0.3,
            hflip_prob=0.5,
            enabled=augment,
        )
        self.stride = stride
        self.gaussian_radius = gaussian_radius
        self.imagenet_norm = imagenet_norm

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        record = self.records[idx]

        # --- load image ---
        img_path = self.images_dir / (record["id"] + ".png")
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]

        # --- parse keypoints ---
        # Each element of record["kps"] is [x, y] (always present in this dataset).
        # We treat any kp with negative coords or out-of-bounds as invisible.
        raw_kps = record["kps"]  # list of 14 [x, y]
        kps: list = []
        for kp in raw_kps:
            if kp is None:
                kps.append(None)
            else:
                x, y = float(kp[0]), float(kp[1])
                if x < 0 or y < 0 or x >= orig_w or y >= orig_h:
                    kps.append(None)
                else:
                    kps.append([x, y])

        # --- augmentation (on original-resolution image) ---
        img, kps = self.augmenter(img, kps)

        # --- save original-space kps for evaluation ---
        kps_orig = np.full((NUM_KEYPOINTS, 2), -1.0, dtype=np.float32)
        for i, kp in enumerate(kps):
            if kp is not None:
                kps_orig[i, 0] = kp[0]
                kps_orig[i, 1] = kp[1]

        # --- resize image to model input size ---
        img_resized = resize_image(img, INPUT_H, INPUT_W)

        # --- normalise to CHW float32 ---
        image_tensor = normalize(img_resized, imagenet=self.imagenet_norm)

        # --- build heatmap at output resolution ---
        out_h = INPUT_H // self.stride
        out_w = INPUT_W // self.stride
        heatmap = make_target_heatmap(
            kps, orig_w, orig_h, out_h, out_w, self.gaussian_radius
        )

        return (
            torch.from_numpy(image_tensor),                          # [3, 360, 640]
            torch.from_numpy(heatmap),                               # [15, out_h, out_w]
            torch.from_numpy(kps_orig),                              # [14, 2]
        )
