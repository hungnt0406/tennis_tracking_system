"""Image preprocessing, augmentation, and heatmap utilities for court keypoint detection."""

import random
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_H = 360
INPUT_W = 640
NUM_KEYPOINTS = 14
NUM_CHANNELS = 15  # 14 kps + court center

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Horizontal flip keypoint swap table (left<->right symmetric pairs).
#
# Keypoint layout (0-indexed, verified from data_train.json + TennisCourtDetector):
#   0 = top-left outer corner
#   1 = top-right outer corner
#   2 = bottom-left outer corner   (note: data shows kps[2] is lower-left)
#   3 = bottom-right outer corner
#   4 = top-left service box corner
#   5 = bottom-left service box corner
#   6 = top-right service box corner
#   7 = bottom-right service box corner
#   8 = top-left T-point (top of service center line, left side)
#   9 = top-right T-point
#  10 = bottom-left T-point
#  11 = bottom-right T-point
#  12 = top center T-point (midpoint of service center line top)
#  13 = bottom center T-point
#
# IMPORTANT: this is a placeholder identity mapping.  The true bilateral-
# symmetry swap table must be confirmed by visualizing annotated images
# before using hflip augmentation in training.
HFLIP_KP_SWAP = tuple(range(NUM_KEYPOINTS))  # placeholder — needs visual verification


# ---------------------------------------------------------------------------
# Basic image utilities
# ---------------------------------------------------------------------------

def resize_image(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize *img* (HWC uint8) to (h, w) using bilinear interpolation."""
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def normalize(img: np.ndarray, imagenet: bool = False) -> np.ndarray:
    """Convert uint8 HWC image to float32 CHW in [0, 1].

    If *imagenet* is True, further subtract ImageNet mean and divide by std
    per channel (standard torchvision normalisation).
    """
    out = img.astype(np.float32) / 255.0
    out = out.transpose(2, 0, 1)  # HWC -> CHW

    if imagenet:
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        out = (out - mean) / std

    return out


# ---------------------------------------------------------------------------
# Heatmap generation (CenterNet-style Gaussian)
# ---------------------------------------------------------------------------

def draw_umich_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> np.ndarray:
    """Draw a 2-D Gaussian blob on *heatmap* in-place and return it.

    Parameters
    ----------
    heatmap : np.ndarray
        Float32 array of shape (H, W).
    center : (cx, cy)
        Integer pixel coordinates of the Gaussian centre.
    radius : int
        Gaussian radius; diameter = 2*radius+1, sigma = diameter/6.
    """
    H, W = heatmap.shape
    cx, cy = int(center[0]), int(center[1])

    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    x = np.arange(0, diameter, 1, np.float32)
    y = x[:, np.newaxis]
    x0, y0 = radius, radius
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    left = min(cx, radius)
    right = min(W - cx, radius + 1)
    top = min(cy, radius)
    bottom = min(H - cy, radius + 1)

    if left + right <= 0 or top + bottom <= 0:
        return heatmap

    heatmap[cy - top:cy + bottom, cx - left:cx + right] = np.maximum(
        heatmap[cy - top:cy + bottom, cx - left:cx + right],
        g[radius - top:radius + bottom, radius - left:radius + right],
    )
    return heatmap


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def line_intersection(
    line1: tuple[tuple[float, float], tuple[float, float]],
    line2: tuple[tuple[float, float], tuple[float, float]],
) -> Optional[tuple[float, float]]:
    """Return the intersection of two infinite lines, or None if parallel.

    Each line is specified as a pair of points ``((x1,y1),(x2,y2))``.
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # parallel / coincident

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)


# ---------------------------------------------------------------------------
# Target heatmap construction
# ---------------------------------------------------------------------------

def make_target_heatmap(
    kps: list,
    orig_w: int,
    orig_h: int,
    out_h: int,
    out_w: int,
    radius: int,
) -> np.ndarray:
    """Build the (NUM_CHANNELS, out_h, out_w) heatmap tensor for one sample.

    Parameters
    ----------
    kps : list of length NUM_KEYPOINTS
        Each element is ``[x, y]`` in original image pixel space, or ``None``
        for an invisible / unlabelled keypoint.
    orig_w, orig_h : int
        Original image dimensions (used for coordinate scaling).
    out_h, out_w : int
        Output spatial resolution.
    radius : int
        Gaussian radius in output space.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(15, out_h, out_w)``.
        Channels 0-13 correspond to individual keypoints.
        Channel 14 is the inferred court centre (diagonal intersection of the
        four outer corners), drawn only when all four corners are visible.
    """
    heatmap = np.zeros((NUM_CHANNELS, out_h, out_w), dtype=np.float32)

    scale_x = out_w / orig_w
    scale_y = out_h / orig_h

    # Channels 0-13: individual keypoints
    for i, kp in enumerate(kps):
        if kp is None:
            continue
        x, y = kp[0], kp[1]
        if x < 0 or y < 0 or x >= orig_w or y >= orig_h:
            continue
        cx = int(x * scale_x)
        cy = int(y * scale_y)
        draw_umich_gaussian(heatmap[i], (cx, cy), radius)

    # Channel 14: court centre = intersection of the two court diagonals.
    # kps[0]=top-left, kps[1]=top-right, kps[2]=bottom-left, kps[3]=bottom-right.
    # Diagonal 1: top-left <-> bottom-right  (kps[0] <-> kps[3])
    # Diagonal 2: top-right <-> bottom-left  (kps[1] <-> kps[2])
    # Only drawn when all four outer corners are visible.
    corners = [kps[0], kps[1], kps[2], kps[3]]
    if all(c is not None and c[0] >= 0 and c[1] >= 0 for c in corners):
        pt = line_intersection(
            (corners[0], corners[3]),
            (corners[1], corners[2]),
        )
        if pt is not None:
            cx_c = int(pt[0] * scale_x)
            cy_c = int(pt[1] * scale_y)
            if 0 <= cx_c < out_w and 0 <= cy_c < out_h:
                draw_umich_gaussian(heatmap[14], (cx_c, cy_c), radius)

    return heatmap


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class Augmenter:
    """Random photometric + geometric augmentations applied to image and keypoints.

    Parameters
    ----------
    brightness : float
        Max fractional brightness shift; delta sampled from U(-b, b).
    contrast : float
        Max fractional contrast shift; factor sampled from U(1-c, 1+c).
    hflip_prob : float
        Probability of horizontal flip.
    enabled : bool
        When False, the callable is a no-op (useful for val/test).
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        hflip_prob: float = 0.5,
        enabled: bool = True,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.hflip_prob = hflip_prob
        self.enabled = enabled

    def __call__(
        self, img: np.ndarray, kps: list
    ) -> tuple[np.ndarray, list]:
        """Apply augmentations to *img* and *kps*.

        Parameters
        ----------
        img : np.ndarray
            HWC uint8 RGB image.
        kps : list
            List of 14 ``[x, y]`` pairs (original pixel space) or ``None``
            for invisible keypoints.

        Returns
        -------
        (img, kps) : (np.ndarray, list)
            Augmented image (uint8 HWC) and adjusted keypoints.
        """
        if not self.enabled:
            return img, kps

        orig_h, orig_w = img.shape[:2]

        # --- photometric ---
        b_delta = random.uniform(-self.brightness, self.brightness)
        c_factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)

        img = img.astype(np.float32)
        mean_val = img.mean()
        img = (img - mean_val) * c_factor + mean_val  # contrast around mean
        img = img + b_delta * 255.0                    # brightness shift
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)

        # --- horizontal flip ---
        do_flip = random.random() < self.hflip_prob
        if do_flip:
            img = img[:, ::-1, :].copy()
            new_kps = []
            for kp in kps:
                if kp is None:
                    new_kps.append(None)
                else:
                    new_kps.append([orig_w - 1 - kp[0], kp[1]])
            # permute according to bilateral symmetry swap table
            kps = [new_kps[HFLIP_KP_SWAP[i]] for i in range(NUM_KEYPOINTS)]

        return img, kps
