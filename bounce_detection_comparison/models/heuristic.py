"""
Heuristic bounce scorer (no learned parameters).

A ground bounce is, in image-y-grows-down coordinates: the ball at its lowest
point (local max of y), with vy flipping +→− and a large |vertical accel| spike.
We turn each cue into a per-clip [0,1] signal, take a weighted sum, smooth, and
renormalize. "Training" only calibrates the decode threshold (train_heuristic.py).
"""

import json
import os

import numpy as np

from train.config import HEURISTIC


def _norm01(a):
    """Min-max to [0,1]; all-equal inputs map to zeros."""
    a = np.asarray(a, dtype=float)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _moving_average(a, w):
    if w <= 1 or len(a) < 2:
        return a
    w = min(int(w), len(a))
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode="same")


def heuristic_score(feats, weights=None) -> np.ndarray:
    """Per-frame bounce score in [0,1], shape (T,)."""
    w = weights or HEURISTIC
    y = np.asarray(feats["y_norm"], dtype=float)
    vy = np.asarray(feats["vy"], dtype=float)
    ay = np.asarray(feats["ay"], dtype=float)
    T = len(y)
    if T == 0:
        return np.zeros(0, dtype=float)

    # vertical local-maximum: ball lowest on screen (y grows downward)
    y_local_max = np.zeros(T)
    if T >= 3:
        is_max = (y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:])
        y_local_max[1:-1] = np.where(is_max, y[1:-1], 0.0)
    cue_ymin = _norm01(y_local_max)

    # vertical-velocity sign flip +→− within ±2 frames
    flip = np.zeros(T)
    for t in range(T):
        lo, hi = max(0, t - 2), min(T, t + 3)
        win = vy[lo:hi]
        if (win > 0).any() and (win < 0).any():
            flip[t] = win.max() - win.min()      # bigger reversal ⇒ stronger cue
    cue_vflip = _norm01(flip)

    # |vertical acceleration| spike
    cue_accel = _norm01(np.abs(ay))

    score = (w["w_ymin"] * cue_ymin
             + w["w_vflip"] * cue_vflip
             + w["w_accel"] * cue_accel)
    score = _moving_average(score, w.get("smooth_window", 1))
    return _norm01(score)


class Scorer:
    """Shared scorer contract: .score(feats, names, traj) -> (T,) float in [0,1]."""

    def __init__(self, checkpoint_path=None, device="cpu"):
        self.weights = dict(HEURISTIC)
        self.threshold = None
        if checkpoint_path and os.path.isfile(checkpoint_path):
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            self.threshold = ckpt.get("threshold")
            if ckpt.get("weights"):
                self.weights = ckpt["weights"]

    def score(self, feats, names, traj):
        s = heuristic_score(feats, self.weights)
        return s[: len(traj)].astype(float)
