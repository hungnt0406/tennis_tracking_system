"""
Shared trajectory → feature/target pipeline (pure numpy/scipy).

Both consumers — the GBM feature table and the TCN window dataset (see
`data/dataset.py`) — build their inputs from the functions here, so the three
model arms are guaranteed to see identical features. Nothing in this module
depends on torch.

Pipeline per clip:
  load_clip_trajectories → clean_trajectory (gap fill + masks)
                         → compute_kinematics (savgol vx,vy,ax,ay)
                         → compute_lag_features (signed/abs diffs + ratios)
                         → compute_frame_features (assembled, ordered)
  make_soft_target builds the Gaussian-in-time training label.

Conventions: image-y grows downward, so a ground bounce is a LOCAL MAX of y.
A coordinate of -1 / visibility 0 means "no ball"; it never enters arithmetic.
All operations are strictly per clip — features never cross clip boundaries.
"""

import csv
import os
from collections import defaultdict, OrderedDict

import cv2
import numpy as np
from scipy.signal import savgol_filter


# ─── splits.csv loading (mirrors the sibling ball-tracking helpers) ───────────
def _load_splits(splits_csv: str, split: str) -> list:
    records = []
    with open(splits_csv) as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                records.append(row)
    return records


def _group_by_clip(records: list) -> dict:
    """Return {(game, clip): [records]} ordered by integer frame_idx."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["game"], r["clip"])].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: int(r["frame_idx"]))
    return groups


class ClipTrajectory:
    """Ordered per-frame arrays for a single clip."""

    def __init__(self, game, clip, frame_idx, x, y, visible, status,
                 frame_paths, orig_w, orig_h):
        self.game = game
        self.clip = clip
        self.frame_idx = frame_idx      # (T,) int
        self.x = x                      # (T,) float, -1 where invisible
        self.y = y                      # (T,) float, -1 where invisible
        self.visible = visible          # (T,) bool
        self.status = status            # (T,) int (2 == bounce, 1 == hit)
        self.frame_paths = frame_paths  # list[str], length T
        self.orig_w = orig_w
        self.orig_h = orig_h

    def __len__(self):
        return len(self.frame_idx)


def _read_frame_size(path, fallback=(720, 1280)):
    img = cv2.imread(path)
    if img is None:
        return fallback
    return img.shape[0], img.shape[1]


def load_clip_trajectories(splits_csv: str, split: str) -> "OrderedDict[tuple, ClipTrajectory]":
    """Load every clip in `split` as an ordered ClipTrajectory."""
    groups = _group_by_clip(_load_splits(splits_csv, split))
    out = OrderedDict()
    for (game, clip), recs in groups.items():
        x = np.array([float(r["x"]) for r in recs], dtype=float)
        y = np.array([float(r["y"]) for r in recs], dtype=float)
        vis = np.array([int(r["visibility"]) > 0 for r in recs], dtype=bool)
        status = np.array([int(r["status"]) for r in recs], dtype=int)
        frame_idx = np.array([int(r["frame_idx"]) for r in recs], dtype=int)
        paths = [r["frame_path"] for r in recs]
        oh, ow = _read_frame_size(paths[0]) if paths else (720, 1280)
        out[(game, clip)] = ClipTrajectory(game, clip, frame_idx, x, y, vis,
                                           status, paths, ow, oh)
    return out


# ─── helpers ──────────────────────────────────────────────────────────────────
def _contiguous_runs(mask: np.ndarray):
    """Yield (start, end) index pairs of contiguous True runs in `mask`."""
    T = len(mask)
    i = 0
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """Shift so out[t] = a[t-k], NaN-filling vacated positions (never wraps)."""
    out = np.full_like(a, np.nan, dtype=float)
    n = len(a)
    if k >= 0:
        if k < n:
            out[k:] = a[: n - k]
    else:
        k = -k
        if k < n:
            out[: n - k] = a[k:]
    return out


# ─── pipeline ─────────────────────────────────────────────────────────────────
def clean_trajectory(x, y, visible, max_gap=4):
    """Replace the -1/invisible sentinel with NaN, linearly interpolate INTERNAL
    gaps of length ≤ max_gap, and return (x, y, interp_mask, valid_mask).

    Long gaps and leading/trailing missing frames stay NaN (valid_mask False) so
    they are never labeled or scored as confident bounces.
    """
    x = np.where(visible, x, np.nan).astype(float)
    y = np.where(visible, y, np.nan).astype(float)
    x[x < 0] = np.nan
    y[y < 0] = np.nan

    interp_mask = np.zeros(len(x), dtype=bool)
    isnan = np.isnan(x)
    for s, e in _contiguous_runs(isnan):
        run_len = e - s
        if s > 0 and e < len(x) and run_len <= max_gap:   # internal short gap
            for arr in (x, y):
                ramp = np.linspace(arr[s - 1], arr[e], run_len + 2)
                arr[s:e] = ramp[1:-1]
            interp_mask[s:e] = True

    valid_mask = ~np.isnan(x)
    return x, y, interp_mask, valid_mask


def compute_kinematics(x, y, window=7, poly=2):
    """Signed velocity (vx, vy) and acceleration (ax, ay) via Savitzky–Golay
    (per contiguous valid run; central-diff fallback when too short)."""
    T = len(x)
    vx = np.full(T, np.nan); vy = np.full(T, np.nan)
    ax = np.full(T, np.nan); ay = np.full(T, np.nan)
    valid = ~np.isnan(x)
    for s, e in _contiguous_runs(valid):
        n = e - s
        xs, ys = x[s:e], y[s:e]
        if n >= window:
            vx[s:e] = savgol_filter(xs, window, poly, deriv=1)
            vy[s:e] = savgol_filter(ys, window, poly, deriv=1)
            ax[s:e] = savgol_filter(xs, window, poly, deriv=2)
            ay[s:e] = savgol_filter(ys, window, poly, deriv=2)
        elif n >= 3:
            vx[s:e] = np.gradient(xs); vy[s:e] = np.gradient(ys)
            ax[s:e] = np.gradient(vx[s:e]); ay[s:e] = np.gradient(vy[s:e])
        elif n == 2:
            vx[s:e] = np.gradient(xs); vy[s:e] = np.gradient(ys)
            ax[s:e] = 0.0; ay[s:e] = 0.0
        else:  # n == 1
            vx[s:e] = vy[s:e] = ax[s:e] = ay[s:e] = 0.0
    return vx, vy, ax, ay


def _angle_diff(theta):
    """Frame-to-frame change in heading angle, wrapped to [-pi, pi]."""
    d = np.zeros_like(theta)
    d[1:] = theta[1:] - theta[:-1]
    return np.arctan2(np.sin(d), np.cos(d))


def compute_lag_features(x, y, n_lags=10, eps=1e-6, clip_val=20.0):
    """Bidirectional lag features (yastrebksv recipe). For each lag i:
        y_diff_back_i = y(t) - y(t-i)         (signed; vertical reversal)
        y_diff_fut_i  = y(t) - y(t+i)
        x_diff_back_i = |x(t) - x(t-i)|       (abs; horizontal direction-agnostic)
        x_diff_fut_i  = |x(t) - x(t+i)|
        y_div_i = y_diff_fut_i / (y_diff_back_i + eps)   (≈-1 flying, ≈+1 at bounce)
        x_div_i = x_diff_fut_i / (x_diff_back_i + eps)
    The scale/perspective-invariant *_div ratios are the accuracy driver.
    Returns an OrderedDict {name: (T,) array} with NaN→0.
    """
    feats = OrderedDict()
    for i in range(1, n_lags + 1):
        x_back, x_fut = _shift(x, i), _shift(x, -i)
        y_back, y_fut = _shift(y, i), _shift(y, -i)
        y_diff_back = y - y_back
        y_diff_fut = y - y_fut
        x_diff_back = np.abs(x - x_back)
        x_diff_fut = np.abs(x - x_fut)
        y_div = np.clip(y_diff_fut / (y_diff_back + eps), -clip_val, clip_val)
        x_div = np.clip(x_diff_fut / (x_diff_back + eps), -clip_val, clip_val)
        feats[f"y_diff_back_{i}"] = y_diff_back
        feats[f"y_diff_fut_{i}"] = y_diff_fut
        feats[f"x_diff_back_{i}"] = x_diff_back
        feats[f"x_diff_fut_{i}"] = x_diff_fut
        feats[f"y_div_{i}"] = y_div
        feats[f"x_div_{i}"] = x_div
    for k in feats:
        feats[k] = np.nan_to_num(feats[k], nan=0.0, posinf=clip_val, neginf=-clip_val)
    return feats


def compute_frame_features(traj: ClipTrajectory, feature_cfg) -> tuple:
    """Assemble the full ordered per-frame feature dict for one clip.

    Returns (feats: OrderedDict[name -> (T,) float], valid: (T,) bool).
    `feats` holds every GBM column; the TCN selects TCN_CHANNELS by name.
    All arrays are finite (NaN→0); `valid` flags frames usable as targets.
    """
    x, y, interp, valid = clean_trajectory(
        traj.x, traj.y, traj.visible, feature_cfg["max_gap"])
    vx, vy, ax, ay = compute_kinematics(
        x, y, feature_cfg["savgol_window"], feature_cfg["savgol_poly"])

    speed = np.hypot(vx, vy)
    accel_mag = np.hypot(ax, ay)
    theta = np.arctan2(vy, vx)
    dtheta = _angle_diff(theta)
    x_norm = x / max(traj.orig_w, 1)
    y_norm = y / max(traj.orig_h, 1)
    visible_f = traj.visible.astype(float)

    feats = OrderedDict([
        ("x_norm", x_norm), ("y_norm", y_norm),
        ("vx", vx), ("vy", vy), ("ax", ax), ("ay", ay),
        ("speed", speed), ("accel_mag", accel_mag), ("dtheta", dtheta),
        ("visible", visible_f), ("interp", interp.astype(float)),
    ])
    feats.update(compute_lag_features(x, y, feature_cfg["n_lags"]))

    for k in feats:
        feats[k] = np.nan_to_num(feats[k], nan=0.0, posinf=0.0, neginf=0.0)
    return feats, valid


def make_soft_target(status, valid, sigma=1.5):
    """Gaussian-in-time bounce target (peak 1.0 at each status==2 frame) and a
    loss mask that excludes long-gap/invalid frames. status==1 hits stay ≈0
    (hard negatives)."""
    T = len(status)
    target = np.zeros(T, dtype=float)
    bounce_idx = np.where(np.asarray(status) == 2)[0]
    if len(bounce_idx):
        t = np.arange(T)
        for b in bounce_idx:
            target = np.maximum(target, np.exp(-0.5 * ((t - b) / sigma) ** 2))
    loss_mask = np.asarray(valid, dtype=float)
    return target, loss_mask


def gt_bounce_frames(traj: ClipTrajectory) -> np.ndarray:
    """Indices (into the clip's ordered frames) where status==2."""
    return np.where(traj.status == 2)[0]
