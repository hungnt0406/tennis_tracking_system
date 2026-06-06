"""
Consumers of the shared trajectory pipeline.

  - build_feature_table : per-frame numpy table for the GBM (training).
  - BounceWindowDataset : torch windows for the TCN (training).
  - iter_clip_features  : per-clip full-length features for EVALUATION/decoding
                          (decoding needs whole sequences, not dropped rows).
  - feature_matrix / channel_stack : turn a clip's feature dict into the GBM
                          matrix (T, F) or the TCN channel stack (C, T).

All feature computation lives in data/trajectory.py, so every arm sees the
same numbers. Windows and rows never cross clip boundaries.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from data.trajectory import (
    load_clip_trajectories, compute_frame_features, make_soft_target,
)
from train.config import FEATURE, TARGET, TCN_CHANNELS


# ─── feature-dict → array helpers (shared by training and evaluation) ─────────
def feature_matrix(feats, names) -> np.ndarray:
    """(T, F) GBM matrix from a clip's feature dict, columns ordered by `names`."""
    return np.stack([feats[n] for n in names], axis=1).astype(np.float32)


def channel_stack(feats, channels=TCN_CHANNELS) -> np.ndarray:
    """(C, T) TCN channel stack selected by name."""
    return np.stack([feats[c] for c in channels], axis=0).astype(np.float32)


def iter_clip_features(splits_csv, split, feature_cfg=FEATURE):
    """Yield (traj, feats, names, valid) per clip — for evaluation/decoding."""
    trajs = load_clip_trajectories(splits_csv, split)
    names = None
    for traj in trajs.values():
        feats, valid = compute_frame_features(traj, feature_cfg)
        if names is None:
            names = list(feats.keys())
        yield traj, feats, names, valid


# ─── GBM training table ───────────────────────────────────────────────────────
def build_feature_table(splits_csv, split, feature_cfg=FEATURE, target_cfg=TARGET,
                         max_samples=None):
    """Per-frame GBM table over all clips in `split`.

    Returns (X, y_soft, y_hard, meta). Rows on invalid (long-gap) frames are
    dropped. `meta` carries parallel arrays game/clip/frame_idx/status plus the
    ordered feature_names.
    """
    trajs = load_clip_trajectories(splits_csv, split)
    Xs, ys, yh = [], [], []
    g_, c_, fi_, st_ = [], [], [], []
    names = None
    for (game, clip), traj in trajs.items():
        feats, valid = compute_frame_features(traj, feature_cfg)
        if names is None:
            names = list(feats.keys())
        X = feature_matrix(feats, names)
        target, _ = make_soft_target(traj.status, valid, target_cfg["sigma_frames"])
        keep = valid
        n_keep = int(keep.sum())
        if n_keep == 0:
            continue
        Xs.append(X[keep])
        ys.append(target[keep])
        yh.append((traj.status[keep] == 2).astype(int))
        g_.append(np.array([game] * n_keep))
        c_.append(np.array([clip] * n_keep))
        fi_.append(traj.frame_idx[keep])
        st_.append(traj.status[keep])

    X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, len(names or [])), np.float32)
    y_soft = np.concatenate(ys) if ys else np.zeros((0,), np.float32)
    y_hard = np.concatenate(yh) if yh else np.zeros((0,), int)
    meta = dict(
        game=np.concatenate(g_) if g_ else np.array([]),
        clip=np.concatenate(c_) if c_ else np.array([]),
        frame_idx=np.concatenate(fi_) if fi_ else np.array([], int),
        status=np.concatenate(st_) if st_ else np.array([], int),
        feature_names=names or [],
    )
    if max_samples is not None:
        X, y_soft, y_hard = X[:max_samples], y_soft[:max_samples], y_hard[:max_samples]
        for k in ("game", "clip", "frame_idx", "status"):
            meta[k] = meta[k][:max_samples]
    return X, y_soft, y_hard, meta


# ─── TCN window dataset ───────────────────────────────────────────────────────
class BounceWindowDataset(Dataset):
    """Sliding windows of the TCN channel stack, with the soft target and a loss
    mask. Windows never cross clip boundaries; the tail is edge-padded (pad
    frames get loss_mask 0)."""

    def __init__(self, splits_csv, split, window=64, stride=8, max_samples=None,
                 feature_cfg=FEATURE, target_cfg=TARGET, channels=TCN_CHANNELS):
        self.window = window
        trajs = load_clip_trajectories(splits_csv, split)
        self.items = []
        for traj in trajs.values():
            feats, valid = compute_frame_features(traj, feature_cfg)
            target, loss_mask = make_soft_target(
                traj.status, valid, target_cfg["sigma_frames"])
            chan = channel_stack(feats, channels)   # (C, T)
            T = chan.shape[1]
            if T == 0:
                continue
            starts = list(range(0, max(1, T - window + 1), stride))
            if starts[-1] + window < T:              # cover the tail
                starts.append(T - window)
            for s in starts:
                e = s + window
                if e <= T:
                    f, tg, lm = chan[:, s:e], target[s:e], loss_mask[s:e]
                else:                                # pad short/last window
                    pad = e - T
                    f = np.pad(chan[:, s:T], ((0, 0), (0, pad)), mode="edge")
                    tg = np.pad(target[s:T], (0, pad), mode="edge")
                    lm = np.pad(loss_mask[s:T], (0, pad), mode="constant")
                self.items.append((f.astype(np.float32),
                                   tg.astype(np.float32),
                                   lm.astype(np.float32)))
        if max_samples is not None:
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        f, tg, lm = self.items[idx]
        return torch.from_numpy(f), torch.from_numpy(tg), torch.from_numpy(lm)
