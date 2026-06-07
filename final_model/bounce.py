"""Bounce detector wrapper over the bounce_detection_comparison GBM arm.

Wraps the shared trajectory→feature→GBM→decode pipeline behind a single
`detect` call that takes a ball trajectory in original-frame pixel space.
"""
import numpy as np

from ._loader import import_project, BOUNCE_ROOT

_mods = import_project(
    BOUNCE_ROOT,
    ["models.gbm", "data.trajectory", "evaluation.decode", "train.config"],
    "_fm_bounce",
)
Scorer = _mods["models.gbm"].Scorer
ClipTrajectory = _mods["data.trajectory"].ClipTrajectory
compute_frame_features = _mods["data.trajectory"].compute_frame_features
decode_clip = _mods["evaluation.decode"].decode_clip
FEATURE = _mods["train.config"].FEATURE
DECODE = _mods["train.config"].DECODE


class BounceDetector:
    def __init__(self, checkpoint: str):
        self.scorer = Scorer(checkpoint, device="cpu")
        self.threshold = (self.scorer.threshold if self.scorer.threshold is not None
                          else DECODE["threshold"])

    def detect(self, traj_xy, visible, orig_w, orig_h):
        """traj_xy (T,2) original-frame pixels (-1 = invisible), visible (T,) bool.
        Returns (N,) int array of bounce frame indices."""
        traj_xy = np.asarray(traj_xy, float)
        T = len(traj_xy)
        frame_idx = np.arange(T)
        x = traj_xy[:, 0].astype(float)
        y = traj_xy[:, 1].astype(float)
        status = np.zeros(T, int)
        frame_paths = [""] * T
        vis = np.asarray(visible, bool)

        traj = ClipTrajectory("infer", "infer", frame_idx, x, y, vis, status,
                              frame_paths, orig_w, orig_h)
        feats, valid = compute_frame_features(traj, FEATURE)
        names = list(feats.keys())
        score = self.scorer.score(feats, names, traj)
        bounce = decode_clip(score, self.threshold, DECODE["min_peak_distance"],
                             valid, DECODE["peak_offset"])
        return np.asarray(bounce, int)
