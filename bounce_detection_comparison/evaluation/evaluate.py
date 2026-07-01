"""
Evaluate one bounce-detection arm on a split.

Each arm (models/<arm>.py) exposes a `Scorer` producing a per-frame bounce score
in [0,1]; this driver runs the SHARED decode + metrics so all three arms are
compared identically — only the score source differs. Following the
court-keypoint project's precedent, each `_eval_*` returns a flat metrics dict,
written to results/<model>_metrics.json.

The heuristic has no trained file, so --checkpoint is optional (it falls back to
config defaults) — the one deliberate divergence from the sibling eval CLI.
"""

import argparse
import json
import os
import time

import numpy as np

from data.dataset import iter_clip_features
from data.trajectory import gt_bounce_frames
from train.config import SPLITS_CSV, RESULTS_DIR, DECODE, FEATURE
from evaluation.metrics import compute_all_metrics

EVAL_FNS = ("heuristic", "gbm", "xgboost", "lightgbm", "catboost", "tcn")
DEFAULT_CKPT = {
    "heuristic": "checkpoints/heuristic_best.json",
    "gbm":       "checkpoints/gbm_best.pkl",
    "xgboost":   "checkpoints/xgboost_best.pkl",
    "lightgbm":  "checkpoints/lightgbm_best.pkl",
    "catboost":  "checkpoints/catboost_best.pkl",
    "tcn":       "checkpoints/tcn_best.pt",
}


def _load_scorer(model, checkpoint, device):
    if model == "heuristic":
        from models.heuristic import Scorer
    elif model in ("gbm", "xgboost", "lightgbm", "catboost"):
        from models.gbm import Scorer        # backend-agnostic; predicts on the .pkl envelope
    elif model == "tcn":
        from models.tcn import Scorer
    else:
        raise ValueError(f"unknown model: {model}")
    return Scorer(checkpoint, device=device)


def _eval_arm(model, checkpoint, splits_csv, split, device):
    scorer = _load_scorer(model, checkpoint, device)
    scores, gt, status, valid = {}, {}, {}, {}
    nframes = 0
    t0 = time.time()
    for traj, feats, names, vmask in iter_clip_features(splits_csv, split, FEATURE):
        key = (traj.game, traj.clip)
        scores[key] = np.asarray(scorer.score(feats, names, traj), dtype=float)
        gt[key] = gt_bounce_frames(traj)
        status[key] = traj.status
        valid[key] = vmask
        nframes += len(traj)
    fps = nframes / max(time.time() - t0, 1e-9)

    decode_cfg = dict(DECODE)
    thr = getattr(scorer, "threshold", None)
    if thr is not None:
        decode_cfg["threshold"] = float(thr)
    return compute_all_metrics(scores, gt, status, valid, decode_cfg, fps)


def _sanitize(o):
    """Recursively make a metrics dict JSON-safe (NaN/inf → None, numpy → py)."""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def evaluate(args):
    metrics = _eval_arm(args.model, args.checkpoint, args.splits_csv,
                        args.split, args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"{args.model}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(_sanitize(metrics), f, indent=2)

    fps = metrics.get("FPS") or 0.0
    print(f"{args.model}: F1@{metrics['k']}={metrics['event_F1@k']:.3f}  "
          f"P={metrics['event_precision@k']:.3f}  R={metrics['event_recall@k']:.3f}  "
          f"AP={metrics['AP']:.3f}  "
          f"TP/FP/FN={metrics['TP']}/{metrics['FP']}/{metrics['FN']}  "
          f"FPS={fps:.1f}")
    print(f"  saved → {out_path}")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a bounce-detection arm.")
    p.add_argument("--model", required=True, choices=sorted(EVAL_FNS))
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint path; optional for heuristic (config fallback).")
    p.add_argument("--splits_csv", default=SPLITS_CSV)
    p.add_argument("--split", default="test")
    p.add_argument("--results_dir", default=RESULTS_DIR)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.checkpoint is None:
        args.checkpoint = DEFAULT_CKPT[args.model]
    if args.device is None:
        try:
            import torch
            args.device = ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
        except Exception:
            args.device = "cpu"
    evaluate(args)
