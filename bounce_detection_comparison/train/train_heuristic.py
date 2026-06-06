"""
"Train" the heuristic = calibrate its decode threshold on the val split.

The heuristic has no learned parameters; we only sweep candidate thresholds,
decode + match bounce events on every val clip, and keep the best-F1 threshold.

Usage:
    python -m train.train_heuristic
"""

import argparse
import json
import os

import numpy as np

from data.dataset import iter_clip_features
from data.trajectory import gt_bounce_frames
from evaluation.decode import decode_clip, match_events
from evaluation.metrics import prf
from models.heuristic import heuristic_score
from train.config import FEATURE, DECODE, HEURISTIC, SPLITS_CSV, CHECKPOINT_DIR


def train(args):
    clips = list(iter_clip_features(args.splits_csv, "val", FEATURE))
    if args.max_samples is not None:
        clips = clips[: args.max_samples]
    print(f"Val clips: {len(clips)}")

    # Precompute scores once; only the threshold changes across the sweep.
    scored = [(heuristic_score(feats, HEURISTIC), valid, gt_bounce_frames(traj))
              for traj, feats, names, valid in clips]

    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = float(thresholds[0]), -1.0
    for th in thresholds:
        tp = fp = fn = 0
        for score, valid, gt in scored:
            pred = decode_clip(score, th, DECODE["min_peak_distance"],
                               valid, DECODE["peak_offset"])
            t, f, n, _ = match_events(pred, gt, DECODE["tolerance_k"])
            tp += t; fp += f; fn += n
        f1 = prf(tp, fp, fn)[2]
        if f1 > best_f1:
            best_f1, best_thr = f1, float(th)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(args.checkpoint_dir, "heuristic_best.json")
    with open(ckpt, "w") as f:
        json.dump({"threshold": best_thr, "weights": HEURISTIC,
                   "k": DECODE["tolerance_k"]}, f, indent=2)

    print(f"Best threshold={best_thr:.3f} | val event-F1={best_f1:.3f}")
    print(f"Saved checkpoint → {ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_csv", default=SPLITS_CSV)
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of val clips (quick check)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Ignored; accepted for CLI-family consistency")
    train(parser.parse_args())
