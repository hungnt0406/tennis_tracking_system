"""
Train the GBM bounce regressor on the shared soft (Gaussian-in-time) target,
then calibrate the decode threshold on the val split's event-F1.

Usage:
    python -m train.train_gbm [--max_samples N] [--epochs N]
"""

import argparse
import os

import numpy as np

from data.dataset import build_feature_table, iter_clip_features, feature_matrix
from data.trajectory import gt_bounce_frames
from evaluation.decode import decode_clip, match_events
from evaluation.metrics import prf
from models.gbm import build_model, save_model
from train.config import GBM, FEATURE, DECODE, SPLITS_CSV, CHECKPOINT_DIR


def _calibrate_threshold(model, splits_csv):
    """Sweep decode thresholds on val per-clip scores; return (best_thr, best_f1)."""
    clips = []
    for traj, feats, names, valid in iter_clip_features(splits_csv, "val", FEATURE):
        score = np.clip(model.predict(feature_matrix(feats, names)), 0.0, 1.0)
        clips.append((score, valid, gt_bounce_frames(traj)))

    best_thr, best_f1 = DECODE["threshold"], -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        tp = fp = fn = 0
        for score, valid, gt in clips:
            pred = decode_clip(score, thr, DECODE["min_peak_distance"], valid,
                               DECODE["peak_offset"])
            t, f, n, _ = match_events(pred, gt, DECODE["tolerance_k"])
            tp += t; fp += f; fn += n
        f1 = prf(tp, fp, fn)[2]
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_csv", default=SPLITS_CSV)
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit train+val rows (quick convergence check)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override cfg iterations / max_iter")
    args = parser.parse_args()

    cfg = dict(GBM)
    if args.epochs is not None:
        cfg["iterations"] = args.epochs

    X_tr, y_tr, _, _ = build_feature_table(args.splits_csv, "train",
                                           max_samples=args.max_samples)
    X_val, y_val, _, _ = build_feature_table(args.splits_csv, "val",
                                             max_samples=args.max_samples)
    print(f"Train rows: {len(X_tr)} | Val rows: {len(X_val)} | Features: {X_tr.shape[1]}")

    model = build_model(cfg)
    sample_weight = 1.0 + GBM["pos_weight"] * y_tr

    try:
        from catboost import CatBoostRegressor  # noqa: F401
        backend = "catboost"
        model.fit(X_tr, y_tr, sample_weight=sample_weight,
                  eval_set=(X_val, y_val),
                  early_stopping_rounds=GBM["early_stopping_rounds"],
                  use_best_model=True)
    except ImportError:
        backend = "sklearn"
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
    print(f"Backend: {backend}")
    if backend == "catboost":
        print(f"Best iteration: {model.get_best_iteration()}")

    best_thr, best_f1 = _calibrate_threshold(model, args.splits_csv)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(args.checkpoint_dir, "gbm_best.pkl")
    save_model(model, best_thr, backend, ckpt)
    print(f"Saved checkpoint → {ckpt}")
    print(f"Best threshold: {best_thr:.3f} | Val event-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
