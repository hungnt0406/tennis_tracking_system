"""
Train a boosted-tree bounce regressor on the shared soft (Gaussian-in-time)
target, then calibrate the decode threshold on the val split's event-F1.

One trainer, four interchangeable backends (selected with --model):
    gbm       scikit-learn HistGradientBoostingRegressor (histogram GBDT)
    xgboost   XGBoost
    lightgbm  LightGBM
    catboost  CatBoost
All arms share identical features, target, sample weighting (1 + pos_weight * y)
and threshold calibration, so the comparison isolates the library, not the
recipe. Each saves a single .pkl envelope at checkpoints/<model>_best.pkl.

Usage:
    python -m train.train_gbm [--model {gbm,xgboost,lightgbm,catboost}]
                              [--max_samples N] [--epochs N]
"""

import argparse
import os

import numpy as np

from data.dataset import build_feature_table, iter_clip_features, feature_matrix
from data.trajectory import gt_bounce_frames
from evaluation.decode import decode_clip, match_events
from evaluation.metrics import prf
from models.gbm import build_model, save_model
from train.config import (GBM, XGBOOST, LIGHTGBM, CATBOOST,
                          FEATURE, DECODE, SPLITS_CSV, CHECKPOINT_DIR)

# arm name → (backend key, config dict)
ARMS = {
    "gbm":      ("histgbm",  GBM),
    "xgboost":  ("xgboost",  XGBOOST),
    "lightgbm": ("lightgbm", LIGHTGBM),
    "catboost": ("catboost", CATBOOST),
}


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


def _fit(backend, model, X_tr, y_tr, sample_weight, X_val, y_val, cfg):
    """Fit with each library's native early-stopping API; return the model."""
    if backend == "histgbm":                       # sklearn: internal early stop
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
    elif backend == "xgboost":                     # early_stopping_rounds set on ctor
        model.fit(X_tr, y_tr, sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)], verbose=False)
    elif backend == "lightgbm":
        import lightgbm as lgb
        model.fit(X_tr, y_tr, sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"]),
                             lgb.log_evaluation(0)])
    elif backend == "catboost":
        model.fit(X_tr, y_tr, sample_weight=sample_weight,
                  eval_set=(X_val, y_val),
                  early_stopping_rounds=cfg["early_stopping_rounds"],
                  use_best_model=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gbm", choices=sorted(ARMS),
                        help="Boosting backend / comparison arm to train.")
    parser.add_argument("--splits_csv", default=SPLITS_CSV)
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit train+val rows (quick convergence check)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override cfg iterations / max_iter")
    args = parser.parse_args()

    backend, base_cfg = ARMS[args.model]
    cfg = dict(base_cfg)
    if args.epochs is not None:
        cfg["iterations"] = args.epochs

    X_tr, y_tr, _, _ = build_feature_table(args.splits_csv, "train",
                                           max_samples=args.max_samples)
    X_val, y_val, _, _ = build_feature_table(args.splits_csv, "val",
                                             max_samples=args.max_samples)
    print(f"[{args.model}] Train rows: {len(X_tr)} | Val rows: {len(X_val)} | "
          f"Features: {X_tr.shape[1]}")

    model = build_model(backend, cfg)
    sample_weight = 1.0 + cfg["pos_weight"] * y_tr
    model = _fit(backend, model, X_tr, y_tr, sample_weight, X_val, y_val, cfg)
    print(f"Backend: {backend}")

    best_thr, best_f1 = _calibrate_threshold(model, args.splits_csv)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(args.checkpoint_dir, f"{args.model}_best.pkl")
    save_model(model, best_thr, backend, ckpt)
    print(f"Saved checkpoint → {ckpt}")
    print(f"Best threshold: {best_thr:.3f} | Val event-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
