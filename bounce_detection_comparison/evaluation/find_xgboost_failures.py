"""Collect every XGBoost bounce misclassification on the test split.

Re-runs the trained XGBoost arm through the SHARED decode + matching (so the
result is identical to evaluation.evaluate), then dumps the two error classes a
bounce detector can make:

  - false positive : a decoded bounce peak with no GT bounce within +/-k frames.
  - false negative : a GT bounce (status==2) with no decoded peak within +/-k.

The two classes are written to separate subfolders of the output dir:

    bounce_xgboost_failures/wrong/   <- false positives (model fired, no bounce)
    bounce_xgboost_failures/missed/  <- false negatives (true bounce missed)

Each subfolder gets its own manifest (failures.csv + failures.json) plus a
frames/ dir holding the actual dataset JPEG for every failure (named
<game>_<clip>_frame<idx>_<type>.jpg, with the labelled ball ringed). Optionally
renders a capped number of failure trajectories into the matching subfolder,
reusing visualize_trajectory.

Run from bounce_detection_comparison/:

    python -m evaluation.find_xgboost_failures
    python -m evaluation.find_xgboost_failures --render 8
"""

import argparse
import csv
import json
import os

import numpy as np

from data.dataset import iter_clip_features
from data.trajectory import gt_bounce_frames
from train.config import SPLITS_CSV, DECODE, FEATURE
from evaluation.decode import decode_clip, match_events
from models.gbm import Scorer

DEFAULT_CKPT = "checkpoints/xgboost_best.pkl"
DEFAULT_OUT = "evaluation/bounce_xgboost_failures"

# Two-folder layout: wrong/ = false positives, missed/ = false negatives.
SUBDIR = {"false_positive": "wrong", "false_negative": "missed"}


def _peak_score(score, frame, k):
    """Max model score within +/-k frames of `frame` (near-miss strength)."""
    lo, hi = max(0, frame - k), min(len(score), frame + k + 1)
    return float(score[lo:hi].max()) if hi > lo else float("nan")


def collect_failures(checkpoint, splits_csv, split, decode_cfg):
    scorer = Scorer(checkpoint)
    # Mirror evaluation.evaluate: the per-checkpoint threshold overrides DECODE.
    if getattr(scorer, "threshold", None) is not None:
        decode_cfg["threshold"] = float(scorer.threshold)
    thr = decode_cfg["threshold"]
    dist = decode_cfg["min_peak_distance"]
    k = decode_cfg["tolerance_k"]
    off = decode_cfg["peak_offset"]

    failures = []
    tp = fp = fn = 0
    n_clips = n_frames = 0

    for traj, feats, names, valid in iter_clip_features(splits_csv, split, FEATURE):
        score = np.asarray(scorer.score(feats, names, traj), dtype=float)
        pred = decode_clip(score, thr, dist, valid, off)
        gt = gt_bounce_frames(traj)
        t, f, n, matched = match_events(pred, gt, k)
        tp += t; fp += f; fn += n
        n_clips += 1
        n_frames += len(traj)

        matched_pred = {p for p, _ in matched}
        matched_gt = {g for _, g in matched}

        # false positives: decoded peaks not matched to any GT bounce
        for p in pred:
            if int(p) in matched_pred:
                continue
            failures.append(dict(
                game=traj.game, clip=traj.clip,
                frame_idx=int(traj.frame_idx[p]), clip_pos=int(p),
                failure_type="false_positive",
                pred_label=1, gt_label=0,
                peak_score=float(score[p]),
                score_near_event=_peak_score(score, int(p), k),
                frame_path=traj.frame_paths[p],
                ball_x=float(traj.x[p]), ball_y=float(traj.y[p]),
            ))

        # false negatives / missed: GT bounces not matched to any peak
        for g in gt:
            if int(g) in matched_gt:
                continue
            failures.append(dict(
                game=traj.game, clip=traj.clip,
                frame_idx=int(traj.frame_idx[g]), clip_pos=int(g),
                failure_type="false_negative",
                pred_label=0, gt_label=1,
                peak_score=float(score[g]),               # raw model score at GT frame
                score_near_event=_peak_score(score, int(g), k),
                frame_path=traj.frame_paths[g],
                ball_x=float(traj.x[g]), ball_y=float(traj.y[g]),
            ))

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    summary = dict(
        model="xgboost", checkpoint=checkpoint, split=split,
        threshold=thr, min_peak_distance=dist, tolerance_k=k, peak_offset=off,
        n_clips=n_clips, n_frames=n_frames,
        TP=tp, FP=fp, FN=fn,
        precision=p, recall=r, F1=f1,
        n_false_positive=sum(1 for x in failures if x["failure_type"] == "false_positive"),
        n_false_negative=sum(1 for x in failures if x["failure_type"] == "false_negative"),
    )
    return failures, summary


def write_manifest(failures, summary, out_dir):
    """Write one manifest (CSV + JSON) per failure type into its subfolder."""
    cols = ["game", "clip", "frame_idx", "clip_pos", "failure_type",
            "pred_label", "gt_label", "peak_score", "score_near_event"]
    paths = {}
    for ftype, sub in SUBDIR.items():
        sub_dir = os.path.join(out_dir, sub)
        os.makedirs(sub_dir, exist_ok=True)
        rows = [r for r in failures if r["failure_type"] == ftype]
        rows.sort(key=lambda r: (r["game"], r["clip"], r["clip_pos"]))
        csv_path = os.path.join(sub_dir, "failures.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in rows:
                w.writerow({c: row[c] for c in cols})
        json_path = os.path.join(sub_dir, "failures.json")
        with open(json_path, "w") as f:
            json.dump({"summary": summary, "failure_type": ftype,
                       "n_failures": len(rows), "failures": rows}, f, indent=2)
        paths[sub] = (csv_path, json_path)
    return paths


def save_frames(failures, out_dir):
    """Copy each failure's dataset JPEG into <sub>/frames/, with the ball
    position ringed when coords are valid. Returns (n_saved_per_sub, missing)."""
    import cv2

    saved = {sub: 0 for sub in SUBDIR.values()}
    missing = []
    for row in failures:
        src = row["frame_path"]
        if not os.path.exists(src):
            missing.append(dict(game=row["game"], clip=row["clip"],
                                frame_idx=row["frame_idx"], frame_path=src))
            continue
        img = cv2.imread(src)
        if img is None:
            missing.append(dict(game=row["game"], clip=row["clip"],
                                frame_idx=row["frame_idx"], frame_path=src))
            continue
        x, y = row["ball_x"], row["ball_y"]
        if x >= 0 and y >= 0:                     # ring the labelled ball (BGR)
            colour = (0, 0, 255) if row["failure_type"] == "false_positive" else (0, 255, 0)
            cv2.circle(img, (int(round(x)), int(round(y))), 18, colour, 2)
        sub = SUBDIR[row["failure_type"]]
        frames_dir = os.path.join(out_dir, sub, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        name = f"{row['game']}_{row['clip']}_frame{row['frame_idx']}_{row['failure_type']}.jpg"
        cv2.imwrite(os.path.join(frames_dir, name), img)
        saved[sub] += 1
    return saved, missing


def render_failures(failures, splits_csv, out_dir, cap):
    """Render failure-clip trajectories into the matching subfolder (wrong/ or
    missed/), reusing visualize_trajectory. Cap is global across both types."""
    from evaluation.visualize_trajectory import load_clip, render
    seen, rendered = set(), 0
    for row in failures:
        if rendered >= cap:
            break
        key = (row["game"], row["clip"], row["failure_type"])
        if key in seen:
            continue
        seen.add(key)
        img_dir = os.path.join(out_dir, SUBDIR[row["failure_type"]], "trajectories")
        os.makedirs(img_dir, exist_ok=True)
        fr, x, y, status = load_clip(splits_csv, row["game"], row["clip"])
        name = f"{row['game']}_{row['clip']}_{row['failure_type']}"
        render(fr, x, y, status, os.path.join(img_dir, f"{name}.png"), dpi=150)
        rendered += 1
    return rendered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--splits_csv", default=SPLITS_CSV)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--render", type=int, default=0,
                    help="render up to N failure-clip trajectories (0 = none)")
    args = ap.parse_args()

    failures, summary = collect_failures(
        args.checkpoint, args.splits_csv, args.split, dict(DECODE))
    paths = write_manifest(failures, summary, args.output_dir)

    print(f"XGBoost @ threshold {summary['threshold']} on {summary['split']} split "
          f"({summary['n_clips']} clips, {summary['n_frames']} frames)")
    print(f"  TP/FP/FN = {summary['TP']}/{summary['FP']}/{summary['FN']}  "
          f"P={summary['precision']:.3f} R={summary['recall']:.3f} F1={summary['F1']:.3f}")
    print(f"  failures: {summary['n_false_positive']} FP + "
          f"{summary['n_false_negative']} FN = {len(failures)} total")
    print(f"  wrong/  ({summary['n_false_positive']} FP) -> {paths['wrong'][0]}, {paths['wrong'][1]}")
    print(f"  missed/ ({summary['n_false_negative']} FN) -> {paths['missed'][0]}, {paths['missed'][1]}")

    saved, missing = save_frames(failures, args.output_dir)
    print(f"  saved frames: wrong/frames={saved['wrong']}, missed/frames={saved['missed']}")
    if missing:
        print(f"  WARNING: {len(missing)} failure(s) had a missing source frame:")
        for m in missing:
            print(f"    {m['game']}/{m['clip']} frame {m['frame_idx']} -> {m['frame_path']}")

    if args.render > 0:
        n = render_failures(failures, args.splits_csv, args.output_dir, args.render)
        print(f"  rendered {n} trajectory plots (capped at {args.render}) "
              f"into wrong/ and missed/")


if __name__ == "__main__":
    main()
