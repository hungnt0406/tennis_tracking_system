"""
Render the test split (game7) to a single annotated video, overlaying all three
bounce-detection arms (heuristic / GBM / TCN) on the original frames.

Uses the SAME path as evaluation — each model's `Scorer.score` for the per-frame
score and `decode_clip` (with the model's calibrated threshold + the shared
DECODE config) for the discrete bounce events — so the overlay matches the
reported metrics exactly.

Run:  python -m inference.make_overlay_video
      python -m inference.make_overlay_video --clip Clip1 --fps 25
"""

import argparse
import os

import cv2
import numpy as np

from data.dataset import iter_clip_features
from data.trajectory import gt_bounce_frames
from evaluation.decode import decode_clip
from train.config import FEATURE, DECODE

# BGR colors per arm + GT.
ARMS = [
    ("heuristic", (0, 165, 255)),   # orange
    ("gbm",       (0, 220, 0)),     # green
    ("tcn",       (255, 120, 0)),   # blue
]
GT_COLOR = (255, 255, 255)         # white
FLASH_AFTER = 5                     # frames a detection banner/ring stays lit


def _load_scorers(device):
    from models.heuristic import Scorer as HScorer
    from models.gbm import Scorer as GScorer
    from models.tcn import Scorer as TScorer
    return {
        "heuristic": HScorer("checkpoints/heuristic_best.json", device=device),
        "gbm":       GScorer("checkpoints/gbm_best.pkl", device=device),
        "tcn":       TScorer("checkpoints/tcn_best.pt", device=device),
    }


def _events_window(frames, T, after=FLASH_AFTER):
    """Map predicted/GT frame indices to the set of frames their flash covers."""
    lit = {}
    for f in frames:
        for t in range(int(f), min(T, int(f) + after + 1)):
            lit.setdefault(t, []).append(int(f))
    return lit


def _draw_panel(img, t, scores, thresholds, lit_by_arm, gt_lit):
    """Bottom overlay: one score bar per arm + a banner when a bounce fires."""
    H, W = img.shape[:2]
    pad, row_h, bar_x = 12, 26, 150
    panel_h = row_h * len(ARMS) + 2 * pad
    y0 = H - panel_h
    cv2.rectangle(img, (0, y0), (W, H), (0, 0, 0), -1)
    overlay_alpha = img  # already drawn solid; keep simple

    bar_w = W - bar_x - 60
    for i, (name, color) in enumerate(ARMS):
        cy = y0 + pad + i * row_h + row_h // 2
        s = float(scores[name][t])
        thr = thresholds[name]
        fired = t in lit_by_arm[name]
        # label
        cv2.putText(img, name, (10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
        # bar track
        cv2.rectangle(img, (bar_x, cy - 8), (bar_x + bar_w, cy + 8),
                      (60, 60, 60), -1)
        # filled value
        fill = int(bar_w * max(0.0, min(1.0, s)))
        cv2.rectangle(img, (bar_x, cy - 8), (bar_x + fill, cy + 8), color, -1)
        # threshold tick
        tx = bar_x + int(bar_w * max(0.0, min(1.0, thr)))
        cv2.line(img, (tx, cy - 12), (tx, cy + 12), (0, 0, 255), 2)
        # numeric value
        cv2.putText(img, f"{s:.2f}", (bar_x + bar_w + 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        if fired:
            cv2.circle(img, (bar_x - 18, cy), 8, color, -1)

    # top banner listing whoever fired on this frame
    banner = []
    for name, color in ARMS:
        if t in lit_by_arm[name]:
            banner.append((name.upper(), color))
    if t in gt_lit:
        banner.append(("GT BOUNCE", GT_COLOR))
    bx = 12
    for text, color in banner:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (bx - 4, 8), (bx + tw + 4, 8 + th + 10), (0, 0, 0), -1)
        cv2.putText(img, text, (bx, 8 + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
        bx += tw + 20


def _draw_ball_and_rings(img, traj, t, lit_by_arm, gt_lit):
    """Yellow ball dot + an expanding colored ring per arm that just fired."""
    x, y = float(traj.x[t]), float(traj.y[t])
    has_ball = bool(traj.visible[t]) and x >= 0 and y >= 0
    if has_ball:
        cv2.circle(img, (int(x), int(y)), 6, (0, 255, 255), 2, cv2.LINE_AA)

    ring_base = 14
    offset = 0
    for name, color in ARMS:
        if t in lit_by_arm[name]:
            fired_at = min(lit_by_arm[name])
            age = t - fired_at
            r = ring_base + age * 6 + offset
            cx, cy = (int(x), int(y)) if has_ball else (img.shape[1] // 2, img.shape[0] // 2)
            cv2.circle(img, (cx, cy), r, color, 2, cv2.LINE_AA)
            offset += 5
    if t in gt_lit and has_ball:
        # white diamond at the ball for GT bounce
        cv2.drawMarker(img, (int(x), int(y)), GT_COLOR, cv2.MARKER_DIAMOND, 26, 2)


def main():
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", default="splits.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--clip", default=None, help="only this clip name, e.g. Clip1")
    ap.add_argument("--arm", default=None, choices=[n for n, _ in ARMS],
                    help="render only this model's overlay")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--out", default="results/videos/game7_bounce_overlay.mp4")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.arm:
        ARMS = [(n, c) for n, c in ARMS if n == args.arm]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    scorers = _load_scorers(args.device)
    thresholds = {
        n: (s.threshold if s.threshold is not None else DECODE["threshold"])
        for n, s in scorers.items()
    }
    print("Thresholds:", {n: round(float(t), 3) for n, t in thresholds.items()})

    writer = None
    size = None
    n_written = 0

    for traj, feats, names, valid in iter_clip_features(args.splits_csv, args.split, FEATURE):
        if args.clip and traj.clip != args.clip:
            continue
        T = len(traj)
        scores = {n: np.asarray(s.score(feats, names, traj), dtype=float)
                  for n, s in scorers.items()}
        lit_by_arm = {}
        for n in scorers:
            preds = decode_clip(scores[n], thresholds[n],
                                DECODE["min_peak_distance"], valid,
                                DECODE["peak_offset"])
            lit_by_arm[n] = _events_window(preds, T)
        gt_lit = _events_window(gt_bounce_frames(traj), T)

        for t in range(T):
            img = cv2.imread(traj.frame_paths[t])
            if img is None:
                continue
            if size is None:
                size = (img.shape[1], img.shape[0])
            if (img.shape[1], img.shape[0]) != size:
                img = cv2.resize(img, size)
            if writer is None:
                writer = cv2.VideoWriter(
                    args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps, size)

            cv2.putText(img, f"{traj.game}/{traj.clip}  frame {t}",
                        (12, size[1] - 6 - 26 * len(ARMS) - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            _draw_ball_and_rings(img, traj, t, lit_by_arm, gt_lit)
            _draw_panel(img, t, scores, thresholds, lit_by_arm, gt_lit)
            writer.write(img)
            n_written += 1
        print(f"  {traj.game}/{traj.clip}: {T} frames")

    if writer is not None:
        writer.release()
    print(f"Wrote {n_written} frames -> {args.out}")


if __name__ == "__main__":
    main()
