"""Run a TrackNet-family model on the test split and dump its failure cases.

Supports the two 9-channel / classmap models that share the same decode path:
``tracknet`` (TrackNet) and ``tracknetv4`` (TrackNetV4). Each test frame is
classified against the ground-truth label and only the failures are written to
disk as annotated overlays:

    missed/   ground-truth ball present but model predicted no ball   (false negative)
    wrong/    model predicted a ball that is absent or mislocalised    (false positive)
                ghost_*  : GT has no ball, model fired anyway
                mislo_*  : both present but distance > --tol_px

Distances and the tolerance are in the resized IMG_H x IMG_W pixel space, the
same space evaluation/evaluate.py reports metrics in. Overlays are drawn on the
original full-resolution frame (GT green, prediction red).

Usage:
    python -m evaluation.find_failures --model tracknet   --checkpoint checkpoints/tracknet_best.pt   --output_dir failure_analysis
    python -m evaluation.find_failures --model tracknetv4 --checkpoint checkpoints/tracknetv4_best.pt --output_dir failure_analysis
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import TrackNetDataset
from data.preprocessing import IMG_H, IMG_W
from models.tracknet import intensity_to_coords
from train.config import SPLITS_CSV


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def _load_model(model_name, checkpoint, device):
    if model_name == "tracknet":
        from models.tracknet import TrackNet
        model = TrackNet().to(device)
    elif model_name == "tracknetv4":
        from models.tracknetv4 import TrackNetV4
        model = TrackNetV4().to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def _scale_to_original(px, py, orig_w, orig_h):
    if px < 0 or py < 0:
        return -1.0, -1.0
    return float(px / IMG_W * orig_w), float(py / IMG_H * orig_h)


def _gt_resized(record, orig_w, orig_h):
    """Ground-truth ball centre in resized space, (-1, -1) when absent."""
    if int(record["visibility"]) <= 0:
        return -1.0, -1.0
    x, y = float(record["x"]), float(record["y"])
    if x < 0 or y < 0:
        return -1.0, -1.0
    return float(x / orig_w * IMG_W), float(y / orig_h * IMG_H)


def _classify(gt_visible, pred_visible, dist, tol_px):
    if gt_visible and not pred_visible:
        return "missed"
    if pred_visible and not gt_visible:
        return "ghost"          # false positive, ball is absent
    if pred_visible and gt_visible:
        return "correct" if dist <= tol_px else "mislocalized"
    return "true_negative"      # both absent — nothing to draw


def _draw_overlay(frame_bgr, record, pred_resized, category, dist, model_name):
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    px, py = _scale_to_original(pred_resized[0], pred_resized[1], w, h)

    if int(record["visibility"]) > 0 and float(record["x"]) >= 0 and float(record["y"]) >= 0:
        gx, gy = int(round(float(record["x"]))), int(round(float(record["y"])))
        cv2.circle(out, (gx, gy), 8, (0, 255, 0), 2)
        cv2.putText(out, "GT", (gx + 6, max(15, gy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    if px >= 0 and py >= 0:
        ipx, ipy = int(round(px)), int(round(py))
        cv2.circle(out, (ipx, ipy), 7, (0, 0, 255), 2)
        cv2.putText(out, model_name, (ipx + 6, min(h - 8, ipy + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    tag = category if dist is None else f"{category}  d={dist:.1f}px"
    cv2.putText(out, tag, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)
    return out


def find_failures(args):
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(args.model, args.checkpoint, device)

    out_dir = Path(args.output_dir) / args.model
    missed_dir = out_dir / "missed"
    wrong_dir = out_dir / "wrong"
    missed_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    dataset = TrackNetDataset(
        args.splits_csv, "test", augment=False,
        max_samples=args.max_samples, target_mode="classmap",
        frame_cache_dir=args.frame_cache_dir,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    counts = {"total": 0, "correct": 0, "missed": 0, "ghost": 0,
              "mislocalized": 0, "true_negative": 0}
    rows = []
    t0 = time.time()

    with torch.no_grad():
        global_idx = 0
        for frames, _, _ in loader:
            frames = frames.to(device, non_blocking=True)
            logits = model(frames)
            pred_xy = intensity_to_coords(
                logits.argmax(dim=1).cpu(), use_hough=True).numpy()

            for b in range(pred_xy.shape[0]):
                _, r_cur, _ = dataset.samples[global_idx]
                frame = cv2.imread(r_cur["frame_path"])
                if frame is None:
                    raise FileNotFoundError(r_cur["frame_path"])
                orig_h, orig_w = frame.shape[:2]

                px_r, py_r = float(pred_xy[b, 0]), float(pred_xy[b, 1])
                gx_r, gy_r = _gt_resized(r_cur, orig_w, orig_h)
                pred_visible = px_r >= 0 and py_r >= 0
                gt_visible = gx_r >= 0 and gy_r >= 0

                dist = (float(np.hypot(px_r - gx_r, py_r - gy_r))
                        if pred_visible and gt_visible else None)
                category = _classify(gt_visible, pred_visible, dist, args.tol_px)

                counts["total"] += 1
                counts[category] += 1

                base = (f"{_safe_name(r_cur['game'])}_"
                        f"{_safe_name(r_cur['clip'])}_"
                        f"{_safe_name(r_cur['frame_name'])}")
                if Path(base).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    base += ".jpg"

                saved_path = ""
                if category == "missed":
                    saved_path = str(missed_dir / base)
                    cv2.imwrite(saved_path,
                                _draw_overlay(frame, r_cur, (px_r, py_r),
                                              category, dist, args.model))
                elif category in ("ghost", "mislocalized"):
                    prefix = "ghost_" if category == "ghost" else "mislo_"
                    saved_path = str(wrong_dir / (prefix + base))
                    cv2.imwrite(saved_path,
                                _draw_overlay(frame, r_cur, (px_r, py_r),
                                              category, dist, args.model))

                rows.append({
                    "game": r_cur["game"], "clip": r_cur["clip"],
                    "frame_name": r_cur["frame_name"],
                    "frame_path": r_cur["frame_path"],
                    "visibility": int(r_cur["visibility"]),
                    "gt_x_orig": float(r_cur["x"]), "gt_y_orig": float(r_cur["y"]),
                    "pred_x_resized": px_r, "pred_y_resized": py_r,
                    "gt_x_resized": gx_r, "gt_y_resized": gy_r,
                    "distance_px_resized": dist,
                    "category": category,
                    "saved_overlay": saved_path,
                })

                global_idx += 1
                if global_idx % args.log_interval == 0:
                    print(f"  {global_idx}/{len(dataset)} processed", flush=True)

    elapsed = time.time() - t0
    counts["wrong_total"] = counts["ghost"] + counts["mislocalized"]
    counts["fps"] = counts["total"] / max(elapsed, 1e-9)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": args.model, "checkpoint": args.checkpoint,
        "split": "test", "tol_px_resized": args.tol_px,
        "img_w": IMG_W, "img_h": IMG_H, "device": str(device),
        "elapsed_seconds": elapsed, "counts": counts,
    }
    with open(out_dir / "counts.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{args.model}] {counts['total']} frames in {elapsed:.1f}s "
          f"({counts['fps']:.1f} FPS)")
    print(f"  missed (FN)      : {counts['missed']}")
    print(f"  ghost  (FP)      : {counts['ghost']}")
    print(f"  mislocalized (FP): {counts['mislocalized']}")
    print(f"  correct          : {counts['correct']}")
    print(f"  true_negative    : {counts['true_negative']}")
    print(f"  -> missed/ and wrong/ overlays in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["tracknet", "tracknetv4"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="failure_analysis")
    parser.add_argument("--splits_csv", default=SPLITS_CSV)
    parser.add_argument("--tol_px", type=float, default=5.0,
                        help="Max distance (resized px) for a detection to count as correct.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--frame_cache_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    find_failures(parser.parse_args())
