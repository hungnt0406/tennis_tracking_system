"""Run TrackNetV4 inference on a fixed number of test samples.

Outputs:
    predictions.csv  Raw metadata, GT coordinates, predictions, and distances.
    metadata.json    Run configuration and summary.
    overlays/        Original frames annotated with GT and prediction.
    tracknetv4_inference_preview.mp4
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
from models.tracknetv4 import TrackNetV4
from train.config import SPLITS_CSV


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def _scale_to_original(pred_xy, orig_w, orig_h):
    if pred_xy[0] < 0 or pred_xy[1] < 0:
        return -1.0, -1.0
    return float(pred_xy[0] / IMG_W * orig_w), float(pred_xy[1] / IMG_H * orig_h)


def _gt_resized(record, orig_w, orig_h):
    if int(record["visibility"]) <= 0:
        return -1.0, -1.0
    x = float(record["x"])
    y = float(record["y"])
    if x < 0 or y < 0:
        return -1.0, -1.0
    return float(x / orig_w * IMG_W), float(y / orig_h * IMG_H)


def _draw_overlay(frame_bgr, record, pred_xy_resized):
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    pred_x_orig, pred_y_orig = _scale_to_original(pred_xy_resized, w, h)

    if int(record["visibility"]) > 0 and float(record["x"]) >= 0 and float(record["y"]) >= 0:
        gx, gy = int(round(float(record["x"]))), int(round(float(record["y"])))
        cv2.circle(out, (gx, gy), 8, (0, 255, 0), 2)
        cv2.putText(out, "GT", (gx + 6, max(15, gy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    if pred_x_orig >= 0 and pred_y_orig >= 0:
        px, py = int(round(pred_x_orig)), int(round(pred_y_orig))
        cv2.circle(out, (px, py), 7, (0, 0, 255), 2)
        cv2.putText(out, "TrackNetV4", (px + 6, min(h - 8, py + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    return out


def _load_model(checkpoint, device):
    model = TrackNetV4().to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def infer(args):
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    dataset = TrackNetDataset(
        args.splits_csv,
        "test",
        augment=False,
        max_samples=args.num_samples,
        target_mode="classmap",
        frame_cache_dir=args.frame_cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = _load_model(args.checkpoint, device)

    rows = []
    overlay_paths = []
    t0 = time.time()

    with torch.no_grad():
        global_idx = 0
        for frames, _, _ in loader:
            frames = frames.to(device, non_blocking=True)
            logits = model(frames)
            pred_xy = intensity_to_coords(
                logits.argmax(dim=1).cpu(), use_hough=True
            ).numpy()

            for batch_i in range(pred_xy.shape[0]):
                r_prev, r_cur, r_next = dataset.samples[global_idx]
                frame = cv2.imread(r_cur["frame_path"])
                if frame is None:
                    raise FileNotFoundError(r_cur["frame_path"])
                orig_h, orig_w = frame.shape[:2]

                pred_x_resized = float(pred_xy[batch_i, 0])
                pred_y_resized = float(pred_xy[batch_i, 1])
                pred_x_orig, pred_y_orig = _scale_to_original(
                    (pred_x_resized, pred_y_resized), orig_w, orig_h
                )
                gt_x_resized, gt_y_resized = _gt_resized(r_cur, orig_w, orig_h)

                if gt_x_resized >= 0 and pred_x_resized >= 0:
                    dist_resized = float(
                        np.hypot(pred_x_resized - gt_x_resized,
                                 pred_y_resized - gt_y_resized)
                    )
                else:
                    dist_resized = None

                overlay = _draw_overlay(frame, r_cur, (pred_x_resized, pred_y_resized))
                overlay_name = (
                    f"{global_idx:04d}_"
                    f"{_safe_name(r_cur['game'])}_"
                    f"{_safe_name(r_cur['clip'])}_"
                    f"{_safe_name(r_cur['frame_name'])}"
                )
                if Path(overlay_name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    overlay_name += ".jpg"
                overlay_path = overlay_dir / overlay_name
                cv2.imwrite(str(overlay_path), overlay)
                overlay_paths.append(overlay_path)

                rows.append({
                    "sample_idx": global_idx,
                    "game": r_cur["game"],
                    "clip": r_cur["clip"],
                    "frame_name": r_cur["frame_name"],
                    "prev_frame_path": r_prev["frame_path"],
                    "frame_path": r_cur["frame_path"],
                    "next_frame_path": r_next["frame_path"],
                    "visibility": int(r_cur["visibility"]),
                    "gt_x_orig": float(r_cur["x"]),
                    "gt_y_orig": float(r_cur["y"]),
                    "gt_x_resized": gt_x_resized,
                    "gt_y_resized": gt_y_resized,
                    "pred_visible": bool(pred_x_resized >= 0 and pred_y_resized >= 0),
                    "pred_x_resized": pred_x_resized,
                    "pred_y_resized": pred_y_resized,
                    "pred_x_orig": pred_x_orig,
                    "pred_y_orig": pred_y_orig,
                    "distance_px_resized": dist_resized,
                    "correct_5px": (
                        None if dist_resized is None else bool(dist_resized <= 5.0)
                    ),
                    "overlay_path": str(overlay_path),
                })

                global_idx += 1
                if global_idx % args.log_interval == 0:
                    print(f"Processed {global_idx}/{len(dataset)} samples", flush=True)

    csv_path = output_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    video_path = output_dir / "tracknetv4_inference_preview.mp4"
    if overlay_paths:
        first = cv2.imread(str(overlay_paths[0]))
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (w, h),
        )
        for path in overlay_paths:
            writer.write(cv2.imread(str(path)))
        writer.release()

    visible_rows = [r for r in rows if int(r["visibility"]) > 0]
    distances = [
        r["distance_px_resized"] for r in visible_rows
        if r["distance_px_resized"] is not None
    ]
    metadata = {
        "model": "tracknetv4",
        "checkpoint": args.checkpoint,
        "split": "test",
        "num_samples": len(rows),
        "batch_size": args.batch_size,
        "device": str(device),
        "img_w": IMG_W,
        "img_h": IMG_H,
        "elapsed_seconds": time.time() - t0,
        "fps": len(rows) / max(time.time() - t0, 1e-9),
        "mean_distance_px_resized": (
            float(np.mean(distances)) if distances else None
        ),
        "acc_at_5px_on_visible": (
            float(np.mean([d <= 5.0 for d in distances])) if distances else None
        ),
        "outputs": {
            "predictions_csv": str(csv_path),
            "overlay_dir": str(overlay_dir),
            "preview_video": str(video_path),
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved predictions -> {csv_path}")
    print(f"Saved overlays -> {overlay_dir}")
    print(f"Saved video -> {video_path}")
    print(f"Saved metadata -> {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits_csv", default=SPLITS_CSV)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--frame_cache_dir", default=None)
    parser.add_argument("--log_interval", type=int, default=25)
    infer(parser.parse_args())
