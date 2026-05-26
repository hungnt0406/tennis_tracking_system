"""
Run evaluation on the test split for a single model.

Usage:
    python -m evaluation.evaluate --model tracknet    --checkpoint checkpoints/tracknet_best.pt
    python -m evaluation.evaluate --model tracknetv2  --checkpoint checkpoints/tracknetv2_best.pt
    python -m evaluation.evaluate --model tracknetv3  --checkpoint checkpoints/tracknetv3_inpaint_best.pt
    python -m evaluation.evaluate --model tracknetv4  --checkpoint checkpoints/tracknetv4_best.pt
    python -m evaluation.evaluate --model tracknetv5  --checkpoint checkpoints/tracknetv5_best.pt
    python -m evaluation.evaluate --model skeeptrack  --checkpoint checkpoints/skeeptrack_best.pt
    python -m evaluation.evaluate --model yolo11m     --checkpoint checkpoints/yolo11m_best.pt
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import TrackNetDataset, YOLODataset, _load_splits, _group_by_clip
from data.preprocessing import IMG_H, IMG_W
from evaluation.metrics import compute_all_metrics
from train.config import (
    SPLITS_CSV, CHECKPOINT_DIR, RESULTS_DIR,
    TRACKNETV2, TRACKNETV3_TRACKER, TRACKNETV3_INPAINT,
)


# ─── Per-model prediction helpers ────────────────────────────────────────────

def _eval_tracknet(checkpoint: str, splits_csv: str, device):
    from models.tracknet import TrackNet, heatmap_to_coords

    model = TrackNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ds = TrackNetDataset(splits_csv, "test", augment=False)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    pred_xys, gt_xys, pred_vis, gt_vis, vis_cls = [], [], [], [], []
    t0 = time.time()
    total_frames = 0

    with torch.no_grad():
        for frames, heatmaps, visibility in dl:
            frames = frames.to(device)
            pred_hm = model(frames)

            pred_coords = heatmap_to_coords(pred_hm.cpu(), threshold=0.5).numpy()
            gt_coords   = heatmap_to_coords(heatmaps, threshold=0.1).numpy()

            B = frames.shape[0]
            total_frames += B

            # Coords are in the resized heatmap's pixel space (IMG_H × IMG_W).
            pred_xys.append(pred_coords)
            gt_xys.append(gt_coords)
            pred_vis.append(pred_coords[:, 0] >= 0)
            gt_vis.append(gt_coords[:, 0] >= 0)
            vis_cls.append(visibility.squeeze(1).numpy())

    elapsed = time.time() - t0
    fps = total_frames / elapsed

    return (np.concatenate(pred_xys), np.concatenate(gt_xys),
            np.concatenate(pred_vis), np.concatenate(gt_vis),
            np.concatenate(vis_cls), fps)


def _eval_tracknetv4(checkpoint: str, splits_csv: str, device):
    from models.tracknetv4 import TrackNetV4
    from models.tracknet import heatmap_to_coords

    model = TrackNetV4().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ds = TrackNetDataset(splits_csv, "test", augment=False)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    pred_xys, gt_xys, pred_vis, gt_vis, vis_cls = [], [], [], [], []
    t0 = time.time()
    total_frames = 0

    with torch.no_grad():
        for frames, heatmaps, visibility in dl:
            frames = frames.to(device)
            pred_hm = model(frames)

            pred_coords = heatmap_to_coords(pred_hm.cpu(), threshold=0.5).numpy()
            gt_coords   = heatmap_to_coords(heatmaps, threshold=0.1).numpy()

            total_frames += frames.shape[0]
            pred_xys.append(pred_coords)
            gt_xys.append(gt_coords)
            pred_vis.append(pred_coords[:, 0] >= 0)
            gt_vis.append(gt_coords[:, 0] >= 0)
            vis_cls.append(visibility.squeeze(1).numpy())

    fps = total_frames / (time.time() - t0)
    return (np.concatenate(pred_xys), np.concatenate(gt_xys),
            np.concatenate(pred_vis), np.concatenate(gt_vis),
            np.concatenate(vis_cls), fps)


def _eval_tracknetv5(checkpoint: str, splits_csv: str, device):
    from models.tracknetv5 import TrackNetV5
    from models.tracknet import heatmap_to_coords

    model = TrackNetV5().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ds = TrackNetDataset(splits_csv, "test", augment=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

    pred_xys, gt_xys, pred_vis, gt_vis, vis_cls = [], [], [], [], []
    t0 = time.time()
    total_frames = 0

    with torch.no_grad():
        for frames, heatmaps, visibility in dl:
            frames = frames.to(device)
            pred_hm = model(frames)

            pred_coords = heatmap_to_coords(pred_hm.cpu(), threshold=0.5).numpy()
            gt_coords   = heatmap_to_coords(heatmaps, threshold=0.1).numpy()

            total_frames += frames.shape[0]
            pred_xys.append(pred_coords)
            gt_xys.append(gt_coords)
            pred_vis.append(pred_coords[:, 0] >= 0)
            gt_vis.append(gt_coords[:, 0] >= 0)
            vis_cls.append(visibility.squeeze(1).numpy())

    fps = total_frames / (time.time() - t0)
    return (np.concatenate(pred_xys), np.concatenate(gt_xys),
            np.concatenate(pred_vis), np.concatenate(gt_vis),
            np.concatenate(vis_cls), fps)



def _eval_yolo(checkpoint: str, splits_csv: str, device, use_ultralytics: bool):
    if use_ultralytics:
        try:
            from ultralytics import YOLO
            from data.dataset import _load_splits
            import cv2

            model = YOLO(checkpoint)
            records = _load_splits(splits_csv, "test")
            pred_xys, gt_xys, pred_vis, gt_vis, vis_cls = [], [], [], [], []
            t0 = time.time()
            for r in records:
                img = cv2.imread(r["frame_path"])
                h, w = img.shape[:2]
                results = model(img, verbose=False)
                boxes = results[0].boxes
                if boxes and len(boxes):
                    best = boxes.conf.argmax()
                    if boxes.conf[best] > 0.3:
                        x1, y1, x2, y2 = boxes.xyxy[best].tolist()
                        px, py = (x1 + x2) / 2, (y1 + y2) / 2
                        pred_vis.append(True)
                    else:
                        px, py = -1.0, -1.0
                        pred_vis.append(False)
                else:
                    px, py = -1.0, -1.0
                    pred_vis.append(False)

                gx, gy = float(r["x"]), float(r["y"])
                vis = int(r["visibility"])
                pred_xys.append([px, py])
                gt_xys.append([gx, gy])
                gt_vis.append(vis > 0)
                vis_cls.append(vis)

            fps = len(records) / (time.time() - t0)
            return (np.array(pred_xys), np.array(gt_xys),
                    np.array(pred_vis), np.array(gt_vis),
                    np.array(vis_cls), fps)
        except ImportError:
            pass

    # Fallback
    from models.yolo11m import LightweightDetector
    model = LightweightDetector().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ds = YOLODataset(splits_csv, "test", augment=False)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    pred_xys, gt_xys, pred_vis, gt_vis, vis_cls = [], [], [], [], []
    t0 = time.time()
    total = 0
    with torch.no_grad():
        for frames, labels, has_ball, visibility in dl:
            frames = frames.to(device)
            out = model(frames)   # (B, 5)
            conf = torch.sigmoid(out[:, 0]).cpu().numpy()
            cx = out[:, 1].cpu().numpy() * IMG_W
            cy = out[:, 2].cpu().numpy() * IMG_H

            gx = labels[:, 1].numpy() * IMG_W
            gy = labels[:, 2].numpy() * IMG_H

            total += frames.shape[0]
            pred_xys.append(np.stack([cx, cy], axis=1))
            gt_xys.append(np.stack([gx, gy], axis=1))
            pred_vis.append(conf > 0.3)
            gt_vis.append(has_ball.numpy().astype(bool))
            vis_cls.append(visibility.numpy())

    fps = total / (time.time() - t0)
    return (np.concatenate(pred_xys), np.concatenate(gt_xys),
            np.concatenate(pred_vis), np.concatenate(gt_vis),
            np.concatenate(vis_cls), fps)


# ─── TrackNetV2 / V3 helpers ─────────────────────────────────────────────────

def _v2_scale_pred_to_orig(coords_v2: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    """Scale (N, 2) predictions from 288x512 space to original-image pixel space.
    Preserves the (-1, -1) sentinel.
    """
    from data.preprocessing_v2 import IMG_H_V2, IMG_W_V2
    out = coords_v2.copy().astype(np.float32)
    missing = (out[:, 0] < 0) | (out[:, 1] < 0)
    out[:, 0] = out[:, 0] / float(IMG_W_V2) * float(orig_w)
    out[:, 1] = out[:, 1] / float(IMG_H_V2) * float(orig_h)
    out[missing] = -1.0
    return out


def _v2_predict_clip(model, device, clip_records, seq_len: int, hm_threshold: float):
    """Run V2 (seq_len=3) over a clip with sliding windows; last-write-wins.
    Returns pred_xy in 288x512 space (T, 2) with -1 sentinel, plus (orig_h, orig_w).
    """
    from data.dataset_v2 import _read_rgb, _normalize_chw
    from data.preprocessing_v2 import resize_v2
    from models.tracknet import heatmap_to_coords

    T = len(clip_records)
    pred_xy = np.full((T, 2), -1.0, dtype=np.float32)

    first = _read_rgb(clip_records[0]["frame_path"])
    orig_h, orig_w = first.shape[:2]

    if T < seq_len:
        return pred_xy, orig_h, orig_w

    resized = []
    for r in clip_records:
        img = _read_rgb(r["frame_path"])
        resized.append(_normalize_chw(resize_v2(img)))

    with torch.no_grad():
        for start in range(0, T - seq_len + 1):
            window = resized[start:start + seq_len]
            inp = np.concatenate(window, axis=0)  # (seq_len*3, H, W)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
            out = model(inp_t)[0].cpu()  # (seq_len, H, W)
            for k in range(seq_len):
                coords = heatmap_to_coords(
                    out[k:k + 1].unsqueeze(0), threshold=hm_threshold,
                )[0]
                pred_xy[start + k] = coords.numpy()
    return pred_xy, orig_h, orig_w


def _eval_tracknetv2(checkpoint: str, splits_csv: str, device):
    from models.tracknetv2 import TrackNetV2

    seq_len = TRACKNETV2["seq_len"]
    hm_threshold = TRACKNETV2["heatmap_threshold"]

    model = TrackNetV2(in_dim=seq_len * 3, out_dim=seq_len).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    records = _load_splits(splits_csv, "test")
    groups = _group_by_clip(records)

    pred_xys, gt_xys, pred_vis, gt_vis, vis_cls = [], [], [], [], []
    t0 = time.time()
    total_frames = 0

    for (_, _), clip_records in groups.items():
        if len(clip_records) < seq_len:
            continue
        pred_xy_v2, orig_h, orig_w = _v2_predict_clip(
            model, device, clip_records, seq_len, hm_threshold,
        )
        pred_xy_orig = _v2_scale_pred_to_orig(pred_xy_v2, orig_w, orig_h)

        T = len(clip_records)
        gt = np.zeros((T, 2), dtype=np.float32)
        vis_bool = np.zeros((T,), dtype=bool)
        vis_class = np.zeros((T,), dtype=np.int64)
        for i, r in enumerate(clip_records):
            v = int(r["visibility"])
            x = float(r["x"])
            y = float(r["y"])
            vis_class[i] = v
            if v > 0 and x >= 0 and y >= 0:
                gt[i] = (x, y)
                vis_bool[i] = True
            else:
                gt[i] = (-1.0, -1.0)

        pred_xys.append(pred_xy_orig)
        gt_xys.append(gt)
        pred_vis.append(pred_xy_orig[:, 0] >= 0)
        gt_vis.append(vis_bool)
        vis_cls.append(vis_class)
        total_frames += T

    elapsed = time.time() - t0
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    return (np.concatenate(pred_xys), np.concatenate(gt_xys),
            np.concatenate(pred_vis), np.concatenate(gt_vis),
            np.concatenate(vis_cls), fps)


def _v3_predict_clip(model, device, clip_records, seq_len: int, hm_threshold: float):
    """Run V3 tracker over a clip; returns pred_xy in 288x512 space and (orig_h, orig_w)."""
    from data.dataset_v2 import _read_rgb, _normalize_chw
    from data.preprocessing_v2 import resize_v2, compute_clip_median
    from models.tracknet import heatmap_to_coords

    T = len(clip_records)
    pred_xy = np.full((T, 2), -1.0, dtype=np.float32)

    first = _read_rgb(clip_records[0]["frame_path"])
    orig_h, orig_w = first.shape[:2]

    if T < seq_len:
        return pred_xy, orig_h, orig_w

    frame_paths = [r["frame_path"] for r in clip_records]
    clip_key = f"{clip_records[0]['game']}_{clip_records[0]['clip']}"
    median = compute_clip_median(frame_paths,
                                  cache_dir=os.path.join("cache", "median"),
                                  clip_key=clip_key)

    resized = [_normalize_chw(resize_v2(_read_rgb(p))) for p in frame_paths]

    with torch.no_grad():
        for start in range(0, T - seq_len + 1):
            window = resized[start:start + seq_len]
            inp = np.concatenate(window + [median], axis=0)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
            out = model(inp_t)[0].cpu()  # (seq_len, H, W)
            for k in range(seq_len):
                coords = heatmap_to_coords(
                    out[k:k + 1].unsqueeze(0), threshold=hm_threshold,
                )[0]
                pred_xy[start + k] = coords.numpy()
    return pred_xy, orig_h, orig_w


def _v3_rectify_clip(inpaint_model, device, pred_xy_v2: np.ndarray, L: int):
    """Pipe a per-frame coord sequence through InpaintNet in non-overlapping L-windows.
    pred_xy_v2: (T, 2) coords in 288x512 space with -1 sentinel.
    Returns rectified (T, 2) in 288x512 space.
    """
    from data.preprocessing_v2 import IMG_H_V2, IMG_W_V2

    T = pred_xy_v2.shape[0]
    rectified = pred_xy_v2.copy()
    if T < L:
        return rectified

    inpaint_model.eval()
    with torch.no_grad():
        for start in range(0, T - L + 1, L):
            window = pred_xy_v2[start:start + L].copy()
            missing = (window[:, 0] < 0) | (window[:, 1] < 0)
            mask = missing.astype(np.float32)

            coords_norm = window.copy()
            coords_norm[missing] = 0.0
            coords_norm[:, 0] /= float(IMG_W_V2)
            coords_norm[:, 1] /= float(IMG_H_V2)

            inp = np.concatenate([coords_norm, mask[:, None]], axis=1).astype(np.float32)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)  # (1, L, 3)
            out = inpaint_model(inp_t)[0].cpu().numpy()  # (L, 2)

            # Only overwrite frames the inpaint mask marked as missing.
            for i in range(L):
                if mask[i] > 0:
                    rx = out[i, 0] * IMG_W_V2
                    ry = out[i, 1] * IMG_H_V2
                    rectified[start + i] = (rx, ry)

        # Trailing remainder (if T not divisible by L): inpaint the last L-block.
        if (T % L) != 0:
            start = T - L
            window = pred_xy_v2[start:start + L].copy()
            missing = (window[:, 0] < 0) | (window[:, 1] < 0)
            mask = missing.astype(np.float32)

            coords_norm = window.copy()
            coords_norm[missing] = 0.0
            coords_norm[:, 0] /= float(IMG_W_V2)
            coords_norm[:, 1] /= float(IMG_H_V2)

            inp = np.concatenate([coords_norm, mask[:, None]], axis=1).astype(np.float32)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
            out = inpaint_model(inp_t)[0].cpu().numpy()

            for i in range(L):
                # Only fill remainder frames that weren't already rectified above.
                global_i = start + i
                if mask[i] > 0 and global_i >= (T - (T % L)):
                    rx = out[i, 0] * IMG_W_V2
                    ry = out[i, 1] * IMG_H_V2
                    rectified[global_i] = (rx, ry)

    return rectified


def _eval_tracknetv3(checkpoint: str, splits_csv: str, device):
    """checkpoint is the InpaintNet path; tracker is loaded from CHECKPOINT_DIR."""
    from models.tracknetv3 import TrackNetV3Tracker, InpaintNet

    seq_len = TRACKNETV3_TRACKER["seq_len"]
    hm_threshold = TRACKNETV3_TRACKER["heatmap_threshold"]
    L = TRACKNETV3_INPAINT["seq_len"]

    tracker = TrackNetV3Tracker(seq_len=seq_len).to(device)
    tracker_ckpt = os.path.join(CHECKPOINT_DIR, "tracknetv3_tracker_best.pt")
    tracker.load_state_dict(torch.load(tracker_ckpt, map_location=device))
    tracker.eval()

    inpaint = InpaintNet().to(device)
    inpaint.load_state_dict(torch.load(checkpoint, map_location=device))
    inpaint.eval()

    records = _load_splits(splits_csv, "test")
    groups = _group_by_clip(records)

    pred_xys, gt_xys, pred_vis, gt_vis, vis_cls = [], [], [], [], []
    t0 = time.time()
    total_frames = 0

    for (_, _), clip_records in groups.items():
        if len(clip_records) < seq_len:
            continue
        pred_xy_v2, orig_h, orig_w = _v3_predict_clip(
            tracker, device, clip_records, seq_len, hm_threshold,
        )
        rectified_v2 = _v3_rectify_clip(inpaint, device, pred_xy_v2, L)
        pred_xy_orig = _v2_scale_pred_to_orig(rectified_v2, orig_w, orig_h)

        T = len(clip_records)
        gt = np.zeros((T, 2), dtype=np.float32)
        vis_bool = np.zeros((T,), dtype=bool)
        vis_class = np.zeros((T,), dtype=np.int64)
        for i, r in enumerate(clip_records):
            v = int(r["visibility"])
            x = float(r["x"])
            y = float(r["y"])
            vis_class[i] = v
            if v > 0 and x >= 0 and y >= 0:
                gt[i] = (x, y)
                vis_bool[i] = True
            else:
                gt[i] = (-1.0, -1.0)

        pred_xys.append(pred_xy_orig)
        gt_xys.append(gt)
        pred_vis.append(pred_xy_orig[:, 0] >= 0)
        gt_vis.append(vis_bool)
        vis_cls.append(vis_class)
        total_frames += T

    elapsed = time.time() - t0
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    return (np.concatenate(pred_xys), np.concatenate(gt_xys),
            np.concatenate(pred_vis), np.concatenate(gt_vis),
            np.concatenate(vis_cls), fps)


# ─── Entry point ─────────────────────────────────────────────────────────────

EVAL_FNS = {
    "tracknet":   _eval_tracknet,
    "tracknetv2": _eval_tracknetv2,
    "tracknetv3": _eval_tracknetv3,
    "tracknetv4": _eval_tracknetv4,
    "tracknetv5": _eval_tracknetv5,
    "yolo11m":    _eval_yolo,
}


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {args.model} on test split …")

    if args.model == "yolo11m":
        use_ult = not args.use_fallback
        arrays = _eval_yolo(args.checkpoint, args.splits_csv, device, use_ult)
    else:
        arrays = EVAL_FNS[args.model](args.checkpoint, args.splits_csv, device)

    pred_xy, gt_xy, pred_vis, gt_vis, vis_cls, fps = arrays
    metrics = compute_all_metrics(pred_xy, gt_xy, pred_vis, gt_vis, vis_cls, fps)

    print(f"\n{'─'*40}")
    print(f"Model: {args.model}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"{args.model}_metrics.json")
    with open(out_path, "w") as f:
        json.dump({k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                   for k, v in metrics.items()}, f, indent=2)
    print(f"\nSaved metrics → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True, choices=list(EVAL_FNS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits_csv", default=SPLITS_CSV)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--use_fallback", action="store_true")
    evaluate(parser.parse_args())
