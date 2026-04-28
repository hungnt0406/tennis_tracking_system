"""
Run evaluation on the test split for a single model.

Usage:
    python -m evaluation.evaluate --model tracknet  --checkpoint checkpoints/tracknet_best.pt
    python -m evaluation.evaluate --model skeeptrack --checkpoint checkpoints/skeeptrack_best.pt
    python -m evaluation.evaluate --model yolo11m    --checkpoint checkpoints/yolo11m_best.pt
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import TrackNetDataset, SKeepTrackDataset, YOLODataset
from data.preprocessing import IMG_H, IMG_W
from evaluation.metrics import compute_all_metrics
from train.config import SPLITS_CSV, CHECKPOINT_DIR, RESULTS_DIR


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

            # Scale heatmap coords → pixel coords (heatmap is already 256×256)
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


def _eval_skeeptrack(checkpoint: str, splits_csv: str, device):
    from models.skeeptrack import SKeepTrack
    from train.config import SKEEPTRACK

    model = SKeepTrack(k=SKEEPTRACK["k_candidates"], pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ds = SKeepTrackDataset(splits_csv, "test", augment=False)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    pred_xys, gt_xys, pred_vis, gt_vis, vis_cls = [], [], [], [], []
    t0 = time.time()
    total_frames = 0

    with torch.no_grad():
        for t1, t2, hm1, hm2, coords1_gt, coords2_gt in dl:
            t1, t2 = t1.to(device), t2.to(device)
            _, _, _, pred_norm = model(t1, t2)   # (B, 2) normalised

            # Convert normalised → pixel (using IMG_W/IMG_H as reference)
            pred_px = pred_norm.cpu().numpy() * np.array([[IMG_W, IMG_H]])
            gt_norm  = coords2_gt[:, :2].numpy()
            gt_px    = gt_norm * np.array([[IMG_W, IMG_H]])

            gt_visible = coords2_gt[:, 2].numpy().astype(bool)
            vis_label  = coords2_gt[:, 2].numpy().astype(int)

            # Presence: mark as detected when pred confidence > threshold
            # (use score of best candidate — stored in pred_px not available directly;
            #  fall back to assuming always detected)
            p_vis = np.ones(len(pred_px), dtype=bool)

            total_frames += t2.shape[0]
            pred_xys.append(pred_px)
            gt_xys.append(gt_px)
            pred_vis.append(p_vis)
            gt_vis.append(gt_visible)
            vis_cls.append(vis_label)

    elapsed = time.time() - t0
    fps = total_frames / elapsed

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


# ─── Entry point ─────────────────────────────────────────────────────────────

EVAL_FNS = {
    "tracknet":  _eval_tracknet,
    "skeeptrack": _eval_skeeptrack,
    "yolo11m":   _eval_yolo,
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
