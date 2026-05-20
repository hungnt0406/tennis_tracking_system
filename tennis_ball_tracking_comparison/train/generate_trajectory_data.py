"""
Generate trajectory training data for InpaintNet (TrackNetV3 stage 2 setup).

Runs the stage-1 tracker on train+val splits and writes:
  cache/trajectory_data_train.npz
  cache/trajectory_data_val.npz

Each npz contains keys:
  coords    (N, L, 2)  predicted normalized (x, y), 0 for missing frames
  gt_coords (N, L, 2)  ground-truth normalized (x, y)
  mask      (N, L)     1 = missing or far from GT, 0 = good prediction

Usage:
    python -m train.generate_trajectory_data \
        --checkpoint checkpoints/tracknetv3_tracker_best.pt
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from data.dataset import _load_splits, _group_by_clip
from data.dataset_v2 import _read_rgb, _normalize_chw
from data.preprocessing_v2 import (
    IMG_H_V2, IMG_W_V2, resize_v2, compute_clip_median,
)
from models.tracknet import heatmap_to_coords
from models.tracknetv3 import TrackNetV3Tracker
from train.config import TRACKNETV3_TRACKER, TRACKNETV3_INPAINT, SPLITS_CSV, SEED


CACHE_DIR = "cache"
MEDIAN_CACHE_DIR = os.path.join(CACHE_DIR, "median")
TRAJ_LEN = TRACKNETV3_INPAINT["seq_len"]  # L = 16


def _scaled_gt_xy(x: float, y: float, vis: int, orig_w: int, orig_h: int):
    if vis <= 0 or x < 0 or y < 0:
        return -1.0, -1.0
    return x / orig_w * IMG_W_V2, y / orig_h * IMG_H_V2


def _predict_clip(model, device, clip_records, seq_len, hm_threshold):
    """Run tracker over a single clip; return predicted xy in 288x512 space.

    Predictions are produced via sliding windows of seq_len frames, with
    last-write-wins stitching. Returns:
        pred_xy   (T, 2) float32 — -1 marks missing.
        gt_xy     (T, 2) float32 — -1 marks invisible/missing GT.
    where T = len(clip_records).
    """
    T = len(clip_records)
    pred_xy = np.full((T, 2), -1.0, dtype=np.float32)
    gt_xy   = np.full((T, 2), -1.0, dtype=np.float32)

    frame_paths = [r["frame_path"] for r in clip_records]

    # GT first (cheap, no model required).
    first = _read_rgb(frame_paths[0])
    orig_h, orig_w = first.shape[:2]
    for i, r in enumerate(clip_records):
        sx, sy = _scaled_gt_xy(float(r["x"]), float(r["y"]),
                                int(r["visibility"]), orig_w, orig_h)
        gt_xy[i] = (sx, sy)

    if T < seq_len:
        return pred_xy, gt_xy

    clip_key = f"{clip_records[0]['game']}_{clip_records[0]['clip']}"
    median = compute_clip_median(frame_paths, cache_dir=MEDIAN_CACHE_DIR,
                                  clip_key=clip_key)  # (3, H, W)

    # Pre-resize all frames once.
    resized = []
    for p in frame_paths:
        img = _read_rgb(p)
        resized.append(_normalize_chw(resize_v2(img)))  # (3, H, W) float32

    model.eval()
    with torch.no_grad():
        for start in range(0, T - seq_len + 1):
            window = resized[start:start + seq_len]
            inp = np.concatenate(window + [median], axis=0)  # ((L+1)*3, H, W)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
            out = model(inp_t)[0].cpu()  # (seq_len, H, W)
            for k in range(seq_len):
                coords = heatmap_to_coords(
                    out[k:k + 1].unsqueeze(0), threshold=hm_threshold,
                )[0]
                pred_xy[start + k] = coords.numpy()

    return pred_xy, gt_xy


def _build_windows(pred_xy, gt_xy, L, coord_threshold):
    """Slice clip arrays into windows of length L. Returns (coords, gts, masks)."""
    T = pred_xy.shape[0]
    if T < L:
        return [], [], []

    coords_list, gts_list, masks_list = [], [], []
    for start in range(0, T - L + 1):
        p = pred_xy[start:start + L].copy()
        g = gt_xy[start:start + L].copy()

        missing_pred = (p[:, 0] < 0) | (p[:, 1] < 0)
        # Distance only meaningful when both pred and gt valid.
        diff = p - g
        dist = np.sqrt((diff ** 2).sum(axis=1))
        far = (dist > coord_threshold) & (~missing_pred)
        mask = (missing_pred | far).astype(np.float32)

        # Normalize and zero-out missing predictions.
        p_norm = p.copy()
        p_norm[missing_pred] = 0.0
        p_norm[:, 0] /= float(IMG_W_V2)
        p_norm[:, 1] /= float(IMG_H_V2)

        # GT: invisible frames mapped to 0 too.
        g_norm = g.copy()
        invis = (g_norm[:, 0] < 0) | (g_norm[:, 1] < 0)
        g_norm[invis] = 0.0
        g_norm[:, 0] /= float(IMG_W_V2)
        g_norm[:, 1] /= float(IMG_H_V2)

        coords_list.append(p_norm.astype(np.float32))
        gts_list.append(g_norm.astype(np.float32))
        masks_list.append(mask)
    return coords_list, gts_list, masks_list


def _process_split(model, device, splits_csv, split, args):
    records = _load_splits(splits_csv, split)
    if args.max_samples is not None:
        records = records[:args.max_samples]
    groups = _group_by_clip(records)

    all_coords, all_gts, all_masks = [], [], []
    seq_len = TRACKNETV3_TRACKER["seq_len"]
    hm_threshold = TRACKNETV3_TRACKER["heatmap_threshold"]

    for (game, clip), clip_records in tqdm(groups.items(),
                                            desc=f"infer {split}"):
        if len(clip_records) < TRAJ_LEN:
            continue
        pred_xy, gt_xy = _predict_clip(model, device, clip_records,
                                        seq_len, hm_threshold)
        coords, gts, masks = _build_windows(pred_xy, gt_xy, TRAJ_LEN,
                                             args.coord_threshold)
        all_coords.extend(coords)
        all_gts.extend(gts)
        all_masks.extend(masks)

    if not all_coords:
        print(f"  No windows produced for split={split}; skipping save.")
        return

    coords_arr = np.stack(all_coords, axis=0).astype(np.float32)
    gts_arr    = np.stack(all_gts,    axis=0).astype(np.float32)
    masks_arr  = np.stack(all_masks,  axis=0).astype(np.float32)

    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CACHE_DIR, f"trajectory_data_{split}.npz")
    np.savez(out_path, coords=coords_arr, gt_coords=gts_arr, mask=masks_arr)
    print(f"  Wrote {out_path} | windows={coords_arr.shape[0]} "
          f"mean_mask={masks_arr.mean():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_csv", default=SPLITS_CSV)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to stage-1 tracker checkpoint")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit train+val records to N each")
    parser.add_argument("--coord_threshold", type=float, default=5.0,
                        help="Pixel distance to mark a prediction as missing")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    seq_len = TRACKNETV3_TRACKER["seq_len"]
    model = TrackNetV3Tracker(seq_len=seq_len).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint {args.checkpoint}")

    os.makedirs(MEDIAN_CACHE_DIR, exist_ok=True)
    for split in ("train", "val"):
        _process_split(model, device, args.splits_csv, split, args)


if __name__ == "__main__":
    main()
