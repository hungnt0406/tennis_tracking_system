"""
Run evaluation on the test split for a single court keypoint detection model.

Usage:
    python -m evaluation.evaluate --model tracknet_court --checkpoint checkpoints/tracknet_court_best.pt
    python -m evaluation.evaluate --model resnet50       --checkpoint checkpoints/resnet50_best.pt
    python -m evaluation.evaluate --model hrnet          --checkpoint checkpoints/hrnet_best.pt
"""

import argparse
import json
import pathlib
import time

import cv2
import numpy as np
import torch

from data.dataset import CourtKeypointDataset
from data.preprocessing import line_intersection
from train.config import TRACKNET_COURT, RESNET50, HRNET, MOBILENETV3
from evaluation.metrics import compute_all_metrics
from evaluation.refine import refine_keypoint
from homography.estimate import estimate_homography


# ─── Heatmap decoding ────────────────────────────────────────────────────────

def heatmap_to_coords(heatmap_ch, stride, confidence_threshold=None):
    """
    heatmap_ch: (H, W) numpy float
    Returns (x, y) in output-stride pixel space, then scale by stride.
    Sub-pixel refinement via local 3x3 weighted mean.
    """
    H, W = heatmap_ch.shape
    flat_idx = heatmap_ch.argmax()
    fy, fx = np.unravel_index(flat_idx, (H, W))
    peak = float(heatmap_ch[fy, fx])
    if confidence_threshold is not None and peak < confidence_threshold:
        return -1.0, -1.0

    # sub-pixel: 3x3 weighted mean
    y1, y2 = max(0, fy - 1), min(H, fy + 2)
    x1, x2 = max(0, fx - 1), min(W, fx + 2)
    patch = heatmap_ch[y1:y2, x1:x2]
    total = patch.sum()
    if total > 1e-8:
        ys = np.arange(y1, y2)
        xs = np.arange(x1, x2)
        fy_sub = (patch.sum(axis=1) @ ys) / total
        fx_sub = (patch.sum(axis=0) @ xs) / total
    else:
        fy_sub, fx_sub = float(fy), float(fx)
    return fx_sub * stride, fy_sub * stride  # x, y in INPUT_W x INPUT_H space


# ─── Shared evaluation loop ───────────────────────────────────────────────────

def _eval_model(model, cfg, checkpoint_path, split, batch_size, device,
                max_samples=None, confidence_threshold=0.3,
                refine=False, refine_crop_size=40, refine_max_drift=20.0):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)

    ds = CourtKeypointDataset(split, augment=False,
                              stride=cfg['stride'],
                              gaussian_radius=cfg['gaussian_radius'],
                              imagenet_norm=cfg.get('use_imagenet_norm', False),
                              max_samples=max_samples)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Count params
    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    stride = cfg['stride']
    input_h, input_w = cfg['input_h'], cfg['input_w']

    all_pred_kps = []    # list of (14, 2) arrays in input (360x640) space
    all_gt_kps = []      # same
    all_masks = []       # (14,) bool
    all_pred_center = []
    all_gt_center = []
    all_center_mask = []

    t0 = time.time()
    with torch.no_grad():
        for imgs, heatmaps, kps_orig in loader:
            imgs = imgs.to(device)
            preds = torch.sigmoid(model(imgs)).cpu().numpy()  # (B, 15, H, W)
            kps_np = kps_orig.numpy()  # (B, 14, 2)
            B = preds.shape[0]
            for b in range(B):
                pred_kps_b = np.zeros((14, 2), dtype=np.float32)
                for k in range(14):
                    px, py = heatmap_to_coords(
                        preds[b, k], stride,
                        confidence_threshold=confidence_threshold,
                    )
                    pred_kps_b[k] = [px, py]

                # court center from channel 14
                px_c, py_c = heatmap_to_coords(
                    preds[b, 14], stride,
                    confidence_threshold=confidence_threshold,
                )

                gt_b = kps_np[b]  # (14, 2)
                mask_b = (gt_b[:, 0] >= 0)  # visible if x >= 0

                # For gt center: compute from gt corners if all visible
                corners_visible = mask_b[:4].all()
                if corners_visible:
                    # line intersection of (kp0,kp2) and (kp1,kp3) — in original pixel space
                    # but we need it in input space: scale by (input_w/orig_w, input_h/orig_h)
                    # Actually kps_orig is in original pixel space; for center metric,
                    # we scale both pred and gt to a comparable space.
                    # Simplest: just use the heatmap-decoded pred vs the 360x640-space gt
                    # (kps_orig are original space, but we don't have orig dims here easily)
                    # Use -1,-1 as placeholder for gt center to skip this in metrics
                    gt_center_b = np.array([-1.0, -1.0])
                    center_visible = False
                else:
                    gt_center_b = np.array([-1.0, -1.0])
                    center_visible = False

                all_pred_kps.append(pred_kps_b)
                all_gt_kps.append(gt_b)
                all_masks.append(mask_b)
                all_pred_center.append(np.array([px_c, py_c]))
                all_gt_center.append(gt_center_b)
                all_center_mask.append(center_visible)

    pred_kps_arr    = np.stack(all_pred_kps)    # (N, 14, 2)
    gt_kps_arr      = np.stack(all_gt_kps)      # (N, 14, 2) — original pixel space
    masks_arr       = np.stack(all_masks)        # (N, 14) bool
    pred_center_arr = np.stack(all_pred_center)
    gt_center_arr   = np.stack(all_gt_center)
    center_mask_arr = np.array(all_center_mask)

    # NOTE: pred_kps_arr is in input (input_h x input_w) space; gt_kps_arr is in original
    # image space.  Scale predictions up so both are in the same pixel space.
    # TODO: pass orig_dims through dataset; for v1, assume all images are 1280x720
    # (most common resolution in TennisCourtDetector dataset).
    ORIG_W, ORIG_H = 1280, 720
    pred_visible = (pred_kps_arr[:, :, 0] >= 0) & (pred_kps_arr[:, :, 1] >= 0)
    pred_kps_arr[:, :, 0] = np.where(
        pred_visible, pred_kps_arr[:, :, 0] * (ORIG_W / input_w), -1.0
    )
    pred_kps_arr[:, :, 1] = np.where(
        pred_visible, pred_kps_arr[:, :, 1] * (ORIG_H / input_h), -1.0
    )

    pred_center_visible = (pred_center_arr[:, 0] >= 0) & (pred_center_arr[:, 1] >= 0)
    pred_center_arr[:, 0] = np.where(
        pred_center_visible, pred_center_arr[:, 0] * (ORIG_W / input_w), -1.0
    )
    pred_center_arr[:, 1] = np.where(
        pred_center_visible, pred_center_arr[:, 1] * (ORIG_H / input_h), -1.0
    )

    if refine:
        n_attempted = 0
        n_success = 0
        for i in range(pred_kps_arr.shape[0]):
            img_path = ds.images_dir / (ds.records[i]["id"] + ".png")
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            for k in range(14):
                if not pred_visible[i, k]:
                    continue
                n_attempted += 1
                x_new, y_new, ok = refine_keypoint(
                    img_bgr,
                    float(pred_kps_arr[i, k, 0]),
                    float(pred_kps_arr[i, k, 1]),
                    crop_size=refine_crop_size,
                    max_drift=refine_max_drift,
                )
                pred_kps_arr[i, k, 0] = x_new
                pred_kps_arr[i, k, 1] = y_new
                if ok:
                    n_success += 1

            corners = pred_kps_arr[i, :4]
            if (corners[:, 0] >= 0).all() and (corners[:, 1] >= 0).all():
                pt = line_intersection(
                    ((float(corners[0, 0]), float(corners[0, 1])),
                     (float(corners[3, 0]), float(corners[3, 1]))),
                    ((float(corners[1, 0]), float(corners[1, 1])),
                     (float(corners[2, 0]), float(corners[2, 1]))),
                )
                if pt is not None and np.isfinite(pt[0]) and np.isfinite(pt[1]):
                    pred_center_arr[i, 0] = pt[0]
                    pred_center_arr[i, 1] = pt[1]

        pct = (100.0 * n_success / n_attempted) if n_attempted else 0.0
        print(f"Refinement: {n_success}/{n_attempted} keypoints refined ({pct:.1f}%)")

    elapsed = time.time() - t0
    fps = len(ds) / elapsed

    homographies = [estimate_homography(kps)[0] for kps in pred_kps_arr]

    metrics = compute_all_metrics(
        pred_kps_arr, gt_kps_arr, masks_arr,
        pred_center_arr, gt_center_arr, center_mask_arr,
        fps=fps, params_M=params_M, homographies=homographies,
    )
    return metrics


def _parse_confidence_threshold(value):
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a court keypoint detection model on a dataset split."
    )
    parser.add_argument('--model', required=True,
                        choices=['tracknet_court', 'resnet50', 'hrnet', 'mobilenetv3'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--confidence_threshold', type=_parse_confidence_threshold, default=0.3)
    parser.add_argument('--refine', action='store_true', default=False)
    parser.add_argument('--refine_crop_size', type=int, default=40)
    parser.add_argument('--refine_max_drift', type=float, default=5)
    args = parser.parse_args()

    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Evaluating {args.model} on {args.split} split … (device={device})")

    if args.model == 'tracknet_court':
        from models.tracknet_court import TrackNetCourt
        model = TrackNetCourt()
        cfg = TRACKNET_COURT
    elif args.model == 'resnet50':
        from models.resnet50_pose import ResNet50Pose
        model = ResNet50Pose()
        cfg = RESNET50
    elif args.model == 'mobilenetv3':
        from models.mobilenetv3_pose import MobileNetV3SmallPose
        model = MobileNetV3SmallPose()
        cfg = MOBILENETV3
    else:
        from models.hrnet import HRNetPose
        model = HRNetPose()
        cfg = HRNET

    metrics = _eval_model(
        model, cfg, args.checkpoint, args.split, args.batch_size, device,
        max_samples=args.max_samples,
        confidence_threshold=args.confidence_threshold,
        refine=args.refine,
        refine_crop_size=args.refine_crop_size,
        refine_max_drift=args.refine_max_drift,
    )

    print(f"\n{'─' * 40}")
    print(f"Model: {args.model}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out_dir = pathlib.Path(__file__).parent.parent / 'results'
    out_dir.mkdir(exist_ok=True)
    suffix = '_refined' if args.refine else ''
    out_path = out_dir / f'{args.model}_metrics{suffix}.json'
    with open(out_path, 'w') as f:
        json.dump({k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                   for k, v in metrics.items()}, f, indent=2)
    print(f"\nSaved metrics → {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
