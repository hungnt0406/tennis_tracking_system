"""
Visualise test frames whose per-frame mean reprojection error exceeds a
threshold (default 100 cm).

For each model and checkpoint, this iterates the test split, estimates a
predicted homography per frame, projects the visible GT keypoints through it
into court meters, and compares against the canonical court template. Frames
with mean error above ``--threshold_cm`` are saved with GT (green) and
predicted (red) keypoints overlaid, plus the error value in the title.

Usage:
    python -m evaluation.visualize_high_error \
        --model mobilenetv3 \
        --checkpoint checkpoints/mobilenetv3_best.pt \
        --threshold_cm 100
"""

import argparse
import pathlib

import cv2
import numpy as np
import torch

from data.dataset import CourtKeypointDataset
from data.preprocessing import INPUT_H, INPUT_W, NUM_KEYPOINTS
from evaluation.evaluate import heatmap_to_coords
from evaluation.visualize import _build_model, overlay
from homography.court_template import COURT_KEYPOINTS_M
from homography.estimate import estimate_homography


def _per_frame_reproj_err_cm(gt_kps_orig, mask, h_img_to_court):
    """Mean reprojection error in cm for one frame, or None if no homography."""
    if h_img_to_court is None:
        return None
    valid = mask & np.isfinite(gt_kps_orig).all(axis=1) & (gt_kps_orig[:, 0] >= 0)
    if not valid.any():
        return None
    pts = gt_kps_orig[valid].astype(np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, h_img_to_court.astype(np.float64)).reshape(-1, 2)
    errors_m = np.linalg.norm(projected - COURT_KEYPOINTS_M[valid], axis=1)
    return float((errors_m * 100.0).mean())


def visualize_high_error(args):
    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Loading {args.model} on {device} …")

    model, cfg = _build_model(args.model)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)

    ds = CourtKeypointDataset(
        args.split, augment=False,
        stride=cfg['stride'],
        gaussian_radius=cfg['gaussian_radius'],
        imagenet_norm=cfg.get('use_imagenet_norm', False),
    )
    stride = cfg['stride']
    images_dir = ds.images_dir

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(ds)
    n_failed_h = 0
    high_error_frames = []  # list of (record_id, err_cm)

    print(f"Scanning {n_total} {args.split} frames for mean_reproj_err > {args.threshold_cm} cm …")

    with torch.no_grad():
        for idx in range(n_total):
            img_tensor, _, kps_orig_t = ds[idx]
            inp = img_tensor.unsqueeze(0).to(device)
            pred = torch.sigmoid(model(inp))[0].cpu().numpy()  # (15, h, w)

            pred_kps_input = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
            for k in range(NUM_KEYPOINTS):
                px, py = heatmap_to_coords(
                    pred[k], stride,
                    confidence_threshold=args.confidence_threshold,
                )
                pred_kps_input[k] = [px, py]
            cx, cy = heatmap_to_coords(
                pred[14], stride,
                confidence_threshold=args.confidence_threshold,
            )

            record = ds.records[idx]
            img_path = images_dir / (record['id'] + '.png')
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  skipping missing image: {img_path}")
                continue
            orig_h, orig_w = img_bgr.shape[:2]

            sx = orig_w / INPUT_W
            sy = orig_h / INPUT_H
            pred_kps_orig = pred_kps_input.copy()
            visible = (pred_kps_orig[:, 0] >= 0) & (pred_kps_orig[:, 1] >= 0)
            pred_kps_orig[:, 0] = np.where(visible, pred_kps_orig[:, 0] * sx, -1.0)
            pred_kps_orig[:, 1] = np.where(visible, pred_kps_orig[:, 1] * sy, -1.0)
            pred_center_orig = (cx * sx, cy * sy) if cx >= 0 else (-1.0, -1.0)

            gt_kps_orig = kps_orig_t.numpy()
            mask = (gt_kps_orig[:, 0] >= 0)

            h_img_to_court, _, _ = estimate_homography(pred_kps_orig)
            err_cm = _per_frame_reproj_err_cm(gt_kps_orig, mask, h_img_to_court)
            if err_cm is None:
                n_failed_h += 1
                continue
            if err_cm <= args.threshold_cm:
                continue

            high_error_frames.append((record['id'], err_cm))

            vis = overlay(img_bgr, pred_kps_orig, gt_kps_orig,
                          pred_center_orig, args.model)
            cv2.putText(
                vis, f"reproj_err: {err_cm:.1f} cm",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
            out_name = f"{args.model}_err{err_cm:07.1f}cm_{record['id']}.png"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), vis)
            print(f"  [{idx+1}/{n_total}] {record['id']}  err={err_cm:.1f} cm  → {out_path.name}")

    high_error_frames.sort(key=lambda t: -t[1])
    print()
    print(f"Frames above {args.threshold_cm} cm: {len(high_error_frames)} / {n_total}")
    print(f"Frames with failed homography: {n_failed_h}")
    if high_error_frames:
        print(f"Worst 10:")
        for rid, err in high_error_frames[:10]:
            print(f"  {err:>8.1f} cm   {rid}")
    print(f"\nSaved visualisations to {out_dir}")


def _parse_confidence_threshold(value):
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def main():
    parser = argparse.ArgumentParser(
        description="Visualise test frames with mean reprojection error above a threshold."
    )
    parser.add_argument('--model', required=True,
                        choices=['tracknet_court', 'resnet50', 'hrnet', 'mobilenetv3'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--threshold_cm', type=float, default=100.0,
                        help='Save frames whose mean reproj error exceeds this (cm).')
    parser.add_argument('--confidence_threshold',
                        type=_parse_confidence_threshold, default=0.3)
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--output_dir',
        default=str(pathlib.Path(__file__).parent.parent / 'results' / 'high_error_frames'),
    )
    args = parser.parse_args()
    visualize_high_error(args)


if __name__ == '__main__':
    main()
