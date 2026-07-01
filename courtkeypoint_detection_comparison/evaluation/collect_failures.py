"""
Rerun the test split and save every wrong or missed court keypoint detection.

For each test frame this decodes the predicted keypoints (same pipeline as
``evaluation.evaluate``), scales them to original image space, and compares
against the GT keypoints.  A keypoint is flagged when, with GT visible:

  * "missed" -- predicted coords are the -1 sentinel (heatmap peak below the
    confidence threshold), i.e. the model produced no detection.
  * "wrong"  -- predicted-visible but the pixel error exceeds the project's
    correctness threshold (PCK threshold, default 7 px).

Outputs go to results/court_failures/<model>/:
  * failures.csv / failures.json -- one row per failed keypoint (ALL failures).
  * <error>_<id>.png             -- annotated frame per failure sample, capped
    at --max_images worst samples (by max failing-keypoint error).

Usage:
    python -m evaluation.collect_failures \
        --model mobilenetv3 --checkpoint checkpoints/mobilenetv3_best.pt
"""

import argparse
import csv
import json
import pathlib

import cv2
import numpy as np
import torch

from data.dataset import CourtKeypointDataset
from data.preprocessing import INPUT_H, INPUT_W, NUM_KEYPOINTS
from evaluation.evaluate import heatmap_to_coords
from evaluation.visualize import _build_model, overlay


def _parse_confidence_threshold(value):
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def collect_failures(args):
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

    out_dir = pathlib.Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(ds)
    failures = []          # one dict per failed keypoint (across all frames)
    failed_samples = []    # (record_id, worst_err, render_payload) per bad frame
    n_wrong = 0
    n_missed = 0

    print(f"Scanning {n_total} {args.split} frames "
          f"(threshold={args.threshold_px}px, conf={args.confidence_threshold}) …")

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
            gt_mask = (gt_kps_orig[:, 0] >= 0)

            sample_failures = []
            for k in range(NUM_KEYPOINTS):
                if not gt_mask[k]:
                    continue  # GT invisible -> nothing to detect
                pred_vis = visible[k]
                if not pred_vis:
                    fail_type = 'missed'
                    err = float('nan')
                else:
                    err = float(np.linalg.norm(pred_kps_orig[k] - gt_kps_orig[k]))
                    if err <= args.threshold_px:
                        continue  # correct
                    fail_type = 'wrong'

                rec = {
                    'id': record['id'],
                    'keypoint': k,
                    'failure_type': fail_type,
                    'pred_x': float(pred_kps_orig[k, 0]),
                    'pred_y': float(pred_kps_orig[k, 1]),
                    'gt_x': float(gt_kps_orig[k, 0]),
                    'gt_y': float(gt_kps_orig[k, 1]),
                    'error_px': err,
                }
                sample_failures.append(rec)
                failures.append(rec)
                if fail_type == 'wrong':
                    n_wrong += 1
                else:
                    n_missed += 1

            if sample_failures:
                # Rank samples by worst localisation error; misses (nan) sort to
                # the top so they are not lost behind small wrong-but-close errors.
                errs = [f['error_px'] for f in sample_failures]
                finite = [e for e in errs if e == e]
                worst = max(finite) if finite else float('inf')
                types = {f['failure_type'] for f in sample_failures}
                failed_samples.append((
                    record['id'], worst, types,
                    (img_bgr, pred_kps_orig, gt_kps_orig, pred_center_orig),
                ))

            if (idx + 1) % 100 == 0:
                print(f"  [{idx + 1}/{n_total}] failures so far: "
                      f"{len(failed_samples)} frames")

    # ── manifest (ALL failures, never truncated) ─────────────────────────────
    wrong_dir = out_dir / 'wrong'
    missed_dir = out_dir / 'missed'
    wrong_dir.mkdir(parents=True, exist_ok=True)
    missed_dir.mkdir(parents=True, exist_ok=True)

    def _write_manifest(folder, rows, n_kps_label):
        with open(folder / 'failures.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'id', 'keypoint', 'failure_type',
                'pred_x', 'pred_y', 'gt_x', 'gt_y', 'error_px',
            ])
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            'model': args.model,
            'checkpoint': args.checkpoint,
            'split': args.split,
            'threshold_px': args.threshold_px,
            'confidence_threshold': args.confidence_threshold,
            'n_total_samples': n_total,
            'n_failure_samples': len(failed_samples),
            'n_wrong_keypoints': n_wrong,
            'n_missed_keypoints': n_missed,
            'max_images_rendered': args.max_images,
            'images_capped': len(failed_samples) > args.max_images,
            n_kps_label: len(rows),
            'failures': rows,
        }
        with open(folder / 'failures.json', 'w') as f:
            json.dump(summary, f, indent=2)

    wrong_rows = [f for f in failures if f['failure_type'] == 'wrong']
    missed_rows = [f for f in failures if f['failure_type'] == 'missed']
    _write_manifest(wrong_dir, wrong_rows, 'n_keypoints_in_folder')
    _write_manifest(missed_dir, missed_rows, 'n_keypoints_in_folder')

    # ── annotated images (capped at the worst --max_images samples) ──────────
    # A frame goes into wrong/ if it has any wrong keypoint and into missed/ if
    # it has any missed keypoint; frames with both land in both folders.
    failed_samples.sort(key=lambda t: -t[1])
    to_render = failed_samples[:args.max_images]
    n_wrong_imgs = 0
    n_missed_imgs = 0
    for rid, worst, types, (img_bgr, pred_kps_orig, gt_kps_orig, pred_center_orig) in to_render:
        vis = overlay(img_bgr, pred_kps_orig, gt_kps_orig, pred_center_orig, args.model)
        err_tag = f"{worst:07.1f}" if np.isfinite(worst) else "miss"
        name = f"{args.model}_err{err_tag}_{rid}.png"
        if 'wrong' in types:
            cv2.imwrite(str(wrong_dir / name), vis)
            n_wrong_imgs += 1
        if 'missed' in types:
            cv2.imwrite(str(missed_dir / name), vis)
            n_missed_imgs += 1

    capped = len(failed_samples) > args.max_images
    print()
    print(f"Total test samples:        {n_total}")
    print(f"Failure samples:           {len(failed_samples)} "
          f"({100.0 * len(failed_samples) / n_total:.1f}%)")
    print(f"Wrong keypoints (>{args.threshold_px}px): {n_wrong}")
    print(f"Missed keypoints (sentinel): {n_missed}")
    print(f"Manifest wrong/:           {len(wrong_rows)} rows → {wrong_dir}")
    print(f"Manifest missed/:          {len(missed_rows)} rows → {missed_dir}")
    print(f"Annotated images:          wrong/ {n_wrong_imgs}, missed/ {n_missed_imgs}"
          + (f"  (CAPPED at {args.max_images} worst frames; {len(failed_samples)} total)" if capped else ""))


def main():
    parser = argparse.ArgumentParser(
        description="Collect every wrong/missed court keypoint on the test split."
    )
    parser.add_argument('--model', required=True,
                        choices=['tracknet_court', 'resnet50', 'hrnet', 'mobilenetv3'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--threshold_px', type=float, default=7.0,
                        help="Pixel error above which a detection is 'wrong' "
                             "(default 7, the project PCK threshold).")
    parser.add_argument('--confidence_threshold',
                        type=_parse_confidence_threshold, default=0.3)
    parser.add_argument('--max_images', type=int, default=100,
                        help="Cap on annotated images saved (worst by error). "
                             "The manifest always lists all failures.")
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--output_dir',
        default=str(pathlib.Path(__file__).parent.parent / 'results' / 'court_failures'),
    )
    args = parser.parse_args()
    collect_failures(args)


if __name__ == '__main__':
    main()
