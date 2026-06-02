"""
Visualise court keypoint predictions on N random test samples.

Usage:
    python -m evaluation.visualize --model hrnet          --checkpoint checkpoints/hrnet_best.pt
    python -m evaluation.visualize --model resnet50       --checkpoint checkpoints/resnet50_best.pt --num_samples 5
    python -m evaluation.visualize --model tracknet_court --checkpoint checkpoints/tracknet_court_best.pt --num_samples 3

Green = ground-truth, Red = prediction. Keypoints are indexed 0..13;
channel 14 (court centre) is drawn as a yellow cross.
"""

import argparse
import pathlib
import random

import cv2
import numpy as np
import torch

from data.dataset import CourtKeypointDataset
from data.preprocessing import INPUT_H, INPUT_W, NUM_KEYPOINTS
from evaluation.evaluate import heatmap_to_coords
from train.config import TRACKNET_COURT, RESNET50, HRNET, MOBILENETV3


GT_COLOR    = (0, 255, 0)     # green (BGR)
PRED_COLOR  = (0, 0, 255)     # red
CENTER_COLOR = (0, 255, 255)  # yellow


def _build_model(name: str):
    if name == 'tracknet_court':
        from models.tracknet_court import TrackNetCourt
        return TrackNetCourt(), TRACKNET_COURT
    if name == 'resnet50':
        from models.resnet50_pose import ResNet50Pose
        return ResNet50Pose(), RESNET50
    if name == 'hrnet':
        from models.hrnet import HRNetPose
        return HRNetPose(), HRNET
    if name == 'mobilenetv3':
        from models.mobilenetv3_pose import MobileNetV3SmallPose
        return MobileNetV3SmallPose(), MOBILENETV3
    raise ValueError(f"Unknown model: {name}")


def _draw_kp(img, x, y, color, label=None, radius=5):
    if x < 0 or y < 0:
        return
    cx, cy = int(round(x)), int(round(y))
    cv2.circle(img, (cx, cy), radius, color, 2)
    if label is not None:
        cv2.putText(img, label, (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def _draw_cross(img, x, y, color, size=8):
    if x < 0 or y < 0:
        return
    cx, cy = int(round(x)), int(round(y))
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, 2)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, 2)


def overlay(img_bgr, pred_kps_orig, gt_kps_orig, pred_center_orig, model_name):
    """Draw GT + predicted keypoints on a copy of *img_bgr* (original size)."""
    out = img_bgr.copy()
    for i in range(NUM_KEYPOINTS):
        gx, gy = gt_kps_orig[i]
        px, py = pred_kps_orig[i]
        _draw_kp(out, gx, gy, GT_COLOR, label=str(i), radius=6)
        _draw_kp(out, px, py, PRED_COLOR, radius=4)

    cx, cy = pred_center_orig
    _draw_cross(out, cx, cy, CENTER_COLOR)

    cv2.putText(out, f"{model_name}  GT=green pred=red", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def visualize(args):
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
    n_total = len(ds)
    if args.num_samples > n_total:
        print(f"Requested {args.num_samples} > available {n_total}; clamping.")
        args.num_samples = n_total

    rng = random.Random(args.seed)
    indices = rng.sample(range(n_total), args.num_samples)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stride = cfg['stride']
    images_dir = ds.images_dir

    with torch.no_grad():
        for idx in indices:
            img_tensor, _, kps_orig = ds[idx]
            inp = img_tensor.unsqueeze(0).to(device)
            pred = torch.sigmoid(model(inp))[0].cpu().numpy()  # (15, h, w)

            # Decode predictions in INPUT_W x INPUT_H space
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

            # Load original-resolution image to draw on
            record = ds.records[idx]
            img_path = images_dir / (record['id'] + '.png')
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"Skipping missing image: {img_path}")
                continue
            orig_h, orig_w = img_bgr.shape[:2]

            # Scale predictions from input space → original image space
            sx = orig_w / INPUT_W
            sy = orig_h / INPUT_H
            pred_kps_orig = pred_kps_input.copy()
            visible = (pred_kps_orig[:, 0] >= 0) & (pred_kps_orig[:, 1] >= 0)
            pred_kps_orig[:, 0] = np.where(visible, pred_kps_orig[:, 0] * sx, -1.0)
            pred_kps_orig[:, 1] = np.where(visible, pred_kps_orig[:, 1] * sy, -1.0)
            pred_center_orig = (cx * sx, cy * sy) if cx >= 0 else (-1.0, -1.0)

            gt_kps_orig = kps_orig.numpy()  # already original pixel space

            vis = overlay(img_bgr, pred_kps_orig, gt_kps_orig,
                          pred_center_orig, args.model)
            out_path = out_dir / f"{args.model}_{record['id']}.png"
            cv2.imwrite(str(out_path), vis)
            print(f"  → {out_path}")

    print(f"\nSaved {len(indices)} visualisation(s) to {out_dir}")


def _parse_confidence_threshold(value):
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def main():
    parser = argparse.ArgumentParser(
        description="Visualise court keypoint predictions on N test samples."
    )
    parser.add_argument('--model', required=True,
                        choices=['tracknet_court', 'resnet50', 'hrnet', 'mobilenetv3'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--num_samples', '-n', type=int, default=1,
                        help='Number of samples to visualise (default: 1).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for sample selection.')
    parser.add_argument('--confidence_threshold',
                        type=_parse_confidence_threshold, default=0.3)
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--output_dir',
        default=str(pathlib.Path(__file__).parent.parent / 'results' / 'visualizations'),
    )
    args = parser.parse_args()
    visualize(args)


if __name__ == '__main__':
    main()
