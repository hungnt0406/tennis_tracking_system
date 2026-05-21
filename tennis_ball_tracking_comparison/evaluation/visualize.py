"""
Visualise model predictions on sample test frames.

Usage:
    python -m evaluation.visualize --model tracknet    --checkpoint checkpoints/tracknet_best.pt
    python -m evaluation.visualize --model tracknetv4  --checkpoint checkpoints/tracknetv4_best.pt
    python -m evaluation.visualize --model tracknetv5  --checkpoint checkpoints/tracknetv5_best.pt
    python -m evaluation.visualize --model yolo11m     --checkpoint checkpoints/yolo11m_best.pt
"""

import argparse
import os

import cv2
import numpy as np
import torch

from data.dataset import _load_splits
from data.preprocessing import IMG_H, IMG_W
from train.config import SPLITS_CSV, RESULTS_DIR, CHECKPOINT_DIR


def _load_tracknet(checkpoint, device):
    from models.tracknet import TrackNet, heatmap_to_coords
    model = TrackNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    def predict(triplet_paths):
        import cv2 as _cv2
        from data.preprocessing import resize_frame, normalize
        frames = []
        for p in triplet_paths:
            img = _cv2.imread(p)
            img = _cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(resize_frame(img))
        tensor = np.concatenate([normalize(f) for f in frames], axis=0)
        inp = torch.from_numpy(tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            hm = model(inp)
        coords = heatmap_to_coords(hm.cpu(), threshold=0.5)[0].numpy()
        return coords  # (2,) or (-1, -1)

    return predict


def _load_tracknetv4(checkpoint, device):
    from models.tracknetv4 import TrackNetV4
    from models.tracknet import heatmap_to_coords
    model = TrackNetV4().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    def predict(triplet_paths):
        import cv2 as _cv2
        from data.preprocessing import resize_frame, normalize
        frames = []
        for p in triplet_paths:
            img = _cv2.imread(p)
            img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
            frames.append(resize_frame(img))
        tensor = np.concatenate([normalize(f) for f in frames], axis=0)
        inp = torch.from_numpy(tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            hm = model(inp)
        return heatmap_to_coords(hm.cpu(), threshold=0.5)[0].numpy()

    return predict


def _load_tracknetv5(checkpoint, device):
    from models.tracknetv5 import TrackNetV5
    from models.tracknet import heatmap_to_coords
    model = TrackNetV5().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    def predict(triplet_paths):
        import cv2 as _cv2
        from data.preprocessing import resize_frame, normalize
        frames = []
        for p in triplet_paths:
            img = _cv2.imread(p)
            img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
            frames.append(resize_frame(img))
        tensor = np.concatenate([normalize(f) for f in frames], axis=0)
        inp = torch.from_numpy(tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            hm = model(inp)
        return heatmap_to_coords(hm.cpu(), threshold=0.5)[0].numpy()

    return predict


def _load_yolo(checkpoint, device):
    try:
        from ultralytics import YOLO
        model = YOLO(checkpoint)

        def predict(frame_paths):
            import cv2 as _cv2
            img = _cv2.imread(frame_paths[-1])
            h, w = img.shape[:2]
            results = model(img, verbose=False)
            boxes = results[0].boxes
            if boxes and len(boxes):
                best = boxes.conf.argmax()
                if boxes.conf[best] > 0.3:
                    x1, y1, x2, y2 = boxes.xyxy[best].tolist()
                    px = ((x1 + x2) / 2) * (IMG_W / w)
                    py = ((y1 + y2) / 2) * (IMG_H / h)
                    return np.array([px, py])
            return np.array([-1.0, -1.0])

        return predict
    except ImportError:
        from models.yolo11m import LightweightDetector
        from data.preprocessing import YOLO_SIZE
        det = LightweightDetector().to(device)
        det.load_state_dict(torch.load(checkpoint, map_location=device))
        det.eval()

        def predict(frame_paths):
            import cv2 as _cv2
            img = _cv2.imread(frame_paths[-1])
            img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
            img = _cv2.resize(img, (YOLO_SIZE, YOLO_SIZE))
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            inp = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                out = det(inp)
            conf = torch.sigmoid(out[0, 0]).item()
            if conf > 0.3:
                return np.array([out[0, 1].item() * IMG_W, out[0, 2].item() * IMG_H])
            return np.array([-1.0, -1.0])

        return predict


def overlay_prediction(img_bgr, pred_xy, gt_xy, model_name):
    """Draw GT (green) and prediction (red) on a copy of the image."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    sx, sy = w / IMG_W, h / IMG_H

    if gt_xy[0] >= 0:
        gx, gy = int(gt_xy[0] * sx), int(gt_xy[1] * sy)
        cv2.circle(out, (gx, gy), 8, (0, 255, 0), 2)
        cv2.putText(out, "GT", (gx + 5, gy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    if pred_xy[0] >= 0:
        px, py = int(pred_xy[0] * sx), int(pred_xy[1] * sy)
        cv2.circle(out, (px, py), 6, (0, 0, 255), 2)
        cv2.putText(out, model_name, (px + 5, py + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = _load_splits(args.splits_csv, "test")

    if args.model == "tracknet":
        predictor = _load_tracknet(args.checkpoint, device)
        need_frames = 3
    elif args.model == "tracknetv4":
        predictor = _load_tracknetv4(args.checkpoint, device)
        need_frames = 3
    elif args.model == "tracknetv5":
        predictor = _load_tracknetv5(args.checkpoint, device)
        need_frames = 3
    elif args.model == "yolo11m":
        predictor = _load_yolo(args.checkpoint, device)
        need_frames = 1
    else:
        raise ValueError(f"Visualize not implemented for {args.model}")

    # Group into sequences
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[(r["game"], r["clip"])].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r["frame_name"])

    os.makedirs(args.output_dir, exist_ok=True)
    count = 0
    for clip_records in groups.values():
        for i in range(need_frames - 1, len(clip_records)):
            if count >= args.num_samples:
                break
            window = clip_records[i - need_frames + 1 : i + 1]
            paths = [r["frame_path"] for r in window]
            cur_r = window[-1]

            pred_xy = predictor(paths)
            gt_xy   = np.array([float(cur_r["x"]), float(cur_r["y"])])
            if int(cur_r["visibility"]) == 0:
                gt_xy = np.array([-1.0, -1.0])

            img = cv2.imread(cur_r["frame_path"])
            vis = overlay_prediction(img, pred_xy, gt_xy, args.model)

            fname = f"{cur_r['game']}_{cur_r['clip']}_{cur_r['frame_name']}"
            cv2.imwrite(os.path.join(args.output_dir, fname), vis)
            count += 1
        if count >= args.num_samples:
            break

    print(f"Saved {count} visualisations → {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True,
                        choices=["tracknet", "tracknetv4", "tracknetv5", "yolo11m"])
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--splits_csv",  default=SPLITS_CSV)
    parser.add_argument("--output_dir",  default=os.path.join(RESULTS_DIR, "visualizations"))
    parser.add_argument("--num_samples", type=int, default=20)
    visualize(parser.parse_args())
