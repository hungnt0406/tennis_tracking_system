"""
Train YOLO11m for tennis ball detection.

Two modes:
  1. Ultralytics mode  (default): generates a YOLO-format dataset on disk and
     calls the Ultralytics training API.
  2. Fallback mode: trains the LightweightDetector from models/yolo11m.py.

Usage:
    python -m train.train_yolo11m [--epochs N] [--use_fallback]
"""

import argparse
import csv
import os
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import YOLODataset
from train.config import YOLO11M, SPLITS_CSV, CHECKPOINT_DIR, SEED


# ─── Ultralytics path ────────────────────────────────────────────────────────

def _export_yolo_dataset(splits_csv: str, out_dir: str):
    """Write images (symlinks) and YOLO .txt labels to <out_dir>/images|labels."""
    for split in ("train", "val", "test"):
        os.makedirs(f"{out_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{out_dir}/labels/{split}", exist_ok=True)

    with open(splits_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            src  = row["frame_path"]
            stem = f"{row['game']}_{row['clip']}_{row['frame_name']}"
            dst_img = f"{out_dir}/images/{split}/{stem}"
            if not os.path.exists(dst_img):
                os.symlink(src, dst_img)

            vis = int(row["visibility"])
            x   = float(row["x"])
            y   = float(row["y"])

            lbl_path = f"{out_dir}/labels/{split}/{stem.replace('.jpg', '.txt')}"
            if vis > 0 and x >= 0 and y >= 0:
                import cv2
                img = cv2.imread(src)
                h, w = img.shape[:2]
                cx, cy = x / w, y / h
                bw = 20 / w
                bh = 20 / h
                with open(lbl_path, "w") as lf:
                    lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            else:
                open(lbl_path, "w").close()  # empty = no ball

    # Write data.yaml
    yaml_path = f"{out_dir}/data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("nc: 1\nnames: ['tennis_ball']\n")
    return yaml_path


def train_ultralytics(args):
    from ultralytics import YOLO
    yolo_dir = os.path.join(args.checkpoint_dir, "yolo_dataset")
    print("Exporting YOLO dataset …")
    yaml_path = _export_yolo_dataset(args.splits_csv, yolo_dir)

    model = YOLO("yolo11m.pt")
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=YOLO11M["img_size"],
        batch=YOLO11M["batch_size"],
        lr0=args.lr,
        patience=args.patience,
        project=args.checkpoint_dir,
        name="yolo11m_tennis",
        exist_ok=True,
        verbose=True,
    )
    print("Ultralytics training complete.")


# ─── Fallback path ───────────────────────────────────────────────────────────

def train_fallback(args):
    """Train the LightweightDetector using binary detection + regression losses."""
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  (fallback detector — Ultralytics not installed)")

    from models.yolo11m import LightweightDetector

    train_ds = YOLODataset(args.splits_csv, "train", augment=True,
                           max_samples=args.max_samples)
    val_ds   = YOLODataset(args.splits_csv, "val",   augment=False,
                           max_samples=args.max_samples)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = LightweightDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    bce = nn.BCEWithLogitsLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for frames, labels, has_ball, _ in train_dl:
            frames   = frames.to(device)
            labels   = labels.to(device)
            has_ball = has_ball.to(device)

            out = model(frames)      # (B, 5): conf, cx, cy, bw, bh
            conf_loss = bce(out[:, 0], has_ball)
            mask = has_ball.bool()
            reg_loss = torch.tensor(0.0, device=device)
            if mask.any():
                reg_loss = nn.functional.smooth_l1_loss(
                    out[mask, 1:], labels[mask, 1:])
            loss = conf_loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, labels, has_ball, _ in val_dl:
                frames   = frames.to(device)
                labels   = labels.to(device)
                has_ball = has_ball.to(device)
                out = model(frames)
                conf_loss = bce(out[:, 0], has_ball)
                mask = has_ball.bool()
                reg_loss = torch.tensor(0.0, device=device)
                if mask.any():
                    reg_loss = nn.functional.smooth_l1_loss(
                        out[mask, 1:], labels[mask, 1:])
                val_loss += (conf_loss + reg_loss).item()

        val_loss /= len(val_dl)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt = os.path.join(args.checkpoint_dir, "yolo11m_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved checkpoint → {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_csv",      default=SPLITS_CSV)
    parser.add_argument("--epochs",     type=int,   default=YOLO11M["epochs"])
    parser.add_argument("--batch_size", type=int,   default=YOLO11M["batch_size"])
    parser.add_argument("--lr",         type=float, default=YOLO11M["lr"])
    parser.add_argument("--patience",   type=int,   default=YOLO11M["patience"])
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--use_fallback", action="store_true",
                        help="Force LightweightDetector even if ultralytics is installed")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit train+val to N samples each (quick convergence check)")
    args = parser.parse_args()

    use_ultralytics = not args.use_fallback
    if use_ultralytics:
        try:
            import ultralytics
        except ImportError:
            print("ultralytics not installed — using fallback detector.")
            use_ultralytics = False

    if use_ultralytics:
        train_ultralytics(args)
    else:
        train_fallback(args)
