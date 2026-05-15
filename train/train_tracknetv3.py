"""
Train TrackNetV3 stage 1 (tracking module).

Usage:
    python -m train.train_tracknetv3 [--epochs N] [--batch_size N] [--mixup]
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_v2 import TrackNetV3Dataset
from models.losses import WBCELoss
from models.tracknet import heatmap_to_coords
from models.tracknetv3 import TrackNetV3Tracker
from train.config import TRACKNETV3_TRACKER, SPLITS_CSV, CHECKPOINT_DIR, SEED


def pixel_accuracy(pred_heatmap, gt_heatmap, threshold_px=5):
    pred_coords = heatmap_to_coords(pred_heatmap, threshold=0.5)
    gt_coords   = heatmap_to_coords(gt_heatmap,   threshold=0.1)
    valid = (gt_coords[:, 0] >= 0)
    if not valid.any():
        return 0.0
    dist = (pred_coords[valid] - gt_coords[valid]).norm(dim=1)
    return (dist <= threshold_px).float().mean().item()


def mixup_batch(frames, heatmaps, alpha: float = 0.2):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(frames.size(0), device=frames.device)
    mixed_frames = lam * frames + (1 - lam) * frames[perm]
    mixed_heatmaps = lam * heatmaps + (1 - lam) * heatmaps[perm]
    return mixed_frames, mixed_heatmaps


def train(args):
    torch.manual_seed(SEED)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    seq_len = TRACKNETV3_TRACKER["seq_len"]
    bg_mode = TRACKNETV3_TRACKER["bg_mode"]

    train_ds = TrackNetV3Dataset(args.splits_csv, "train", seq_len=seq_len,
                                 bg_mode=bg_mode, augment=True,
                                 max_samples=args.max_samples)
    val_ds   = TrackNetV3Dataset(args.splits_csv, "val", seq_len=seq_len,
                                 bg_mode=bg_mode, augment=False,
                                 max_samples=args.max_samples)
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=pin)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    model = TrackNetV3Tracker(seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=TRACKNETV3_TRACKER["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    criterion = WBCELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for frames, heatmaps in pbar:
            frames   = frames.to(device)
            heatmaps = heatmaps.to(device)
            if args.mixup:
                frames, heatmaps = mixup_batch(frames, heatmaps)
            pred = model(frames)
            loss = criterion(pred, heatmaps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= max(1, len(train_dl))

        model.eval()
        val_loss = 0.0
        val_acc  = 0.0
        with torch.no_grad():
            for frames, heatmaps in val_dl:
                frames   = frames.to(device)
                heatmaps = heatmaps.to(device)
                pred = model(frames)
                val_loss += criterion(pred, heatmaps).item()
                mid = pred.shape[1] // 2
                val_acc += pixel_accuracy(
                    pred[:, mid:mid + 1].cpu(),
                    heatmaps[:, mid:mid + 1].cpu(),
                )
        val_loss /= max(1, len(val_dl))
        val_acc  /= max(1, len(val_dl))

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} val_acc@5px={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt = os.path.join(args.checkpoint_dir, "tracknetv3_tracker_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved checkpoint -> {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_csv",      default=SPLITS_CSV)
    parser.add_argument("--epochs",     type=int,   default=TRACKNETV3_TRACKER["epochs"])
    parser.add_argument("--batch_size", type=int,   default=TRACKNETV3_TRACKER["batch_size"])
    parser.add_argument("--lr",         type=float, default=TRACKNETV3_TRACKER["lr"])
    parser.add_argument("--patience",   type=int,   default=TRACKNETV3_TRACKER["patience"])
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit train+val to N samples each (quick convergence check)")
    parser.add_argument("--mixup", action="store_true",
                        help="Enable mixup augmentation (Beta(0.2, 0.2))")
    parser.add_argument("--device", default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
