"""
Train TrackNetV4.

Usage:
    python -m train.train_tracknetv4 [--epochs N] [--batch_size N] [--lr LR]
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TrackNetDataset
from models.tracknet import heatmap_to_coords
from models.tracknetv4 import TrackNetV4
from train.config import TRACKNETV4, SPLITS_CSV, CHECKPOINT_DIR, SEED


def pixel_accuracy(pred_heatmap, gt_heatmap, threshold_px=5):
    """Fraction of frames where predicted peak ≤ threshold_px from GT peak."""
    pred_coords = heatmap_to_coords(pred_heatmap, threshold=0.5)
    gt_coords   = heatmap_to_coords(gt_heatmap,   threshold=0.1)
    valid = (gt_coords[:, 0] >= 0)
    if not valid.any():
        return 0.0
    dist = (pred_coords[valid] - gt_coords[valid]).norm(dim=1)
    return (dist <= threshold_px).float().mean().item()


def train(args):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = TrackNetDataset(args.splits_csv, "train", augment=True,
                               max_samples=args.max_samples)
    val_ds   = TrackNetDataset(args.splits_csv, "val",   augment=False,
                               max_samples=args.max_samples)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    model = TrackNetV4().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.BCELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for frames, heatmaps, _ in pbar:
            frames   = frames.to(device)
            heatmaps = heatmaps.to(device)
            pred = model(frames)
            loss = criterion(pred, heatmaps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_dl)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0
        with torch.no_grad():
            for frames, heatmaps, _ in val_dl:
                frames   = frames.to(device)
                heatmaps = heatmaps.to(device)
                pred = model(frames)
                val_loss += criterion(pred, heatmaps).item()
                val_acc  += pixel_accuracy(pred.cpu(), heatmaps.cpu())
        val_loss /= len(val_dl)
        val_acc  /= len(val_dl)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc@5px={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt = os.path.join(args.checkpoint_dir, "tracknetv4_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved checkpoint → {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_csv",      default=SPLITS_CSV)
    parser.add_argument("--epochs",     type=int,   default=TRACKNETV4["epochs"])
    parser.add_argument("--batch_size", type=int,   default=TRACKNETV4["batch_size"])
    parser.add_argument("--lr",         type=float, default=TRACKNETV4["lr"])
    parser.add_argument("--patience",   type=int,   default=TRACKNETV4["patience"])
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit train+val to N samples each (quick convergence check)")
    train(parser.parse_args())
