"""
Train TrackNet.

Usage:
    python -m train.train_tracknet [--epochs N] [--batch_size N] [--lr LR]
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TrackNetDataset
from models.tracknet import TrackNet, intensity_to_coords
from train.config import TRACKNET, SPLITS_CSV, CHECKPOINT_DIR, SEED


def pixel_accuracy(logits, gt_classmap, threshold_px=5):
    """Fraction of GT-visible frames where predicted peak ≤ threshold_px from GT.

    Uses the fast argmax-peak readout (no Hough) on the (B, 256, H, W) logits and
    the (B, H, W) GT class-map. Distance is in the resized pixel space.
    """
    pred_coords = intensity_to_coords(logits.argmax(dim=1), use_hough=False)
    gt_coords   = intensity_to_coords(gt_classmap, use_hough=False, threshold=1)
    # Ignore frames where GT has no ball
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
                               max_samples=args.max_samples, target_mode="classmap")
    val_ds   = TrackNetDataset(args.splits_csv, "val",   augment=False,
                               max_samples=args.max_samples, target_mode="classmap")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    model = TrackNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr,
                                     weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = -1.0   # checkpoint on val_acc; val_loss decouples from it at high res
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for frames, targets, _ in pbar:
            frames  = frames.to(device)
            targets = targets.to(device)
            logits = model(frames)
            loss = F.cross_entropy(logits, targets)
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
            for frames, targets, _ in val_dl:
                frames  = frames.to(device)
                targets = targets.to(device)
                logits = model(frames)
                val_loss += F.cross_entropy(logits, targets).item()
                val_acc  += pixel_accuracy(logits.cpu(), targets.cpu())
        val_loss /= len(val_dl)
        val_acc  /= len(val_dl)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc@5px={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt = os.path.join(args.checkpoint_dir, "tracknet_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved checkpoint → {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    print(f"Training complete. Best val_acc@5px={best_val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_csv",      default=SPLITS_CSV)
    parser.add_argument("--epochs",     type=int,   default=TRACKNET["epochs"])
    parser.add_argument("--batch_size", type=int,   default=TRACKNET["batch_size"])
    parser.add_argument("--lr",         type=float, default=TRACKNET["lr"])
    parser.add_argument("--patience",   type=int,   default=TRACKNET["patience"])
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit train+val to N samples each (quick convergence check)")
    train(parser.parse_args())
