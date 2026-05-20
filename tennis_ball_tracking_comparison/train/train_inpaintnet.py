"""
Train InpaintNet (TrackNetV3 stage 3).

Usage:
    python -m train.train_inpaintnet [--epochs N] [--batch_size N]
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_v2 import TrajectoryDataset
from models.tracknetv3 import InpaintNet
from train.config import TRACKNETV3_INPAINT, CHECKPOINT_DIR, SEED


CACHE_DIR = "cache"


def masked_mse(pred, gt, mask):
    """Sum of squared error over masked positions / count of masked positions.

    pred, gt : (B, L, 2)
    mask     : (B, L)  — 1 where loss should be applied.

    Returns a scalar loss or None if there are no masked positions.
    """
    m = mask.unsqueeze(-1)  # (B, L, 1)
    se = ((pred - gt) ** 2) * m
    denom = m.sum() * 2.0  # 2 coords per masked position
    if denom.item() == 0:
        return None
    return se.sum() / denom


def _subset(ds, max_samples):
    if max_samples is None:
        return ds
    ds.coords = ds.coords[:max_samples]
    ds.gt_coords = ds.gt_coords[:max_samples]
    ds.mask = ds.mask[:max_samples]
    return ds


def train(args):
    torch.manual_seed(SEED)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    mask_ratio = TRACKNETV3_INPAINT["mask_ratio"]
    train_npz = os.path.join(CACHE_DIR, "trajectory_data_train.npz")
    val_npz   = os.path.join(CACHE_DIR, "trajectory_data_val.npz")

    train_ds = TrajectoryDataset(train_npz, mask_ratio=mask_ratio)
    val_ds   = TrajectoryDataset(val_npz,   mask_ratio=0.0)
    train_ds = _subset(train_ds, args.max_samples)
    val_ds   = _subset(val_ds,   args.max_samples)

    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=pin)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    model = InpaintNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=TRACKNETV3_INPAINT["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for coords_with_mask, gt, mask in pbar:
            coords_with_mask = coords_with_mask.to(device)
            gt   = gt.to(device)
            mask = mask.to(device)
            pred = model(coords_with_mask)
            loss = masked_mse(pred, gt, mask)
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        train_loss /= max(1, n_train)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for coords_with_mask, gt, mask in val_dl:
                coords_with_mask = coords_with_mask.to(device)
                gt   = gt.to(device)
                mask = mask.to(device)
                pred = model(coords_with_mask)
                loss = masked_mse(pred, gt, mask)
                if loss is None:
                    continue
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(1, n_val)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.6f} | "
              f"val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt = os.path.join(args.checkpoint_dir, "tracknetv3_inpaint_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved checkpoint -> {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    print(f"Training complete. Best val_loss={best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=TRACKNETV3_INPAINT["epochs"])
    parser.add_argument("--batch_size", type=int,   default=TRACKNETV3_INPAINT["batch_size"])
    parser.add_argument("--lr",         type=float, default=TRACKNETV3_INPAINT["lr"])
    parser.add_argument("--patience",   type=int,   default=TRACKNETV3_INPAINT["patience"])
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit train+val to N samples each (quick convergence check)")
    parser.add_argument("--device", default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
