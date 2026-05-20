"""
Train ResNet50Pose for court keypoint detection.

Usage:
    python -m train.train_resnet50 [--max_samples N] [--epochs N] [--batch_size N] [--device DEVICE]
"""

import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CourtKeypointDataset
from models.resnet50_pose import ResNet50Pose
from train.config import RESNET50

SEED = 42
NUM_WORKERS = 4
PIN_MEMORY = True

CHECKPOINT_DIR = pathlib.Path(__file__).parent.parent / "checkpoints"


def _val_pck(model, loader, device, cfg):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, heatmaps, kps_orig in loader:
            imgs = imgs.to(device)
            preds = torch.sigmoid(model(imgs))
            # decode each of 14 channels (skip channel 14 = court center)
            B, C, H, W = preds.shape
            stride = cfg['input_h'] // H  # 1 for tracknet_court, 4 for others
            for b in range(B):
                for k in range(14):
                    # ground truth
                    gt = kps_orig[b, k].numpy()
                    if gt[0] < 0:  # invisible
                        continue
                    # argmax decode in output space
                    hm = preds[b, k].cpu().numpy()
                    fy, fx = np.unravel_index(hm.argmax(), hm.shape)
                    px = fx * stride
                    py = fy * stride
                    dist = np.sqrt((px - gt[0])**2 + (py - gt[1])**2)
                    if dist <= cfg['pck_threshold_px']:
                        correct += 1
                    total += 1
    return correct / total if total > 0 else 0.0


def train(args):
    torch.manual_seed(SEED)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
    print(f"Device: {device}")

    cfg = RESNET50
    epochs     = args.epochs     if args.epochs     is not None else cfg['epochs']
    batch_size = args.batch_size if args.batch_size is not None else cfg['batch_size']

    train_ds = CourtKeypointDataset(
        'train', augment=True, max_samples=args.max_samples,
        stride=cfg['stride'], gaussian_radius=cfg['gaussian_radius'],
        imagenet_norm=cfg['use_imagenet_norm'], data_dir=args.data_dir,
    )
    val_ds = CourtKeypointDataset(
        'val', augment=False, max_samples=args.max_samples,
        stride=cfg['stride'], gaussian_radius=cfg['gaussian_radius'],
        imagenet_norm=cfg['use_imagenet_norm'], data_dir=args.data_dir,
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    model = ResNet50Pose().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                 weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_pck = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for imgs, heatmaps, _ in pbar:
            imgs      = imgs.to(device)
            heatmaps  = heatmaps.to(device)
            pred_logits = model(imgs)
            loss = criterion(torch.sigmoid(pred_logits), heatmaps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_dl)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps, _ in val_dl:
                imgs     = imgs.to(device)
                heatmaps = heatmaps.to(device)
                pred_logits = model(imgs)
                val_loss += criterion(torch.sigmoid(pred_logits), heatmaps).item()
        val_loss /= len(val_dl)

        val_pck = _val_pck(model, val_dl, device, cfg)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | val_pck@7px={val_pck:.4f}")

        if val_pck > best_pck:
            best_pck = val_pck
            patience_counter = 0
            ckpt = CHECKPOINT_DIR / "resnet50_best.pt"
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict(), 'best_pck': best_pck},
                ckpt
            )
            print(f"  Saved checkpoint -> {ckpt}")
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print("Early stopping.")
                break

    print(f"Training complete. Best val_pck@7px={best_pck:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--epochs',      type=int, default=None)
    parser.add_argument('--batch_size',  type=int, default=None)
    parser.add_argument('--device',      type=str, default=None)
    parser.add_argument('--data_dir',    type=str, default=None,
                        help='Path to data dir containing data_train.json, data_val.json, images/. '
                             'Defaults to the data/ folder in this repo.')
    train(parser.parse_args())
