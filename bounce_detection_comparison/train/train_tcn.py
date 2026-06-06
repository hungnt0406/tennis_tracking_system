"""
Train the bounce-detection TCN.

Usage:
    python -m train.train_tcn [--epochs N] [--batch_size N] [--max_samples N]
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import BounceWindowDataset, iter_clip_features
from data.trajectory import gt_bounce_frames
from evaluation.decode import decode_clip, match_events
from evaluation.metrics import prf
from models.tcn import BounceTCN, Scorer
from train.config import (TCN, TCN_CHANNELS, FEATURE, DECODE, SPLITS_CSV,
                          CHECKPOINT_DIR, SEED)


def masked_soft_focal_bce(logits, target, loss_mask, gamma, pos_weight=10.0):
    """Per-frame focal BCE against the soft Gaussian target, masked to valid
    frames. Focal term (|p - target|)**gamma down-weights easy frames; positives
    (target > 0) are upweighted to fight the ~2.6% bounce-frame imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    focal = (p - target).abs().pow(gamma)
    w = 1.0 + (pos_weight - 1.0) * target          # ramps 1→pos_weight with target
    loss = focal * bce * w * loss_mask
    return loss.sum() / loss_mask.sum().clamp(min=1.0)


def calibrate_threshold(model, splits_csv, device):
    """Sweep the decode threshold on val (same windowed scoring as Scorer) and
    return the best-event-F1 threshold."""
    model.eval()
    clips = list(iter_clip_features(splits_csv, "val", FEATURE))
    if not clips:
        return DECODE["threshold"]
    window, stride = TCN["window"], TCN["stride"]
    scored = []
    with torch.no_grad():
        for traj, feats, names, valid in clips:
            cs = np.stack([feats[c] for c in TCN_CHANNELS], 0).astype(np.float32)
            C, T = cs.shape
            if T <= window:
                x = torch.from_numpy(
                    np.pad(cs, ((0, 0), (0, window - T)), mode="edge")
                ).unsqueeze(0).to(device)
                s = torch.sigmoid(model(x))[0].cpu().numpy()[:T]
            else:
                starts = list(range(0, T - window + 1, stride))
                if starts[-1] + window < T:
                    starts.append(T - window)
                acc = np.zeros(T); cnt = np.zeros(T)
                for st in starts:
                    e = st + window
                    x = torch.from_numpy(cs[:, st:e]).unsqueeze(0).to(device)
                    acc[st:e] += torch.sigmoid(model(x))[0].cpu().numpy()
                    cnt[st:e] += 1.0
                s = acc / np.maximum(cnt, 1.0)
            scored.append((s, valid, gt_bounce_frames(traj)))

    best_thr, best_f1 = DECODE["threshold"], -1.0
    for th in np.linspace(0.05, 0.95, 19):
        tp = fp = fn = 0
        for s, valid, gt in scored:
            pred = decode_clip(s, th, DECODE["min_peak_distance"], valid,
                               DECODE["peak_offset"])
            t, f, n, _ = match_events(pred, gt, DECODE["tolerance_k"])
            tp += t; fp += f; fn += n
        f1 = prf(tp, fp, fn)[2]
        if f1 > best_f1:
            best_f1, best_thr = f1, float(th)
    print(f"Calibrated threshold={best_thr:.3f} | val event-F1={best_f1:.3f}")
    return best_thr


def train(args):
    torch.manual_seed(SEED)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    train_ds = BounceWindowDataset(args.splits_csv, "train", TCN["window"],
                                   TCN["stride"], max_samples=args.max_samples)
    val_ds   = BounceWindowDataset(args.splits_csv, "val", TCN["window"],
                                   TCN["stride"], max_samples=args.max_samples)
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=pin)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=pin)
    print(f"Train: {len(train_ds)} windows | Val: {len(val_ds)} windows")

    cfg = dict(in_ch=len(TCN_CHANNELS), hidden=TCN["hidden"], levels=TCN["levels"],
               kernel=TCN["kernel"], dropout=TCN["dropout"])
    model = BounceTCN(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BounceTCN params: {n_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=TCN["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    gamma = TCN["focal_gamma"]

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "tcn_best.pt")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for feat, target, loss_mask in pbar:
            feat       = feat.to(device)
            target     = target.to(device)
            loss_mask  = loss_mask.to(device)
            logits = model(feat)
            loss = masked_soft_focal_bce(logits, target, loss_mask, gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= max(1, len(train_dl))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feat, target, loss_mask in val_dl:
                feat      = feat.to(device)
                target    = target.to(device)
                loss_mask = loss_mask.to(device)
                logits = model(feat)
                val_loss += masked_soft_focal_bce(
                    logits, target, loss_mask, gamma).item()
        val_loss /= max(1, len(val_dl))

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                        "threshold": DECODE["threshold"]}, ckpt_path)
            print(f"  Saved checkpoint -> {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")

    # Calibrate the decode threshold on val with the best checkpoint, store it.
    if os.path.isfile(ckpt_path):
        best = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best["state_dict"])
        thr = calibrate_threshold(model, args.splits_csv, device)
        best["threshold"] = thr
        torch.save(best, ckpt_path)
        print(f"  Updated checkpoint threshold -> {thr:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_csv",      default=SPLITS_CSV)
    parser.add_argument("--epochs",     type=int,   default=TCN["epochs"])
    parser.add_argument("--batch_size", type=int,   default=TCN["batch_size"])
    parser.add_argument("--lr",         type=float, default=TCN["lr"])
    parser.add_argument("--patience",   type=int,   default=TCN["patience"])
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit train+val to N windows each (quick convergence check)")
    parser.add_argument("--device", default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
