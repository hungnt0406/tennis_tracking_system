"""
Train S-KeepTrack.

Usage:
    python -m train.train_skeeptrack [--epochs N] [--batch_size N] [--lr LR]
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SKeepTrackDataset
from models.skeeptrack import SKeepTrack
from train.config import SKEEPTRACK, SPLITS_CSV, CHECKPOINT_DIR, SEED


def build_assoc_targets(coords1, coords2, k: int, threshold: float = 0.05):
    """
    Build soft binary association targets.
    Two candidates are "matched" if they are close in normalised coordinate space.

    coords1, coords2 : (B, k, 2) — normalised positions from extract_candidates
    Returns          : (B, k, k) float target matrix
    """
    diff = coords1.unsqueeze(2) - coords2.unsqueeze(1)          # (B, k, k, 2)
    dist = diff.norm(dim=-1)                                     # (B, k, k)
    return (dist < threshold).float()


def classification_loss(score_map, gt_heatmap):
    """BCE between predicted score map and (downsampled) ground-truth heatmap."""
    gt_down = F.adaptive_avg_pool2d(gt_heatmap, score_map.shape[-2:])
    gt_down = (gt_down > 0.1).float()
    return F.binary_cross_entropy(score_map, gt_down)


def train(args):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = SKeepTrackDataset(args.splits_csv, "train", augment=True,
                                 max_samples=args.max_samples)
    val_ds   = SKeepTrackDataset(args.splits_csv, "val",   augment=False,
                                 max_samples=args.max_samples)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} pairs | Val: {len(val_ds)} pairs")

    model = SKeepTrack(k=args.k_candidates, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for t1, t2, hm1, hm2, coords1_gt, coords2_gt in pbar:
            t1, t2 = t1.to(device), t2.to(device)
            hm1, hm2 = hm1.to(device), hm2.to(device)

            sm1, sm2, assoc, pred_coords = model(t1, t2)

            cls_loss = (classification_loss(sm1, hm1)
                        + classification_loss(sm2, hm2))

            # Association targets: use candidate coords extracted inside model
            # We recompute them here — simpler than exporting from forward pass
            from models.skeeptrack import extract_candidates
            with torch.no_grad():
                feat1 = model.backbone(t1)
                feat2 = model.backbone(t2)
                c1, _, _ = extract_candidates(sm1.detach(), feat1.detach(), args.k_candidates)
                c2, _, _ = extract_candidates(sm2.detach(), feat2.detach(), args.k_candidates)
                assoc_gt = build_assoc_targets(c1, c2, args.k_candidates)

            assoc_loss = F.binary_cross_entropy(assoc, assoc_gt)
            loss = args.cls_weight * cls_loss + args.assoc_weight * assoc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_dl)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_dist = 0.0
        n_valid  = 0
        with torch.no_grad():
            for t1, t2, hm1, hm2, coords1_gt, coords2_gt in val_dl:
                t1, t2 = t1.to(device), t2.to(device)
                hm1, hm2 = hm1.to(device), hm2.to(device)
                sm1, sm2, assoc, pred_coords = model(t1, t2)

                cls_loss = (classification_loss(sm1, hm1)
                            + classification_loss(sm2, hm2))

                from models.skeeptrack import extract_candidates
                feat1 = model.backbone(t1)
                feat2 = model.backbone(t2)
                c1, _, _ = extract_candidates(sm1, feat1, args.k_candidates)
                c2, _, _ = extract_candidates(sm2, feat2, args.k_candidates)
                assoc_gt = build_assoc_targets(c1, c2, args.k_candidates)
                assoc_loss = F.binary_cross_entropy(assoc, assoc_gt)
                val_loss += (args.cls_weight * cls_loss
                             + args.assoc_weight * assoc_loss).item()

                # Distance to GT (normalised)
                gt_vis = coords2_gt[:, 2]
                valid  = gt_vis > 0
                if valid.any():
                    gt_xy   = coords2_gt[valid, :2]
                    pr_xy   = pred_coords[valid].cpu()
                    val_dist += (pr_xy - gt_xy).norm(dim=1).mean().item()
                    n_valid  += 1

        val_loss /= len(val_dl)
        mean_dist = val_dist / max(1, n_valid)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_dist(norm)={mean_dist:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt = os.path.join(args.checkpoint_dir, "skeeptrack_best.pt")
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
    parser.add_argument("--epochs",          type=int,   default=SKEEPTRACK["epochs"])
    parser.add_argument("--batch_size",      type=int,   default=SKEEPTRACK["batch_size"])
    parser.add_argument("--lr",              type=float, default=SKEEPTRACK["lr"])
    parser.add_argument("--patience",        type=int,   default=SKEEPTRACK["patience"])
    parser.add_argument("--k_candidates",   type=int,   default=SKEEPTRACK["k_candidates"])
    parser.add_argument("--cls_weight",     type=float, default=SKEEPTRACK["cls_weight"])
    parser.add_argument("--assoc_weight",   type=float, default=SKEEPTRACK["assoc_weight"])
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit train+val to N samples each (quick convergence check)")
    train(parser.parse_args())
