"""
Train TrackNet.

Usage:
    python -m train.train_tracknet [--epochs N] [--batch_size N] [--lr LR]
    python -m train.train_tracknet --auto_batch --target_vram_gb 23
"""

import argparse
import os
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TrackNetDataset
from models.tracknet import TrackNet, intensity_to_coords
from train.config import TRACKNET, SPLITS_CSV, CHECKPOINT_DIR, SEED


BYTES_IN_GIB = 1024 ** 3


def _gib(num_bytes):
    return num_bytes / BYTES_IN_GIB


def _clear_cuda_cache(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


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


def _probe_batch_memory(sample, batch_size, device, lr):
    """Return peak CUDA reserved memory for one train step, or None on OOM."""
    frames, targets, _ = sample
    frames = frames.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    targets = targets.unsqueeze(0).repeat(batch_size, 1, 1)

    model = None
    optimizer = None
    try:
        _clear_cuda_cache(device)
        torch.cuda.reset_peak_memory_stats(device)

        model = TrackNet().to(device)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=1e-5)
        frames = frames.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(frames)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)

        peak_reserved = torch.cuda.max_memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        return max(peak_reserved, peak_allocated)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return None
    finally:
        del optimizer, model, frames, targets
        _clear_cuda_cache(device)


def auto_select_batch_size(train_ds, args, device):
    """Find the largest train batch that fits within the requested VRAM budget."""
    if device.type != "cuda":
        print("Auto batch is only available on CUDA; using --batch_size.")
        return args.batch_size
    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty; cannot auto-select batch size.")

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    target_bytes = int(args.target_vram_gb * BYTES_IN_GIB)
    reserve_bytes = int(args.reserve_vram_gb * BYTES_IN_GIB)
    usable_bytes = min(target_bytes, max(0, free_bytes - reserve_bytes))
    if usable_bytes <= 0:
        raise RuntimeError(
            "No usable CUDA memory remains after reserve. "
            f"Free={_gib(free_bytes):.2f} GiB reserve={args.reserve_vram_gb:.2f} GiB."
        )

    print(
        "Auto batch target: "
        f"{_gib(usable_bytes):.2f} GiB usable "
        f"(free={_gib(free_bytes):.2f}/{_gib(total_bytes):.2f} GiB, "
        f"requested={args.target_vram_gb:.2f} GiB, "
        f"reserve={args.reserve_vram_gb:.2f} GiB)"
    )

    sample = train_ds[0]
    max_batch = max(1, args.max_auto_batch_size)
    best_batch = 0
    best_peak = 0
    probe_batch = 1
    failed_at = None

    while probe_batch <= max_batch:
        peak = _probe_batch_memory(sample, probe_batch, device, args.lr)
        fits = peak is not None and peak <= usable_bytes
        if peak is None:
            print(f"  probe batch={probe_batch}: OOM")
        else:
            print(f"  probe batch={probe_batch}: peak={_gib(peak):.2f} GiB")

        if not fits:
            failed_at = probe_batch
            break

        best_batch = probe_batch
        best_peak = peak
        probe_batch *= 2

    if best_batch == 0:
        raise RuntimeError(
            "Batch size 1 does not fit the current CUDA memory budget. "
            "Free the GPU or lower --reserve_vram_gb."
        )

    high = max_batch if failed_at is None else min(max_batch, failed_at - 1)
    low = best_batch + 1
    while low <= high:
        mid = (low + high) // 2
        peak = _probe_batch_memory(sample, mid, device, args.lr)
        fits = peak is not None and peak <= usable_bytes
        if peak is None:
            print(f"  probe batch={mid}: OOM")
        else:
            print(f"  probe batch={mid}: peak={_gib(peak):.2f} GiB")

        if fits:
            best_batch = mid
            best_peak = peak
            low = mid + 1
        else:
            high = mid - 1

    print(
        f"Selected batch_size={best_batch} "
        f"(probe peak={_gib(best_peak):.2f} GiB <= target={_gib(usable_bytes):.2f} GiB)"
    )
    return best_batch


def train(args):
    torch.manual_seed(SEED)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    train_ds = TrackNetDataset(args.splits_csv, "train", augment=True,
                               max_samples=args.max_samples, target_mode="classmap")
    val_ds   = TrackNetDataset(args.splits_csv, "val",   augment=False,
                               max_samples=args.max_samples, target_mode="classmap")
    batch_size = auto_select_batch_size(train_ds, args, device) if args.auto_batch else args.batch_size
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=pin)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples | Batch: {batch_size}")

    model = TrackNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr,
                                     weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = -1.0   # checkpoint on val_acc; val_loss decouples from it at high res
    patience_counter = 0
    start_epoch = 1
    last_ckpt = os.path.join(args.checkpoint_dir, "tracknet_last.pt")

    if args.resume:
        resume_path = args.resume if isinstance(args.resume, str) else last_ckpt
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch       = state["epoch"] + 1
        best_val_acc      = state["best_val_acc"]
        patience_counter  = state["patience_counter"]
        print(f"Resumed from {resume_path} at epoch {start_epoch} "
              f"(best val_acc@5px={best_val_acc:.3f})")

    for epoch in range(start_epoch, args.epochs + 1):
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

        # full training state for --resume (written every epoch, after best update)
        torch.save({
            "model":            model.state_dict(),
            "optimizer":        optimizer.state_dict(),
            "scheduler":        scheduler.state_dict(),
            "epoch":            epoch,
            "best_val_acc":     best_val_acc,
            "patience_counter": patience_counter,
        }, last_ckpt)

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
    parser.add_argument("--resume", nargs="?", const=True, default=False,
                        help="Resume training. Bare flag uses "
                             "<checkpoint_dir>/tracknet_last.pt; pass a path to override.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit train+val to N samples each (quick convergence check)")
    parser.add_argument("--device", default=None,
                        help="Torch device, e.g. cuda, cuda:2, or cpu.")
    parser.add_argument("--auto_batch", action="store_true",
                        help="Probe CUDA memory and use the largest safe batch size.")
    parser.add_argument("--target_vram_gb", type=float, default=23.0,
                        help="Target peak CUDA reserved memory for auto batch.")
    parser.add_argument("--reserve_vram_gb", type=float, default=0.5,
                        help="VRAM to leave free when auto-selecting batch size.")
    parser.add_argument("--max_auto_batch_size", type=int, default=64,
                        help="Upper bound for automatic batch-size probing.")
    train(parser.parse_args())
