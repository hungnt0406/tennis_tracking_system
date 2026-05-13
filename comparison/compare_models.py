"""
Run all models on the test split and collect their metrics.

Usage:
    python -m comparison.compare_models \\
        --tracknet_ckpt    checkpoints/tracknet_best.pt \\
        --tracknetv4_ckpt  checkpoints/tracknetv4_best.pt \\
        --tracknetv5_ckpt  checkpoints/tracknetv5_best.pt \\
        --yolo_ckpt        checkpoints/yolo11m_best.pt

Saves per-model JSON files and a combined summary JSON to results/.
"""

import argparse
import json
import os
import subprocess
import sys

from train.config import SPLITS_CSV, RESULTS_DIR, CHECKPOINT_DIR


def _run_evaluate(model: str, checkpoint: str, splits_csv: str, results_dir: str):
    """Launch evaluation as a subprocess so models don't share GPU memory."""
    cmd = [
        sys.executable, "-m", "evaluation.evaluate",
        "--model",      model,
        "--checkpoint", checkpoint,
        "--splits_csv", splits_csv,
        "--results_dir", results_dir,
    ]
    print(f"\n{'='*60}")
    print(f"Evaluating {model} …")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def load_metrics(model: str, results_dir: str) -> dict:
    path = os.path.join(results_dir, f"{model}_metrics.json")
    with open(path) as f:
        return json.load(f)


def compare(args):
    os.makedirs(args.results_dir, exist_ok=True)

    checkpoints = {
        "tracknet":   args.tracknet_ckpt,
        "tracknetv4": args.tracknetv4_ckpt,
        "tracknetv5": args.tracknetv5_ckpt,
        "yolo11m":    args.yolo_ckpt,
    }

    # Only evaluate models whose checkpoints are provided
    available = {m: c for m, c in checkpoints.items() if c and os.path.exists(c)}
    if not available:
        print("No checkpoint files found. Run training first.")
        return

    for model, ckpt in available.items():
        _run_evaluate(model, ckpt, args.splits_csv, args.results_dir)

    # Aggregate
    all_metrics = {}
    for model in available:
        all_metrics[model] = load_metrics(model, args.results_dir)

    out_path = os.path.join(args.results_dir, "comparison_summary.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved comparison summary → {out_path}")

    # Quick console table
    metrics_keys = ["acc@5px", "acc@10px", "acc@20px",
                    "MAE_px", "precision", "recall", "F1",
                    "tracking_consistency", "FPS"]
    col_w = 18
    header = f"{'Metric':<30}" + "".join(f"{m:<{col_w}}" for m in available)
    print(f"\n{'─'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")
    for key in metrics_keys:
        row = f"{key:<30}"
        for model in available:
            val = all_metrics[model].get(key)
            row += f"{(f'{val:.4f}' if isinstance(val, (int, float)) and val is not None else 'N/A'):<{col_w}}"
        print(row)
    print(f"{'─'*len(header)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracknet_ckpt",   default=os.path.join(CHECKPOINT_DIR, "tracknet_best.pt"))
    parser.add_argument("--tracknetv4_ckpt", default=os.path.join(CHECKPOINT_DIR, "tracknetv4_best.pt"))
    parser.add_argument("--tracknetv5_ckpt", default=os.path.join(CHECKPOINT_DIR, "tracknetv5_best.pt"))
    parser.add_argument("--yolo_ckpt",       default=os.path.join(CHECKPOINT_DIR, "yolo11m_best.pt"))
    parser.add_argument("--splits_csv",      default=SPLITS_CSV)
    parser.add_argument("--results_dir",     default=RESULTS_DIR)
    compare(parser.parse_args())
