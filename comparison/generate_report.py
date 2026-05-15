"""
Generate comparison tables and visualisation plots from saved metric JSONs.

Usage:
    python -m comparison.generate_report --results_dir results/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train.config import RESULTS_DIR


MODEL_COLORS = {
    "tracknet":   "#2196F3",
    "tracknetv2": "#00BCD4",
    "tracknetv3": "#4CAF50",
    "tracknetv4": "#9C27B0",
    "tracknetv5": "#E91E63",
    "yolo11m":    "#FF5722",
}

DISPLAY_NAMES = {
    "tracknet":   "TrackNet V1",
    "tracknetv2": "TrackNet V2",
    "tracknetv3": "TrackNet V3",
    "tracknetv4": "TrackNet V4",
    "tracknetv5": "TrackNet V5",
    "yolo11m":    "YOLO11m",
}


def load_all(results_dir: str) -> dict:
    all_metrics = {}
    for model in ("tracknet", "tracknetv2", "tracknetv3", "tracknetv4", "tracknetv5", "yolo11m"):
        path = os.path.join(results_dir, f"{model}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                all_metrics[model] = json.load(f)
    return all_metrics


def plot_accuracy_curves(all_metrics: dict, out_path: str):
    thresholds = [5, 10, 20]
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, metrics in all_metrics.items():
        vals = [metrics.get(f"acc@{t}px", 0) or 0 for t in thresholds]
        ax.plot(thresholds, vals, "o-", label=DISPLAY_NAMES[model],
                color=MODEL_COLORS.get(model, "gray"), linewidth=2, markersize=7)
    ax.set_xlabel("Distance threshold (pixels)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Distance Threshold")
    ax.legend()
    ax.set_xticks(thresholds)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_bar_metric(all_metrics: dict, metric_key: str, ylabel: str,
                    title: str, out_path: str, invert: bool = False):
    models = list(all_metrics.keys())
    vals = []
    for m in models:
        v = all_metrics[m].get(metric_key)
        vals.append(v if v is not None else 0.0)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar([DISPLAY_NAMES[m] for m in models], vals,
                  color=[MODEL_COLORS.get(m, "gray") for m in models], width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if invert:
        ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_per_visibility(all_metrics: dict, out_path: str):
    vis_classes = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(vis_classes))
    n_models = max(len(all_metrics), 1)
    width = 0.8 / n_models
    for i, (model, metrics) in enumerate(all_metrics.items()):
        vals = [metrics.get(f"MAE_vis{c}") or 0 for c in vis_classes]
        ax.bar(x + i * width, vals, width, label=DISPLAY_NAMES[model],
               color=MODEL_COLORS.get(model, "gray"))
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(["Visible (1)", "Hard (2)", "Occluded (3)"])
    ax.set_ylabel("MAE (pixels)")
    ax.set_title("MAE by Visibility Class")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def print_latex_table(all_metrics: dict):
    metrics = ["acc@5px", "acc@10px", "acc@20px", "MAE_px",
               "precision", "recall", "F1", "FPS"]
    models = list(all_metrics.keys())

    print("\n% LaTeX comparison table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l" + "r" * len(models) + "}")
    print("\\hline")
    header = "Metric & " + " & ".join(DISPLAY_NAMES[m] for m in models) + " \\\\"
    print(header)
    print("\\hline")
    for key in metrics:
        row = key.replace("_", "\\_") + " & "
        vals = []
        for m in models:
            v = all_metrics[m].get(key)
            vals.append(f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "—")
        row += " & ".join(vals) + " \\\\"
        print(row)
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Tennis ball tracking model comparison on test set.}")
    print("\\end{table}")


def generate_report(args):
    all_metrics = load_all(args.results_dir)
    if not all_metrics:
        print("No metric files found in results_dir. Run evaluation first.")
        return

    os.makedirs(args.results_dir, exist_ok=True)

    plot_accuracy_curves(
        all_metrics,
        os.path.join(args.results_dir, "accuracy_curves.png"))

    plot_bar_metric(
        all_metrics, "MAE_px", "MAE (pixels)", "Mean Absolute Error",
        os.path.join(args.results_dir, "mae_comparison.png"), invert=False)

    plot_bar_metric(
        all_metrics, "F1", "F1 Score", "Detection F1 Score",
        os.path.join(args.results_dir, "f1_comparison.png"))

    plot_bar_metric(
        all_metrics, "FPS", "Frames per Second", "Inference Speed",
        os.path.join(args.results_dir, "fps_comparison.png"))

    plot_per_visibility(
        all_metrics,
        os.path.join(args.results_dir, "visibility_mae.png"))

    print_latex_table(all_metrics)
    print(f"\nReport generated in {args.results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    generate_report(parser.parse_args())
