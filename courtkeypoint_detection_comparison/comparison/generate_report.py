"""
Generate comparison tables and visualisation plots from saved metric JSONs.

Usage:
    python -m comparison.generate_report [--results_dir results/]
"""

import argparse
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = str(pathlib.Path(__file__).parent.parent / "results")

MODEL_COLORS = {
    "tracknet_court": "#2196F3",
    "resnet50":        "#FF5722",
    "hrnet":           "#4CAF50",
}

DISPLAY_NAMES = {
    "tracknet_court": "TrackNet-Court",
    "resnet50":        "ResNet50+Deconv",
    "hrnet":           "HRNet-W32",
}

# Human-readable labels for the 14 court keypoints (TennisCourtDetector ordering)
KP_LABELS = [
    "TL-outer", "TR-outer", "BL-outer", "BR-outer",
    "TL-singles", "BL-singles", "TR-singles", "BR-singles",
    "TL-service", "TR-service", "BL-service", "BR-service",
    "Top-center-T", "Bot-center-T",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all(results_dir: str) -> dict:
    all_metrics = {}
    for model in ("tracknet_court", "resnet50", "hrnet"):
        path = os.path.join(results_dir, f"{model}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                all_metrics[model] = json.load(f)
    return all_metrics


def _safe(v):
    """Return 0.0 for None/NaN so matplotlib never chokes."""
    if v is None:
        return 0.0
    try:
        return float(v) if not (v != v) else 0.0  # NaN check
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_pck_curves(all_metrics: dict, out_path: str):
    thresholds = [5, 7, 10, 25]
    keys = [f"pck@{t}px" for t in thresholds]
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, metrics in all_metrics.items():
        vals = [_safe(metrics.get(k)) for k in keys]
        ax.plot(thresholds, vals, "o-",
                label=DISPLAY_NAMES[model],
                color=MODEL_COLORS.get(model, "gray"),
                linewidth=2, markersize=7)
    ax.set_xlabel("Distance threshold (pixels)")
    ax.set_ylabel("PCK")
    ax.set_title("PCK vs Distance Threshold (14 keypoints)")
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
    vals = [_safe(all_metrics[m].get(metric_key)) for m in models]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        [DISPLAY_NAMES[m] for m in models], vals,
        color=[MODEL_COLORS.get(m, "gray") for m in models],
        width=0.5,
    )
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


def plot_per_keypoint_pck(all_metrics: dict, out_path: str):
    """Horizontal grouped bar chart: one group per keypoint, one bar per model."""
    n_kp = 14
    x = np.arange(n_kp)
    models = list(all_metrics.keys())
    n_models = len(models)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, model in enumerate(models):
        raw = all_metrics[model].get("per_kp_pck@7px", [])
        vals = [_safe(raw[k]) if k < len(raw) else 0.0 for k in range(n_kp)]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width,
               label=DISPLAY_NAMES[model],
               color=MODEL_COLORS.get(model, "gray"),
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(KP_LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("PCK @ 7px")
    ax.set_title("Per-Keypoint PCK @ 7px")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_params_vs_pck(all_metrics: dict, out_path: str):
    """Scatter: x=params_M, y=pck@7px, annotated with model names."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, metrics in all_metrics.items():
        x = _safe(metrics.get("params_M"))
        y = _safe(metrics.get("pck@7px"))
        ax.scatter(x, y, s=120, color=MODEL_COLORS.get(model, "gray"),
                   zorder=3, label=DISPLAY_NAMES[model])
        ax.annotate(DISPLAY_NAMES[model], (x, y),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("PCK @ 7px")
    ax.set_title("Accuracy vs Model Size")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_radar(all_metrics: dict, out_path: str):
    """Radar chart comparing models on the five headline PCK thresholds +
    homography success and normalised error metrics."""
    categories = ["PCK@5px", "PCK@7px", "PCK@10px", "PCK@25px",
                  "H Success", "1−NormPxErr", "1−NormCmErr"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for model, metrics in all_metrics.items():
        # Normalise mean_kp_error to [0,1]: assume 50px is the "zero" bound
        err = _safe(metrics.get("mean_kp_error_px"))
        norm_err_inv = max(0.0, 1.0 - err / 50.0)
        reproj_err = _safe(metrics.get("mean_reproj_err_cm"))
        norm_reproj_inv = max(0.0, 1.0 - reproj_err / 100.0)
        vals = [
            _safe(metrics.get("pck@5px")),
            _safe(metrics.get("pck@7px")),
            _safe(metrics.get("pck@10px")),
            _safe(metrics.get("pck@25px")),
            _safe(metrics.get("homography_success_rate")),
            norm_err_inv,
            norm_reproj_inv,
        ]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2,
                color=MODEL_COLORS.get(model, "gray"),
                label=DISPLAY_NAMES[model])
        ax.fill(angles, vals, alpha=0.1, color=MODEL_COLORS.get(model, "gray"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison Radar", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Console + LaTeX tables
# ---------------------------------------------------------------------------

SUMMARY_METRICS = [
    ("pck@5px",               "PCK@5px"),
    ("pck@7px",               "PCK@7px"),
    ("pck@10px",              "PCK@10px"),
    ("pck@25px",              "PCK@25px"),
    ("mean_kp_error_px",      "Mean KP Err (px)"),
    ("mean_reproj_err_cm",    "Mean Reproj Err (cm)"),
    ("max_reproj_err_cm",     "Max Reproj Err (cm)"),
    ("homography_success_rate", "Homography Success"),
    ("court_center_pck@7px",  "Center@7px"),
    ("params_M",              "Params (M)"),
    ("FPS",                   "FPS"),
]


def print_console_table(all_metrics: dict):
    models = list(all_metrics.keys())
    col_w = 20
    header = f"{'Metric':<25}" + "".join(f"{DISPLAY_NAMES[m]:>{col_w}}" for m in models)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for key, label in SUMMARY_METRICS:
        row = f"{label:<25}"
        for m in models:
            v = all_metrics[m].get(key)
            if isinstance(v, float):
                row += f"{v:>{col_w}.4f}"
            elif v is None:
                row += f"{'—':>{col_w}}"
            else:
                row += f"{str(v):>{col_w}}"
        print(row)
    print("=" * len(header))


def print_latex_table(all_metrics: dict):
    models = list(all_metrics.keys())
    print("\n% ---- LaTeX comparison table ----")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l" + "r" * len(models) + "}")
    print("\\hline")
    header = "Metric & " + " & ".join(DISPLAY_NAMES[m] for m in models) + " \\\\"
    print(header)
    print("\\hline")
    for key, label in SUMMARY_METRICS:
        row = label.replace("_", "\\_") + " & "
        vals = []
        for m in models:
            v = all_metrics[m].get(key)
            if isinstance(v, float) and v == v:
                vals.append(f"{v:.4f}")
            elif v is None or (isinstance(v, float) and v != v):
                vals.append("—")
            else:
                vals.append(str(v))
        row += " & ".join(vals) + " \\\\"
        print(row)
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Court keypoint detection model comparison on test set.}")
    print("\\label{tab:court-kp-comparison}")
    print("\\end{table}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_report(args):
    all_metrics = load_all(args.results_dir)
    if not all_metrics:
        print("No metric files found. Run evaluation first:")
        print("  python -m evaluation.evaluate --model tracknet_court --checkpoint ...")
        return

    os.makedirs(args.results_dir, exist_ok=True)

    plot_pck_curves(all_metrics,
        os.path.join(args.results_dir, "pck_curves.png"))

    plot_bar_metric(all_metrics, "pck@7px", "PCK @ 7px",
        "PCK @ 7px Comparison",
        os.path.join(args.results_dir, "pck7_comparison.png"))

    plot_bar_metric(all_metrics, "mean_kp_error_px", "Mean Error (pixels)",
        "Mean Keypoint Error",
        os.path.join(args.results_dir, "mean_kp_error.png"))

    plot_bar_metric(all_metrics, "mean_reproj_err_cm", "Mean Error (cm)",
        "Mean Court-Plane Reprojection Error",
        os.path.join(args.results_dir, "mean_reproj_error_cm.png"))

    plot_bar_metric(all_metrics, "homography_success_rate", "Success Rate",
        "Homography Estimation Success Rate",
        os.path.join(args.results_dir, "homography_success_rate.png"))

    plot_bar_metric(all_metrics, "court_center_pck@7px", "PCK @ 7px",
        "Court Center Accuracy @ 7px",
        os.path.join(args.results_dir, "court_center_accuracy.png"))

    plot_bar_metric(all_metrics, "FPS", "Frames per Second",
        "Inference Speed",
        os.path.join(args.results_dir, "fps_comparison.png"))

    plot_per_keypoint_pck(all_metrics,
        os.path.join(args.results_dir, "per_keypoint_pck.png"))

    plot_params_vs_pck(all_metrics,
        os.path.join(args.results_dir, "params_vs_pck.png"))

    plot_radar(all_metrics,
        os.path.join(args.results_dir, "radar_comparison.png"))

    print_console_table(all_metrics)
    print_latex_table(all_metrics)
    print(f"\nAll plots saved to {args.results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate report from court keypoint evaluation results.")
    parser.add_argument("--results_dir", default=RESULTS_DIR,
                        help="Directory containing *_metrics.json files")
    generate_report(parser.parse_args())
