#!/usr/bin/env python3
"""Generate Chapter 5 (Court Keypoint Detection) figures from locked metrics JSON.

Reads results/comparison_summary.json and writes 6 PNGs into thesis/img/.
No training or evaluation is run; all numbers come from the JSON.
"""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path("/Users/hungcucu/Documents/usth/tennis_tracking_system")
SRC_JSON = REPO / "courtkeypoint_detection_comparison/results/comparison_summary.json"
VIS_DIR = REPO / "courtkeypoint_detection_comparison/results/visualizations"
OUT_DIR = REPO / "thesis/img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERLAY_SRC = VIS_DIR / "mobilenetv3_PIpT0JzKjRA_2350.png"

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
MODEL_ORDER = ["tracknet_court", "resnet50", "hrnet", "mobilenetv3"]
DISPLAY = {
    "tracknet_court": "TrackNet-court",
    "resnet50": "ResNet50",
    "hrnet": "HRNet",
    "mobilenetv3": "MobileNetV3",
}
# MobileNetV3 (selected model) gets an emphasis green; others muted blue-greys.
COLORS = {
    "tracknet_court": "#5b7c99",  # muted slate blue
    "resnet50": "#9aa6b2",        # light blue-grey
    "hrnet": "#34495e",           # dark blue-grey
    "mobilenetv3": "#2ca02c",     # strong green (emphasis)
}

TITLE_FS = 13
LABEL_FS = 11
TICK_FS = 10

PCK_TOLS = [5, 7, 10, 25]


def load_data():
    with open(SRC_JSON) as f:
        return json.load(f)


def style_axes(ax):
    ax.tick_params(labelsize=TICK_FS)


# ---------------------------------------------------------------------------
# 1. PCK vs distance tolerance (line plot)
# ---------------------------------------------------------------------------
def fig_pck_curves(data):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for m in MODEL_ORDER:
        ys = [data[m][f"pck@{t}px"] for t in PCK_TOLS]
        lw = 2.2 if m == "mobilenetv3" else 1.6
        ax.plot(
            PCK_TOLS, ys, marker="o", markersize=5, linewidth=lw,
            color=COLORS[m], label=DISPLAY[m],
        )
    ax.set_xlabel("Pixel tolerance (px)", fontsize=LABEL_FS)
    ax.set_ylabel("PCK", fontsize=LABEL_FS)
    ax.set_title("PCK vs distance tolerance", fontsize=TITLE_FS)
    ax.set_xticks(PCK_TOLS)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "court_pck_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Mean keypoint error (bar chart)
# ---------------------------------------------------------------------------
def fig_mean_kp_error(data):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vals = [data[m]["mean_kp_error_px"] for m in MODEL_ORDER]
    xs = np.arange(len(MODEL_ORDER))
    bars = ax.bar(xs, vals, color=[COLORS[m] for m in MODEL_ORDER], width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[m] for m in MODEL_ORDER], fontsize=TICK_FS, rotation=15)
    ax.set_ylabel("Mean keypoint error (px)", fontsize=LABEL_FS)
    ax.set_ylim(0, max(vals) * 1.18)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=TICK_FS)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "court_mean_kp_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Mean reprojection error (bar chart)
# ---------------------------------------------------------------------------
def fig_reproj_error(data):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vals = [data[m]["mean_reproj_err_cm"] for m in MODEL_ORDER]
    xs = np.arange(len(MODEL_ORDER))
    bars = ax.bar(xs, vals, color=[COLORS[m] for m in MODEL_ORDER], width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[m] for m in MODEL_ORDER], fontsize=TICK_FS, rotation=15)
    ax.set_ylabel("Mean reprojection error (cm)", fontsize=LABEL_FS)
    ax.set_ylim(0, max(vals) * 1.18)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=TICK_FS)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "court_reproj_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Accuracy vs model size (scatter, log-x)
# ---------------------------------------------------------------------------
def fig_params_vs_pck(data):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for m in MODEL_ORDER:
        x = data[m]["params_M"]
        y = data[m]["pck@7px"]
        size = 130 if m == "mobilenetv3" else 80
        edge = "black" if m == "mobilenetv3" else "none"
        ax.scatter(x, y, s=size, color=COLORS[m], edgecolors=edge,
                   linewidths=1.2, zorder=3)
    # Annotate each point; nudge labels to avoid overlap.
    offsets = {
        "tracknet_court": (8, 6),
        "resnet50": (8, -12),
        "hrnet": (8, 8),
        "mobilenetv3": (10, -4),
    }
    for m in MODEL_ORDER:
        x = data[m]["params_M"]
        y = data[m]["pck@7px"]
        dx, dy = offsets[m]
        ax.annotate(DISPLAY[m], (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9,
                    fontweight="bold" if m == "mobilenetv3" else "normal",
                    color=COLORS[m])
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log scale)", fontsize=LABEL_FS)
    ax.set_ylabel("PCK@7px", fontsize=LABEL_FS)
    ax.set_title("Accuracy vs model size", fontsize=TITLE_FS)
    ax.set_ylim(0.6, 1.02)
    ax.grid(alpha=0.3, which="both")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "court_params_vs_pck.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Per-keypoint PCK@7px (line plot, 14 keypoints)
# ---------------------------------------------------------------------------
def fig_per_keypoint_pck(data):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    kp_idx = np.arange(14)
    for m in MODEL_ORDER:
        ys = data[m]["per_kp_pck@7px"]
        lw = 2.2 if m == "mobilenetv3" else 1.5
        ax.plot(kp_idx, ys, marker="o", markersize=4, linewidth=lw,
                color=COLORS[m], label=DISPLAY[m])
    ax.set_xlabel("Keypoint index", fontsize=LABEL_FS)
    ax.set_ylabel("PCK@7px", fontsize=LABEL_FS)
    ax.set_title("Per-keypoint accuracy", fontsize=TITLE_FS)
    ax.set_xticks(kp_idx)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left", ncol=2)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "court_per_keypoint_pck.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Qualitative overlay (copy existing image)
# ---------------------------------------------------------------------------
def copy_overlay():
    dst = OUT_DIR / "court_overlay_mobilenetv3.png"
    src = OVERLAY_SRC
    if not src.exists():
        candidates = sorted(VIS_DIR.glob("mobilenetv3_*.png"))
        if not candidates:
            raise FileNotFoundError("No mobilenetv3_*.png overlay found.")
        src = candidates[0]
    shutil.copyfile(src, dst)
    print(f"Copied overlay: {src.name} -> {dst.name}")


def main():
    data = load_data()
    fig_pck_curves(data)
    fig_mean_kp_error(data)
    fig_reproj_error(data)
    fig_params_vs_pck(data)
    fig_per_keypoint_pck(data)
    copy_overlay()
    print("All figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
