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
    "heuristic": "#2196F3",
    "gbm":       "#FF5722",
    "tcn":       "#4CAF50",
}

DISPLAY_NAMES = {
    "heuristic": "Heuristic",
    "gbm":       "GBM",
    "tcn":       "TCN",
}

# Tolerance values (frames) for the F1@k* sweep.
TOLERANCES = [0, 1, 2, 3, 5, 7]

# Games in canonical order for the per-game grouped bar chart.
GAMES = [f"game{i}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all(results_dir: str) -> dict:
    all_metrics = {}
    for model in ("heuristic", "gbm", "tcn"):
        path = os.path.join(results_dir, f"{model}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                all_metrics[model] = json.load(f)
        else:
            print(f"[WARN] missing metrics for '{model}': {path} (skipping)")
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

def plot_pr_curve(all_metrics: dict, out_path: str):
    """Precision (y) vs recall (x), one line per model, AP in the legend."""
    fig, ax = plt.subplots(figsize=(7, 5))
    drew = False
    for model, metrics in all_metrics.items():
        pr = metrics.get("pr_curve")
        if not pr:
            continue
        recall = [_safe(r) for r in pr.get("recall", [])]
        precision = [_safe(p) for p in pr.get("precision", [])]
        if not recall or not precision:
            continue
        ap = _safe(metrics.get("AP"))
        ax.plot(recall, precision, "o-",
                label=f"{DISPLAY_NAMES[model]} (AP={ap:.3f})",
                color=MODEL_COLORS.get(model, "gray"),
                linewidth=2, markersize=5)
        drew = True
    if not drew:
        plt.close(fig)
        print(f"[WARN] no pr_curve data — skipping {out_path}")
        return
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_f1_vs_tolerance(all_metrics: dict, out_path: str):
    """x = tolerance k, y = F1, one line per model (from F1@k* keys)."""
    keys = [f"F1@k{k}" for k in TOLERANCES]
    fig, ax = plt.subplots(figsize=(7, 5))
    drew = False
    for model, metrics in all_metrics.items():
        vals = [_safe(metrics.get(k)) for k in keys]
        if not any(k in metrics for k in keys):
            continue
        ax.plot(TOLERANCES, vals, "o-",
                label=DISPLAY_NAMES[model],
                color=MODEL_COLORS.get(model, "gray"),
                linewidth=2, markersize=7)
        drew = True
    if not drew:
        plt.close(fig)
        print(f"[WARN] no F1@k data — skipping {out_path}")
        return
    ax.set_xlabel("Temporal tolerance k (frames)")
    ax.set_ylabel("Event F1")
    ax.set_title("Event F1 vs Temporal Tolerance")
    ax.legend()
    ax.set_xticks(TOLERANCES)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_per_game_f1(all_metrics: dict, out_path: str):
    """Grouped bar chart: one group per game, one bar per model."""
    models = list(all_metrics.keys())
    if not models:
        return
    # Union of games present across models, sorted in canonical order.
    present = set()
    for m in models:
        present.update((all_metrics[m].get("per_game_F1") or {}).keys())
    games = [g for g in GAMES if g in present]
    games += sorted(g for g in present if g not in GAMES)
    if not games:
        print(f"[WARN] no per_game_F1 data — skipping {out_path}")
        return

    x = np.arange(len(games))
    n_models = len(models)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(8, len(games) * 1.1), 5))
    for i, model in enumerate(models):
        per_game = all_metrics[model].get("per_game_F1") or {}
        vals = [_safe(per_game.get(g)) for g in games]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width,
               label=DISPLAY_NAMES[model],
               color=MODEL_COLORS.get(model, "gray"),
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Event F1@k")
    ax.set_title("Per-Game Event F1")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_bounce_vs_hit_confusion(all_metrics: dict, out_path: str):
    """Stacked bar per model of [bounce, hit, none] predicted positives."""
    models = list(all_metrics.keys())
    rows = {m: all_metrics[m].get("confusion_bounce_hit_none")
            for m in models}
    models = [m for m in models if rows[m] and len(rows[m]) == 3]
    if not models:
        print(f"[WARN] no confusion_bounce_hit_none data — skipping {out_path}")
        return

    bounce = [_safe(rows[m][0]) for m in models]
    hit    = [_safe(rows[m][1]) for m in models]
    none   = [_safe(rows[m][2]) for m in models]
    labels = [DISPLAY_NAMES[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, bounce, width=0.5, label="bounce",
           color="#4CAF50")
    ax.bar(labels, hit, width=0.5, bottom=bounce, label="hit",
           color="#FF9800")
    ax.bar(labels, none, width=0.5,
           bottom=[b + h for b, h in zip(bounce, hit)], label="none",
           color="#9E9E9E")
    ax.set_ylabel("Predicted positives (count)")
    ax.set_title("What Predicted Bounces Actually Are")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
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


# ---------------------------------------------------------------------------
# Console + LaTeX tables
# ---------------------------------------------------------------------------

SUMMARY_METRICS = [
    ("event_F1@k",        "Event F1@k"),
    ("event_precision@k", "Precision@k"),
    ("event_recall@k",    "Recall@k"),
    ("AP",                "AP"),
    ("F1@k1",             "F1@k1"),
    ("FPS",               "FPS"),
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
    print("\\caption{Bounce detection model comparison on test set.}")
    print("\\label{tab:bounce-detection-comparison}")
    print("\\end{table}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_report(args):
    all_metrics = load_all(args.results_dir)
    if not all_metrics:
        print("No metric files found. Run evaluation first:")
        print("  python -m evaluation.evaluate --model heuristic --checkpoint ...")
        return

    os.makedirs(args.results_dir, exist_ok=True)

    plot_pr_curve(all_metrics,
        os.path.join(args.results_dir, "pr_curve.png"))

    plot_f1_vs_tolerance(all_metrics,
        os.path.join(args.results_dir, "f1_vs_tolerance.png"))

    plot_per_game_f1(all_metrics,
        os.path.join(args.results_dir, "per_game_f1.png"))

    plot_bounce_vs_hit_confusion(all_metrics,
        os.path.join(args.results_dir, "bounce_vs_hit_confusion.png"))

    plot_bar_metric(all_metrics, "event_F1@k", "Event F1@k",
        "Event F1@k Comparison",
        os.path.join(args.results_dir, "event_f1_bar.png"))

    plot_bar_metric(all_metrics, "FPS", "Frames per Second",
        "Inference Speed",
        os.path.join(args.results_dir, "fps_bar.png"))

    print_console_table(all_metrics)
    print_latex_table(all_metrics)
    print(f"\nAll plots saved to {args.results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate report from bounce detection evaluation results.")
    parser.add_argument("--results_dir", default=RESULTS_DIR,
                        help="Directory containing *_metrics.json files")
    generate_report(parser.parse_args())
