"""
Run all models on the test split and collect their metrics.

Usage:
    python -m comparison.compare_models \\
        --tracknet_court_ckpt  checkpoints/tracknet_court_best.pt \\
        --resnet50_ckpt        checkpoints/resnet50_best.pt \\
        --hrnet_ckpt           checkpoints/hrnet_best.pt

Saves per-model JSON files and a combined summary JSON to results/.
"""

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent.parent


def run_eval(
    model: str,
    ckpt,
    split: str = "test",
    max_samples: int | None = None,
    confidence_threshold: str | None = None,
):
    """Launch evaluation as a subprocess so models don't share GPU memory."""
    cmd = [
        sys.executable, "-m", "evaluation.evaluate",
        "--model",      model,
        "--checkpoint", str(ckpt),
        "--split",      split,
    ]
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])
    if confidence_threshold is not None:
        cmd.extend(["--confidence_threshold", confidence_threshold])

    print(f"\n{'='*60}")
    print(f"Evaluating {model} ...")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] {model} eval failed:\n{result.stderr}")
        return None
    metrics_path = ROOT / "results" / f"{model}_metrics.json"
    with open(metrics_path) as f:
        return json.load(f)


def compare(args):
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "tracknet_court": args.tracknet_court_ckpt,
        "resnet50":       args.resnet50_ckpt,
        "hrnet":          args.hrnet_ckpt,
        "mobilenetv3":    args.mobilenetv3_ckpt,
    }

    # Only evaluate models whose checkpoints are provided
    available = {m: c for m, c in checkpoints.items() if c and pathlib.Path(c).exists()}
    if not available:
        print("No checkpoint files found. Run training first.")
        return

    all_metrics = {}
    for model, ckpt in available.items():
        metrics = run_eval(
            model,
            ckpt,
            split=args.split,
            max_samples=args.max_samples,
            confidence_threshold=args.confidence_threshold,
        )
        if metrics is not None:
            all_metrics[model] = metrics

    if not all_metrics:
        print("No evaluation results collected.")
        return

    out_path = results_dir / "comparison_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved comparison summary -> {out_path}")

    # Console table
    metrics_keys = [
        "pck@5px",
        "pck@7px",
        "pck@10px",
        "pck@25px",
        "mean_kp_error_px",
        "mean_reproj_err_cm",
        "max_reproj_err_cm",
        "homography_success_rate",
        "court_center_pck@7px",
        "params_M",
        "FPS",
    ]

    col_w = 25
    models_evaluated = list(all_metrics.keys())
    header = f"{'Metric':<25}" + "".join(f"{m:<{col_w}}" for m in models_evaluated)
    print(f"\n{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")
    for key in metrics_keys:
        row = f"{key:<25}"
        for model in models_evaluated:
            val = all_metrics[model].get(key)
            if isinstance(val, (int, float)) and val is not None:
                cell = f"{val:.4f}"
            else:
                cell = "N/A"
            row += f"{cell:<{col_w}}"
        print(row)
    print(f"{'─' * len(header)}")


if __name__ == "__main__":
    _ckpt_dir = ROOT / "checkpoints"
    parser = argparse.ArgumentParser(
        description="Compare court keypoint detection models on the test split."
    )
    parser.add_argument(
        "--tracknet_court_ckpt",
        default=str(_ckpt_dir / "tracknet_court_best.pt"),
    )
    parser.add_argument(
        "--resnet50_ckpt",
        default=str(_ckpt_dir / "resnet50_best.pt"),
    )
    parser.add_argument(
        "--hrnet_ckpt",
        default=str(_ckpt_dir / "hrnet_best.pt"),
    )
    parser.add_argument(
        "--mobilenetv3_ckpt",
        default=str(_ckpt_dir / "mobilenetv3_best.pt"),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional sample cap forwarded to evaluation.evaluate.",
    )
    parser.add_argument(
        "--confidence_threshold",
        default=None,
        help='Optional heatmap threshold forwarded to evaluation.evaluate, e.g. "0.3" or "none".',
    )
    compare(parser.parse_args())
