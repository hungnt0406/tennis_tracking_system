"""
Run all three bounce-detection arms and aggregate to comparison_summary.json.

Each arm is evaluated in its own subprocess (`python -m evaluation.evaluate`) so
they don't share memory; a failing arm is tolerated with a [WARN] rather than
aborting the run (court-keypoint project precedent). The heuristic always runs
(config fallback); gbm/tcn require a trained checkpoint.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["heuristic", "gbm", "tcn"]
DEFAULT_CKPT = {
    "heuristic": "checkpoints/heuristic_best.json",
    "gbm":       "checkpoints/gbm_best.pkl",
    "tcn":       "checkpoints/tcn_best.pt",
}


def run_eval(model, checkpoint, splits_csv, split, results_dir):
    cmd = [sys.executable, "-m", "evaluation.evaluate",
           "--model", model, "--checkpoint", checkpoint,
           "--splits_csv", splits_csv, "--split", split,
           "--results_dir", results_dir]
    print(f"\n{'=' * 60}\nEvaluating {model} …\n{'=' * 60}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print(f"[WARN] {model} failed:\n{r.stderr}")
        return None
    path = os.path.join(results_dir, f"{model}_metrics.json")
    if os.path.exists(os.path.join(ROOT, path)) or os.path.exists(path):
        with open(os.path.join(ROOT, path) if not os.path.isabs(path) else path) as f:
            return json.load(f)
    return None


def compare(args):
    os.makedirs(os.path.join(ROOT, args.results_dir), exist_ok=True)
    all_metrics = {}
    for m in MODELS:
        ckpt = DEFAULT_CKPT[m]
        if m != "heuristic" and not os.path.exists(os.path.join(ROOT, ckpt)):
            print(f"[WARN] {m}: checkpoint {ckpt} missing — train it first; skipping.")
            continue
        res = run_eval(m, ckpt, args.splits_csv, args.split, args.results_dir)
        if res is not None:
            all_metrics[m] = res

    out = os.path.join(ROOT, args.results_dir, "comparison_summary.json")
    with open(out, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved comparison summary → {out}")

    if all_metrics:
        cols = ["event_F1@k", "event_precision@k", "event_recall@k", "AP", "F1@k1", "FPS"]
        print(f"\n{'model':<12}" + "".join(f"{c:>20}" for c in cols))
        for m, mt in all_metrics.items():
            row = f"{m:<12}"
            for c in cols:
                v = mt.get(c)
                row += f"{v:>20.3f}" if isinstance(v, (int, float)) else f"{'—':>20}"
            print(row)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits_csv", default="splits.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--results_dir", default="results")
    compare(p.parse_args())
