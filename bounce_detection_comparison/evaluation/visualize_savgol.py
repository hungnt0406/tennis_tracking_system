"""Contrast the two ways to get velocity/acceleration from a noisy ball track:
plain finite differences (NO Savitzky-Golay) vs the Savitzky-Golay
smoothing-differentiation filter the pipeline actually uses.

Both signals start from the SAME gap-filled positions (`clean_trajectory`), so
the only difference is the differentiation operator:

  WITHOUT savgol : vy = np.gradient(y),  ay = np.gradient(vy)   (central diff)
  WITH    savgol : vy = savgol_filter(y, w, p, deriv=1)
                   ay = savgol_filter(y, w, p, deriv=2)         (`compute_kinematics`)

A Savitzky-Golay filter fits a low-order polynomial to a sliding window and
reads the derivative off that polynomial, so it differentiates and denoises in
one pass. Finite differences amplify pixel jitter, badly so for the second
derivative (acceleration), which is exactly the signal the bounce detector
leans on (a vertical-velocity sign flip and an |ay| spike). This figure is the
"why we smooth" companion to evaluation/visualize_trajectory.py.

Reads coordinates straight from splits.csv (no image IO). Bounce = status 2.

Run from bounce_detection_comparison/:

    python -m evaluation.visualize_savgol --clip game7/Clip2
    python -m evaluation.visualize_savgol --clip game7/Clip2 --frame_start 80 --frame_end 105
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from data.trajectory import clean_trajectory, compute_kinematics, _contiguous_runs
from train.config import FEATURE


def load_clip(splits_csv, game, clip):
    rows = [r for r in csv.DictReader(open(splits_csv))
            if r["game"] == game and r["clip"] == clip]
    if not rows:
        raise SystemExit(f"clip {game}/{clip} not found in {splits_csv}")
    rows.sort(key=lambda r: int(r["frame_idx"]))
    f = np.array([int(r["frame_idx"]) for r in rows])
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    vis = np.array([int(r["visibility"]) > 0 for r in rows])
    status = np.array([int(r["status"]) for r in rows])
    return f, x, y, vis, status


def kinematics_no_savgol(y):
    """Example WITHOUT the filter: vertical velocity/acceleration by plain
    central finite differences, per contiguous valid run (mirrors the run
    handling of compute_kinematics but with NO smoothing)."""
    T = len(y)
    vy = np.full(T, np.nan)
    ay = np.full(T, np.nan)
    valid = ~np.isnan(y)
    for s, e in _contiguous_runs(valid):
        if e - s >= 2:
            vy[s:e] = np.gradient(y[s:e])
            ay[s:e] = np.gradient(vy[s:e])
        else:
            vy[s:e] = ay[s:e] = 0.0
    return vy, ay


def render(f, y, vy_raw, ay_raw, vy_sg, ay_sg, bounce, out_path, dpi):
    fig, (ay_y, ax_v, ax_a) = plt.subplots(3, 1, sharex=True, figsize=(7.5, 6.2))

    # row 0: the cleaned vertical position the two methods both start from
    ay_y.plot(f, y, color="0.35", linewidth=1.2)
    ay_y.set_ylabel("vertical $y$\n(pixels)")
    ay_y.invert_yaxis()  # image y grows downward; ground bounce is a local max
    ay_y.grid(True, alpha=0.25, linewidth=0.5)

    # row 1: first derivative (velocity)
    ax_v.plot(f, vy_raw, color="tab:red", linewidth=0.9, alpha=0.85,
              label="finite difference (no Savitzky-Golay)")
    ax_v.plot(f, vy_sg, color="tab:blue", linewidth=1.8,
              label="Savitzky-Golay (deriv=1)")
    ax_v.axhline(0.0, color="0.6", linewidth=0.6)
    ax_v.set_ylabel("vertical velocity\n$v_y$")
    ax_v.grid(True, alpha=0.25, linewidth=0.5)
    ax_v.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # row 2: second derivative (acceleration) — where noise hurts most
    ax_a.plot(f, ay_raw, color="tab:red", linewidth=0.9, alpha=0.85,
              label="finite difference (no Savitzky-Golay)")
    ax_a.plot(f, ay_sg, color="tab:blue", linewidth=1.8,
              label="Savitzky-Golay (deriv=2)")
    ax_a.axhline(0.0, color="0.6", linewidth=0.6)
    ax_a.set_ylabel("vertical accel.\n$a_y$")
    ax_a.set_xlabel("frame index")
    ax_a.grid(True, alpha=0.25, linewidth=0.5)
    ax_a.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for ax in (ay_y, ax_v, ax_a):
        for b in f[bounce]:
            ax.axvline(b, color="forestgreen", linewidth=1.0, alpha=0.5,
                       zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", default="splits.csv")
    ap.add_argument("--clip", default="game7/Clip2", help="'game/clip'")
    ap.add_argument("--frame_start", type=int, default=None,
                    help="zoom: first frame index to plot")
    ap.add_argument("--frame_end", type=int, default=None,
                    help="zoom: last frame index to plot")
    ap.add_argument("--output_dir", default="../thesis/img")
    ap.add_argument("--name", default="savgol_vs_finite_difference")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    game, clip = args.clip.split("/")
    f, x, y, vis, status = load_clip(args.splits_csv, game, clip)

    # shared starting point: gap-filled positions, NaN where there is no ball
    xc, yc, _, valid = clean_trajectory(x, y, vis, FEATURE["max_gap"])

    # WITHOUT the filter (plain central differences)
    vy_raw, ay_raw = kinematics_no_savgol(yc)
    # WITH the filter (the pipeline's compute_kinematics)
    _, vy_sg, _, ay_sg = compute_kinematics(
        xc, yc, FEATURE["savgol_window"], FEATURE["savgol_poly"])

    bounce = status == 2

    # quantify the noise the smoothing removes (valid frames only)
    m = valid & ~np.isnan(ay_raw) & ~np.isnan(ay_sg)
    print(f"clip {args.clip}: {len(f)} frames, {int(bounce.sum())} bounces")
    print(f"std(a_y)  no-savgol = {np.std(ay_raw[m]):.3f}   "
          f"savgol = {np.std(ay_sg[m]):.3f}   "
          f"(noise x{np.std(ay_raw[m]) / (np.std(ay_sg[m]) + 1e-9):.1f} smaller)")

    # optional zoom window
    if args.frame_start is not None or args.frame_end is not None:
        lo = args.frame_start if args.frame_start is not None else f.min()
        hi = args.frame_end if args.frame_end is not None else f.max()
        sel = (f >= lo) & (f <= hi)
        f, yc = f[sel], yc[sel]
        vy_raw, ay_raw = vy_raw[sel], ay_raw[sel]
        vy_sg, ay_sg = vy_sg[sel], ay_sg[sel]
        bounce = bounce[sel]

    os.makedirs(args.output_dir, exist_ok=True)
    render(f, yc, vy_raw, ay_raw, vy_sg, ay_sg, bounce,
           os.path.join(args.output_dir, f"{args.name}.png"), args.dpi)


if __name__ == "__main__":
    main()
