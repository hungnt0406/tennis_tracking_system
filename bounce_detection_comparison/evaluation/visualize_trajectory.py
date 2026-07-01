"""Plot a rally's ball-coordinate time series with bounces and hits marked.

This is the motivating figure for the bounce chapter: it shows that the bounce
signature lives in the coordinate time series (a reversal of the vertical
motion) and that a racket hit produces the same kind of reversal, so a
hand-coded "detect the reversal" rule cannot separate the two on its own.

Reads coordinates straight from splits.csv (no image IO), so it does not need
the Dataset frames to be present. Bounce = status 2 (green), hit = status 1
(brown).

Run from bounce_detection_comparison/:

    python -m evaluation.visualize_trajectory --clip game7/Clip2
"""

import argparse
import csv
from collections import defaultdict
import os

import matplotlib.pyplot as plt
import numpy as np


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
    # break the plotted line at missing/invisible frames
    x = np.where((x >= 0) & vis, x, np.nan)
    y = np.where((y >= 0) & vis, y, np.nan)
    return f, x, y, status


def render(f, x, y, status, out_path, dpi):
    bounce = status == 2
    hit = status == 1
    fig, (axx, axy) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 4.2))

    for ax, coord, name in ((axx, x, "horizontal $x$"), (axy, y, "vertical $y$")):
        ax.plot(f, coord, color="0.45", linewidth=1.1, zorder=1)
        ax.scatter(f[hit], coord[hit], marker="s", s=46, c="saddlebrown",
                   edgecolor="black", linewidth=0.4, zorder=3,
                   label="racket hit (status 1)")
        ax.scatter(f[bounce], coord[bounce], marker="o", s=46, c="forestgreen",
                   edgecolor="black", linewidth=0.4, zorder=4,
                   label="bounce (status 2)")
        ax.set_ylabel(f"{name}\n(pixels)")
        ax.grid(True, alpha=0.25, linewidth=0.5)
    axy.invert_yaxis()  # image y grows downward; court toward the bottom
    axy.set_xlabel("frame index")
    axx.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_path}  ({int(bounce.sum())} bounces, "
          f"{int(hit.sum())} hits, {len(f)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", default="splits.csv")
    ap.add_argument("--clip", default="game7/Clip2", help="'game/clip'")
    ap.add_argument("--output_dir", default="../thesis/img")
    ap.add_argument("--name", default="bounce_trajectory_motivation")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    game, clip = args.clip.split("/")
    f, x, y, status = load_clip(args.splits_csv, game, clip)
    os.makedirs(args.output_dir, exist_ok=True)
    render(f, x, y, status, os.path.join(args.output_dir, f"{args.name}.png"),
           args.dpi)


if __name__ == "__main__":
    main()
