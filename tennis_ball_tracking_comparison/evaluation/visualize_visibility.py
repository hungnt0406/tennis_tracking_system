"""Render one example frame per TrackNet visibility class with its GT ball overlay.

Produces a clean panel per visibility class (0..3) showing the full 1280x720
frame, the ground-truth ball position, and a magnified inset around the ball so
the (few-pixel) ball is actually legible. Used to illustrate the dataset's
four-level visibility annotation in the thesis.

Run from tennis_ball_tracking_comparison/:

    python -m evaluation.visualize_visibility \
        --splits_csv splits.csv \
        --output_dir ../thesis/img

Each class example can be overridden, e.g. --vis2 game6/Clip3/0094.
"""

import argparse
import csv
import os

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# Curated, representative example per visibility class: "game/clip/frame_stem".
# Class 0 has no ball, so its coordinate is the (-1,-1) sentinel.
DEFAULT_EXAMPLES = {
    0: "game4/Clip5/0046",   # no visible ball (sentinel -1,-1)
    1: "game2/Clip5/0000",   # easily visible
    2: "game3/Clip7/0023",   # present but hard to identify
    3: "game7/Clip5/0013",   # occluded by player/racket
}

CLASS_LABEL = {
    0: "Visibility 0: ball not visible",
    1: "Visibility 1: easily visible",
    2: "Visibility 2: hard to identify",
    3: "Visibility 3: occluded",
}

INSET_HALF = 70  # half-size (px) of the square zoom region around the ball


def load_rows(splits_csv):
    with open(splits_csv) as f:
        return list(csv.DictReader(f))


def find_row(rows, spec):
    """spec = 'game/clip/frame_stem' -> matching CSV row."""
    game, clip, stem = spec.split("/")
    target = f"{stem}.jpg"
    for r in rows:
        if r["game"] == game and r["clip"] == clip and r["frame_name"] == target:
            return r
    raise SystemExit(f"No frame matches {spec!r} in the split manifest.")


def render_panel(row, out_path, dpi):
    img = cv2.imread(row["frame_path"])
    if img is None:
        raise SystemExit(f"Could not read image: {row['frame_path']}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    x, y = float(row["x"]), float(row["y"])
    visible = x >= 0 and y >= 0

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.imshow(img)
    ax.set_axis_off()

    if visible:
        # Ball marker on the full frame.
        ax.add_patch(Circle((x, y), radius=14, fill=False,
                            edgecolor="lime", linewidth=1.8))
        # Zoom region rectangle on the full frame.
        x0, y0 = max(0, x - INSET_HALF), max(0, y - INSET_HALF)
        x1, y1 = min(w, x + INSET_HALF), min(h, y + INSET_HALF)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                              edgecolor="yellow", linewidth=1.2))

        # Magnified inset (top-right corner) showing the ball region.
        crop = img[int(y0):int(y1), int(x0):int(x1)]
        axins = ax.inset_axes([0.66, 0.60, 0.34, 0.40])
        axins.imshow(crop, extent=[x0, x1, y1, y0])
        axins.add_patch(Circle((x, y), radius=12, fill=False,
                              edgecolor="lime", linewidth=1.8))
        axins.set_xlim(x0, x1)
        axins.set_ylim(y1, y0)
        axins.set_xticks([])
        axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_edgecolor("yellow")
            spine.set_linewidth(1.2)
    else:
        ax.text(0.5, 0.06, "no ball annotated (sentinel x, y = -1, -1)",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.6))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out_path}  ({row['game']}/{row['clip']}/{row['frame_name']}, "
          f"vis={row['visibility']}, xy=({row['x']},{row['y']}))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", default="splits.csv")
    ap.add_argument("--output_dir", default="../thesis/img")
    ap.add_argument("--dpi", type=int, default=200)
    for c in range(4):
        ap.add_argument(f"--vis{c}", default=DEFAULT_EXAMPLES[c],
                        help=f"override class {c} example, 'game/clip/frame_stem'")
    args = ap.parse_args()

    rows = load_rows(args.splits_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    for c in range(4):
        spec = getattr(args, f"vis{c}")
        row = find_row(rows, spec)
        if int(row["visibility"]) != c:
            print(f"  warning: {spec} has visibility {row['visibility']}, "
                  f"requested class {c}")
        out_path = os.path.join(args.output_dir, f"visibility_class{c}.png")
        render_panel(row, out_path, args.dpi)


if __name__ == "__main__":
    main()
