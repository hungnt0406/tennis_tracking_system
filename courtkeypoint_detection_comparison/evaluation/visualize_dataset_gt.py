"""Overlay the 14 ground-truth court keypoints on example frames.

Renders a panel per record showing the full 1280x720 frame, the 14 annotated
court keypoints (numbered in the dataset ordering), and the court lines implied
by those keypoints. Used to illustrate the court-keypoint dataset in the thesis.

Run from courtkeypoint_detection_comparison/:

    python -m evaluation.visualize_dataset_gt \
        --split train \
        --output_dir ../thesis/img \
        --ids PuXlxKdUIes_2450 7qCfURaFMpQ_1350
"""

import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Curated, representative records (different camera angles / surfaces).
DEFAULT_IDS = ["PuXlxKdUIes_2450", "7qCfURaFMpQ_1350"]

# Court lines as keypoint-index pairs, in the dataset's 14-keypoint ordering
# (0-3 outer doubles corners, 4-7 singles baseline corners, 8-11 service
# points, 12-13 centre T-points).
COURT_LINES = [
    (0, 1), (2, 3),     # doubles baselines (near, far)
    (0, 2), (1, 3),     # doubles sidelines (left, right)
    (4, 5), (6, 7),     # singles sidelines (left, right)
    (8, 9), (10, 11),   # service lines (near, far)
    (12, 13),           # centre service line
]


def load_records(split, data_dir):
    path = os.path.join(data_dir, f"data_{split}.json")
    with open(path) as f:
        return {r["id"]: r for r in json.load(f)}


def valid_pt(p):
    return p is not None and p[0] is not None and p[1] is not None and p[0] >= 0 and p[1] >= 0


def render_panel(rec, img_path, out_path, dpi):
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kps = rec["kps"]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.imshow(img)
    ax.set_axis_off()

    # Court lines first (so the markers sit on top).
    for a, b in COURT_LINES:
        if a < len(kps) and b < len(kps) and valid_pt(kps[a]) and valid_pt(kps[b]):
            ax.plot([kps[a][0], kps[b][0]], [kps[a][1], kps[b][1]],
                    color="yellow", linewidth=1.0, alpha=0.8)

    # Numbered keypoints.
    for i, p in enumerate(kps):
        if not valid_pt(p):
            continue
        x, y = p
        ax.add_patch(Circle((x, y), radius=7, facecolor="lime",
                            edgecolor="black", linewidth=0.8, zorder=3))
        ax.text(x + 9, y - 9, str(i), color="white", fontsize=7,
                zorder=4, bbox=dict(boxstyle="round,pad=0.1",
                                    facecolor="black", alpha=0.6, linewidth=0))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    n_valid = sum(valid_pt(p) for p in kps)
    print(f"wrote {out_path}  ({rec['id']}, {n_valid}/14 keypoints in-frame)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="../thesis/img")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--ids", nargs="+", default=DEFAULT_IDS,
                    help="record ids to render (one panel each)")
    args = ap.parse_args()

    records = load_records(args.split, args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    for n, rid in enumerate(args.ids, start=1):
        if rid not in records:
            raise SystemExit(f"id {rid!r} not found in data_{args.split}.json")
        img_path = os.path.join(args.data_dir, "images", f"{rid}.png")
        out_path = os.path.join(args.output_dir, f"court_gt_example{n}.png")
        render_panel(records[rid], img_path, out_path, args.dpi)


if __name__ == "__main__":
    main()
