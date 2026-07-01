"""Illustrate the TrackNet status field (0 in-flight, 1 hit, 2 bounce).

A single frame cannot show what makes a frame a hit or a bounce; the cue is
kinematic. Each panel therefore overlays the ball positions in a short window
around one event onto the event frame, coloured by time so the direction of
motion (and the reversal at a hit or bounce) is visible. The event frame's ball
is ringed.

Run from bounce_detection_comparison/:

    python -m evaluation.visualize_status \
        --output_dir ../thesis/img \
        --status0 game7/Clip4/0760 \
        --status1 game4/Clip6/0043 \
        --status2 game2/Clip8/0007
"""

import argparse
import csv
from collections import defaultdict
import os

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

WINDOW = 7  # frames on each side of the event

DEFAULT_EVENTS = {
    0: "game5/Clip4/0192",   # in flight: smooth arc, event-free window
    1: "game4/Clip6/0043",   # hit: ball struck at the racket
    2: "game2/Clip8/0007",   # bounce: ball lands on the court
}


def load_clips(splits_csv):
    rows = list(csv.DictReader(open(splits_csv)))
    clips = defaultdict(list)
    for r in rows:
        clips[(r["game"], r["clip"])].append(r)
    for seq in clips.values():
        seq.sort(key=lambda r: int(r["frame_idx"]))
    return clips


def find_index(seq, stem):
    target = f"{stem}.jpg"
    for i, r in enumerate(seq):
        if r["frame_name"] == target:
            return i
    raise SystemExit(f"frame {stem} not found in clip")


def render_panel(seq, i, out_path, dpi):
    event = seq[i]
    img = cv2.cvtColor(cv2.imread(event["frame_path"]), cv2.COLOR_BGR2RGB)
    lo, hi = max(0, i - WINDOW), min(len(seq), i + WINDOW + 1)
    xs, ys, ts = [], [], []
    for t, r in enumerate(seq[lo:hi]):
        x, y = float(r["x"]), float(r["y"])
        if x >= 0 and y >= 0:
            xs.append(x); ys.append(y); ts.append(t)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.imshow(img)
    ax.set_axis_off()
    # Trajectory path, then time-coloured points on top.
    ax.plot(xs, ys, color="white", linewidth=1.0, alpha=0.7, zorder=2)
    ax.scatter(xs, ys, c=ts, cmap="plasma", s=26, edgecolor="black",
               linewidth=0.4, zorder=3)
    # Ring the event frame's ball.
    ex, ey = float(event["x"]), float(event["y"])
    if ex >= 0:
        ax.add_patch(Circle((ex, ey), radius=18, fill=False,
                            edgecolor="lime", linewidth=2.0, zorder=4))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out_path}  ({event['game']}/{event['clip']}/"
          f"{event['frame_name']}, status={event['status']}, "
          f"{len(xs)} visible points in window)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", default="splits.csv")
    ap.add_argument("--output_dir", default="../thesis/img")
    ap.add_argument("--dpi", type=int, default=200)
    for s in range(3):
        ap.add_argument(f"--status{s}", default=DEFAULT_EVENTS[s],
                        help=f"status {s} example, 'game/clip/frame_stem'")
    args = ap.parse_args()

    clips = load_clips(args.splits_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    for s in range(3):
        game, clip, stem = getattr(args, f"status{s}").split("/")
        seq = clips[(game, clip)]
        if not seq:
            raise SystemExit(f"clip {game}/{clip} not found")
        i = find_index(seq, stem)
        if seq[i]["status"] != str(s):
            print(f"  warning: {game}/{clip}/{stem} has status "
                  f"{seq[i]['status']}, requested {s}")
        render_panel(seq, i, os.path.join(args.output_dir, f"status_class{s}.png"),
                     args.dpi)


if __name__ == "__main__":
    main()
