"""
Create train/val/test splits for bounce detection from ALL 10 games.

Bounce events (status==2) are scarce — 523 across the whole dataset — so unlike
the sibling ball-tracking project (which uses only game1/2/7) this pulls every
game. One game (default game7) is held out entirely as the test set, matching
the sibling's held-out-game policy for cross-project comparability. The
remaining 9 games are split by clip into train/val, *stratified on per-clip
bounce density* so the scarce positives are balanced across splits.

Outputs a CSV with columns:
  game, clip, frame_path, label_path, frame_name,
  visibility, x, y, status, frame_idx, is_bounce, split

`frame_idx` (int parsed from the frame filename) is the authoritative ordering
within a clip; `is_bounce` is 1 iff status==2.
"""

import os
import csv
import random
import argparse
from pathlib import Path


ALL_GAMES = [f"game{i}" for i in range(1, 11)]
TRAIN_RATIO = 0.825   # of the non-test games; val = 1 - TRAIN_RATIO.
# game7 held out as test ≈ 10% of clips → overall ≈ 74/16/10 train/val/test.


def collect_clips(dataset_root: Path, game: str):
    game_dir = dataset_root / game
    clips = sorted(
        [d for d in game_dir.iterdir() if d.is_dir()],
        key=lambda p: int(p.name.replace("Clip", "")),
    )
    return clips


def count_bounces(clip_dir: Path) -> int:
    label_file = clip_dir / "Label.csv"
    if not label_file.exists():
        return 0
    n = 0
    with open(label_file) as f:
        for row in csv.DictReader(f):
            if row["status"].strip() == "2":
                n += 1
    return n


def stratified_split(clips, train_ratio, seed=42):
    """Split (clip_dir, game) pairs into train/val, stratified by bounce count
    (tertiles) so bounce density is balanced across the two splits."""
    counted = [(c, g, count_bounces(c)) for (c, g) in clips]
    counted.sort(key=lambda t: t[2])
    n = len(counted)
    strata = [counted[: n // 3], counted[n // 3 : 2 * n // 3], counted[2 * n // 3 :]]
    rng = random.Random(seed)
    train, val = [], []
    for stratum in strata:
        s = stratum[:]
        rng.shuffle(s)
        n_train = max(1, int(round(len(s) * train_ratio)))
        train.extend(s[:n_train])
        val.extend(s[n_train:])
    return [(c, g) for (c, g, _) in train], [(c, g) for (c, g, _) in val]


def build_records(clips, split):
    """clips: list of (clip_dir, game)."""
    records = []
    for clip_dir, game in clips:
        label_file = clip_dir / "Label.csv"
        if not label_file.exists():
            continue
        with open(label_file) as f:
            for row in csv.DictReader(f):
                fname = row["file name"]
                frame_path = clip_dir / fname
                if not frame_path.exists():
                    continue
                vis = int(row["visibility"])
                x_str = row["x-coordinate"].strip()
                y_str = row["y-coordinate"].strip()
                x = float(x_str) if x_str else -1.0
                y = float(y_str) if y_str else -1.0
                status = int(row["status"]) if row["status"].strip() else 0
                try:
                    frame_idx = int(os.path.splitext(fname)[0])
                except ValueError:
                    frame_idx = len(records)
                records.append({
                    "game": game,
                    "clip": clip_dir.name,
                    "frame_path": str(frame_path),
                    "label_path": str(label_file),
                    "frame_name": fname,
                    "visibility": vis,
                    "x": x,
                    "y": y,
                    "status": status,
                    "frame_idx": frame_idx,
                    "is_bounce": 1 if status == 2 else 0,
                    "split": split,
                })
    return records


def create_splits(dataset_root, output_path, test_games, seed=42):
    dataset_root = Path(dataset_root)
    test_games = [g.strip() for g in test_games if g.strip()]

    trainval_clips = []
    for game in ALL_GAMES:
        if game in test_games:
            continue
        trainval_clips.extend((c, game) for c in collect_clips(dataset_root, game))

    train_clips, val_clips = stratified_split(trainval_clips, TRAIN_RATIO, seed)

    test_clips = []
    for game in test_games:
        test_clips.extend((c, game) for c in collect_clips(dataset_root, game))

    all_records = []
    all_records.extend(build_records(train_clips, "train"))
    all_records.extend(build_records(val_clips, "val"))
    all_records.extend(build_records(test_clips, "test"))

    fieldnames = ["game", "clip", "frame_path", "label_path", "frame_name",
                  "visibility", "x", "y", "status", "frame_idx", "is_bounce", "split"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    frames, bounces = {}, {}
    for r in all_records:
        frames[r["split"]] = frames.get(r["split"], 0) + 1
        bounces[r["split"]] = bounces.get(r["split"], 0) + r["is_bounce"]

    print(f"Saved {len(all_records)} records to {output_path}")
    print(f"Held-out test game(s): {', '.join(test_games)}")
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {frames.get(s, 0):6d} frames, {bounces.get(s, 0):4d} bounces")
    print(f"  TOTAL bounces: {sum(bounces.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="../Dataset")
    parser.add_argument("--output", default="splits.csv")
    parser.add_argument("--test_games", default="game7",
                        help="Comma-separated game(s) held out entirely as test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_splits(args.dataset_root, args.output,
                  args.test_games.split(","), args.seed)
