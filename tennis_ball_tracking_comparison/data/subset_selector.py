"""
Create train/val/test splits: games 1-9 for train/val, game 10 held out for test.
Outputs a CSV with columns: game, clip, frame_path, label_path, split
"""

import os
import csv
import random
import argparse
from pathlib import Path


GAMES = {
    "train_val": [f"game{i}" for i in range(1, 10)],  # games 1-9
    "test_only": ["game10"],
}
VAL_RATIO = 0.15
# train = 1 - VAL_RATIO of each train_val game's clips; all clips for test_only games


def collect_clips(dataset_root: Path, game: str):
    game_dir = dataset_root / game
    clips = sorted(
        [d for d in game_dir.iterdir() if d.is_dir()],
        key=lambda p: int(p.name.replace("Clip", "")),
    )
    return clips


def split_train_val(clips, val_ratio, seed=42):
    rng = random.Random(seed)
    shuffled = clips[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


def build_records(clips, split, game):
    records = []
    for clip_dir in clips:
        label_file = clip_dir / "Label.csv"
        if not label_file.exists():
            continue
        with open(label_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["file name"]
                frame_path = clip_dir / fname
                if frame_path.exists():
                    vis = int(row["visibility"])
                    x_str = row["x-coordinate"].strip()
                    y_str = row["y-coordinate"].strip()
                    x = float(x_str) if x_str else -1.0
                    y = float(y_str) if y_str else -1.0
                    records.append(
                        {
                            "game": game,
                            "clip": clip_dir.name,
                            "frame_path": str(frame_path),
                            "label_path": str(label_file),
                            "frame_name": fname,
                            "visibility": vis,
                            "x": x,
                            "y": y,
                            "status": int(row["status"]) if row["status"].strip() else 0,
                            "split": split,
                        }
                    )
    return records


def create_splits(dataset_root: str, output_path: str, seed: int = 42):
    dataset_root = Path(dataset_root)
    all_records = []

    for game in GAMES["train_val"]:
        clips = collect_clips(dataset_root, game)
        train_clips, val_clips = split_train_val(clips, VAL_RATIO, seed)
        all_records.extend(build_records(train_clips, "train", game))
        all_records.extend(build_records(val_clips, "val", game))

    for game in GAMES["test_only"]:
        clips = collect_clips(dataset_root, game)
        all_records.extend(build_records(clips, "test", game))

    fieldnames = ["game", "clip", "frame_path", "label_path", "frame_name",
                  "visibility", "x", "y", "status", "split"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    splits = {}
    for r in all_records:
        splits.setdefault(r["split"], 0)
        splits[r["split"]] += 1

    print(f"Saved {len(all_records)} records to {output_path}")
    for s, n in sorted(splits.items()):
        print(f"  {s}: {n} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="Dataset")
    parser.add_argument("--output", default="splits.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_splits(args.dataset_root, args.output, args.seed)
