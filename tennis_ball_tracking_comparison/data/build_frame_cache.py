"""Build a consolidated resized-frame cache for TrackNet-style datasets.

The training dataset normally opens and decodes three image files per sample.
For TrackNetV4 this causes severe GPU starvation on slow shared storage. This
builder reads every unique frame once and stores resized RGB uint8 frames in one
memory-mappable `.npy` file, while preserving original image dimensions for
target generation.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

from data.preprocessing import IMG_H, IMG_W, resize_frame


def _read_rows(splits_csv, selected_splits):
    selected = set(selected_splits)
    rows = []
    with open(splits_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] in selected:
                rows.append(row)
    return rows


def _unique_frame_paths(rows):
    seen = set()
    paths = []
    for row in rows:
        frame_path = row["frame_path"]
        if frame_path not in seen:
            seen.add(frame_path)
            paths.append(frame_path)
    return paths


def _resolve_frame_path(frame_path, splits_csv):
    path = Path(frame_path)
    if path.is_absolute():
        return path
    return (Path(splits_csv).resolve().parent / path).resolve()


def _load_rgb(frame_path, splits_csv):
    resolved = _resolve_frame_path(frame_path, splits_csv)
    img = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read frame: {resolved}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _metadata_matches(meta_path, expected_count, selected_splits):
    if not meta_path.exists():
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    return (
        meta.get("version") == 1
        and meta.get("img_h") == IMG_H
        and meta.get("img_w") == IMG_W
        and meta.get("frame_count") == expected_count
        and meta.get("splits") == list(selected_splits)
        and (meta_path.parent / meta.get("data_file", "frames.npy")).exists()
    )


def build_cache(args):
    cv2.setNumThreads(0)
    rows = _read_rows(args.splits_csv, args.splits)
    frame_paths = _unique_frame_paths(rows)
    if not frame_paths:
        raise RuntimeError(f"No frames found for splits: {', '.join(args.splits)}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "metadata.json"
    data_path = cache_dir / "frames.npy"

    if not args.force and _metadata_matches(meta_path, len(frame_paths), args.splits):
        print(f"Frame cache already exists: {cache_dir}")
        return

    tmp_data_path = cache_dir / "frames.tmp.npy"
    tmp_meta_path = cache_dir / "metadata.tmp.json"
    for path in (tmp_data_path, tmp_meta_path):
        if path.exists():
            path.unlink()

    frames = open_memmap(
        tmp_data_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(frame_paths), IMG_H, IMG_W, 3),
    )

    orig_sizes = []
    for idx, frame_path in enumerate(tqdm(frame_paths, desc="Caching frames")):
        img = _load_rgb(frame_path, args.splits_csv)
        orig_h, orig_w = img.shape[:2]
        frames[idx] = resize_frame(img)
        orig_sizes.append([int(orig_h), int(orig_w)])
        if idx % args.flush_every == 0:
            frames.flush()

    frames.flush()
    del frames

    metadata = {
        "version": 1,
        "img_h": IMG_H,
        "img_w": IMG_W,
        "frame_count": len(frame_paths),
        "data_file": "frames.npy",
        "splits_csv": os.path.abspath(args.splits_csv),
        "splits": list(args.splits),
        "paths": frame_paths,
        "path_to_index": {path: idx for idx, path in enumerate(frame_paths)},
        "orig_sizes": orig_sizes,
    }
    with open(tmp_meta_path, "w") as f:
        json.dump(metadata, f)

    tmp_data_path.replace(data_path)
    tmp_meta_path.replace(meta_path)
    print(f"Built frame cache: {cache_dir}")
    print(f"Frames: {len(frame_paths)} | Shape: {IMG_W}x{IMG_H} RGB uint8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_csv", default="splits.csv")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Dataset splits to include in the cache.",
    )
    parser.add_argument("--flush_every", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    build_cache(parser.parse_args())


if __name__ == "__main__":
    main()
