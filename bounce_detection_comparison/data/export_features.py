"""
Export the exact per-frame feature table the models train on, to a CSV for
inspection. This is `build_feature_table`'s output verbatim — valid frames only
(long-gap frames dropped), the 71 engineered features per frame, plus the soft
target (`y_soft`) and the hard label (`is_bounce`). The GBM trains on this table
directly; the TCN trains on sliding windows of the TCN_CHANNELS subset of these
same columns.

Columns: split, game, clip, frame_idx, status, is_bounce, y_soft, <71 features…>
"""

import argparse

import pandas as pd

from data.dataset import build_feature_table
from train.config import SPLITS_CSV


def export(splits_csv, output, splits=("train", "val", "test")):
    frames, names = [], None
    for split in splits:
        X, y_soft, y_hard, meta = build_feature_table(splits_csv, split)
        if names is None:
            names = meta["feature_names"]
        if len(X) == 0:
            continue
        df = pd.DataFrame(X, columns=names)
        df.insert(0, "y_soft", y_soft)
        df.insert(0, "is_bounce", y_hard)
        df.insert(0, "status", meta["status"])
        df.insert(0, "frame_idx", meta["frame_idx"])
        df.insert(0, "clip", meta["clip"])
        df.insert(0, "game", meta["game"])
        df.insert(0, "split", split)
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full.to_csv(output, index=False)

    print(f"Saved {len(full)} rows × {full.shape[1]} cols → {output}")
    for split in splits:
        sub = full[full["split"] == split]
        print(f"  {split:5s}: {len(sub):6d} frames, {int(sub['is_bounce'].sum()):4d} bounces")
    print(f"  {len(names)} feature columns: {names[:6]} … {names[-3:]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits_csv", default=SPLITS_CSV)
    p.add_argument("--output", default="training_features.csv")
    args = p.parse_args()
    export(args.splits_csv, args.output)
