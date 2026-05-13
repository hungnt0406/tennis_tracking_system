"""Shared hyperparameters for all training scripts."""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_ROOT = os.environ.get("DATASET_ROOT", "../Dataset")
SPLITS_CSV   = os.environ.get("SPLITS_CSV",   "splits.csv")
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"

# ─── Common training settings ─────────────────────────────────────────────────
SEED           = 42
NUM_WORKERS    = 4
PIN_MEMORY     = True

# ─── TrackNet ─────────────────────────────────────────────────────────────────
TRACKNET = dict(
    batch_size    = 8,
    epochs        = 50,
    lr            = 1e-4,
    weight_decay  = 1e-5,
    patience      = 10,          # early stopping
    heatmap_threshold = 0.5,     # for peak → coord conversion
)

# ─── TrackNetV4 ───────────────────────────────────────────────────────────────
TRACKNETV4 = dict(
    batch_size    = 8,
    epochs        = 50,
    lr            = 1e-4,
    weight_decay  = 1e-5,
    patience      = 10,
    heatmap_threshold = 0.5,
)

# ─── TrackNetV5 ───────────────────────────────────────────────────────────────
TRACKNETV5 = dict(
    batch_size    = 8,            # smaller due to 3× encoder + attention memory
    epochs        = 50,
    lr            = 1e-4,
    weight_decay  = 1e-5,
    patience      = 10,
    heatmap_threshold = 0.5,
)

# ─── YOLO11m ──────────────────────────────────────────────────────────────────
YOLO11M = dict(
    epochs        = 50,
    batch_size    = 16,
    img_size      = 640,
    lr            = 1e-4,
    patience      = 10,
    conf_threshold = 0.3,
)
