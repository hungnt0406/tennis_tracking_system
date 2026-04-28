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

# ─── S-KeepTrack ──────────────────────────────────────────────────────────────
SKEEPTRACK = dict(
    batch_size    = 8,
    epochs        = 50,
    lr            = 1e-4,
    weight_decay  = 1e-5,
    patience      = 10,
    k_candidates  = 8,           # top-k candidates per frame
    cls_weight    = 1.0,         # weight for classification loss
    assoc_weight  = 1.0,         # weight for association loss
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
