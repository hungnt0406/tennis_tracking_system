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
    batch_size        = 8,        # smaller due to 3× encoder + attention memory
    epochs            = 60,
    lr                = 5e-5,     # lower than V1; attention modules need gentler LR
    warmup_epochs     = 5,        # linear LR warmup before ReduceLROnPlateau takes over
    weight_decay      = 1e-5,
    patience          = 15,       # attention needs more epochs to converge
    grad_clip         = 1.0,      # clip gradient norm; attention can spike early
    heatmap_threshold = 0.5,
)

# ─── TrackNetV2 ───────────────────────────────────────────────────────────────
TRACKNETV2 = dict(
    batch_size    = 10,
    epochs        = 30,
    lr            = 1e-3,
    weight_decay  = 1e-5,
    patience      = 8,
    heatmap_threshold = 0.5,
    seq_len       = 3,
    sigma         = 2.5,
    img_h         = 288,
    img_w         = 512,
)

# ─── TrackNetV3 tracker ───────────────────────────────────────────────────────
TRACKNETV3_TRACKER = dict(
    batch_size    = 10,
    epochs        = 30,
    lr            = 1e-3,
    weight_decay  = 1e-5,
    patience      = 8,
    heatmap_threshold = 0.5,
    seq_len       = 8,
    sigma         = 2.5,
    img_h         = 288,
    img_w         = 512,
    bg_mode       = 'concat',
)

# ─── TrackNetV3 inpaint ───────────────────────────────────────────────────────
TRACKNETV3_INPAINT = dict(
    batch_size    = 32,
    epochs        = 300,
    lr            = 1e-3,
    weight_decay  = 1e-5,
    patience      = 20,
    seq_len       = 16,
    mask_ratio    = 0.3,
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
