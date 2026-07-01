"""Shared configuration for the bounce-detection comparison.

The FEATURE / TARGET / DECODE dicts are the cross-cutting contract: the data
pipeline (`data/trajectory.py`, `data/dataset.py`), every model arm, and the
evaluator all read these so the three approaches are compared on identical
features, targets, and event-decoding.
"""

import os

# ─── Paths ──────────────────────────────────────────────────────────────────
DATASET_ROOT   = os.environ.get("DATASET_ROOT", "../Dataset")
SPLITS_CSV     = os.environ.get("SPLITS_CSV", "splits.csv")
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"
SEED           = 42

# ─── Shared feature pipeline ──────────────────────────────────────────────────
FEATURE = dict(
    n_lags        = 10,      # bidirectional lag/ratio features (yastrebksv recipe)
    savgol_window = 7,       # Savitzky–Golay smoothing window for kinematics
    savgol_poly   = 2,
    max_gap       = 4,       # interpolate invisible runs ≤ this many frames
)

# Channels fed to the temporal NN (subset of the GBM table, by name).
TCN_CHANNELS = ["x_norm", "y_norm", "vx", "vy", "ax", "ay", "visible"]

# ─── Shared target ────────────────────────────────────────────────────────────
TARGET = dict(
    sigma_frames = 1.5,      # std (in frames) of the Gaussian-in-time bounce target
)

# ─── Shared event decoding / matching ─────────────────────────────────────────
DECODE = dict(
    threshold         = 0.5,  # default score threshold (per-model checkpoint overrides)
    min_peak_distance = 5,    # min frames between two predicted bounces
    tolerance_k       = 3,    # ±k-frame window for matching a prediction to a GT bounce
    peak_offset       = 0,    # systematic frame offset applied to predictions before matching
)

# ─── Per-model hyperparameters ────────────────────────────────────────────────
HEURISTIC = dict(
    smooth_window = 5,        # extra smoothing on the score
    w_ymin        = 1.0,      # weight: vertical local-maximum (ball lowest on screen)
    w_vflip       = 1.0,      # weight: vertical-velocity sign flip
    w_accel       = 1.0,      # weight: |vertical acceleration| spike
)

GBM = dict(
    backend             = "histgbm",   # sklearn HistGradientBoostingRegressor
    iterations          = 1000,        # maps to max_iter
    learning_rate       = 0.05,
    depth               = 6,
    early_stopping_rounds = 50,
    pos_weight          = 20.0,        # upweight near-bounce rows (imbalance)
)

# Additional gradient-boosting libraries, compared as separate arms. All share
# GBM's recipe (same iterations / lr / depth / pos_weight) so the comparison
# isolates the library, not the hyperparameters.
XGBOOST = dict(
    iterations          = 1000,
    learning_rate       = 0.05,
    depth               = 6,
    early_stopping_rounds = 50,
    pos_weight          = 20.0,
)

LIGHTGBM = dict(
    iterations          = 1000,
    learning_rate       = 0.05,
    depth               = 6,
    num_leaves          = 31,
    early_stopping_rounds = 50,
    pos_weight          = 20.0,
)

CATBOOST = dict(
    iterations          = 1000,
    learning_rate       = 0.05,
    depth               = 6,
    early_stopping_rounds = 50,
    pos_weight          = 20.0,
)

TCN = dict(
    batch_size   = 16,
    epochs       = 60,
    lr           = 1e-3,
    weight_decay = 1e-5,
    patience     = 10,
    window       = 64,
    stride       = 8,
    hidden       = 32,
    levels       = 4,
    kernel       = 3,
    dropout      = 0.1,
    focal_gamma  = 2.0,       # focal weighting for the masked soft-target loss
)
