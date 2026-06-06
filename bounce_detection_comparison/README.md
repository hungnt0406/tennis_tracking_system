# Bounce Detection Comparison

This folder compares three approaches for detecting tennis ball bounce frames from
TrackNet-style ball trajectories:

- `heuristic`: a calibrated hand-built scorer using trajectory shape cues.
- `gbm`: a per-frame gradient boosting regressor over engineered trajectory features.
- `tcn`: a small dilated 1-D temporal convolutional network over trajectory windows.

The scored task is event timing: detect the frame where `Label.csv` marks
`status == 2`. The ball pixel coordinate at that frame is the image-space bounce
location, but only the frame event is benchmarked here.

Run commands from this directory:

```bash
cd bounce_detection_comparison
```

## Project Structure

```text
bounce_detection_comparison/
  data/
    subset_selector.py      # builds splits.csv from TrackNet Label.csv files
    trajectory.py           # shared trajectory cleaning, kinematics, features, targets
    dataset.py              # GBM tables, TCN windows, per-clip evaluation features
    export_features.py      # exports training_features.csv for inspection
  models/
    heuristic.py            # rule-based per-frame scorer
    gbm.py                  # CatBoost / sklearn HistGradientBoosting scorer
    tcn.py                  # dilated Conv1d temporal model and scorer
  train/
    config.py               # shared paths, feature config, decode config, hyperparameters
    train_heuristic.py      # calibrates heuristic event threshold
    train_gbm.py            # trains GBM and calibrates threshold
    train_tcn.py            # trains TCN and calibrates threshold
  evaluation/
    decode.py               # peak decoding and event matching
    metrics.py              # event metrics, PR curve, tolerance sweep, confusion
    evaluate.py             # evaluates one model arm
  comparison/
    compare_models.py       # evaluates all available arms into one summary JSON
    generate_report.py      # plots and console/LaTeX tables from metric JSONs
  inference/
    make_overlay_video.py   # annotated video overlays for scores/events
    project_to_court.py     # optional qualitative court-meter projection
  checkpoints/              # trained/calibrated model artifacts
  results/                  # metric JSONs, plots, and rendered videos
  splits.csv                # frame-level split manifest
  training_features.csv     # exported valid-frame feature table
  requirements.txt
```

## Data Expectations

The split builder expects a TrackNet-like dataset rooted at `../Dataset` by
default, with game and clip folders such as:

```text
../Dataset/game7/Clip1/
  0000.jpg
  0001.jpg
  ...
  Label.csv
```

`Label.csv` rows must contain the TrackNet columns used by
`data/subset_selector.py`: `file name`, `visibility`, `x-coordinate`,
`y-coordinate`, and `status`.

The generated `splits.csv` is the main contract for the rest of the project. It
contains:

```text
game, clip, frame_path, label_path, frame_name, visibility, x, y,
status, frame_idx, is_bounce, split
```

Current checked artifact counts:

| split | frames in `splits.csv` | bounces |
| --- | ---: | ---: |
| train | 14697 | 387 |
| val | 3257 | 86 |
| test | 1881 | 50 |

The feature pipeline treats `visibility == 0`, blank coordinates, and negative
coordinates as missing ball observations. Short internal gaps up to 4 frames are
linearly interpolated; long gaps remain invalid and are masked out during
training/evaluation.

## Feature and Target Pipeline

All three model arms share `data/trajectory.py`, so the comparison changes the
scorer while keeping preprocessing, targets, decoding, and metrics fixed.

Per clip, the pipeline is:

```text
load_clip_trajectories
  -> clean_trajectory
  -> compute_kinematics
  -> compute_lag_features
  -> compute_frame_features
  -> make_soft_target
```

Important conventions:

- Image `y` grows downward, so a ground bounce is usually near a local maximum of
  `y`.
- `status == 2` is a bounce event.
- `status == 1` is treated as a hit hard negative.
- Targets are Gaussian-in-time soft labels with `sigma_frames = 1.5`.
- Event decoding uses score peaks, not frame classification accuracy.

`training_features.csv` is produced by `data/export_features.py`. It currently
has 78 columns: 7 metadata/label columns plus 71 model feature columns. The
feature columns include normalized position, velocity, acceleration, speed,
acceleration magnitude, heading change, visibility/interpolation flags, and
bidirectional lag/difference/ratio features for lags 1 through 10.

Current checked valid-frame counts in `training_features.csv`:

| split | valid feature rows | bounces |
| --- | ---: | ---: |
| train | 14200 | 387 |
| val | 3179 | 86 |
| test | 1779 | 50 |

## Models

### Heuristic

`models/heuristic.py` computes a per-frame score from three normalized cues:

- vertical local maximum of `y_norm`;
- vertical velocity sign flip within a local window;
- absolute vertical acceleration spike.

The heuristic has no learned model weights. Training only sweeps decode
thresholds on the validation split and writes `checkpoints/heuristic_best.json`.
The current checkpoint stores threshold `0.2`, tolerance `k = 3`, and equal cue
weights.

### GBM

`models/gbm.py` trains a regressor on the full 71-feature per-frame table and the
soft Gaussian bounce target. CatBoost is preferred. If CatBoost is unavailable,
the code falls back to scikit-learn `HistGradientBoostingRegressor`.

Training uses sample weights `1 + pos_weight * y_soft` with `pos_weight = 20.0`,
then calibrates the decode threshold on validation event F1. The checkpoint is a
single pickle envelope at `checkpoints/gbm_best.pkl` containing the backend, the
model, and the calibrated threshold.

### TCN

`models/tcn.py` defines a compact dilated Conv1d network with residual blocks.
It consumes sliding windows over these 7 shared channels:

```text
x_norm, y_norm, vx, vy, ax, ay, visible
```

Training uses masked focal BCE against the soft target, Adam,
`ReduceLROnPlateau`, early stopping, and validation threshold calibration. The
checkpoint at `checkpoints/tcn_best.pt` stores the model config, state dict, and
calibrated threshold.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`catboost` is listed in `requirements.txt`; if it is not installed, GBM training
still runs with the sklearn fallback.

Path defaults are in `train/config.py`:

- `DATASET_ROOT`: defaults to `../Dataset`, overridable by environment variable.
- `SPLITS_CSV`: defaults to `splits.csv`, overridable by environment variable.
- checkpoints: `checkpoints/`.
- results: `results/`.

## Build Splits and Export Features

Build the frame-level split manifest:

```bash
python -m data.subset_selector --dataset_root ../Dataset --output splits.csv
```

By default this uses all games `game1` through `game10`, holds out `game7` as
the test game, and stratifies train/validation clips by bounce density. To choose
different held-out games:

```bash
python -m data.subset_selector \
  --dataset_root ../Dataset \
  --output splits.csv \
  --test_games game7,game8 \
  --seed 42
```

Export the exact valid-frame feature table used by the GBM:

```bash
python -m data.export_features --splits_csv splits.csv --output training_features.csv
```

## Training

Train or calibrate all three arms:

```bash
python -m train.train_heuristic
python -m train.train_gbm
python -m train.train_tcn
```

Useful quick checks:

```bash
python -m train.train_heuristic --max_samples 20
python -m train.train_gbm --max_samples 1000 --epochs 20
python -m train.train_tcn --max_samples 256 --epochs 2
```

Common options:

- `--splits_csv`: path to the split manifest.
- `--checkpoint_dir`: checkpoint output directory.
- `--max_samples`: caps validation clips, GBM rows, or TCN windows depending on
  the trainer.
- `train_tcn.py` also accepts `--batch_size`, `--lr`, `--patience`, and
  `--device`.

## Evaluation and Comparison

Evaluate one arm on a split:

```bash
python -m evaluation.evaluate --model heuristic --checkpoint checkpoints/heuristic_best.json
python -m evaluation.evaluate --model gbm --checkpoint checkpoints/gbm_best.pkl
python -m evaluation.evaluate --model tcn --checkpoint checkpoints/tcn_best.pt
```

If `--checkpoint` is omitted, `evaluation/evaluate.py` uses the default
checkpoint path for that model. The heuristic can run without a checkpoint, in
which case it falls back to config defaults.

Run all available arms and write `results/comparison_summary.json`:

```bash
python -m comparison.compare_models --splits_csv splits.csv --split test --results_dir results
```

Generate plots and console/LaTeX tables from saved metric JSONs:

```bash
python -m comparison.generate_report --results_dir results
```

Metrics are event-level, using greedy one-to-one matching within `k = 3` frames
by default:

- `event_precision@k`, `event_recall@k`, `event_F1@k`;
- `AP` from a threshold sweep;
- `F1@k0`, `F1@k1`, `F1@k2`, `F1@k3`, `F1@k5`, `F1@k7`;
- per-game F1;
- predicted-positive confusion against nearby `bounce`, `hit`, or `none`;
- measured scorer FPS.

Per-frame accuracy is intentionally not reported because bounce frames are very
sparse.

## Current Results Artifacts

The checked `results/comparison_summary.json` reports the following on the
current held-out `game7` test split with 50 bounce events:

| model | threshold | F1@k=3 | precision | recall | AP | TP/FP/FN | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| heuristic | 0.20 | 0.578 | 0.459 | 0.780 | 0.394 | 39/46/11 | 32706.3 |
| gbm | 0.60 | 0.951 | 0.925 | 0.980 | 0.874 | 49/4/1 | 21216.4 |
| tcn | 0.60 | 0.925 | 0.875 | 0.980 | 0.972 | 49/7/1 | 4703.0 |

Generated report files include:

```text
results/heuristic_metrics.json
results/gbm_metrics.json
results/tcn_metrics.json
results/comparison_summary.json
results/pr_curve.png
results/f1_vs_tolerance.png
results/per_game_f1.png
results/bounce_vs_hit_confusion.png
results/event_f1_bar.png
results/fps_bar.png
```

## Inference and Visualization

Render annotated videos using the same scorers, thresholds, and decoder used by
evaluation:

```bash
python -m inference.make_overlay_video
python -m inference.make_overlay_video --clip Clip1 --fps 25
python -m inference.make_overlay_video --arm gbm --out results/videos/game7_gbm.mp4
```

The overlay draws the ball position, ground-truth bounce markers, model score
bars, threshold ticks, and short flashes/rings when a decoded event fires.

The repository also contains example videos:

```text
results/videos/game7_heuristic.mp4
results/videos/game7_gbm.mp4
results/videos/game7_tcn.mp4
```

## Optional Court Projection

`inference/project_to_court.py` maps detected bounce pixels to court meters and
classifies them as in/out against singles or doubles bounds:

```bash
python -m inference.project_to_court \
  --court_keypoints path/to/court_keypoints.json \
  --bounce_pixels path/to/bounce_pixels.json \
  --bounds singles
```

This is qualitative/demo-only. It depends on homography helpers from the sibling
`courtkeypoint_detection_comparison` project and requires a 14x2 array of court
keypoint pixel coordinates for the clip. TrackNet bounce frames do not provide
court-keypoint ground truth, so court-meter locations and in/out labels are not
benchmarked by this comparison.

## Checkpoints

Current checkpoint files:

```text
checkpoints/heuristic_best.json
checkpoints/gbm_best.pkl
checkpoints/tcn_best.pt
```

Checkpoint thresholds are used automatically by evaluation and overlay tools
when present.

## Caveats

- Reported scores use ground-truth TrackNet ball coordinates from `Label.csv`.
  They are an upper bound relative to deployment with a predicted ball tracker.
- Splits and features are generated from local dataset paths embedded in
  `splits.csv`; rebuild the manifest if the dataset moves.
- Features are strictly per clip. Windows, lag features, and event decoding do
  not cross clip boundaries.
- Invalid long-gap frames are masked during event decoding.
- The GBM backend can change depending on whether CatBoost is importable, which
  may change metrics and checkpoint contents.
- `comparison.compare_models` skips missing GBM/TCN checkpoints but always tries
  the heuristic path.
