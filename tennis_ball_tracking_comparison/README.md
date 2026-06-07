# Tennis Ball Tracking Comparison

This folder compares several tennis-ball localization and tracking approaches on
the TrackNet-style tennis dataset. It contains dataset split generation,
preprocessing, model definitions, training scripts, per-model evaluation, sample
visualization, and comparison report generation.

The code is written as a local Python package with imports such as `from data...`
and `from models...`. Run the commands below from this directory:

```bash
cd tennis_ball_tracking_comparison
```

## Contents

```text
tennis_ball_tracking_comparison/
|-- data/                  # split loading, datasets, augmentation, preprocessing
|-- models/                # TrackNet, TrackNetV2-V5, YOLO wrapper
|-- train/                 # training and trajectory-data generation scripts
|-- evaluation/            # metrics, single-model evaluation, visualizations
|-- comparison/            # batch evaluation and report plotting
|-- checkpoints/           # model checkpoints used by the scripts
|-- cache/                 # generated median-background and trajectory caches
|-- results/               # generated metrics, plots, and visualizations
|-- requirements.txt
`-- splits.csv             # checked-in train/val/test manifest
```

## Environment

Install dependencies from this folder:

```bash
pip install -r requirements.txt
```

Main dependencies are PyTorch, torchvision, Ultralytics, OpenCV, NumPy, pandas,
matplotlib, scikit-learn, tqdm, and Pillow.

The scripts automatically use CUDA when available unless a script exposes
`--device`. TrackNet and TrackNetV4 also support CUDA memory probing with
`--auto_batch`.

## Data

The expected raw dataset layout is:

```text
Dataset/
|-- game1/
|   `-- Clip*/
|       |-- 0000.jpg
|       |-- ...
|       `-- Label.csv
|-- ...
`-- game10/
```

Each `Label.csv` is expected to contain TrackNet-style columns:

```text
file name, visibility, x-coordinate, y-coordinate, status
```

`visibility == 0` or negative coordinates mean no visible ball. Positive
visibility classes are evaluated separately as visible/hard/occluded classes.
The `status` column is preserved in `splits.csv`; the tracking models here use
ball visibility and coordinates, not bounce status.

Generate a fresh split manifest from the raw dataset with:

```bash
python -m data.subset_selector \
  --dataset_root ../Dataset \
  --output splits.csv \
  --seed 42
```

The split code uses games 1-9 for train/validation and game 10 for test. Within
games 1-9, clips are split per game with a 15% validation ratio. The checked-in
`splits.csv` currently contains:

| Split | Frames | Clips |
| --- | ---: | ---: |
| train | 14812 | 73 |
| val | 2879 | 10 |
| test | 2144 | 12 |

The manifest columns are:

```text
game, clip, frame_path, label_path, frame_name, visibility, x, y, status, split
```

By default, `train/config.py` reads:

```python
DATASET_ROOT = os.environ.get("DATASET_ROOT", "../Dataset")
SPLITS_CSV = os.environ.get("SPLITS_CSV", "splits.csv")
```

Most scripts use `--splits_csv` directly, so the important requirement is that
the frame paths inside the CSV resolve from the working directory.

## Models

| Model | File | Input/target | Notes |
| --- | --- | --- | --- |
| TrackNet | `models/tracknet.py` | 3 RGB frames stacked to 9 channels at 640x368, 256-class intensity map | VGG-style encoder-decoder with skip connections. Trained with cross-entropy on quantized Gaussian class maps. |
| TrackNetV2 | `models/tracknetv2.py` | 3-frame 9-channel input at 512x288, 3 sigmoid heatmaps | MIMO heatmap model trained with focal-style weighted BCE. |
| TrackNetV3 tracker | `models/tracknetv3.py` | 8 frames plus median background at 512x288 | Extends V2 by concatenating a per-clip median background. |
| TrackNetV3 InpaintNet | `models/tracknetv3.py` | `(x, y, mask)` coordinate sequences | 1D U-Net that rectifies missing or low-quality trajectory points after tracker inference. |
| TrackNetV4 | `models/tracknetv4.py` | Same contract as TrackNet | TrackNet V1 backbone with a motion-attention module from inter-frame differences. |
| TrackNetV5 | `models/tracknetv5.py` | 3 frames at 640x368, 1 sigmoid heatmap | Shared per-frame encoder with temporal self-attention at the bottleneck, decoding the middle frame. |
| YOLO11m | `models/yolo11m.py` | Single 640x640 frame, one ball class | Uses Ultralytics YOLO11m when installed; otherwise a local lightweight fallback detector is available. |
| SKeepTrack | `models/skeeptrack.py` | Consecutive-frame pair | Simplified tracking-by-detection model with ResNet-18 features, top-k candidate extraction, and candidate association. See caveats below. |

Preprocessing details:

- TrackNet/TrackNetV4/TrackNetV5 use `IMG_W=640`, `IMG_H=368`.
- TrackNetV2/TrackNetV3 use `IMG_W_V2=512`, `IMG_H_V2=288`.
- V3 median backgrounds are cached as `cache/median/<game>_<clip>.npz`.
- YOLO labels use a single class named `tennis_ball` and a fixed 20-pixel box
  size around the annotated center.

## Training

All commands below are run from `tennis_ball_tracking_comparison/`.

Create train/val/test splits:

```bash
python -m data.subset_selector --dataset_root ../Dataset --output splits.csv
```

Train TrackNet:

```bash
python -m train.train_tracknet
python -m train.train_tracknet --epochs 50 --batch_size 4 --lr 1.0
python -m train.train_tracknet --auto_batch --target_vram_gb 23
python -m train.train_tracknet --resume
```

Outputs:

- `checkpoints/tracknet_best.pt`
- `checkpoints/tracknet_last.pt` when training/resuming

Train TrackNetV2:

```bash
python -m train.train_tracknetv2
python -m train.train_tracknetv2 --epochs 30 --batch_size 10 --mixup
```

Output:

- `checkpoints/tracknetv2_best.pt`

Train TrackNetV3 tracker, then generate trajectory data, then train InpaintNet:

```bash
python -m train.train_tracknetv3

python -m train.generate_trajectory_data \
  --checkpoint checkpoints/tracknetv3_tracker_best.pt

python -m train.train_inpaintnet
```

Outputs:

- `checkpoints/tracknetv3_tracker_best.pt`
- `cache/trajectory_data_train.npz`
- `cache/trajectory_data_val.npz`
- `checkpoints/tracknetv3_inpaint_best.pt`

The trajectory `.npz` files contain:

- `coords`: predicted normalized `(x, y)` with zeros for missing frames
- `gt_coords`: normalized ground-truth `(x, y)`
- `mask`: `1` for missing or far-from-GT predictions, `0` for good predictions

Train TrackNetV4:

```bash
python -m train.train_tracknetv4
python -m train.train_tracknetv4 --auto_batch --target_vram_gb 23
python -m train.train_tracknetv4 --resume
```

Outputs:

- `checkpoints/tracknetv4_best.pt`
- `checkpoints/tracknetv4_last.pt`

Train TrackNetV5:

```bash
python -m train.train_tracknetv5
python -m train.train_tracknetv5 --epochs 60 --batch_size 8 --lr 5e-5
```

Output:

- `checkpoints/tracknetv5_best.pt`

Train YOLO11m with Ultralytics:

```bash
python -m train.train_yolo11m
```

This exports a YOLO-format dataset under `checkpoints/yolo_dataset/` using
symlinked images and generated label files, then runs Ultralytics training with
project `checkpoints/` and run name `yolo11m_tennis`.

Train the local fallback detector instead:

```bash
python -m train.train_yolo11m --use_fallback
```

Fallback output:

- `checkpoints/yolo11m_best.pt`

Existing checkpoint files in this folder:

```text
checkpoints/skeeptrack_best.pt
checkpoints/tracknet_best.pt
checkpoints/tracknetv2_best.pt
checkpoints/tracknetv3_tracker_best.pt
checkpoints/tracknetv3_inpaint_best.pt
checkpoints/tracknetv4_best.pt
checkpoints/tracknetv4_last.pt
checkpoints/tracknetv5_best.pt
checkpoints/yolo11m_best.pt
```

## Evaluation

Evaluate one model on the test split:

```bash
python -m evaluation.evaluate --model tracknet \
  --checkpoint checkpoints/tracknet_best.pt

python -m evaluation.evaluate --model tracknetv2 \
  --checkpoint checkpoints/tracknetv2_best.pt

python -m evaluation.evaluate --model tracknetv3 \
  --checkpoint checkpoints/tracknetv3_inpaint_best.pt

python -m evaluation.evaluate --model tracknetv4 \
  --checkpoint checkpoints/tracknetv4_best.pt

python -m evaluation.evaluate --model tracknetv5 \
  --checkpoint checkpoints/tracknetv5_best.pt

python -m evaluation.evaluate --model yolo11m \
  --checkpoint checkpoints/yolo11m_best.pt
```

Use fallback YOLO evaluation when `yolo11m_best.pt` is from the local fallback
detector:

```bash
python -m evaluation.evaluate --model yolo11m \
  --checkpoint checkpoints/yolo11m_best.pt \
  --use_fallback
```

Each run writes:

```text
results/<model>_metrics.json
```

Metrics include:

- `acc@5px`, `acc@10px`, `acc@20px`
- `MAE_px`
- binary `precision`, `recall`, `F1`
- location-aware `precision@5px`, `recall@5px`, `F1@5px`
- `tracking_consistency`
- per-visibility `MAE_vis0` through `MAE_vis3`
- `FPS`

For V2 and V3, predictions are made in 512x288 space and scaled back to the
original frame resolution before metrics are computed. For TrackNet and V4,
evaluation converts 256-class intensity maps to coordinates with Hough circle
detection.

## Compare Models

Run all available checkpoint evaluations and aggregate metrics:

```bash
python -m comparison.compare_models \
  --tracknet_ckpt checkpoints/tracknet_best.pt \
  --tracknetv2_ckpt checkpoints/tracknetv2_best.pt \
  --tracknetv3_ckpt checkpoints/tracknetv3_inpaint_best.pt \
  --tracknetv4_ckpt checkpoints/tracknetv4_best.pt \
  --tracknetv5_ckpt checkpoints/tracknetv5_best.pt \
  --yolo_ckpt checkpoints/yolo11m_best.pt
```

The comparison script skips models whose checkpoint path is missing. It writes:

```text
results/<model>_metrics.json
results/comparison_summary.json
```

Generate plots and a LaTeX table from saved metric JSON files:

```bash
python -m comparison.generate_report --results_dir results
```

Report outputs:

```text
results/accuracy_curves.png
results/mae_comparison.png
results/f1_comparison.png
results/fps_comparison.png
results/visibility_mae.png
```

## Visualizations

Create prediction overlays and a preview video:

```bash
python -m evaluation.visualize --model tracknet \
  --checkpoint checkpoints/tracknet_best.pt \
  --output_dir results/visualizations \
  --num_samples 200 \
  --fps 25
```

Supported visualization models are `tracknet`, `tracknetv2`, `tracknetv3`,
`tracknetv4`, `tracknetv5`, and `yolo11m`. Output files are sample frame images
with GT in green and prediction in red, plus:

```text
results/visualizations/<model>_preview.mp4
```

## Notes and Caveats

- Run commands from `tennis_ball_tracking_comparison/`; otherwise the absolute
  imports like `from data.dataset import ...` may not resolve unless
  `PYTHONPATH` is adjusted.
- `results/` currently contains no useful metrics JSON files in this checkout;
  generate them with `evaluation.evaluate` or `comparison.compare_models`.
- `cache/median/` contains generated V3 median-background `.npz` files. They can
  be regenerated from the dataset if removed.
- `train/train_skeeptrack.py` imports `SKEEPTRACK` from `train.config`, but the
  current `train/config.py` does not define that configuration block. As written,
  SKeepTrack training needs that config added or CLI defaults changed before it
  will run.
- `evaluation.evaluate` has no `skeeptrack` parser choice even though the file
  docstring lists an SKeepTrack command. SKeepTrack therefore has model/training
  code but is not part of the current automated evaluation/comparison path.
- In `evaluation.visualize`, the YOLO fallback branch imports `YOLO_SIZE` from
  `data.preprocessing`, but the constant is defined as `YOLODataset.YOLO_SIZE`
  in `data.dataset`. Ultralytics visualization is unaffected; fallback
  visualization needs that import fixed.
- Ultralytics YOLO training does not copy its best checkpoint to
  `checkpoints/yolo11m_best.pt`; it writes under the Ultralytics run directory.
  The file `checkpoints/yolo11m_best.pt` is useful for the fallback path or must
  be manually aligned with the trained Ultralytics weight path.
- V3 evaluation requires both `checkpoints/tracknetv3_tracker_best.pt` and the
  InpaintNet checkpoint passed via `--checkpoint`.
- Classic TrackNet and TrackNetV4 save `*_last.pt` as full training-state
  checkpoints for `--resume`; `*_best.pt` files are plain model state dicts.
