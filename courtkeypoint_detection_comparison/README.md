# Court Keypoint Detection Comparison

This folder compares heatmap-based neural models for detecting tennis court
landmarks in single RGB frames. Each model predicts 15 heatmap channels:
14 annotated court keypoints plus an inferred court-center channel. The predicted
keypoints are also used to estimate an image-to-court homography against a
canonical tennis court template in meters.

Run the commands below from this directory so the local absolute imports
(`data.*`, `models.*`, `train.*`, etc.) resolve correctly:

```bash
cd courtkeypoint_detection_comparison
pip install -r requirements.txt
```

## Directory Structure

```text
courtkeypoint_detection_comparison/
|-- checkpoints/              # Best model checkpoints, one .pt per model
|-- comparison/               # Multi-model evaluation and report plotting
|-- data/                     # JSON annotations, split loader, dataset, images
|-- evaluation/               # Metrics, inference evaluation, visualization
|-- homography/               # Canonical court geometry and RANSAC homography
|-- models/                   # TrackNet, ResNet50, HRNet, MobileNetV3 models
|-- train/                    # Training loops, shared config, heatmap loss
|-- results/                  # Saved metrics JSONs, plots, visual diagnostics
`-- requirements.txt
```

## Data

The dataset loader expects a data directory with:

```text
data/
|-- data_train.json
|-- data_val.json
`-- images/
    `-- {record_id}.png
```

Each JSON record contains:

```json
{
  "id": "PuXlxKdUIes_2450",
  "metric": 0.28256459215943674,
  "kps": [[378, 186], [905, 184], "... 14 total [x, y] points ..."]
}
```

The bundled annotation counts are:

| Source | Count | Use |
| --- | ---: | --- |
| `data/data_train.json` | 6630 | train |
| `data/data_val.json` | 2211 | split into validation/test |

`data.splits.load_records()` uses `data_train.json` as-is for training. It
sorts `data_val.json` deterministically by MD5 hash of record id, uses the first
1105 records for validation, and the remaining records for test. Six known bad
test annotations are excluded, leaving 1100 test records:

- `zKIU4fWsRTM_1500`
- `PuAPCalPLM4_1700`
- `-5zNAhwRoPE_200`
- `UJHVcyTNo-k_2150`
- `1ueaSm-2-lo_1650`
- `oTsZKnpPiRw_800`

Images are resized from the original frame to `360x640` before model input. The
current evaluation code assumes original frames are `1280x720` when scaling
predictions back to original pixel space.

## Keypoint Order

The 14 annotated channels use the TennisCourtDetector ordering used throughout
`data/preprocessing.py`, `comparison/generate_report.py`, and
`homography/court_template.py`:

| Index | Meaning |
| ---: | --- |
| 0 | top-left outer doubles corner |
| 1 | top-right outer doubles corner |
| 2 | bottom-left outer doubles corner |
| 3 | bottom-right outer doubles corner |
| 4 | top-left singles baseline corner |
| 5 | bottom-left singles baseline corner |
| 6 | top-right singles baseline corner |
| 7 | bottom-right singles baseline corner |
| 8 | top-left service point |
| 9 | top-right service point |
| 10 | bottom-left service point |
| 11 | bottom-right service point |
| 12 | top center T-point |
| 13 | bottom center T-point |

Channel 14 is not annotated directly. It is generated as the intersection of the
two outer-court diagonals when all four outer corners are visible.

## Preprocessing And Targets

`CourtKeypointDataset` returns:

- image tensor: `[3, 360, 640]`, float32
- heatmap tensor: `[15, out_h, out_w]`, float32
- original-space keypoints: `[14, 2]`, float32, with `[-1, -1]` for invisible or invalid points

Training augmentation is intentionally simple:

- random brightness shift
- random contrast change
- random horizontal flip with keypoint remapping

Target heatmaps use CenterNet-style Gaussian blobs. Model output logits are
passed through `torch.sigmoid()` before loss/evaluation.

## Models

Shared training defaults live in `train/config.py`:

| Setting | Value |
| --- | ---: |
| input size | `360x640` |
| heatmap channels | 15 |
| keypoints | 14 |
| batch size | 8 |
| epochs | 100 |
| learning rate | `1e-4` |
| weight decay | `1e-5` |
| early-stopping patience | 30 epochs |
| validation metric | `PCK@7px` in resized input space |

Model-specific settings:

| Model id | Class | Backbone/decoder | Output stride | Gaussian radius | ImageNet norm |
| --- | --- | --- | ---: | ---: | --- |
| `tracknet_court` | `TrackNetCourt` | VGG-style encoder-decoder adapted from TrackNet | 1 | 15 | no |
| `resnet50` | `ResNet50Pose` | torchvision ResNet-50 + 3 deconv blocks | 4 | 8 | yes |
| `hrnet` | `HRNetPose` | timm `hrnet_w32` features + 1x1 head | 4 | 8 | yes |
| `mobilenetv3` | `MobileNetV3SmallPose` | torchvision MobileNetV3-Small + U-Net decoder | 1 | 15 | yes |

The ResNet50, HRNet, and MobileNetV3 model constructors request pretrained
weights by default. On a fresh machine, the first run may need network access to
download those weights unless they are already cached.

## Training

Each trainer has the same CLI pattern and saves the best validation checkpoint
under `checkpoints/`.

```bash
python -m train.train_tracknet_court
python -m train.train_resnet50
python -m train.train_hrnet
python -m train.train_mobilenetv3
```

Useful options:

```bash
python -m train.train_mobilenetv3 \
  --epochs 20 \
  --batch_size 4 \
  --device cuda \
  --max_samples 200
```

Resume training from an existing checkpoint:

```bash
python -m train.train_hrnet \
  --resume checkpoints/hrnet_best.pt \
  --device cuda
```

Use a different dataset directory for training:

```bash
python -m train.train_resnet50 \
  --data_dir /path/to/data_dir
```

The override directory must contain `data_train.json`, `data_val.json`, and an
`images/` subdirectory.

## Checkpoints

Current checkpoint files:

| File | Size | Stored epoch | Best validation PCK@7px |
| --- | ---: | ---: | ---: |
| `checkpoints/tracknet_court_best.pt` | 80 MB | 27 | 0.9968 |
| `checkpoints/resnet50_best.pt` | 130 MB | 75 | 0.9920 |
| `checkpoints/hrnet_best.pt` | 119 MB | 21 | 0.9960 |
| `checkpoints/mobilenetv3_best.pt` | 5.0 MB | 52 | 0.9968 |

Each checkpoint is a PyTorch dictionary with `model_state`, `epoch`, and
`best_pck`.

## Evaluation

Evaluate one checkpoint on a split:

```bash
python -m evaluation.evaluate \
  --model mobilenetv3 \
  --checkpoint checkpoints/mobilenetv3_best.pt \
  --split test \
  --device cuda
```

Supported model ids are:

- `tracknet_court`
- `resnet50`
- `hrnet`
- `mobilenetv3`

Useful evaluation options:

```bash
python -m evaluation.evaluate \
  --model hrnet \
  --checkpoint checkpoints/hrnet_best.pt \
  --split test \
  --batch_size 8 \
  --confidence_threshold 0.3
```

Disable confidence thresholding:

```bash
python -m evaluation.evaluate \
  --model tracknet_court \
  --checkpoint checkpoints/tracknet_court_best.pt \
  --confidence_threshold none
```

Enable Hough-line local refinement:

```bash
python -m evaluation.evaluate \
  --model mobilenetv3 \
  --checkpoint checkpoints/mobilenetv3_best.pt \
  --refine \
  --refine_crop_size 30 \
  --refine_max_drift 5
```

Evaluation writes:

```text
results/{model}_metrics.json
results/{model}_metrics_refined.json   # when --refine is used
```

Saved metrics include:

- `pck@5px`, `pck@7px`, `pck@10px`, `pck@25px`
- `mean_kp_error_px`
- `per_kp_pck@7px`
- `court_center_pck@7px`
- `params_M`
- `FPS`
- `mean_reproj_err_cm`
- `max_reproj_err_cm`
- `homography_success_rate`

The current evaluator scales predictions back to original pixel space using a
hard-coded `1280x720` original resolution. The dataset images bundled here
match that assumption.

## Current Recorded Results

The current `results/` directory contains JSON metrics for HRNet and
MobileNetV3, plus a refined MobileNetV3 run. TrackNet-Court and ResNet50
checkpoints exist, but their metric JSONs are not currently present.

| Result file | PCK@7px | Mean KP error px | Mean reproj error cm | Homography success | Params M | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `results/hrnet_metrics.json` | 0.7385 | 8.9530 | 30.0489 | 1.0000 | 30.8838 | 2.1092 |
| `results/mobilenetv3_metrics.json` | 0.9891 | 2.6604 | 9.0609 | 0.9991 | 1.2828 | 25.1977 |
| `results/mobilenetv3_metrics_refined.json` | 0.9262 | 3.2147 | 8.9437 | 1.0000 | 1.2828 | 2.3669 |

`court_center_pck@7px` is `null` in these JSON files because the current
evaluation loop does not build a ground-truth center target for that metric.

## Model Comparison

Run all models whose checkpoint paths exist:

```bash
python -m comparison.compare_models
```

Override checkpoints or run a smaller smoke comparison:

```bash
python -m comparison.compare_models \
  --tracknet_court_ckpt checkpoints/tracknet_court_best.pt \
  --resnet50_ckpt checkpoints/resnet50_best.pt \
  --hrnet_ckpt checkpoints/hrnet_best.pt \
  --mobilenetv3_ckpt checkpoints/mobilenetv3_best.pt \
  --split test \
  --max_samples 100 \
  --confidence_threshold 0.3
```

This script launches `evaluation.evaluate` in subprocesses so models do not
share GPU memory. It writes per-model metric JSON files plus:

```text
results/comparison_summary.json
```

Generate plots and console/LaTeX tables from saved metrics:

```bash
python -m comparison.generate_report --results_dir results
```

Generated report images include:

- `results/pck_curves.png`
- `results/pck7_comparison.png`
- `results/mean_kp_error.png`
- `results/mean_reproj_error_cm.png`
- `results/homography_success_rate.png`
- `results/court_center_accuracy.png`
- `results/fps_comparison.png`
- `results/per_keypoint_pck.png`
- `results/params_vs_pck.png`
- `results/radar_comparison.png`

Note: `comparison/generate_report.py` currently loads only
`tracknet_court`, `resnet50`, and `hrnet` metric files. It does not include
`mobilenetv3` in its report plots unless the script is extended.

## Visualization

Overlay predictions and ground truth on random samples:

```bash
python -m evaluation.visualize \
  --model mobilenetv3 \
  --checkpoint checkpoints/mobilenetv3_best.pt \
  --split test \
  --num_samples 5 \
  --output_dir results/visualizations
```

Colors:

- green: ground-truth keypoints
- red: predicted keypoints
- yellow cross: predicted court center

Find frames whose predicted homography produces large court-plane reprojection
error:

```bash
python -m evaluation.visualize_high_error \
  --model mobilenetv3 \
  --checkpoint checkpoints/mobilenetv3_best.pt \
  --threshold_cm 100 \
  --output_dir results/high_error_frames
```

The existing `results/gt_audit/` directory contains ground-truth audit overlays,
neighbor contact sheets, and `gt_audit_report.json` for manually inspected
high-error or suspicious annotations.

## Homography Utilities

`homography/court_template.py` defines a canonical tennis court in meters:

- length: `23.77 m`
- doubles width: `10.97 m`
- singles width: `8.23 m`
- service line distance from net: `6.40 m`

`homography/estimate.py` estimates an image-to-court homography from predicted
keypoints:

```python
from homography.estimate import estimate_homography

H_img_to_court, inlier_mask, mean_reproj_err_px = estimate_homography(pred_kps)
```

Inputs must be an array of shape `(14, 2)` in original-image pixel coordinates.
Rows equal to `[-1, -1]` or non-finite rows are treated as missing. The function
uses OpenCV RANSAC, requires at least four visible points to attempt estimation,
and returns `None` unless at least six inliers remain by default.

The evaluation metric `mean_reproj_err_cm` estimates homographies from predicted
keypoints, projects visible ground-truth points into court meters through that
predicted homography, and compares them to the canonical template.

## Notes And Caveats

- Run module commands from `courtkeypoint_detection_comparison/`, not the repo
  root, unless you set `PYTHONPATH` yourself.
- Evaluation and visualization currently use the bundled `data/` directory;
  unlike the training scripts, they do not expose a `--data_dir` CLI argument.
- Original image size is assumed to be `1280x720` in evaluation. The
  visualization script reads actual image dimensions, but `evaluation.evaluate`
  uses the hard-coded assumption.
- `court_center_pck@7px` is present in the metrics schema but currently null
  because ground-truth center computation is disabled in `evaluation.evaluate`.
- Hough refinement can improve homography reprojection error while reducing PCK
  and speed, as shown by the existing MobileNetV3 refined metrics.
- `requirements.txt` includes broad dependencies used across experiments. The
  court keypoint code mainly relies on PyTorch, torchvision, timm, OpenCV,
  NumPy, matplotlib, scikit-learn, tqdm, and Pillow.
