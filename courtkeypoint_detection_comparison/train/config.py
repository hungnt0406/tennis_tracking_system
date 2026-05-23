"""Shared hyperparameters for all court keypoint training scripts."""

SHARED = dict(
    input_h=360,
    input_w=640,
    num_kps=14,
    num_channels=15,      # 14 keypoints + court center
    batch_size=8,
    epochs=100,
    lr=1e-4,
    weight_decay=1e-5,
    patience=30,          # early stopping
    pck_threshold_px=7,   # primary monitoring metric
)

TRACKNET_COURT = {**SHARED, "stride": 1, "gaussian_radius": 15, "use_imagenet_norm": False}
RESNET50       = {**SHARED, "stride": 4, "gaussian_radius": 8,  "use_imagenet_norm": True}
HRNET          = {**SHARED, "stride": 4, "gaussian_radius": 8,  "use_imagenet_norm": True}
MOBILENETV3    = {**SHARED, "stride": 1, "gaussian_radius": 15, "use_imagenet_norm": True}
