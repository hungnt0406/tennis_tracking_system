"""Court keypoint detection + homography wrapper (MobileNetV3-Small pose)."""
import numpy as np
import torch

from ._loader import import_project, COURT_ROOT

_mods = import_project(
    COURT_ROOT,
    ["models.mobilenetv3_pose", "data.preprocessing", "homography.estimate",
     "homography.court_template"],
    "_fm_court",
)
MobileNetV3SmallPose = _mods["models.mobilenetv3_pose"].MobileNetV3SmallPose
resize_image = _mods["data.preprocessing"].resize_image
normalize = _mods["data.preprocessing"].normalize
INPUT_H = _mods["data.preprocessing"].INPUT_H
INPUT_W = _mods["data.preprocessing"].INPUT_W
estimate_homography = _mods["homography.estimate"].estimate_homography
_tmpl = _mods["homography.court_template"]
COURT_KEYPOINTS_M = _tmpl.COURT_KEYPOINTS_M
SINGLES_BOUNDS_M = _tmpl.SINGLES_BOUNDS_M
DOUBLES_BOUNDS_M = _tmpl.DOUBLES_BOUNDS_M


def heatmap_to_coords(heatmap_ch, stride, confidence_threshold=None):
    """heatmap_ch: (H, W) numpy float. Returns (x, y) in input-stride pixel space."""
    H, W = heatmap_ch.shape
    flat_idx = heatmap_ch.argmax()
    fy, fx = np.unravel_index(flat_idx, (H, W))
    peak = float(heatmap_ch[fy, fx])
    if confidence_threshold is not None and peak < confidence_threshold:
        return -1.0, -1.0
    y1, y2 = max(0, fy - 1), min(H, fy + 2)
    x1, x2 = max(0, fx - 1), min(W, fx + 2)
    patch = heatmap_ch[y1:y2, x1:x2]
    total = patch.sum()
    if total > 1e-8:
        ys = np.arange(y1, y2)
        xs = np.arange(x1, x2)
        fy_sub = (patch.sum(axis=1) @ ys) / total
        fx_sub = (patch.sum(axis=0) @ xs) / total
    else:
        fy_sub, fx_sub = float(fy), float(fx)
    return fx_sub * stride, fy_sub * stride


class CourtDetector:
    def __init__(self, checkpoint: str, device: str = "cpu", confidence_threshold: float = 0.3):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = MobileNetV3SmallPose(num_channels=15, dec_ch=64, pretrained=False).to(device)
        ckpt = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def detect(self, frame: np.ndarray):
        """frame: RGB HWC uint8. Returns (kps, H).

        kps: (14, 2) float32 in 640x360 space, (-1, -1) for low-confidence kps.
        H:   (3, 3) float64 image->court-meters homography, or None if estimation failed.
        """
        img = resize_image(frame, INPUT_H, INPUT_W)
        x = normalize(img, imagenet=True)
        tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(tensor))[0].cpu().numpy()  # (15, 360, 640)
        stride = INPUT_W / probs.shape[-1]
        kps = np.full((14, 2), -1.0, np.float32)
        for ch in range(14):
            kx, ky = heatmap_to_coords(probs[ch], stride, self.confidence_threshold)
            kps[ch] = (kx, ky)
        H, _mask, _err = estimate_homography(kps)
        return kps, H
