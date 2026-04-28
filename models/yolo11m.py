"""
YOLO11m wrapper for tennis ball detection.

Uses Ultralytics YOLO11m pretrained on COCO and fine-tuned as a single-class
detector (tennis ball).  When Ultralytics is not installed we provide a
lightweight fallback CNN for standalone testing.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fallback lightweight detector (used when ultralytics is unavailable)
# ---------------------------------------------------------------------------

class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class _CSPBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.cv1 = _ConvBnRelu(ch, ch // 2, 1, p=0)
        self.cv2 = _ConvBnRelu(ch, ch // 2, 1, p=0)
        self.cv3 = _ConvBnRelu(ch // 2, ch // 2, 3)
        self.merge = _ConvBnRelu(ch, ch, 1, p=0)

    def forward(self, x):
        y1 = self.cv3(self.cv1(x))
        y2 = self.cv2(x)
        return self.merge(torch.cat([y1, y2], 1))


class LightweightDetector(nn.Module):
    """
    Fallback single-class detector when Ultralytics is not installed.
    Input : (B, 3, 640, 640)
    Output: (B, 5) — [conf, cx, cy, bw, bh] (unnormalised logit for conf,
                       normalised coords for box)
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            _ConvBnRelu(3, 32, 3, s=2),    # 320
            _ConvBnRelu(32, 64, 3, s=2),   # 160
            _CSPBlock(64),
            _ConvBnRelu(64, 128, 3, s=2),  # 80
            _CSPBlock(128),
            _ConvBnRelu(128, 256, 3, s=2), # 40
            _CSPBlock(256),
            _ConvBnRelu(256, 512, 3, s=2), # 20
            _CSPBlock(512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 5),   # conf, cx, cy, bw, bh
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat)
        out = self.head(feat)
        conf = out[:, :1]                        # raw logit
        box = torch.sigmoid(out[:, 1:])          # normalised [0,1]
        return torch.cat([conf, box], dim=1)     # (B, 5)


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class YOLO11mDetector(nn.Module):
    """
    Wraps Ultralytics YOLO11m if available, otherwise uses LightweightDetector.

    For training / fine-tuning with Ultralytics:
        Use train_yolo11m.py which calls ultralytics directly.

    For inference via this wrapper:
        model = YOLO11mDetector(weights_path='path/to/best.pt')
        pred  = model.predict_centre(frame_tensor)
    """

    def __init__(self, weights_path: str = None, use_ultralytics: bool = True):
        super().__init__()
        self.use_ultralytics = False
        self._yolo = None

        if use_ultralytics:
            try:
                from ultralytics import YOLO
                if weights_path:
                    self._yolo = YOLO(weights_path)
                else:
                    self._yolo = YOLO("yolo11m.pt")
                self.use_ultralytics = True
            except ImportError:
                pass

        if not self.use_ultralytics:
            self.detector = LightweightDetector()
            if weights_path:
                state = torch.load(weights_path, map_location="cpu")
                self.detector.load_state_dict(state)

    def forward(self, x: torch.Tensor):
        """Only used for the fallback detector."""
        if self.use_ultralytics:
            raise RuntimeError("Use predict_centre() for the Ultralytics model.")
        return self.detector(x)

    @torch.no_grad()
    def predict_centre(self, frame_bgr, conf_threshold: float = 0.3):
        """
        frame_bgr: HWC BGR numpy array
        Returns (cx, cy) in pixel coordinates, or (-1, -1) if no detection.
        """
        if self.use_ultralytics:
            results = self._yolo(frame_bgr, verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return -1.0, -1.0
            best = boxes.conf.argmax()
            if boxes.conf[best] < conf_threshold:
                return -1.0, -1.0
            x1, y1, x2, y2 = boxes.xyxy[best].tolist()
            return (x1 + x2) / 2, (y1 + y2) / 2

        # Fallback path
        import numpy as np, cv2
        img = cv2.resize(frame_bgr, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        out = self.detector(t)[0]
        conf = torch.sigmoid(out[0]).item()
        if conf < conf_threshold:
            return -1.0, -1.0
        h, w = frame_bgr.shape[:2]
        cx = out[1].item() * w
        cy = out[2].item() * h
        return cx, cy
