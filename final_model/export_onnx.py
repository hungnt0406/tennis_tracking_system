"""Export the ball (TrackNetV4) and court (MobileNetV3SmallPose) models to ONNX.

Loads both projects in one process via ``import_project`` (separate prefixes so
the two ``models``/``data`` package trees don't collide), exports each to ONNX,
runs a one-batch parity check against onnxruntime, and writes a metadata.json
describing input/output shapes, normalization, and decoding for downstream use.
"""
import argparse
import json
import os

import numpy as np
import torch

from ._loader import import_project, BALL_ROOT, COURT_ROOT


def _parity_check(model, onnx_path, sample, input_name):
    """Compare PyTorch vs onnxruntime output on the same random input."""
    import onnxruntime

    with torch.no_grad():
        torch_out = model(sample).cpu().numpy()

    sess = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {input_name: sample.cpu().numpy()})[0]

    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"  parity max abs diff: {max_diff:.3e}")
    assert max_diff < 1e-3, f"parity check failed: {max_diff} >= 1e-3"


def export_ball(ckpt_path, out_dir):
    TrackNetV4 = import_project(BALL_ROOT, "models.tracknetv4", "_fm_ball")[
        "models.tracknetv4"
    ].TrackNetV4

    model = TrackNetV4().eval().cpu()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    onnx_path = os.path.join(out_dir, "ball_tracknetv4.onnx")
    torch.onnx.export(
        model,
        torch.zeros(1, 9, 368, 640),
        onnx_path,
        opset_version=16,
        input_names=["frames"],
        output_names=["logits"],
        dynamic_axes={"frames": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"exported ball -> {onnx_path}")
    _parity_check(model, onnx_path, torch.randn(1, 9, 368, 640), "frames")


def export_court(ckpt_path, out_dir):
    MobileNetV3SmallPose = import_project(
        COURT_ROOT, "models.mobilenetv3_pose", "_fm_court"
    )["models.mobilenetv3_pose"].MobileNetV3SmallPose

    model = MobileNetV3SmallPose(num_channels=15, dec_ch=64, pretrained=False).eval().cpu()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    onnx_path = os.path.join(out_dir, "court_mobilenetv3.onnx")
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 360, 640),
        onnx_path,
        opset_version=16,
        input_names=["image"],
        output_names=["heatmaps"],
        dynamic_axes={"image": {0: "batch"}, "heatmaps": {0: "batch"}},
    )
    print(f"exported court -> {onnx_path}")
    _parity_check(model, onnx_path, torch.randn(1, 3, 360, 640), "image")


def _write_metadata(out_dir):
    metadata = {
        "ball": {
            "onnx_file": "ball_tracknetv4.onnx",
            "input": {
                "name": "frames",
                "shape": [1, 9, 368, 640],
                "description": "3 RGB frames stacked channel-wise (9 = 3 frames x 3 channels)",
            },
            "output": {
                "name": "logits",
                "shape": [1, 256, 368, 640],
            },
            "normalization": "/255.0 (NO imagenet)",
            "decode": (
                "argmax over the 256 output channels -> (H,W) intensity in [0,255], "
                "then peak/HoughCircles to (x,y) in 640x368 space; (-1,-1) = no ball"
            ),
        },
        "court": {
            "onnx_file": "court_mobilenetv3.onnx",
            "input": {
                "name": "image",
                "shape": [1, 3, 360, 640],
                "description": "single RGB frame",
            },
            "output": {
                "name": "heatmaps",
                "shape": [1, 15, 360, 640],
            },
            "normalization": (
                "/255.0 then ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]"
            ),
            "decode": (
                "sigmoid -> per-channel argmax (sub-pixel) on channels 0..13 -> "
                "14 keypoints (x,y) in 640x360 space; channel 14 = court center (ignored)"
            ),
        },
    }
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"wrote {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Export ball + court models to ONNX.")
    parser.add_argument("--ball", required=True, help="path to tracknetv4 checkpoint")
    parser.add_argument("--court", required=True, help="path to mobilenetv3 checkpoint")
    parser.add_argument("--out-dir", required=True, help="directory for ONNX outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    export_ball(args.ball, args.out_dir)
    export_court(args.court, args.out_dir)
    _write_metadata(args.out_dir)

    print("bounce model (XGBoost) ships as a pickle — ship as-is or switch to TCN for a fully-ONNX bundle")


if __name__ == "__main__":
    main()
