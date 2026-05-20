"""
Evaluation metrics for tennis court keypoint detection.
All coordinate inputs are in pixel space unless noted.
"""

import numpy as np


def keypoint_pck(pred_kps, gt_kps, in_image_mask, threshold_px):
    """
    pred_kps: (N, 14, 2) float, pixel coords in original image space
    gt_kps:   (N, 14, 2) float
    in_image_mask: (N, 14) bool — True where keypoint is visible/labeled
    threshold_px: float
    Returns scalar: fraction of visible kps within threshold.
    """
    dists = np.linalg.norm(pred_kps - gt_kps, axis=-1)  # (N, 14)
    within = (dists <= threshold_px) & in_image_mask
    return within.sum() / in_image_mask.sum() if in_image_mask.sum() > 0 else 0.0


def mean_keypoint_error(pred_kps, gt_kps, in_image_mask):
    """Mean Euclidean pixel error over visible kps."""
    dists = np.linalg.norm(pred_kps - gt_kps, axis=-1)
    if in_image_mask.sum() == 0:
        return float('nan')
    return float(dists[in_image_mask].mean())


def per_keypoint_pck(pred_kps, gt_kps, in_image_mask, threshold_px):
    """Returns (14,) array of per-keypoint PCK."""
    dists = np.linalg.norm(pred_kps - gt_kps, axis=-1)  # (N, 14)
    result = np.zeros(14)
    for k in range(14):
        mask_k = in_image_mask[:, k]
        if mask_k.sum() == 0:
            result[k] = float('nan')
        else:
            result[k] = ((dists[:, k] <= threshold_px) & mask_k).sum() / mask_k.sum()
    return result


def court_center_accuracy(pred_center, gt_center, center_mask, threshold_px):
    """
    pred_center: (N, 2)
    gt_center:   (N, 2)
    center_mask: (N,) bool
    Returns scalar PCK for court center channel.
    """
    if center_mask.sum() == 0:
        return float('nan')
    dists = np.linalg.norm(pred_center - gt_center, axis=-1)
    return float(((dists <= threshold_px) & center_mask).sum() / center_mask.sum())


def compute_all_metrics(pred_kps, gt_kps, in_image_mask,
                        pred_center, gt_center, center_mask,
                        fps, params_M):
    """
    All arrays in original image pixel space.
    pred_kps:  (N, 14, 2)
    gt_kps:    (N, 14, 2)
    in_image_mask: (N, 14) bool
    pred_center: (N, 2), gt_center: (N, 2), center_mask: (N,) bool
    fps: float, params_M: float
    Returns dict.
    """
    return {
        "pck@5px":  keypoint_pck(pred_kps, gt_kps, in_image_mask, 5.0),
        "pck@7px":  keypoint_pck(pred_kps, gt_kps, in_image_mask, 7.0),
        "pck@10px": keypoint_pck(pred_kps, gt_kps, in_image_mask, 10.0),
        "pck@25px": keypoint_pck(pred_kps, gt_kps, in_image_mask, 25.0),
        "mean_kp_error_px": mean_keypoint_error(pred_kps, gt_kps, in_image_mask),
        "per_kp_pck@7px":   per_keypoint_pck(pred_kps, gt_kps, in_image_mask, 7.0).tolist(),
        "court_center_pck@7px": court_center_accuracy(pred_center, gt_center, center_mask, 7.0),
        "params_M": params_M,
        "FPS": fps,
    }
