"""
Evaluation metrics for tennis court keypoint detection.
All coordinate inputs are in pixel space unless noted.
"""

import cv2
import numpy as np

from homography.court_template import COURT_KEYPOINTS_M
from homography.estimate import estimate_homography


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


def _project_points(points_px, h_img_to_court):
    """Project image-space points into court meters with a 3x3 homography."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, h_img_to_court.astype(np.float64))
    return projected.reshape(-1, 2)


def homography_metrics(pred_kps, gt_kps, in_image_mask, homographies=None):
    """
    Compute court-plane reprojection metrics from predicted homographies.

    Homographies map image pixels to canonical court meters and are estimated
    from predicted keypoints. GT keypoints are then projected through each
    predicted homography and compared against the canonical template.
    """
    if len(pred_kps) == 0:
        return {
            "mean_reproj_err_cm": float("nan"),
            "max_reproj_err_cm": float("nan"),
            "homography_success_rate": 0.0,
        }

    if homographies is None:
        homographies = [estimate_homography(kps)[0] for kps in pred_kps]

    frame_errors_cm = []
    point_errors_cm = []
    success_count = 0

    for gt_frame, mask_frame, h_img_to_court in zip(gt_kps, in_image_mask, homographies):
        if h_img_to_court is None:
            continue

        success_count += 1
        valid = mask_frame & np.isfinite(gt_frame).all(axis=1) & (gt_frame[:, 0] >= 0)
        if not valid.any():
            continue

        gt_court = _project_points(gt_frame[valid], h_img_to_court)
        errors_m = np.linalg.norm(gt_court - COURT_KEYPOINTS_M[valid], axis=1)
        errors_cm = errors_m * 100.0
        frame_errors_cm.append(float(errors_cm.mean()))
        point_errors_cm.extend(errors_cm.tolist())

    return {
        "mean_reproj_err_cm": float(np.mean(frame_errors_cm)) if frame_errors_cm else float("nan"),
        "max_reproj_err_cm": float(np.max(point_errors_cm)) if point_errors_cm else float("nan"),
        "homography_success_rate": success_count / len(pred_kps),
    }


def compute_all_metrics(pred_kps, gt_kps, in_image_mask,
                        pred_center, gt_center, center_mask,
                        fps, params_M, homographies=None):
    """
    All arrays in original image pixel space.
    pred_kps:  (N, 14, 2)
    gt_kps:    (N, 14, 2)
    in_image_mask: (N, 14) bool
    pred_center: (N, 2), gt_center: (N, 2), center_mask: (N,) bool
    fps: float, params_M: float
    homographies: optional iterable of image->court 3x3 matrices or None
    Returns dict.
    """
    metrics = {
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
    metrics.update(homography_metrics(pred_kps, gt_kps, in_image_mask, homographies))
    return metrics
