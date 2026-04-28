"""
Evaluation metrics for tennis ball tracking.
All coordinate inputs are in pixel space unless noted.
"""

import numpy as np


def pixel_distance(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    """
    pred_xy, gt_xy : (N, 2) float arrays — predicted and GT ball centres (px)
    Returns        : (N,) Euclidean distances
    """
    return np.linalg.norm(pred_xy - gt_xy, axis=1)


def accuracy_at_threshold(pred_xy, gt_xy, visible_mask, threshold_px):
    """
    Fraction of *visible* frames where distance ≤ threshold_px.
    pred_xy, gt_xy : (N, 2)
    visible_mask   : (N,) bool — True for frames with a visible ball
    """
    if not visible_mask.any():
        return 0.0
    dist = pixel_distance(pred_xy[visible_mask], gt_xy[visible_mask])
    return float((dist <= threshold_px).mean())


def mean_absolute_error(pred_xy, gt_xy, visible_mask):
    """MAE in pixels for visible frames."""
    if not visible_mask.any():
        return float("nan")
    return float(pixel_distance(pred_xy[visible_mask], gt_xy[visible_mask]).mean())


def detection_precision_recall(pred_visible: np.ndarray, gt_visible: np.ndarray):
    """
    Binary precision / recall for ball visibility prediction.
    pred_visible, gt_visible : (N,) bool
    """
    tp = float((pred_visible & gt_visible).sum())
    fp = float((pred_visible & ~gt_visible).sum())
    fn = float((~pred_visible & gt_visible).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def tracking_consistency(pred_xy: np.ndarray, visible_mask: np.ndarray):
    """
    Mean frame-to-frame displacement of predictions (temporal smoothness).
    Lower = smoother trajectory.
    """
    vis_preds = pred_xy[visible_mask]
    if len(vis_preds) < 2:
        return float("nan")
    diffs = np.linalg.norm(np.diff(vis_preds, axis=0), axis=1)
    return float(diffs.mean())


def per_visibility_class_mae(pred_xy, gt_xy, visibility_class):
    """
    MAE broken down by visibility class 0-3.
    Returns dict {cls: mae}
    """
    result = {}
    for cls in range(4):
        mask = (visibility_class == cls)
        if not mask.any():
            result[cls] = float("nan")
            continue
        if cls == 0:
            # No ball — skip distance computation
            result[cls] = float("nan")
        else:
            result[cls] = float(pixel_distance(pred_xy[mask], gt_xy[mask]).mean())
    return result


def compute_all_metrics(pred_xy: np.ndarray,
                         gt_xy: np.ndarray,
                         pred_visible: np.ndarray,
                         gt_visible: np.ndarray,
                         visibility_class: np.ndarray,
                         fps: float = None):
    """
    Aggregate metrics into a single dict.

    pred_xy         : (N, 2) — predicted centre (-1 means no detection)
    gt_xy           : (N, 2) — ground-truth centre
    pred_visible    : (N,)   bool
    gt_visible      : (N,)   bool
    visibility_class: (N,)   int 0-3
    fps             : optional float
    """
    metrics = {}

    for thr in (5, 10, 20):
        metrics[f"acc@{thr}px"] = accuracy_at_threshold(
            pred_xy, gt_xy, gt_visible, thr)

    metrics["MAE_px"] = mean_absolute_error(pred_xy, gt_xy, gt_visible)
    prec, rec, f1 = detection_precision_recall(pred_visible, gt_visible)
    metrics["precision"] = prec
    metrics["recall"]    = rec
    metrics["F1"]        = f1
    metrics["tracking_consistency"] = tracking_consistency(pred_xy, pred_visible)

    per_cls = per_visibility_class_mae(pred_xy, gt_xy, visibility_class)
    for cls, v in per_cls.items():
        metrics[f"MAE_vis{cls}"] = v

    if fps is not None:
        metrics["FPS"] = fps

    return metrics
