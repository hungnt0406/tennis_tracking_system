"""Homography estimation from detected tennis court keypoints."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .court_template import COURT_KEYPOINTS_M


def estimate_homography(
    pred_kps: NDArray[np.floating],
    min_inliers: int = 6,
    ransac_threshold_px: float = 5.0,
) -> tuple[NDArray[np.float64] | None, NDArray[np.bool_], float]:
    """Estimate an image-to-court homography from 14 image keypoints.

    Parameters
    ----------
    pred_kps:
        Array of shape ``(14, 2)`` with original-image pixel coordinates.
        Rows equal to ``[-1, -1]`` and non-finite rows are treated as missing.
    min_inliers:
        Minimum number of RANSAC inliers required before returning a homography.
    ransac_threshold_px:
        Pixel reprojection threshold passed to OpenCV RANSAC.

    Returns
    -------
    tuple
        ``(H_img_to_court, inlier_mask, mean_reproj_err_px)``. ``H_img_to_court``
        is ``None`` when there are fewer than four visible keypoints, OpenCV
        fails, the transform is singular, or the inlier count is below
        ``min_inliers``. ``inlier_mask`` always has length 14.
    """
    image_kps = np.asarray(pred_kps, dtype=np.float64)
    if image_kps.shape != (14, 2):
        raise ValueError("pred_kps must have shape (14, 2)")

    inlier_mask = np.zeros(14, dtype=bool)
    finite_rows = np.isfinite(image_kps).all(axis=1)
    sentinel_rows = (image_kps == -1.0).all(axis=1)
    visible_mask = finite_rows & ~sentinel_rows

    if int(visible_mask.sum()) < 4:
        return None, inlier_mask, float("inf")

    template_visible = COURT_KEYPOINTS_M[visible_mask]
    image_visible = image_kps[visible_mask]

    H_court_to_img, ransac_mask = cv2.findHomography(
        srcPoints=template_visible,
        dstPoints=image_visible,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_px),
    )
    if H_court_to_img is None or ransac_mask is None:
        return None, inlier_mask, float("inf")

    visible_inliers = ransac_mask.ravel().astype(bool)
    visible_indices = np.flatnonzero(visible_mask)
    inlier_mask[visible_indices] = visible_inliers

    projected = cv2.perspectiveTransform(
        template_visible.reshape(-1, 1, 2),
        H_court_to_img,
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - image_visible, axis=1)
    mean_reproj_err_px = (
        float(errors[visible_inliers].mean()) if visible_inliers.any() else float("inf")
    )

    if int(visible_inliers.sum()) < min_inliers:
        return None, inlier_mask, mean_reproj_err_px

    try:
        H_img_to_court = np.linalg.inv(H_court_to_img).astype(np.float64)
    except np.linalg.LinAlgError:
        return None, inlier_mask, mean_reproj_err_px

    if not np.isfinite(H_img_to_court).all():
        return None, inlier_mask, mean_reproj_err_px

    if abs(H_img_to_court[2, 2]) > 1e-12:
        H_img_to_court /= H_img_to_court[2, 2]

    return H_img_to_court, inlier_mask, mean_reproj_err_px
