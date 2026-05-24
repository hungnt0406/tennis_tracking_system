"""Hough-line post-processing refinement for court keypoint predictions.

Refines a single (x, y) prediction by detecting two crossing court lines in a
small crop around the CNN's estimate, then returns their intersection.
Falls back to the original prediction when refinement is unreliable.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

from data.preprocessing import line_intersection


def refine_keypoint(
    img_bgr: np.ndarray,
    x: float,
    y: float,
    *,
    crop_size: int = 40,
    max_drift: float = 20.0,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 20,
    hough_min_length: int = 10,
    hough_max_gap: int = 5,
) -> tuple[float, float, bool]:
    if x < 0 or y < 0:
        return x, y, False

    H, W = img_bgr.shape[:2]
    half = crop_size // 2
    x0 = int(round(x)) - half
    y0 = int(round(y)) - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size
    x0_c, y0_c = max(0, x0), max(0, y0)
    x1_c, y1_c = min(W, x1), min(H, y1)
    if (x1_c - x0_c) < 10 or (y1_c - y0_c) < 10:
        return x, y, False

    crop = img_bgr[y0_c:y1_c, x0_c:x1_c]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=hough_min_length,
        maxLineGap=hough_max_gap,
    )
    if lines is None or len(lines) < 2:
        return x, y, False

    segs = lines.reshape(-1, 4).astype(np.float32)
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    angles = np.arctan2(dy, dx)
    angles = np.where(angles >= np.pi / 2, angles - np.pi, angles)
    angles = np.where(angles < -np.pi / 2, angles + np.pi, angles)

    km = KMeans(n_clusters=2, n_init=3, random_state=0)
    labels = km.fit_predict(angles.reshape(-1, 1))
    centers = km.cluster_centers_.flatten()

    g0 = np.where(labels == 0)[0]
    g1 = np.where(labels == 1)[0]
    if len(g0) == 0 or len(g1) == 0:
        return x, y, False

    if abs(centers[0] - centers[1]) < np.deg2rad(20):
        return x, y, False

    lengths = np.hypot(dx, dy)
    rep0 = g0[np.argmax(lengths[g0])]
    rep1 = g1[np.argmax(lengths[g1])]
    l0 = segs[rep0]
    l1 = segs[rep1]

    pt = line_intersection(
        ((float(l0[0]), float(l0[1])), (float(l0[2]), float(l0[3]))),
        ((float(l1[0]), float(l1[1])), (float(l1[2]), float(l1[3]))),
    )
    if pt is None:
        return x, y, False

    x_refined = pt[0] + x0_c
    y_refined = pt[1] + y0_c
    if np.hypot(x_refined - x, y_refined - y) > max_drift:
        return x, y, False

    return float(x_refined), float(y_refined), True
