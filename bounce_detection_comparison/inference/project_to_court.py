"""Project bounce pixels to real-world court meters and classify in/out.

QUALITATIVE / DEMO ONLY -- NOT part of any scored metric.

The bounce-frame detector yields a bounce frame; the ball's pixel ``(x, y)`` at
that frame is the bounce location in image space. This utility maps that pixel
to court-plane METERS using the sibling court-keypoint project's homography
(``../../courtkeypoint_detection_comparison``), then classifies in/out against
the singles or doubles court polygon.

There is NO court-keypoint ground truth for the TrackNet bounce frames, so the
homography here must be supplied by hand (or from a court-keypoint model run) and
the in/out result is unverified. Use this for visualization/demo only; do not
report its output as a benchmarked number.

The heavy sibling import is lazy/guarded so importing this module never breaks
the rest of the project even when the sibling is not importable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
from matplotlib.path import Path

_SIB = os.path.join(os.path.dirname(__file__), "..", "..", "courtkeypoint_detection_comparison")


def _load_sibling():
    """Import the sibling homography API lazily, raising a clear error on failure."""
    sys.path.insert(0, os.path.abspath(_SIB))
    try:
        from homography.estimate import estimate_homography
        from homography.court_template import (
            COURT_KEYPOINTS_M,
            SINGLES_BOUNDS_M,
            DOUBLES_BOUNDS_M,
        )
    except Exception as exc:  # pragma: no cover - depends on sibling availability
        raise RuntimeError(
            "Could not import the court-keypoint project's homography API from "
            f"{os.path.abspath(_SIB)!r}. This optional demo util needs the sibling "
            "project 'courtkeypoint_detection_comparison' on disk and importable. "
            f"Original error: {exc}"
        ) from exc
    return estimate_homography, COURT_KEYPOINTS_M, SINGLES_BOUNDS_M, DOUBLES_BOUNDS_M


def project_points(pixels_xy, H_img_to_court):
    """Map (N, 2) image pixels to (N, 2) court meters via ``cv2.perspectiveTransform``."""
    pixels = np.asarray(pixels_xy, dtype=np.float64).reshape(-1, 1, 2)
    court = cv2.perspectiveTransform(pixels, np.asarray(H_img_to_court, dtype=np.float64))
    return court.reshape(-1, 2)


def homography_from_keypoints(court_keypoints_px):
    """Estimate ``H_img_to_court`` from 14x2 court keypoint pixels; raise if it fails."""
    estimate_homography, _, _, _ = _load_sibling()
    H = estimate_homography(np.asarray(court_keypoints_px, dtype=np.float64))[0]
    if H is None:
        raise ValueError(
            "estimate_homography returned None (too few visible keypoints, RANSAC "
            "failed, or a singular/low-inlier transform). Check the court keypoints."
        )
    return H


def classify_in_out(court_xy_m, bounds):
    """Point-in-polygon test; True when a court-meter point lies inside ``bounds``."""
    polygon = Path(np.asarray(bounds, dtype=np.float64))
    pts = np.asarray(court_xy_m, dtype=np.float64).reshape(-1, 2)
    return [bool(polygon.contains_point((float(x), float(y)))) for x, y in pts]


def project_bounces(bounce_pixels_xy, court_keypoints_px, bounds="singles"):
    """Project bounce pixels to court meters and classify in/out.

    Returns a list of ``{"pixel": (x, y), "court_m": (X, Y), "in_court": bool}``.
    """
    _, _, singles, doubles = _load_sibling()
    poly = doubles if bounds == "doubles" else singles
    H = homography_from_keypoints(court_keypoints_px)
    pixels = np.asarray(bounce_pixels_xy, dtype=np.float64).reshape(-1, 2)
    court = project_points(pixels, H)
    inside = classify_in_out(court, poly)
    return [
        {
            "pixel": (float(px), float(py)),
            "court_m": (float(cx), float(cy)),
            "in_court": flag,
        }
        for (px, py), (cx, cy), flag in zip(pixels, court, inside)
    ]


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Project bounce pixels to court meters and classify in/out "
        "(QUALITATIVE DEMO -- not a benchmarked metric).",
    )
    parser.add_argument(
        "--court_keypoints",
        help="Path to a JSON file with a 14x2 array of court keypoint pixel coords.",
    )
    parser.add_argument(
        "--bounce_pixels",
        help="Path to a JSON list of [x, y] bounce pixel coords. "
        "If omitted, a few hardcoded demo points are used.",
    )
    parser.add_argument(
        "--bounds",
        choices=("singles", "doubles"),
        default="singles",
        help="Court polygon to test against (default: singles).",
    )
    args = parser.parse_args()

    if not args.court_keypoints:
        print(
            "No --court_keypoints given.\n\n"
            "This is a QUALITATIVE demo only -- it is not part of any scored metric.\n"
            "It needs a 14x2 array of court-keypoint pixel coords (TrackNet keypoint\n"
            "order) for the clip, which the TrackNet bounce dataset does NOT provide.\n"
            "Supply them by hand or from a court-keypoint model run, then pass:\n"
            "  --court_keypoints kps.json [--bounce_pixels bounces.json] [--bounds singles|doubles]\n"
            "where kps.json is a JSON 14x2 array and bounces.json is a JSON list of [x, y]."
        )
        return 0

    with open(args.court_keypoints) as f:
        court_kps = json.load(f)

    if args.bounce_pixels:
        with open(args.bounce_pixels) as f:
            bounce_pixels = json.load(f)
    else:
        bounce_pixels = [[640.0, 360.0], [120.0, 700.0], [1180.0, 80.0]]
        print("No --bounce_pixels given; using hardcoded demo points:", bounce_pixels)

    results = project_bounces(bounce_pixels, court_kps, bounds=args.bounds)
    print(f"\nIn/out tested against {args.bounds} bounds (court meters, origin = center):")
    for r in results:
        px, py = r["pixel"]
        cx, cy = r["court_m"]
        verdict = "IN " if r["in_court"] else "OUT"
        print(f"  pixel ({px:8.1f}, {py:8.1f}) -> court ({cx:7.2f}, {cy:7.2f}) m  [{verdict}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
