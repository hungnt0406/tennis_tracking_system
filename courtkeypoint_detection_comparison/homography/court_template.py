"""Canonical tennis court keypoints in court-plane meters."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


COURT_LENGTH_M = 23.77
DOUBLES_WIDTH_M = 10.97
SINGLES_WIDTH_M = 8.23
SERVICE_LINE_FROM_NET_M = 6.40

HALF_COURT_LENGTH_M = COURT_LENGTH_M / 2.0
HALF_DOUBLES_WIDTH_M = DOUBLES_WIDTH_M / 2.0
HALF_SINGLES_WIDTH_M = SINGLES_WIDTH_M / 2.0


# Local annotations place keypoints 4/6 on the top baseline and 5/7 on the
# bottom baseline, so these are singles baseline corners.
COURT_KEYPOINTS_M: NDArray[np.float64] = np.array(
    [
        [-HALF_DOUBLES_WIDTH_M, HALF_COURT_LENGTH_M],  # 0: top-left outer corner
        [HALF_DOUBLES_WIDTH_M, HALF_COURT_LENGTH_M],  # 1: top-right outer corner
        [-HALF_DOUBLES_WIDTH_M, -HALF_COURT_LENGTH_M],  # 2: bottom-left outer corner
        [HALF_DOUBLES_WIDTH_M, -HALF_COURT_LENGTH_M],  # 3: bottom-right outer corner
        [-HALF_SINGLES_WIDTH_M, HALF_COURT_LENGTH_M],  # 4: top-left singles corner
        [-HALF_SINGLES_WIDTH_M, -HALF_COURT_LENGTH_M],  # 5: bottom-left singles corner
        [HALF_SINGLES_WIDTH_M, HALF_COURT_LENGTH_M],  # 6: top-right singles corner
        [HALF_SINGLES_WIDTH_M, -HALF_COURT_LENGTH_M],  # 7: bottom-right singles corner
        [-HALF_SINGLES_WIDTH_M, SERVICE_LINE_FROM_NET_M],  # 8: top-left service point
        [HALF_SINGLES_WIDTH_M, SERVICE_LINE_FROM_NET_M],  # 9: top-right service point
        [-HALF_SINGLES_WIDTH_M, -SERVICE_LINE_FROM_NET_M],  # 10: bottom-left service point
        [HALF_SINGLES_WIDTH_M, -SERVICE_LINE_FROM_NET_M],  # 11: bottom-right service point
        [0.0, SERVICE_LINE_FROM_NET_M],  # 12: top center T-point
        [0.0, -SERVICE_LINE_FROM_NET_M],  # 13: bottom center T-point
    ],
    dtype=np.float64,
)
"""Fourteen canonical court landmarks in TennisCourtDetector keypoint order.

The origin is the court center. Positive ``y`` points toward the top baseline;
positive ``x`` points toward the right sideline.
"""


SINGLES_BOUNDS_M: NDArray[np.float64] = np.array(
    [
        [-HALF_SINGLES_WIDTH_M, HALF_COURT_LENGTH_M],
        [HALF_SINGLES_WIDTH_M, HALF_COURT_LENGTH_M],
        [HALF_SINGLES_WIDTH_M, -HALF_COURT_LENGTH_M],
        [-HALF_SINGLES_WIDTH_M, -HALF_COURT_LENGTH_M],
    ],
    dtype=np.float64,
)
"""Singles court polygon in meters, ordered clockwise from the top-left corner."""


DOUBLES_BOUNDS_M: NDArray[np.float64] = np.array(
    [
        [-HALF_DOUBLES_WIDTH_M, HALF_COURT_LENGTH_M],
        [HALF_DOUBLES_WIDTH_M, HALF_COURT_LENGTH_M],
        [HALF_DOUBLES_WIDTH_M, -HALF_COURT_LENGTH_M],
        [-HALF_DOUBLES_WIDTH_M, -HALF_COURT_LENGTH_M],
    ],
    dtype=np.float64,
)
"""Doubles court polygon in meters, ordered clockwise from the top-left corner."""
