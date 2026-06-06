"""
Shared event decoding + matching. ALL THREE arms route through this module so
the comparison is fair: only the per-frame score differs, never the decoding.
"""

import numpy as np
from scipy.signal import find_peaks


def decode_clip(score, threshold, min_peak_distance, valid=None, peak_offset=0):
    """Per-frame score (T,) in [0,1] → predicted bounce frame indices.

    Peaks above `threshold`, separated by ≥ `min_peak_distance` frames. Invalid
    (long-gap) frames are zeroed first; `peak_offset` applies any systematic
    shift before returning.
    """
    s = np.asarray(score, dtype=float).copy()
    if valid is not None:
        s = np.where(np.asarray(valid, bool), s, 0.0)
    peaks, _ = find_peaks(s, height=threshold, distance=max(1, int(min_peak_distance)))
    peaks = peaks + int(peak_offset)
    peaks = peaks[(peaks >= 0) & (peaks < len(s))]
    return np.sort(peaks)


def match_events(pred_frames, gt_frames, k):
    """Greedy one-to-one matching within ±k frames (closest pairs first).
    Returns (tp, fp, fn, matched_pairs)."""
    pred = np.sort(np.asarray(pred_frames, dtype=int))
    gt = np.sort(np.asarray(gt_frames, dtype=int))

    pairs = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            d = abs(int(p) - int(g))
            if d <= k:
                pairs.append((d, pi, gi))
    pairs.sort()

    used_p, used_g, matched = set(), set(), []
    for d, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched.append((int(pred[pi]), int(gt[gi])))
    tp = len(matched)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn, matched
