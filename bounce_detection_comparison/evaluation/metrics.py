"""
Event-level bounce-detection metrics (flat-dict contract, mirroring the
court-keypoint project's evaluator). Never report per-frame accuracy — at ~2.6%
positives it is meaningless. Everything is computed on decoded bounce events
matched within a ±k-frame tolerance.
"""

from collections import defaultdict

import numpy as np

from evaluation.decode import decode_clip, match_events

# np.trapz was removed in NumPy 2.x in favour of np.trapezoid.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _count_events(scores_by_clip, gt_by_clip, valid_by_clip,
                  threshold, dist, k, offset):
    tp = fp = fn = 0
    per_game = defaultdict(lambda: [0, 0, 0])
    for key, score in scores_by_clip.items():
        pred = decode_clip(score, threshold, dist, valid_by_clip[key], offset)
        t, f, n, _ = match_events(pred, gt_by_clip[key], k)
        tp += t; fp += f; fn += n
        g = key[0]
        per_game[g][0] += t; per_game[g][1] += f; per_game[g][2] += n
    return tp, fp, fn, per_game


def f1_vs_tolerance(scores_by_clip, gt_by_clip, valid_by_clip, threshold, dist,
                    offset, ks=(0, 1, 2, 3, 5, 7)):
    out = {}
    for kk in ks:
        tp, fp, fn, _ = _count_events(scores_by_clip, gt_by_clip, valid_by_clip,
                                      threshold, dist, kk, offset)
        out[f"F1@k{kk}"] = prf(tp, fp, fn)[2]
    return out


def pr_curve(scores_by_clip, gt_by_clip, valid_by_clip, dist, k, offset, n_thresh=50):
    """Sweep the decode threshold; return (AP, precisions, recalls, thresholds)."""
    thresholds = np.linspace(0.02, 0.98, n_thresh)
    precs, recs = [], []
    for th in thresholds:
        tp, fp, fn, _ = _count_events(scores_by_clip, gt_by_clip, valid_by_clip,
                                      th, dist, k, offset)
        p, r, _ = prf(tp, fp, fn)
        precs.append(p); recs.append(r)
    precs, recs = np.array(precs), np.array(recs)
    order = np.argsort(recs)
    ap = float(_trapz(precs[order], recs[order]))
    return ap, precs.tolist(), recs.tolist(), thresholds.tolist()


def bounce_vs_hit_confusion(scores_by_clip, status_by_clip, valid_by_clip,
                            threshold, dist, k, offset):
    """For each predicted bounce, bucket the nearest GT event within ±k as
    [bounce, hit, none]. Quantifies bounce↔hit confusion."""
    counts = [0, 0, 0]
    for key, score in scores_by_clip.items():
        pred = decode_clip(score, threshold, dist, valid_by_clip[key], offset)
        status = np.asarray(status_by_clip[key])
        T = len(status)
        for p in pred:
            lo, hi = max(0, p - k), min(T, p + k + 1)
            window = status[lo:hi]
            if (window == 2).any():
                counts[0] += 1
            elif (window == 1).any():
                counts[1] += 1
            else:
                counts[2] += 1
    return counts


def compute_all_metrics(scores_by_clip, gt_by_clip, status_by_clip, valid_by_clip,
                        decode_cfg, fps=None):
    """Aggregate everything into a flat JSON-able dict."""
    thr = decode_cfg["threshold"]
    dist = decode_cfg["min_peak_distance"]
    k = decode_cfg["tolerance_k"]
    off = decode_cfg["peak_offset"]

    tp, fp, fn, per_game = _count_events(
        scores_by_clip, gt_by_clip, valid_by_clip, thr, dist, k, off)
    p, r, f1 = prf(tp, fp, fn)
    ap, precs, recs, thr_sweep = pr_curve(
        scores_by_clip, gt_by_clip, valid_by_clip, dist, k, off)
    conf = bounce_vs_hit_confusion(
        scores_by_clip, status_by_clip, valid_by_clip, thr, dist, k, off)

    out = {
        "event_F1@k": f1,
        "event_precision@k": p,
        "event_recall@k": r,
        "AP": ap,
        "TP": tp, "FP": fp, "FN": fn,
        "k": k, "threshold": thr,
        "per_game_F1": {g: prf(*v)[2] for g, v in per_game.items()},
        "confusion_bounce_hit_none": conf,
        "n_bounces_eval": int(tp + fn),
        "FPS": fps,
        # curve data for generate_report.py
        "pr_curve": {"precision": precs, "recall": recs, "threshold": thr_sweep},
    }
    out.update(f1_vs_tolerance(scores_by_clip, gt_by_clip, valid_by_clip, thr, dist, off))
    return out
