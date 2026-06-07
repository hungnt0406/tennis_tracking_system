"""End-to-end tennis tracking pipeline.

Wires the three component wrappers (ball / court / bounce) into a single
``run(src, output_path)`` call that produces an annotated MP4 plus a summary
dict. Coordinate spaces:

- ball model output is 640x368; court keypoints are 640x360; both are scaled
  up to the source frame size before anything is combined.
- homography ``H`` maps *frame pixels* -> *court meters* (re-estimated in frame
  space so it composes with frame-pixel ball positions).
"""
import os
import re
import glob
import time

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib.path import Path

from ._loader import REPO_ROOT
from .ball import BallTracker
from .court import (
    CourtDetector,
    estimate_homography,
    SINGLES_BOUNDS_M,
    DOUBLES_BOUNDS_M,
)
from .bounce import BounceDetector

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(it, **_kwargs):
        return it

# Source spaces the component wrappers emit coordinates in.
BALL_W, BALL_H = 640, 368
COURT_W, COURT_H = 640, 360

# Minimap geometry (court meters -> minimap pixels).
MINIMAP_W, MINIMAP_H = 200, 300
MINIMAP_MARGIN = 12
COURT_HALF_X_M = 5.485   # doubles half-width
COURT_HALF_Y_M = 11.885  # half court length

# Rendering tunables.
TRAIL_LEN = 10
BOUNCE_BANNER_FRAMES = 15
BOUNCE_RING_FRAMES = 12

_IMG_EXTS = (".jpg", ".jpeg", ".png")


class TennisPipeline:
    def __init__(
        self,
        ball_ckpt: str = None,
        court_ckpt: str = None,
        bounce_ckpt: str = None,
        device: str = "cpu",
        bounds: str = "singles",
        ball_decode: str = "hough",
        fps: float = 25,
    ):
        ball_ckpt = ball_ckpt or os.path.join(
            REPO_ROOT, "final_model/tracknetv4_best.pt")
        court_ckpt = court_ckpt or os.path.join(
            REPO_ROOT, "courtkeypoint_detection_comparison/checkpoints/mobilenetv3_best.pt")
        bounce_ckpt = bounce_ckpt or os.path.join(
            REPO_ROOT, "bounce_detection_comparison/checkpoints/gbm_best.pkl")

        self.bounds = bounds
        self.fps = fps
        self.ball = BallTracker(ball_ckpt, device=device, decode=ball_decode)
        self.court = CourtDetector(court_ckpt, device=device)
        self.bounce = BounceDetector(bounce_ckpt)

    # ------------------------------------------------------------------ #
    # Frame extraction
    # ------------------------------------------------------------------ #
    def _read_frames(self, src):
        """Returns (frames list[RGB HWC uint8], frame_w, frame_h, out_fps)."""
        if os.path.isdir(src):
            paths = [
                p for p in glob.glob(os.path.join(src, "*"))
                if os.path.splitext(p)[1].lower() in _IMG_EXTS
            ]

            def _key(p):
                name = os.path.basename(p)
                digits = re.sub(r"\D", "", name)
                return (0, int(digits)) if digits else (1, name)

            paths.sort(key=_key)
            frames = []
            for p in paths:
                bgr = cv2.imread(p)
                if bgr is None:
                    continue
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            out_fps = self.fps
        else:
            cap = cv2.VideoCapture(src)
            frames = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            out_fps = cap_fps if (cap_fps and not np.isnan(cap_fps)) else self.fps

        if not frames:
            raise ValueError(f"No frames read from source: {src}")
        h, w = frames[0].shape[:2]
        return frames, int(w), int(h), float(out_fps)

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    def run(self, src: str, output_path: str, bounds: str = None) -> dict:
        bounds = bounds if bounds is not None else self.bounds
        polygon = SINGLES_BOUNDS_M if bounds == "singles" else DOUBLES_BOUNDS_M

        frames, frame_w, frame_h, out_fps = self._read_frames(src)
        T = len(frames)

        # 1. Ball ------------------------------------------------------- #
        t0 = time.time()
        traj_640 = self.ball.infer_frames(frames)  # (T, 2) in 640x368 space
        ball_elapsed = time.time() - t0
        fps_ball = T / ball_elapsed if ball_elapsed > 0 else 0.0

        visible = traj_640[:, 0] >= 0
        traj_orig = np.full((T, 2), -1.0, np.float32)
        traj_orig[visible, 0] = traj_640[visible, 0] * (frame_w / BALL_W)
        traj_orig[visible, 1] = traj_640[visible, 1] * (frame_h / BALL_H)

        # 2. Court ------------------------------------------------------ #
        H = None
        kps_frame = None
        n_detect = 0
        t0 = time.time()
        for i in range(min(10, T)):
            n_detect += 1
            kps, H_det = self.court.detect(frames[i])
            if H_det is not None:
                kf = kps.astype(np.float64).copy()
                ok = (kf[:, 0] >= 0) & (kf[:, 1] >= 0)
                kf[ok, 0] *= frame_w / COURT_W
                kf[ok, 1] *= frame_h / COURT_H
                H, _, _ = estimate_homography(kf.astype(np.float32))
                if H is not None:
                    kps_frame = kf
                    break
        court_elapsed = time.time() - t0
        fps_court = n_detect / court_elapsed if court_elapsed > 0 else 0.0

        # 3. Bounce ----------------------------------------------------- #
        bounce_frames = self.bounce.detect(
            traj_orig, visible, orig_w=frame_w, orig_h=frame_h)

        # 4. Project + classify in/out --------------------------------- #
        bounce_labels = {}  # frame_idx -> "in"/"out"/"unknown"
        demo_in_count = None
        demo_out_count = None
        if H is not None:
            demo_in_count = 0
            demo_out_count = 0
            poly_path = Path(np.asarray(polygon, np.float64))
            for idx in bounce_frames:
                if not visible[idx]:
                    bounce_labels[int(idx)] = "unknown"
                    continue
                X, Y = _project_point(H, traj_orig[idx])
                if poly_path.contains_point((X, Y)):
                    bounce_labels[int(idx)] = "in"
                    demo_in_count += 1
                else:
                    bounce_labels[int(idx)] = "out"
                    demo_out_count += 1
        else:
            for idx in bounce_frames:
                bounce_labels[int(idx)] = "unknown"

        # 5. Render ----------------------------------------------------- #
        self._render_video(
            frames, traj_orig, visible, bounce_frames, bounce_labels,
            H, polygon, output_path, frame_w, frame_h, out_fps)

        return {
            "n_bounces": int(len(bounce_frames)),
            "demo_in_count": demo_in_count,
            "demo_out_count": demo_out_count,
            "in_out_quality": "qualitative" if H is not None else "unavailable",
            "fps_ball": float(fps_ball),
            "fps_court": float(fps_court),
        }

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_video(self, frames, traj_orig, visible, bounce_frames,
                      bounce_labels, H, polygon, output_path,
                      frame_w, frame_h, out_fps):
        # Prefer H.264 (avc1) so the MP4 plays in browsers / Gradio; mp4v
        # (MPEG-4 Part 2) is readable by OpenCV but HTML5 <video> can't decode
        # it (shows 0:00 / NaN:NaN). Fall back to mp4v if avc1 is unavailable.
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"avc1"),
            out_fps, (frame_w, frame_h))
        if not writer.isOpened():
            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                out_fps, (frame_w, frame_h))

        color_for = {"in": (0, 200, 0), "out": (0, 0, 230), "unknown": (160, 160, 160)}
        banner_for = {"in": "IN", "out": "OUT", "unknown": "BOUNCE"}
        bounce_set = set(int(b) for b in bounce_frames)

        # Pre-render the static minimap court (lines never move) once.
        minimap_base = None
        traj_mini = None  # (T, 2) minimap px for visible pts, NaN elsewhere
        if H is not None:
            minimap_base = self._build_minimap(polygon)
            traj_mini = np.full((len(frames), 2), np.nan, np.float32)
            for i in range(len(frames)):
                if visible[i]:
                    X, Y = _project_point(H, traj_orig[i])
                    traj_mini[i] = _meters_to_minimap(X, Y)

        for i in tqdm(range(len(frames)), desc="render"):
            bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)

            # Ball trail (older = fainter).
            for k in range(TRAIL_LEN, 0, -1):
                j = i - k
                if j < 0 or not visible[j]:
                    continue
                alpha = 1.0 - k / (TRAIL_LEN + 1)
                pt = (int(traj_orig[j, 0]), int(traj_orig[j, 1]))
                overlay = bgr.copy()
                cv2.circle(overlay, pt, 4, (0, 255, 255), -1)
                cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0, bgr)
            if visible[i]:
                cv2.circle(bgr, (int(traj_orig[i, 0]), int(traj_orig[i, 1])),
                           6, (0, 255, 255), -1)

            # Bounce ring + banner for any recent bounce.
            active_label = None
            for b in bounce_set:
                age = i - b
                if 0 <= age < BOUNCE_RING_FRAMES and visible[b]:
                    label = bounce_labels.get(b, "unknown")
                    radius = 8 + age * 3
                    cv2.circle(bgr, (int(traj_orig[b, 0]), int(traj_orig[b, 1])),
                               radius, color_for[label], 2)
                if 0 <= age < BOUNCE_BANNER_FRAMES:
                    active_label = bounce_labels.get(b, "unknown")
            if active_label is not None:
                cv2.putText(bgr, banner_for[active_label], (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                            color_for[active_label], 3, cv2.LINE_AA)

            # Court minimap PiP.
            if H is not None:
                self._blit_minimap(bgr, minimap_base, traj_mini, bounce_set,
                                   i, frame_w, frame_h)

            writer.write(bgr)

        writer.release()

    def _build_minimap(self, polygon):
        """White panel with doubles + singles + chosen-bounds court lines."""
        panel = np.full((MINIMAP_H, MINIMAP_W, 3), 255, np.uint8)
        for poly, color, thick in (
            (DOUBLES_BOUNDS_M, (120, 120, 120), 1),
            (SINGLES_BOUNDS_M, (120, 120, 120), 1),
            (polygon, (0, 0, 0), 2),
        ):
            pts = np.array([_meters_to_minimap(x, y) for x, y in poly], np.int32)
            cv2.polylines(panel, [pts], True, color, thick, cv2.LINE_AA)
        # Net line across the middle (y = 0 meters).
        net_l = _meters_to_minimap(-COURT_HALF_X_M, 0.0)
        net_r = _meters_to_minimap(COURT_HALF_X_M, 0.0)
        cv2.line(panel, tuple(np.int32(net_l)), tuple(np.int32(net_r)),
                 (120, 120, 120), 1, cv2.LINE_AA)
        return panel

    def _blit_minimap(self, bgr, base, traj_mini, bounce_set, i,
                      frame_w, frame_h):
        panel = base.copy()

        # Ball trajectory polyline (visible points up to current frame).
        pts = []
        for j in range(i + 1):
            p = traj_mini[j]
            if not np.isnan(p[0]):
                pts.append([int(p[0]), int(p[1])])
        if len(pts) >= 2:
            cv2.polylines(panel, [np.array(pts, np.int32)], False,
                          (255, 0, 0), 1, cv2.LINE_AA)

        # Bounce positions seen so far.
        for b in bounce_set:
            if b <= i and not np.isnan(traj_mini[b, 0]):
                cv2.circle(panel, (int(traj_mini[b, 0]), int(traj_mini[b, 1])),
                           3, (0, 0, 230), -1)

        # Current ball position.
        if not np.isnan(traj_mini[i, 0]):
            cv2.circle(panel, (int(traj_mini[i, 0]), int(traj_mini[i, 1])),
                       4, (0, 255, 255), -1)

        # Composite into bottom-right corner.
        pad = 10
        x0 = frame_w - MINIMAP_W - pad
        y0 = frame_h - MINIMAP_H - pad
        if x0 < 0 or y0 < 0:
            return
        cv2.rectangle(bgr, (x0 - 2, y0 - 2),
                      (x0 + MINIMAP_W + 1, y0 + MINIMAP_H + 1), (0, 0, 0), 1)
        bgr[y0:y0 + MINIMAP_H, x0:x0 + MINIMAP_W] = panel


# ---------------------------------------------------------------------- #
# Small math helpers
# ---------------------------------------------------------------------- #
def _project_point(H, xy):
    """Project a single frame-pixel point through H -> (X, Y) court meters."""
    src = np.array([[[float(xy[0]), float(xy[1])]]], np.float32)
    dst = cv2.perspectiveTransform(src, H)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


def _meters_to_minimap(x_m, y_m):
    """Court meters -> minimap pixel (x right, y flipped so top baseline up)."""
    usable_w = MINIMAP_W - 2 * MINIMAP_MARGIN
    usable_h = MINIMAP_H - 2 * MINIMAP_MARGIN
    px = MINIMAP_MARGIN + (x_m + COURT_HALF_X_M) / (2 * COURT_HALF_X_M) * usable_w
    py = MINIMAP_MARGIN + (COURT_HALF_Y_M - y_m) / (2 * COURT_HALF_Y_M) * usable_h
    return np.array([px, py], np.float32)
