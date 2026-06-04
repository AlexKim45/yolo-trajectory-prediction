#!/usr/bin/env python3
"""YOLO detection with multi-object tracking, monocular distance, and trajectory prediction.
Designed for batting cage use: tracks ball + batter, draws trail and predicted flight path.

Usage:
  python scripts/batting_cage_detect.py --video path/to/video.mp4
  python scripts/batting_cage_detect.py --cam 0
  python scripts/batting_cage_detect.py --video v.mp4 --out result.mp4 --headless
"""
from __future__ import annotations

import argparse
import collections
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from depth_bev import PredDetection, infer_yolo_detections

# ── Classes of interest ──────────────────────────────────────────────────────
BATTING_CLASSES = {0, 32, 34, 35}  # person, sports ball, baseball bat, glove
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0:  (60,  60,  255),   # person   — red
    32: (0,   220, 255),   # ball     — yellow
    34: (50,  200,  50),   # bat      — green
    35: (255, 140,  50),   # glove    — orange
}

# Real-world sizes used for monocular distance estimation (meters)
# Ball uses diameter; person uses height; bat uses length; glove uses width.
REAL_SIZE_M: dict[int, float] = {
    0:  1.70,   # person height
    32: 0.073,  # baseball diameter
    34: 0.86,   # bat length
    35: 0.28,   # glove width
}
# Which bbox dimension to use as the apparent size denominator
USE_HEIGHT_FOR = {0, 34}  # person, bat: measure bbox height

# ── Tracking config ──────────────────────────────────────────────────────────
TRAIL_LEN    = 40   # max past positions kept per track
PRED_FRAMES  = 50   # frames to extrapolate forward
MIN_FIT_PTS  = 14   # minimum history points before attempting a trajectory fit
MATCH_DIST   = 90   # px — max center distance to match detection to track
MAX_MISSED   = 10   # frames a track can go unmatched before deletion

# Trajectory confidence gates — both must pass before the prediction is drawn.
# RMSE of the parabolic fit must be below this fraction of total displacement.
FIT_RMSE_RATIO  = 0.20   # fit residuals ≤ 20 % of total displacement
MIN_MOTION_PX   = 40     # object must have moved at least this many pixels


class Track:
    _next_id = 0

    def __init__(self, det: PredDetection) -> None:
        self.id = Track._next_id
        Track._next_id += 1
        self.cls_id  = det.cls_id
        self.name    = det.name
        self.history: collections.deque[tuple[float, float]] = collections.deque(maxlen=TRAIL_LEN)
        self.dists:   collections.deque[float]               = collections.deque(maxlen=TRAIL_LEN)
        self.missed  = 0
        self.last    = det
        self._absorb(det)

    def _absorb(self, det: PredDetection) -> None:
        cx = 0.5 * (det.x1 + det.x2)
        cy = 0.5 * (det.y1 + det.y2)
        self.history.append((cx, cy))
        if det.depth_m > 0:
            self.dists.append(det.depth_m)
        self.last   = det
        self.missed = 0

    def update(self, det: PredDetection) -> None:
        self._absorb(det)

    def mark_missed(self) -> None:
        self.missed += 1

    @property
    def center(self) -> tuple[float, float]:
        return self.history[-1]

    @property
    def dist_m(self) -> float:
        if not self.dists:
            return 0.0
        return float(np.median(list(self.dists)[-6:]))

    def predict_trajectory(self, n_frames: int) -> list[tuple[int, int]]:
        """Parabolic fit through track history, gated on fit quality.

        Returns [] if there isn't enough history or the fit residuals are too
        large relative to how far the object has actually moved — this prevents
        wild, jittery predictions on early/noisy tracks.
        """
        pts = list(self.history)
        n   = len(pts)
        if n < MIN_FIT_PTS:
            return []

        xs = np.array([p[0] for p in pts], dtype=np.float64)
        ys = np.array([p[1] for p in pts], dtype=np.float64)
        t  = np.arange(n, dtype=np.float64)

        # Object must have moved enough to establish a direction.
        displacement = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
        if displacement < MIN_MOTION_PX:
            return []

        # Fit degree-2 polynomial (parabola captures gravity).
        px = np.polyfit(t, xs, 2)
        py = np.polyfit(t, ys, 2)

        # Reject if the fit explains the data poorly.
        rmse = math.sqrt(
            float(np.mean((xs - np.polyval(px, t)) ** 2 +
                          (ys - np.polyval(py, t)) ** 2))
        )
        if rmse > FIT_RMSE_RATIO * displacement:
            return []

        tf = np.arange(n, n + n_frames, dtype=np.float64)
        fx = np.polyval(px, tf)
        fy = np.polyval(py, tf)
        return [(int(x), int(y)) for x, y in zip(fx, fy)]


class Tracker:
    def __init__(self) -> None:
        self.tracks: list[Track] = []

    def update(self, dets: list[PredDetection]) -> list[Track]:
        unmatched = list(range(len(dets)))
        matched   = set()

        for trk in self.tracks:
            if not unmatched:
                break
            tx, ty   = trk.center
            best_i   = None
            best_d   = MATCH_DIST
            for i in unmatched:
                d  = dets[i]
                if d.cls_id != trk.cls_id:
                    continue
                cx = 0.5 * (d.x1 + d.x2)
                cy = 0.5 * (d.y1 + d.y2)
                dist = math.hypot(cx - tx, cy - ty)
                if dist < best_d:
                    best_d, best_i = dist, i
            if best_i is not None:
                trk.update(dets[best_i])
                matched.add(id(trk))
                unmatched.remove(best_i)

        for trk in self.tracks:
            if id(trk) not in matched:
                trk.mark_missed()

        for i in unmatched:
            self.tracks.append(Track(dets[i]))

        self.tracks = [t for t in self.tracks if t.missed <= MAX_MISSED]
        return self.tracks


def estimate_distance(det: PredDetection, frame_w: int, fov_deg: float) -> float:
    """Monocular distance from apparent bbox size vs known real-world size."""
    real = REAL_SIZE_M.get(det.cls_id)
    if real is None:
        return 0.0
    f_px = frame_w / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    if det.cls_id in USE_HEIGHT_FOR:
        apparent = max(1, det.y2 - det.y1)
    else:
        apparent = max(1, det.x2 - det.x1)
    return (f_px * real) / apparent


def _dist_color(dist_m: float) -> tuple[int, int, int]:
    """BGR color: green (far) → yellow → red (close/incoming)."""
    if dist_m <= 0:
        return (180, 180, 180)
    t = max(0.0, min(1.0, 1.0 - (dist_m - 1.0) / 8.0))  # 1 m → red, 9 m → green
    r = int(t * 255)
    g = int((1 - t) * 255)
    return (0, g, r)


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    out = frame.copy()
    h_frame, w_frame = out.shape[:2]

    for trk in tracks:
        if trk.missed > 0:
            continue
        d     = trk.last
        color = CLASS_COLORS.get(d.cls_id, (180, 180, 180))
        dcol  = _dist_color(trk.dist_m)

        # ── Bounding box ─────────────────────────────────────────────────────
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, 2)

        # ── Label ────────────────────────────────────────────────────────────
        dist_txt = f"{trk.dist_m:.1f}m" if trk.dist_m > 0 else "?"
        label    = f"#{trk.id} {d.name} {d.conf:.2f}  {dist_txt}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(0, d.y1 - th - 6)
        cv2.rectangle(out, (d.x1, ly), (d.x1 + tw + 6, d.y1), color, -1)
        cv2.putText(out, label, (d.x1 + 3, d.y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Distance ring on the bounding box ────────────────────────────────
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), dcol, 1)

        # ── Trail (fading dots) ───────────────────────────────────────────────
        pts = list(trk.history)
        for k, (px, py) in enumerate(pts):
            alpha = (k + 1) / len(pts)
            r     = max(2, int(2 + alpha * 5))
            c     = tuple(int(v * alpha * 0.9) for v in color)
            cv2.circle(out, (int(px), int(py)), r, c, -1, cv2.LINE_AA)
        # solid dot at current position
        if pts:
            cv2.circle(out, (int(pts[-1][0]), int(pts[-1][1])), 5, color, -1, cv2.LINE_AA)

        # ── Trajectory prediction ─────────────────────────────────────────────
        future = trk.predict_trajectory(PRED_FRAMES)
        if len(future) >= 2:
            # clip to frame bounds
            future = [
                (max(0, min(w_frame - 1, x)), max(0, min(h_frame - 1, y)))
                for x, y in future
            ]
            # dashed prediction curve: draw every other segment
            for k in range(0, len(future) - 1, 2):
                alpha = 1.0 - k / len(future)
                c     = tuple(int(v * alpha) for v in color)
                cv2.line(out, future[k], future[k + 1], c, 2, cv2.LINE_AA)
            # arrowhead at predicted endpoint
            tip_idx = min(len(future) - 1, PRED_FRAMES - 1)
            tail_idx = max(0, tip_idx - 6)
            if future[tail_idx] != future[tip_idx]:
                cv2.arrowedLine(out, future[tail_idx], future[tip_idx],
                                color, 2, cv2.LINE_AA, tipLength=0.5)

    # ── HUD ──────────────────────────────────────────────────────────────────
    n_live = sum(1 for t in tracks if t.missed == 0)
    cv2.putText(out, f"tracking={n_live}", (8, h_frame - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)
    return out


def run_loop(
    args: argparse.Namespace,
    model: YOLO,
    cap: cv2.VideoCapture,
) -> int:
    tracker = Tracker()
    writer: cv2.VideoWriter | None = None
    window = "Batting Cage  (q=quit)"

    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    fps_ema = 0.0
    t_prev  = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            h, w = frame.shape[:2]

            # Run YOLO, keep only batting-cage classes
            dets = infer_yolo_detections(model, frame,
                                         imgsz=args.imgsz,
                                         conf=args.conf,
                                         iou=args.iou)
            dets = [d for d in dets if d.cls_id in BATTING_CLASSES]

            # Monocular distance for each detection
            for d in dets:
                d.depth_m = estimate_distance(d, w, args.fov)

            # Update tracker
            tracks = tracker.update(dets)

            # Draw
            vis = draw_tracks(frame, tracks)

            # FPS overlay
            now     = time.time()
            inst    = 1.0 / max(1e-6, now - t_prev)
            fps_ema = 0.9 * fps_ema + 0.1 * inst
            t_prev  = now
            cv2.putText(vis, f"{fps_ema:.1f} fps",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)

            if writer is None and args.out:
                out_p = Path(args.out).expanduser().resolve()
                out_p.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_fps = args.fps if args.fps > 0 else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
                writer = cv2.VideoWriter(str(out_p.with_suffix(".tmp.mp4")), fourcc, out_fps, (w, h))
            if writer:
                writer.write(vis)

            if not args.headless:
                cv2.imshow(window, vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    if args.out and writer:
        tmp  = Path(args.out).with_suffix(".tmp.mp4")
        out  = Path(args.out).expanduser().resolve()
        print(f"[encode] → {out}")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
            capture_output=True, check=False,
        )
        tmp.unlink(missing_ok=True)
        print(f"[done] {out}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--cam",   type=int,  default=None)
    src.add_argument("--video", type=Path, default=None)
    ap.add_argument("--model",  type=Path, default=Path("yolo11n.pt"))
    ap.add_argument("--conf",   type=float, default=0.20)
    ap.add_argument("--iou",    type=float, default=0.45)
    ap.add_argument("--imgsz",  type=int,   default=640)
    ap.add_argument("--fov",    type=float, default=70.0,
                    help="Camera horizontal FOV in degrees (default 70).")
    ap.add_argument("--out",    type=Path, default=None)
    ap.add_argument("--fps",    type=float, default=0.0)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    model_path = args.model.expanduser().resolve()
    print(f"[yolo] loading {model_path}")
    model = YOLO(str(model_path))

    if args.video is not None:
        cap = cv2.VideoCapture(str(args.video.expanduser().resolve()))
    else:
        cam = 0 if args.cam is None else args.cam
        cap = cv2.VideoCapture(cam)

    if not cap.isOpened():
        print("ERROR: could not open video source", file=sys.stderr)
        return 1

    return run_loop(args, model, cap)


if __name__ == "__main__":
    sys.exit(main())
