#!/usr/bin/env python3
"""Camera + BEV side-by-side with YOLO detection, tracking, and trajectory prediction.

Left panel  — camera frame with bounding boxes, fading trails, parabolic arcs.
Right panel — BEV (segmentation colors when seg_maps.npz available, otherwise
              a plain grid) with detections and trajectories projected to ground.

Pulls BEV geometry from drive-by-segmentation/camera_calibration.json.

Usage:
  python scripts/combined_view.py --video path/to/video.mp4
  python scripts/combined_view.py --video v.mp4 --seg /path/to/seg_maps.npz --out out.mp4
  python scripts/combined_view.py --cam 0 --out live.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DBS_ROOT     = Path.home() / "drive-by-segmentation"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(DBS_ROOT))

from depth_bev import PredDetection, infer_yolo_detections
from batting_cage_detect import (
    Tracker, BATTING_CLASSES, CLASS_COLORS,
    estimate_distance, draw_tracks, PRED_FRAMES,
)

try:
    from render import create_bev, CITYSCAPES_COLORS
    _HAS_DBS = True
except ImportError:
    _HAS_DBS = False
    print("[warn] drive-by-segmentation/render.py not found — using blank BEV", file=sys.stderr)

BEV_SIZE = 500
FT_TO_M  = 0.3048


# ── Calibration helpers ───────────────────────────────────────────────────────

def scale_calib(calib: dict, src_w: int, src_h: int) -> dict:
    """Scale intrinsics to match a different frame resolution."""
    cal_w, cal_h = calib["intrinsics"]["resolution"]
    if cal_w == src_w and cal_h == src_h:
        return calib
    sx, sy = src_w / cal_w, src_h / cal_h
    c = json.loads(json.dumps(calib))
    c["intrinsics"]["focal_length"] *= sx
    c["intrinsics"]["cx"]           *= sx
    c["intrinsics"]["cy"]           *= sy
    c["intrinsics"]["resolution"]    = [src_w, src_h]
    return c


def _build_R(calib: dict) -> np.ndarray:
    pitch = math.radians(calib["extrinsics"]["pitch_deg"])
    roll  = math.radians(calib["extrinsics"]["roll_deg"])
    yaw   = math.radians(calib["extrinsics"]["yaw_deg"])
    cp, sp   = math.cos(pitch), math.sin(pitch)
    cr, sr   = math.cos(roll),  math.sin(roll)
    cyw, syw = math.cos(yaw),   math.sin(yaw)

    Ryaw   = np.array([[cyw,-syw,0],[syw,cyw,0],[0,0,1]], dtype=np.float64)
    Rbase  = np.array([[1,0,0],[0,0,-1],[0,1,0]],         dtype=np.float64)
    Rpitch = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]],      dtype=np.float64)
    Rroll  = np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]],      dtype=np.float64)
    return Rroll @ Rpitch @ Rbase @ Ryaw


# ── Ground-plane projection ───────────────────────────────────────────────────

def image_to_bev_px(
    us: np.ndarray,
    vs: np.ndarray,
    calib: dict,
    range_fwd: float,
    range_side: float,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project image-space points (u, v) → BEV pixel (bx, by).

    Returns (bx, by, valid_mask).  Uses the same equidistant fisheye
    model as drive-by-segmentation/render.py, run in reverse.
    """
    f  = float(calib["intrinsics"]["focal_length"])
    cx = float(calib["intrinsics"]["cx"])
    cy = float(calib["intrinsics"]["cy"])
    k1 = float(calib["intrinsics"]["k1"])
    k2 = float(calib["intrinsics"]["k2"])
    h  = float(calib["extrinsics"]["height_m"])

    du  = np.asarray(us, dtype=np.float64) - cx
    dv  = np.asarray(vs, dtype=np.float64) - cy
    r_d = np.hypot(du, dv)
    ok  = r_d > 1e-8

    # Equidistant: r_d = f * theta * (1 + k1*theta^2 + k2*theta^4)
    theta = np.where(ok, r_d / f, 0.0)
    if k1 != 0.0 or k2 != 0.0:
        for _ in range(10):
            t2  = theta ** 2
            td  = theta * (1 + k1 * t2 + k2 * t2 ** 2)
            dtd = 1 + 3 * k1 * t2 + 5 * k2 * t2 ** 2
            theta = np.where(ok, theta - (f * td - r_d) / (f * dtd + 1e-12), theta)

    r3d   = np.sin(theta)
    safe  = np.where(r_d > 1e-8, r_d, 1.0)
    cam_x = np.where(ok, r3d * du / safe, 0.0)
    cam_y = np.where(ok, r3d * dv / safe, 0.0)
    cam_z = np.cos(theta)

    # Rotate camera ray to ego frame: ego_vec = R^T @ cam_vec
    Rt    = R.T
    cam   = np.stack([cam_x, cam_y, cam_z], axis=0)   # (3, N)
    ego   = Rt @ cam                                    # (3, N)

    # Intersect with ground plane: t * ego[2] = -h  →  t = -h / ego[2]
    gok   = ok & (np.abs(ego[2]) > 1e-6)
    t_ray = np.where(gok, -h / ego[2], 0.0)
    gok  &= t_ray > 0

    side_m = np.where(gok, t_ray * ego[0], np.nan)
    fwd_m  = np.where(gok, t_ray * ego[1], np.nan)

    bx = (side_m / range_side * 0.5 + 0.5) * BEV_SIZE
    by = (1.0 - fwd_m / range_fwd)          * BEV_SIZE

    valid = (
        gok
        & (fwd_m  >= 0)          & (fwd_m  <= range_fwd)
        & (np.abs(side_m) <= range_side)
        & (bx >= 0) & (bx < BEV_SIZE)
        & (by >= 0) & (by < BEV_SIZE)
    )
    return bx.astype(np.int32), by.astype(np.int32), valid


# ── BEV drawing ───────────────────────────────────────────────────────────────

def make_blank_bev(range_fwd: float, range_side: float) -> np.ndarray:
    img = np.full((BEV_SIZE, BEV_SIZE, 3), 28, dtype=np.uint8)
    for fwd_m in np.arange(5, range_fwd + 1, 5):
        by = int((1 - fwd_m / range_fwd) * BEV_SIZE)
        if 0 <= by < BEV_SIZE:
            cv2.line(img, (0, by), (BEV_SIZE, by), (55, 52, 48), 1)
            cv2.putText(img, f"{fwd_m / FT_TO_M:.0f}ft", (4, max(12, by - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130, 130, 130), 1, cv2.LINE_AA)
    for side_frac in np.linspace(-1, 1, 11):
        bx = int((side_frac * 0.5 + 0.5) * BEV_SIZE)
        if 0 <= bx < BEV_SIZE:
            cv2.line(img, (bx, 0), (bx, BEV_SIZE), (55, 52, 48), 1)
    ex, ey = BEV_SIZE // 2, BEV_SIZE - 8
    cv2.fillPoly(img, [np.array([[ex, ey - 14], [ex - 7, ey], [ex + 7, ey]])], (255, 255, 255))
    cv2.putText(img, "BEV", (BEV_SIZE // 2 - 18, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return img


def draw_bev_tracks(
    bev: np.ndarray,
    tracks: list,
    calib: dict,
    range_fwd: float,
    range_side: float,
    R: np.ndarray,
) -> np.ndarray:
    out = bev.copy()

    for trk in tracks:
        if trk.missed > 0:
            continue
        color = CLASS_COLORS.get(trk.cls_id, (180, 180, 180))
        pts   = list(trk.history)
        if not pts:
            continue

        # ── Past trail ───────────────────────────────────────────────────────
        us = np.array([p[0] for p in pts], dtype=np.float64)
        vs = np.array([p[1] for p in pts], dtype=np.float64)
        bxs, bys, valids = image_to_bev_px(us, vs, calib, range_fwd, range_side, R)

        prev = None
        for k in range(len(pts)):
            if not valids[k]:
                prev = None
                continue
            alpha = (k + 1) / len(pts)
            c     = tuple(int(v * alpha) for v in color)
            r     = max(2, int(2 + alpha * 3))
            pt    = (int(bxs[k]), int(bys[k]))
            cv2.circle(out, pt, r, c, -1, cv2.LINE_AA)
            if prev:
                cv2.line(out, prev, pt, c, 1, cv2.LINE_AA)
            prev = pt

        # Bright dot at current position
        if valids[-1]:
            cp = (int(bxs[-1]), int(bys[-1]))
            cv2.circle(out, cp, 6, color, -1, cv2.LINE_AA)
            cv2.circle(out, cp, 7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(out, f"#{trk.id}", (cp[0] + 8, cp[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # ── Predicted trajectory on BEV ──────────────────────────────────────
        future = trk.predict_trajectory(PRED_FRAMES)
        if len(future) < 2:
            continue

        fus = np.array([p[0] for p in future], dtype=np.float64)
        fvs = np.array([p[1] for p in future], dtype=np.float64)
        fbx, fby, fval = image_to_bev_px(fus, fvs, calib, range_fwd, range_side, R)

        prev_f = None
        for k in range(len(future)):
            if not fval[k]:
                prev_f = None
                continue
            if k % 2 != 0:
                prev_f = (int(fbx[k]), int(fby[k]))
                continue  # dashed
            alpha = 1.0 - k / len(future)
            c     = tuple(int(v * alpha) for v in color)
            fp    = (int(fbx[k]), int(fby[k]))
            if prev_f:
                cv2.line(out, prev_f, fp, c, 2, cv2.LINE_AA)
            prev_f = fp

        valid_pts = [(int(fbx[k]), int(fby[k])) for k in range(len(future)) if fval[k]]
        if len(valid_pts) >= 4:
            cv2.arrowedLine(out, valid_pts[-4], valid_pts[-1],
                            color, 2, cv2.LINE_AA, tipLength=0.5)

    return out


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_loop(
    args: argparse.Namespace,
    model: YOLO,
    calib_orig: dict,
    cap: cv2.VideoCapture,
) -> int:
    bev_range  = calib_orig.get("bev_range", {})
    range_fwd  = bev_range.get("forward_ft", 50)  * FT_TO_M
    range_side = bev_range.get("side_ft",    25)   * FT_TO_M

    # Optional pre-computed segmentation maps
    seg_maps: np.ndarray | None      = None
    frame_indices: np.ndarray | None = None
    if args.seg and Path(args.seg).is_file():
        d             = np.load(args.seg)
        seg_maps      = d["seg_maps"]
        frame_indices = d["frame_indices"]
        print(f"[seg] loaded {len(seg_maps)} segmentation frames")

    tracker  = Tracker()
    writer: cv2.VideoWriter | None = None
    window   = "YOLO + BEV  (q=quit)"
    fps_ema  = 0.0
    t_prev   = time.time()
    frame_no = 0
    calib    = calib_orig  # will be rescaled on first frame
    R: np.ndarray | None = None
    last_bev_base: np.ndarray | None = None  # holds last rendered seg BEV

    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            fh, fw = frame.shape[:2]

            # Rescale calibration intrinsics once to match actual frame size
            if R is None:
                calib = scale_calib(calib_orig, fw, fh)
                R     = _build_R(calib)

            # ── YOLO + tracker ───────────────────────────────────────────────
            dets = infer_yolo_detections(model, frame,
                                         imgsz=args.imgsz,
                                         conf=args.conf,
                                         iou=args.iou)
            dets = [d for d in dets if d.cls_id in BATTING_CLASSES]
            for d in dets:
                d.depth_m = estimate_distance(d, fw, args.fov)
            tracks = tracker.update(dets)

            # ── Left: camera view with overlays ──────────────────────────────
            cam_vis = draw_tracks(frame, tracks)

            # ── Right: BEV ───────────────────────────────────────────────────
            if seg_maps is not None and frame_indices is not None and _HAS_DBS:
                idx = min(int(np.searchsorted(frame_indices, frame_no)),
                          len(seg_maps) - 1)
                # Only re-render when we reach a new segmentation frame
                if last_bev_base is None or int(frame_indices[idx]) == frame_no:
                    last_bev_base = cv2.cvtColor(
                        create_bev(seg_maps[idx], calib_orig, BEV_SIZE),
                        cv2.COLOR_RGB2BGR,
                    )

            bev_base = last_bev_base if last_bev_base is not None \
                else make_blank_bev(range_fwd, range_side)

            bev_vis = draw_bev_tracks(bev_base, tracks, calib, range_fwd, range_side, R)

            # ── Composite side-by-side ───────────────────────────────────────
            bev_resized = cv2.resize(bev_vis, (fh, fh))
            composite   = np.hstack([cam_vis, bev_resized])

            # Panel labels
            cw = composite.shape[1]
            cv2.putText(composite, "Camera + YOLO", (10, fh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
            cv2.putText(composite, "BEV + Trajectories", (fw + 10, fh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
            # Divider
            cv2.line(composite, (fw, 0), (fw, fh), (80, 80, 80), 2)

            # FPS
            now     = time.time()
            fps_ema = 0.9 * fps_ema + 0.1 / max(1e-6, now - t_prev)
            t_prev  = now
            cv2.putText(composite, f"{fps_ema:.1f} fps",
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            if writer is None and args.out:
                out_p   = Path(args.out).expanduser().resolve()
                out_p.parent.mkdir(parents=True, exist_ok=True)
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                out_fps = args.fps if args.fps > 0 else src_fps
                fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
                writer  = cv2.VideoWriter(
                    str(out_p.with_suffix(".tmp.mp4")), fourcc, out_fps,
                    (composite.shape[1], composite.shape[0]),
                )
            if writer:
                writer.write(composite)

            if not args.headless:
                cv2.imshow(window, composite)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            frame_no += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    if args.out and writer:
        tmp = Path(args.out).with_suffix(".tmp.mp4")
        out = Path(args.out).expanduser().resolve()
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
    src.add_argument("--video", type=Path, default=None)
    src.add_argument("--cam",   type=int,  default=None)
    ap.add_argument("--calib", type=Path,
                    default=DBS_ROOT / "camera_calibration.json",
                    help="camera_calibration.json from drive-by-segmentation.")
    ap.add_argument("--seg",  type=Path, default=None,
                    help="Optional seg_maps.npz for BEV segmentation colors.")
    ap.add_argument("--model",    type=Path,  default=Path("yolo11n.pt"))
    ap.add_argument("--conf",     type=float, default=0.20)
    ap.add_argument("--iou",      type=float, default=0.45)
    ap.add_argument("--imgsz",    type=int,   default=640)
    ap.add_argument("--fov",      type=float, default=70.0,
                    help="Camera horizontal FOV for monocular distance (degrees).")
    ap.add_argument("--out",      type=Path,  default=None)
    ap.add_argument("--fps",      type=float, default=0.0)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    calib_path = args.calib.expanduser().resolve()
    if not calib_path.is_file():
        print(f"ERROR: calibration not found: {calib_path}", file=sys.stderr)
        return 1
    calib = json.loads(calib_path.read_text())
    print(f"[calib] {calib_path.name}  camera={calib.get('camera', {}).get('model', '?')}")

    model_path = (PROJECT_ROOT / args.model).resolve()
    if not model_path.is_file():
        model_path = args.model.expanduser().resolve()
    print(f"[yolo] loading {model_path}")
    model = YOLO(str(model_path))

    if args.video is not None:
        cap = cv2.VideoCapture(str(args.video.expanduser().resolve()))
    else:
        cap = cv2.VideoCapture(0 if args.cam is None else args.cam)
    if not cap.isOpened():
        print("ERROR: could not open video source", file=sys.stderr)
        return 1

    return run_loop(args, model, calib, cap)


if __name__ == "__main__":
    sys.exit(main())
