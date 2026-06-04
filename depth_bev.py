"""Lift-splat BEV occupancy from RealSense metric depth (LSS-style, oracle depth).

Ego frame: +X forward, +Y left, +Z up. BEV image: ego at bottom-center, forward = up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class BevGridSpec:
    range_fwd_m: float = 25.0
    range_side_m: float = 12.0
    bev_size: int = 400
    cell_m: float | None = None

    @property
    def meters_per_px_fwd(self) -> float:
        return self.range_fwd_m / self.bev_size

    @property
    def meters_per_px_side(self) -> float:
        return (2.0 * self.range_side_m) / self.bev_size

    def local_to_bev(self, fwd: np.ndarray, left: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bx = (left / self.range_side_m * 0.5 + 0.5) * self.bev_size
        by = (1.0 - fwd / self.range_fwd_m) * self.bev_size
        return bx, by


@dataclass(frozen=True)
class CameraExtrinsics:
    """RealSense optical frame → ego (forward, left, up)."""

    height_m: float = 1.35
    pitch_deg: float = 8.0
    roll_deg: float = 0.0
    yaw_deg: float = 0.0
    cam_offset_fwd_m: float = 0.45
    cam_offset_left_m: float = 0.0

    def rotation_cam_to_ego(self) -> np.ndarray:
        pitch = math.radians(self.pitch_deg)
        roll = math.radians(self.roll_deg)
        yaw = math.radians(self.yaw_deg)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # Optical: +X right, +Y down, +Z forward
        r_opt_to_cam = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        ryaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        rpitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        rroll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
        r_mount = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        return ryaw @ rpitch @ rroll @ r_mount @ r_opt_to_cam

    def translation_cam_origin_in_ego(self) -> np.ndarray:
        return np.array([self.cam_offset_fwd_m, self.cam_offset_left_m, self.height_m], dtype=np.float64)


@dataclass
class DepthBevFrame:
    occupancy: np.ndarray
    density: np.ndarray
    height_max_m: np.ndarray
    height_min_m: np.ndarray
    nearest_fwd_m: np.ndarray
    spec: BevGridSpec = field(default_factory=BevGridSpec)

    @property
    def obstacle_mask(self) -> np.ndarray:
        """Cells with occupied points below 2.2 m (pedestrian / vehicle bulk)."""
        occ = self.occupancy > 0
        low = self.height_min_m < 2.2
        return occ & low

    def corridor_min_depth_m(
        self,
        half_width_m: float = 1.2,
        fwd_min_m: float = 0.5,
        fwd_max_m: float | None = None,
    ) -> float:
        s = self.spec
        if fwd_max_m is None:
            fwd_max_m = s.range_fwd_m
        gx = np.arange(s.bev_size, dtype=np.float32)
        gy = np.arange(s.bev_size, dtype=np.float32)
        bx, by = np.meshgrid(gx, gy)
        left = (bx / s.bev_size - 0.5) * 2.0 * s.range_side_m
        fwd = (1.0 - by / s.bev_size) * s.range_fwd_m
        mask = (
            (self.nearest_fwd_m > 0)
            & (fwd >= fwd_min_m)
            & (fwd <= fwd_max_m)
            & (np.abs(left) <= half_width_m)
        )
        vals = self.nearest_fwd_m[mask]
        if vals.size == 0:
            return 99.0
        return float(np.percentile(vals, 10))


class DepthBevSplat:
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        extrinsics: CameraExtrinsics | None = None,
        spec: BevGridSpec | None = None,
        min_depth_m: float = 0.35,
        max_depth_m: float = 22.0,
    ) -> None:
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.extrinsics = extrinsics or CameraExtrinsics()
        self.spec = spec or BevGridSpec()
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self._r_ce = self.extrinsics.rotation_cam_to_ego()
        self._t_ego = self.extrinsics.translation_cam_origin_in_ego()

    @classmethod
    def from_realsense_meta(cls, meta: dict, **kwargs) -> "DepthBevSplat":
        intr = meta["intrinsics"]
        ext = CameraExtrinsics(
            height_m=float(kwargs.pop("height_m", 1.35)),
            pitch_deg=float(kwargs.pop("pitch_deg", 8.0)),
        )
        return cls(
            fx=float(intr["fx"]),
            fy=float(intr["fy"]),
            cx=float(intr["ppx"]),
            cy=float(intr["ppy"]),
            extrinsics=ext,
            **kwargs,
        )

    def unproject_cam(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=-1)

    def cam_to_ego(self, pts_cam: np.ndarray) -> np.ndarray:
        return (self._r_ce @ pts_cam.T).T + self._t_ego

    def splat(
        self,
        depth_m: np.ndarray,
        rgb: np.ndarray | None = None,
        stride: int = 3,
        ground_z_max_m: float = 0.45,
    ) -> DepthBevFrame:
        h, w = depth_m.shape[:2]
        v, u = np.mgrid[0:h:stride, 0:w:stride]
        z = depth_m[v, u].astype(np.float32)
        valid = (z >= self.min_depth_m) & (z <= self.max_depth_m) & np.isfinite(z)
        if not np.any(valid):
            return self._empty_frame()

        u_f = u[valid].astype(np.float32).ravel()
        v_f = v[valid].astype(np.float32).ravel()
        z_f = z[valid].ravel()
        pts_cam = self.unproject_cam(u_f, v_f, z_f)
        pts_ego = self.cam_to_ego(pts_cam)
        fwd, left, up = pts_ego[:, 0], pts_ego[:, 1], pts_ego[:, 2]

        s = self.spec
        in_range = (
            (fwd > 0.2)
            & (fwd <= s.range_fwd_m)
            & (np.abs(left) <= s.range_side_m)
        )
        if not np.any(in_range):
            return self._empty_frame()

        fwd, left, up = fwd[in_range], left[in_range], up[in_range]
        bx, by = s.local_to_bev(fwd, left)
        ix = np.clip(bx.astype(np.int32), 0, s.bev_size - 1)
        iy = np.clip(by.astype(np.int32), 0, s.bev_size - 1)

        occ = np.zeros((s.bev_size, s.bev_size), dtype=np.float32)
        dens = np.zeros_like(occ)
        zmax = np.full_like(occ, -99.0)
        zmin = np.full_like(occ, 99.0)
        nfwd = np.full_like(occ, 99.0)

        for i in range(len(ix)):
            x, y = ix[i], iy[i]
            dens[y, x] += 1.0
            occ[y, x] = 1.0
            zmax[y, x] = max(zmax[y, x], up[i])
            zmin[y, x] = min(zmin[y, x], up[i])
            nfwd[y, x] = min(nfwd[y, x], fwd[i])

        ground = (zmax < ground_z_max_m) & (occ > 0)
        occ[ground] = 0.5

        return DepthBevFrame(
            occupancy=occ,
            density=dens,
            height_max_m=zmax,
            height_min_m=zmin,
            nearest_fwd_m=nfwd,
            spec=s,
        )

    def _empty_frame(self) -> DepthBevFrame:
        s = self.spec
        z = np.full((s.bev_size, s.bev_size), 99.0, dtype=np.float32)
        return DepthBevFrame(
            occupancy=np.zeros((s.bev_size, s.bev_size), dtype=np.float32),
            density=np.zeros((s.bev_size, s.bev_size), dtype=np.float32),
            height_max_m=np.full((s.bev_size, s.bev_size), -99.0, dtype=np.float32),
            height_min_m=z.copy(),
            nearest_fwd_m=z.copy(),
            spec=s,
        )

    def render(
        self,
        frame: DepthBevFrame,
        show_grid: bool = True,
        show_ego: bool = True,
    ) -> np.ndarray:
        s = frame.spec
        img = np.full((s.bev_size, s.bev_size, 3), (28, 24, 20), dtype=np.uint8)

        occ = frame.occupancy
        obst = frame.obstacle_mask
        free = (occ == 0) & (frame.nearest_fwd_m > 0.5)
        ground = (occ > 0) & (occ < 1.0)
        img[free] = (35, 42, 35)
        img[ground] = (55, 90, 55)
        img[obst] = (50, 50, 220)

        dens_norm = np.clip(frame.density / max(1.0, frame.density.max()), 0, 1)
        heat = (dens_norm * 180).astype(np.uint8)
        heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
        blend = obst.astype(np.float32)[..., None]
        img = np.where(blend, img, (img.astype(np.float32) * 0.55 + heatmap.astype(np.float32) * 0.45).astype(np.uint8))

        if show_grid:
            for dist in range(5, int(s.range_fwd_m) + 1, 5):
                y = int((1 - dist / s.range_fwd_m) * s.bev_size)
                if 0 <= y < s.bev_size:
                    cv2.line(img, (0, y), (s.bev_size, y), (55, 50, 45), 1)
                    cv2.putText(img, f"{dist}m", (4, max(12, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
            for dist in range(-10, 11, 5):
                x = int((dist / s.range_side_m * 0.5 + 0.5) * s.bev_size)
                if 0 <= x < s.bev_size:
                    cv2.line(img, (x, 0), (x, s.bev_size), (55, 50, 45), 1)

        danger_y = int((1 - 4.0 / s.range_fwd_m) * s.bev_size)
        if 0 <= danger_y < s.bev_size:
            tint = img[danger_y:, :].copy()
            tint[:, :, 2] = np.clip(tint[:, :, 2].astype(np.int16) + 25, 0, 255).astype(np.uint8)
            img[danger_y:, :] = tint

        if show_ego:
            ex, ey = s.bev_size // 2, s.bev_size - 10
            pts = np.array([[ex, ey - 12], [ex - 8, ey], [ex + 8, ey]])
            cv2.fillPoly(img, [pts], (255, 255, 255))

        cv2.putText(img, "Depth BEV", (s.bev_size // 2 - 42, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return img

    def render_depth_overlay(self, depth_m: np.ndarray, max_m: float = 12.0) -> np.ndarray:
        vis = np.clip(depth_m / max_m, 0, 1)
        vis = (255 - vis * 255).astype(np.uint8)
        return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


# COCO classes we care about for driving (BGR colors)
PRED_COLORS: dict[int, tuple[int, int, int]] = {
    0: (60, 60, 255),    # person
    1: (0, 165, 255),    # bicycle
    2: (50, 150, 255),   # car
    3: (0, 255, 255),    # motorcycle
    5: (200, 100, 255),  # bus
    7: (50, 220, 50),    # truck
    9: (250, 170, 30),   # traffic light
    11: (50, 50, 255),   # stop sign
}
PRED_CLASS_IDS = set(PRED_COLORS)


@dataclass
class PredDetection:
    cls_id: int
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    name: str = ""
    depth_m: float = 0.0
    forward_m: float = 0.0
    left_m: float = 0.0


def median_depth_in_box(depth_m: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    z, _, _ = depth_at_detection_foot(depth_m, x1, y1, x2, y2)
    return z


def depth_at_detection_foot(
    depth_m: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    foot_frac: float = 0.28,
    center_frac: float = 0.42,
    percentile: float = 12.0,
    min_z: float = 0.35,
    max_z: float = 40.0,
) -> tuple[float, float, float]:
    """Robust metric depth at the object foot (lower-center bbox patch).

    Returns ``(z_m, foot_u, foot_v)`` for pinhole unprojection. Uses a low
    percentile in the foot patch to approximate nearest visible surface.
    """
    h, w = depth_m.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0

    bw = x2 - x1
    bh = max(1, y2 - y1)
    foot_h = max(2, int(bh * foot_frac))
    cx1 = int(x1 + (1.0 - center_frac) * 0.5 * bw)
    cx2 = int(x2 - (1.0 - center_frac) * 0.5 * bw)
    fy1 = max(y1, y2 - foot_h)
    roi = depth_m[fy1:y2, cx1:cx2]
    valid = roi[(roi >= min_z) & (roi <= max_z) & np.isfinite(roi)]
    if valid.size == 0:
        return 0.0, 0.0, 0.0

    z = float(np.percentile(valid, percentile))
    foot_u = 0.5 * (cx1 + cx2)
    foot_v = y2 - 0.5 * foot_h
    return z, foot_u, foot_v


def detection_range_m(d: PredDetection) -> float | None:
    """Best ego-range estimate for overlay / logging (forward preferred)."""
    if d.forward_m > 0.5:
        return d.forward_m
    if d.depth_m > 0.35:
        return d.depth_m
    return None


def infer_yolo_detections(
    model,
    frame_bgr: np.ndarray,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
) -> list[PredDetection]:
    """Run YOLO on a BGR frame; returns all COCO classes above confidence."""
    result = model.predict(frame_bgr, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    names = result.names or {}
    dets: list[PredDetection] = []
    for box, score, cls_id in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
    ):
        x1, y1, x2, y2 = [int(v) for v in box]
        cid = int(cls_id)
        dets.append(
            PredDetection(
                cls_id=cid,
                conf=float(score),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                name=str(names.get(cid, cid)),
            )
        )
    return dets


def enrich_detections_metric(
    dets: list[PredDetection],
    *,
    depth_m: np.ndarray | None = None,
    splat: DepthBevSplat | None = None,
    calib: dict | None = None,
    image_shape: tuple[int, int] | None = None,
) -> list[PredDetection]:
    """Attach metric distance to every detection (depth ROI, else ground-plane calib)."""
    ground_fn = None
    if calib is not None and image_shape is not None:
        try:
            import sys
            from pathlib import Path

            seg_root = Path(__file__).resolve().parent.parent.parent / "drive-by-segmentation"
            if str(seg_root) not in sys.path:
                sys.path.insert(0, str(seg_root))
            from bev_geometry import image_pixel_to_ground_m

            ground_fn = image_pixel_to_ground_m
        except Exception:
            ground_fn = None

    out: list[PredDetection] = []
    for d in dets:
        z = 0.0
        fwd = 0.0
        left = 0.0
        if depth_m is not None:
            z, foot_u, foot_v = depth_at_detection_foot(depth_m, d.x1, d.y1, d.x2, d.y2)
            if z > 0.0 and splat is not None:
                pts_cam = splat.unproject_cam(
                    np.array([foot_u], dtype=np.float32),
                    np.array([foot_v], dtype=np.float32),
                    np.array([z], dtype=np.float32),
                )
                pts_ego = splat.cam_to_ego(pts_cam)[0]
                fwd = float(pts_ego[0])
                left = float(pts_ego[1])
        elif ground_fn is not None and image_shape is not None:
            foot_x = 0.5 * (d.x1 + d.x2)
            foot_y = float(d.y2)
            ground = ground_fn(foot_x, foot_y, image_shape, calib)
            if ground is not None:
                fwd, left = ground
                z = math.hypot(fwd, left)

        out.append(
            PredDetection(
                cls_id=d.cls_id,
                conf=d.conf,
                x1=d.x1,
                y1=d.y1,
                x2=d.x2,
                y2=d.y2,
                name=d.name,
                depth_m=z,
                forward_m=fwd,
                left_m=left,
            )
        )
    return out


def enrich_detections(
    dets: list[PredDetection],
    depth_m: np.ndarray,
    splat: DepthBevSplat,
) -> list[PredDetection]:
    out: list[PredDetection] = []
    for d in dets:
        z, foot_u, foot_v = depth_at_detection_foot(depth_m, d.x1, d.y1, d.x2, d.y2)
        if z <= 0.0:
            continue
        pts_cam = splat.unproject_cam(
            np.array([foot_u], dtype=np.float32),
            np.array([foot_v], dtype=np.float32),
            np.array([z], dtype=np.float32),
        )
        pts_ego = splat.cam_to_ego(pts_cam)[0]
        out.append(
            PredDetection(
                cls_id=d.cls_id,
                conf=d.conf,
                x1=d.x1,
                y1=d.y1,
                x2=d.x2,
                y2=d.y2,
                name=d.name,
                depth_m=z,
                forward_m=float(pts_ego[0]),
                left_m=float(pts_ego[1]),
            )
        )
    return out


def draw_yolo_on_color(frame_bgr: np.ndarray, dets: list[PredDetection]) -> np.ndarray:
    out = frame_bgr.copy()
    for d in dets:
        color = PRED_COLORS.get(d.cls_id, (180, 180, 180))
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        dist = detection_range_m(d)
        dist_txt = f"{dist:.2f}m" if dist is not None else "?"
        label = f"{d.name or d.cls_id} {d.conf:.2f} {dist_txt}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (d.x1, max(0, d.y1 - th - 6)), (d.x1 + tw + 4, d.y1), color, -1)
        cv2.putText(out, label, (d.x1 + 2, d.y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(
        out,
        f"YOLO preds: {len(dets)}",
        (8, out.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        2,
    )
    return out


def draw_predictions_on_bev(bev_img: np.ndarray, dets: list[PredDetection], spec: BevGridSpec) -> np.ndarray:
    out = bev_img.copy()
    for d in dets:
        if d.cls_id not in PRED_CLASS_IDS or d.forward_m <= 0.5:
            continue
        if d.forward_m > spec.range_fwd_m or abs(d.left_m) > spec.range_side_m:
            continue
        bx, by = spec.local_to_bev(
            np.array([d.forward_m], dtype=np.float32),
            np.array([d.left_m], dtype=np.float32),
        )
        bev_x, bev_y = int(bx[0]), int(by[0])
        if not (0 <= bev_x < spec.bev_size and 0 <= bev_y < spec.bev_size):
            continue
        color = PRED_COLORS.get(d.cls_id, (150, 150, 150))
        cv2.circle(out, (bev_x, bev_y), 7, color, -1, cv2.LINE_AA)
        cv2.circle(out, (bev_x, bev_y), 7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            out,
            f"{d.name[:4]} {d.forward_m:.0f}m",
            (bev_x + 8, bev_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(out, "BEV preds", (spec.bev_size // 2 - 38, spec.bev_size - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return out
