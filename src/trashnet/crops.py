from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from . import config
from .data import load_labels

VARIANTS: dict[str, dict] = {
    "fixed336_288": {"mode": "window", "window": 336, "pad": 0.0, "px": 288},
    "fixed420_288": {"mode": "window", "window": 420, "pad": 0.0, "px": 288},
    "fixed560_288": {"mode": "window", "window": 560, "pad": 0.0, "px": 288},
    "fixed420_384": {"mode": "window", "window": 420, "pad": 0.0, "px": 384},
    "prop_288": {"mode": "window", "window": 0, "pad": 0.0, "px": 288},
    "bbox_warp": {"mode": "bbox", "window": 0, "pad": 0.0, "px": 288},
    "bbox_warp_pad25": {"mode": "bbox", "window": 0, "pad": 0.25, "px": 288},
    "bbox_letterbox": {"mode": "bbox_letterbox", "window": 0, "pad": 0.0, "px": 288},
    "bbox_square": {"mode": "square", "window": 0, "pad": 0.25, "px": 288},
    "frame_mask": {"mode": "frame_mask", "window": 0, "pad": 0.0, "px": 288},
    "frame_bboxdraw": {"mode": "frame_draw", "window": 0, "pad": 0.0, "px": 288},
}
PROP_CONTEXT = 3.0
PROP_MIN_WINDOW = 180


def quat_to_matrix(q: dict) -> np.ndarray:
    w, x, y, z = float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"])
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


def camera_pose(meta: dict) -> tuple[float, float] | None:
    pose = (meta or {}).get("color_cam_in_world")
    if not pose:
        return None
    axis = quat_to_matrix(pose["orientation"])[:, 2]
    angle_to_normal = float(np.degrees(np.arccos(float(np.clip(axis[2], -1.0, 1.0)))))
    return float(pose["position"][2]), 180.0 - angle_to_normal


def height_above_floor(depth: np.ndarray | None,
                       meta: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    cam = (meta or {}).get("color_camera")
    pose = (meta or {}).get("color_cam_in_world")
    if depth is None or not cam or not pose or not cam.get("fx"):
        return None, None
    fx, fy = float(cam["fx"]), float(cam["fy"])
    ppx, ppy = float(cam["ppx"]), float(cam["ppy"])
    rot = quat_to_matrix(pose["orientation"])
    t = np.asarray(pose["position"], np.float64)
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    valid = np.isfinite(depth) & (depth > 0.15) & (depth < 5.0)
    if not valid.any():
        return None, None
    d = np.where(valid, depth, 1.0).astype(np.float64)
    points = np.stack([(uu - ppx) * d / fx, (vv - ppy) * d / fy, d], axis=-1)
    zw = (points @ rot.T + t)[..., 2]
    floor = float(np.percentile(zw[valid], 10.0))
    return (zw - floor).astype(np.float32), valid


def mm_per_pixel(depth: np.ndarray | None, valid: np.ndarray | None, meta: dict) -> float | None:
    cam = (meta or {}).get("color_camera") or {}
    if not cam.get("fx") or depth is None or valid is None or not valid.any():
        return None
    return 1000.0 * float(np.median(depth[valid])) / float(cam["fx"])


def descriptors(depth: np.ndarray | None, meta: dict, bbox: Sequence[float]) -> dict[str, float]:
    out = {k: float("nan") for k in config.SCALARS}
    rel, valid = height_above_floor(depth, meta)
    mmpx = mm_per_pixel(depth, valid, meta)
    if rel is None or mmpx is None:
        return out
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, rel.shape[1]), min(y1, rel.shape[0])
    if x1 <= x0 or y1 <= y0:
        return out
    sub = rel[y0:y1, x0:x1]
    sok = valid[y0:y1, x0:x1] & np.isfinite(sub)
    if not sok.any():
        return out
    hmm = sub[sok] * 1000.0
    elevated = hmm > 3.0
    px = int(elevated.sum())
    area = px * (mmpx / 10.0) ** 2
    box_px = (x1 - x0) * (y1 - y0)
    ring = np.zeros_like(valid, bool)
    pad = max(int(0.5 * max(x1 - x0, y1 - y0)), 10)
    ring[max(y0 - pad, 0):min(y1 + pad, rel.shape[0]),
         max(x0 - pad, 0):min(x1 + pad, rel.shape[1])] = True
    ring[y0:y1, x0:x1] = False
    rok = ring & valid & np.isfinite(rel)
    out["area_cm2"] = area
    out["h_p90_mm"] = float(np.percentile(hmm, 90))
    out["h_med_mm"] = float(np.median(hmm))
    out["bulk_mm_per_cm"] = (float(hmm[elevated].mean()) / max(np.sqrt(max(area, 1e-6)), 1e-6)
                             if px else 0.0)
    out["solidity"] = float(px) / max(box_px, 1)
    out["elongation"] = float(max(x1 - x0, y1 - y0)) / max(min(x1 - x0, y1 - y0), 1)
    out["floor_noise_mm"] = float(np.std(rel[rok] * 1000.0)) if rok.sum() > 20 else 10.0
    out["z_m"] = float(np.median(depth[y0:y1, x0:x1][sok]))
    out["on_fixture_height_mm"] = float(np.percentile(hmm, 99))
    return out


def context_window(bbox: Sequence[float], w: int, h: int, context: float,
                   min_window: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = min(max(max(x1 - x0, y1 - y0, 8.0) * context, min_window), min(w, h))
    half = side / 2.0
    cx = min(max(cx, half), w - half)
    cy = min(max(cy, half), h - half)
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def bbox_window(bbox: Sequence[float], w: int, h: int, pad: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bw, bh = max(x1 - x0, 4.0), max(y1 - y0, 4.0)
    x0, x1 = x0 - pad * bw, x1 + pad * bw
    y0, y1 = y0 - pad * bh, y1 + pad * bh
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    if x1 - x0 < 4:
        x0, x1 = max(x0 - 2, 0), min(x0 + 4, w)
    if y1 - y0 < 4:
        y0, y1 = max(y0 - 2, 0), min(y0 + 4, h)
    return x0, y0, x1, y1


def square_window(bbox: Sequence[float], w: int, h: int, pad: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = min(max(x1 - x0, y1 - y0, 8.0) * (1.0 + 2.0 * pad), min(w, h))
    half = side / 2.0
    cx = min(max(cx, half), w - half)
    cy = min(max(cy, half), h - half)
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def pick_window(bbox: Sequence[float], w: int, h: int, spec: dict) -> tuple[int, int, int, int]:
    mode = spec["mode"]
    if mode in ("bbox", "bbox_letterbox"):
        return bbox_window(bbox, w, h, spec["pad"])
    if mode == "square":
        return square_window(bbox, w, h, spec["pad"])
    if mode in ("frame_mask", "frame_draw"):
        return 0, 0, w, h
    side = int(spec["window"])
    if side:
        return context_window(bbox, w, h, 0.0, side)
    return context_window(bbox, w, h, PROP_CONTEXT, PROP_MIN_WINDOW)


def letterbox(a: np.ndarray, px: int, interp: int) -> np.ndarray:
    h, w = a.shape[:2]
    s = float(px) / max(h, w)
    nh, nw = max(int(round(h * s)), 1), max(int(round(w * s)), 1)
    out = np.zeros((px, px) + a.shape[2:], a.dtype)
    oy, ox = (px - nh) // 2, (px - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = cv2.resize(a, (nw, nh), interpolation=interp)
    return out


def render(img: np.ndarray, win: tuple[int, int, int, int], px: int,
           mode: str = "resize") -> np.ndarray:
    x0, y0, x1, y1 = win
    crop = img[y0:y1, x0:x1]
    interp = cv2.INTER_AREA if img.dtype == np.uint8 else cv2.INTER_NEAREST
    if mode == "letterbox":
        return letterbox(crop, px, interp)
    return cv2.resize(crop, (px, px), interpolation=interp)


def read_depth(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is not None:
        return a if a.ndim == 2 else a[..., 0]
    import OpenEXR

    with OpenEXR.File(path) as fh:
        ch = fh.channels()
        key = "Y" if "Y" in ch else next(iter(ch))
        return np.asarray(ch[key].pixels, np.float32)


def load_frame(rgb_path: str) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    bgr = cv2.imread(rgb_path)
    depth = read_depth(rgb_path.replace("_rgb.jpg", "_depth.exr"))
    meta_path = rgb_path.replace("_rgb.jpg", "_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if depth is not None and bgr is not None and depth.shape[:2] != bgr.shape[:2]:
        depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return bgr, depth, meta


def frame_path(item_id: str) -> str:
    return str(config.FRAMES / f"{item_id}_rgb.jpg")


def bbox_mask_frame(shape: tuple[int, int], bbox: Sequence[float], px: int) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    m[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = 255
    return cv2.resize(m, (px, px), interpolation=cv2.INTER_NEAREST)


def draw_bbox(bgr: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), max(int(round(min(h, w) / 96.0)), 3))
    return out


def build_one(row: dict, spec: dict) -> dict | None:
    px = int(spec["px"])
    bgr, depth, meta = load_frame(frame_path(row["id"]))
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    win = pick_window(row["bbox_xyxy"], w, h, spec)
    fit = "letterbox" if spec["mode"] == "bbox_letterbox" else "resize"
    src = draw_bbox(bgr, row["bbox_xyxy"]) if spec["mode"] == "frame_draw" else bgr
    rgb = render(src, win, px, fit)
    if rgb.size == 0:
        return None
    rel, valid = height_above_floor(depth, meta)
    if rel is None:
        geo = np.zeros((3, px, px), np.float32)
    else:
        d = np.where(valid, depth, 0.0).astype(np.float32)
        r = np.where(valid, rel, 0.0).astype(np.float32)
        v = valid.astype(np.float32)
        dep = render(d, win, px, fit) / config.DEPTH_SCALE_M
        hei = np.clip(render(r, win, px, fit), *config.HEIGHT_CLIP_M) / config.HEIGHT_SCALE_M
        geo = np.stack([dep, hei, render(v, win, px, fit)])
    if spec["mode"] == "frame_mask":
        geo[2] = bbox_mask_frame((h, w), row["bbox_xyxy"], px).astype(np.float32) / 255.0
    desc = descriptors(depth, meta, row["bbox_xyxy"])
    return {"id": row["id"], "rgb": rgb, "geo": geo.astype(np.float16),
            "sc": np.array([desc.get(k, np.nan) for k in config.SCALARS], np.float32),
            "y": int(row["y"]), "mat": row.get("trash_type") or "other",
            "group": row["group"], "source": row["source"],
            "sample": row.get("sample", ""), "wet": bool(row.get("wet"))}


def _job(arg: tuple[dict, dict]) -> dict | None:
    return build_one(arg[0], arg[1])


def build_cache(variant: str, out: Path | None = None, workers: int = 8,
                with_geo: bool = True) -> Path:
    spec = VARIANTS[variant]
    rows = load_labels()
    out = Path(out) if out is not None else config.CACHE / f"{variant}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    got: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_job, [(row, spec) for row in rows], chunksize=8):
            if r:
                got.append(r)
    assert len(got) == len(rows), f"{len(got)} of {len(rows)} items built"
    payload = {
        "rgb": np.stack([g["rgb"] for g in got]),
        "sc": np.stack([g["sc"] for g in got]),
        "y": np.array([g["y"] for g in got], np.int64),
        "mat": np.array([g["mat"] for g in got]),
        "group": np.array([g["group"] for g in got]),
        "source": np.array([g["source"] for g in got]),
        "sample": np.array([g["sample"] for g in got]),
        "wet": np.array([g["wet"] for g in got]),
        "ids": np.array([g["id"] for g in got]),
        "scalars": np.array(config.SCALARS),
    }
    if with_geo:
        payload["geo"] = np.stack([g["geo"] for g in got])
    np.savez(out, **payload)
    return out
