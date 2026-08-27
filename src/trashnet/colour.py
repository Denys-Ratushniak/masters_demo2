from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from . import config
from .data import Items

COLUMNS: tuple[str, ...] = ("rgb_mean_r", "rgb_mean_g", "rgb_mean_b", "rgb_std_r", "rgb_std_g",
                            "rgb_std_b", "brightness", "whiteness")


def cache_path() -> Path:
    return config.CACHE / "bbox_colour.npz"


def stats_for(args: tuple[str, tuple[float, float, float, float]]) -> np.ndarray:
    item_id, bbox = args
    bgr = cv2.imread(str(config.FRAMES / f"{item_id}_rgb.jpg"))
    if bgr is None:
        return np.full(len(COLUMNS), np.nan, np.float32)
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(max(x1, x0 + 1), w), min(max(y1, y0 + 1), h)
    patch = bgr[y0:y1, x0:x1, ::-1].astype(np.float32)
    flat = patch.reshape(-1, 3)
    return np.concatenate([flat.mean(0), flat.std(0),
                           [flat.mean(), flat.min(1).mean()]]).astype(np.float32)


def build(ids: Sequence[str], bboxes: np.ndarray, workers: int = 8) -> np.ndarray:
    jobs = [(str(i), tuple(float(v) for v in b)) for i, b in zip(ids, bboxes)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(stats_for, jobs, chunksize=16))
    return np.stack(rows).astype(np.float32)


def features(items: Items, workers: int = 8) -> np.ndarray:
    path = cache_path()
    if path.exists():
        with np.load(path, allow_pickle=True) as z:
            pos = {str(v): k for k, v in enumerate(z["ids"].astype(str))}
            if all(str(i) in pos for i in items.ids):
                return z["x"][np.array([pos[str(i)] for i in items.ids], np.int64)]
    x = build(items.ids, items.bbox, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.asarray(items.ids, object).astype(str), x=x,
                        cols=np.asarray(COLUMNS))
    return x
