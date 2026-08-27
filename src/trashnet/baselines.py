from __future__ import annotations

import numpy as np

from .data import Items

FLOOR = -1.0e9
RULES: tuple[str, ...] = ("flush_all", "oracle_tp", "oracle_tp_area_px", "oracle_tp_area_cm2",
                          "oracle_tp_support")


def bbox_wh(items: Items) -> np.ndarray:
    b = np.asarray(items.bbox, np.float32)
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    return np.stack([w, h, w * h], 1)


def bbox_area_px(items: Items) -> np.ndarray:
    return bbox_wh(items)[:, 2]


def _gated(mask: np.ndarray, quantity: np.ndarray) -> np.ndarray:
    s = np.full(len(mask), FLOOR, np.float64)
    ok = mask & np.isfinite(quantity)
    s[ok] = -np.asarray(quantity, np.float64)[ok]
    return s


def rule_score(name: str, items: Items) -> np.ndarray:
    tp = items.mat == "tp"
    if name == "flush_all":
        return np.ones(len(items), np.float64)
    if name == "oracle_tp":
        return tp.astype(np.float64)
    if name == "oracle_tp_area_px":
        return _gated(tp, bbox_area_px(items))
    if name == "oracle_tp_area_cm2":
        return _gated(tp, np.asarray(items.area_cm2, np.float64))
    if name == "oracle_tp_support":
        return _gated(tp, np.asarray(items.support_rel_mm, np.float64))
    raise ValueError(name)


def native_point(name: str, items: Items, idx: np.ndarray) -> tuple[float, float]:
    s = rule_score(name, items)[idx]
    y = items.y[idx]
    admitted = np.ones(len(idx), bool) if name == "flush_all" else s > FLOOR
    if name == "oracle_tp":
        admitted = s > 0.5
    pos, neg = y == 1, y == 0
    return (float(admitted[pos].mean()) if pos.any() else float("nan"),
            float(admitted[neg].mean()) if neg.any() else float("nan"))
