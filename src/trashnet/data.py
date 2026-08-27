from __future__ import annotations

import datetime as dt
import inspect
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from . import config


@dataclass
class Items:
    ids: np.ndarray
    y: np.ndarray
    mat: np.ndarray
    robot: np.ndarray
    group: np.ndarray
    source: np.ndarray
    sample: np.ndarray
    wet: np.ndarray
    bbox: np.ndarray
    area_cm2: np.ndarray
    support_rel_mm: np.ndarray
    box_quality: np.ndarray
    rgb: np.ndarray | None = None
    geo: np.ndarray | None = None
    sc: np.ndarray | None = None

    def __len__(self) -> int:
        return int(len(self.ids))

    def subset(self, mask: np.ndarray) -> "Items":
        idx = np.asarray(mask)
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = None if v is None else v[idx]
        return Items(**out)


def load_labels(path: Path | str | None = None) -> list[dict]:
    src = Path(path) if path is not None else config.LABELS
    with open(src) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return sorted(rows, key=lambda r: r["id"])


def _material(row: dict) -> str:
    m = row.get("trash_type") or "other"
    return m if m in config.MATERIALS else "other"


def _float_or_nan(row: dict, key: str) -> float:
    v = row.get(key)
    return float(v) if v is not None else float("nan")


def build_items(rows: list[dict]) -> Items:
    def col(fn, dtype):
        return np.array([fn(r) for r in rows], dtype=dtype)

    return Items(
        ids=col(lambda r: r["id"], object).astype(str),
        y=col(lambda r: int(r["y"]), np.int64),
        mat=col(_material, object).astype(str),
        robot=col(lambda r: r["group"].split("_")[0], object).astype(str),
        group=col(lambda r: r["group"], object).astype(str),
        source=col(lambda r: r["source"], object).astype(str),
        sample=col(lambda r: r.get("sample", ""), object).astype(str),
        wet=col(lambda r: bool(r.get("wet")), bool),
        bbox=np.array([[float(v) for v in r["bbox_xyxy"]] for r in rows], np.float32),
        area_cm2=col(lambda r: _float_or_nan(r, "area_cm2"), np.float32),
        support_rel_mm=col(lambda r: _float_or_nan(r, "support_rel_mm"), np.float32),
        box_quality=col(lambda r: r.get("box_quality", ""), object).astype(str),
    )


def load_all() -> Items:
    return build_items(load_labels())


def load_pool() -> Items:
    items = load_all()
    return items.subset(np.isin(items.robot, config.TRAIN_ROBOTS))


def load_test(reason: str) -> Items:
    frame = inspect.stack()[1]
    config.TEST_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(config.TEST_ACCESS_LOG, "a") as fh:
        fh.write(f"{stamp}\t{reason}\t{frame.filename}:{frame.lineno}\n")
    items = load_all()
    return items.subset(~np.isin(items.robot, config.TRAIN_ROBOTS))


def load_cache(name: str = config.DEFAULT_CROP,
               keys: Sequence[str] = ("rgb", "sc")) -> dict[str, np.ndarray]:
    path = config.cache_path(name)
    assert path.exists(), f"crop cache missing: {path}"
    with np.load(path, allow_pickle=True) as d:
        out = {k: d[k] for k in set(keys) | {"ids"}}
    out["ids"] = out["ids"].astype(str)
    return out


def attach(items: Items, cache: dict[str, np.ndarray],
           keys: Sequence[str] = ("rgb", "geo", "sc")) -> Items:
    pos = {k: i for i, k in enumerate(cache["ids"])}
    missing = [i for i in items.ids if i not in pos]
    assert not missing, f"{len(missing)} ids missing from cache, first={missing[:3]}"
    sel = np.array([pos[i] for i in items.ids], np.int64)
    return replace(items, **{k: cache[k][sel] for k in keys if k in cache})


def prep_rgb(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, np.float32)[..., ::-1] / 255.0
    mean = np.asarray(config.IMAGENET_MEAN, np.float32)
    std = np.asarray(config.IMAGENET_STD, np.float32)
    x = (x - mean) / std
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def geo_channels(in_chans: int, channels: Sequence[int] | None = None,
                 use_rgb: bool = True) -> list[int]:
    if channels is not None:
        return [int(c) for c in channels]
    if not use_rgb:
        return [0, 1, 2]
    return [1] if in_chans == 4 else list(range(in_chans - 3))


def prep_input(rgb: np.ndarray, geo: np.ndarray | None = None, in_chans: int = 3,
               channels: Sequence[int] | None = None, use_rgb: bool = True,
               chunk: int = 256) -> np.ndarray:
    ch = geo_channels(in_chans, channels, use_rgb)
    assert in_chans == (3 if use_rgb else 0) + len(ch), (in_chans, ch, use_rgb)
    assert not ch or geo is not None, "geometry channels asked for without the geo stack"
    n = len(rgb) if use_rgb else len(geo)
    px = rgb.shape[1] if use_rgb else geo.shape[-1]
    out = np.empty((n, in_chans, px, px), np.float16)
    for i in range(0, n, chunk):
        parts = []
        if use_rgb:
            parts.append(prep_rgb(rgb[i:i + chunk]).astype(np.float32))
        if ch:
            parts.append(prep_geo(geo[i:i + chunk])[:, ch].astype(np.float32))
        out[i:i + chunk] = np.concatenate(parts, 1).astype(np.float16)
    return out


def prep_geo(geo: np.ndarray, chunk: int = 256) -> np.ndarray:
    n = len(geo)
    px = geo.shape[-1]
    out = np.empty((n, 3, px, px), np.float16)
    mean = np.asarray(config.IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(config.IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)
    for i in range(0, n, chunk):
        g = np.asarray(geo[i:i + chunk], np.float32)
        g[:, 0] = np.clip(g[:, 0], 0.0, 1.0)
        g[:, 1] = np.clip((g[:, 1] + 1.0) / 4.0, 0.0, 1.0)
        g[:, 2] = np.clip(g[:, 2], 0.0, 1.0)
        out[i:i + chunk] = ((g - mean) / std).astype(np.float16)
    return out


def material_index(mat: np.ndarray, classes: Sequence[str] = config.MATERIALS) -> np.ndarray:
    if tuple(classes) == config.MATERIALS3:
        mat = np.array([config.MATERIAL_TO3[m] for m in mat])
    lut = {c: i for i, c in enumerate(classes)}
    return np.array([lut[m] for m in mat], np.int64)


def fit_scaler(x: np.ndarray) -> dict[str, np.ndarray]:
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isfinite(x), x, med)
    return {"median": med.astype(np.float32),
            "mean": filled.mean(0).astype(np.float32),
            "std": np.maximum(filled.std(0), 1e-6).astype(np.float32)}


def apply_scaler(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    filled = np.where(np.isfinite(x), x, stats["median"])
    return ((filled - stats["mean"]) / stats["std"]).astype(np.float32)


def raw_scalars(items: Items, names: Sequence[str]) -> np.ndarray:
    cols = []
    for nm in names:
        if nm == "support_rel_mm":
            cols.append(items.support_rel_mm.astype(np.float32))
        elif nm == "support_missing":
            cols.append((~np.isfinite(items.support_rel_mm)).astype(np.float32))
        elif nm == "area_cm2_label":
            cols.append(items.area_cm2.astype(np.float32))
        elif nm == "bbox_w":
            cols.append((items.bbox[:, 2] - items.bbox[:, 0]).astype(np.float32))
        elif nm == "bbox_h":
            cols.append((items.bbox[:, 3] - items.bbox[:, 1]).astype(np.float32))
        elif nm == "bbox_px_area":
            cols.append(((items.bbox[:, 2] - items.bbox[:, 0])
                         * (items.bbox[:, 3] - items.bbox[:, 1])).astype(np.float32))
        else:
            assert items.sc is not None, "descriptor scalars are not attached"
            cols.append(items.sc[:, config.SCALARS.index(nm)].astype(np.float32))
    return np.stack(cols, 1) if cols else np.zeros((len(items), 0), np.float32)


def scalar_matrix(items: Items, names: Sequence[str],
                  fit_idx: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = raw_scalars(items, names)
    stats = fit_scaler(raw[fit_idx])
    return apply_scaler(raw, stats), stats
