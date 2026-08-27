from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import config, data, metrics

SIZE_NAMES: tuple[str, ...] = config.SCALARS + ("support_rel_mm", "support_missing",
                                                "area_cm2_label", "bbox_px_area",
                                                "bbox_w", "bbox_h")
WITH_SUPPORT: tuple[str, ...] = tuple(n for n in SIZE_NAMES if not n.startswith("support"))
SET_A: tuple[str, ...] = ("v2_bbox_px", "v3_area_cm2", "v4_support", "v5_logsize")
SET_B: tuple[str, ...] = ("v2_bbox_px", "v3_area_cm2", "v5b_logsize_nosup")


@dataclass
class RuleConfig:
    delta: float = 0.05
    q_points: int = 120
    frontier_points: int = 96
    logreg_c: float = 1.0


def gate_grid(gate: np.ndarray, fit: np.ndarray, points: int) -> np.ndarray:
    q = np.quantile(gate[fit], np.linspace(0.0, 0.995, points))
    return np.unique(np.concatenate([[-np.inf], q]))


def best_threshold(y: np.ndarray, inner: np.ndarray, sel: np.ndarray, budget: float,
                   crit: str, n_neg: int, n_pos: int, k_cert: int) -> tuple[float, float, float]:
    idx = np.flatnonzero(sel)
    if idx.size == 0:
        return float("inf"), 0.0, 0.0
    v = inner[idx]
    order = np.argsort(-v, kind="mergesort")
    idx, v = idx[order], v[order]
    c_neg = np.cumsum(y[idx] == 0)
    c_pos = np.cumsum(y[idx] == 1)
    cand = np.flatnonzero(np.r_[v[1:] != v[:-1], np.array([True])])
    limit = budget * n_neg + 1e-9
    if crit == "cert":
        limit = min(float(k_cert), limit)
    ok = cand[c_neg[cand] <= limit]
    if ok.size == 0:
        return float("inf"), 0.0, 0.0
    j = int(ok[-1])
    return float(v[j]), float(c_pos[j] / max(n_pos, 1)), float(c_neg[j] / max(n_neg, 1))


def fit_pair(y: np.ndarray, gate: np.ndarray, inner: np.ndarray, valid: np.ndarray,
             grid: np.ndarray, budget: float, crit: str,
             delta: float) -> tuple[float, float, float, float]:
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    k_cert = metrics.certified_count(n_neg, budget, delta) if crit == "cert" else 0
    best_key = (-1.0, -np.inf, -np.inf)
    best = (-np.inf, float("inf"), 0.0, 0.0)
    for q in grid:
        sel = valid & (gate >= q)
        t, rec, f = best_threshold(y, inner, sel, budget, crit, n_neg, n_pos, k_cert)
        key = (rec, -f, float(q))
        if key > best_key:
            best_key, best = key, (float(q), t, rec, f)
    return best


def variant_bank(box: dict, p_size: np.ndarray, p_size_nosup: np.ndarray,
                 every: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {"v2_bbox_px": (-box["bbox_area"], np.isfinite(box["bbox_area"])),
            "v3_area_cm2": (-box["area_cm2"], np.isfinite(box["area_cm2"])),
            "v4_support": (-box["support"], np.isfinite(box["support"])),
            "v5_logsize": (p_size, every),
            "v5b_logsize_nosup": (p_size_nosup, every)}


def size_model(feats: np.ndarray, mat: np.ndarray, y: np.ndarray, fit: np.ndarray,
               c_reg: float) -> tuple[np.ndarray, dict]:
    tp = fit & (mat == "tp")
    stats = data.fit_scaler(feats[tp])
    z = data.apply_scaler(feats, stats)
    model = LogisticRegression(C=c_reg, max_iter=5000)
    model.fit(z[tp], y[tp])
    p = model.predict_proba(z)[:, 1]
    return p.astype(float), {"n_fit": int(tp.sum()), "auroc_fit": float(roc_auc_score(y[tp], p[tp]))}


def size_inputs(ids: np.ndarray) -> dict:
    items = data.load_pool()
    items = data.attach(items, data.load_cache(config.DEFAULT_CROP, ("sc",)), ("sc",))
    pos = {str(v): i for i, v in enumerate(items.ids)}
    idx = np.array([pos[str(v)] for v in ids], np.int64)
    raw = data.raw_scalars(items, SIZE_NAMES)[idx]
    return {"feats": raw.astype(np.float32),
            "keep": np.array([n in WITH_SUPPORT for n in SIZE_NAMES]),
            "bbox_area": raw[:, SIZE_NAMES.index("bbox_px_area")].astype(float),
            "area_cm2": items.area_cm2[idx].astype(float),
            "support": items.support_rel_mm[idx].astype(float),
            "mat": items.mat[idx]}


def build_banks(box: dict, y: np.ndarray, robot: np.ndarray, cfg: RuleConfig,
                with_inner: bool = False) -> tuple[dict, dict, dict, dict]:
    feats, keep, mat = box["feats"], box["keep"], box["mat"]
    every = np.ones(len(y), bool)
    bank: dict = {}
    bank_inner: dict = {}
    p_size: dict = {}
    info: dict = {}
    for r in np.unique(robot):
        fit = robot != r
        po, io = size_model(feats, mat, y, fit, cfg.logreg_c)
        pn, inn = size_model(feats[:, keep], mat, y, fit, cfg.logreg_c)
        p_size[str(r)] = po
        bank[r] = variant_bank(box, po, pn, every)
        info[str(r)] = {"outer": io, "outer_nosup": inn, "inner": {}}
        if not with_inner:
            continue
        bank_inner[r] = {}
        for u in np.unique(robot):
            if u == r:
                continue
            sub = fit & (robot != u)
            pou, iou = size_model(feats, mat, y, sub, cfg.logreg_c)
            pnu, inu = size_model(feats[:, keep], mat, y, sub, cfg.logreg_c)
            bank_inner[r][u] = variant_bank(box, pou, pnu, every)
            info[str(r)]["inner"][str(u)] = {"outer": iou, "outer_nosup": inu}
    return bank, bank_inner, p_size, info


def nested_rule(y: np.ndarray, robot: np.ndarray, gate: np.ndarray, bank: dict,
                variants: Sequence[str], budget: float, crit: str,
                cfg: RuleConfig) -> tuple[np.ndarray, dict]:
    flags = np.zeros(len(y), bool)
    picks: dict[str, dict] = {}
    for r in np.unique(robot):
        te = robot == r
        fit = ~te
        grid = gate_grid(gate, fit, cfg.q_points)
        best = None
        for v in variants:
            inner, valid = bank[r][v]
            q, t, rec, ff = fit_pair(y[fit], gate[fit], inner[fit], valid[fit], grid,
                                     budget, crit, cfg.delta)
            key = (rec, -ff)
            if best is None or key > best[0]:
                best = (key, v, q, t, rec, ff)
        _, v, q, t, rec, ff = best
        inner, valid = bank[r][v]
        flags[te] = valid[te] & (gate[te] >= q) & (inner[te] >= t)
        picks[str(r)] = {"variant": v, "q": q, "t": t, "fit_recall": rec, "fit_ff": ff}
    return flags, picks


def nested_cv_rule(y: np.ndarray, robot: np.ndarray, gate: np.ndarray, bank: dict,
                   bank_inner: dict, variants: Sequence[str], budget: float, crit: str,
                   cfg: RuleConfig) -> tuple[np.ndarray, dict]:
    flags = np.zeros(len(y), bool)
    picks: dict[str, dict] = {}
    for r in np.unique(robot):
        te = robot == r
        fit = ~te
        best = None
        for v in variants:
            inner_flags = np.zeros(len(y), bool)
            for u in np.unique(robot):
                if u == r:
                    continue
                is_u = robot == u
                sub = fit & ~is_u
                inner, valid = bank_inner[r][u][v]
                q, t, _, _ = fit_pair(y[sub], gate[sub], inner[sub], valid[sub],
                                      gate_grid(gate, sub, cfg.q_points), budget, crit,
                                      cfg.delta)
                inner_flags[is_u] = valid[is_u] & (gate[is_u] >= q) & (inner[is_u] >= t)
            rec = float(inner_flags[fit & (y == 1)].mean())
            ff = float(inner_flags[fit & (y == 0)].mean())
            key = (ff <= budget + 1e-9, rec, -ff)
            if best is None or key > best[0]:
                best = (key, v, rec, ff)
        _, v, rec_in, ff_in = best
        inner, valid = bank[r][v]
        q, t, rec, ff = fit_pair(y[fit], gate[fit], inner[fit], valid[fit],
                                 gate_grid(gate, fit, cfg.q_points), budget, crit, cfg.delta)
        flags[te] = valid[te] & (gate[te] >= q) & (inner[te] >= t)
        picks[str(r)] = {"variant": v, "q": q, "t": t, "inner_recall": rec_in,
                         "inner_ff": ff_in, "fit_recall": rec, "fit_ff": ff}
    return flags, picks


def fixed_rule(y: np.ndarray, robot: np.ndarray, gate: np.ndarray, bank: dict, variant: str,
               budget: float, crit: str, cfg: RuleConfig) -> tuple[np.ndarray, dict]:
    return nested_rule(y, robot, gate, bank, (variant,), budget, crit, cfg)


def gate_rule(y: np.ndarray, robot: np.ndarray, gate: np.ndarray, budget: float, crit: str,
              cfg: RuleConfig) -> tuple[np.ndarray, dict]:
    every = np.ones(len(y), bool)
    bank = {r: {"v1_gate_only": (gate, every)} for r in np.unique(robot)}
    return nested_rule(y, robot, gate, bank, ("v1_gate_only",), budget, crit, cfg)


def two_stage_score(gate: np.ndarray, size: np.ndarray, robot: np.ndarray,
                    q_per_robot: dict[str, float]) -> np.ndarray:
    keep = np.zeros(len(gate), bool)
    for r in np.unique(robot):
        m = robot == r
        keep[m] = gate[m] >= q_per_robot[str(r)]
    return np.where(keep, size, -np.inf)


def nested_score(gate: np.ndarray, robot: np.ndarray, bank: dict, picks: dict) -> np.ndarray:
    out = np.full(len(gate), -np.inf, float)
    for r in np.unique(robot):
        m = robot == r
        pick = picks[str(r)]
        inner, valid = bank[r][pick["variant"]]
        keep = m & valid & (gate >= pick["q"])
        out[keep] = inner[keep]
    return out
