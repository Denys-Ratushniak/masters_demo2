from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import config, metrics

EPS = 1e-6
Q_GRID: tuple[float, ...] = tuple(np.round(np.arange(0.0, 1.0, 0.05), 3)) + (0.97, 0.99)
STRATEGIES: dict[str, tuple[str, str, float]] = {
    "emp": ("none", "pooled", 1.0),
    "mean_loc": ("none", "mean", 1.0),
    "worst_loc": ("none", "worst", 1.0),
    "cert": ("none", "cert", 1.0),
    "platt": ("platt", "pooled", 1.0),
    "temp": ("temp", "pooled", 1.0),
    "iso": ("iso", "pooled", 1.0),
    "iso_mean_loc": ("iso", "mean", 1.0),
    "qnorm": ("qnorm", "pooled", 1.0),
    "qnorm_mean_loc": ("qnorm", "mean", 1.0),
    "fuse_pm": ("none", "fuse", 1.0),
    "inflate1.5": ("none", "pooled", 1.5),
    "inflate2": ("none", "pooled", 2.0),
    "deflate0.5": ("none", "pooled", 0.5),
}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _fit_platt(s_fit: np.ndarray, y_fit: np.ndarray) -> Callable:
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logit(s_fit)[:, None], y_fit)
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
    return lambda s, robot=None: 1.0 / (1.0 + np.exp(-(a * logit(s) + b)))


def _fit_temp(s_fit: np.ndarray, y_fit: np.ndarray) -> Callable:
    z, y = logit(s_fit), np.asarray(y_fit, float)

    def nll(lt: float) -> float:
        p = np.clip(1.0 / (1.0 + np.exp(-z / np.exp(lt))), EPS, 1 - EPS)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    t = float(np.exp(minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded").x))
    return lambda s, robot=None: 1.0 / (1.0 + np.exp(-logit(s) / t))


def _fit_iso(s_fit: np.ndarray, y_fit: np.ndarray) -> Callable:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(s_fit, float), np.asarray(y_fit, float))
    return lambda s, robot=None: iso.predict(np.asarray(s, float))


def _rank_map(ref: np.ndarray) -> Callable:
    ref = np.sort(np.asarray(ref, float))
    n = max(len(ref), 1)

    def f(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, float)
        return 0.5 * (np.searchsorted(ref, s, "left") + np.searchsorted(ref, s, "right")) / n

    return f


def _make_transform(kind: str, pool: dict, fit_mask: np.ndarray) -> Callable:
    s, y, robot = pool["score"], pool["y"], pool["robot"]
    if kind == "none":
        return lambda v, r=None: np.asarray(v, float)
    if kind == "qnorm":
        maps = {r: _rank_map(s[robot == r]) for r in np.unique(robot)}
        return lambda v, r: maps[r](v)
    fit = {"platt": _fit_platt, "temp": _fit_temp, "iso": _fit_iso}[kind]
    g = fit(s[fit_mask], y[fit_mask])
    return lambda v, r=None: g(v)


def aggregate_threshold(kind: str, y: np.ndarray, s: np.ndarray, robot: np.ndarray,
                        budget: float, delta: float = 0.05) -> float:
    if kind == "pooled":
        return metrics.threshold_at_ff(y, s, budget)
    if kind == "cert":
        return metrics.certified_threshold(y, s, budget, delta)
    return metrics.group_threshold(y, s, budget, robot, "max" if kind == "worst" else "mean")


def fuse_q(y: np.ndarray, s: np.ndarray, p_tp: np.ndarray, budget: float) -> tuple[float, float]:
    best = (-1.0, 0.0, np.inf)
    for q in Q_GRID:
        gated = np.where(p_tp >= q, s, -np.inf)
        t = metrics.threshold_at_ff(y, gated, budget)
        r = metrics.recall(y, gated, t)
        if r > best[0] + 1e-12:
            best = (r, float(q), float(t))
    return best[1], best[2]


def apply_strategy(pool: dict, strategy: str, budget: float) -> dict:
    tkind, akind, mult = STRATEGIES[strategy]
    s, y, robot = pool["score"], pool["y"], pool["robot"]
    b = float(np.clip(budget * mult, 1e-6, 0.99))
    flags = np.zeros(len(s), bool)
    thresholds: dict[str, float] = {}
    qs: dict[str, float] = {}
    for r in np.unique(robot):
        te = robot == r
        fit = ~te
        tf = _make_transform(tkind, pool, fit)
        sa = np.empty(len(s), float)
        for u in np.unique(robot):
            sa[robot == u] = tf(s[robot == u], u)
        if akind == "fuse":
            p_tp = pool["p_material"][:, 0]
            q, t = fuse_q(y[fit], sa[fit], p_tp[fit], b)
            flags[te] = (sa[te] >= t) & (p_tp[te] >= q)
            qs[str(r)] = q
        else:
            t = aggregate_threshold(akind, y[fit], sa[fit], robot[fit], b)
            flags[te] = sa[te] >= t
        thresholds[str(r)] = float(t)
    return {"flags": flags, "thresholds": thresholds, "q": qs, "fit_budget": b}


def fleet_thresholds(pool: dict, budgets: Sequence[float] = config.BUDGETS,
                     delta: float = 0.05) -> dict:
    y, s = pool["y"], pool["score"]
    out: dict = {"emp": {}, "cert": {}, "n_fit": int(y.size), "n_fit_neg": int((y == 0).sum())}
    for b in budgets:
        k = config.budget_key(b)
        out["emp"][k] = metrics.threshold_at_ff(y, s, b)
        out["cert"][k] = metrics.certified_threshold(y, s, b, delta)
        for strat in ("emp", "cert"):
            t = out[strat][k]
            out[f"fit_ff_{strat}_{k}"] = metrics.ff(y, s, t)
            out[f"fit_recall_{strat}_{k}"] = metrics.recall(y, s, t)
    return out


def score_flags(pool: dict, flags: np.ndarray, budget: float) -> dict:
    y, robot, mat = pool["y"], pool["robot"], pool["mat"]
    pos, neg = y == 1, y == 0
    ffs, recs = {}, {}
    for r in np.unique(robot):
        m = robot == r
        ffs[str(r)] = float(flags[m & neg].mean()) if (m & neg).any() else float("nan")
        recs[str(r)] = float(flags[m & pos].mean()) if (m & pos).any() else float("nan")
    v = np.array(list(ffs.values()), float)
    out = {"recall": float(flags[pos].mean()),
           "recall_robot_mean": float(np.nanmean(list(recs.values()))),
           "ff": float(flags[neg].mean()), "ff_mean": float(np.nanmean(v)),
           "ff_max": float(np.nanmax(v)),
           "within_budget": float(np.mean(v <= budget + 1e-9)),
           "ff_per_robot": ffs, "recall_per_robot": recs, "n_flush": int(flags.sum())}
    for m in config.MATERIALS:
        sel = neg & (mat == m)
        out[f"leak_{m}"] = float(flags[sel].mean()) if sel.any() else float("nan")
    for v_src in config.SOURCES:
        sel = neg & (pool["source"] == v_src)
        out[f"leak_source_{v_src}"] = float(flags[sel].mean()) if sel.any() else float("nan")
    return out


def operating_point(pool: dict, threshold: float, budget: float) -> dict:
    flags = pool["score"] >= threshold
    out = score_flags(pool, flags, budget)
    y = pool["y"]
    out["threshold"] = float(threshold)
    out["n_false_flush"] = int((flags & (y == 0)).sum())
    out["n_missed"] = int((~flags & (y == 1)).sum())
    out["accuracy"] = float((flags == (y == 1)).mean())
    out["recall_at_ff_oracle"] = metrics.recall_at_ff_oracle(y, pool["score"], budget)
    return out


def optimal_threshold_spread(pool: dict, budget: float = 0.05, min_neg: int = 5) -> dict:
    y, s, robot = pool["y"], pool["score"], pool["robot"]
    ts = {str(r): float(metrics.threshold_at_ff(y[robot == r], s[robot == r], budget))
          for r in np.unique(robot) if int((y[robot == r] == 0).sum()) >= min_neg}
    v = np.array(list(ts.values()), float)
    lg = logit(v)
    return {"per_robot": ts, "min": float(v.min()), "max": float(v.max()),
            "spread": float(v.max() - v.min()),
            "logit_spread": float(lg.max() - lg.min()),
            "pooled": float(metrics.threshold_at_ff(y, s, budget))}


def per_robot_table(pool: dict, threshold: float, min_n: int = 10,
                    other_key: str = "other_robots") -> dict:
    y, robot, s = pool["y"], pool["robot"], pool["score"]
    flags = s >= threshold
    out: dict = {}
    small: list[str] = []
    for r in np.unique(robot):
        m = robot == r
        row = {"n": int(m.sum()), "n_pos": int((m & (y == 1)).sum()),
               "n_neg": int((m & (y == 0)).sum()),
               "recall": float(flags[m & (y == 1)].mean()) if (m & (y == 1)).any() else float("nan"),
               "ff": float(flags[m & (y == 0)].mean()) if (m & (y == 0)).any() else float("nan"),
               "n_false_flush": int((flags & m & (y == 0)).sum())}
        if row["n"] >= min_n:
            out[str(r)] = row
        else:
            small.append(str(r))
    if small:
        m = np.isin(robot, small)
        out[other_key] = {
            "robots": small, "n": int(m.sum()), "n_pos": int((m & (y == 1)).sum()),
            "n_neg": int((m & (y == 0)).sum()),
            "recall": float(flags[m & (y == 1)].mean()) if (m & (y == 1)).any() else float("nan"),
            "ff": float(flags[m & (y == 0)].mean()) if (m & (y == 0)).any() else float("nan"),
            "n_false_flush": int((flags & m & (y == 0)).sum())}
    return out


def material_table(pool: dict, threshold: float) -> dict:
    y, mat, s = pool["y"], pool["mat"], pool["score"]
    flags = s >= threshold
    out = {}
    for m in config.MATERIALS:
        sel = mat == m
        out[m] = {
            "flushable_flush": int((sel & (y == 1) & flags).sum()),
            "flushable_bin": int((sel & (y == 1) & ~flags).sum()),
            "nonflushable_flush": int((sel & (y == 0) & flags).sum()),
            "nonflushable_bin": int((sel & (y == 0) & ~flags).sum())}
    return out


def distribution_block(pool: dict, hard: Sequence[str] = ("wrapper", "plastic", "other")) -> dict:
    y, mat, robot, group = pool["y"], pool["mat"], pool["robot"], pool["group"]
    neg = y == 0
    n_neg_per_robot = {str(r): int((neg & (robot == r)).sum()) for r in np.unique(robot)}
    counts = np.sort(np.array(list(n_neg_per_robot.values()), float))[::-1]
    total_neg = max(counts.sum(), 1.0)
    return {
        "n": int(y.size), "n_pos": int((y == 1).sum()), "n_neg": int(neg.sum()),
        "flush_rate": float((y == 1).mean()),
        "n_robots": int(np.unique(robot).size), "n_robot_days": int(np.unique(group).size),
        "material_share_all": {m: float((mat == m).mean()) for m in config.MATERIALS},
        "material_share_neg": {m: float((mat[neg] == m).mean()) for m in config.MATERIALS},
        "hard_share_neg": float(np.isin(mat[neg], list(hard)).mean()),
        "wipe_towel_share_neg": float(np.isin(mat[neg], ["wipe", "towel"]).mean()),
        "tp_share_neg": float((mat[neg] == "tp").mean()),
        "staged_share_all": float((pool["source"] == "staged").mean()),
        "staged_share_neg": float((pool["source"][neg] == "staged").mean()),
        "staged_share_pos": float((pool["source"][y == 1] == "staged").mean()),
        "neg_per_robot": n_neg_per_robot,
        "neg_concentration_top2": float(counts[:2].sum() / total_neg),
        "neg_concentration_top3": float(counts[:3].sum() / total_neg),
    }


def safety_curve(pool: dict) -> dict[str, list[float]]:
    return metrics.safety_curve(pool["y"], pool["score"])
