from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy.stats import beta, binom
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config

THRESHOLD_STRATEGIES: tuple[str, ...] = ("emp", "halved", "cert", "worst_loc", "mean_loc")


def ff(y: np.ndarray, s: np.ndarray, t: float) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    n = int((y == 0).sum())
    return float(((s >= t) & (y == 0)).sum() / n) if n else float("nan")


def recall(y: np.ndarray, s: np.ndarray, t: float) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    n = int((y == 1).sum())
    return float(((s >= t) & (y == 1)).sum() / n) if n else float("nan")


def threshold_at_ff(y: np.ndarray, s: np.ndarray, budget: float) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    neg = np.sort(s[y == 0])
    cand = np.unique(s)
    if neg.size == 0:
        return float(cand[0])
    admitted = neg.size - np.searchsorted(neg, cand, side="left")
    ok = np.flatnonzero(admitted <= budget * neg.size + 1e-9)
    if ok.size:
        return float(cand[ok[0]])
    return float(np.nextafter(cand[-1], np.inf))


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))


def min_n_to_certify(eps: float, delta: float = 0.05) -> int:
    return int(math.ceil(math.log(delta) / math.log(1.0 - eps)))


def certified_count(n: int, eps: float, delta: float = 0.05) -> int:
    best = 0
    for k in range(1, int(n) + 1):
        if binom.cdf(k - 1, n, eps) <= delta:
            best = k
        else:
            break
    return best


def certified_threshold(y: np.ndarray, s: np.ndarray, eps: float, delta: float = 0.05) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    sb = np.sort(s[y == 0])[::-1]
    n = sb.size
    if n == 0:
        return float(np.nextafter(s.max(), np.inf))
    best = certified_count(n, eps, delta)
    if not best:
        return float(np.nextafter(s.max(), np.inf))
    return float(max(np.nextafter(sb[best - 1], np.inf), threshold_at_ff(y, s, eps)))


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(np.asarray(y), np.asarray(s, float)))


def auprc_bin(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(1 - np.asarray(y), -np.asarray(s, float)))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(np.asarray(y), np.asarray(s, float)))


def recall_at_ff_oracle(y: np.ndarray, s: np.ndarray, budget: float) -> float:
    return recall(y, s, threshold_at_ff(y, s, budget))


def group_threshold(y: np.ndarray, s: np.ndarray, budget: float, groups: np.ndarray,
                    agg: str, min_neg: int = 5) -> float:
    ts = [threshold_at_ff(y[groups == g], s[groups == g], budget)
          for g in np.unique(groups) if int((y[groups == g] == 0).sum()) >= min_neg]
    if not ts:
        return threshold_at_ff(y, s, budget)
    return float(max(ts) if agg == "max" else np.mean(ts))


def fit_threshold(y_fit: np.ndarray, s_fit: np.ndarray, budget: float, strategy: str = "emp",
                  groups: np.ndarray | None = None, delta: float = 0.05) -> float:
    y_fit, s_fit = np.asarray(y_fit), np.asarray(s_fit, float)
    if strategy == "emp":
        return threshold_at_ff(y_fit, s_fit, budget)
    if strategy == "halved":
        return threshold_at_ff(y_fit, s_fit, budget / 2.0)
    if strategy == "cert":
        return certified_threshold(y_fit, s_fit, budget, delta)
    if strategy in ("worst_loc", "mean_loc"):
        if groups is None:
            return threshold_at_ff(y_fit, s_fit, budget)
        return group_threshold(y_fit, s_fit, budget, np.asarray(groups),
                               "max" if strategy == "worst_loc" else "mean")
    raise ValueError(strategy)


def eval_transferred(y_te: np.ndarray, s_te: np.ndarray, y_fit: np.ndarray, s_fit: np.ndarray,
                     budgets: Sequence[float] = config.BUDGETS, strategy: str = "emp",
                     groups_fit: np.ndarray | None = None) -> dict:
    out: dict[str, dict[str, float]] = {"threshold_used": {}, "recall_at_ff": {}, "ff_actual": {}}
    for b in budgets:
        k = config.budget_key(b)
        t = fit_threshold(y_fit, s_fit, b, strategy, groups_fit)
        out["threshold_used"][k] = t
        out["recall_at_ff"][k] = recall(y_te, s_te, t)
        out["ff_actual"][k] = ff(y_te, s_te, t)
    return out


def rate_by(y: np.ndarray, s: np.ndarray, key: np.ndarray, t: float,
            values: Sequence[str]) -> dict[str, float]:
    y, s, key = np.asarray(y), np.asarray(s, float), np.asarray(key)
    out = {}
    for v in values:
        m = (key == v) & (y == 0)
        out[str(v)] = float((s[m] >= t).mean()) if m.any() else float("nan")
    return out


def ff_per_material(y: np.ndarray, s: np.ndarray, mat: np.ndarray, t: float,
                    classes: Sequence[str] = config.MATERIALS) -> dict[str, float]:
    return rate_by(y, s, mat, t, classes)


def ff_by_source(y: np.ndarray, s: np.ndarray, source: np.ndarray, t: float,
                 values: Sequence[str] = config.SOURCES) -> dict[str, float]:
    return rate_by(y, s, source, t, values)


def summarise(per_fold: list[dict], keys: Sequence[str] | None = None,
              reference: list[dict] | None = None) -> dict:
    if not per_fold:
        return {}
    if keys is None:
        keys = [k for k, v in per_fold[0].items() if isinstance(v, (int, float))]
    out: dict = {"n": len(per_fold), "mean": {}, "sd": {}, "per_fold": {}, "wins": {}}
    for k in keys:
        v = np.array([float(r.get(k, np.nan)) for r in per_fold], float)
        out["mean"][k] = float(np.nanmean(v))
        out["sd"][k] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
        out["per_fold"][k] = [float(x) for x in v]
        if reference is not None:
            r = np.array([float(x.get(k, np.nan)) for x in reference], float)
            out["wins"][k] = int(np.nansum(v > r))
    return out


def fold_mean(rows: list[dict], key: Callable[[dict], float]) -> np.ndarray:
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["fold"], []).append(float(key(r)))
    return np.array([np.mean(v) for _, v in sorted(by.items())], float)


def mean_sd(v: np.ndarray) -> str:
    v = np.asarray(v, float)
    sd = float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0
    return f"{np.nanmean(v):.3f}±{sd:.3f}"


def group_bootstrap_index(groups: np.ndarray, draws: int, seed: int):
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in ug}
    for _ in range(draws):
        yield np.concatenate([by[g] for g in rng.choice(ug, len(ug), replace=True)])


def group_bootstrap_sets(groups: np.ndarray, y: np.ndarray, draws: int = 1000,
                         seed: int = 0, min_each: int = 2) -> list[np.ndarray]:
    groups, y = np.asarray(groups), np.asarray(y)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in ug}
    out: list[np.ndarray] = []
    while len(out) < draws:
        sel = np.concatenate([by[g] for g in rng.choice(ug, len(ug), replace=True)])
        yy = y[sel]
        if int((yy == 1).sum()) >= min_each and int((yy == 0).sum()) >= min_each:
            out.append(sel)
    return out


def bootstrap_ci(y: np.ndarray, s: np.ndarray, groups: np.ndarray,
                 fn: Callable[[np.ndarray, np.ndarray], float],
                 draws: int = 2000, seed: int = 0) -> dict:
    y, s = np.asarray(y), np.asarray(s, float)
    vals = [float(fn(y[sel], s[sel])) for sel in group_bootstrap_index(groups, draws, seed)
            if np.unique(y[sel]).size >= 2]
    if not vals:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean": float(np.mean(vals)), "lo": float(lo), "hi": float(hi), "n": len(vals)}


def paired_bootstrap(a: dict, b: dict, groups: np.ndarray, draws: int = 2000, seed: int = 0,
                     fn: Callable[[np.ndarray, np.ndarray], float] = auprc) -> dict:
    y = np.asarray(a["y"])
    sa, sb = np.asarray(a["s"], float), np.asarray(b["s"], float)
    deltas = [float(fn(y[sel], sa[sel]) - fn(y[sel], sb[sel]))
              for sel in group_bootstrap_index(groups, draws, seed)
              if np.unique(y[sel]).size >= 2]
    if not deltas:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_gt0": float("nan"), "n": 0}
    d = np.array(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(np.mean(d)), "lo": float(lo), "hi": float(hi),
            "p_gt0": float((d > 0).mean()), "n": int(d.size)}


def paired_bootstrap_flags(y: np.ndarray, groups: np.ndarray, fa: np.ndarray, fb: np.ndarray,
                           draws: int = 2000, seed: int = 0,
                           sets: list[np.ndarray] | None = None) -> dict:
    y, fa, fb = np.asarray(y), np.asarray(fa), np.asarray(fb)
    if sets is None:
        sets = [sel for sel in group_bootstrap_index(groups, draws, seed)
                if int((y[sel] == 1).sum()) >= 2 and int((y[sel] == 0).sum()) >= 2]
    dr, df = [], []
    for sel in sets:
        yy = y[sel]
        p, n = yy == 1, yy == 0
        dr.append(float(fa[sel][p].mean() - fb[sel][p].mean()))
        df.append(float(fa[sel][n].mean() - fb[sel][n].mean()))
    if not dr:
        return {"d_recall": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_gt0": float("nan"), "d_ff": float("nan"), "ff_lo": float("nan"),
                "ff_hi": float("nan"), "n": 0}
    a, b = np.array(dr), np.array(df)
    lo, hi = np.percentile(a, [2.5, 97.5])
    flo, fhi = np.percentile(b, [2.5, 97.5])
    return {"d_recall": float(a.mean()), "lo": float(lo), "hi": float(hi),
            "p_gt0": float((a > 0).mean()), "d_ff": float(b.mean()),
            "ff_lo": float(flo), "ff_hi": float(fhi), "n": int(a.size)}


def recall_at_ff_targets(y: np.ndarray, s: np.ndarray,
                         targets: Sequence[float]) -> np.ndarray:
    y, s = np.asarray(y), np.asarray(s, float)
    npos = max(int((y == 1).sum()), 1)
    nneg = max(int((y == 0).sum()), 1)
    keep = np.isfinite(s)
    yy, ss = y[keep], s[keep]
    if yy.size == 0:
        return np.zeros(len(targets), float)
    order = np.argsort(-ss, kind="mergesort")
    yy = yy[order]
    tp = np.cumsum(yy == 1) / npos
    fp = np.cumsum(yy == 0) / nneg
    last = np.r_[fp[1:] != fp[:-1], True]
    xs = np.r_[0.0, fp[last]]
    rs = np.r_[0.0, tp[last]]
    uniq = np.r_[np.diff(xs) > 0, True]
    return np.interp(np.asarray(targets, float), xs[uniq], rs[uniq])


def safety_curve(y: np.ndarray, s: np.ndarray) -> dict[str, list[float]]:
    y, s = np.asarray(y), np.asarray(s, float)
    ys = y[np.argsort(-s, kind="mergesort")]
    npos, nneg = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    return {"ff": (np.cumsum(ys == 0) / nneg).tolist(),
            "recall": (np.cumsum(ys == 1) / npos).tolist()}


def fit_trend(n: np.ndarray, v: np.ndarray, target: float) -> dict:
    n = np.asarray(n, float)
    v = np.clip(np.asarray(v, float), 1e-6, None)
    out: dict = {}
    lg = np.log(n)
    for kind, yy, back in (("power", np.log(v), np.exp), ("log", v, lambda z: z)):
        b, a = np.polyfit(lg, yy, 1)
        pred = a + b * lg
        ss = float(np.sum((yy - pred) ** 2))
        tot = float(np.sum((yy - yy.mean()) ** 2))
        out[kind] = {"a": float(a), "b": float(b),
                     "r2": 1.0 - ss / tot if tot > 0 else float("nan"),
                     "pred_target": float(back(a + b * np.log(target)))}
    return out
