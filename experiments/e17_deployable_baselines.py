from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from trashnet import baselines, colour, config, data, evaluate, features, metrics, transfer
from trashnet.folds import inner_folds, loro_folds

EXP = "e17"
SEEDS: tuple[int, ...] = (0, 1, 2)
BUDGETS: tuple[float, ...] = (0.02, 0.05, 0.10)
BUDGET = 0.05
C_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
FROZEN_BACKBONE = "convnext_tiny"
DEPTH_COLUMNS: tuple[str, ...] = config.SCALARS
RF_COLUMNS: tuple[str, ...] = DEPTH_COLUMNS + ("bbox_w", "bbox_h") + colour.COLUMNS[:6]
RF_KWARGS: dict = {"n_estimators": 300, "n_jobs": -1, "min_samples_leaf": 2,
                   "class_weight": "balanced_subsample"}
QUESTION = ("what a deployable method with no label at inference reaches, so the gap the "
            "proposed model has to justify is measured and not asserted")
RULES: tuple[tuple[str, str], ...] = (
    ("area_rule", "flush the smallest bounding boxes first"),
    ("height_rule", "flush the flattest objects first, null support never flushes"),
    ("bright_rule_white", "flush the whitest objects first"),
    ("bright_rule_mean", "flush the brightest objects first"),
)
REUSED: tuple[tuple[str, str, str, tuple[int, ...]], ...] = (
    ("flush_all", "e1", "a_flush_all", (0,)),
    ("logreg_depth", "e1", "f_logreg_scalars", (0, 1, 2)),
    ("tinycnn", "e1", "h_tinycnn", (0,)),
)
AGG_KEYS: tuple[str, ...] = ("auprc", "matched_ff5", "recall_ff5_emp", "ff_ff5_emp",
                             "within_budget_ff5_emp", "recall_ff5_cert", "ff_ff5_cert")


@dataclass
class BaselineConfig:
    seeds: tuple[int, ...] = SEEDS
    budgets: tuple[float, ...] = BUDGETS
    budget: float = BUDGET
    with_frozen: bool = True


def as_pool(items: data.Items, score: np.ndarray) -> dict:
    return {"ids": items.ids, "score": np.asarray(score, float), "y": items.y.astype(int),
            "mat": items.mat, "group": items.group, "source": items.source,
            "robot": items.robot}


def rule_score(name: str, items: data.Items, tint: np.ndarray) -> np.ndarray:
    if name == "area_rule":
        return -baselines.bbox_area_px(items).astype(float)
    if name == "height_rule":
        v = items.support_rel_mm.astype(float)
        return np.where(np.isfinite(v), -v, baselines.FLOOR)
    index = colour.COLUMNS.index("whiteness" if name == "bright_rule_white" else "brightness")
    return tint[:, index].astype(float)


def fold_local(items: data.Items, x: np.ndarray, seed: int, fit: str) -> np.ndarray:
    score = np.zeros(len(items), float)
    picks: dict[str, float] = {}
    for fold in loro_folds(items):
        stats = data.fit_scaler(x[fold.train])
        z = data.apply_scaler(x, stats)
        if fit == "rf":
            model = RandomForestClassifier(random_state=seed, **RF_KWARGS)
        else:
            model = LogisticRegression(C=pick_c(items, z, fold, seed), max_iter=3000,
                                       class_weight="balanced")
            picks[fold.name] = float(model.C)
        model.fit(z[fold.train], items.y[fold.train])
        column = list(model.classes_).index(1)
        score[fold.test] = model.predict_proba(z[fold.test])[:, column]
    return score, picks


def pick_c(items: data.Items, z: np.ndarray, fold, seed: int) -> float:
    best = (-np.inf, C_GRID[0])
    for c in C_GRID:
        vals = []
        for tr, va in inner_folds(items, fold.train, seed=seed):
            model = LogisticRegression(C=c, max_iter=3000, class_weight="balanced")
            model.fit(z[tr], items.y[tr])
            column = list(model.classes_).index(1)
            vals.append(metrics.auprc(items.y[va], model.predict_proba(z[va])[:, column]))
        score = float(np.mean(vals))
        if score > best[0]:
            best = (score, c)
    return best[1]


def frozen_features(items: data.Items) -> np.ndarray:
    def images() -> np.ndarray:
        it = data.attach(items, data.load_cache(config.DEFAULT_CROP, ("rgb",)), ("rgb",))
        return data.prep_input(it.rgb, None, 3, None, True)

    return features.embed_cached(f"{config.DEFAULT_CROP}_c3", FROZEN_BACKBONE, images,
                                 len(items))


def summarise(pool: dict, cfg: BaselineConfig) -> dict:
    y, s = pool["y"], pool["score"]
    out: dict = {"auprc": metrics.auprc(y, s), "auroc": metrics.auroc(y, s),
                 "n_admissible": int(np.sum(np.isfinite(s) & (s > baselines.FLOOR))),
                 "matched": {}, "transferred": {}}
    for b in cfg.budgets:
        bk = config.budget_key(b)
        out["matched"][bk] = float(metrics.recall_at_ff_targets(y, s, [b])[0])
        out["transferred"][bk] = {}
        for crit in ("emp", "cert"):
            applied = transfer.apply_strategy(pool, crit, b)
            entry = transfer.score_flags(pool, applied["flags"], b)
            entry["thresholds"] = applied["thresholds"]
            out["transferred"][bk][crit] = entry
    return out


def aggregate_seeds(rows: list[dict], cfg: BaselineConfig) -> dict:
    key = config.budget_key(cfg.budget)
    table = {"auprc": [r["auprc"] for r in rows],
             "matched_ff5": [r["matched"][key] for r in rows],
             "recall_ff5_emp": [r["transferred"][key]["emp"]["recall"] for r in rows],
             "ff_ff5_emp": [r["transferred"][key]["emp"]["ff"] for r in rows],
             "within_budget_ff5_emp": [r["transferred"][key]["emp"]["within_budget"]
                                       for r in rows],
             "recall_ff5_cert": [r["transferred"][key]["cert"]["recall"] for r in rows],
             "ff_ff5_cert": [r["transferred"][key]["cert"]["ff"] for r in rows]}
    out: dict = {"n_seeds": len(rows)}
    for name in AGG_KEYS:
        v = np.array(table[name], float)
        out[name] = {"mean": float(np.nanmean(v)),
                     "sd": float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0}
    return out


def reused_pool(exp: str, name: str, ids: np.ndarray, seed: int) -> dict | None:
    if not (config.results_dir(exp) / name).exists():
        return None
    pool = evaluate.load_scores(exp, name, seed=seed)
    pos = {str(v): i for i, v in enumerate(pool["ids"])}
    idx = np.array([pos[str(v)] for v in ids], np.int64)
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(pool["ids"]) else v)
            for k, v in pool.items() if k != "admitted"}


def run(cfg: BaselineConfig) -> dict:
    items = data.load_pool()
    items = data.attach(items, data.load_cache(config.DEFAULT_CROP, ("sc",)), ("sc",))
    tint = colour.features(items)
    out: dict = {"exp": EXP, "name": "summary", "question": QUESTION,
                 "n": len(items), "n_pos": int((items.y == 1).sum()),
                 "n_neg": int((items.y == 0).sum()),
                 "robots": [str(r) for r in np.unique(items.robot)],
                 "budgets": list(cfg.budgets), "seeds": list(cfg.seeds),
                 "c_grid": list(C_GRID), "rf_features": list(RF_COLUMNS),
                 "configs": {}, "summary": {}, "runtime_s": {},
                 "native": {"flush_all": {"recall": 1.0, "ff": 1.0}}}
    t0 = time.time()
    for name, note in RULES:
        block = summarise(as_pool(items, rule_score(name, items, tint)), cfg)
        out["configs"][name] = {"notes": note, "per_seed": [block]}
        out["summary"][name] = aggregate_seeds([block], cfg)
    out["runtime_s"]["rules"] = round(time.time() - t0, 2)
    raw = np.concatenate([data.raw_scalars(items, DEPTH_COLUMNS + ("bbox_w", "bbox_h")),
                          tint[:, :6]], 1).astype(np.float32)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    t0 = time.time()
    rows = []
    for seed in cfg.seeds:
        score, _ = fold_local(items, raw, seed, "rf")
        rows.append(summarise(as_pool(items, score), cfg))
    out["configs"]["rf_depth_colour"] = {
        "notes": "random forest on depth scalars, bbox size and bbox colour", "per_seed": rows}
    out["summary"]["rf_depth_colour"] = aggregate_seeds(rows, cfg)
    out["runtime_s"]["rf"] = round(time.time() - t0, 2)
    if cfg.with_frozen:
        t0 = time.time()
        z = frozen_features(items)
        rows, picks = [], {}
        for seed in cfg.seeds:
            score, chosen = fold_local(items, z, seed, "logreg")
            picks[str(seed)] = chosen
            rows.append(summarise(as_pool(items, score), cfg))
        out["configs"]["frozen_convnext_logreg"] = {
            "notes": "logistic regression on frozen ImageNet features, C chosen inner CV",
            "per_seed": rows, "picks": picks}
        out["summary"]["frozen_convnext_logreg"] = aggregate_seeds(rows, cfg)
        out["runtime_s"]["frozen"] = round(time.time() - t0, 2)
    for key, exp, name, seeds in REUSED:
        rows = [reused_pool(exp, name, items.ids, s) for s in seeds]
        rows = [summarise(p, cfg) for p in rows if p is not None]
        if not rows:
            continue
        out["configs"][key] = {"source": f"{exp}/{name}", "per_seed": rows}
        out["summary"][key] = aggregate_seeds(rows, cfg)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=EXP)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--skip-frozen", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = replace(BaselineConfig(), seeds=tuple(int(s) for s in args.seeds.split(",") if s != ""),
                  with_frozen=not args.skip_frozen)
    if args.dry_run:
        for name, _ in RULES:
            print(f"{EXP}/{name} deterministic rule over the pooled folds")
        print(f"{EXP}/rf_depth_colour seeds {','.join(str(s) for s in cfg.seeds)}")
        if cfg.with_frozen:
            print(f"{EXP}/frozen_convnext_logreg seeds {','.join(str(s) for s in cfg.seeds)}")
        for key, exp, name, seeds in REUSED:
            print(f"{EXP}/{key} reused from {exp}/{name}")
        return 0
    out = run(cfg)
    evaluate.write_result(out)
    for name, agg in out["summary"].items():
        print(f"{name:24s} auprc={agg['auprc']['mean']:.3f} "
              f"matched={agg['matched_ff5']['mean']:.3f} "
              f"recall={agg['recall_ff5_emp']['mean']:.3f} ff={agg['ff_ff5_emp']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
