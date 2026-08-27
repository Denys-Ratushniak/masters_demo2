from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np

from trashnet import config, evaluate, metrics, transfer
from trashnet.arms import align, matched_bootstrap, matched_recall, operating_point

EXP = "e19"
SOURCE_EXP = "e9"
SEEDS: tuple[int, ...] = tuple(range(7))
KS: tuple[int, ...] = (1, 3, 5, 7)
BUDGET = 0.05
MAX_SUBSETS = 20
DRAWS = 2000
MS_PER_NET = 6.46
DEVICE = "RTX A2000, FP16"
RECIPES: dict[str, str] = {"a_es": "loro_focal", "b_noes": "loro_focal_noes"}
LABELS: dict[str, str] = {"a_es": "final recipe with early stopping",
                          "b_noes": "final recipe, full 12 epochs"}
SUMMARY_KEYS: tuple[str, ...] = ("recall_ff5_matched", "auprc_pooled", "auprc_fold_mean",
                                 "emp_recall", "emp_ff", "emp_within_budget", "emp_ff_max",
                                 "cert_recall", "cert_ff", "cert_within_budget", "cert_ff_max",
                                 "thr_spread", "thr_spread_over_score_sd")
GRID: list[dict] = []


@dataclass
class SizeConfig:
    seeds: tuple[int, ...] = SEEDS
    ks: tuple[int, ...] = KS
    budget: float = BUDGET
    max_subsets: int = MAX_SUBSETS
    draws: int = DRAWS
    boot_seed: int = 0
    ms_per_net: float = MS_PER_NET


def score_matrix(name: str, seeds) -> tuple[np.ndarray, dict]:
    base = evaluate.load_scores(SOURCE_EXP, name, seed=seeds[0], n_folds=5)
    rows = []
    for s in seeds:
        pool = evaluate.load_scores(SOURCE_EXP, name, seed=s, n_folds=5)
        idx = align(base["ids"], pool["ids"])
        assert (pool["y"][idx].astype(int) == base["y"]).all(), f"seed {s} misaligned"
        rows.append(pool["score"][idx].astype(float))
    return np.stack(rows), base


def subsets_for(k: int, seeds, limit: int, seed: int = 0) -> list[tuple[int, ...]]:
    every = list(combinations(range(len(seeds)), k))
    if len(every) <= limit:
        return every
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(every), limit, replace=False)
    return [every[i] for i in sorted(picked)]


def subset_block(base: dict, scores: np.ndarray, subset, budget: float) -> dict:
    pool = dict(base)
    pool["score"] = scores[list(subset)].mean(0)
    fold_auprc = [metrics.auprc(pool["y"][pool["robot"] == r], pool["score"][pool["robot"] == r])
                  for r in np.unique(pool["robot"])]
    out = {"subset": list(subset),
           "recall_ff5_matched": matched_recall(pool, budget),
           "auprc_pooled": metrics.auprc(pool["y"], pool["score"]),
           "auprc_fold_mean": float(np.mean(fold_auprc)),
           "auprc_per_fold": [float(v) for v in fold_auprc]}
    for strategy in ("emp", "cert"):
        point = operating_point(pool, budget, strategy)
        for key in ("recall", "ff", "within_budget", "ff_max"):
            out[f"{strategy}_{key}"] = point[key]
        out[f"{strategy}_thresholds"] = point["thresholds"]
    spread = transfer.optimal_threshold_spread(pool, budget)
    out["thr_spread"] = spread["spread"]
    out["thr_per_robot"] = spread["per_robot"]
    out["thr_pooled"] = spread["pooled"]
    out["thr_spread_over_score_sd"] = float(spread["spread"] / max(pool["score"].std(), 1e-9))
    return out


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for key in SUMMARY_KEYS:
        v = np.array([r[key] for r in rows], float)
        out[key] = {"mean": float(np.nanmean(v)),
                    "sd": float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0,
                    "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
    return out


def recipe_block(name: str, cfg: SizeConfig) -> dict:
    scores, base = score_matrix(name, cfg.seeds)
    out: dict = {"config": name, "n": int(base["y"].size),
                 "n_pos": int((base["y"] == 1).sum()), "n_neg": int((base["y"] == 0).sum()),
                 "robots": [str(r) for r in np.unique(base["robot"])],
                 "k": {}, "canonical": {}, "bootstrap": {}}
    for k in cfg.ks:
        picks = subsets_for(k, cfg.seeds, cfg.max_subsets)
        rows = [subset_block(base, scores, s, cfg.budget) for s in picks]
        out["k"][str(k)] = {"subsets": [list(s) for s in picks], "n_subsets": len(picks),
                            "latency_ms": round(k * cfg.ms_per_net, 2),
                            "per_subset": rows, "summary": summarise(rows)}
        out["canonical"][str(k)] = list(range(k))
    sets = metrics.group_bootstrap_sets(base["group"], base["y"], cfg.draws, cfg.boot_seed)
    ref = tuple(range(3))
    for k in (5, 7):
        if k not in cfg.ks or 3 not in cfg.ks:
            continue
        a = dict(base, score=scores[list(range(k))].mean(0))
        b = dict(base, score=scores[list(ref)].mean(0))
        for strategy in ("emp", "cert"):
            fa = transfer.apply_strategy(a, strategy, cfg.budget)["flags"].astype(float)
            fb = transfer.apply_strategy(b, strategy, cfg.budget)["flags"].astype(float)
            out["bootstrap"][f"k{k}_minus_k3_{strategy}"] = metrics.paired_bootstrap_flags(
                base["y"], base["group"], fa, fb, sets=sets)
        out["bootstrap"][f"k{k}_minus_k3_matched_ff5"] = matched_bootstrap(
            base, a["score"], b["score"], cfg.budget, sets)
        out["bootstrap"][f"k{k}_minus_k3_auprc"] = metrics.paired_bootstrap(
            {"y": base["y"], "s": a["score"]}, {"y": base["y"], "s": b["score"]},
            base["group"], draws=cfg.draws, seed=cfg.boot_seed)
    return out


def run(cfg: SizeConfig) -> dict:
    out: dict = {"exp": EXP, "name": "summary", "budget": cfg.budget, "seeds": list(cfg.seeds),
                 "ks": list(cfg.ks), "draws": cfg.draws, "max_subsets": cfg.max_subsets,
                 "ms_per_net": cfg.ms_per_net, "device": DEVICE, "labels": LABELS,
                 "recipes": {}}
    for key, name in RECIPES.items():
        if (config.results_dir(SOURCE_EXP) / name).exists():
            out["recipes"][key] = recipe_block(name, cfg)
    assert out["recipes"], "no leave-one-robot-out scores under the source experiment"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=EXP)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--max-subsets", type=int, default=MAX_SUBSETS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = replace(SizeConfig(), seeds=tuple(int(s) for s in args.seeds.split(",") if s != ""),
                  ks=tuple(int(k) for k in args.ks.split(",") if k != ""),
                  draws=args.draws, max_subsets=args.max_subsets)
    if args.dry_run:
        for key, name in RECIPES.items():
            print(f"{EXP}/{key} reads {SOURCE_EXP}/{name} seeds "
                  f"{','.join(str(s) for s in cfg.seeds)}")
        print(f"{EXP}: ensemble sizes {list(cfg.ks)}, at most {cfg.max_subsets} subsets each")
        return 0
    out = run(cfg)
    evaluate.write_result(out)
    for key, block in out["recipes"].items():
        for k in cfg.ks:
            s = block["k"][str(k)]["summary"]["recall_ff5_matched"]
            print(f"{key:7s} k={k} recall at exactly FF5 = {s['mean']:.3f}±{s['sd']:.3f} "
                  f"[{s['min']:.3f}, {s['max']:.3f}] "
                  f"latency={block['k'][str(k)]['latency_ms']:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
