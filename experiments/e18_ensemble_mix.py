from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np

from trashnet import config, evaluate, metrics, transfer
from trashnet.arms import (blend, matched_bootstrap, matched_recall, operating_point)

EXP = "e18"
SEEDS: tuple[int, ...] = (0, 1, 2)
BUDGET = 0.05
BUDGETS: tuple[float, ...] = (0.02, 0.05, 0.10)
DRAWS = 2000
REFERENCE = "a_seed3"
QUESTION = ("does mixing architectures beat averaging three seeds of one architecture, "
            "at the same or a larger inference cost")
SOURCES: dict[str, tuple[str, str]] = {
    "convnext_noes": ("e9", "loro_focal_noes"),
    "convnext_es": ("e9", "loro_focal"),
    "mnv3_noes": ("e11", "mnv3_focal_noes"),
    "mnv3_es": ("e11", "mnv3_focal"),
    "rn50_noes": ("e11", "rn50_focal_noes"),
}
MEMBERS: dict[str, dict[str, list[tuple[str, int]]]] = {
    "noes": {
        "a_seed3": [("convnext_noes", s) for s in SEEDS],
        "b_arch3": [("convnext_noes", 0), ("mnv3_noes", 0), ("rn50_noes", 0)],
        "c_arch9": [(m, s) for m in ("convnext_noes", "mnv3_noes", "rn50_noes") for s in SEEDS],
        "d_convnext_mnv3": [(m, s) for m in ("convnext_noes", "mnv3_noes") for s in SEEDS],
        "e_single": [("convnext_noes", 0)],
    },
    "es": {
        "a_seed3": [("convnext_es", s) for s in SEEDS],
        "b_arch3": [("convnext_es", 0), ("mnv3_es", 0)],
        "d_convnext_mnv3": [(m, s) for m in ("convnext_es", "mnv3_es") for s in SEEDS],
        "e_single": [("convnext_es", 0)],
    },
}
LABELS: dict[str, str] = {
    "a_seed3": "three seeds of one architecture",
    "b_arch3": "one seed of each architecture",
    "c_arch9": "three seeds of each of three architectures",
    "d_convnext_mnv3": "three seeds of two architectures",
    "e_single": "a single network",
}
GRID: list[dict] = []


@dataclass
class MixConfig:
    budget: float = BUDGET
    budgets: tuple[float, ...] = BUDGETS
    draws: int = DRAWS
    boot_seed: int = 0


def member_pool(key: str, seed: int) -> dict:
    exp, name = SOURCES[key]
    return evaluate.load_scores(exp, name, seed=seed, n_folds=5)


def available(members: list[tuple[str, int]]) -> bool:
    return all((config.results_dir(SOURCES[k][0]) / SOURCES[k][1]).exists() and
               len(list((config.results_dir(SOURCES[k][0]) / SOURCES[k][1])
                        .glob(f"fold*_seed{s}_scores.npz"))) == 5 for k, s in members)


def calibration(pool: dict, budget: float) -> dict:
    spread = transfer.optimal_threshold_spread(pool, budget)
    v = np.array(list(spread["per_robot"].values()), float)
    sd = float(pool["score"].std())
    return {"per_robot": spread["per_robot"], "pooled": spread["pooled"],
            "spread": spread["spread"], "sd": float(np.std(v, ddof=1)),
            "score_sd": sd, "spread_over_score_sd": float(spread["spread"] / max(sd, 1e-9)),
            "mean_abs_deviation": float(np.mean(np.abs(v - spread["pooled"])))}


def arm_block(pool: dict, cfg: MixConfig) -> dict:
    fold_auprc = [metrics.auprc(pool["y"][pool["robot"] == r], pool["score"][pool["robot"] == r])
                  for r in np.unique(pool["robot"])]
    out = {"auprc_pooled": metrics.auprc(pool["y"], pool["score"]),
           "auprc_fold_mean": float(np.mean(fold_auprc)),
           "auprc_per_fold": [float(v) for v in fold_auprc],
           "matched_ff": {config.budget_key(b): matched_recall(pool, b) for b in cfg.budgets},
           "calibration": calibration(pool, cfg.budget)}
    for strategy in ("emp", "cert"):
        out[strategy] = operating_point(pool, cfg.budget, strategy)
    return out


def recipe_block(name: str, arms: dict[str, list[tuple[str, int]]], cfg: MixConfig) -> dict:
    pools = {key: blend([member_pool(k, s) for k, s in members])
             for key, members in arms.items() if available(members)}
    assert REFERENCE in pools, f"{name}: the seed ensemble reference is missing"
    ref = pools[REFERENCE]
    out: dict = {"n": int(ref["y"].size), "n_pos": int((ref["y"] == 1).sum()),
                 "n_neg": int((ref["y"] == 0).sum()),
                 "robots": [str(r) for r in np.unique(ref["robot"])],
                 "members": {k: [[m, int(s)] for m, s in arms[k]] for k in pools},
                 "arms": {k: arm_block(p, cfg) for k, p in pools.items()},
                 "bootstrap": {}}
    sets = metrics.group_bootstrap_sets(ref["group"], ref["y"], cfg.draws, cfg.boot_seed)
    for key, pool in pools.items():
        if key == REFERENCE:
            continue
        for strategy in ("emp", "cert"):
            fa = transfer.apply_strategy(pool, strategy, cfg.budget)["flags"].astype(float)
            fb = transfer.apply_strategy(ref, strategy, cfg.budget)["flags"].astype(float)
            out["bootstrap"][f"{key}_minus_{REFERENCE}_{strategy}"] = \
                metrics.paired_bootstrap_flags(ref["y"], ref["group"], fa, fb, sets=sets)
        for b in cfg.budgets:
            out["bootstrap"][f"{key}_minus_{REFERENCE}_matched_{config.budget_key(b)}"] = \
                matched_bootstrap(ref, pool["score"], ref["score"], b, sets)
        out["bootstrap"][f"{key}_minus_{REFERENCE}_auprc"] = metrics.paired_bootstrap(
            {"y": ref["y"], "s": pool["score"]}, {"y": ref["y"], "s": ref["score"]},
            ref["group"], draws=cfg.draws, seed=cfg.boot_seed)
    return out


def run(cfg: MixConfig) -> dict:
    out: dict = {"exp": EXP, "name": "summary", "question": QUESTION, "budget": cfg.budget,
                 "budgets": list(cfg.budgets), "seeds": list(SEEDS), "draws": cfg.draws,
                 "labels": LABELS, "sources": {k: list(v) for k, v in SOURCES.items()},
                 "recipes": {}}
    for name, arms in MEMBERS.items():
        out["recipes"][name] = recipe_block(name, arms, cfg)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=EXP)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = replace(MixConfig(), draws=args.draws)
    if args.dry_run:
        for name, arms in MEMBERS.items():
            for key, members in arms.items():
                print(f"{EXP}/{name}/{key}: {len(members)} networks, "
                      f"{'ready' if available(members) else 'missing scores'}")
        return 0
    out = run(cfg)
    evaluate.write_result(out)
    key5 = config.budget_key(BUDGET)
    for name, block in out["recipes"].items():
        for key, arm in sorted(block["arms"].items()):
            print(f"{name:5s} {key:16s} auprc={arm['auprc_pooled']:.3f} "
                  f"matched={arm['matched_ff'][key5]:.3f} "
                  f"recall={arm['emp']['recall']:.3f} ff={arm['emp']['ff']:.3f} "
                  f"spread={arm['calibration']['spread']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
