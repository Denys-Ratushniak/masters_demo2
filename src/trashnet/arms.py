from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import config, evaluate, metrics, transfer

STRATEGIES: tuple[str, ...] = ("emp", "cert")
POINT_KEYS: tuple[str, ...] = ("recall", "ff", "ff_max", "within_budget")


@dataclass(frozen=True)
class Arm:
    key: str
    exp: str
    name: str
    label: str = ""


def align(ids: np.ndarray, other: np.ndarray) -> np.ndarray:
    pos = {str(v): i for i, v in enumerate(other)}
    return np.array([pos[str(v)] for v in ids], np.int64)


def seeds_present(arm: Arm, seeds: Sequence[int], n_folds: int = 5) -> list[int]:
    root = config.results_dir(arm.exp) / arm.name
    return [int(s) for s in seeds
            if len(list(root.glob(f"fold*_seed{s}_scores.npz"))) == n_folds]


def auprc_by_fold(pool: dict) -> tuple[float, list[float]]:
    vals = [metrics.auprc(pool["y"][pool["robot"] == r], pool["score"][pool["robot"] == r])
            for r in np.unique(pool["robot"])]
    return float(np.mean(vals)), [float(v) for v in vals]


def operating_point(pool: dict, budget: float, strategy: str) -> dict:
    applied = transfer.apply_strategy(pool, strategy, budget)
    out = transfer.score_flags(pool, applied["flags"], budget)
    out["thresholds"] = applied["thresholds"]
    return out


def matched_recall(pool: dict, budget: float) -> float:
    return float(metrics.recall_at_ff_targets(pool["y"], pool["score"], [budget])[0])


def cost_block(arm: Arm, seed: int) -> dict:
    epochs, best, spent, mem = [], [], [], []
    for path in sorted((config.results_dir(arm.exp) / arm.name).glob(f"fold*_seed{seed}.json")):
        row = json.loads(path.read_text())
        epochs.append(int(row["epochs_run"]))
        best.append(int(row["best_epoch"]))
        spent.append(float(row["train_time_s"]) + float(row.get("oof_time_s", 0.0)))
        mem.append(float(row["gpu_mem_mb"]))
    return {"epochs_run": epochs, "best_epoch": best, "fit_time_s": float(np.sum(spent)),
            "gpu_mem_mb": float(np.max(mem)) if mem else float("nan")}


def seed_block(pool: dict, budget: float, strategies: Sequence[str] = STRATEGIES) -> dict:
    fold_mean, per_fold = auprc_by_fold(pool)
    out = {"auprc_fold_mean": fold_mean, "auprc_per_fold": per_fold,
           "auprc_pooled": metrics.auprc(pool["y"], pool["score"]),
           "matched_recall": matched_recall(pool, budget),
           "n": int(pool["y"].size), "n_pos": int((pool["y"] == 1).sum()),
           "n_neg": int((pool["y"] == 0).sum())}
    for strategy in strategies:
        out[strategy] = operating_point(pool, budget, strategy)
    return out


def aggregate(per_seed: dict, seeds: Sequence[int],
              strategies: Sequence[str] = STRATEGIES) -> dict:
    agg: dict = {"mean": {}, "sd": {}, "per_seed": {}}
    fields = [("auprc_fold_mean", None), ("auprc_pooled", None), ("matched_recall", None),
              ("fit_time_s", None)]
    fields += [(f"{s}_{k}", (s, k)) for s in strategies for k in POINT_KEYS]
    fields += [(f"{s}_leak_{m}", (s, f"leak_{m}"))
               for s in strategies for m in config.MATERIALS]
    for key, path in fields:
        try:
            v = np.array([per_seed[str(s)][key] if path is None
                          else per_seed[str(s)][path[0]][path[1]] for s in seeds], float)
        except KeyError:
            continue
        agg["mean"][key] = float(np.nanmean(v))
        agg["sd"][key] = float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0
        agg["per_seed"][key] = v.tolist()
    return agg


def arm_block(arm: Arm, seeds: Sequence[int], budget: float, with_cost: bool = True) -> dict:
    per_seed: dict[str, dict] = {}
    for s in seeds:
        pool = evaluate.load_scores(arm.exp, arm.name, seed=s, n_folds=5)
        block = seed_block(pool, budget)
        if with_cost:
            block.update(cost_block(arm, s))
        per_seed[str(s)] = block
    out = {"exp": arm.exp, "config": arm.name, "label": arm.label, "seeds": list(seeds),
           "per_seed": per_seed, "agg": aggregate(per_seed, seeds)}
    if len(seeds) > 1:
        out["ensemble"] = seed_block(evaluate.pooled_scores(arm.exp, arm.name, seeds, "mean"),
                                     budget)
    return out


def compare(a: dict, b: dict, strategy: str, budget: float, draws: int = 2000,
            seed: int = 0) -> dict:
    order = align(a["ids"], b["ids"])
    other = dict(a)
    other["score"] = b["score"][order]
    fa = transfer.apply_strategy(a, strategy, budget)["flags"].astype(float)
    fb = transfer.apply_strategy(other, strategy, budget)["flags"].astype(float)
    out = metrics.paired_bootstrap_flags(a["y"], a["group"], fa, fb, draws=draws, seed=seed)
    out["auprc"] = metrics.paired_bootstrap({"y": a["y"], "s": a["score"]},
                                            {"y": a["y"], "s": other["score"]}, a["group"],
                                            draws=draws, seed=seed)
    out["recall_a"] = float(fa[a["y"] == 1].mean())
    out["recall_b"] = float(fb[a["y"] == 1].mean())
    out["ff_a"] = float(fa[a["y"] == 0].mean())
    out["ff_b"] = float(fb[a["y"] == 0].mean())
    return out


def matched_bootstrap(pool: dict, sa: np.ndarray, sb: np.ndarray, budget: float,
                      sets: list[np.ndarray]) -> dict:
    y = pool["y"]
    deltas = np.array([float(metrics.recall_at_ff_targets(y[sel], sa[sel], [budget])[0]
                             - metrics.recall_at_ff_targets(y[sel], sb[sel], [budget])[0])
                       for sel in sets], float)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"d_recall": float(deltas.mean()), "lo": float(lo), "hi": float(hi),
            "p_gt0": float((deltas > 0).mean()), "n": int(deltas.size)}


def blend(members: Sequence[dict]) -> dict:
    base = dict(members[0])
    stack = [members[0]["score"]]
    for pool in members[1:]:
        stack.append(pool["score"][align(base["ids"], pool["ids"])])
    base["score"] = np.mean(stack, 0)
    return base
