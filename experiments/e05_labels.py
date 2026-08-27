from __future__ import annotations

import numpy as np

from trashnet import data, evaluate, metrics, transfer

EXP = "e5"
SEEDS: tuple[int, ...] = (0,)
BUDGET = 0.05
FUSION_Q_GRID: tuple[float, ...] = transfer.Q_GRID
BASE: dict = {"backbone": "convnext_tiny", "regime": "finetune", "crop": "fixed420_288",
              "aug": "flip", "batch": 32, "early_stop": True, "epochs": 12,
              "focal_gamma": 2.0, "freeze_frac": 0.0, "in_chans": 3, "loss": "focal",
              "lr_head": 1e-3, "lr_trunk": 1e-4, "oof_mode": "holdout", "patience": 4,
              "pos_weight": "balanced", "px": 288}
REFERENCE = "b_mat3_w1"

GRID: list[dict] = [
    dict(BASE, name="a_binary", label_space="binary", material_weight=0.0,
         notes="single action head"),
    dict(BASE, name="b_mat3_w1", label_space="binary_plus_material3", material_weight=1.0,
         notes="two heads, 3 material classes"),
    dict(BASE, name="c_mat6_w1", label_space="binary_plus_material6", material_weight=1.0,
         notes="two heads, 6 material classes"),
    dict(BASE, name="d_mat3_w2", label_space="binary_plus_material3", material_weight=2.0,
         notes="two heads, material loss weight 2"),
    dict(BASE, name="e_matonly", label_space="binary_plus_material3", material_weight=1.0,
         action_weight=0.0, early_stop=False, notes="material head only, no action loss"),
    dict(BASE, name="b2_mat3_full", label_space="binary_plus_material3", material_weight=1.0,
         early_stop=False, notes="two heads, full fold fit, no early stopping"),
    dict(BASE, name="b3_noaug_full", label_space="binary_plus_material3", material_weight=1.0,
         early_stop=False, aug="none", notes="two heads, no augmentation, full fold fit"),
]
MATERIAL_SCORE_CONFIGS: tuple[str, ...] = ("e_matonly", "b_mat3_w1")


def material_pool(exp: str, name: str, seed: int = 0) -> dict:
    pool = evaluate.load_scores(exp, name, seed=seed)
    assert "p_material" in pool, f"{name} has no material head output"
    out = dict(pool)
    out["score"] = pool["p_material"][:, 0].astype(float)
    return out


def loro_point(pool: dict, strategy: str, budget: float = BUDGET) -> dict:
    applied = transfer.apply_strategy(pool, strategy, budget)
    out = transfer.score_flags(pool, applied["flags"], budget)
    out["thresholds"] = applied["thresholds"]
    out["q"] = applied["q"]
    out["auprc"] = metrics.auprc(pool["y"], pool["score"])
    return out


def fusion_sweep(pool: dict, budget: float = BUDGET) -> list[dict]:
    p_tp = pool["p_material"][:, 0].astype(float)
    y, s, robot = pool["y"], pool["score"], pool["robot"]
    rows = []
    for q in FUSION_Q_GRID:
        flags = np.zeros(len(s), bool)
        thresholds = {}
        for r in np.unique(robot):
            te = robot == r
            fit = ~te
            t = metrics.threshold_at_ff(y[fit], np.where(p_tp[fit] >= q, s[fit], -np.inf), budget)
            flags[te] = (s[te] >= t) & (p_tp[te] >= q)
            thresholds[str(r)] = float(t)
        row = transfer.score_flags(pool, flags, budget)
        row["q"] = float(q)
        row["thresholds"] = thresholds
        rows.append(row)
    return rows


def in_budget_tolerance(pool: dict, budget: float = BUDGET) -> float:
    return budget + 0.5 / max(int((pool["y"] == 0).sum()), 1)


def main(argv: list[str] | None = None) -> int:
    args = evaluate.build_parser(EXP, SEEDS).parse_args(argv)
    if args.dry_run:
        evaluate.run_grid(EXP, GRID, args)
        return 0
    evaluate.run_grid(EXP, GRID, args, items=data.load_pool())
    order = [c["name"] for c in GRID]
    sums = evaluate.summaries(EXP, order)
    block = evaluate.variant_block(sums, order, REFERENCE)
    pools = {n: evaluate.load_scores(EXP, n, seed=0) for n in block}
    loro = {n: {strat: loro_point(pools[n], strat) for strat in ("emp", "cert")}
            for n in pools}
    for n in MATERIAL_SCORE_CONFIGS:
        if n in pools and "p_material" in pools[n]:
            loro[f"{n}_material_score"] = {
                strat: loro_point(material_pool(EXP, n), strat) for strat in ("emp", "cert")}
    fusion = fusion_sweep(pools[REFERENCE]) if "p_material" in pools.get(REFERENCE, {}) else []
    tol = in_budget_tolerance(pools[REFERENCE]) if REFERENCE in pools else BUDGET
    candidates = {n: v["emp"] for n, v in loro.items()}
    in_budget = {n: v for n, v in candidates.items() if v["ff"] <= tol}
    chosen = max(in_budget or candidates, key=lambda n: candidates[n]["recall"])
    summary = {"exp": EXP, "name": "summary", "reference": REFERENCE, "base": BASE,
               "budget": BUDGET, "in_budget_tolerance": tol,
               "label_space": block, "loro": loro,
               "fusion_sweep": fusion,
               "fusion_fitted": loro_point(pools[REFERENCE], "fuse_pm")
               if "p_material" in pools.get(REFERENCE, {}) else {},
               "paired": evaluate.paired_vs_reference(EXP, list(block), REFERENCE),
               "chosen": chosen, "issues": evaluate.check(EXP)}
    evaluate.write_result(summary)
    print(evaluate.variant_table(sums, order, REFERENCE))
    for n, v in loro.items():
        print(f"{n}: LORO recall={v['emp']['recall']:.3f} ff={v['emp']['ff']:.3f} "
              f"cert recall={v['cert']['recall']:.3f} ff={v['cert']['ff']:.3f}")
    print("chosen:", chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
