from __future__ import annotations

import numpy as np

from trashnet import config, data, evaluate, metrics, transfer

EXP = "e6"
SEEDS: tuple[int, ...] = (0,)
BUDGETS: tuple[float, ...] = config.BUDGETS
STRATEGIES: tuple[str, ...] = tuple(transfer.STRATEGIES)
PART_A_SOURCES: tuple[tuple[str, str, str, tuple[int, ...]], ...] = (
    ("convnext_tiny_ft", "e4", "convnext_tiny_finetune", (0,)),
    ("resnet50_ft", "e4", "resnet50_finetune", (0,)),
    ("mobilenetv3_ft", "e4", "mobilenetv3_large_100_finetune", (0,)),
    ("efficientnet_b0_ft", "e4", "efficientnet_b0_finetune", (0,)),
    ("convnext_tiny_frozen_e3", "e3", "m3_rgb_scalars_support", (0, 1, 2, 3, 4)),
)
BASELINE_CURVES: tuple[tuple[str, str, str], ...] = (
    ("flush_all", "e1", "a_flush_all"),
    ("oracle_tp", "e1", "b_oracle_tp"),
    ("oracle_tp_area", "e1", "c_oracle_tp_area_px"),
)
BASE: dict = {"backbone": "convnext_tiny", "regime": "finetune", "crop": "fixed420_288",
              "epochs": 12, "batch": 32, "lr_trunk": 1e-4, "lr_head": 1e-3,
              "label_space": "binary_plus_material3", "in_chans": 3, "px": 288, "aug": "none",
              "patience": 4, "early_stop": True, "oof_mode": "holdout"}
REFERENCE = "cw1"
GRID: list[dict] = [
    dict(BASE, name="cw1", class_weight_ff=1.0, notes="no extra weight on false flushes"),
    dict(BASE, name="cw3", class_weight_ff=3.0),
    dict(BASE, name="cw10", class_weight_ff=10.0),
    dict(BASE, name="cw30", class_weight_ff=30.0),
    dict(BASE, name="focal", loss="focal", focal_gamma=2.0),
    dict(BASE, name="ls01", label_smooth=0.1),
]
PART_B_STRATEGIES: tuple[str, ...] = ("emp", "cert", "worst_loc", "inflate1.5", "qnorm")


def inner_cv_reference(exp: str, name: str) -> dict:
    rows = evaluate.load_results(exp, name)
    if not rows:
        return {}
    ff = np.array([r["ff_actual"]["ff5"] for r in rows], float)
    thr = np.array([r["threshold_used"]["ff5"] for r in rows], float)
    return {"n": len(rows),
            "recall": float(np.mean([r["recall_at_ff5"] for r in rows])),
            "ff_mean": float(ff.mean()), "ff_max": float(ff.max()),
            "within_budget": float(np.mean(ff <= 0.05 + 1e-9)),
            "thr_mean": float(thr.mean()), "thr_min": float(thr.min()),
            "thr_max": float(thr.max()),
            "oof_ff": float(np.mean([r["oof_ff_actual"]["ff5"] for r in rows]))}


def strategy_block(pool: dict, budgets=BUDGETS, strategies=STRATEGIES) -> dict:
    out: dict = {}
    for strat in strategies:
        if strat == "fuse_pm" and "p_material" not in pool:
            continue
        out[strat] = {}
        for b in budgets:
            applied = transfer.apply_strategy(pool, strat, b)
            row = transfer.score_flags(pool, applied["flags"], b)
            row["thresholds"] = applied["thresholds"]
            out[strat][config.budget_key(b)] = row
    return out


def part_a() -> dict:
    out: dict = {"configs": {}, "baselines": {}}
    for label, exp, name, seeds in PART_A_SOURCES:
        if not (config.results_dir(exp) / name).exists():
            continue
        pools = [evaluate.load_scores(exp, name, seed=s) for s in seeds]
        pool = pools[0]
        sets = metrics.group_bootstrap_sets(pool["group"], pool["y"], 1000, 0)
        blocks = [strategy_block(p) for p in pools]
        merged: dict = {}
        for strat in blocks[0]:
            merged[strat] = {}
            for k in blocks[0][strat]:
                vals = [b[strat][k] for b in blocks]
                merged[strat][k] = {kk: float(np.mean([v[kk] for v in vals]))
                                    for kk, vv in vals[0].items() if isinstance(vv, float)}
                merged[strat][k]["thresholds"] = vals[0]["thresholds"]
        ref_flags = transfer.apply_strategy(pool, "emp", 0.05)["flags"]
        merged_vs_emp = {}
        for strat in blocks[0]:
            flags = transfer.apply_strategy(pool, strat, 0.05)["flags"]
            merged_vs_emp[strat] = metrics.paired_bootstrap_flags(
                pool["y"], pool["group"], flags, ref_flags, sets=sets)
        out["configs"][label] = {
            "exp": exp, "config": name, "seeds": list(seeds),
            "inner_cv_reference": inner_cv_reference(exp, name),
            "strategies": merged, "vs_emp": merged_vs_emp,
            "threshold_spread": transfer.optimal_threshold_spread(pool),
            "auprc": metrics.auprc(pool["y"], pool["score"])}
    for label, exp, name in BASELINE_CURVES:
        if (config.results_dir(exp) / name).exists():
            pool = evaluate.load_scores(exp, name, seed=0)
            out["baselines"][label] = transfer.safety_curve(pool)
    return out


def part_b() -> dict:
    order = [c["name"] for c in GRID]
    sums = evaluate.summaries(EXP, order)
    block = evaluate.variant_block(sums, order, REFERENCE)
    out: dict = {"configs": block, "loro": {}, "spread": {}, "vs_reference": {}}
    if REFERENCE not in block:
        return out
    ref_pool = evaluate.load_scores(EXP, REFERENCE, seed=0)
    sets = metrics.group_bootstrap_sets(ref_pool["group"], ref_pool["y"], 1000, 0)
    ref_flags = transfer.apply_strategy(ref_pool, "emp", 0.05)["flags"]
    for name in block:
        pool = evaluate.load_scores(EXP, name, seed=0)
        out["loro"][name] = strategy_block(pool, strategies=PART_B_STRATEGIES)
        out["spread"][name] = transfer.optimal_threshold_spread(pool)
        if name != REFERENCE and list(pool["ids"]) == list(ref_pool["ids"]):
            flags = transfer.apply_strategy(pool, "emp", 0.05)["flags"]
            out["vs_reference"][name] = metrics.paired_bootstrap_flags(
                ref_pool["y"], ref_pool["group"], flags, ref_flags, sets=sets)
    in_budget = {n: v["emp"]["ff5"] for n, v in out["loro"].items()
                 if v["emp"]["ff5"]["ff"] <= 0.05 + 1e-9}
    pool = in_budget or {n: v["emp"]["ff5"] for n, v in out["loro"].items()}
    out["best"] = max(pool, key=lambda n: pool[n]["recall"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = evaluate.build_parser(EXP, SEEDS)
    ap.add_argument("--stage", default="all", choices=("a", "b", "all"))
    args = ap.parse_args(argv)
    if args.dry_run:
        print(f"{EXP}/part_a over {[s[0] for s in PART_A_SOURCES]}")
        evaluate.run_grid(EXP, GRID, args)
        return 0
    summary: dict = {"exp": EXP, "name": "summary", "base": BASE, "budgets": list(BUDGETS),
                     "strategies": list(STRATEGIES)}
    if args.stage in ("a", "all"):
        summary["part_a"] = part_a()
    if args.stage in ("b", "all"):
        evaluate.run_grid(EXP, GRID, args, items=data.load_pool())
        summary["part_b"] = part_b()
        summary["issues"] = evaluate.check(EXP)
    evaluate.write_result(summary)
    if "part_b" in summary and summary["part_b"].get("best"):
        print("best loss variant:", summary["part_b"]["best"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
