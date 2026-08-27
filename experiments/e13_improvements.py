from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from trashnet import config, evaluate, metrics, transfer
from trashnet.arms import Arm, align, arm_block, seeds_present
from trashnet.sizerule import RuleConfig, build_banks, fixed_rule, size_inputs

EXP = "e13"
REFERENCE_EXP = "e9"
REFERENCE = "base"
SEEDS: tuple[int, ...] = (0, 1, 2)
BUDGET = 0.05
BUDGETS: tuple[float, ...] = (0.02, 0.05, 0.10)
SIZE_VARIANT = "v5_logsize"
BASE: dict = {"action_weight": 1.0, "aug": "none", "backbone": "convnext_tiny", "batch": 32,
              "crop": "fixed420_288", "early_stop": False, "epochs": 12, "freeze_frac": 0.0,
              "in_chans": 3, "label_space": "binary_plus_material3", "loss": "focal",
              "lr_head": 1e-3, "lr_trunk": 1e-4, "material_weight": 1.0, "oof_mode": "holdout",
              "patience": 4, "px": 288, "regime": "finetune"}
LATE_SCALARS: tuple[str, ...] = config.SCALARS + ("support_rel_mm", "support_missing",
                                                  "area_cm2_label", "bbox_w", "bbox_h")
GRID: list[dict] = [
    dict(BASE, name="ls01", label_smooth=0.1,
         notes="label smoothing 0.1, with the focal branch reading the smoothed target"),
    dict(BASE, name="late_scalars", scalar_dim=len(LATE_SCALARS), scalars=list(LATE_SCALARS),
         notes="late fusion of 14 size and depth scalars into the head"),
    dict(BASE, name="ema999", ema_decay=0.999, notes="exponential moving average of the weights"),
    dict(BASE, name="aux_size", size_aux_weight=0.5,
         notes="auxiliary regression head on log area and support height"),
    dict(BASE, name="tta_flip", tta_flip=True,
         notes="mirror averaging at inference, identical weights to the base recipe"),
    dict(BASE, name="px384", crop="fixed420_384", px=384,
         notes="the same 420 px window rendered and fed at 384 px"),
    dict(BASE, name="cbase", backbone="convnext_base",
         notes="a heavier trunk on the same recipe, two folds only"),
]
ARMS: tuple[Arm, ...] = ((Arm(REFERENCE, REFERENCE_EXP, "loro_focal_noes", "base recipe"),)
                         + tuple(Arm(c["name"], EXP, c["name"], c["notes"]) for c in GRID))
PROBE_FOLDS: tuple[str, ...] = config.TRAIN_ROBOTS[3:5]


@dataclass
class ImprovementConfig:
    seeds: tuple[int, ...] = SEEDS
    budget: float = BUDGET
    budgets: tuple[float, ...] = BUDGETS
    draws: int = 2000
    boot_seed: int = 0
    delta: float = 0.05

    def rules(self) -> RuleConfig:
        return RuleConfig(delta=self.delta)


def ensemble_pool(arm: Arm, seeds) -> dict:
    if len(seeds) > 1:
        return evaluate.pooled_scores(arm.exp, arm.name, seeds, "mean")
    return evaluate.load_scores(arm.exp, arm.name, seed=seeds[0])


def matched(y: np.ndarray, score: np.ndarray, budgets) -> dict:
    return {config.budget_key(b): float(metrics.recall_at_ff_targets(y, score, [b])[0])
            for b in budgets}


def transferred(pool: dict, budgets) -> dict:
    out: dict = {}
    for b in budgets:
        bk = config.budget_key(b)
        out[bk] = {}
        for crit in ("emp", "cert"):
            applied = transfer.apply_strategy(pool, crit, b)
            entry = transfer.score_flags(pool, applied["flags"], b)
            entry["thresholds"] = applied["thresholds"]
            out[bk][crit] = entry
    return out


def rule_on_top(pool: dict, bank: dict, size: np.ndarray, cfg: ImprovementConfig) -> dict:
    y, robot, gate = pool["y"], pool["robot"], pool["score"]
    out: dict = {"and_rule": {}, "fused": {}}
    for b in cfg.budgets:
        bk = config.budget_key(b)
        out["and_rule"][bk] = {}
        for crit in ("emp", "cert"):
            flags, picks = fixed_rule(y, robot, gate, bank, SIZE_VARIANT, b, crit, cfg.rules())
            entry = transfer.score_flags(pool, flags, b)
            entry["picks"] = picks
            out["and_rule"][bk][crit] = entry
    fused = dict(pool, score=gate * size)
    out["fused"] = transferred(fused, cfg.budgets)
    out["fused_matched"] = matched(y, fused["score"], cfg.budgets)
    return out


def run(cfg: ImprovementConfig) -> dict:
    present = {a.key: (a, seeds_present(a, cfg.seeds)) for a in ARMS}
    partial = {k: a.name for k, (a, s) in present.items() if not s}
    present = {k: v for k, v in present.items() if v[1]}
    assert REFERENCE in present, "the base recipe has no leave-one-robot-out scores"
    ref_arm, ref_seeds = present[REFERENCE]
    ref_pool = ensemble_pool(ref_arm, ref_seeds)
    ids, y = ref_pool["ids"], ref_pool["y"]
    box = size_inputs(ids)
    bank, _, p_size, size_info = build_banks(box, y, ref_pool["robot"], cfg.rules())
    size = np.zeros(len(y), float)
    for r in np.unique(ref_pool["robot"]):
        m = ref_pool["robot"] == r
        size[m] = p_size[str(r)][m]
    out: dict = {"exp": EXP, "name": "summary", "budget": cfg.budget,
                 "budgets": list(cfg.budgets), "seeds": list(cfg.seeds),
                 "reference": REFERENCE, "arms_present": sorted(present),
                 "arms_partial": partial, "size_variant": SIZE_VARIANT,
                 "size_model": size_info, "arms": {}, "matched_ff": {}, "rule_on_top": {},
                 "bootstrap": {}}
    sets = metrics.group_bootstrap_sets(ref_pool["group"], y, cfg.draws, cfg.boot_seed)
    for key, (arm, seeds) in present.items():
        out["arms"][key] = arm_block(arm, seeds, cfg.budget)
        pool = ensemble_pool(arm, seeds)
        idx = align(ids, pool["ids"])
        aligned = dict(ref_pool, score=pool["score"][idx].astype(float))
        out["arms"][key]["transferred"] = transferred(aligned, cfg.budgets)
        out["matched_ff"][key] = matched(y, aligned["score"], cfg.budgets)
        block = rule_on_top(aligned, bank, size, cfg)
        out["rule_on_top"][key] = block
        out["matched_ff"][f"{key}+fused"] = block["fused_matched"]
        if key == REFERENCE:
            continue
        out["bootstrap"][f"matched_ff5__{key}__{REFERENCE}"] = _matched_delta(
            y, aligned["score"], ref_pool["score"], cfg.budget, sets)
        out["bootstrap"][f"matched_ff5__{key}+fused__{REFERENCE}"] = _matched_delta(
            y, aligned["score"] * size, ref_pool["score"], cfg.budget, sets)
    out["issues"] = evaluate.check(EXP)
    return out


def _matched_delta(y: np.ndarray, sa: np.ndarray, sb: np.ndarray, budget: float,
                   sets: list[np.ndarray]) -> dict:
    deltas = np.array([float(metrics.recall_at_ff_targets(y[sel], sa[sel], [budget])[0]
                             - metrics.recall_at_ff_targets(y[sel], sb[sel], [budget])[0])
                       for sel in sets], float)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"d_recall": float(deltas.mean()), "lo": float(lo), "hi": float(hi),
            "p_gt0": float((deltas > 0).mean()), "n": int(deltas.size)}


def main(argv: list[str] | None = None) -> int:
    ap = evaluate.build_parser(EXP, SEEDS)
    ap.add_argument("--stage", default="loro", choices=("loro", "eval"))
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.dry_run:
        evaluate.run_grid(EXP, GRID, args)
        print(f"{EXP}/stage eval")
        return 0
    if args.stage == "loro":
        for cfg in evaluate.select(GRID, args.only):
            sub = evaluate.build_parser(EXP, SEEDS).parse_args([])
            sub.folds, sub.force, sub.seeds = args.folds, args.force, args.seeds
            if cfg["name"] == "cbase" and sub.folds == "all":
                sub.folds = ",".join(PROBE_FOLDS)
            evaluate.run_grid(EXP, [cfg], sub)
        return 0
    cfg = replace(ImprovementConfig(),
                  seeds=tuple(int(s) for s in args.seeds.split(",") if s != ""),
                  draws=args.draws)
    out = run(cfg)
    evaluate.write_result(out)
    key5 = config.budget_key(BUDGET)
    for key in out["arms_present"]:
        print(f"{key:14s} matched={out['matched_ff'][key][key5]:.3f} "
              f"fused={out['matched_ff'][key + '+fused'][key5]:.3f} "
              f"transferred={out['arms'][key]['transferred'][key5]['emp']['recall']:.3f} "
              f"ff={out['arms'][key]['transferred'][key5]['emp']['ff']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
