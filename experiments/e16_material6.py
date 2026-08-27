from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from sklearn.metrics import f1_score

from trashnet import config, data, evaluate
from trashnet.arms import Arm, align, arm_block, compare, seeds_present

EXP = "e16"
REFERENCE_EXP = "e9"
REFERENCE = "mat3"
SEEDS: tuple[int, ...] = (0, 1, 2)
BUDGET = 0.05
QUESTION = ("does a six class material head change the flush decision, or only the "
            "human readable material report")
BASE: dict = {"action_weight": 1.0, "aug": "none", "backbone": "convnext_tiny", "batch": 32,
              "crop": "fixed420_288", "early_stop": False, "epochs": 12, "freeze_frac": 0.0,
              "in_chans": 3, "loss": "focal", "lr_head": 1e-3, "lr_trunk": 1e-4,
              "material_weight": 1.0, "oof_mode": "holdout", "patience": 4, "px": 288,
              "regime": "finetune"}
GRID: list[dict] = [
    dict(BASE, name="mat6", label_space="binary_plus_material6",
         notes="final recipe with the material head at 6 classes instead of 3"),
]
ARMS: tuple[Arm, ...] = (
    Arm(REFERENCE, REFERENCE_EXP, "loro_focal_noes", "material head with 3 classes"),
    Arm("mat6", EXP, "mat6", "material head with 6 classes"),
)


@dataclass
class MaterialConfig:
    seeds: tuple[int, ...] = SEEDS
    budget: float = BUDGET
    draws: int = 2000
    boot_seed: int = 0


def material_metrics(pool: dict) -> dict:
    if "p_material" not in pool:
        return {}
    p = np.asarray(pool["p_material"], float)
    classes = config.MATERIALS if p.shape[1] == 6 else config.MATERIALS3
    truth6 = np.asarray(pool["mat"])
    out: dict = {"n_classes": int(p.shape[1])}
    pred = np.asarray(classes)[p.argmax(1)]
    if p.shape[1] == 6:
        out["acc6"] = float((pred == truth6).mean())
        out["macro_f1_6"] = float(f1_score(truth6, pred, labels=list(config.MATERIALS),
                                           average="macro", zero_division=0))
        out["per_class_f1_6"] = {m: float(v) for m, v in zip(
            config.MATERIALS, f1_score(truth6, pred, labels=list(config.MATERIALS),
                                       average=None, zero_division=0))}
        pred3 = np.array([config.MATERIAL_TO3[m] for m in pred])
    else:
        pred3 = pred
    truth3 = np.array([config.MATERIAL_TO3[m] for m in truth6])
    out["acc3"] = float((pred3 == truth3).mean())
    out["macro_f1_3"] = float(f1_score(truth3, pred3, labels=list(config.MATERIALS3),
                                       average="macro", zero_division=0))
    out["per_class_f1_3"] = {m: float(v) for m, v in zip(
        config.MATERIALS3, f1_score(truth3, pred3, labels=list(config.MATERIALS3),
                                    average=None, zero_division=0))}
    out["per_robot_acc3"] = {str(r): float((pred3[pool["robot"] == r]
                                            == truth3[pool["robot"] == r]).mean())
                             for r in np.unique(pool["robot"])}
    out["class_counts"] = {m: int((truth6 == m).sum()) for m in config.MATERIALS}
    return out


def material_ensemble(arm: Arm, seeds) -> dict:
    pool = evaluate.pooled_scores(arm.exp, arm.name, seeds, "mean")
    stack = []
    for s in seeds:
        one = evaluate.load_scores(arm.exp, arm.name, seed=s)
        stack.append(one["p_material"][align(pool["ids"], one["ids"])])
    pool["p_material"] = np.mean(stack, 0)
    return material_metrics(pool)


def run(cfg: MaterialConfig) -> dict:
    present = {a.key: (a, seeds_present(a, cfg.seeds)) for a in ARMS}
    present = {k: v for k, v in present.items() if v[1]}
    assert REFERENCE in present, "the three class reference has no scores"
    out: dict = {"exp": EXP, "name": "summary", "question": QUESTION, "budget": cfg.budget,
                 "reference": REFERENCE, "seeds": list(cfg.seeds),
                 "arms_present": sorted(present), "arms": {}, "paired_bootstrap": {}}
    for key, (arm, seeds) in present.items():
        block = arm_block(arm, seeds, cfg.budget)
        block["material"] = {}
        for s in seeds:
            pool = evaluate.load_scores(arm.exp, arm.name, seed=s)
            block["material"][str(s)] = material_metrics(pool)
        if len(seeds) > 1:
            block["material"]["ensemble"] = material_ensemble(arm, seeds)
        out["arms"][key] = block
    ref_arm, ref_seeds = present[REFERENCE]
    for key, (arm, seeds) in present.items():
        if key == REFERENCE:
            continue
        shared = [s for s in seeds if s in ref_seeds]
        for s in shared:
            a = evaluate.load_scores(arm.exp, arm.name, seed=s)
            b = evaluate.load_scores(ref_arm.exp, ref_arm.name, seed=s)
            for strategy in ("emp", "cert"):
                out["paired_bootstrap"][f"{key}_minus_{REFERENCE}_seed{s}_{strategy}"] = compare(
                    a, b, strategy, cfg.budget, cfg.draws, cfg.boot_seed)
        if len(shared) > 1:
            a = evaluate.pooled_scores(arm.exp, arm.name, shared, "mean")
            b = evaluate.pooled_scores(ref_arm.exp, ref_arm.name, shared, "mean")
            for strategy in ("emp", "cert"):
                out["paired_bootstrap"][f"{key}_minus_{REFERENCE}_ensemble_{strategy}"] = compare(
                    a, b, strategy, cfg.budget, cfg.draws, cfg.boot_seed)
    out["issues"] = evaluate.check(EXP)
    return out


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
        evaluate.run_grid(EXP, GRID, args, items=data.load_pool())
        return 0
    cfg = replace(MaterialConfig(), seeds=tuple(int(s) for s in args.seeds.split(",") if s != ""),
                  draws=args.draws)
    out = run(cfg)
    evaluate.write_result(out)
    for key in out["arms_present"]:
        agg = out["arms"][key]["agg"]
        mat = out["arms"][key]["material"].get("ensemble", {})
        print(f"{key:6s} auprc={agg['mean']['auprc_fold_mean']:.3f} "
              f"recall={agg['mean']['emp_recall']:.3f}±{agg['sd']['emp_recall']:.3f} "
              f"matched={agg['mean']['matched_recall']:.3f} "
              f"material acc3={mat.get('acc3', float('nan')):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
