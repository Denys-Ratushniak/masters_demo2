from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np

from trashnet import config, evaluate
from trashnet.arms import Arm, arm_block, compare, seeds_present

EXP = "e11"
REFERENCE_EXP = "e9"
SEEDS: tuple[int, ...] = (0, 1, 2)
BUDGET = 0.05
STRATEGIES: tuple[str, ...] = ("emp", "cert")
BASE: dict = {"action_weight": 1.0, "aug": "none", "batch": 32, "crop": "fixed420_288",
              "epochs": 12, "freeze_frac": 0.0, "in_chans": 3,
              "label_space": "binary_plus_material3", "loss": "focal", "lr_head": 1e-3,
              "lr_trunk": 1e-4, "material_weight": 1.0, "oof_mode": "holdout", "patience": 4,
              "px": 288, "regime": "finetune"}
GRID: list[dict] = [
    dict(BASE, name="mnv3_focal", backbone="mobilenetv3_large_100", early_stop=True,
         notes="MobileNetV3-L on the final recipe with early stopping"),
    dict(BASE, name="mnv3_focal_noes", backbone="mobilenetv3_large_100", early_stop=False,
         notes="MobileNetV3-L on the final recipe, full 12 epochs"),
    dict(BASE, name="rn50_focal_noes", backbone="resnet50", early_stop=False,
         notes="ResNet50 on the final recipe, full 12 epochs"),
]
ARMS: tuple[Arm, ...] = (
    Arm("convnext_es", REFERENCE_EXP, "loro_focal", "ConvNeXt-T, early stopping"),
    Arm("convnext_noes", REFERENCE_EXP, "loro_focal_noes", "ConvNeXt-T, full 12 epochs"),
    Arm("mnv3_es", EXP, "mnv3_focal", "MobileNetV3-L, early stopping"),
    Arm("mnv3_noes", EXP, "mnv3_focal_noes", "MobileNetV3-L, full 12 epochs"),
    Arm("rn50_noes", EXP, "rn50_focal_noes", "ResNet50, full 12 epochs"),
)
PAIRS: tuple[tuple[str, str], ...] = (("mnv3_es", "convnext_es"),
                                      ("mnv3_noes", "convnext_noes"),
                                      ("rn50_noes", "convnext_noes"))
RECIPE = ("focal gamma 2, no augmentation, freeze_frac 0, lr_trunk 1e-4, lr_head 1e-3, "
          "12 epochs, batch 32, fixed420_288, RGB, binary_plus_material3, holdout "
          "out-of-fold scores, leave-one-robot-out pooled threshold")


@dataclass
class BackboneConfig:
    seeds: tuple[int, ...] = SEEDS
    budget: float = BUDGET
    draws: int = 2000
    boot_seed: int = 0


def backbone_cost() -> dict:
    summary = config.results_dir("e4") / "summary.json"
    if not summary.exists():
        return {}
    return json.loads(summary.read_text()).get("cost", {})


def run(cfg: BackboneConfig) -> dict:
    present = {a.key: (a, seeds_present(a, cfg.seeds)) for a in ARMS}
    present = {k: v for k, v in present.items() if v[1]}
    assert present, "no leave-one-robot-out scores for any backbone arm"
    out: dict = {"exp": EXP, "name": "summary", "budget": cfg.budget,
                 "seeds": list(cfg.seeds), "arms_present": sorted(present),
                 "arms": {}, "paired_bootstrap": {}, "cost": backbone_cost(),
                 "recipe": RECIPE}
    for key, (arm, seeds) in present.items():
        out["arms"][key] = arm_block(arm, seeds, cfg.budget)
    for left, right in PAIRS:
        if left not in present or right not in present:
            continue
        a_arm, a_seeds = present[left]
        b_arm, b_seeds = present[right]
        shared = [s for s in a_seeds if s in b_seeds]
        for s in shared:
            a = evaluate.load_scores(a_arm.exp, a_arm.name, seed=s)
            b = evaluate.load_scores(b_arm.exp, b_arm.name, seed=s)
            for strategy in STRATEGIES:
                out["paired_bootstrap"][f"{left}_minus_{right}_seed{s}_{strategy}"] = compare(
                    a, b, strategy, cfg.budget, cfg.draws, cfg.boot_seed)
        if len(shared) > 1:
            a = evaluate.pooled_scores(a_arm.exp, a_arm.name, shared, "mean")
            b = evaluate.pooled_scores(b_arm.exp, b_arm.name, shared, "mean")
            for strategy in STRATEGIES:
                out["paired_bootstrap"][f"{left}_minus_{right}_ensemble_{strategy}"] = compare(
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
        evaluate.run_grid(EXP, GRID, args)
        return 0
    cfg = replace(BackboneConfig(), seeds=tuple(int(s) for s in args.seeds.split(",") if s),
                  draws=args.draws)
    out = run(cfg)
    evaluate.write_result(out)
    for key in out["arms_present"]:
        agg = out["arms"][key]["agg"]
        print(f"{key:14s} auprc={agg['mean']['auprc_fold_mean']:.3f} "
              f"recall={agg['mean']['emp_recall']:.3f}±{agg['sd']['emp_recall']:.3f} "
              f"ff={agg['mean']['emp_ff']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
