from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np

from trashnet import config, data, evaluate
from trashnet.arms import Arm, arm_block, compare, seeds_present

EXP = "e12"
REFERENCE_EXP = "e9"
REFERENCE = "rgb"
SEEDS: tuple[int, ...] = (0, 1, 2)
BUDGET = 0.05
STRATEGIES: tuple[str, ...] = ("emp", "cert")
GEO_SCALARS: tuple[str, ...] = config.SCALARS + ("support_rel_mm", "support_missing")
BASE: dict = {"action_weight": 1.0, "aug": "none", "backbone": "convnext_tiny", "batch": 32,
              "crop": "fixed420_288", "early_stop": False, "epochs": 12, "freeze_frac": 0.0,
              "label_space": "binary_plus_material3", "loss": "focal", "lr_head": 1e-3,
              "lr_trunk": 1e-4, "material_weight": 1.0, "oof_mode": "holdout", "patience": 4,
              "px": 288, "regime": "finetune"}
GRID: list[dict] = [
    dict(BASE, name="early_depth", in_chans=4, geo_channels=(0,),
         notes="early fusion, RGB plus raw depth"),
    dict(BASE, name="early_height", in_chans=4, geo_channels=(1,),
         notes="early fusion, RGB plus floor relative height"),
    dict(BASE, name="early_geo3", in_chans=6, geo_channels=(0, 1, 2),
         notes="early fusion, RGB plus depth, height and validity mask"),
    dict(BASE, name="mid_geo", in_chans=6, geo_channels=(0, 1, 2), fusion="mid",
         notes="mid fusion, a second trunk on the geometry image"),
    dict(BASE, name="late_scalars", in_chans=3, scalar_dim=len(GEO_SCALARS),
         scalars=list(GEO_SCALARS),
         notes="late fusion, RGB trunk plus depth scalars and the support height"),
    dict(BASE, name="depth_only", in_chans=3, geo_channels=(0, 1, 2), use_rgb=False,
         notes="geometry only, the three channel geometry image through the trunk"),
]
ARMS: tuple[Arm, ...] = (
    Arm(REFERENCE, REFERENCE_EXP, "loro_focal_noes", "RGB only"),
    Arm("early_depth", EXP, "early_depth", "early fusion, depth"),
    Arm("early_height", EXP, "early_height", "early fusion, height"),
    Arm("early_geo3", EXP, "early_geo3", "early fusion, three geometry channels"),
    Arm("mid_geo", EXP, "mid_geo", "mid fusion, second trunk"),
    Arm("late_scalars", EXP, "late_scalars", "late fusion, depth scalars"),
    Arm("depth_only", EXP, "depth_only", "geometry only"),
)
RECIPE = ("convnext_tiny, focal gamma 2, no augmentation, no early stopping, 12 epochs, "
          "batch 32, fixed420_288, binary_plus_material3, holdout out-of-fold scores, "
          "leave-one-robot-out pooled threshold")


@dataclass
class FusionConfig:
    seeds: tuple[int, ...] = SEEDS
    budget: float = BUDGET
    draws: int = 2000
    boot_seed: int = 0


def frozen_reference() -> dict:
    summary = config.results_dir("e3") / "summary.json"
    if not summary.exists():
        return {}
    block = json.loads(summary.read_text()).get("modality", {})
    out: dict = {}
    for name, row in block.items():
        if isinstance(row, dict) and "recall_ff5_mean" in row:
            out[name] = {"recall_ff5_mean": row["recall_ff5_mean"],
                         "auprc_mean": row["auprc_mean"], "ff5_mean": row["ff5_mean"]}
    return out


def geometry_validity() -> dict:
    path = config.cache_path(config.DEFAULT_CROP)
    if not path.exists():
        return {}
    items = data.load_pool()
    cache = data.load_cache(config.DEFAULT_CROP, ("geo",))
    idx = np.array([{str(v): k for k, v in enumerate(cache["ids"])}[str(i)]
                    for i in items.ids], np.int64)
    valid = np.asarray(cache["geo"][idx, 2], np.float32).reshape(len(idx), -1).mean(1)
    return {"n": int(len(idx)), "valid_mean": float(valid.mean()),
            "valid_p10": float(np.percentile(valid, 10)), "valid_min": float(valid.min()),
            "share_below_0.8": float((valid < 0.8).mean())}


def run(cfg: FusionConfig) -> dict:
    present = {a.key: (a, seeds_present(a, cfg.seeds)) for a in ARMS}
    present = {k: v for k, v in present.items() if v[1]}
    assert REFERENCE in present, "the RGB reference arm has no scores"
    out: dict = {"exp": EXP, "name": "summary", "budget": cfg.budget, "reference": REFERENCE,
                 "seeds": list(cfg.seeds), "arms_present": sorted(present), "arms": {},
                 "paired_bootstrap": {}, "frozen_reference": frozen_reference(),
                 "geometry_validity": geometry_validity(), "recipe": RECIPE}
    for key, (arm, seeds) in present.items():
        out["arms"][key] = arm_block(arm, seeds, cfg.budget)
    ref_arm, ref_seeds = present[REFERENCE]
    for key, (arm, seeds) in present.items():
        if key == REFERENCE:
            continue
        shared = [s for s in seeds if s in ref_seeds]
        for s in shared:
            a = evaluate.load_scores(arm.exp, arm.name, seed=s)
            b = evaluate.load_scores(ref_arm.exp, ref_arm.name, seed=s)
            for strategy in STRATEGIES:
                out["paired_bootstrap"][f"{key}_minus_{REFERENCE}_seed{s}_{strategy}"] = compare(
                    a, b, strategy, cfg.budget, cfg.draws, cfg.boot_seed)
        if len(shared) > 1:
            a = evaluate.pooled_scores(arm.exp, arm.name, shared, "mean")
            b = evaluate.pooled_scores(ref_arm.exp, ref_arm.name, shared, "mean")
            for strategy in STRATEGIES:
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
        evaluate.run_grid(EXP, GRID, args)
        return 0
    cfg = replace(FusionConfig(), seeds=tuple(int(s) for s in args.seeds.split(",") if s),
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
