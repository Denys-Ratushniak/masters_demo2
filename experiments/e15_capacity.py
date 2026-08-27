from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from trashnet import config, evaluate, features, models
from trashnet.arms import Arm, arm_block, compare, matched_bootstrap, seeds_present
from trashnet.metrics import group_bootstrap_sets

EXP = "e15"
REFERENCE_EXP = "e9"
REFERENCE = "ct_h128d2"
SEEDS: tuple[int, ...] = (0, 1, 2)
BUDGET = 0.05
LATENCY_ITERS, LATENCY_WARMUP = 200, 40
EDGE_FACTOR = 2.832
EDGE_DEVICE = "RTX A2000 12 GB"
BASE: dict = {"action_weight": 1.0, "aug": "none", "backbone": "convnext_tiny", "batch": 32,
              "crop": "fixed420_288", "early_stop": False, "epochs": 12, "freeze_frac": 0.0,
              "in_chans": 3, "label_space": "binary_plus_material3", "loss": "focal",
              "lr_head": 1e-3, "lr_trunk": 1e-4, "material_weight": 1.0, "oof_mode": "holdout",
              "patience": 4, "px": 288, "regime": "finetune"}
GRID: list[dict] = [
    dict(BASE, name="h256d3", hidden=256, depth=3, notes="wider and deeper head"),
    dict(BASE, name="h512d2", hidden=512, depth=2, notes="wider head"),
    dict(BASE, name="h512d3", hidden=512, depth=3, notes="widest and deepest head"),
    dict(BASE, name="swin", backbone="swin_tiny_patch4_window7_224", px=224,
         notes="Swin-T trunk, the crop is resized to its native 224 px"),
    dict(BASE, name="swin_lr3e-5", backbone="swin_tiny_patch4_window7_224", px=224,
         lr_trunk=3e-5, notes="Swin-T fallback for a lower trunk learning rate"),
]
ARMS: tuple[Arm, ...] = (
    Arm(REFERENCE, REFERENCE_EXP, "loro_focal_noes", "ConvNeXt-T, head 128 by 2"),
    Arm("ct_h256d3", EXP, "h256d3", "ConvNeXt-T, head 256 by 3"),
    Arm("ct_h512d2", EXP, "h512d2", "ConvNeXt-T, head 512 by 2"),
    Arm("ct_h512d3", EXP, "h512d3", "ConvNeXt-T, head 512 by 3"),
    Arm("swin_tiny", EXP, "swin", "Swin-T trunk"),
    Arm("swin_tiny_lr3e-5", EXP, "swin_lr3e-5", "Swin-T trunk, lower trunk learning rate"),
    Arm("convnext_base", "e13", "cbase", "ConvNeXt-B trunk"),
    Arm("vit_s_lr3e-5", "e4", "vit_small_patch14_dinov2_finetune_lr3e-5",
        "ViT-S DINOv2 trunk, a different recipe"),
)
COST_ROWS: tuple[tuple[str, str, int, int, int], ...] = (
    ("ct_h128d2", "convnext_tiny", 128, 2, 288),
    ("ct_h256d3", "convnext_tiny", 256, 3, 288),
    ("ct_h512d2", "convnext_tiny", 512, 2, 288),
    ("ct_h512d3", "convnext_tiny", 512, 3, 288),
    ("convnext_base", "convnext_base", 128, 2, 288),
    ("swin_tiny", "swin_tiny_patch4_window7_224", 128, 2, 288),
    ("vit_s_lr3e-5", "vit_small_patch14_dinov2", 128, 2, 288),
)


@dataclass
class CapacityConfig:
    seeds: tuple[int, ...] = SEEDS
    budget: float = BUDGET
    draws: int = 2000
    boot_seed: int = 0


def cost_table() -> dict:
    rows = {}
    for key, backbone, hidden, depth, px in COST_ROWS:
        model = models.TrashNet(backbone, 3, hidden=hidden, depth=depth, in_chans=3,
                                pretrained=False, px=px)
        row = {"backbone": backbone, "hidden": hidden, "depth": depth, "px_in": px,
               "px_model": features.required_px(backbone) or px,
               "params_m": models.param_count_m(model),
               "params_head_m": models.param_count_m(model.head)}
        try:
            bench = models.benchmark_latency(model, px, "fp16", LATENCY_ITERS, LATENCY_WARMUP)
            row.update({"median_ms": bench["median_ms"], "gpu_mem_mb": bench["gpu_mem_mb"],
                        "edge_ms": bench["median_ms"] * EDGE_FACTOR})
        except Exception as err:
            row["error"] = str(err)[:300]
        rows[key] = row
        del model
    return {"rows": rows, "edge_device": EDGE_DEVICE, "edge_factor": EDGE_FACTOR,
            "iters": LATENCY_ITERS, "warmup": LATENCY_WARMUP}


def run(cfg: CapacityConfig, with_cost: bool = False) -> dict:
    present = {a.key: (a, seeds_present(a, cfg.seeds)) for a in ARMS}
    present = {k: v for k, v in present.items() if v[1]}
    assert REFERENCE in present, "the reference head has no leave-one-robot-out scores"
    out: dict = {"exp": EXP, "name": "summary", "budget": cfg.budget, "reference": REFERENCE,
                 "seeds": list(cfg.seeds), "arms_present": sorted(present), "arms": {},
                 "paired_bootstrap": {}, "cost": cost_table() if with_cost else {}}
    for key, (arm, seeds) in present.items():
        out["arms"][key] = arm_block(arm, seeds, cfg.budget)
    ref_arm, ref_seeds = present[REFERENCE]
    ref_pool = evaluate.load_scores(ref_arm.exp, ref_arm.name, seed=ref_seeds[0])
    sets = group_bootstrap_sets(ref_pool["group"], ref_pool["y"], cfg.draws, cfg.boot_seed)
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
            out["paired_bootstrap"][f"{key}_minus_{REFERENCE}_ensemble_matched"] = \
                matched_bootstrap(b, a["score"], b["score"], cfg.budget, sets)
    out["issues"] = evaluate.check(EXP)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = evaluate.build_parser(EXP, SEEDS)
    ap.add_argument("--stage", default="loro", choices=("loro", "eval"))
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--cost", action="store_true")
    args = ap.parse_args(argv)
    if args.dry_run:
        evaluate.run_grid(EXP, GRID, args)
        print(f"{EXP}/stage eval")
        return 0
    if args.stage == "loro":
        evaluate.run_grid(EXP, GRID, args)
        return 0
    cfg = replace(CapacityConfig(),
                  seeds=tuple(int(s) for s in args.seeds.split(",") if s != ""),
                  draws=args.draws)
    out = run(cfg, args.cost)
    evaluate.write_result(out)
    for key in out["arms_present"]:
        agg = out["arms"][key]["agg"]
        print(f"{key:18s} auprc={agg['mean']['auprc_fold_mean']:.3f} "
              f"matched={agg['mean']['matched_recall']:.3f} "
              f"recall={agg['mean']['emp_recall']:.3f}±{agg['sd']['emp_recall']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
