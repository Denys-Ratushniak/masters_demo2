from __future__ import annotations

from trashnet import config, data, evaluate

EXP = "e4b"
SEEDS: tuple[int, ...] = (0,)
BASE: dict = {"backbone": "convnext_tiny", "regime": "finetune", "crop": "fixed420_288",
              "label_space": "binary_plus_material3", "material_weight": 1.0, "aug": "none",
              "epochs": 12, "batch": 32, "lr_trunk": 1e-4, "lr_head": 1e-3, "px": 288,
              "early_stop": True, "in_chans": 3, "patience": 4, "freeze_frac": 0.0,
              "oof_mode": "holdout"}
REFERENCE = "freeze0.0"
DEPTH_SCALARS = list(config.SCALARS)
SUPPORT_SCALARS = DEPTH_SCALARS + list(config.EXTRA_SCALARS[:2])

VARIATIONS: list[dict] = [
    {"name": "freeze0.0", "freeze_frac": 0.0},
    {"name": "freeze0.5", "freeze_frac": 0.5},
    {"name": "freeze0.8", "freeze_frac": 0.8},
    {"name": "freeze1.0", "freeze_frac": 1.0},
    {"name": "aug_flip", "aug": "flip"},
    {"name": "aug_color", "aug": "color"},
    {"name": "aug_geom", "aug": "geom"},
    {"name": "aug_all", "aug": "all"},
    {"name": "mat0.0", "material_weight": 0.0, "label_space": "binary"},
    {"name": "crop_prop_288", "crop": "prop_288"},
    {"name": "crop_bbox_square", "crop": "bbox_square"},
    {"name": "crop_bbox_warp", "crop": "bbox_warp"},
    {"name": "patience8", "epochs": 20, "patience": 8},
    {"name": "sc_depth9", "scalar_dim": 9, "scalars": DEPTH_SCALARS},
    {"name": "sc_support", "scalar_dim": 11, "scalars": SUPPORT_SCALARS},
    {"name": "focal", "loss": "focal"},
    {"name": "focal_flip", "loss": "focal", "aug": "flip"},
    {"name": "labelsmooth0.1", "label_smooth": 0.1},
]
GRID: list[dict] = [dict(BASE, **v) for v in VARIATIONS]


def main(argv: list[str] | None = None) -> int:
    args = evaluate.build_parser(EXP, SEEDS).parse_args(argv)
    if args.dry_run:
        evaluate.run_grid(EXP, GRID, args)
        return 0
    evaluate.run_grid(EXP, GRID, args, items=data.load_pool())
    order = [c["name"] for c in GRID]
    sums = evaluate.summaries(EXP, order)
    block = evaluate.variant_block(sums, order, REFERENCE)
    summary = {"exp": EXP, "name": "summary", "reference": REFERENCE, "base": BASE,
               "regime": block,
               "paired": evaluate.paired_vs_reference(EXP, list(block), REFERENCE),
               "best_recall": evaluate.best_by(block),
               "best_auprc": max(block, key=lambda n: block[n]["auprc_mean"]) if block else None,
               "issues": evaluate.check(EXP)}
    evaluate.write_result(summary)
    print(evaluate.variant_table(sums, order, REFERENCE))
    print("best by recall@FF5:", summary["best_recall"],
          "| best by AUPRC:", summary["best_auprc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
