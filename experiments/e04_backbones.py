from __future__ import annotations


from trashnet import config, data, evaluate, features, models

EXP = "e4"
LR_EXP = "e4lr"
SEEDS: tuple[int, ...] = (0,)
FROZEN_SEEDS: tuple[int, ...] = (0, 1, 2)
BACKBONES: tuple[str, ...] = ("convnext_tiny", "resnet50", "efficientnet_b0", "resnet18",
                              "mobilenetv3_large_100", "vit_small_patch14_dinov2")
PROBE_BACKBONES: tuple[str, ...] = ("convnext_base",)
PROBE_FOLDS: tuple[str, ...] = config.TRAIN_ROBOTS[3:5]
LR_GRID: tuple[float, ...] = (1e-4, 2e-4, 5e-4)
EPOCH_GRID: tuple[int, ...] = (12, 20)
LR_FOLD = config.TRAIN_ROBOTS[3] if config.TRAIN_ROBOTS else ""
CHOSEN_LR, CHOSEN_EPOCHS = 1e-4, 12
BASE: dict = {"crop": "fixed420_288", "label_space": "binary_plus_material3",
              "material_weight": 1.0, "aug": "none", "batch": 32, "lr_head": 1e-3,
              "px": 288, "early_stop": True, "in_chans": 3, "patience": 4, "oof_mode": "cv"}

LR_TUNE_GRID: list[dict] = [
    dict(BASE, name=f"ct_lr{lr:g}_ep{ep}", backbone="convnext_tiny", regime="finetune",
         epochs=ep, lr_trunk=lr, oof_mode="holdout",
         notes="single fold learning rate and epoch screen, selected on inner validation AUPRC")
    for lr in LR_GRID for ep in EPOCH_GRID
]
GRID: list[dict] = (
    [dict(BASE, name=f"{b}_finetune", backbone=b, regime="finetune", epochs=CHOSEN_EPOCHS,
          lr_trunk=CHOSEN_LR, freeze_frac=0.0)
     for b in BACKBONES + PROBE_BACKBONES]
    + [dict(BASE, name=f"{b}_frozen", backbone=b, regime="frozen", head_epochs=200,
            head_batch=256)
       for b in BACKBONES + PROBE_BACKBONES]
    + [dict(BASE, name="vit_small_patch14_dinov2_finetune_lr3e-5",
            backbone="vit_small_patch14_dinov2", regime="finetune", epochs=CHOSEN_EPOCHS,
            lr_trunk=3e-5, freeze_frac=0.0, oof_mode="holdout",
            notes="ViT rerun with a lower trunk learning rate after collapse at 1e-4")]
)


def pick_lr() -> dict:
    rows = []
    for cfg in LR_TUNE_GRID:
        for r in evaluate.load_results(LR_EXP, cfg["name"]):
            if r.get("best_val_auprc") is not None:
                rows.append(r)
    if not rows:
        return {"chosen": {"lr_trunk": CHOSEN_LR, "epochs": CHOSEN_EPOCHS}, "runs": []}
    best = max(rows, key=lambda r: float(r["best_val_auprc"]))
    return {"chosen": {"lr_trunk": best["config"]["lr_trunk"],
                       "epochs": best["config"]["epochs"]},
            "criterion": "maximum inner validation AUPRC on a single fold",
            "runs": [{"name": r["config"]["name"], "best_val_auprc": r["best_val_auprc"],
                      "auprc": r["auprc"], "recall_ff5": r["recall_at_ff5"]} for r in rows]}


def backbone_cost() -> dict:
    out = {}
    for b in BACKBONES + PROBE_BACKBONES:
        trunk, dim = features.build_trunk(b, pretrained=False)
        out[b] = {"params_m": models.param_count_m(trunk), "feature_dim": dim,
                  "required_px": features.required_px(b)}
        del trunk
    return out


def main(argv: list[str] | None = None) -> int:
    ap = evaluate.build_parser(EXP, SEEDS)
    ap.add_argument("--stage", default="main", choices=("lr", "main"))
    ap.add_argument("--cost", action="store_true")
    args = ap.parse_args(argv)
    if args.stage == "lr":
        args.folds = LR_FOLD if args.folds == "all" else args.folds
        evaluate.run_grid(LR_EXP, LR_TUNE_GRID, args)
        if not args.dry_run:
            chosen = pick_lr()
            evaluate.write_result(dict(chosen, exp=LR_EXP, name="summary"))
            print("chosen:", chosen["chosen"])
        return 0
    if args.dry_run:
        evaluate.run_grid(EXP, GRID, args)
        return 0
    items = data.load_pool()
    for cfg in evaluate.select(GRID, args.only):
        sub = evaluate.build_parser(EXP, SEEDS).parse_args([])
        sub.folds = args.folds
        sub.force = args.force
        sub.seeds = (",".join(str(s) for s in FROZEN_SEEDS)
                     if cfg["regime"] == "frozen" else args.seeds)
        if cfg["backbone"] in PROBE_BACKBONES and sub.folds == "all":
            sub.folds = ",".join(PROBE_FOLDS)
        evaluate.run_grid(EXP, [cfg], sub, items=items)
    order = [c["name"] for c in GRID]
    sums = evaluate.summaries(EXP, order)
    reference = "convnext_tiny_frozen" if "convnext_tiny_frozen" in sums else order[0]
    block = evaluate.variant_block(sums, order, reference)
    finetuned = {n: v for n, v in block.items()
                 if n.endswith("_finetune") and not n.startswith(tuple(PROBE_BACKBONES))}
    summary = {"exp": EXP, "name": "summary", "reference": reference, "backbones": block,
               "lr_tune": pick_lr(),
               "best_finetune_recall": evaluate.best_by(finetuned) if finetuned else None,
               "best_finetune_auprc": (max(finetuned, key=lambda n: finetuned[n]["auprc_mean"])
                                       if finetuned else None),
               "cost": backbone_cost() if args.cost else {},
               "issues": evaluate.check(EXP)}
    evaluate.write_result(summary)
    print(evaluate.variant_table(sums, order, reference))
    print("best by recall@FF5:", summary["best_finetune_recall"],
          "| best by AUPRC:", summary["best_finetune_auprc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
