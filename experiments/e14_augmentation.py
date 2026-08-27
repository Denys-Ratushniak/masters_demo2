from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np

from trashnet import config, evaluate, metrics
from trashnet.arms import Arm, align, arm_block, compare, matched_recall, seeds_present

EXP = "e14"
REFERENCE_EXP = "e9"
REFERENCE = "none"
SEEDS: tuple[int, ...] = (0,)
BUDGET = 0.05
BASE: dict = {"action_weight": 1.0, "backbone": "convnext_tiny", "batch": 32,
              "crop": "fixed420_288", "early_stop": False, "epochs": 12, "freeze_frac": 0.0,
              "in_chans": 3, "label_space": "binary_plus_material3", "loss": "focal",
              "lr_head": 1e-3, "lr_trunk": 1e-4, "material_weight": 1.0, "oof_mode": "holdout",
              "patience": 4, "px": 288, "regime": "finetune"}
AUGMENTATIONS: tuple[tuple[str, str, str], ...] = (
    ("flip", "loro_focal_flip", "horizontal mirror"),
    ("color", "loro_focal_color", "brightness, contrast and saturation jitter"),
    ("geom", "loro_focal_geom", "small rotation, scale and shift"),
    ("all", "loro_focal_all", "flip, colour and geometry together"),
)
GRID: list[dict] = [
    dict(BASE, name=name, aug=aug,
         notes=f"final recipe with augmentation {aug}: {note}")
    for aug, name, note in AUGMENTATIONS
] + [dict(BASE, name="px224", aug="none", px=224,
          notes="final recipe with the trunk input downscaled to 224 px")]
ARMS: tuple[Arm, ...] = ((Arm(REFERENCE, REFERENCE_EXP, "loro_focal_noes", "no augmentation"),)
                         + tuple(Arm(aug, EXP, name, note) for aug, name, note in AUGMENTATIONS))
INPUT_SIZES: tuple[tuple[int, str, str], ...] = (
    (224, EXP, "px224"),
    (288, REFERENCE_EXP, "loro_focal_noes"),
    (384, "e13", "px384"),
)
FOLD_KEYS: tuple[str, ...] = ("auprc", "auroc", "recall_at_ff5_oracle", "train_time_s",
                              "gpu_mem_mb", "epochs_run")


@dataclass
class AugmentationConfig:
    seeds: tuple[int, ...] = SEEDS
    budget: float = BUDGET
    draws: int = 2000
    boot_seed: int = 0


def fold_stats(exp: str, name: str, seed: int) -> dict:
    rows = evaluate.load_results(exp, name, seed=seed)
    out = {}
    for key in FOLD_KEYS:
        v = np.array([float(r[key]) for r in rows], float)
        out[key] = {"mean": float(v.mean()),
                    "sd": float(np.std(v, ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}
    return out


def latency_lookup() -> dict[int, float]:
    summary = config.results_dir("e8") / "summary.json"
    if not summary.exists():
        return {}
    rows = json.loads(summary.read_text()).get("latency", {}).get("rows", [])
    return {int(r["px"]): float(r["median_ms"]) for r in rows
            if r.get("mode") == "fp16" and "median_ms" in r}


def input_size_block(cfg: AugmentationConfig) -> dict:
    latency = latency_lookup()
    out: dict = {}
    for px, exp, name in INPUT_SIZES:
        if not (config.results_dir(exp) / name).exists():
            continue
        pool = evaluate.load_scores(exp, name, seed=cfg.seeds[0])
        block = arm_block(Arm(str(px), exp, name), [cfg.seeds[0]], cfg.budget)
        out[str(px)] = {"exp": exp, "config": name,
                        "recall_matched_ff5": matched_recall(pool, cfg.budget),
                        "auprc_pooled": metrics.auprc(pool["y"], pool["score"]),
                        "folds": fold_stats(exp, name, cfg.seeds[0]),
                        "latency_ms_fp16": latency.get(px),
                        "point": block["per_seed"][str(cfg.seeds[0])]}
    return out


def run(cfg: AugmentationConfig) -> dict:
    present = {a.key: (a, seeds_present(a, cfg.seeds)) for a in ARMS}
    present = {k: v for k, v in present.items() if v[1]}
    assert REFERENCE in present, "the no-augmentation reference has no scores"
    out: dict = {"exp": EXP, "name": "summary", "budget": cfg.budget, "reference": REFERENCE,
                 "seeds": list(cfg.seeds), "arms_present": sorted(present), "arms": {},
                 "input_size": input_size_block(cfg), "bootstrap": {}}
    ref_arm, ref_seeds = present[REFERENCE]
    for key, (arm, seeds) in present.items():
        block = arm_block(arm, seeds, cfg.budget)
        block["folds"] = fold_stats(arm.exp, arm.name, seeds[0])
        out["arms"][key] = block
        if key == REFERENCE:
            continue
        a = evaluate.load_scores(arm.exp, arm.name, seed=seeds[0])
        b = evaluate.load_scores(ref_arm.exp, ref_arm.name, seed=ref_seeds[0])
        for strategy in ("emp", "cert"):
            out["bootstrap"][f"{key}_minus_{REFERENCE}_{strategy}"] = compare(
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
    cfg = replace(AugmentationConfig(),
                  seeds=tuple(int(s) for s in args.seeds.split(",") if s != ""),
                  draws=args.draws)
    out = run(cfg)
    evaluate.write_result(out)
    for key in out["arms_present"]:
        block = out["arms"][key]["per_seed"][str(cfg.seeds[0])]
        print(f"{key:8s} auprc={block['auprc_fold_mean']:.3f} "
              f"matched={block['matched_recall']:.3f} "
              f"recall={block['emp']['recall']:.3f} ff={block['emp']['ff']:.3f}")
    for px, block in sorted(out["input_size"].items(), key=lambda kv: int(kv[0])):
        print(f"{px:>4s} px matched={block['recall_matched_ff5']:.3f} "
              f"auprc={block['auprc_pooled']:.3f} latency={block['latency_ms_fp16']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
