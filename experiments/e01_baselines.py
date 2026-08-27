from __future__ import annotations

import numpy as np

from trashnet import baselines, config, data, evaluate
from trashnet.folds import loro_folds

EXP = "e1"
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
SINGLE_SEED_CONFIGS: tuple[str, ...] = ("a_flush_all", "b_oracle_tp", "c_oracle_tp_area_px",
                                        "d_oracle_tp_area_cm2", "e_oracle_tp_support",
                                        "h_tinycnn")
CROP = "fixed420_288"
NATIVE_RULES: dict[str, str] = {"a_flush_all": "flush_all", "b_oracle_tp": "oracle_tp"}

GRID: list[dict] = [
    {"name": "a_flush_all", "regime": "rule", "rule": "flush_all", "crop": CROP,
     "notes": "flush everything"},
    {"name": "b_oracle_tp", "regime": "rule", "rule": "oracle_tp", "crop": CROP,
     "notes": "flush iff the material label is tp"},
    {"name": "c_oracle_tp_area_px", "regime": "rule", "rule": "oracle_tp_area_px", "crop": CROP,
     "notes": "oracle tp and bbox pixel area below the out-of-fold threshold"},
    {"name": "d_oracle_tp_area_cm2", "regime": "rule", "rule": "oracle_tp_area_cm2", "crop": CROP,
     "notes": "oracle tp and labelled area_cm2 below the out-of-fold threshold"},
    {"name": "e_oracle_tp_support", "regime": "rule", "rule": "oracle_tp_support", "crop": CROP,
     "notes": "oracle tp and support_rel_mm below the threshold, null support never flushes"},
    {"name": "f_logreg_scalars", "regime": "frozen", "head": "logreg", "logreg_c": 0.1,
     "feature_source": "scalars,bbox", "crop": CROP, "early_stop": False,
     "notes": "logistic regression on 9 depth scalars plus bbox width, height, area"},
    {"name": "g_logreg_scalars_tp", "regime": "frozen", "head": "logreg", "logreg_c": 0.1,
     "feature_source": "scalars,bbox,tpflag", "crop": CROP, "early_stop": False,
     "notes": "logistic regression on scalars, bbox and the oracle tp flag"},
    {"name": "h_tinycnn", "regime": "finetune", "backbone": "tinycnn", "crop": CROP,
     "epochs": 30, "batch": 32, "lr_trunk": 1e-3, "lr_head": 1e-3, "aug": "none", "px": 64,
     "early_stop": True, "patience": 6,
     "notes": "three block CNN trained from scratch at 64 px"},
]


def seeds_for(name: str, requested: list[int]) -> list[int]:
    return requested[:1] if name in SINGLE_SEED_CONFIGS else requested


def native_point(name: str, items: data.Items, folds) -> tuple[float, float]:
    rule = NATIVE_RULES.get(name)
    if rule is None:
        return float("nan"), float("nan")
    pairs = [baselines.native_point(rule, items, f.test) for f in folds]
    return float(np.mean([a for a, _ in pairs])), float(np.mean([b for _, b in pairs]))


def report(items: data.Items) -> dict:
    folds = loro_folds(items)
    names = [c["name"] for c in GRID if (config.results_dir(EXP) / c["name"]).exists()]
    out: dict = {"exp": EXP, "name": "summary", "configs": {}, "order": names}
    for name in names:
        s = evaluate.config_summary(EXP, name)
        recall, ff = native_point(name, items, folds)
        out["configs"][name] = {
            "n": s["n"], "folds": s["folds"], "seeds": s["seeds"],
            "auprc_mean": float(np.nanmean(s["auprc"])),
            "auprc_sd": float(np.nanstd(s["auprc"], ddof=1)),
            "recall_ff5_mean": float(np.nanmean(s["r_ff5"])),
            "recall_ff5_sd": float(np.nanstd(s["r_ff5"], ddof=1)),
            "recall_ff5_per_fold": s["r_ff5"].tolist(),
            "ff5_mean": float(np.nanmean(s["ff5"])),
            "ff5_sd": float(np.nanstd(s["ff5"], ddof=1)),
            "recall_ff5_oracle_mean": float(np.nanmean(s["r_ff5_oracle"])),
            "recall_ff5_oracle_sd": float(np.nanstd(s["r_ff5_oracle"], ddof=1)),
            "pooled_ff1": s.get("pooled_ff1"), "pooled_ff2": s.get("pooled_ff2"),
            "ff_per_material_ff5": s.get("mat_ff5"),
            "native_recall": recall, "native_ff": ff,
        }
    out["issues"] = evaluate.check(EXP)
    return out


def table(summary: dict) -> str:
    cols = ["baseline", "n", "AUPRC", "recall@FF5", "FF5", "recall@FF5 oracle",
            "pooled FF1", "pooled FF2", "native recall", "native FF"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name in summary["order"]:
        c = summary["configs"][name]
        cells = [name, str(c["n"]),
                 f"{c['auprc_mean']:.3f}±{c['auprc_sd']:.3f}",
                 f"{c['recall_ff5_mean']:.3f}±{c['recall_ff5_sd']:.3f}",
                 f"{c['ff5_mean']:.3f}±{c['ff5_sd']:.3f}",
                 f"{c['recall_ff5_oracle_mean']:.3f}±{c['recall_ff5_oracle_sd']:.3f}",
                 f"{c['pooled_ff1']:.3f}" if c["pooled_ff1"] is not None else "-",
                 f"{c['pooled_ff2']:.3f}" if c["pooled_ff2"] is not None else "-",
                 f"{c['native_recall']:.3f}" if np.isfinite(c["native_recall"]) else "-",
                 f"{c['native_ff']:.3f}" if np.isfinite(c["native_ff"]) else "-"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = evaluate.build_parser(EXP, SEEDS).parse_args(argv)
    requested = [int(s) for s in args.seeds.split(",") if s != ""]
    cfgs = evaluate.select(GRID, args.only)
    if args.dry_run:
        for cfg in cfgs:
            for seed in seeds_for(cfg["name"], requested):
                print(f"{EXP}/{cfg['name']} folds={args.folds} seed={seed}")
        return 0
    items = data.load_pool()
    args.only = ""
    for cfg in cfgs:
        args.seeds = ",".join(str(s) for s in seeds_for(cfg["name"], requested))
        evaluate.run_grid(EXP, [cfg], args, items=items)
    summary = report(items)
    evaluate.write_result(summary)
    print(table(summary))
    print("issues:", summary["issues"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
