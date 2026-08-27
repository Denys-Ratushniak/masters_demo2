from __future__ import annotations

import numpy as np

from trashnet import config, data, evaluate, metrics, transfer
from trashnet.folds import loro_folds, subsample_groups

EXP = "e7b"
SEEDS: tuple[int, ...] = (0,)
FRAC_SEED = 0
TARGET_N = 3400
BUDGET = 0.05
BASE: dict = {"action_weight": 1.0, "aug": "none", "backbone": "convnext_tiny", "batch": 32,
              "crop": "fixed420_288", "early_stop": False, "epochs": 12, "freeze_frac": 0.0,
              "in_chans": 3, "label_space": "binary_plus_material3", "loss": "focal",
              "lr_head": 1e-3, "lr_trunk": 1e-4, "material_weight": 1.0, "oof_mode": "holdout",
              "patience": 4, "px": 288, "regime": "finetune"}
POINTS: tuple[tuple[float, str, str], ...] = (
    (0.125, EXP, "e7b_f0p125"),
    (0.25, EXP, "e7b_f0p25"),
    (0.5, EXP, "e7b_f0p5"),
    (0.75, EXP, "e7b_f0p75"),
    (1.0, "e9", "loro_focal_noes"),
)
GRID: list[dict] = [
    dict(BASE, name=name, train_frac=frac,
         oof_mode="cv" if frac <= 0.125 else BASE["oof_mode"],
         notes=f"{int(frac * 100)} percent of the training robot-days under the final recipe")
    for frac, exp, name in POINTS if exp == EXP
]


def train_sizes(items: data.Items, frac: float) -> dict:
    rows = []
    for fold in loro_folds(items):
        idx = subsample_groups(items, fold.train, frac, FRAC_SEED)
        rows.append({"fold": fold.name, "n_items": int(idx.size),
                     "n_groups": int(np.unique(items.group[idx]).size),
                     "n_pos": int((items.y[idx] == 1).sum()),
                     "n_neg": int((items.y[idx] == 0).sum())})
    return {"per_fold": rows,
            "n_items_mean": float(np.mean([r["n_items"] for r in rows])),
            "n_groups_mean": float(np.mean([r["n_groups"] for r in rows]))}


def point_metrics(exp: str, name: str, frac: float, items: data.Items) -> dict:
    pool = evaluate.load_scores(exp, name, seed=0)
    out: dict = {"exp": exp, "config": name, "frac": frac,
                 "auprc_pooled": metrics.auprc(pool["y"], pool["score"]),
                 "auroc_pooled": metrics.auroc(pool["y"], pool["score"]),
                 "auprc_per_robot": {str(r): metrics.auprc(pool["y"][pool["robot"] == r],
                                                           pool["score"][pool["robot"] == r])
                                     for r in np.unique(pool["robot"])},
                 "train": train_sizes(items, frac)}
    for b in config.BUDGETS:
        k = config.budget_key(b)
        for strat in ("emp", "cert"):
            applied = transfer.apply_strategy(pool, strat, b)
            block = transfer.score_flags(pool, applied["flags"], b)
            block["thresholds"] = applied["thresholds"]
            out[f"loro_{strat}_{k}"] = block
        out[f"oracle_recall_{k}"] = metrics.recall_at_ff_oracle(pool["y"], pool["score"], b)
    per_robot = out[f"loro_emp_{config.budget_key(BUDGET)}"]["recall_per_robot"]
    v = np.array([per_robot[r] for r in sorted(per_robot)], float)
    out["recall_ff5_robot_mean"] = float(np.nanmean(v))
    out["recall_ff5_robot_sd"] = float(np.nanstd(v, ddof=1))
    return out


def main(argv: list[str] | None = None) -> int:
    args = evaluate.build_parser(EXP, SEEDS).parse_args(argv)
    if args.dry_run:
        for cfg in evaluate.select(GRID, args.only):
            print(f"{EXP}/{cfg['name']} train_frac={cfg['train_frac']} folds=all seed=0")
        print(f"{EXP}/1.0 point reused from e9/loro_focal_noes")
        return 0
    items = data.load_pool()
    for cfg in evaluate.select(GRID, args.only):
        sub = evaluate.build_parser(EXP, SEEDS).parse_args([])
        sub.folds, sub.force, sub.seeds = args.folds, args.force, args.seeds
        sub.train_frac, sub.frac_seed = float(cfg["train_frac"]), FRAC_SEED
        evaluate.run_grid(EXP, [cfg], sub, items=items)
    points = {}
    for frac, exp, name in POINTS:
        if (config.results_dir(exp) / name).exists():
            points[f"{frac:g}"] = point_metrics(exp, name, frac, items)
    fracs = sorted(points, key=float)
    n = np.array([points[f]["train"]["n_items_mean"] for f in fracs], float)
    summary = {"exp": EXP, "name": "summary", "base": BASE, "frac_seed": FRAC_SEED,
               "target_n": TARGET_N, "budget": BUDGET, "points": points,
               "series": {}, "trend": {}}
    key5 = config.budget_key(BUDGET)
    for key, getter in (("auprc_pooled", lambda p: p["auprc_pooled"]),
                        ("recall_ff5_pooled", lambda p: p[f"loro_emp_{key5}"]["recall"]),
                        ("ff5_pooled", lambda p: p[f"loro_emp_{key5}"]["ff"]),
                        ("recall_ff5_robot_mean", lambda p: p["recall_ff5_robot_mean"])):
        v = np.array([getter(points[f]) for f in fracs], float)
        summary["series"][key] = {"n_train": n.tolist(), "value": v.tolist(),
                                  "fracs": [float(f) for f in fracs]}
        if len(v) >= 3:
            summary["trend"][key] = metrics.fit_trend(n, v, TARGET_N)
    if len(fracs) >= 2:
        last = summary["series"]["recall_ff5_robot_mean"]["value"]
        summary["last_step_delta_recall"] = float(last[-1] - last[-2])
        summary["last_step_delta_auprc"] = float(
            summary["series"]["auprc_pooled"]["value"][-1]
            - summary["series"]["auprc_pooled"]["value"][-2])
    summary["issues"] = evaluate.check(EXP)
    evaluate.write_result(summary)
    for f in fracs:
        p = points[f]
        print(f"frac={f} n={p['train']['n_items_mean']:.0f} auprc={p['auprc_pooled']:.3f} "
              f"recall@FF5={p[f'loro_emp_{key5}']['recall']:.3f} "
              f"robot mean={p['recall_ff5_robot_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
