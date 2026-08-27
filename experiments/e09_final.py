from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from trashnet import baselines, config, data, evaluate, metrics, models, transfer
from trashnet.folds import inner_folds
from trashnet.train import Batch, TrainConfig, predict, train_finetune

EXP = "e9"
SEEDS: tuple[int, ...] = (0, 1, 2)
BUDGET = 0.05
DELTA = 0.05
MIN_ROBOT_N = 10
BOOTSTRAP_DRAWS = 2000
TEST_REASON = "e9 final evaluation on the held out mixed robot set"
BASE: dict = {"action_weight": 1.0, "backbone": "convnext_tiny", "batch": 32,
              "crop": "fixed420_288", "epochs": 12, "freeze_frac": 0.0, "in_chans": 3,
              "label_space": "binary_plus_material3", "loss": "focal", "lr_head": 1e-3,
              "lr_trunk": 1e-4, "material_weight": 1.0, "patience": 4, "px": 288,
              "regime": "finetune", "oof_mode": "holdout"}
GRID: list[dict] = [
    dict(BASE, name="loro_focal", aug="none", early_stop=True,
         notes="focal loss, no augmentation, early stopping on the inner split"),
    dict(BASE, name="loro_focal_flip", aug="flip", early_stop=True,
         notes="focal loss with horizontal flip augmentation"),
    dict(BASE, name="loro_focal_noes", aug="none", early_stop=False,
         notes="focal loss, no augmentation, full 12 epochs on the whole fold"),
]
SELECTION_KEY = "cert"
MODEL_DIR = config.RESULTS / "models"


def model_path(name: str, seed: int) -> Path:
    return MODEL_DIR / f"final_{name}_seed{seed}.pt"


def loro_block(pool: dict) -> dict:
    out: dict = {"auprc": metrics.auprc(pool["y"], pool["score"]),
                 "auroc": metrics.auroc(pool["y"], pool["score"])}
    for b in config.BUDGETS:
        k = config.budget_key(b)
        for strat in ("emp", "cert"):
            applied = transfer.apply_strategy(pool, strat, b)
            block = transfer.score_flags(pool, applied["flags"], b)
            block["thresholds"] = applied["thresholds"]
            out[f"{k}_{strat}"] = block
    return out


def select_variant(names: list[str], seeds: list[int]) -> dict:
    out: dict = {"variants": {}}
    for name in names:
        per_seed = {}
        for s in seeds:
            try:
                per_seed[str(s)] = loro_block(evaluate.load_scores(EXP, name, seed=s))
            except AssertionError:
                continue
        if not per_seed:
            continue
        ens = loro_block(evaluate.pooled_scores(EXP, name, [int(s) for s in per_seed], "mean"))
        key = f"{config.budget_key(BUDGET)}_{SELECTION_KEY}"
        recalls = [v[key]["recall"] for v in per_seed.values()]
        out["variants"][name] = {"per_seed": per_seed, "ensemble": ens,
                                 "mean_recall": float(np.mean(recalls)),
                                 "sd_recall": float(np.std(recalls, ddof=1))
                                 if len(recalls) > 1 else 0.0}
    assert out["variants"], "no LORO results to select from"
    out["chosen_variant"] = max(out["variants"], key=lambda n: out["variants"][n]["mean_recall"])
    out["chosen_by"] = (f"mean leave-one-robot-out certified recall at FF{int(BUDGET * 100)} "
                        "over seeds")
    chosen = out["variants"][out["chosen_variant"]]
    key = f"{config.budget_key(BUDGET)}_{SELECTION_KEY}"
    ens_recall = chosen["ensemble"][key]["recall"]
    seed0 = chosen["per_seed"].get("0", {}).get(key, {}).get("recall", float("-inf"))
    use_ens = (ens_recall - chosen["mean_recall"]) > 0 and (ens_recall - seed0) > 0
    out["score_aggregation"] = "mean" if use_ens else "seed0"
    out["gain_vs_mean_seed"] = float(ens_recall - chosen["mean_recall"])
    out["gain_vs_seed0"] = float(ens_recall - seed0)
    return out


def train_deployment(name: str, cfg: dict, items: data.Items, seeds: list[int]) -> dict:
    tcfg = TrainConfig.from_dict(cfg)
    prep = evaluate.prepare(items, cfg, tcfg, "finetune")
    items = prep["items"]
    pool = Batch(prep["x"], items.y, prep.get("m"), None)
    allidx = np.arange(len(items))
    out: dict = {"config": cfg, "n_pool": len(items), "seeds": {}}
    for seed in seeds:
        if tcfg.early_stop:
            core, es = inner_folds(items, allidx, seed=seed)[0]
            val = pool.take(es)
        else:
            core, es, val = allidx, np.zeros(0, np.int64), None
        model, info = train_finetune(tcfg, pool.take(core), val=val, seed=seed)
        models.save_checkpoint(model, tcfg.as_dict(), model_path(name, seed),
                               extra={"seed": seed, "n_train": int(core.size),
                                      "n_val": int(es.size),
                                      "classes3": list(config.MATERIALS3)})
        out["seeds"][str(seed)] = {"n_train": int(core.size), "n_val": int(es.size),
                                   "epochs_run": info["epochs_run"],
                                   "train_time_s": info["train_time_s"],
                                   "gpu_mem_mb": info["gpu_mem_mb"]}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def score_items(name: str, cfg: dict, items: data.Items,
                seeds: list[int]) -> tuple[np.ndarray, np.ndarray | None]:
    tcfg = TrainConfig.from_dict(cfg)
    x = data.prep_input(items.rgb, items.geo, tcfg.in_chans, tcfg.geo_channels,
                        tcfg.use_rgb)
    per_seed, materials = [], []
    for seed in seeds:
        model = models.load_checkpoint(model_path(name, seed), px=tcfg.px)
        action, material = predict(model, x, None, batch=64)
        per_seed.append(action)
        if material is not None:
            materials.append(material)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.stack(per_seed), (np.mean(materials, 0) if materials else None)


def as_pool(items: data.Items, score: np.ndarray) -> dict:
    return {"ids": items.ids, "score": score.astype(float), "y": items.y.astype(int),
            "mat": items.mat, "group": items.group, "source": items.source,
            "robot": items.robot}


def baseline_block(test_items: data.Items, train_items: data.Items) -> dict:
    out = {}
    y_te = test_items.y.astype(int)
    y_fit = train_items.y.astype(int)
    for rule in ("flush_all", "oracle_tp", "oracle_tp_area_px"):
        s_te = baselines.rule_score(rule, test_items)
        s_fit = baselines.rule_score(rule, train_items)
        recall, ff = baselines.native_point(rule, test_items, np.arange(len(test_items)))
        block = {"auprc": metrics.auprc(y_te, s_te), "auroc": metrics.auroc(y_te, s_te),
                 "native_recall": recall, "native_ff": ff}
        for b in config.BUDGETS:
            k = config.budget_key(b)
            t = metrics.threshold_at_ff(y_fit, s_fit, b)
            block[k] = {"threshold": float(t), "recall": metrics.recall(y_te, s_te, t),
                        "ff": metrics.ff(y_te, s_te, t),
                        "recall_oracle": metrics.recall_at_ff_oracle(y_te, s_te, b)}
        out[rule] = block
    return out


def evaluate_test(name: str, cfg: dict, seeds: list[int], mode: str, thresholds: dict,
                  train_pool: dict) -> dict:
    items = data.load_test(reason=TEST_REASON)
    items = data.attach(items, data.load_cache(cfg.get("crop", config.DEFAULT_CROP),
                                               ("rgb", "sc")), ("rgb", "sc"))
    per_seed, p_material = score_items(name, cfg, items,
                                       seeds if mode == "mean" else seeds[:1])
    score = per_seed.mean(0)
    pool = as_pool(items, score)
    out: dict = {
        "variant": name, "score_aggregation": mode, "seeds": seeds,
        "n": len(items), "n_pos": int((items.y == 1).sum()), "n_neg": int((items.y == 0).sum()),
        "n_robots": int(np.unique(items.robot).size),
        "n_robot_days": int(np.unique(items.group).size),
        "auprc": metrics.auprc(pool["y"], pool["score"]),
        "auprc_bin": metrics.auprc_bin(pool["y"], pool["score"]),
        "auroc": metrics.auroc(pool["y"], pool["score"]),
        "fleet_thresholds": thresholds,
        "operating_points": {}, "bootstrap": {}, "per_robot": {}, "material_table": {},
        "shift": {"test": transfer.distribution_block(pool),
                  "train": transfer.distribution_block(train_pool)},
    }
    for strat in ("cert", "emp"):
        for b in config.BUDGETS:
            k = config.budget_key(b)
            t = thresholds[strat][k]
            key = f"{strat}_{k}"
            out["operating_points"][key] = transfer.operating_point(pool, t, b)
            out["bootstrap"][f"{key}_recall"] = metrics.bootstrap_ci(
                pool["y"], pool["score"], pool["group"],
                lambda y, s, t=t: metrics.recall(y, s, t), draws=BOOTSTRAP_DRAWS)
            out["bootstrap"][f"{key}_ff"] = metrics.bootstrap_ci(
                pool["y"], pool["score"], pool["group"],
                lambda y, s, t=t: metrics.ff(y, s, t), draws=BOOTSTRAP_DRAWS)
        t5 = thresholds[strat][config.budget_key(BUDGET)]
        out["per_robot"][strat] = transfer.per_robot_table(pool, t5, MIN_ROBOT_N)
        out["material_table"][strat] = transfer.material_table(pool, t5)
    out["bootstrap"]["auprc"] = metrics.bootstrap_ci(pool["y"], pool["score"], pool["group"],
                                                     metrics.auprc, draws=BOOTSTRAP_DRAWS)
    out["bootstrap"]["auroc"] = metrics.bootstrap_ci(pool["y"], pool["score"], pool["group"],
                                                     metrics.auroc, draws=BOOTSTRAP_DRAWS)
    out["deployment_threshold"] = thresholds[SELECTION_KEY][config.budget_key(BUDGET)]
    out["one_item_ff_pct"] = 100.0 / max(int((items.y == 0).sum()), 1)
    stored = {"ids": items.ids, "score": score.astype(np.float32),
              "y": items.y.astype(np.int8), "mat": items.mat, "group": items.group,
              "source": items.source, "robot": items.robot, "per_seed": per_seed}
    if p_material is not None:
        stored["p_material"] = np.asarray(p_material, np.float32)
    np.savez_compressed(config.results_dir(EXP) / "test_scores.npz", **stored)
    out["items"] = items
    return out


def main(argv: list[str] | None = None) -> int:
    ap = evaluate.build_parser(EXP, SEEDS)
    ap.add_argument("--stage", default="loro", choices=("loro", "select", "deploy", "test"))
    args = ap.parse_args(argv)
    if args.dry_run:
        evaluate.run_grid(EXP, GRID, args)
        for stage in ("select", "deploy", "test"):
            print(f"{EXP}/stage {stage}")
        return 0
    items = data.load_pool()
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    if args.stage == "loro":
        evaluate.run_grid(EXP, GRID, args, items=items)
        return 0
    selection = select_variant([c["name"] for c in GRID], seeds)
    name = selection["chosen_variant"]
    cfg = next(c for c in GRID if c["name"] == name)
    if args.stage == "select":
        evaluate.write_result(dict(selection, exp=EXP, name="selection"))
        print("chosen:", name, "| aggregation:", selection["score_aggregation"])
        return 0
    if args.stage == "deploy":
        block = train_deployment(name, cfg, items, seeds)
        pool = evaluate.load_scores(EXP, name, seed=0)
        block.update({"exp": EXP, "name": "deployment", "variant": name,
                      "score_aggregation": selection["score_aggregation"],
                      "fleet_thresholds": transfer.fleet_thresholds(pool, delta=DELTA)})
        evaluate.write_result(block)
        print("deployment threshold:",
              block["fleet_thresholds"][SELECTION_KEY][config.budget_key(BUDGET)])
        return 0
    mode = selection["score_aggregation"]
    loro = evaluate.pooled_scores(EXP, name, seeds, mode) if mode == "mean" else \
        evaluate.load_scores(EXP, name, seed=0)
    thresholds = transfer.fleet_thresholds(loro, delta=DELTA)
    result = evaluate_test(name, cfg, seeds, mode, thresholds, loro)
    test_items = result.pop("items")
    result["baselines"] = baseline_block(test_items, items)
    result.update({"exp": EXP, "name": "test", "test_access_log": str(config.TEST_ACCESS_LOG)})
    evaluate.write_result(result)
    key = f"{SELECTION_KEY}_{config.budget_key(BUDGET)}"
    op = result["operating_points"][key]
    print(f"variant={name} aggregation={mode} auprc={result['auprc']:.3f}")
    print(f"threshold={op['threshold']:.4f} recall={op['recall']:.3f} ff={op['ff']:.3f} "
          f"(false flushes {op['n_false_flush']} of {result['n_neg']})")
    print("test access logged to", config.TEST_ACCESS_LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
