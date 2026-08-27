from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from trashnet import config, data, evaluate, metrics, models, transfer

EXP = "e8"
SEEDS: tuple[int, ...] = (0,)
BUDGET = 0.05
PX_GRID: tuple[int, ...] = (224, 288, 384)
MODES: tuple[str, ...] = ("fp32", "fp16", "compile")
ITERS, WARMUP = 300, 50
EDGE_DEVICE = "RTX A2000 12 GB"
EDGE_MULTIPLIER: tuple[float, float] = (2.5, 3.0)
DEPLOY_EXP, DEPLOY_CONFIG = "e9", "loro_focal_noes"
ACCURACY_ARMS: tuple[tuple[str, str, str], ...] = (
    ("288", DEPLOY_EXP, DEPLOY_CONFIG),
    ("224", EXP, f"px224_{DEPLOY_CONFIG}"),
)
GRID: list[dict] = [
    {"name": f"px224_{DEPLOY_CONFIG}", "px": 224,
     "notes": "same recipe as the deployed model, trunk input downscaled to 224"},
]


def checkpoint_path(seed: int = 0) -> Path:
    return config.RESULTS / "models" / f"final_convnext_tiny_seed{seed}.pt"


def latency_table(px_grid=PX_GRID, modes=MODES) -> dict:
    path = checkpoint_path()
    if not path.exists():
        return {"available": False, "reason": f"missing checkpoint {path}"}
    rows = []
    for px in px_grid:
        for mode in modes:
            model = models.load_checkpoint(path, px=None)
            try:
                row = models.benchmark_latency(model, px, mode, ITERS, WARMUP)
                row["edge_lo_ms"] = row["median_ms"] * EDGE_MULTIPLIER[0]
                row["edge_hi_ms"] = row["median_ms"] * EDGE_MULTIPLIER[1]
            except Exception as err:
                row = {"px": px, "mode": mode, "error": str(err)[:300]}
            rows.append(row)
            del model
    return {"available": True, "rows": rows,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__, "edge_device": EDGE_DEVICE,
            "edge_multiplier": list(EDGE_MULTIPLIER),
            "edge_note": "estimate, the multiplier is not measured on the edge device"}


def accuracy_arms() -> dict:
    out: dict = {}
    for label, exp, name in ACCURACY_ARMS:
        if not (config.results_dir(exp) / name).exists():
            continue
        pool = evaluate.load_scores(exp, name, seed=0)
        rows = evaluate.load_results(exp, name, seed=0)
        block = {"exp": exp, "config": name, "n": int(pool["y"].size),
                 "auprc": metrics.auprc(pool["y"], pool["score"]),
                 "auroc": metrics.auroc(pool["y"], pool["score"]),
                 "recall_ff5_oracle": metrics.recall_at_ff_oracle(pool["y"], pool["score"],
                                                                  BUDGET),
                 "train_time_s": float(np.mean([r["train_time_s"] for r in rows])),
                 "gpu_mem_mb": float(np.mean([r["gpu_mem_mb"] for r in rows]))}
        for strat in ("emp", "cert"):
            applied = transfer.apply_strategy(pool, strat, BUDGET)
            block[strat] = transfer.score_flags(pool, applied["flags"], BUDGET)
            block[strat]["thresholds"] = applied["thresholds"]
        out[label] = block
    if {"224", "288"} <= set(out):
        a = evaluate.load_scores(EXP, ACCURACY_ARMS[1][2], seed=0)
        b = evaluate.load_scores(DEPLOY_EXP, DEPLOY_CONFIG, seed=0)
        oa, ob = np.argsort(a["ids"], kind="mergesort"), np.argsort(b["ids"], kind="mergesort")
        assert (a["ids"][oa] == b["ids"][ob]).all()
        out["paired_224_vs_288"] = metrics.paired_bootstrap(
            {"y": a["y"][oa], "s": a["score"][oa]}, {"y": b["y"][ob], "s": b["score"][ob]},
            b["group"][ob], draws=2000, seed=0, fn=metrics.auprc)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = evaluate.build_parser(EXP, SEEDS)
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args(argv)
    if args.dry_run:
        evaluate.run_grid(EXP, GRID, args)
        for px in PX_GRID:
            for mode in MODES:
                print(f"{EXP}/latency px={px} mode={mode}")
        return 0
    deploy = evaluate.load_results(DEPLOY_EXP, DEPLOY_CONFIG)
    assert deploy, f"run e9_final.py stage loro first, {DEPLOY_CONFIG} has no results"
    base = dict(deploy[0]["config"])
    base.pop("name", None)
    base.pop("hash", None)
    grid = [dict(base, **cfg) for cfg in GRID]
    evaluate.run_grid(EXP, grid, args, items=data.load_pool())
    summary = {"exp": EXP, "name": "summary", "px_grid": list(PX_GRID), "modes": list(MODES),
               "accuracy": accuracy_arms(),
               "latency": {} if args.skip_latency else latency_table(),
               "issues": evaluate.check(EXP)}
    evaluate.write_result(summary)
    for row in summary["latency"].get("rows", []):
        if "median_ms" in row:
            print(f"px={row['px']} {row['mode']}: {row['median_ms']:.3f} ms "
                  f"(edge {row['edge_lo_ms']:.1f}-{row['edge_hi_ms']:.1f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
