from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from trashnet import config, evaluate, metrics, transfer
from trashnet.arms import align
from trashnet.sizerule import (SET_A, SET_B, SIZE_NAMES, RuleConfig, build_banks, fixed_rule,
                               gate_rule, nested_cv_rule, nested_rule, size_inputs,
                               two_stage_score)

EXP = "e10"
GATE_KEY = "ts_gate"


@dataclass(frozen=True)
class Source:
    exp: str
    name: str

    def as_dict(self) -> dict:
        return {"exp": self.exp, "config": self.name}


@dataclass
class TwoStageConfig:
    stage1: Source = Source("e5", "e_matonly")
    stage1_by_seed: dict[int, Source] = field(default_factory=dict)
    joint: tuple[Source, ...] = (Source("e9", "loro_focal"), Source("e9", "loro_focal_noes"))
    oracle: Source = Source("e1", "c_oracle_tp_area_px")
    seeds: tuple[int, ...] = (0, 1, 2)
    budgets: tuple[float, ...] = (0.02, 0.05, 0.10)
    delta: float = 0.05
    q_points: int = 120
    frontier_points: int = 96
    curve_max_ff: float = 0.25
    curve_points: int = 251
    draws: int = 2000
    boot_seed: int = 0
    logreg_c: float = 1.0
    min_neg: int = 5
    with_nested_cv: bool = False
    with_per_seed: bool = False

    def stage1_for(self, seed: int) -> Source:
        return self.stage1_by_seed.get(int(seed), self.stage1)

    def rules(self) -> RuleConfig:
        return RuleConfig(delta=self.delta, q_points=self.q_points,
                          frontier_points=self.frontier_points, logreg_c=self.logreg_c)


def parse_source(text: str) -> Source:
    exp, sep, name = text.partition("/")
    assert sep and name, f"expected exp/config, got {text!r}"
    return Source(exp, name)


def parse_stage1(text: str, base: Source) -> tuple[Source, dict[int, Source]]:
    per: dict[int, Source] = {}
    for chunk in text.split(","):
        if not chunk:
            continue
        head, sep, spec = chunk.partition(":")
        if sep:
            per[int(head)] = parse_source(spec)
        else:
            base = parse_source(chunk)
    return base, per


def summarise_flags(pool: dict, flags: np.ndarray, budget: float) -> dict:
    out = transfer.score_flags(pool, flags, budget)
    per_robot = out["recall_per_robot"]
    v = np.array([per_robot[k] for k in sorted(per_robot)], float)
    out["recall_robot_sd"] = float(np.nanstd(v, ddof=1))
    return out


def rule_makers(y: np.ndarray, robot: np.ndarray, gate: np.ndarray, bank: dict,
                bank_inner: dict, cfg: TwoStageConfig) -> dict[str, Callable]:
    rc = cfg.rules()
    out: dict[str, Callable] = {
        "ts_nested": lambda b, c: nested_rule(y, robot, gate, bank, SET_A, b, c, rc),
        "ts_nested_nosup": lambda b, c: nested_rule(y, robot, gate, bank, SET_B, b, c, rc),
        GATE_KEY: lambda b, c: gate_rule(y, robot, gate, b, c, rc),
        "ts_logsize": lambda b, c: fixed_rule(y, robot, gate, bank, "v5_logsize", b, c, rc),
        "ts_bbox": lambda b, c: fixed_rule(y, robot, gate, bank, "v2_bbox_px", b, c, rc),
    }
    if cfg.with_nested_cv:
        out["ts_nested_cv"] = lambda b, c: nested_cv_rule(y, robot, gate, bank, bank_inner,
                                                          SET_A, b, c, rc)
    return out


def evaluate_rules(pool: dict, makers: dict[str, Callable],
                   cfg: TwoStageConfig) -> tuple[dict, dict]:
    block: dict[str, dict] = {}
    flags: dict[tuple[str, str, str], np.ndarray] = {}
    for key, make in makers.items():
        block[key] = {"budgets": {}}
        for b in cfg.budgets:
            bk = config.budget_key(b)
            block[key]["budgets"][bk] = {}
            for crit in ("emp", "cert"):
                flag, picks = make(b, crit)
                entry = summarise_flags(pool, flag, b)
                entry["picks"] = picks
                block[key]["budgets"][bk][crit] = entry
                flags[(key, bk, crit)] = flag
    return block, flags


def evaluate_joint(pools: dict[str, dict], cfg: TwoStageConfig) -> tuple[dict, dict]:
    block: dict[str, dict] = {}
    flags: dict[tuple[str, str, str], np.ndarray] = {}
    for key, pool in pools.items():
        block[key] = {"budgets": {}}
        for b in cfg.budgets:
            bk = config.budget_key(b)
            block[key]["budgets"][bk] = {}
            for crit in ("emp", "cert"):
                applied = transfer.apply_strategy(pool, crit, b)
                entry = summarise_flags(pool, applied["flags"], b)
                entry["thresholds"] = applied["thresholds"]
                block[key]["budgets"][bk][crit] = entry
                flags[(key, bk, crit)] = applied["flags"]
    return block, flags


def load_stage1(cfg: TwoStageConfig) -> tuple[dict, dict[int, np.ndarray]]:
    first = cfg.stage1_for(cfg.seeds[0])
    base = evaluate.load_scores(first.exp, first.name, seed=cfg.seeds[0])
    assert "p_material" in base, f"{first.name} carries no material head output"
    ids = base["ids"]
    p_tp: dict[int, np.ndarray] = {}
    for s in cfg.seeds:
        src = cfg.stage1_for(s)
        pool = evaluate.load_scores(src.exp, src.name, seed=s)
        idx = align(ids, pool["ids"])
        assert (pool["y"][idx].astype(int) == base["y"]).all(), f"stage 1 seed {s} misaligned"
        p_tp[s] = pool["p_material"][idx, 0].astype(float)
    return base, p_tp


def load_joint(cfg: TwoStageConfig, ids: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for src in cfg.joint:
        stack = []
        for s in cfg.seeds:
            pool = evaluate.load_scores(src.exp, src.name, seed=s)
            idx = align(ids, pool["ids"])
            assert (pool["y"][idx].astype(int) == y).all(), f"{src.name} seed {s} misaligned"
            stack.append(pool["score"][idx].astype(float))
        out[f"joint_{src.name}"] = np.mean(stack, 0)
    return out


def tp_classifier_block(mat: np.ndarray, robot: np.ndarray, p_tp: dict[int, np.ndarray],
                        joint: dict[str, np.ndarray]) -> dict:
    is_tp = (mat == "tp").astype(int)
    out: dict = {"n_tp": int(is_tp.sum()), "seeds": {}}
    for s, g in p_tp.items():
        out["seeds"][str(s)] = {
            "auroc": float(roc_auc_score(is_tp, g)),
            "auprc": float(average_precision_score(is_tp, g)),
            "err_rate_at_half": float(((g >= 0.5).astype(int) != is_tp).mean()),
            "per_robot_auroc": {str(r): float(roc_auc_score(is_tp[robot == r], g[robot == r]))
                                for r in np.unique(robot)}}
    for key, g in joint.items():
        out[key] = {"auprc": float(average_precision_score(is_tp, g))}
    return out


def matched_block(pool: dict, cfg: TwoStageConfig, gate: np.ndarray, size: np.ndarray,
                  joint: dict[str, np.ndarray], rules: dict,
                  oracle: np.ndarray) -> tuple[dict, dict, np.ndarray]:
    y, robot = pool["y"], pool["robot"]
    grid = np.linspace(0.0, cfg.curve_max_ff, cfg.curve_points)
    curves: dict[str, np.ndarray] = {"grid_ff": grid}
    matched: dict[str, dict] = {}
    at_budget = {}
    for b in cfg.budgets:
        bk = config.budget_key(b)
        picks = rules["ts_logsize"]["budgets"][bk]["emp"]["picks"]
        qs = {r: picks[r]["q"] for r in picks}
        at_budget[bk] = float(metrics.recall_at_ff_targets(
            y, two_stage_score(gate, size, robot, qs), [b])[0])
    matched["two_stage_qfit"] = at_budget
    key5 = config.budget_key(0.05)
    picks5 = rules["ts_logsize"]["budgets"][key5]["emp"]["picks"]
    s_ts = two_stage_score(gate, size, robot, {r: picks5[r]["q"] for r in picks5})
    curves["two_stage_rec"] = metrics.recall_at_ff_targets(y, s_ts, grid)
    matched["two_stage_q_at_ff5"] = {config.budget_key(b): float(
        metrics.recall_at_ff_targets(y, s_ts, [b])[0]) for b in cfg.budgets}
    q_grid = np.unique(np.r_[-np.inf, np.quantile(gate, np.linspace(0.0, 0.995,
                                                                   cfg.frontier_points))])
    front = np.zeros(len(cfg.budgets))
    front_curve = np.zeros(len(grid))
    for q in q_grid:
        s = np.where(gate >= q, size, -np.inf)
        front = np.maximum(front, metrics.recall_at_ff_targets(y, s, cfg.budgets))
        front_curve = np.maximum(front_curve, metrics.recall_at_ff_targets(y, s, grid))
    matched["two_stage_frontier"] = {config.budget_key(b): float(v)
                                     for b, v in zip(cfg.budgets, front)}
    curves["frontier_rec"] = front_curve
    for key, score in joint.items():
        curves[f"{key}_rec"] = metrics.recall_at_ff_targets(y, score, grid)
        matched[key] = {config.budget_key(b): float(
            metrics.recall_at_ff_targets(y, score, [b])[0]) for b in cfg.budgets}
    curves["gate_rec"] = metrics.recall_at_ff_targets(y, gate, grid)
    matched[GATE_KEY] = {config.budget_key(b): float(
        metrics.recall_at_ff_targets(y, gate, [b])[0]) for b in cfg.budgets}
    if oracle is not None:
        curves["oracle_rec"] = metrics.recall_at_ff_targets(y, oracle, grid)
        matched["oracle"] = {config.budget_key(b): float(
            metrics.recall_at_ff_targets(y, oracle, [b])[0]) for b in cfg.budgets}
    return matched, curves, s_ts


def load_oracle(cfg: TwoStageConfig, ids: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    root = config.results_dir(cfg.oracle.exp) / cfg.oracle.name
    if not root.exists():
        return None
    pool = evaluate.load_scores(cfg.oracle.exp, cfg.oracle.name, seed=0)
    idx = align(ids, pool["ids"])
    assert (pool["y"][idx].astype(int) == y).all(), "oracle scores misaligned"
    s = pool["score"][idx].astype(float)
    return np.where(s <= -1e8, -np.inf, s)


def bootstrap_block(pool: dict, flags: dict, joint: dict[str, np.ndarray],
                    gate: np.ndarray, s_ts: np.ndarray, cfg: TwoStageConfig) -> dict:
    sets = metrics.group_bootstrap_sets(pool["group"], pool["y"], cfg.draws, cfg.boot_seed)
    y = pool["y"]
    out: dict = {}
    pairs = [("ts_nested", key) for key in joint]
    pairs += [("ts_nested", GATE_KEY), ("ts_logsize", list(joint)[0]),
              ("ts_nested_nosup", list(joint)[-1])]
    key5 = config.budget_key(0.05)
    for crit in ("emp", "cert"):
        for a, b in pairs:
            if (a, key5, crit) not in flags or (b, key5, crit) not in flags:
                continue
            out[f"transferred_{crit}__{a}__{b}"] = metrics.paired_bootstrap_flags(
                y, pool["group"], flags[(a, key5, crit)].astype(float),
                flags[(b, key5, crit)].astype(float), sets=sets)
    for key, score in list(joint.items()) + [(GATE_KEY, gate)]:
        deltas = np.array([float(metrics.recall_at_ff_targets(y[sel], s_ts[sel], [0.05])[0]
                                 - metrics.recall_at_ff_targets(y[sel], score[sel], [0.05])[0])
                           for sel in sets], float)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        out[f"matched_ff5__two_stage__{key}"] = {
            "d_recall": float(deltas.mean()), "lo": float(lo), "hi": float(hi),
            "p_gt0": float((deltas > 0).mean()), "n": int(deltas.size)}
    return out


def per_seed_block(pool: dict, robot: np.ndarray, p_tp: dict[int, np.ndarray], bank: dict,
                   bank_inner: dict, joint_sources: dict[str, list[np.ndarray]],
                   cfg: TwoStageConfig) -> dict:
    out: dict = {}
    y = pool["y"]
    keys = ("recall", "ff", "ff_max", "within_budget", "leak_tp", "leak_wipe", "leak_towel")
    per_rule: dict[str, dict] = {}
    for s in cfg.seeds:
        makers = rule_makers(y, robot, p_tp[s], bank, bank_inner, cfg)
        block, _ = evaluate_rules(pool, makers, cfg)
        for name, entry in block.items():
            per_rule.setdefault(name, {})[s] = entry
    for name, scores in joint_sources.items():
        for s, score in zip(cfg.seeds, scores):
            jp = dict(pool)
            jp["score"] = score
            block, _ = evaluate_joint({name: jp}, cfg)
            per_rule.setdefault(name, {})[s] = block[name]
    for name, by_seed in per_rule.items():
        out[name] = {}
        for b in cfg.budgets:
            bk = config.budget_key(b)
            out[name][bk] = {}
            for crit in ("emp", "cert"):
                row = {}
                for k in keys:
                    v = np.array([by_seed[s]["budgets"][bk][crit][k] for s in cfg.seeds], float)
                    row[k] = float(np.nanmean(v))
                    row[f"{k}_sd"] = float(np.nanstd(v, ddof=1))
                out[name][bk][crit] = row
    return out


def run(cfg: TwoStageConfig) -> dict:
    base, p_tp = load_stage1(cfg)
    ids, y, robot = base["ids"], base["y"].astype(int), base["robot"]
    box = size_inputs(ids)
    pool = {"ids": ids, "y": y, "robot": robot, "mat": base["mat"], "group": base["group"],
            "source": base["source"], "score": np.mean([p_tp[s] for s in cfg.seeds], 0)}
    gate = pool["score"]
    bank, bank_inner, p_size, size_info = build_banks(box, y, robot, cfg.rules(),
                                                     cfg.with_nested_cv)
    joint = load_joint(cfg, ids, y)
    makers = rule_makers(y, robot, gate, bank, bank_inner, cfg)
    rules, flags = evaluate_rules(pool, makers, cfg)
    joint_pools = {k: dict(pool, score=v) for k, v in joint.items()}
    joint_block, joint_flags = evaluate_joint(joint_pools, cfg)
    ensemble = dict(rules)
    ensemble.update(joint_block)
    flags.update(joint_flags)
    size = np.zeros(len(y), float)
    for r in np.unique(robot):
        m = robot == r
        size[m] = p_size[str(r)][m]
    oracle = load_oracle(cfg, ids, y)
    matched, curves, s_ts = matched_block(pool, cfg, gate, size, joint, rules, oracle)
    out: dict = {
        "exp": EXP, "name": "summary",
        "n": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "robots": [str(r) for r in np.unique(robot)],
        "seeds": list(cfg.seeds), "budgets": list(cfg.budgets),
        "stage1": {str(s): cfg.stage1_for(s).as_dict() for s in cfg.seeds},
        "joint": [s.as_dict() for s in cfg.joint],
        "oracle": cfg.oracle.as_dict() if oracle is not None else None,
        "variant_set_a": list(SET_A), "variant_set_b": list(SET_B),
        "size_features": list(SIZE_NAMES), "size_model": size_info,
        "tp_classifier": tp_classifier_block(pool["mat"], robot, p_tp, joint),
        "ensemble": ensemble, "matched_ff": matched,
        "bootstrap": bootstrap_block(pool, flags, joint, gate, s_ts, cfg),
        "curves": str(config.results_dir(EXP) / "curves.npz"),
    }
    if cfg.with_per_seed:
        sources = {}
        for src in cfg.joint:
            stack = []
            for s in cfg.seeds:
                p = evaluate.load_scores(src.exp, src.name, seed=s)
                stack.append(p["score"][align(ids, p["ids"])].astype(float))
            sources[f"joint_{src.name}"] = stack
        out["per_seed"] = per_seed_block(pool, robot, p_tp, bank, bank_inner, sources, cfg)
    config.results_dir(EXP).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out["curves"], **{k: np.asarray(v) for k, v in curves.items()})
    return out


def build_config(args: argparse.Namespace) -> TwoStageConfig:
    cfg = TwoStageConfig()
    stage1, per_seed = parse_stage1(args.stage1, cfg.stage1)
    joint = tuple(parse_source(t) for t in args.joint.split(",") if t)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s != "")
    return replace(cfg, stage1=stage1, stage1_by_seed=per_seed, joint=joint,
                   oracle=parse_source(args.oracle), seeds=seeds, draws=args.draws,
                   with_nested_cv=args.with_nested_cv, with_per_seed=args.with_per_seed)


def main(argv: list[str] | None = None) -> int:
    base = TwoStageConfig()
    ap = argparse.ArgumentParser(prog=EXP)
    ap.add_argument("--stage1", default=f"{base.stage1.exp}/{base.stage1.name}",
                    help="exp/config, optionally seed:exp/config entries")
    ap.add_argument("--joint", default=",".join(f"{s.exp}/{s.name}" for s in base.joint))
    ap.add_argument("--oracle", default=f"{base.oracle.exp}/{base.oracle.name}")
    ap.add_argument("--seeds", default=",".join(str(s) for s in base.seeds))
    ap.add_argument("--draws", type=int, default=base.draws)
    ap.add_argument("--with-nested-cv", action="store_true")
    ap.add_argument("--with-per-seed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args, _ = ap.parse_known_args(argv)
    cfg = build_config(args)
    if args.dry_run:
        print(f"{EXP}: stage 1 {cfg.stage1.exp}/{cfg.stage1.name} seeds "
              f"{','.join(str(s) for s in cfg.seeds)}")
        for s in cfg.joint:
            print(f"{EXP}: joint model {s.exp}/{s.name}")
        print(f"{EXP}: nested selection, ensembles, matched false-flush comparison")
        return 0
    out = run(cfg)
    evaluate.write_result(out)
    key = config.budget_key(0.05)
    for name in sorted(out["ensemble"]):
        entry = out["ensemble"][name]["budgets"][key]["emp"]
        print(f"{name:24s} recall={entry['recall']:.3f} ff={entry['ff']:.3f}")
    print("matched at FF 5 percent:", {k: round(v[key], 3)
                                       for k, v in sorted(out["matched_ff"].items())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
