from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace

import numpy as np

from trashnet import config, evaluate, metrics, transfer
from trashnet.arms import align
from trashnet.sizerule import (SET_A, RuleConfig, build_banks, nested_rule, nested_score,
                               size_inputs)

EXP = "e13b"
JOINT = "e9/loro_focal_noes"
STAGE1 = "e5/e_matonly"
ORACLE = "e1/c_oracle_tp_area_px"
BUDGETS: tuple[float, ...] = (0.02, 0.05, 0.10)
SEEDS: tuple[int, ...] = (0, 1, 2)
BUDGET = 0.05
CURVE_MAX_FF = 0.25
CURVE_POINTS = 251
QUESTION = ("does a size rule on top of the joint model survive honest nested selection, "
            "where the rule variant and both thresholds are chosen on the fitting robots only")


@dataclass
class NestedConfig:
    joint: str = JOINT
    stage1: str = STAGE1
    stage1_by_seed: dict[int, str] = field(default_factory=dict)
    oracle: str = ORACLE
    seeds: tuple[int, ...] = SEEDS
    budgets: tuple[float, ...] = BUDGETS
    variants: tuple[str, ...] = SET_A
    draws: int = 2000
    boot_seed: int = 0
    delta: float = 0.05
    tta_flip: bool = False

    def rules(self) -> RuleConfig:
        return RuleConfig(delta=self.delta)

    def stage1_for(self, seed: int) -> str:
        return self.stage1_by_seed.get(int(seed), self.stage1)


def source(text: str) -> tuple[str, str]:
    exp, sep, name = text.partition("/")
    assert sep and name, f"expected exp/config, got {text!r}"
    return exp, name


def parse_stage1(text: str, base: str) -> tuple[str, dict[int, str]]:
    per: dict[int, str] = {}
    for chunk in text.split(","):
        if not chunk:
            continue
        head, sep, spec = chunk.partition(":")
        if sep:
            per[int(head)] = spec
        else:
            base = chunk
    return base, per


def ensemble_scores(spec: str, seeds, ids: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    exp, name = source(spec)
    base = evaluate.load_scores(exp, name, seed=seeds[0])
    ids = base["ids"] if ids is None else ids
    stack = []
    for s in seeds:
        pool = evaluate.load_scores(exp, name, seed=s)
        idx = align(ids, pool["ids"])
        assert (pool["y"][idx].astype(int) == base["y"][align(ids, base["ids"])]).all()
        stack.append(pool["score"][idx].astype(float))
    return np.mean(stack, 0), base


def material_gate(cfg: NestedConfig, ids: np.ndarray) -> np.ndarray:
    stack = []
    for s in cfg.seeds:
        exp, name = source(cfg.stage1_for(s))
        pool = evaluate.load_scores(exp, name, seed=s)
        assert "p_material" in pool, f"{name} carries no material head output"
        stack.append(pool["p_material"][align(ids, pool["ids"]), 0].astype(float))
    return np.mean(stack, 0)


def summarise(pool: dict, flags: np.ndarray, budget: float) -> dict:
    out = transfer.score_flags(pool, flags, budget)
    per_robot = out["recall_per_robot"]
    v = np.array([per_robot[k] for k in sorted(per_robot)], float)
    out["recall_robot_sd"] = float(np.nanstd(v, ddof=1))
    return out


def rule_block(pool: dict, gate: np.ndarray, bank: dict, cfg: NestedConfig) -> tuple[dict, dict]:
    y, robot = pool["y"], pool["robot"]
    block: dict = {"budgets": {}}
    flags: dict[tuple[str, str], np.ndarray] = {}
    picks_all: dict = {}
    for b in cfg.budgets:
        bk = config.budget_key(b)
        block["budgets"][bk] = {}
        picks_all[bk] = {}
        for crit in ("emp", "cert"):
            flag, picks = nested_rule(y, robot, gate, bank, cfg.variants, b, crit, cfg.rules())
            entry = summarise(pool, flag, b)
            entry["picks"] = picks
            block["budgets"][bk][crit] = entry
            flags[(bk, crit)] = flag
            picks_all[bk][crit] = picks
    block["picks_all"] = picks_all
    return block, flags


def strategy_block(pool: dict, cfg: NestedConfig) -> tuple[dict, dict]:
    block: dict = {"budgets": {}}
    flags: dict[tuple[str, str], np.ndarray] = {}
    for b in cfg.budgets:
        bk = config.budget_key(b)
        block["budgets"][bk] = {}
        for crit in ("emp", "cert"):
            applied = transfer.apply_strategy(pool, crit, b)
            entry = summarise(pool, applied["flags"], b)
            entry["thresholds"] = applied["thresholds"]
            block["budgets"][bk][crit] = entry
            flags[(bk, crit)] = applied["flags"]
    return block, flags


def matched(y: np.ndarray, score: np.ndarray, budgets) -> dict:
    return {config.budget_key(b): float(metrics.recall_at_ff_targets(y, score, [b])[0])
            for b in budgets}


def run(cfg: NestedConfig) -> dict:
    joint, base = ensemble_scores(cfg.joint, cfg.seeds)
    ids = base["ids"]
    y, robot = base["y"].astype(int), base["robot"]
    pool = {"ids": ids, "y": y, "robot": robot, "mat": base["mat"], "group": base["group"],
            "source": base["source"], "score": joint}
    box = size_inputs(ids)
    bank, _, _, size_info = build_banks(box, y, robot, cfg.rules())
    gate_mat = material_gate(cfg, ids)
    nested, nested_flags = rule_block(pool, joint, bank, cfg)
    plain, plain_flags = strategy_block(pool, cfg)
    two_stage, ts_flags = rule_block(dict(pool, score=gate_mat), gate_mat, bank, cfg)
    key5 = config.budget_key(BUDGET)
    s_nested = nested_score(joint, robot, bank, nested["picks_all"][key5]["emp"])
    s_two_stage = nested_score(gate_mat, robot, bank, two_stage["picks_all"][key5]["emp"])
    curves = {"grid_ff": np.linspace(0.0, CURVE_MAX_FF, CURVE_POINTS)}
    for key, score in (("joint_nested_size", s_nested), ("joint_base", joint),
                       ("two_stage_nested", s_two_stage), ("material_gate", gate_mat)):
        curves[f"{key}_rec"] = metrics.recall_at_ff_targets(y, score, curves["grid_ff"])
    out: dict = {
        "exp": EXP, "name": "summary", "question": QUESTION,
        "n": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "robots": [str(r) for r in np.unique(robot)], "seeds": list(cfg.seeds),
        "budgets": list(cfg.budgets), "variants": list(cfg.variants),
        "joint": cfg.joint, "stage1": {str(s): cfg.stage1_for(s) for s in cfg.seeds},
        "tta_flip": cfg.tta_flip, "size_model": size_info,
        "ensemble": {"joint_nested_size": nested, "joint_base": plain,
                     "two_stage_nested": two_stage},
        "picks": {"joint_nested_size": nested["picks_all"][key5]["emp"],
                  "two_stage_nested": two_stage["picks_all"][key5]["emp"]},
        "matched_ff": {"joint_nested_size": matched(y, s_nested, cfg.budgets),
                       "joint_base": matched(y, joint, cfg.budgets),
                       "two_stage_nested": matched(y, s_two_stage, cfg.budgets),
                       "material_gate": matched(y, gate_mat, cfg.budgets)},
        "bootstrap": {},
        "curves": str(config.results_dir(EXP) / "curves.npz"),
    }
    oracle_exp, oracle_name = source(cfg.oracle)
    if (config.results_dir(oracle_exp) / oracle_name).exists():
        op = evaluate.load_scores(oracle_exp, oracle_name, seed=0)
        s = op["score"][align(ids, op["ids"])].astype(float)
        s = np.where(s <= -1e8, -np.inf, s)
        out["matched_ff"]["oracle"] = matched(y, s, cfg.budgets)
        curves["oracle_rec"] = metrics.recall_at_ff_targets(y, s, curves["grid_ff"])
    sets = metrics.group_bootstrap_sets(base["group"], y, cfg.draws, cfg.boot_seed)
    for ref, ref_flags, ref_score in (("joint_base", plain_flags, joint),
                                      ("two_stage_nested", ts_flags, s_two_stage)):
        for crit in ("emp", "cert"):
            out["bootstrap"][f"transferred_{crit}__joint_nested_size__{ref}"] = \
                metrics.paired_bootstrap_flags(y, base["group"],
                                               nested_flags[(key5, crit)].astype(float),
                                               ref_flags[(key5, crit)].astype(float), sets=sets)
        deltas = np.array([float(metrics.recall_at_ff_targets(y[sel], s_nested[sel], [BUDGET])[0]
                                 - metrics.recall_at_ff_targets(y[sel], ref_score[sel],
                                                                [BUDGET])[0])
                           for sel in sets], float)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        out["bootstrap"][f"matched_{key5}__joint_nested_size__{ref}"] = {
            "d_recall": float(deltas.mean()), "lo": float(lo), "hi": float(hi),
            "p_gt0": float((deltas > 0).mean()), "n": int(deltas.size)}
    config.results_dir(EXP).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out["curves"], **{k: np.asarray(v) for k, v in curves.items()})
    return out


def build_config(args: argparse.Namespace) -> NestedConfig:
    stage1, per_seed = parse_stage1(args.stage1, STAGE1)
    return replace(NestedConfig(), joint=args.joint, stage1=stage1, stage1_by_seed=per_seed,
                   oracle=args.oracle, draws=args.draws,
                   seeds=tuple(int(s) for s in args.seeds.split(",") if s != ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=EXP)
    ap.add_argument("--joint", default=JOINT, help="exp/config of the joint model")
    ap.add_argument("--stage1", default=STAGE1,
                    help="exp/config, optionally seed:exp/config entries")
    ap.add_argument("--oracle", default=ORACLE)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    args, _ = ap.parse_known_args(argv)
    cfg = build_config(args)
    if args.dry_run:
        print(f"{EXP}: joint {cfg.joint} seeds {','.join(str(s) for s in cfg.seeds)}")
        print(f"{EXP}: nested selection over {list(cfg.variants)} on the fitting robots only")
        return 0
    out = run(cfg)
    evaluate.write_result(out)
    key5 = config.budget_key(BUDGET)
    for name, block in out["ensemble"].items():
        entry = block["budgets"][key5]["emp"]
        print(f"{name:20s} transferred recall={entry['recall']:.3f} ff={entry['ff']:.3f}")
    print("matched at FF 5 percent:",
          {k: round(v[key5], 3) for k, v in sorted(out["matched_ff"].items())})
    print("picks:", {r: p["variant"] for r, p in out["picks"]["joint_nested_size"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
