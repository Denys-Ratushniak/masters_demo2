from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from . import baselines, config, data, features, metrics
from .data import Items
from .folds import Fold, assert_disjoint, inner_folds, loro_folds, subsample_groups
from .train import Batch, TrainConfig, fit_predict, oof_scores

SCHEMA = 1
REQUIRED = ("schema", "exp", "config", "fold", "seed", "n_train", "n_test", "auprc", "auroc",
            "recall_at_ff1", "recall_at_ff2", "recall_at_ff5", "recall_at_ff10",
            "threshold_used", "threshold_source", "ff_actual", "ff_per_material",
            "per_item_scores", "train_time_s", "gpu_mem_mb")
SCORE_KEYS = ("ids", "score", "y", "mat", "group", "source")
SIZE_TARGETS: tuple[str, ...] = ("area_cm2_label", "support_rel_mm")
SUMMARY_KEYS = ("auprc", "auprc_bin", "auroc", "recall_at_ff1", "recall_at_ff2",
                "recall_at_ff5", "recall_at_ff10", "recall_at_ff5_oracle",
                "train_time_s", "gpu_mem_mb")


def config_hash(cfg: dict) -> str:
    payload = {k: v for k, v in cfg.items() if k not in ("name", "hash")}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]


def result_path(exp: str, name: str, fold: str, seed: int) -> Path:
    return config.results_dir(exp) / name / f"fold{fold}_seed{seed}.json"


def scores_path(exp: str, name: str, fold: str, seed: int) -> Path:
    return config.results_dir(exp) / name / f"fold{fold}_seed{seed}_scores.npz"


def oof_scores_path(exp: str, name: str, fold: str, seed: int) -> Path:
    return config.results_dir(exp) / name / f"fold{fold}_seed{seed}_oof.npz"


def summary_path(exp: str, name: str) -> Path:
    return config.results_dir(exp) / f"{name}.json"


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _scrub(o):
    if isinstance(o, dict):
        return {k: _scrub(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_scrub(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(float(o)):
        return None
    return o


def dump_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_scrub(obj), fh, indent=2, sort_keys=True, default=_jsonable, allow_nan=False)
    return path


_PREPARED: dict[tuple, dict] = {}


def sources(cfg: dict) -> list[str]:
    return [t for t in str(cfg.get("feature_source", "trunk")).split(",") if t]


def prepare(items: Items, cfg: dict, tcfg: TrainConfig, regime: str) -> dict:
    crop = cfg.get("crop", config.DEFAULT_CROP)
    srcs = sources(cfg)
    chans = tuple(data.geo_channels(tcfg.in_chans, tcfg.geo_channels, tcfg.use_rgb))
    key = (crop, tcfg.in_chans, regime, tcfg.backbone, tcfg.label_space, len(items),
           tuple(srcs), chans, tcfg.use_rgb)
    if key in _PREPARED:
        return _PREPARED[key]
    need_geo = bool(chans) or "geo_trunk" in srcs
    keys = ("rgb", "geo", "sc") if need_geo else ("rgb", "sc")
    out: dict = {}
    if regime == "rule":
        out["x"] = np.zeros((len(items), 0), np.float32)
    elif regime == "frozen":
        items = data.attach(items, data.load_cache(crop, ("sc",)), ("sc",))
        n = len(items)

        def images() -> np.ndarray:
            it = data.attach(items, data.load_cache(crop, keys), keys)
            return data.prep_input(it.rgb, it.geo, tcfg.in_chans, tcfg.geo_channels,
                                   tcfg.use_rgb)

        def geo_images() -> np.ndarray:
            it = data.attach(items, data.load_cache(crop, ("geo",)), ("geo",))
            return data.prep_geo(it.geo)

        out["x"] = (features.embed_cached(f"{crop}_c{tcfg.in_chans}", tcfg.backbone, images, n)
                    if "trunk" in srcs else np.zeros((n, 0), np.float32))
        if "geo_trunk" in srcs:
            out["x_geo"] = features.embed_cached(f"{crop}_geo", tcfg.backbone, geo_images, n)
    else:
        if items.rgb is None:
            items = data.attach(items, data.load_cache(crop, keys), keys)
        out["x"] = data.prep_input(items.rgb, items.geo, tcfg.in_chans, tcfg.geo_channels,
                                   tcfg.use_rgb)
    out["items"] = items
    if tcfg.scalar_dim:
        out["scalar_names"] = list(cfg.get("scalars", config.SCALARS))
    if tcfg.label_space != "binary":
        classes = config.MATERIALS3 if tcfg.label_space.endswith("3") else config.MATERIALS
        out["m"] = data.material_index(items.mat, classes)
    _PREPARED.clear()
    _PREPARED[key] = out
    return out


def feature_matrix(cfg: dict, items: Items, x: np.ndarray, fold: Fold, prep: dict) -> np.ndarray:
    srcs = sources(cfg)
    if srcs == ["trunk"]:
        return x
    parts = []
    for s in srcs:
        if s == "trunk":
            parts.append(np.asarray(x, np.float32))
        elif s == "geo_trunk":
            parts.append(np.asarray(prep["x_geo"], np.float32))
        elif s == "scalars":
            names = list(cfg.get("fuse_scalars", config.SCALARS))
            v, _ = data.scalar_matrix(items, names, fold.train)
            parts.append(v)
            if cfg.get("missing_indicator"):
                parts.append(np.isfinite(data.raw_scalars(items, names)).astype(np.float32))
        elif s == "bbox":
            raw = baselines.bbox_wh(items)
            parts.append(data.apply_scaler(raw, data.fit_scaler(raw[fold.train])))
        elif s == "tpflag":
            parts.append((items.mat == "tp").astype(np.float32)[:, None])
        else:
            raise ValueError(s)
    return np.concatenate(parts, 1).astype(np.float32)


def size_targets(items: Items, tcfg: TrainConfig, fold: Fold) -> np.ndarray | None:
    if tcfg.size_aux_weight <= 0.0:
        return None
    raw = data.raw_scalars(items, SIZE_TARGETS).copy()
    raw[:, 0] = np.log1p(np.maximum(raw[:, 0], 0.0))
    stats = data.fit_scaler(raw[fold.train])
    return np.concatenate([data.apply_scaler(raw, stats),
                           np.isfinite(raw).astype(np.float32)], 1)


def run_fold(exp: str, name: str, cfg: dict, fold: Fold, seed: int, items: Items) -> dict:
    started = dt.datetime.now(dt.timezone.utc)
    regime = cfg.get("regime", "finetune")
    tcfg = TrainConfig.from_dict(cfg)
    assert_disjoint(items, fold.train, fold.test, level="robot")
    prep = prepare(items, cfg, tcfg, regime)
    items = prep["items"]
    x, m = prep["x"], prep.get("m")
    sc = None
    if tcfg.scalar_dim:
        sc, _ = data.scalar_matrix(items, prep["scalar_names"], fold.train)
        assert sc.shape[1] == tcfg.scalar_dim, "scalar_dim does not match the scalar list"
    y = items.y
    if regime == "rule":
        s_all = baselines.rule_score(cfg["rule"], items)
        info = {"train_time_s": 0.0, "gpu_mem_mb": 0.0, "epochs_run": 0, "best_epoch": 0}
        return _finish(exp, name, cfg, fold, seed, items, y, s_all[fold.train], s_all[fold.test],
                       None, fold.train, 0.0, info, started)
    x = feature_matrix(cfg, items, x, fold, prep)
    pool = Batch(x, y, m, sc, size_targets(items, tcfg, fold))
    inner = inner_folds(items, fold.train, seed=seed)
    if cfg.get("oof_mode", "cv") == "holdout":
        inner = inner[:1]
    t0 = time.time()
    oof = oof_scores(items, fold.train, tcfg, seed, inner, pool, regime)
    oof_time = time.time() - t0
    assert np.isfinite(oof).sum() >= 50, "too few out-of-fold scores for the threshold"
    core, es = inner_folds(items, fold.train, seed=seed)[0]
    mode = cfg.get("final_fit", "holdout") if tcfg.early_stop else "full"
    fit_idx = fold.train if mode == "full" else core
    s_te, pm_te, info = fit_predict(tcfg, regime, pool.take(fit_idx), pool.take(fold.test),
                                    val=pool.take(es), seed=seed)
    if mode == "refit":
        rcfg = replace(tcfg, early_stop=False, epochs=int(info["best_epoch"]) + 1,
                       head_epochs=int(info["best_epoch"]) + 1)
        fit_idx = fold.train
        s_te, pm_te, info2 = fit_predict(rcfg, regime, pool.take(fit_idx), pool.take(fold.test),
                                         val=None, seed=seed)
        info2["train_time_s"] += info["train_time_s"]
        info2["best_epoch"], info = info["best_epoch"], info2
    return _finish(exp, name, cfg, fold, seed, items, y, oof, s_te, pm_te, fit_idx, oof_time,
                   info, started)


def _finish(exp: str, name: str, cfg: dict, fold: Fold, seed: int, items: Items, y: np.ndarray,
            oof: np.ndarray, s_te: np.ndarray, pm_te: np.ndarray | None, fit_idx: np.ndarray,
            oof_time: float, info: dict, started: dt.datetime) -> dict:
    oof = np.asarray(oof, np.float64)
    s_te = np.asarray(s_te, np.float64)
    ok = np.isfinite(oof)
    y_te = y[fold.test]
    y_fit, s_fit = y[fold.train][ok], oof[ok]
    g_fit = items.group[fold.train][ok]
    strategy = cfg.get("threshold_strategy", "emp")
    tr = metrics.eval_transferred(y_te, s_te, y_fit, s_fit, config.BUDGETS, strategy, g_fit)
    res: dict = {
        "schema": SCHEMA, "exp": exp,
        "config": dict(cfg, name=name, hash=config_hash(cfg)),
        "fold": fold.name, "seed": seed,
        "n_train": int(len(fit_idx)), "n_test": int(len(fold.test)),
        "n_train_pos": int((y[fit_idx] == 1).sum()), "n_train_neg": int((y[fit_idx] == 0).sum()),
        "n_test_pos": int((y_te == 1).sum()), "n_test_neg": int((y_te == 0).sum()),
        "n_oof": int(ok.sum()),
        "auprc": metrics.auprc(y_te, s_te), "auprc_bin": metrics.auprc_bin(y_te, s_te),
        "auroc": metrics.auroc(y_te, s_te),
        "threshold_used": tr["threshold_used"], "threshold_source": "oof_train",
        "threshold_strategy": strategy, "ff_actual": tr["ff_actual"],
        "ff_per_material": {}, "ff_by_source": {},
        "recall_at_ff_oracle": {}, "oof_ff_actual": {},
        "per_item_scores": str(scores_path(exp, name, fold.name, seed)),
        "oof_per_item_scores": str(oof_scores_path(exp, name, fold.name, seed)),
        "oof_mode": cfg.get("oof_mode", "cv"),
        "train_time_s": float(info["train_time_s"]), "oof_time_s": round(oof_time, 2),
        "eval_time_s": float(info.get("eval_time_s", 0.0)),
        "gpu_mem_mb": float(info["gpu_mem_mb"]), "epochs_run": int(info["epochs_run"]),
        "best_epoch": int(info["best_epoch"]), "best_val_auprc": info.get("best_val_auprc"),
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": cfg.get("notes", ""),
    }
    for b in config.BUDGETS:
        k = config.budget_key(b)
        t = tr["threshold_used"][k]
        res[f"recall_at_{k}"] = tr["recall_at_ff"][k]
        res["ff_per_material"][k] = metrics.ff_per_material(y_te, s_te, items.mat[fold.test], t)
        res["ff_by_source"][k] = metrics.ff_by_source(y_te, s_te, items.source[fold.test], t)
        res["recall_at_ff_oracle"][k] = metrics.recall_at_ff_oracle(y_te, s_te, b)
        res["oof_ff_actual"][k] = metrics.ff(y_fit, s_fit, t)
    res["recall_at_ff5_oracle"] = res["recall_at_ff_oracle"]["ff5"]
    tr_idx = np.asarray(fold.train)[ok]
    res["_oof_scores"] = {"ids": items.ids[tr_idx], "score": np.asarray(s_fit, np.float32),
                          "y": y_fit.astype(np.int8), "mat": items.mat[tr_idx],
                          "group": items.group[tr_idx], "source": items.source[tr_idx]}
    res["_scores"] = {"ids": items.ids[fold.test], "score": np.asarray(s_te, np.float32),
                      "y": y_te.astype(np.int8), "mat": items.mat[fold.test],
                      "group": items.group[fold.test], "source": items.source[fold.test]}
    if pm_te is not None:
        res["_scores"]["p_material"] = np.asarray(pm_te, np.float32)
    return res


def validate(res: dict) -> None:
    missing = [k for k in REQUIRED if k not in res]
    assert not missing, f"missing keys: {missing}"
    assert res["threshold_source"] == "oof_train", res["threshold_source"]
    flat = [res[k] for k in ("auprc", "auroc", "recall_at_ff1", "recall_at_ff2", "recall_at_ff5",
                             "recall_at_ff10", "train_time_s", "gpu_mem_mb")]
    flat += list(res["threshold_used"].values()) + list(res["ff_actual"].values())
    bad = [v for v in flat if not np.isfinite(float(v))]
    assert not bad, f"non-finite values: {bad}"
    for b in config.BUDGETS:
        k = config.budget_key(b)
        assert k in res["ff_per_material"], f"ff_per_material missing {k}"


def write_result(res: dict) -> Path:
    if "fold" not in res:
        return dump_json(res, summary_path(res["exp"], res["name"]))
    validate(res)
    path = result_path(res["exp"], res["config"]["name"], res["fold"], res["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = res.pop("_scores", None)
    oof = res.pop("_oof_scores", None)
    if scores is not None:
        np.savez_compressed(res["per_item_scores"], **scores)
    if oof is not None:
        np.savez_compressed(res["oof_per_item_scores"], **oof)
    return dump_json(res, path)


def config_names(exp: str) -> list[str]:
    root = config.results_dir(exp)
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def load_results(exp: str, name: str, seed: int | None = None) -> list[dict]:
    root = config.results_dir(exp) / name
    pattern = "fold*_seed*.json" if seed is None else f"fold*_seed{seed}.json"
    return [json.loads(p.read_text()) for p in sorted(root.glob(pattern))]


def load_scores(exp: str, name: str, seed: int | None = 0,
                n_folds: int | None = None) -> dict:
    root = config.results_dir(exp) / name
    tag = "*" if seed is None else str(seed)
    files = sorted(root.glob(f"fold*_seed{tag}_scores.npz"))
    assert files, f"no per-item scores under {root} for seed {seed}"
    if n_folds is not None:
        assert len(files) == n_folds, f"{len(files)} folds under {root}, expected {n_folds}"
    parts = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            parts.append({k: z[k] for k in z.files})
    pool = {k: np.concatenate([p[k] for p in parts]) for k in SCORE_KEYS}
    if all("p_material" in p for p in parts):
        pool["p_material"] = np.concatenate([p["p_material"] for p in parts])
    pool["score"] = pool["score"].astype(np.float64)
    pool["y"] = pool["y"].astype(int)
    pool["robot"] = np.array([str(g).split("_")[0] for g in pool["group"]])
    thresholds = [json.loads(p.read_text())["threshold_used"]
                  for p in sorted(root.glob(f"fold*_seed{tag}.json"))]
    if len(thresholds) == len(parts):
        pool["admitted"] = {config.budget_key(b): np.concatenate(
            [p["score"].astype(float) >= t[config.budget_key(b)] for p, t in zip(parts, thresholds)])
            for b in config.BUDGETS}
    return pool


def pooled_scores(exp: str, name: str, seeds: Sequence[int], mode: str = "mean") -> dict:
    pools = [load_scores(exp, name, s) for s in seeds]
    base = dict(pools[0])
    for p in pools[1:]:
        assert list(p["ids"]) == list(base["ids"]), "seed pools are not aligned by item id"
    stacked = np.stack([p["score"] for p in pools])
    base["per_seed"] = stacked
    base["score"] = stacked.mean(0) if mode == "mean" else pools[0]["score"]
    return base


def aggregate(exp: str, keys: Sequence[str] = SUMMARY_KEYS,
              reference: str | None = None) -> dict:
    rows: dict[str, list[dict]] = {}
    for path in sorted(config.results_dir(exp).glob("*/fold*_seed*.json")):
        res = json.loads(path.read_text())
        flat = dict(res)
        for k in ("ff_actual", "threshold_used"):
            flat.update({f"{k}_{kk}": vv for kk, vv in res[k].items()})
        rows.setdefault(path.parent.name, []).append(flat)
    ref_rows = rows.get(reference) if reference else None
    out: dict = {"exp": exp, "configs": {}}
    for name, rs in rows.items():
        rs = sorted(rs, key=lambda r: (r["fold"], r["seed"]))
        ref = None
        if ref_rows is not None and name != reference:
            index = {(r["fold"], r["seed"]): r for r in ref_rows}
            ref = [index.get((r["fold"], r["seed"]), {}) for r in rs]
        summary = metrics.summarise(rs, list(keys) + ["ff_actual_ff5"], ref)
        summary["folds"] = [f"{r['fold']}/s{r['seed']}" for r in rs]
        out["configs"][name] = summary
    out["table"] = markdown(out, keys)
    return out


def config_summary(exp: str, name: str) -> dict:
    rows = load_results(exp, name)
    pool = load_scores(exp, name, seed=None)
    out = {"name": name, "n": len(rows),
           "folds": sorted({r["fold"] for r in rows}),
           "seeds": sorted({r["seed"] for r in rows})}
    for key, getter in (("auprc", lambda r: r["auprc"]),
                        ("auroc", lambda r: r["auroc"]),
                        ("r_ff5", lambda r: r["recall_at_ff5"]),
                        ("r_ff5_oracle", lambda r: r["recall_at_ff5_oracle"]),
                        ("r_ff10", lambda r: r["recall_at_ff10"]),
                        ("ff5", lambda r: r["ff_actual"]["ff5"])):
        out[key] = metrics.fold_mean(rows, getter)
    y = pool["y"]
    if "admitted" in pool:
        for b in config.BUDGETS:
            k = config.budget_key(b)
            out[f"pooled_{k}"] = float(pool["admitted"][k][y == 0].mean())
        adm = pool["admitted"]["ff5"]
        out["mat_ff5"] = {m: float(adm[(pool["mat"] == m) & (y == 0)].mean())
                          if ((pool["mat"] == m) & (y == 0)).any() else float("nan")
                          for m in config.MATERIALS}
    return out


VARIANT_COLUMNS = ("config", "n", "AUPRC", "recall@FF5", "FF5", "recall@FF5 oracle",
                   "FF wipe", "FF towel", "FF tp", "wins vs ref")


def summaries(exp: str, names: list[str]) -> dict:
    return {n: config_summary(exp, n) for n in names
            if (config.results_dir(exp) / n).exists()}


def wins(a: dict, b: dict, key: str = "r_ff5") -> int:
    return int(np.sum(a[key] > b[key]))


def variant_table(sums: dict, order: list[str], reference: str) -> str:
    lines = ["| " + " | ".join(VARIANT_COLUMNS) + " |", "|" + "---|" * len(VARIANT_COLUMNS)]
    for n in order:
        if n not in sums:
            continue
        s = sums[n]
        w = "-" if n == reference else f"{wins(s, sums[reference])}/{len(s['r_ff5'])}"
        mat = s.get("mat_ff5", {})
        lines.append("| " + " | ".join([
            n, str(s["n"]), metrics.mean_sd(s["auprc"]), metrics.mean_sd(s["r_ff5"]),
            metrics.mean_sd(s["ff5"]), metrics.mean_sd(s["r_ff5_oracle"]),
            f"{mat.get('wipe', float('nan')):.3f}", f"{mat.get('towel', float('nan')):.3f}",
            f"{mat.get('tp', float('nan')):.3f}", w]) + " |")
    return "\n".join(lines)


def variant_block(sums: dict, order: list[str], reference: str) -> dict:
    out: dict = {}
    for n in order:
        if n not in sums:
            continue
        s = sums[n]
        out[n] = {
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
            "ff_per_material_ff5": s.get("mat_ff5"),
            "wins_vs_reference": None if n == reference else wins(s, sums[reference]),
        }
    return out


def paired_vs_reference(exp: str, names: list[str], reference: str, draws: int = 2000,
                        seed: int = 0) -> dict:
    ref = load_scores(exp, reference, seed=0)
    order_ref = np.argsort(ref["ids"], kind="mergesort")
    out: dict = {}
    for n in names:
        if n == reference or not (config.results_dir(exp) / n).exists():
            continue
        cur = load_scores(exp, n, seed=0)
        order = np.argsort(cur["ids"], kind="mergesort")
        assert (cur["ids"][order] == ref["ids"][order_ref]).all(), n
        a = {"y": cur["y"][order], "s": cur["score"][order]}
        b = {"y": ref["y"][order_ref], "s": ref["score"][order_ref]}
        groups = ref["group"][order_ref]
        out[n] = {
            "auprc": metrics.paired_bootstrap(a, b, groups, draws=draws, seed=seed,
                                              fn=metrics.auprc),
            "recall_ff5_oracle": metrics.paired_bootstrap(
                a, b, groups, draws=draws, seed=seed,
                fn=lambda y, s: metrics.recall_at_ff_oracle(y, s, 0.05)),
        }
    return out


def best_by(block: dict, key: str = "recall_ff5_mean") -> str:
    return max(block, key=lambda n: block[n][key])


def markdown(agg: dict, keys: Sequence[str] = SUMMARY_KEYS) -> str:
    cols = ["config", "n"] + list(keys) + ["ff_actual_ff5"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, s in sorted(agg["configs"].items()):
        cells = [name, str(s["n"])]
        for k in list(keys) + ["ff_actual_ff5"]:
            m, sd = s["mean"].get(k, float("nan")), s["sd"].get(k, float("nan"))
            win = s["wins"].get(k)
            cell = f"{m:.3f}±{sd:.3f}" if abs(m) < 1e4 else f"{m:.0f}"
            if win is not None:
                cell += f" ({win}/{s['n']})"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def check(exp: str, robots: Sequence[str] = config.TRAIN_ROBOTS) -> list[str]:
    bad: list[str] = []
    for name in config_names(exp):
        rows = load_results(exp, name)
        for r in rows:
            try:
                validate(r)
            except AssertionError as e:
                bad.append(f"{name}/{r['fold']}/s{r['seed']}: {e}")
        got = {(r["fold"], r["seed"]) for r in rows}
        for f in robots:
            for s in sorted({s for _, s in got}):
                if (f, s) not in got:
                    bad.append(f"{name}: missing fold {f} seed {s}")
    return bad


def build_parser(exp: str, seeds: Sequence[int] = (0,)) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"{exp.lower()}")
    ap.add_argument("--only", default="", help="comma separated config names to run")
    ap.add_argument("--folds", default="all", help="comma separated robot names or 'all'")
    ap.add_argument("--seeds", default=",".join(str(s) for s in seeds))
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--frac-seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def select(grid: Sequence[dict], only: str) -> list[dict]:
    if not only:
        return list(grid)
    want = {n for n in only.split(",") if n}
    picked = [c for c in grid if c["name"] in want]
    unknown = want - {c["name"] for c in picked}
    assert not unknown, f"unknown config names: {sorted(unknown)}"
    return picked


def build_folds(items: Items, args: argparse.Namespace) -> list[Fold]:
    folds = loro_folds(items)
    if args.folds != "all":
        want = set(args.folds.split(","))
        folds = [f for f in folds if f.name in want]
        assert folds, f"no folds match {args.folds}"
    if args.train_frac < 1.0:
        folds = [Fold(f.name, subsample_groups(items, f.train, args.train_frac, args.frac_seed),
                      f.test) for f in folds]
    return folds


def run_grid(exp: str, grid: Sequence[dict], args: argparse.Namespace,
             items: Items | None = None) -> dict:
    cfgs = select(grid, args.only)
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    if args.dry_run:
        for cfg in cfgs:
            for robot in (config.TRAIN_ROBOTS if args.folds == "all"
                          else tuple(args.folds.split(","))):
                for seed in seeds:
                    print(f"{exp}/{cfg['name']} fold={robot} seed={seed}")
        return {}
    items = data.load_pool() if items is None else items
    folds = build_folds(items, args)
    for cfg in cfgs:
        t0 = time.time()
        for fold in folds:
            for seed in seeds:
                path = result_path(exp, cfg["name"], fold.name, seed)
                if path.exists() and not args.force:
                    print(f"skip {path.name}")
                    continue
                res = run_fold(exp, cfg["name"], cfg, fold, seed, items)
                write_result(res)
                print(f"{exp}/{cfg['name']} {fold.name} s{seed}: "
                      f"auprc={res['auprc']:.3f} r@ff5={res['recall_at_ff5']:.3f} "
                      f"ff5={res['ff_actual']['ff5']:.3f} ({res['train_time_s']:.0f}s)")
        print(f"{cfg['name']} done in {time.time() - t0:.0f}s")
    agg = aggregate(exp)
    print(agg["table"])
    return agg
