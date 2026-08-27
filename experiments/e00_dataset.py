from __future__ import annotations

import argparse
import json

import numpy as np

from trashnet import config, crops, data, evaluate

EXP = "e0"
MIXED_GROUP = "other_robots"
MATERIAL_ORDER: tuple[str, ...] = config.MATERIALS
GEOMETRY_PERCENTILES: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)
DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("bbox_area_px", "bbox pixel area"),
    ("area_cm2", "labelled area in cm2"),
    ("support_rel_mm", "height above the support in mm"),
)
GALLERY: tuple[dict, ...] = (
    {"name": "per_material", "per_group": 3, "key": "material"},
    {"name": "same_material_two_actions", "per_group": 3, "key": "tp_action"},
    {"name": "wipe_vs_tp", "per_group": 3, "key": "wipe_vs_tp"},
)


def robot_group(robot: str) -> str:
    return robot if robot in config.TRAIN_ROBOTS else MIXED_GROUP


def group_order() -> list[str]:
    return list(config.TRAIN_ROBOTS) + [MIXED_GROUP]


def per_robot_stats(items: data.Items, groups: np.ndarray) -> dict:
    out = {}
    for g in group_order() + ["all"]:
        m = np.ones(len(items), bool) if g == "all" else groups == g
        if not m.any():
            continue
        tp = m & (items.mat == "tp")
        wipe = m & (items.mat == "wipe")
        out[g] = {
            "n": int(m.sum()), "robot_days": int(np.unique(items.group[m]).size),
            "flush": int((items.y[m] == 1).sum()), "bin": int((items.y[m] == 0).sum()),
            "flush_rate": float((items.y[m] == 1).mean()),
            "tp": int(tp.sum()), "tp_binned": int((tp & (items.y == 0)).sum()),
            "wipe": int(wipe.sum()),
            "wipe_staged": int((wipe & (items.source == "staged")).sum()),
            "wipe_natural": int((wipe & (items.source != "staged")).sum()),
        }
    return out


def material_matrix(items: data.Items, groups: np.ndarray) -> dict:
    out = {}
    for g in group_order() + ["all"]:
        m = np.ones(len(items), bool) if g == "all" else groups == g
        if not m.any():
            continue
        out[g] = {mat: {"n": int((m & (items.mat == mat)).sum()),
                        "flush_rate": (float(items.y[m & (items.mat == mat)].mean())
                                       if (m & (items.mat == mat)).any() else float("nan"))}
                  for mat in MATERIAL_ORDER}
    return out


def camera_geometry(items: data.Items, groups: np.ndarray) -> dict:
    heights, tilts, ok = [], [], []
    for item_id in items.ids:
        path = config.FRAMES / f"{item_id}_meta.json"
        pose = crops.camera_pose(json.loads(path.read_text())) if path.exists() else None
        ok.append(pose is not None)
        heights.append(100.0 * pose[0] if pose else np.nan)
        tilts.append(pose[1] if pose else np.nan)
    h, t = np.array(heights, float), np.array(tilts, float)
    out: dict = {"n_with_pose": int(np.sum(ok)), "per_group": {}}
    for g in group_order() + ["all"]:
        m = np.ones(len(items), bool) if g == "all" else groups == g
        m = m & np.isfinite(h)
        if not m.any():
            continue
        out["per_group"][g] = {
            "n": int(m.sum()),
            "height_cm": {f"p{int(p)}": float(np.percentile(h[m], p))
                          for p in GEOMETRY_PERCENTILES},
            "tilt_deg": {f"p{int(p)}": float(np.percentile(t[m], p))
                         for p in GEOMETRY_PERCENTILES}}
    out["height_cm"] = h.tolist()
    out["tilt_deg"] = t.tolist()
    return out


def quantity_distributions(items: data.Items) -> dict:
    from trashnet import baselines

    values = {"bbox_area_px": baselines.bbox_area_px(items).astype(float),
              "area_cm2": items.area_cm2.astype(float),
              "support_rel_mm": items.support_rel_mm.astype(float)}
    out: dict = {}
    for key, label in DISTRIBUTIONS:
        v = values[key]
        out[key] = {"label": label}
        for action, mask in (("flush", items.y == 1), ("bin", items.y == 0)):
            sel = v[mask & np.isfinite(v)]
            out[key][action] = {
                "n": int(sel.size),
                "percentiles": {f"p{int(p)}": float(np.percentile(sel, p))
                                for p in GEOMETRY_PERCENTILES} if sel.size else {},
                "values": sel.tolist()}
        tp = (items.mat == "tp") & np.isfinite(v)
        out[key]["tp_flush"] = v[tp & (items.y == 1)].tolist()
        out[key]["tp_bin"] = v[tp & (items.y == 0)].tolist()
    return out


def gallery(items: data.Items) -> dict:
    rng = np.random.default_rng(config.SEED)

    def pick(mask: np.ndarray, k: int) -> list[str]:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        return [str(i) for i in items.ids[rng.permutation(idx)[:k]]]

    out: dict = {}
    for spec in GALLERY:
        k = int(spec["per_group"])
        if spec["key"] == "material":
            out[spec["name"]] = {m: pick(items.mat == m, k) for m in MATERIAL_ORDER}
        elif spec["key"] == "tp_action":
            out[spec["name"]] = {
                "tp_flush": pick((items.mat == "tp") & (items.y == 1), k),
                "tp_bin": pick((items.mat == "tp") & (items.y == 0), k)}
        else:
            out[spec["name"]] = {"wipe": pick(items.mat == "wipe", k),
                                 "tp": pick(items.mat == "tp", k)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=EXP)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-geometry", action="store_true")
    args = ap.parse_args(argv)
    if args.dry_run:
        print(f"{EXP}/dataset stats over {config.LABELS}")
        return 0
    items = data.load_all()
    groups = np.array([robot_group(r) for r in items.robot])
    res = {
        "exp": EXP, "name": "dataset",
        "n_items": len(items), "n_robots": int(np.unique(items.robot).size),
        "n_robot_days": int(np.unique(items.group).size),
        "train_robots": list(config.TRAIN_ROBOTS),
        "per_robot": per_robot_stats(items, groups),
        "material_matrix": material_matrix(items, groups),
        "quantities": quantity_distributions(items),
        "gallery": gallery(items),
        "geometry": {} if args.skip_geometry else camera_geometry(items, groups),
    }
    path = evaluate.write_result(res)
    pool = res["per_robot"]
    print(f"items={res['n_items']} robots={res['n_robots']} robot_days={res['n_robot_days']}")
    print(f"train pool flush rate={pool['all']['flush_rate']:.3f} "
          f"tp binned={pool['all']['tp_binned']}")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
