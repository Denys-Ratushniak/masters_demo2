from __future__ import annotations

from trashnet import config, data, evaluate

EXP = "e3"
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
BACKBONE = "convnext_tiny"
CROP = "prop_288"
SCALARS = list(config.SCALARS)
REFERENCE = "m1_rgb"


def _cfg(name: str, feature_source: str, **kw) -> dict:
    return dict({"name": name, "regime": "frozen", "backbone": BACKBONE, "crop": CROP,
                 "head": "mlp", "feature_source": feature_source}, **kw)


GRID: list[dict] = [
    _cfg("m1_rgb", "trunk", notes="RGB only"),
    _cfg("m2_rgb_scalars", "trunk,scalars", fuse_scalars=SCALARS,
         notes="RGB plus 9 depth scalars, late fusion, fold local standardisation"),
    _cfg("m3_rgb_scalars_support", "trunk,scalars",
         fuse_scalars=SCALARS + ["support_rel_mm"], missing_indicator=True,
         notes="RGB plus 9 scalars plus support_rel_mm and a missing indicator"),
    _cfg("m4_scalars", "scalars", fuse_scalars=SCALARS, notes="9 depth scalars only"),
    _cfg("m5_depth_image", "geo_trunk",
         notes="depth, floor relative height and validity as pseudo RGB through the same trunk"),
    _cfg("m6_rgb_plus_depth_image", "trunk,geo_trunk",
         notes="RGB features concatenated with depth image features"),
]
ABLATION: list[dict] = [
    _cfg(f"li_{s}", "trunk,scalars", fuse_scalars=[s],
         missing_indicator=(s == "support_rel_mm"), notes=f"leave one in: RGB plus {s}")
    for s in SCALARS + ["support_rel_mm"]
]
FULL_GRID = GRID + ABLATION


def main(argv: list[str] | None = None) -> int:
    ap = evaluate.build_parser(EXP, SEEDS)
    ap.add_argument("--with-ablation", action="store_true")
    args = ap.parse_args(argv)
    grid = FULL_GRID if args.with_ablation else GRID
    if args.dry_run:
        evaluate.run_grid(EXP, grid, args)
        return 0
    evaluate.run_grid(EXP, grid, args, items=data.load_pool())
    order = [c["name"] for c in grid]
    sums = evaluate.summaries(EXP, order)
    block = evaluate.variant_block(sums, order, REFERENCE)
    summary = {"exp": EXP, "name": "summary", "reference": REFERENCE, "crop": CROP,
               "modality": block,
               "paired": evaluate.paired_vs_reference(EXP, list(block), REFERENCE),
               "best": evaluate.best_by(block), "issues": evaluate.check(EXP)}
    evaluate.write_result(summary)
    print(evaluate.variant_table(sums, order, REFERENCE))
    print("best:", summary["best"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
