from __future__ import annotations

from trashnet import data, evaluate

EXP = "e2"
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
BACKBONES: tuple[str, ...] = ("convnext_tiny", "resnet50")
CROPS: tuple[tuple[str, str], ...] = (
    ("fixed336_288", "fixed 336 px window resized to 288"),
    ("fixed420_288", "fixed 420 px window resized to 288"),
    ("fixed560_288", "fixed 560 px window resized to 288"),
    ("prop_288", "proportional 3x bbox window, min side 180 px"),
    ("bbox_warp", "bbox warped to 288 square"),
    ("bbox_warp_pad25", "bbox padded 25 percent and warped"),
    ("bbox_letterbox", "bbox letterboxed into 288 square"),
    ("bbox_square", "bbox expanded to a square with 25 percent pad"),
    ("frame_bboxdraw", "whole frame with the bbox drawn on the RGB"),
)
REFERENCE_CROP = "fixed420_288"

GRID: list[dict] = [
    {"name": f"{backbone}_{crop}", "regime": "frozen", "backbone": backbone, "crop": crop,
     "head": "mlp", "notes": note}
    for backbone in BACKBONES for crop, note in CROPS
]


def reference(backbone: str) -> str:
    return f"{backbone}_{REFERENCE_CROP}"


def main(argv: list[str] | None = None) -> int:
    args = evaluate.build_parser(EXP, SEEDS).parse_args(argv)
    if args.dry_run:
        evaluate.run_grid(EXP, GRID, args)
        return 0
    evaluate.run_grid(EXP, GRID, args, items=data.load_pool())
    summary: dict = {"exp": EXP, "name": "summary", "backbones": list(BACKBONES),
                     "reference_crop": REFERENCE_CROP, "blocks": {}, "paired": {}, "best": {}}
    for backbone in BACKBONES:
        order = [f"{backbone}_{crop}" for crop, _ in CROPS]
        sums = evaluate.summaries(EXP, order)
        if reference(backbone) not in sums:
            continue
        block = evaluate.variant_block(sums, order, reference(backbone))
        summary["blocks"][backbone] = block
        summary["paired"][backbone] = evaluate.paired_vs_reference(EXP, list(block),
                                                                  reference(backbone))
        summary["best"][backbone] = evaluate.best_by(block)
        print(evaluate.variant_table(sums, order, reference(backbone)))
    summary["issues"] = evaluate.check(EXP)
    evaluate.write_result(summary)
    print("best:", summary["best"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
