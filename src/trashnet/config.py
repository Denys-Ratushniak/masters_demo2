from __future__ import annotations

import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent


def _dir(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, str(default))).expanduser()


DATASET = _dir("TRASH_DATA", REPO_ROOT / "trash_dataset_v1")
FRAMES = DATASET / "frames"
LABELS = DATASET / "labels.jsonl"
RESULTS = _dir("TRASH_RESULTS", PACKAGE_ROOT / "results")
FIGURES = _dir("TRASH_FIGURES", PACKAGE_ROOT / "figures")
CACHE = _dir("TRASH_CACHE", PACKAGE_ROOT / "cache")
TEST_ACCESS_LOG = RESULTS / "TEST_ACCESS.log"

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

DEFAULT_TRAIN_ROBOTS: tuple[str, ...] = ("robot_a", "robot_b", "robot_c", "robot_d", "robot_e")
ROBOTS_FILE = _dir("TRASH_ROBOTS", PACKAGE_ROOT / "robots.json")


def robots_file() -> dict:
    return json.loads(ROBOTS_FILE.read_text()) if ROBOTS_FILE.exists() else {}


def _train_robots() -> tuple[str, ...]:
    raw = os.environ.get("TRASH_TRAIN_ROBOTS", "")
    if raw:
        return tuple(v.strip() for v in raw.split(",") if v.strip())
    names = robots_file().get("train_robots")
    if names:
        return tuple(str(n) for n in names)
    return DEFAULT_TRAIN_ROBOTS


TRAIN_ROBOTS: tuple[str, ...] = _train_robots()
MATERIALS: tuple[str, ...] = ("tp", "towel", "wipe", "wrapper", "plastic", "other")
MATERIALS3: tuple[str, ...] = ("tp", "paper_other", "nonpaper")
MATERIAL_TO3: dict[str, str] = {"tp": "tp", "towel": "paper_other", "wrapper": "paper_other",
                                "wipe": "nonpaper", "plastic": "nonpaper", "other": "nonpaper"}
SOURCES: tuple[str, ...] = ("staged", "robot_detection")
BUDGETS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
SCALARS: tuple[str, ...] = ("area_cm2", "bulk_mm_per_cm", "h_p90_mm", "h_med_mm", "solidity",
                            "elongation", "floor_noise_mm", "z_m", "on_fixture_height_mm")
EXTRA_SCALARS: tuple[str, ...] = ("support_rel_mm", "support_missing", "area_cm2_label",
                                  "bbox_w", "bbox_h", "bbox_px_area")

CROP_PX = 288
DEFAULT_CROP = "fixed420_288"
LARGE_CROP = "fixed420_384"
HEIGHT_CLIP_M: tuple[float, float] = (-0.05, 0.15)
DEPTH_SCALE_M = 1.5
HEIGHT_SCALE_M = 0.05
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
SEED = 20260827
INNER_FOLDS = 4
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
ROBOT_NAMES = os.environ.get("TRASH_ROBOT_NAMES", "")


def cache_path(crop: str = DEFAULT_CROP) -> Path:
    return CACHE / f"{crop}.npz"


def budget_key(budget: float) -> str:
    return f"ff{int(round(budget * 100))}"


def results_dir(exp: str) -> Path:
    direct = RESULTS / exp
    alias = RESULTS / exp.upper()
    return alias if not direct.exists() and alias.exists() else direct
