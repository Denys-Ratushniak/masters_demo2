from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from . import config

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FLEET_WORD = "тестових роботів"
AGGREGATE: tuple[str, ...] = ("all", "other_robots", "pool", "test")


def overrides() -> dict[str, str]:
    table = {str(k): str(v) for k, v in config.robots_file().get("display", {}).items()}
    raw = config.ROBOT_NAMES
    if raw:
        path = Path(raw).expanduser()
        if path.exists():
            table.update({str(k): str(v) for k, v in json.loads(path.read_text()).items()})
    return table


def mapping(names: Sequence[str] = ()) -> dict[str, str]:
    known = list(config.TRAIN_ROBOTS)
    extra = sorted({str(n) for n in names} - set(known) - set(AGGREGATE))
    out = {r: ALPHABET[i % len(ALPHABET)] for i, r in enumerate(known + extra)}
    out.update(overrides())
    return out


def robot(name: str, names: Sequence[str] = ()) -> str:
    return mapping(list(names) or [name]).get(str(name), str(name))


def robots(names: Sequence[str]) -> list[str]:
    table = mapping(names)
    return [table.get(str(n), str(n)) for n in names]


def fleet(n: int, word: str = FLEET_WORD) -> str:
    return f"{int(n)} {word}"
