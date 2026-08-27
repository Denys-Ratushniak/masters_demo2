from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .data import Items


@dataclass
class Fold:
    name: str
    train: np.ndarray
    test: np.ndarray


def loro_folds(items: Items) -> list[Fold]:
    pool = np.flatnonzero(np.isin(items.robot, config.TRAIN_ROBOTS))
    folds = []
    for robot in config.TRAIN_ROBOTS:
        test = pool[items.robot[pool] == robot]
        train = pool[items.robot[pool] != robot]
        assert len(test) and len(train), f"empty LORO fold for {robot}"
        folds.append(Fold(name=robot, train=train, test=test))
    return folds


def inner_folds(items: Items, train_idx: np.ndarray, n: int = config.INNER_FOLDS,
                seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    train_idx = np.asarray(train_idx)
    splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    return [(train_idx[tr], train_idx[va])
            for tr, va in splitter.split(train_idx, items.y[train_idx], items.group[train_idx])]


def subsample_groups(items: Items, idx: np.ndarray, frac: float, seed: int) -> np.ndarray:
    idx = np.asarray(idx)
    if frac >= 1.0:
        return idx
    groups = np.unique(items.group[idx])
    rng = np.random.default_rng(seed)
    keep = rng.permutation(groups)[:max(int(round(frac * len(groups))), 1)]
    return idx[np.isin(items.group[idx], keep)]


def assert_disjoint(items: Items, a: np.ndarray, b: np.ndarray, level: str = "robot") -> None:
    key = {"robot": items.robot, "robot_day": items.group, "item": items.ids}[level]
    shared = np.intersect1d(key[np.asarray(a)], key[np.asarray(b)])
    assert shared.size == 0, f"{level} overlap: {shared[:5].tolist()}"
    ids = np.intersect1d(items.ids[np.asarray(a)], items.ids[np.asarray(b)])
    assert ids.size == 0, f"item overlap: {ids[:5].tolist()}"
