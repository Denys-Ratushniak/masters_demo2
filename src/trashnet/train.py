from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, fields
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from . import metrics
from .data import Items
from .models import TrashNet, build_head, logreg_fit_predict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    backbone: str = "convnext_tiny"
    epochs: int = 20
    batch: int = 32
    lr_trunk: float = 2e-4
    lr_head: float = 1e-3
    wd: float = 1e-2
    aug: str = "none"
    material_weight: float = 1.0
    pos_weight: str = "balanced"
    freeze_frac: float = 0.0
    in_chans: int = 3
    geo_channels: tuple[int, ...] | None = None
    use_rgb: bool = True
    fusion: str = "early"
    scalar_dim: int = 0
    label_space: str = "binary"
    early_stop: bool = True
    patience: int = 4
    amp: str = "bf16"
    px: int = 288
    loss: str = "bce"
    class_weight_ff: float = 1.0
    focal_gamma: float = 2.0
    label_smooth: float = 0.0
    action_weight: float = 1.0
    hidden: int = 128
    depth: int = 2
    dropout: float = 0.5
    head: str = "mlp"
    head_epochs: int = 200
    head_batch: int = 256
    logreg_c: float = 0.01
    material_balanced: bool = True
    ema_decay: float = 0.0
    tta_flip: bool = False
    size_aux_weight: float = 0.0

    @property
    def size_aux(self) -> int:
        return 2 if self.size_aux_weight > 0.0 else 0

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def n_materials(self) -> int:
        return {"binary_plus_material3": 3, "binary_plus_material6": 6}.get(self.label_space, 0)

    @property
    def amp_dtype(self):
        return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.amp)


@dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray
    m: np.ndarray | None = None
    sc: np.ndarray | None = None
    tg: np.ndarray | None = None

    def take(self, idx: np.ndarray) -> "Batch":
        return Batch(self.x[idx], self.y[idx],
                     None if self.m is None else self.m[idx],
                     None if self.sc is None else self.sc[idx],
                     None if self.tg is None else self.tg[idx])


def pos_weight_value(y: np.ndarray, mode: str) -> float | None:
    if mode in (None, "none", ""):
        return None
    if mode == "balanced":
        return float((y == 0).sum()) / max(int((y == 1).sum()), 1)
    return float(mode)


def augment(x: Tensor, mode: str, gen: torch.Generator) -> Tensor:
    if mode == "none":
        return x
    n = x.shape[0]
    if mode in ("flip", "geom", "all"):
        flip = torch.rand(n, generator=gen, device=x.device) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
    if mode in ("color", "all"):
        b = 1.0 + 0.3 * (torch.rand(n, 1, 1, 1, generator=gen, device=x.device) - 0.5)
        c = 0.3 * (torch.rand(n, 1, 1, 1, generator=gen, device=x.device) - 0.5)
        x = x * b + c
    if mode in ("geom", "all"):
        ang = (torch.rand(n, generator=gen, device=x.device) - 0.5) * (20.0 * np.pi / 180.0)
        scale = 1.0 + 0.2 * (torch.rand(n, generator=gen, device=x.device) - 0.5)
        tx = 0.1 * (torch.rand(n, generator=gen, device=x.device) - 0.5)
        ty = 0.1 * (torch.rand(n, generator=gen, device=x.device) - 0.5)
        cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
        theta = torch.stack([torch.stack([cos, -sin, tx], 1),
                             torch.stack([sin, cos, ty], 1)], 1)
        grid = nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
        x = nn.functional.grid_sample(x, grid, align_corners=False, padding_mode="border")
    return x


def action_loss(logit: Tensor, y: Tensor, cfg: TrainConfig, pw: float | None) -> Tensor:
    w = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, cfg.class_weight_ff))
    if cfg.label_smooth > 0.0:
        y = y * (1.0 - cfg.label_smooth) + 0.5 * cfg.label_smooth
    if cfg.loss == "focal" and cfg.label_smooth > 0.0:
        p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
        alpha = pw if pw else 1.0
        loss = -(alpha * y * (1.0 - p) ** cfg.focal_gamma * torch.log(p)
                 + (1.0 - y) * p ** cfg.focal_gamma * torch.log(1.0 - p))
    elif cfg.loss == "focal":
        p = torch.sigmoid(logit)
        pt = torch.where(y > 0.5, p, 1.0 - p).clamp(1e-6, 1 - 1e-6)
        alpha = torch.where(y > 0.5, torch.full_like(y, pw if pw else 1.0), torch.ones_like(y))
        loss = -alpha * (1.0 - pt) ** cfg.focal_gamma * torch.log(pt)
    else:
        pos = None if pw is None else torch.tensor([pw], device=logit.device)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=pos,
                                                              reduction="none")
    return (loss * w).mean()


def material_weights(m: np.ndarray, k: int, balanced: bool) -> Tensor | None:
    if not balanced or k <= 0:
        return None
    cnt = np.bincount(m, minlength=k).astype(np.float64)
    inv = np.where(cnt > 0, len(m) / np.maximum(cnt, 1) / k, 0.0)
    return torch.tensor(inv, dtype=torch.float32, device=DEVICE)


def _gpu(a: np.ndarray | None, dtype) -> Tensor | None:
    return None if a is None else torch.as_tensor(np.asarray(a), dtype=dtype, device=DEVICE)


def _snapshot(model: nn.Module) -> dict:
    return copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})


def _gpu_mem_mb() -> float:
    return round(torch.cuda.max_memory_allocated() / 1e6, 1) if DEVICE.type == "cuda" else 0.0


def predict(model: nn.Module, x: np.ndarray, sc: np.ndarray | None = None, batch: int = 64,
            amp=torch.bfloat16, flip: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    pa, pm = [], []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.as_tensor(np.asarray(x[i:i + batch], np.float32), device=DEVICE)
            sb = _gpu(sc[i:i + batch], torch.float32) if sc is not None else None
            with torch.autocast("cuda", dtype=amp, enabled=DEVICE.type == "cuda"):
                la, lm = model(xb, sb)
                action = torch.sigmoid(la.float())
                material = torch.softmax(lm.float(), -1) if lm is not None else None
                if flip:
                    la2, lm2 = model(xb.flip(-1), sb)
                    action = 0.5 * (action + torch.sigmoid(la2.float()))
                    if material is not None and lm2 is not None:
                        material = 0.5 * (material + torch.softmax(lm2.float(), -1))
            pa.append(action.cpu().numpy())
            if material is not None:
                pm.append(material.cpu().numpy())
    return np.concatenate(pa), (np.concatenate(pm) if pm else None)


def size_loss(pred: Tensor, target: Tensor) -> Tensor:
    weight = target[:, 2:]
    sq = (pred.float() - target[:, :2]) ** 2
    return (sq * weight).sum() / weight.sum().clamp(min=1.0)


def ema_update(shadow: dict, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                shadow[key].mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                shadow[key].copy_(value)


def train_finetune(cfg: TrainConfig, train: Batch, val: Batch | None = None,
                   seed: int = 0) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    k = cfg.n_materials
    model = TrashNet(cfg.backbone, k, hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout,
                     in_chans=cfg.in_chans, scalar_dim=cfg.scalar_dim,
                     freeze_frac=cfg.freeze_frac, px=cfg.px, fusion=cfg.fusion,
                     size_aux=cfg.size_aux).to(DEVICE)
    trunk_p = [p for p in model.trunk_parameters() if p.requires_grad]
    groups, max_lr = [], []
    if trunk_p:
        groups.append({"params": trunk_p, "lr": cfg.lr_trunk})
        max_lr.append(cfg.lr_trunk)
    groups.append({"params": list(model.head.parameters()), "lr": cfg.lr_head})
    max_lr.append(cfg.lr_head)
    opt = torch.optim.AdamW(groups, weight_decay=cfg.wd)
    n = len(train.x)
    steps = max(int(np.ceil(n / cfg.batch)), 1) * max(cfg.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=steps)
    pw = pos_weight_value(train.y, cfg.pos_weight)
    ce_w = material_weights(train.m, k, cfg.material_balanced) if (k and train.m is not None) else None
    ce = nn.CrossEntropyLoss(weight=ce_w)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    rng = np.random.default_rng(seed)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best: dict = {"score": -np.inf, "epoch": -1, "state": None}
    hist: list[float] = []
    ema = ({k: v.detach().clone() for k, v in model.state_dict().items()}
           if cfg.ema_decay > 0.0 else None)
    t0, done = time.time(), 0
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(n)
        for i in range(0, n, cfg.batch):
            sel = order[i:i + cfg.batch]
            xb = augment(torch.as_tensor(np.asarray(train.x[sel], np.float32), device=DEVICE),
                         cfg.aug, gen)
            yb = _gpu(train.y[sel].astype(np.float32), torch.float32)
            sb = _gpu(train.sc[sel], torch.float32) if train.sc is not None else None
            mb = _gpu(train.m[sel].astype(np.int64), torch.long) if (k and train.m is not None) else None
            tb = _gpu(train.tg[sel], torch.float32) if train.tg is not None else None
            with torch.autocast("cuda", dtype=cfg.amp_dtype, enabled=DEVICE.type == "cuda"):
                la, lm = model(xb, sb)
                loss = cfg.action_weight * action_loss(la.float(), yb, cfg, pw)
                if mb is not None and lm is not None:
                    loss = loss + cfg.material_weight * ce(lm.float(), mb)
                if tb is not None and model.size_out is not None:
                    loss = loss + cfg.size_aux_weight * size_loss(model.size_out, tb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema is not None:
                ema_update(ema, model, cfg.ema_decay)
            if sched.last_epoch + 1 < steps:
                sched.step()
        done = epoch + 1
        if val is None:
            continue
        pv, _ = predict(model, val.x, val.sc, batch=cfg.batch * 2, flip=cfg.tta_flip)
        score = metrics.auprc(val.y, pv) if np.unique(val.y).size > 1 else float("nan")
        hist.append(float(score))
        if score > best["score"]:
            best = {"score": float(score), "epoch": epoch, "state": _snapshot(model)}
        if cfg.early_stop and epoch - best["epoch"] >= cfg.patience:
            break
    if cfg.early_stop and best["state"] is not None:
        model.load_state_dict(best["state"])
    if ema is not None:
        model.load_state_dict(ema)
    info = {"epochs_run": done,
            "best_epoch": int(best["epoch"]) if best["epoch"] >= 0 else done - 1,
            "train_time_s": round(time.time() - t0, 2), "gpu_mem_mb": _gpu_mem_mb(),
            "val_auprc": hist,
            "best_val_auprc": float(best["score"]) if np.isfinite(best["score"]) else None}
    return model, info


def train_frozen(cfg: TrainConfig, train: Batch, z_te: np.ndarray, val: Batch | None = None,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray | None, dict]:
    t0 = time.time()
    if cfg.head == "logreg":
        p = logreg_fit_predict(train.x, train.y, z_te, c_reg=cfg.logreg_c)
        return p.astype(np.float32), None, {"epochs_run": 1, "best_epoch": 0,
                                            "train_time_s": round(time.time() - t0, 2),
                                            "gpu_mem_mb": 0.0, "best_val_auprc": None}
    torch.manual_seed(seed)
    k = cfg.n_materials
    mu, sd = train.x.mean(0, keepdims=True), train.x.std(0, keepdims=True) + 1e-6
    ztr, zte = (train.x - mu) / sd, (z_te - mu) / sd
    model = build_head(ztr.shape[1], k, cfg.hidden, cfg.depth, cfg.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_head, weight_decay=cfg.wd)
    pw = pos_weight_value(train.y, cfg.pos_weight)
    ce_w = material_weights(train.m, k, cfg.material_balanced) if (k and train.m is not None) else None
    ce = nn.CrossEntropyLoss(weight=ce_w)
    xb_all = _gpu(ztr.astype(np.float32), torch.float32)
    yb_all = _gpu(train.y.astype(np.float32), torch.float32)
    mb_all = _gpu(train.m.astype(np.int64), torch.long) if (k and train.m is not None) else None
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    zv = ((val.x - mu) / sd).astype(np.float32) if val is not None else None
    rng = np.random.default_rng(seed)
    n = len(ztr)
    best: dict = {"score": -np.inf, "epoch": -1, "state": None}
    epoch = 0
    for epoch in range(cfg.head_epochs):
        model.train()
        order = rng.permutation(n)
        for i in range(0, n, cfg.head_batch):
            sel = torch.as_tensor(order[i:i + cfg.head_batch], device=DEVICE)
            la, lm = model(xb_all[sel])
            loss = cfg.action_weight * action_loss(la.float(), yb_all[sel], cfg, pw)
            if mb_all is not None and lm is not None:
                loss = loss + cfg.material_weight * ce(lm.float(), mb_all[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if zv is None or not (epoch % 5 == 4 or epoch == cfg.head_epochs - 1):
            continue
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(_gpu(zv, torch.float32))[0]).cpu().numpy()
        score = metrics.auprc(val.y, pv) if np.unique(val.y).size > 1 else float("nan")
        if score > best["score"]:
            best = {"score": float(score), "epoch": epoch, "state": _snapshot(model)}
        elif cfg.early_stop and epoch - best["epoch"] >= 5 * cfg.patience:
            break
    if cfg.early_stop and best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        la, lm = model(_gpu(zte.astype(np.float32), torch.float32))
        pa = torch.sigmoid(la.float()).cpu().numpy()
        pm = torch.softmax(lm.float(), -1).cpu().numpy() if lm is not None else None
    info = {"epochs_run": epoch + 1, "best_epoch": int(max(best["epoch"], 0)),
            "train_time_s": round(time.time() - t0, 2), "gpu_mem_mb": _gpu_mem_mb(),
            "best_val_auprc": float(best["score"]) if np.isfinite(best["score"]) else None}
    return pa.astype(np.float32), pm, info


def fit_predict(cfg: TrainConfig, regime: str, train: Batch, test: Batch,
                val: Batch | None = None,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray | None, dict]:
    if regime == "frozen":
        return train_frozen(cfg, train, test.x,
                            val=None if val is None else Batch(val.x, val.y), seed=seed)
    model, info = train_finetune(cfg, train, val=val, seed=seed)
    t1 = time.time()
    s_te, pm_te = predict(model, test.x, test.sc, batch=max(cfg.batch * 2, 32),
                          flip=cfg.tta_flip)
    info["eval_time_s"] = round(time.time() - t1, 2)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return s_te, pm_te, info


def oof_scores(items: Items, train_idx: np.ndarray, cfg: TrainConfig, seed: int,
               inner: Sequence[tuple[np.ndarray, np.ndarray]], pool: Batch,
               regime: str = "finetune") -> np.ndarray:
    from .folds import inner_folds

    pos = {int(g): i for i, g in enumerate(train_idx)}
    oof = np.full(len(train_idx), np.nan, np.float32)
    for fit_idx, val_idx in inner:
        core, es = inner_folds(items, fit_idx, seed=seed)[0]
        p, _, _ = fit_predict(cfg, regime, pool.take(core), pool.take(val_idx),
                              val=pool.take(es), seed=seed)
        for g, v in zip(val_idx, p):
            oof[pos[int(g)]] = v
    return oof
