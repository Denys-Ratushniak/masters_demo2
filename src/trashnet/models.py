from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from . import features


def _mlp_body(dim: int, hidden: int, depth: int, dropout: float) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    d = dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden
    return nn.Sequential(*layers), d


class OneHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, depth: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        body, d = _mlp_body(dim, hidden, depth, dropout)
        self.net = nn.Sequential(body, nn.Linear(d, 1))

    def forward(self, x: Tensor) -> tuple[Tensor, None]:
        return self.net(x).squeeze(-1), None


class TwoHead(nn.Module):
    def __init__(self, dim: int, n_materials: int, hidden: int = 128, depth: int = 2,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.body, d = _mlp_body(dim, hidden, depth, dropout)
        self.action = nn.Linear(d, 1)
        self.material = nn.Linear(d, n_materials)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.body(x)
        return self.action(h).squeeze(-1), self.material(h)


def build_head(dim: int, n_materials: int, hidden: int, depth: int, dropout: float) -> nn.Module:
    if n_materials > 0:
        return TwoHead(dim, n_materials, hidden, depth, dropout)
    return OneHead(dim, hidden, depth, dropout)


class TrashNet(nn.Module):
    def __init__(self, backbone: str, n_materials: int, hidden: int = 128, depth: int = 2,
                 dropout: float = 0.5, in_chans: int = 3, scalar_dim: int = 0,
                 freeze_frac: float = 0.0, pretrained: bool = True,
                 px: int | None = None, fusion: str = "early",
                 size_aux: int = 0) -> None:
        super().__init__()
        self.fusion = fusion
        self.trunk_geo: nn.Module | None = None
        if fusion == "mid":
            assert in_chans > 3, "mid fusion needs geometry channels beside RGB"
            self.trunk, dim = features.build_trunk(backbone, in_chans=3, pretrained=pretrained)
            self.trunk_geo, dim_geo = features.build_trunk(backbone, in_chans=in_chans - 3,
                                                           pretrained=pretrained)
            dim += dim_geo
        else:
            self.trunk, dim = features.build_trunk(backbone, in_chans=in_chans,
                                                   pretrained=pretrained)
        self.px = features.required_px(backbone) or px
        self.scalar_dim = scalar_dim
        self.n_materials = n_materials
        self.head = build_head(dim + scalar_dim, n_materials, hidden, depth, dropout)
        self.size_head = nn.Linear(dim + scalar_dim, int(size_aux)) if size_aux else None
        self.size_out: Tensor | None = None
        freeze_trunk(self, freeze_frac)

    def trunk_parameters(self) -> list[torch.nn.Parameter]:
        out = list(self.trunk.parameters())
        if self.trunk_geo is not None:
            out += list(self.trunk_geo.parameters())
        return out

    def forward(self, x: Tensor, sc: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        x = features.resize(x, self.px)
        if self.trunk_geo is None:
            z = self.trunk(x)
        else:
            z = torch.cat([self.trunk(x[:, :3]), self.trunk_geo(x[:, 3:])], dim=1)
        if self.scalar_dim and sc is not None:
            z = torch.cat([z, sc], dim=1)
        self.size_out = self.size_head(z) if self.size_head is not None else None
        return self.head(z)


def freeze_trunk(model: TrashNet, frac: float) -> None:
    params = model.trunk_parameters()
    for p in params:
        p.requires_grad_(True)
    if frac <= 0.0:
        return
    n = len(params) if frac >= 1.0 else int(round(frac * len(params)))
    for p in params[:n]:
        p.requires_grad_(False)


def logreg_fit_predict(z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray,
                       c_reg: float = 0.01, balanced: bool = True) -> np.ndarray:
    scaler = StandardScaler().fit(z_tr)
    model = LogisticRegression(C=c_reg, max_iter=5000,
                               class_weight="balanced" if balanced else None)
    model.fit(scaler.transform(z_tr), y_tr)
    return model.predict_proba(scaler.transform(z_te))[:, list(model.classes_).index(1)]


def load_checkpoint(path: Path, pretrained: bool = False,
                    px: int | None = None) -> TrashNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    model = TrashNet(cfg["backbone"], {"binary_plus_material3": 3,
                                       "binary_plus_material6": 6}.get(cfg["label_space"], 0),
                     hidden=cfg["hidden"], depth=cfg["depth"], dropout=cfg["dropout"],
                     in_chans=cfg["in_chans"], scalar_dim=cfg["scalar_dim"],
                     pretrained=pretrained, px=px, fusion=cfg.get("fusion", "early"),
                     size_aux=2 if cfg.get("size_aux_weight", 0.0) > 0 else 0)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def save_checkpoint(model: TrashNet, cfg: dict, path: Path,
                    extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "cfg": cfg}
    payload.update(extra or {})
    torch.save(payload, path)


def benchmark_latency(model: nn.Module, px: int, mode: str = "fp32", iters: int = 300,
                      warmup: int = 50, in_chans: int = 3, batch: int = 1) -> dict:
    device = features.DEVICE
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, in_chans, px, px, device=device)
    amp = mode == "fp16"
    run = torch.compile(model) if mode == "compile" else model

    def step() -> None:
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            run(x, None)

    times: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(iters):
            t0 = time.perf_counter()
            step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(times, float)
    return {"px": px, "mode": mode, "iters": iters, "warmup": warmup,
            "median_ms": float(np.median(a)), "mean_ms": float(a.mean()),
            "p90_ms": float(np.percentile(a, 90)),
            "gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 1e6, 1)
                           if device.type == "cuda" else 0.0)}


def param_count_m(model: nn.Module) -> float:
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
