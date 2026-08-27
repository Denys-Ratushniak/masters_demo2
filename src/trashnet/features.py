from __future__ import annotations

import numpy as np
import timm
import torch
from torch import Tensor, nn

from . import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECS: dict[str, dict] = {
    "tinycnn": {"px": 64},
    "efficientnet_b0": {},
    "resnet18": {},
    "resnet50": {},
    "convnext_tiny": {},
    "convnext_base": {},
    "mobilenetv3_large_100": {},
    "swin_tiny_patch4_window7_224": {"px": 224},
    "vit_small_patch14_dinov2": {"timm": "vit_small_patch14_dinov2.lvd142m", "px": 224,
                                 "kwargs": {"img_size": 224}},
    "vit_small_patch16_224.augreg": {"timm": "vit_small_patch16_224.augreg_in21k_ft_in1k",
                                     "px": 224},
}


def spec(name: str) -> dict:
    return SPECS.get(name, {})


def timm_name(name: str) -> str:
    return spec(name).get("timm", name)


def required_px(name: str) -> int | None:
    return spec(name).get("px")


class TinyCNN(nn.Module):
    def __init__(self, in_chans: int = 3, width: int = 32) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        c = in_chans
        for i in range(3):
            o = width * (2 ** i)
            blocks += [nn.Conv2d(c, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                       nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c = o
        self.net = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.num_features = c

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def first_conv(trunk: nn.Module) -> tuple[nn.Module, str]:
    name = (getattr(trunk, "pretrained_cfg", None) or {}).get("first_conv")
    assert name, "trunk exposes no first_conv in its pretrained config"
    parts = str(name).split(".")
    mod = trunk
    for q in parts[:-1]:
        mod = getattr(mod, q)
    return mod, parts[-1]


def expand_in_chans(trunk: nn.Module, in_chans: int) -> nn.Module:
    parent, attr = first_conv(trunk)
    conv = getattr(parent, attr)
    w = conv.weight.data
    assert w.shape[1] == 3 and in_chans > 3, (tuple(w.shape), in_chans)
    extra = w.mean(1, keepdim=True).repeat(1, in_chans - 3, 1, 1)
    wide = nn.Conv2d(in_chans, conv.out_channels, conv.kernel_size, stride=conv.stride,
                     padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
                     bias=conv.bias is not None)
    with torch.no_grad():
        wide.weight.copy_(torch.cat([w, extra], 1))
        if conv.bias is not None:
            wide.bias.copy_(conv.bias.data)
    setattr(parent, attr, wide)
    return trunk


def build_trunk(name: str, in_chans: int = 3, pretrained: bool = True) -> tuple[nn.Module, int]:
    if name == "tinycnn":
        trunk = TinyCNN(in_chans)
        return trunk, int(trunk.num_features)
    kwargs = dict(spec(name).get("kwargs", {}))
    if in_chans > 3 and pretrained:
        trunk = timm.create_model(timm_name(name), pretrained=True, num_classes=0,
                                  in_chans=3, **kwargs)
        expand_in_chans(trunk, in_chans)
    else:
        trunk = timm.create_model(timm_name(name), pretrained=pretrained, num_classes=0,
                                  in_chans=in_chans, **kwargs)
    px = required_px(name) or config.CROP_PX
    was_training = trunk.training
    trunk.eval()
    with torch.no_grad():
        dim = int(trunk(torch.zeros(1, in_chans, px, px)).shape[-1])
    trunk.train(was_training)
    return trunk, dim


def resize(x: Tensor, px: int | None) -> Tensor:
    if px is None or x.shape[-1] == px:
        return x
    return nn.functional.interpolate(x, size=(px, px), mode="bilinear", align_corners=False)


def embed(trunk: nn.Module, x: np.ndarray, batch: int = 64, amp: bool = True,
          px: int | None = None) -> np.ndarray:
    trunk = trunk.to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(np.asarray(x[i:i + batch], np.float32)).to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=amp and DEVICE.type == "cuda"):
                out.append(trunk(resize(xb, px)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def embed_cached(tag: str, trunk_name: str, build, n: int, batch: int = 32) -> np.ndarray:
    d = config.CACHE / "features"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{tag}_{trunk_name}_n{n}.npy"
    if path.exists():
        z = np.load(path)
        if len(z) == n:
            return z
    x = build()
    trunk, _ = build_trunk(trunk_name, in_chans=x.shape[1])
    z = embed(trunk, x, batch=batch, px=required_px(trunk_name))
    del trunk, x
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    np.save(path, z)
    return z
