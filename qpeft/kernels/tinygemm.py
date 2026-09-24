"""Run a merged qpeft model on PyTorch's int4 tinygemm kernel (CUDA).

tinygemm computes w = (q - 8) * s + z_f with a FLOAT zero z_f per group. A merged QuantLinear
computes w = (q - z) * s, where z is float too after a QA-LoRA fold. Both are the same weight with
    z_f = (8 - z) * s,
so the merged artifact maps onto tinygemm without re-rounding the codes. The kernel runs in bf16:
scale and z_f are rounded to bf16 once, at export.

Kernel ops (PyTorch >= 2.5): aten._convert_weight_to_int4pack (uint8 input, two codes per byte,
even column in the high nibble) and aten._weight_int4pack_mm.
"""
from __future__ import annotations

import torch
from torch import nn

from ..quant_schemes import UnsupportedSchemeError
from ..tuners.tuners_utils import QuantLinear

GROUP_SIZES = (32, 64, 128, 256)
INNER_K_TILES = (8, 4, 2)


def tinygemm_zero(scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """qpeft zero-point z -> tinygemm float zero z_f, so that (q - 8) * s + z_f == (q - z) * s."""
    return (8 - zero_point) * scale


def pack_int4_pairs(codes: torch.Tensor) -> torch.Tensor:
    """(N, K) codes in [0, 15] -> (N, K // 2) uint8, column 2i in the high nibble."""
    return (codes[:, ::2] << 4 | codes[:, 1::2]).to(torch.uint8)


def _inner_k_tiles(in_features: int) -> int:
    for t in INNER_K_TILES:
        if in_features % (t * 16) == 0:
            return t
    raise UnsupportedSchemeError(f"tinygemm needs in_features to be a multiple of 32, got {in_features}.")


def check_supported(layer: QuantLinear) -> None:
    """Refuse every layer the kernel cannot run exactly as the merged artifact describes."""
    if not layer.merged:
        raise UnsupportedSchemeError("tinygemm export needs a merged layer; call merge_and_unload() first.")
    if layer.config.bits != 4:
        raise UnsupportedSchemeError(f"tinygemm is int4 only, got {layer.config.bits} bits.")
    if layer.config.group_size not in GROUP_SIZES:
        raise UnsupportedSchemeError(f"tinygemm group sizes are {GROUP_SIZES}, got {layer.config.group_size}.")
    if layer.out_features % 8:
        raise UnsupportedSchemeError(f"tinygemm needs out_features to be a multiple of 8, got {layer.out_features}.")
    _inner_k_tiles(layer.in_features)
    if not torch.cuda.is_available():
        raise UnsupportedSchemeError("tinygemm is a CUDA kernel; no CUDA device is available.")
    if tuple(int(v) for v in torch.__version__.split(".")[:2]) < (2, 5):
        raise UnsupportedSchemeError(f"tinygemm export needs torch >= 2.5 (uint8 int4pack), got {torch.__version__}.")


class TinyGemmLinear(nn.Module):
    """Inference-only linear on the tinygemm kernel, built from a merged QuantLinear."""

    def __init__(self, layer: QuantLinear):
        super().__init__()
        check_supported(layer)
        dev = torch.device("cuda")
        self.in_features, self.out_features = layer.in_features, layer.out_features
        self.group_size = layer.config.group_size
        self.inner_k_tiles = _inner_k_tiles(self.in_features)
        self.register_buffer("weight_int4pack", torch.ops.aten._convert_weight_to_int4pack(
            pack_int4_pairs(layer.codes.to(dev)), self.inner_k_tiles))
        s = layer.scale.detach().to(dev, torch.float32)
        z_f = tinygemm_zero(s, layer.zero_point.detach().to(dev, torch.float32))
        self.register_buffer("scales_and_zeros",
                             torch.stack([s, z_f], -1).transpose(0, 1).contiguous().to(torch.bfloat16))
        self.register_buffer("bias", None if layer.bias is None else layer.bias.detach().to(dev))

    def forward(self, x):
        lead = x.shape[:-1]
        y = torch.ops.aten._weight_int4pack_mm(x.reshape(-1, self.in_features).to(torch.bfloat16),
                                              self.weight_int4pack, self.group_size, self.scales_and_zeros)
        y = y.reshape(*lead, self.out_features).to(x.dtype)
        return y if self.bias is None else y + self.bias.to(x.dtype)


def to_tinygemm(model: nn.Module) -> nn.Module:
    """Replace every merged QuantLinear in `model` (a merged QuantModel or its base) with a
    TinyGemmLinear, in place. All layers are checked before the first one is replaced."""
    root = getattr(model, "base", model)
    targets = [(n, m) for n, m in root.named_modules() if isinstance(m, QuantLinear)]
    if not targets:
        raise ValueError("no QuantLinear found; nothing to export.")
    for _, m in targets:
        check_supported(m)
    for name, m in targets:
        parent = root.get_submodule(name.rsplit(".", 1)[0]) if "." in name else root
        setattr(parent, name.rsplit(".", 1)[-1], TinyGemmLinear(m))
    return model
