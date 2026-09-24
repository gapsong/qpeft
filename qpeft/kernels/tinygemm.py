"""Run a merged qpeft model on PyTorch's int4 tinygemm kernel (CUDA).

tinygemm computes    w = (q - 8) * s + z_f    with a float zero z_f per group.
A merged QuantLinear computes    w = (q - z) * s    (z is fractional after a QA-LoRA fold).
Both give the same weight when
    z_f = (8 - z) * s,
so the merged codes go into tinygemm as they are, without rounding again.
The kernel runs in bf16: scale and z_f are rounded to bf16 once, at export.

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


def to_tinygemm(model: nn.Module) -> nn.Module:
    """Replace every merged QuantLinear in `model` (a merged QuantModel or its base) with a
    TinyGemmLinear, in place. All layers are checked before the first one is replaced."""
    root = getattr(model, "base", model)
    targets = [(name, m) for name, m in root.named_modules() if isinstance(m, QuantLinear)]
    if not targets:
        raise ValueError("no QuantLinear found; nothing to export.")
    for _, layer in targets:
        check_supported(layer)
    for name, layer in targets:
        parent_name, _, attr = name.rpartition(".")
        setattr(root.get_submodule(parent_name), attr, TinyGemmLinear(layer))
    return model


class TinyGemmLinear(nn.Module):
    """Inference-only linear on the tinygemm kernel, built from a merged QuantLinear."""

    def __init__(self, layer: QuantLinear):
        super().__init__()
        check_supported(layer)
        cuda = torch.device("cuda")
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.group_size = layer.config.group_size
        self.inner_k_tiles = _inner_k_tiles(self.in_features)

        packed = pack_int4_pairs(layer.codes.to(cuda))
        self.register_buffer("weight_int4pack",
                             torch.ops.aten._convert_weight_to_int4pack(packed, self.inner_k_tiles))
        scale = layer.scale.detach().to(cuda, torch.float32)
        zero = tinygemm_zero(scale, layer.zero_point.detach().to(cuda, torch.float32))
        # Kernel layout: (n_groups, out_features, 2) with [scale, zero] in the last dim.
        self.register_buffer("scales_and_zeros",
                             torch.stack([scale, zero], -1).transpose(0, 1).contiguous().to(torch.bfloat16))
        self.register_buffer("bias", None if layer.bias is None else layer.bias.detach().to(cuda))

    def forward(self, x):
        lead = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features).to(torch.bfloat16)
        y = torch.ops.aten._weight_int4pack_mm(x_2d, self.weight_int4pack, self.group_size, self.scales_and_zeros)
        y = y.reshape(*lead, self.out_features).to(x.dtype)
        return y if self.bias is None else y + self.bias.to(x.dtype)


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


def tinygemm_zero(scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """qpeft zero-point z -> tinygemm float zero z_f, so that (q - 8) * s + z_f == (q - z) * s."""
    return (8 - zero_point) * scale


def pack_int4_pairs(codes: torch.Tensor) -> torch.Tensor:
    """(N, K) codes in [0, 15] -> (N, K // 2) uint8, column 2i in the high nibble."""
    even_columns, odd_columns = codes[:, ::2], codes[:, 1::2]
    return (even_columns << 4 | odd_columns).to(torch.uint8)


def _inner_k_tiles(in_features: int) -> int:
    """The largest kernel tile count that divides in_features."""
    for tiles in INNER_K_TILES:
        if in_features % (tiles * 16) == 0:
            return tiles
    raise UnsupportedSchemeError(f"tinygemm needs in_features to be a multiple of 32, got {in_features}.")
