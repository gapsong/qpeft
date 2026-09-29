"""A Triton kernel for training on frozen integer codes (QA-LoRA, PEQA, EfficientQAT E2E-QP).

The torch path unpacks the GPTQ-packed `qweight` with int64 gathers and keeps the dense
weight alive until backward. Here one kernel reads the packed codes and writes the dense weight
    w = (code - z) * s
in the compute dtype, and `FrozenCodesLinear` recomputes it in backward instead of keeping it.
The matmuls stay cuBLAS (`F.linear` / `mm`), the same calls torch's autograd makes, so the
forward output and every gradient are bit-identical to the torch path.

The unpack follows the flat dequant kernel pattern of GPTQModel
(gptqmodel/nn_modules/triton_utils/dequant.py, Apache-2.0); no code is copied.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

from ..quant_schemes.reference import ReferenceIntUniformScheme

TRITON_AVAILABLE = triton is not None and torch.cuda.is_available()
KERNEL_BITS = (2, 4, 8)             # 3-bit codes cross word boundaries; they stay on the torch path
KERNEL_DTYPES = (torch.bfloat16, torch.float16)
BLOCK_OUT, BLOCK_IN = 32, 64


def kernel_supports(layer, x) -> bool:
    """True when the kernel computes exactly what the layer's torch path computes for input x."""
    if not TRITON_AVAILABLE or not x.is_cuda or x.dim() < 2 or x.dtype not in KERNEL_DTYPES:
        return False
    if not layer.codes_frozen or not layer.quant_enabled:
        return False
    # The kernel reproduces the reference scheme's dequant; torchao rounds in its own way.
    if not isinstance(layer.scheme, ReferenceIntUniformScheme) or layer.config.bits not in KERNEL_BITS:
        return False
    # A trainable zero-point gets its gradient through round_zero_point's straight-through
    # estimator, which the kernel's backward does not compute.
    if layer.zero_point.requires_grad:
        return False
    # Under autocast to another dtype, F.linear would cast the weight; the backward here would not.
    if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != x.dtype:
        return False
    return True


def dequantize(qweight, scale, zero_point, bits: int, group_size: int, apply_scale: bool = True):
    """(in * bits // 32, out) packed codes -> contiguous (out, in) weight in scale's dtype.
    apply_scale=False gives (code - z) only, which the scale gradient needs."""
    out_features, n_groups = scale.shape
    in_features = n_groups * group_size
    weight = torch.empty(out_features, in_features, dtype=scale.dtype, device=scale.device)
    grid = (triton.cdiv(out_features, BLOCK_OUT), triton.cdiv(in_features, BLOCK_IN))
    with torch.cuda.device(scale.device):
        _dequantize_kernel[grid](
            qweight, scale.contiguous(), zero_point.contiguous(), weight,
            out_features, in_features,
            BITS=bits, GROUP_SIZE=group_size, APPLY_SCALE=apply_scale,
            BLOCK_OUT=BLOCK_OUT, BLOCK_IN=BLOCK_IN)
    return weight


if triton is not None:
    @triton.jit
    def _dequantize_kernel(qweight_ptr, scale_ptr, zero_ptr, weight_ptr,
                           out_features, in_features,
                           BITS: tl.constexpr, GROUP_SIZE: tl.constexpr, APPLY_SCALE: tl.constexpr,
                           BLOCK_OUT: tl.constexpr, BLOCK_IN: tl.constexpr):
        rows = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)     # output features
        cols = tl.program_id(1) * BLOCK_IN + tl.arange(0, BLOCK_IN)       # input features
        mask = (rows[:, None] < out_features) & (cols[None, :] < in_features)

        # qweight is (in * BITS // 32, out): code k of output o sits in word k // (32 // BITS)
        # at bit (k % (32 // BITS)) * BITS. The mask drops the sign bits an arithmetic shift copies in.
        codes_per_word: tl.constexpr = 32 // BITS
        words = tl.load(qweight_ptr + (cols[None, :] // codes_per_word) * out_features + rows[:, None],
                        mask=mask, other=0)
        codes = (words >> ((cols[None, :] % codes_per_word) * BITS)) & ((1 << BITS) - 1)

        n_groups = in_features // GROUP_SIZE
        group = rows[:, None] * n_groups + cols[None, :] // GROUP_SIZE
        zero = tl.load(zero_ptr + group, mask=mask, other=0.0)
        # Round to the storage dtype after each op, as torch does for (code - z) * s in bf16 / fp16.
        weight = (codes.to(tl.float32) - zero.to(tl.float32)).to(zero.dtype)
        if APPLY_SCALE:
            scale = tl.load(scale_ptr + group, mask=mask, other=0.0)
            weight = (weight.to(tl.float32) * scale.to(tl.float32)).to(zero.dtype)
        tl.store(weight_ptr + rows[:, None] * in_features + cols[None, :], weight, mask=mask)


class FrozenCodesLinear(torch.autograd.Function):
    """y = F.linear(x, (code - z) * s, bias) with the dense weight recomputed in backward.
    Gradients for x and for s (PEQA, E2E-QP); z, the codes and the bias are frozen here."""

    @staticmethod
    def forward(ctx, x, qweight, scale, zero_point, bias, bits, group_size):
        weight = dequantize(qweight, scale, zero_point, bits, group_size)
        ctx.save_for_backward(x if ctx.needs_input_grad[2] else None, qweight, scale, zero_point)
        ctx.bits, ctx.group_size = bits, group_size
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_y):
        x, qweight, scale, zero_point = ctx.saved_tensors
        bits, group_size = ctx.bits, ctx.group_size
        grad_y_2d = grad_y.reshape(-1, grad_y.shape[-1])
        grad_x = grad_scale = None
        if ctx.needs_input_grad[0]:
            weight = dequantize(qweight, scale, zero_point, bits, group_size)
            grad_x = grad_y_2d.mm(weight).reshape(*grad_y.shape[:-1], weight.shape[1])
        if ctx.needs_input_grad[2]:
            # The same ops as torch's autograd through (code - z) * s and the per-group expand.
            grad_weight = grad_y_2d.t().mm(x.reshape(-1, x.shape[-1]))
            code_minus_zero = dequantize(qweight, scale, zero_point, bits, group_size, apply_scale=False)
            out_features, n_groups = scale.shape
            grad_scale = (grad_weight * code_minus_zero).reshape(out_features, n_groups, group_size).sum(-1)
        return grad_x, None, grad_scale, None, None, None, None
