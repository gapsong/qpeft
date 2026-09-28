"""Merge checks: the merged (integer) model must compute what the trained model computes.
This is the property the whole library is built around."""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from .quant_schemes import QuantScheme
from .tuners.tuners_utils import QuantLinear


class MergeMismatchError(RuntimeError):
    """The merged (integer) path does not compute what the training path computes.
    A raise, not an assert: `python -O` would strip an assert and let a wrong artifact through."""


@torch.no_grad()
def check_merge_equivalence(scheme: QuantScheme, w, s, z, adapter, x, atol: float = 1e-4,
                            codes=None):
    """Scheme-level check: fake_quant (+ adapter) must equal dequant(merge(...)) on input x.
    Run it before trusting a scheme.

    `codes`: frozen integer codes (when the weight does not train). The training path is
    then dequant(codes, s, z), and the merge starts from these codes."""
    if codes is None:
        train_out = F.linear(x, scheme.fake_quant(w, s, z))
        codes = scheme.quantize(w, s, z)
    else:
        train_out = F.linear(x, scheme.dequant(codes, s, z))
    if adapter is not None:
        train_out = train_out + adapter(x)
    merged_codes, merged_s, merged_z = scheme.merge(codes, s, z, adapter)
    merged_out = F.linear(x, scheme.dequant(merged_codes, merged_s, merged_z))
    max_err = (train_out - merged_out).abs().max().item()
    if not torch.allclose(train_out, merged_out, atol=atol):
        raise MergeMismatchError(f"merge != fake_quant, max|delta|={max_err}")
    return max_err


@torch.no_grad()
def check_layer_merge_equivalence(layer, x=None) -> float:
    """Layer-level check: the layer's own forward must equal the forward of a merged copy.
    Unlike check_merge_equivalence, this uses what the layer really computes with,
    including codes frozen at a phase switch while the scale kept training."""
    if layer.merged:
        return 0.0
    # eval: dropout would make the training forward random. The mode is restored below.
    was_training = layer.training
    layer.eval()
    try:
        return _compare_with_merged_copy(layer, x)
    finally:
        layer.train(was_training)


def _compare_with_merged_copy(layer, x):
    if x is None:
        in_features = layer.scale.shape[-1] * layer.config.group_size
        x = torch.randn(4, in_features, dtype=layer.compute_dtype, device=layer.scale.device)
    train_out = layer(x)
    merged = copy.deepcopy(layer)
    merged.merge()
    merged_out = merged(x)
    max_err = (train_out - merged_out).abs().max().item()
    tol = _merge_tolerance(layer.compute_dtype, train_out)
    if math.isnan(max_err) or max_err > tol:
        raise MergeMismatchError(f"merge != training forward, max|delta|={max_err} > {tol}")
    return max_err


def verify_quant_model(model) -> dict:
    """check_layer_merge_equivalence on every unmerged QuantLinear.
    Raises on the first failing layer; otherwise returns {name: max_err}."""
    base = getattr(model, "base", model)
    results = {name: check_layer_merge_equivalence(m)
               for name, m in base.named_modules() if isinstance(m, QuantLinear)}
    if not results:
        raise ValueError("no QuantLinear found -- nothing to verify")
    return results


def _merge_tolerance(dtype: torch.dtype, reference: torch.Tensor) -> float:
    """fp32: the fold is exact up to float reordering. Half precision: the fold z - delta / s
    is itself computed in half, so allow a relative 1e-2."""
    if dtype in (torch.float32, torch.float64):
        return 1e-4
    return 1e-2 * max(reference.abs().max().item(), 1.0)
