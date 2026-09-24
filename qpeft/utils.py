"""Correctness utilities. The merge-equivalence check is the spine of the whole lib."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .quant_schemes import QuantScheme


@torch.no_grad()
def check_merge_equivalence(scheme: QuantScheme, w, s, z, adapter, x, atol: float = 1e-4):
    """The spine: the training fake_quant path must equal the merged (still-quantized) path.

    If this fails, the fake_quant does not match the fuse -- and a fake-quant that
    does not match fuse is worse than none. Run this per layer before trusting a scheme."""
    train = F.linear(x, scheme.fake_quant(w, s, z))
    if adapter is not None:
        train = train + adapter(x)
    wq, s2, z2 = scheme.merge(scheme.quantize(w, s, z), s, z, adapter)
    infer = F.linear(x, scheme.dequant(wq, s2, z2))
    max_err = (train - infer).abs().max().item()
    assert torch.allclose(train, infer, atol=atol), f"merge != fake_quant, max|delta|={max_err}"
    return max_err


def _merge_tolerance(dtype: torch.dtype, ref: torch.Tensor) -> float:
    """fp32: the fold is exact up to float reassociation. Half precision: the
    fold z - delta/s itself is computed in half, so allow a relative 1e-2."""
    if dtype == torch.float32 or dtype == torch.float64:
        return 1e-4
    return 1e-2 * max(ref.abs().max().item(), 1.0)


@torch.no_grad()
def check_layer_merge_equivalence(layer, x=None) -> float:
    """Layer-level spine check that also covers FROZEN codes: the layer's own
    training forward must equal the forward of a merged copy of it.

    Unlike check_merge_equivalence (which re-derives codes from w, s, z), this
    uses whatever the layer really computes with -- including codes frozen at a
    phase switch while the scale kept training."""
    import copy
    if layer.merged:
        return 0.0
    in_f = layer.scale.shape[-1] * layer.config.group_size
    if x is None:
        x = torch.randn(4, in_f, dtype=layer.compute_dtype, device=layer.scale.device)
    train = layer(x)
    merged = copy.deepcopy(layer)
    merged.merge()
    infer = merged(x)
    max_err = (train - infer).abs().max().item()
    tol = _merge_tolerance(layer.compute_dtype, train)
    assert max_err <= tol, f"merge != training forward, max|delta|={max_err} > {tol}"
    return max_err


def verify_quant_model(model) -> dict:
    """Run check_layer_merge_equivalence on every unmerged QuantLinear.
    Raises on the first red layer; returns {name: max_err} otherwise."""
    from .tuners.tuners_utils import QuantLinear
    base = getattr(model, "base", model)
    results = {name: check_layer_merge_equivalence(m)
               for name, m in base.named_modules() if isinstance(m, QuantLinear)}
    if not results:
        raise ValueError("no QuantLinear found -- nothing to verify")
    return results
