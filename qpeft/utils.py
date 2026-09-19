"""Correctness utilities. The merge-equivalence check is the spine of the whole lib."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .schemes import QuantScheme


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
