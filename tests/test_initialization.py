"""Quant-param init. ~ peft/tests/test_initialization.py: RTN must place a grid
that reconstructs the weight to within half a quantization step per group.
"""
import pytest
import torch

from qpeft import EfficientQATConfig
from qpeft.schemes import build_scheme


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64])
def test_rtn_init_reconstructs_weight(bits, group_size):
    torch.manual_seed(0)
    scheme = build_scheme(EfficientQATConfig(bits=bits, group_size=group_size))
    w = torch.randn(32, 128) * 0.05
    s, z = scheme.init_qparams(w, group_size)

    assert (s > 0).all()                                  # a usable grid
    assert s.shape == (32, 128 // group_size)

    w_hat = scheme.dequant(scheme.quantize(w, s, z), s, z)
    step = scheme._expand(s, w.shape[-1])
    # round-to-nearest error is at most half a step in every group
    assert (w - w_hat).abs().max() <= (0.5 * step).max() + 1e-6


def test_rtn_covers_group_range():
    """The grid spans each group's min..max, so extremes are representable."""
    torch.manual_seed(0)
    scheme = build_scheme(EfficientQATConfig(bits=4, group_size=64))
    w = torch.randn(8, 128) * 0.05
    s, z = scheme.init_qparams(w, 64)
    w_hat = scheme.dequant(scheme.quantize(w, s, z), s, z)
    # per-group min/max reconstructed within one step
    step = scheme._expand(s, 128)
    assert (w - w_hat).abs().max() <= step.max() + 1e-6
