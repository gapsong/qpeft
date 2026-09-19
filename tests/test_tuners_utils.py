"""Injection / target matching. ~ peft/tests/test_tuners_utils.py: which nn.Linear
modules get swapped for a QuantLinear, and the group-size guard.
"""
import pytest
import torch.nn as nn

from qpeft import EfficientQATConfig, get_quant_model
from qpeft.tuners.tuners_utils import QuantLinear


class Attn(nn.Module):
    """A tiny transformer-block-like module with HF-style submodule names."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 128, bias=False)
        self.k_proj = nn.Linear(128, 128, bias=False)
        self.v_proj = nn.Linear(128, 128, bias=False)
        self.o_proj = nn.Linear(128, 128, bias=False)
        self.mlp = nn.Linear(128, 64, bias=False)

    def forward(self, x):
        return self.mlp(self.o_proj(self.q_proj(x)))


def _kinds(model):
    return {n.split(".")[-1]: type(m).__name__
            for n, m in model.named_modules() if isinstance(m, (nn.Linear, QuantLinear))}


def test_target_modules_list_selects_named_linears():
    cfg = EfficientQATConfig(bits=4, group_size=64, target_modules=["q_proj", "v_proj"])
    kinds = _kinds(get_quant_model(Attn(), cfg))
    assert kinds["q_proj"] == "QuantLinear"
    assert kinds["v_proj"] == "QuantLinear"
    assert kinds["k_proj"] == "Linear"          # not targeted
    assert kinds["mlp"] == "Linear"


def test_target_modules_none_targets_all_linears():
    cfg = EfficientQATConfig(bits=4, group_size=64, target_modules=None)
    kinds = _kinds(get_quant_model(Attn(), cfg))
    assert set(kinds.values()) == {"QuantLinear"}


def test_injection_count():
    cfg = EfficientQATConfig(bits=4, group_size=64, target_modules=["proj"])
    model = get_quant_model(Attn(), cfg)
    n_quant = sum(isinstance(m, QuantLinear) for m in model.modules())
    assert n_quant == 4                          # q/k/v/o_proj, not mlp


def test_non_divisible_group_size_raises():
    with pytest.raises(ValueError):
        get_quant_model(nn.Sequential(nn.Linear(100, 64)),
                        EfficientQATConfig(bits=4, group_size=64))
