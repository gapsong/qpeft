"""~ peft/tuners/lora/layer.py: the QA-LoRA adapter and its default dispatch."""
from __future__ import annotations

import torch
from torch import nn

from ...schemes import build_scheme
from ..tuners_utils import QuantLinear


class ZeroPointFoldLoRA(nn.Module):          # group-structured LoRA that folds into z on merge
    """LoRA whose input is pooled per quantization group, so its weight-delta is
    constant within each group. That is exactly the structure a per-group
    zero-point can absorb, which is what makes the QA-LoRA merge exact and int."""

    def __init__(self, in_features, out_features, r, alpha, group_size):
        super().__init__()
        self.group_size = group_size
        self.n_groups = in_features // group_size
        # A over GROUPS (not raw inputs); B = 0 so the initial delta is zero.
        self.A = nn.Parameter(torch.randn(r, self.n_groups) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r

    def forward(self, x):
        *lead, in_f = x.shape
        xg = x.reshape(*lead, self.n_groups, self.group_size).sum(-1)   # pool per group
        return (xg @ self.A.t() @ self.B.t()) * self.scaling

    def folded_delta(self):
        """Per-group weight shift M, shape (out_features, n_groups). The effective
        full weight-delta is D[o, i] = M[o, group(i)] -- constant inside a group."""
        return (self.B @ self.A) * self.scaling


def dispatch_default(target: nn.Linear, config):    # ~ peft dispatch_default
    adapter = ZeroPointFoldLoRA(target.in_features, target.out_features,
                                config.r, config.lora_alpha, config.group_size)
    return QuantLinear(target, build_scheme(config), config, adapter=adapter)
