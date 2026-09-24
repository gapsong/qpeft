"""~ peft/tuners/lora/layer.py: the QA-LoRA adapter and its default dispatch."""
from __future__ import annotations

import torch
from torch import nn

from ...quant_schemes import build_scheme
from ..tuners_utils import QuantLinear


class ZeroPointFoldLoRA(nn.Module):          # group-structured LoRA that folds into z on merge
    """LoRA whose input is average-pooled per quantization group, so its
    weight-delta is constant within each group. That is exactly the structure a
    per-group zero-point can absorb, which is what makes the QA-LoRA merge exact
    and int. Average (not sum) pooling as in the official QA-LoRA
    (https://github.com/yuhuixu1993/qa-lora, peft_utils.py)
    (`nn.AvgPool1d(group_size)`), so the adapter's scale does not grow with
    group_size."""

    def __init__(self, in_features, out_features, r, alpha, group_size, *, device=None):
        super().__init__()
        self.group_size = group_size
        self.n_groups = in_features // group_size
        # A over GROUPS (not raw inputs); B = 0 so the initial delta is zero.
        # fp32 masters on the base layer's device, as peft's autocast_adapter_dtype:
        # a bf16 adapter would not move at a typical LoRA learning rate.
        self.A = nn.Parameter(torch.randn(r, self.n_groups, device=device) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r, device=device))
        self.scaling = alpha / r

    def forward(self, x):
        *lead, in_f = x.shape
        xg = x.reshape(*lead, self.n_groups, self.group_size).mean(-1)  # avg-pool per group
        return (xg @ self.A.to(x.dtype).t() @ self.B.to(x.dtype).t()) * self.scaling

    def folded_delta(self):
        """Per-group weight shift M, shape (out_features, n_groups). The effective
        full weight-delta is D[o, i] = M[o, group(i)] -- constant inside a group.
        The 1/group_size is the average pooling moved from the input onto M."""
        return (self.B @ self.A) * (self.scaling / self.group_size)


def dispatch_default(target: nn.Linear, config):    # ~ peft dispatch_default
    adapter = ZeroPointFoldLoRA(target.in_features, target.out_features,
                                config.r, config.lora_alpha, config.group_size,
                                device=target.weight.device)
    return QuantLinear(target, build_scheme(config), config, adapter=adapter)
