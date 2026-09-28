"""The QA-LoRA adapter and the function that builds a QA-LoRA layer (peft: lora/layer.py)."""
from __future__ import annotations

import torch
from torch import nn

from ...quant_schemes import build_scheme
from ..tuners_utils import QuantLinear


class ZeroPointFoldLoRA(nn.Module):
    """LoRA on the per-group average of the input.

    Because A sees one value per quantization group, the weight change B @ A is the same
    for every input of a group. A per-group zero-point can absorb exactly that, which is
    why the QA-LoRA merge is exact and stays integer.
    Average (not sum) pooling and dropout on the pooled input, as in the official QA-LoRA:
    lora_B(lora_A(lora_dropout(qa_pool(x)))) (https://github.com/yuhuixu1993/qa-lora,
    peft_utils.py). Dropout is active only in training, so the merge is unaffected."""

    def __init__(self, in_features, out_features, r, alpha, group_size, *, dropout=0.0, device=None):
        super().__init__()
        self.group_size = group_size
        self.n_groups = in_features // group_size
        self.dropout = nn.Dropout(dropout)
        # fp32 on the base layer's device, as peft's autocast_adapter_dtype:
        # a bf16 adapter would not move at a typical LoRA learning rate.
        # B = 0, so the adapter starts as a no-op.
        self.A = nn.Parameter(torch.randn(r, self.n_groups, device=device) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r, device=device))
        self.scaling = alpha / r

    def forward(self, x):
        *lead, _ = x.shape
        group_means = self.dropout(x.reshape(*lead, self.n_groups, self.group_size).mean(-1))
        return (group_means @ self.A.to(x.dtype).t() @ self.B.to(x.dtype).t()) * self.scaling

    def folded_delta(self):
        """The weight change per group, (out_features, n_groups): every input i of group g
        gets delta[:, g]. The 1 / group_size is the averaging, moved from x onto the weight."""
        return (self.B @ self.A) * (self.scaling / self.group_size)


def dispatch_default(target: nn.Linear, config):
    adapter = ZeroPointFoldLoRA(target.in_features, target.out_features,
                                config.r, config.lora_alpha, config.group_size,
                                dropout=config.lora_dropout, device=target.weight.device)
    return QuantLinear(target, build_scheme(config), config, adapter=adapter)
