"""Optimizer param groups per parameter KIND, as in the official EfficientQAT
(quant_parameters / weight_parameters with separate learning rates)."""
from __future__ import annotations

from typing import Iterable

from torch import nn

from .tuners.tuners_utils import QuantLinear


def param_groups(model: nn.Module, *, weight_lr: float, quant_lr: float,
                 adapter_lr: float, weight_decay: float = 0.0,
                 extra_weights: Iterable[nn.Parameter] = ()) -> list[dict]:
    """One optimizer group per kind among the TRAINABLE parameters:
    weight -> weight_lr, scale/zero_point -> quant_lr, adapter -> adapter_lr.

    A trainable parameter outside any QuantLinear is refused: qpeft only knows
    how to train its own axes, so anything else is a configuration mistake.
    `extra_weights` is the one explicit exception: fp parameters outside
    QuantLinear that join the weight group (Block-AP's norm weights)."""
    kinds = {"weight": [], "quant": [], "adapter": []}
    seen = set()
    for layer in (m for m in model.modules() if isinstance(m, QuantLinear)):
        if layer.weight is not None and layer.weight.requires_grad:
            kinds["weight"].append(layer.weight)
        for p in (layer.scale, layer.zero_point):
            if p.requires_grad:
                kinds["quant"].append(p)
        if layer.adapter is not None:
            kinds["adapter"] += [p for p in layer.adapter.parameters() if p.requires_grad]
        seen.update(id(p) for p in layer.parameters())
    for p in extra_weights:
        if p.requires_grad and id(p) not in seen:
            kinds["weight"].append(p)
        seen.add(id(p))

    stray = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in seen]
    if stray:
        raise ValueError(f"trainable parameters outside QuantLinear: {stray[:5]}... "
                         f"qpeft trains only weight/scale/zero_point/adapter. Freeze them first.")

    lrs = {"weight": weight_lr, "quant": quant_lr, "adapter": adapter_lr}
    groups = [{"params": ps, "lr": lrs[k], "weight_decay": weight_decay, "name": k}
              for k, ps in kinds.items() if ps]
    if not groups:
        raise ValueError("nothing is trainable")
    return groups
