"""Optimizer groups with one learning rate per parameter kind, as in the official EfficientQAT
(weight_parameters / quant_parameters)."""
from __future__ import annotations

from typing import Iterable

from torch import nn

from .tuners.tuners_utils import quant_layers


def param_groups(model: nn.Module, *, weight_lr: float, quant_lr: float,
                 adapter_lr: float, weight_decay: float = 0.0,
                 extra_weights: Iterable[nn.Parameter] = ()) -> list[dict]:
    """One optimizer group per kind of trainable parameter:
        weight              -> weight_lr
        scale, zero_point   -> quant_lr
        adapter             -> adapter_lr

    A trainable parameter outside every QuantLinear is refused: it would be a configuration
    mistake. The one exception is `extra_weights`, which join the weight group
    (Block-AP's norm weights)."""
    kinds = {"weight": [], "quant": [], "adapter": []}
    known = set()
    for layer in quant_layers(model):
        if layer.weight is not None and layer.weight.requires_grad:
            kinds["weight"].append(layer.weight)
        for p in (layer.scale, layer.zero_point):
            if p.requires_grad:
                kinds["quant"].append(p)
        if layer.adapter is not None:
            kinds["adapter"] += [p for p in layer.adapter.parameters() if p.requires_grad]
        known.update(id(p) for p in layer.parameters())
    for p in extra_weights:
        if p.requires_grad and id(p) not in known:
            kinds["weight"].append(p)
        known.add(id(p))

    stray = [name for name, p in model.named_parameters() if p.requires_grad and id(p) not in known]
    if stray:
        raise ValueError(f"trainable parameters outside QuantLinear: {stray[:5]}... "
                         f"qpeft trains only weight/scale/zero_point/adapter. Freeze them first.")

    lrs = {"weight": weight_lr, "quant": quant_lr, "adapter": adapter_lr}
    groups = [{"params": params, "lr": lrs[kind], "weight_decay": weight_decay, "name": kind}
              for kind, params in kinds.items() if params]
    if not groups:
        raise ValueError("nothing is trainable")
    return groups
