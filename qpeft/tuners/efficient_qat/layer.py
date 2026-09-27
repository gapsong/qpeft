"""Builds an EfficientQAT layer: a QuantLinear without adapter (peft: lora/layer.py)."""
from __future__ import annotations

from torch import nn

from ...quant_schemes import build_scheme
from ..tuners_utils import QuantLinear


def dispatch_default(target: nn.Linear, config):
    return QuantLinear(target, build_scheme(config), config, adapter=None)
