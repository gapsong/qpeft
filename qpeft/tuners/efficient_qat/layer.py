"""~ peft/tuners/lora/layer.py: the default dispatch for this method."""
from __future__ import annotations

from torch import nn

from ...schemes import build_scheme
from ..tuners_utils import QuantLinear


def dispatch_default(target: nn.Linear, config):    # ~ peft dispatch_default
    return QuantLinear(target, build_scheme(config), config, adapter=None)
