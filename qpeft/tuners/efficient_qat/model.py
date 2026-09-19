"""~ peft/tuners/lora/model.py: LoraModel."""
from __future__ import annotations

from ..tuners_utils import BaseQuantTuner
from .layer import dispatch_default


class EfficientQATModel(BaseQuantTuner):
    # Block-AP reconstruction loss lives on the training step, not in the config.
    def _create_new_module(self, target):
        return dispatch_default(target, self.config)
