"""The tuner for this method (peft: LoraModel)."""
from __future__ import annotations

from ..tuners_utils import BaseQuantTuner
from .layer import dispatch_default


class EfficientQATModel(BaseQuantTuner):
    def _create_new_module(self, target):
        return dispatch_default(target, self.config)
