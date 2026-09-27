"""The tuner for this method (peft: LoraModel)."""
from __future__ import annotations

from ..efficient_qat.layer import dispatch_default
from ..tuners_utils import BaseQuantTuner


class PEQAModel(BaseQuantTuner):
    def _create_new_module(self, target):
        return dispatch_default(target, self.config)
