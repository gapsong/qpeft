"""~ peft/tuners/lora/model.py: LoraModel."""
from __future__ import annotations

from ..tuners_utils import BaseQuantTuner
from .layer import dispatch_default


class QALoraModel(BaseQuantTuner):
    def _create_new_module(self, target):
        return dispatch_default(target, self.config)
