"""~ peft/peft_model.py: the model wrapper returned to the user."""
from __future__ import annotations

from torch import nn

from .config import QuantTuningConfig, QuantTuningType
from .tuners.efficient_qat import EfficientQATModel
from .tuners.qa_lora import QALoraModel
from .tuners.tuners_utils import QuantLinear


class QuantModel(nn.Module):                  # ~ peft PeftModel
    _TUNERS = {
        QuantTuningType.EFFICIENT_QAT: EfficientQATModel,
        QuantTuningType.QA_LORA: QALoraModel,
    }

    def __init__(self, model: nn.Module, config: QuantTuningConfig):
        super().__init__()
        self.base = self._TUNERS[config.quant_tuning_type](model, config).model
        self.config = config

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def merge_and_unload(self):               # ~ peft; here the result STAYS quantized
        for module in self.base.modules():
            if isinstance(module, QuantLinear):
                module.merge()
        return self.base
