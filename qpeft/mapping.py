"""~ peft/mapping.py: registries + the get_quant_model entrypoint."""
from __future__ import annotations

from torch import nn

from .config import QuantTuningConfig, QuantTuningType
from .peft_model import QuantModel
from .tuners.efficient_qat import EfficientQATConfig, EfficientQATModel
from .tuners.qa_lora import QALoraConfig, QALoraModel

QUANT_TUNING_TYPE_TO_CONFIG_MAPPING = {       # ~ peft PEFT_TYPE_TO_CONFIG_MAPPING
    QuantTuningType.EFFICIENT_QAT: EfficientQATConfig,
    QuantTuningType.QA_LORA: QALoraConfig,
}
QUANT_TUNING_TYPE_TO_MODEL_MAPPING = {        # ~ peft PEFT_TYPE_TO_TUNER_MAPPING
    QuantTuningType.EFFICIENT_QAT: EfficientQATModel,
    QuantTuningType.QA_LORA: QALoraModel,
}


def get_quant_model(model: nn.Module, config: QuantTuningConfig) -> QuantModel:   # ~ get_peft_model
    return QuantModel(model, config)
