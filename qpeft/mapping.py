"""Which config and which tuner belong to each method, and get_quant_model (peft: mapping.py)."""
from __future__ import annotations

from torch import nn

from .config import QuantTuningConfig, QuantTuningType
from .tuners.efficient_qat import EfficientQATConfig, EfficientQATModel
from .tuners.peqa import PEQAConfig, PEQAModel
from .tuners.qa_lora import QALoraConfig, QALoraModel

QUANT_TUNING_TYPE_TO_CONFIG_MAPPING = {       # used to load a saved config
    QuantTuningType.EFFICIENT_QAT: EfficientQATConfig,
    QuantTuningType.QA_LORA: QALoraConfig,
    QuantTuningType.PEQA: PEQAConfig,
}
QUANT_TUNING_TYPE_TO_TUNER_MAPPING = {        # used to swap in QuantLinear
    QuantTuningType.EFFICIENT_QAT: EfficientQATModel,
    QuantTuningType.QA_LORA: QALoraModel,
    QuantTuningType.PEQA: PEQAModel,
}


def get_quant_model(model: nn.Module, config: QuantTuningConfig):
    """Quantize the target linears of `model` for `config` (peft: get_peft_model)."""
    from .peft_model import QuantModel        # imported here: peft_model imports the mappings above
    return QuantModel(model, config)
