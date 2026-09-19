"""~ peft/tuners/lora/config.py: LoraConfig field names, adapter folds into zero-points."""
from __future__ import annotations

from dataclasses import dataclass

from ...config import QuantTuningConfig, QuantTuningType, TrainableParams


@dataclass
class QALoraConfig(QuantTuningConfig):
    quant_tuning_type: QuantTuningType = QuantTuningType.QA_LORA
    r: int = 64                                   # exact peft LoRA field names
    lora_alpha: int = 128
    lora_dropout: float = 0.0

    def __post_init__(self):
        self.trainable_params = (TrainableParams.ADAPTER,)
