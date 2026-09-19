"""~ peft/tuners/lora/config.py, but for a QAT method with no adapter."""
from __future__ import annotations

from dataclasses import dataclass

from ...config import QuantTuningConfig, QuantTuningType, TrainableParams


@dataclass
class EfficientQATConfig(QuantTuningConfig):
    quant_tuning_type: QuantTuningType = QuantTuningType.EFFICIENT_QAT
    phase: str = "block_ap"                       # "block_ap" | "e2e_qp"

    def __post_init__(self):
        if self.phase == "block_ap":              # train weights + both quant params (STE)
            self.trainable_params = (
                TrainableParams.WEIGHT, TrainableParams.SCALE, TrainableParams.ZERO_POINT,
            )
        elif self.phase == "e2e_qp":              # freeze int weights, train only step size
            self.trainable_params = (TrainableParams.SCALE,)
        else:
            raise ValueError(f"unknown phase {self.phase!r}")


def efficient_qat_schedule(**kw) -> list[EfficientQATConfig]:
    """EfficientQAT = two PEFT-style configs run in order (Block-AP then E2E-QP).
    This replaces the old single 'Recipe' object."""
    return [EfficientQATConfig(phase="block_ap", **kw),
            EfficientQATConfig(phase="e2e_qp", **kw)]
