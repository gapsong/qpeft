"""Core config primitives. ~ peft/utils/peft_types.py + peft/config.py"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QuantTuningType(str, Enum):          # ~ peft.utils.PeftType
    EFFICIENT_QAT = "EFFICIENT_QAT"
    QA_LORA = "QA_LORA"
    PEQA = "PEQA"


class TrainableParams(str, Enum):          # ~ torchtune TrainableParams(FULL/LORA/FROZEN)
    WEIGHT = "weight"
    SCALE = "scale"
    ZERO_POINT = "zero_point"
    ADAPTER = "adapter"


@dataclass
class QuantTuningConfig:                    # ~ peft.PeftConfig (base for every method config)
    quant_tuning_type: QuantTuningType
    bits: int = 4
    group_size: int = 64
    qat_scheme: str = "int_uniform"         # names the (fake_quant, fuse) CONTRACT (unsloth-style)
    backend: str = "auto"                   # names the IMPLEMENTATION: "torchao_cuda" | "mlx" | "auto"
    target_modules: Optional[list[str]] = None
    task_type: Optional[str] = None
    init_weights: str = "rtn"               # ~ peft init_lora_weights: "rtn"|"loftq"|"lqlora"|"apiq"
    trainable_params: tuple[TrainableParams, ...] = (TrainableParams.SCALE,)
