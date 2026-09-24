"""~ peft/tuners/lora/config.py, but for a QAT method with no adapter."""
from __future__ import annotations

from dataclasses import dataclass

from ...config import QuantTuningConfig, QuantTuningType, TrainableParams


@dataclass
class EfficientQATConfig(QuantTuningConfig):
    """Defaults follow the official EfficientQAT code (OpenGVLab/EfficientQAT,
    main_block_ap.py argparse): wbits=4, group_size=128.

    Training hyperparameters live in the caller's training loop, not in this
    config. Official values for reference:
      Block-AP  quant_lr=1e-4, weight_lr=1e-5 (2e-5 for 2-bit), wd=0,
                epochs=2, train_size=4096, training_seqlen=2048, batch_size=2,
                cosine schedule down to lr / min_lr_factor (=20),
                loss = per-block MSE vs the fp block output
      E2E-QP    lr=1e-5 (2e-5 for 2-bit), 1 epoch, batch 4 x grad-accum 8,
                cosine with warmup_ratio=0.03, max_grad_norm=0.3, only scales
                trainable (QATTrainingArguments carries these defaults)
    """
    quant_tuning_type: QuantTuningType = QuantTuningType.EFFICIENT_QAT
    bits: int = 4                                 # official --wbits default
    group_size: int = 128                         # official --group_size default
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
