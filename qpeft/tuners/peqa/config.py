"""PEQA config: the integer codes and zero-points are frozen, only the scales train."""
from __future__ import annotations

from dataclasses import dataclass

from ...config import QuantTuningConfig, QuantTuningType, TrainableParams


@dataclass
class PEQAConfig(QuantTuningConfig):
    """PEQA (Kim et al., NeurIPS 2023, arXiv 2305.14152), Eq. 2:
        w_hat = (s0 + delta_s) * (clamp(round(w0 / s0) + z0, 0, 2**b - 1) - z0)
    RTN init, then only s0 trains; codes and z0 stay frozen. There is no official code,
    so the paper's equations are the reference (tests/test_peqa.py).
    One difference: qpeft clamps the scale to [1e-4, 1e4] like every int_uniform scheme;
    the paper does not clamp, so the two differ only once a scale falls below 1e-4.

    The paper's main results are per-channel; qpeft is group-wise only, so the default is
    the paper's grouped setting for LLaMA2 (Table 13: 4 bits, group size 256)."""
    quant_tuning_type: QuantTuningType = QuantTuningType.PEQA
    bits: int = 4
    group_size: int = 256

    def __post_init__(self):
        self.trainable_params = (TrainableParams.SCALE,)
