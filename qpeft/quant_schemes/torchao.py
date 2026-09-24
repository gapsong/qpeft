"""torchao backend of the int_uniform scheme.

The same scheme as the pure-torch reference, on torchao's affine primitives:
_fake_quantize_affine for training, quantize_affine / dequantize_affine for export,
choose_qparams_affine for the RTN init.

torchao stores the zero-point as an integer, so a zero-point that trains continuously
cannot be expressed: configs with ZERO_POINT in the trainable set are refused.
Scale-only training (EfficientQAT E2E-QP) and adapter training (QA-LoRA) work.

Imported only when backend="torchao" is asked for, so `import qpeft` does not need torchao.
"""
from __future__ import annotations

import torch
from torchao.quantization.quant_primitives import (
    MappingType, _fake_quantize_affine, choose_qparams_affine, dequantize_affine,
    quantize_affine,
)

from ..config import QuantTuningConfig, TrainableParams
from .base import _IntUniformScheme

CODE_DTYPE = torch.int32


class TorchaoIntUniformScheme(_IntUniformScheme):

    def supports(self, config: QuantTuningConfig) -> bool:
        if TrainableParams.ZERO_POINT in config.trainable_params:
            return False
        return super().supports(config)

    def init_qparams(self, weight, group_size: int):
        scale, zero_point = choose_qparams_affine(
            weight, MappingType.ASYMMETRIC, (1, group_size), CODE_DTYPE, self.qmin, self.qmax)
        return self.clamp_scale(scale), self.round_zero_point(zero_point.to(weight.dtype))

    def fake_quant(self, w, s, z):
        return _fake_quantize_affine(
            w, self._block_size(), self.clamp_scale(s), self.round_zero_point(z).to(CODE_DTYPE), CODE_DTYPE,
            self.qmin, self.qmax)

    def quantize(self, w, s, z):
        with torch.no_grad():
            return quantize_affine(
                w, self._block_size(), self.clamp_scale(s), self.round_zero_point(z).to(CODE_DTYPE), CODE_DTYPE,
                self.qmin, self.qmax)

    def dequant(self, wq, s, z):
        # z is not rounded here: after a QA-LoRA fold the stored zero-point is fractional.
        return dequantize_affine(
            wq, self._block_size(), self.clamp_scale(s), z, CODE_DTYPE, self.qmin, self.qmax, output_dtype=s.dtype)

    def _block_size(self):
        """torchao's name for the group: one row, group_size inputs."""
        return (1, self.group_size)
