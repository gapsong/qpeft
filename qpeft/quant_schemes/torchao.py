"""torchao backend for the int_uniform contract.

A second implementation of the SAME (fake_quant, merge) contract as the pure-torch
reference, this time on torchao's affine primitives (`_fake_quantize_affine` for
training, `quantize_affine`/`dequantize_affine` for export, `choose_qparams_affine`
for RTN init). It is measured at the SAME gate (`check_merge_equivalence`).

torchao's affine quant uses an INTEGER zero-point domain (q = round(w/s) + z with z
integer, w_hat = (q - z) * s). A continuously *trainable* zero-point is therefore
not expressible here, so this scheme REFUSES configs that put ZERO_POINT in the
trainable set -- refuse rather than approximate. Scale-only training (PEQA, the
EfficientQAT E2E-QP phase) and adapter training (QA-LoRA) are supported.

This module is imported lazily by `build_scheme`, so `import qpeft` never requires
torchao; only `backend="torchao*"` does.
"""
from __future__ import annotations

import torch
from torchao.quantization.quant_primitives import (
    MappingType, _fake_quantize_affine, choose_qparams_affine, dequantize_affine,
    quantize_affine,
)

from ..config import QuantTuningConfig, TrainableParams
from .base import _IntUniformScheme

_QDTYPE = torch.int32


class TorchaoIntUniformScheme(_IntUniformScheme):
    """Group-wise asymmetric affine quantization on torchao primitives.

    Shares bit-width, level range and the adapter fold (`merge`) with the pure-torch
    reference via `_IntUniformScheme`; only the four primitives differ."""

    def _block(self):
        return (1, self.group_size)

    # -- capability: INT zero-point domain cannot train the zero-point ----------
    def supports(self, config: QuantTuningConfig) -> bool:
        # torchao's integer zero-point domain cannot carry a continuously trained
        # zero-point, so refuse those configs (the base assert_supported raises).
        if TrainableParams.ZERO_POINT in config.trainable_params:
            return False
        return super().supports(config)

    # -- init / primitives ------------------------------------------------------
    def init_qparams(self, weight, group_size: int):
        scale, zp = choose_qparams_affine(
            weight, MappingType.ASYMMETRIC, (1, group_size), _QDTYPE, self.qmin, self.qmax)
        # zp is stored as an integer-valued float param
        return self.clamp_scale(scale), self.round_zero_point(zp.to(weight.dtype))

    def fake_quant(self, w, s, z):                 # STE surrogate used in TRAINING
        return _fake_quantize_affine(
            w, self._block(), self.clamp_scale(s), self.round_zero_point(z).to(_QDTYPE), _QDTYPE,
            self.qmin, self.qmax)

    def quantize(self, w, s, z):                   # -> integer artifact (the codes)
        with torch.no_grad():
            return quantize_affine(
                w, self._block(), self.clamp_scale(s), self.round_zero_point(z).to(_QDTYPE), _QDTYPE,
                self.qmin, self.qmax)

    def dequant(self, wq, s, z):                   # (q - z) * s, z may be float after a fold
        return dequantize_affine(
            wq, self._block(), self.clamp_scale(s), z, _QDTYPE, self.qmin, self.qmax, output_dtype=s.dtype)

    # merge (adapter fold) is inherited from _IntUniformScheme -- defined once.
