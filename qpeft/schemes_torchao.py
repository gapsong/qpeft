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

from .config import QuantTuningConfig, TrainableParams
from .schemes import FakeQuantizeConfig, QuantScheme

_QDTYPE = torch.int32


class TorchaoIntUniformScheme(QuantScheme):
    """Group-wise asymmetric affine quantization on torchao primitives."""

    def __init__(self, fq: FakeQuantizeConfig, *, name: str = "int_uniform",
                 backend: str = "torchao", supports_adapter: bool = True):
        super().__init__(fq, name=name, backend=backend, supports_adapter=supports_adapter)
        digits = "".join(c for c in str(fq.dtype) if c.isdigit())
        self.bits = int(digits) if digits else 4
        self.group_size = fq.group_size

    @property
    def qmin(self) -> int:
        return 0

    @property
    def qmax(self) -> int:
        return (1 << self.bits) - 1

    def _block(self):
        return (1, self.group_size)

    # -- capability: INT zero-point domain cannot train the zero-point ----------
    def supports(self, config: QuantTuningConfig) -> bool:
        if TrainableParams.ZERO_POINT in config.trainable_params:
            return False
        return super().supports(config)

    def assert_supported(self, config: QuantTuningConfig) -> None:
        if TrainableParams.ZERO_POINT in config.trainable_params:
            from .schemes import UnsupportedSchemeError
            raise UnsupportedSchemeError(
                f"scheme {self.name!r} on backend {self.backend!r} uses torchao's "
                f"integer zero-point domain, which cannot train the zero-point. "
                f"Drop ZERO_POINT from trainable_params (e.g. use the E2E-QP phase) "
                f"or the pure-torch backend. Refusing rather than approximating.")
        super().assert_supported(config)

    # -- init / primitives ------------------------------------------------------
    def init_qparams(self, weight, group_size: int):
        scale, zp = choose_qparams_affine(
            weight, MappingType.ASYMMETRIC, (1, group_size), _QDTYPE, self.qmin, self.qmax)
        return scale, zp.to(weight.dtype)          # store zp as an (integer-valued) float param

    def fake_quant(self, w, s, z):                 # STE surrogate used in TRAINING
        return _fake_quantize_affine(
            w, self._block(), s, z.round().to(_QDTYPE), _QDTYPE, self.qmin, self.qmax)

    def quantize(self, w, s, z):                   # -> integer artifact (the codes)
        with torch.no_grad():
            return quantize_affine(
                w, self._block(), s, z.round().to(_QDTYPE), _QDTYPE, self.qmin, self.qmax)

    def dequant(self, wq, s, z):                   # (q - z) * s, z may be float after a fold
        return dequantize_affine(
            wq, self._block(), s, z, _QDTYPE, self.qmin, self.qmax, output_dtype=s.dtype)

    def merge(self, wq, s, z, adapter=None):
        """int in, int out. No adapter -> identity. A group-structured adapter folds
        exactly into the zero-points; codes and scale untouched."""
        if adapter is None:
            return wq, s, z
        delta = adapter.folded_delta()             # (out, n_groups) per-group weight shift
        z_new = z - delta / s                       # (q - z_new)*s == (q - z)*s + delta
        return wq, s, z_new
