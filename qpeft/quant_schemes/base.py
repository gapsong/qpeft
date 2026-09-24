"""Quantization schemes: the (fake_quant, fuse) CONTRACT.

This module is where the central lesson lives, encoded rather than documented:

  - fake_quant (training) and merge (export) are methods on ONE object, so they
    cannot drift into separate, silently-disagreeing code paths.
  - A scheme must be able to REFUSE a config whose fuse cannot match its
    fake_quant (assert_supported / UnsupportedSchemeError) -- a fake-quant that
    does not match fuse is worse than none.
  - The backend (CUDA/torchao, MLX, ...) is an IMPLEMENTATION of the contract,
    selected separately from the contract name, so the primitive is not welded
    to one framework or accelerator.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import QuantTuningConfig, TrainableParams


# Bounds on the effective scale, as in the official EfficientQAT quantizer
# (`clamp_ste(self.scale, 1e-4, 1e4)`). A trainable scale can otherwise step to
# zero or below, where (code - z) * s is no longer a quantization grid.
SCALE_MIN, SCALE_MAX = 1e-4, 1e4


def _ste(value, x):
    """Straight-through estimator: forward is EXACTLY `value`, gradient is that
    of `x`. (`x + (value - x).detach()` is not exact in floating point.)"""
    return value.detach() + (x - x.detach())


class UnsupportedSchemeError(RuntimeError):
    """Raised when a scheme cannot guarantee fake_quant == fuse for a config.
    Refusing here is the whole point: never approximate a mismatched fuse."""


@dataclass
class FakeQuantizeConfig:                   # ~ torchao / torchtune FakeQuantizeConfig
    dtype: str = "int4"
    group_size: int = 64
    is_symmetric: bool = False


class QuantScheme:
    """The (fake_quant, fuse) contract for ONE quantized representation on ONE backend.

    Subclasses implement the four primitives AND declare, via supports(), for
    which configs their merge provably matches their fake_quant. `merge` takes an
    integer artifact and returns an integer artifact -- it never dequantizes."""

    def __init__(self, fq: FakeQuantizeConfig, *, name: str, backend: str,
                 supports_adapter: bool = False):
        self.fq = fq
        self.name = name
        self.backend = backend
        self.supports_adapter = supports_adapter

    # -- capability / refusal ------------------------------------------------
    def supports(self, config: QuantTuningConfig) -> bool:
        """Can this scheme's fuse match its fake_quant for `config`? Subclasses
        narrow this (bit-widths, group sizes, adapter, activation quant, ...)."""
        needs_adapter = TrainableParams.ADAPTER in config.trainable_params
        return self.supports_adapter or not needs_adapter

    def assert_supported(self, config: QuantTuningConfig) -> None:
        if not self.supports(config):
            raise UnsupportedSchemeError(
                f"scheme {self.name!r} on backend {self.backend!r} cannot guarantee "
                f"fake_quant == fuse for this config (bits={config.bits}, "
                f"group_size={config.group_size}, trainable={[p.value for p in config.trainable_params]}). "
                f"Refusing rather than approximating.")

    # -- the (fake_quant, fuse) pair -----------------------------------------
    def fake_quant(self, w, s, z):          # STE surrogate used during TRAINING
        raise NotImplementedError("wrap the backend FakeQuantizer (STE round)")

    def quantize(self, w, s, z):            # -> packed int artifact (e.g. AffineQuantizedTensor)
        raise NotImplementedError("build the backend's quantized tensor")

    def dequant(self, wq, s, z):            # w_hat = (w_q - z) * s
        return (wq.to(s.dtype) - z) * s

    def merge(self, wq, s, z, adapter=None):
        """Fold the adapter into (wq, s, z) and return an artifact that is STILL
        integer. The one op qpeft owns; must match fake_quant (see check_merge_equivalence)."""
        raise NotImplementedError("fold adapter into zero-points; stay quantized")


class _IntUniformScheme(QuantScheme):
    """Shared int_uniform machinery: bit-width, level range, and the adapter fold
    (`merge`). Backends subclass this and implement fake_quant/quantize/dequant/
    init_qparams. The fold lives here ONCE so it cannot drift between backends --
    the very divergence this project exists to prevent."""

    def __init__(self, fq: FakeQuantizeConfig, *, name: str = "int_uniform",
                 backend: str = "torch", supports_adapter: bool = True):
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

    @staticmethod
    def clamp_scale(s):
        """The effective scale: s clamped to [SCALE_MIN, SCALE_MAX] with a
        straight-through gradient. Every primitive and the merge go through it,
        so training and the exported artifact use the same scale."""
        return _ste(s.clamp(SCALE_MIN, SCALE_MAX), s)

    def round_zero_point(self, z):
        """The effective zero-point: rounded to an integer in [qmin, qmax], with a
        straight-through gradient (official EfficientQAT:
        clamp_ste(round_ste(z), qmin, qmax)). int_uniform has an INTEGER
        zero-point domain, so the artifact stays GPTQ-style packable; only an
        adapter folded in by `merge` makes the stored zero-point fractional."""
        return _ste(z.round().clamp(self.qmin, self.qmax), z)

    def merge(self, wq, s, z, adapter=None):
        """int in, int out. No adapter -> identity. A group-structured adapter
        (QA-LoRA) folds EXACTLY into the zero-points; codes and scale are untouched,
        so the merged artifact stays quantized. Defined once for every backend."""
        s = self.clamp_scale(s)                         # the scale training actually used
        z = self.round_zero_point(z)                    # the zero-point training actually used
        if adapter is None:
            return wq, s, z
        delta = adapter.folded_delta().to(s.dtype)      # (out, n_groups) per-group weight shift
        return wq, s, z - delta / s                     # (code - z')*s == (code - z)*s + delta
