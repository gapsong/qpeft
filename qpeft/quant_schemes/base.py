"""The contract every quantization scheme follows.

A scheme owns both halves of quantization in one object:
    fake_quant   the differentiable stand-in used during training
    merge        folds the adapter in and returns the integer artifact used at inference
Keeping both in one class means they cannot drift apart.
If a scheme cannot make merge match fake_quant for a config, it refuses the config
(UnsupportedSchemeError) instead of approximating.

The backend (pure torch, torchao, ...) is an implementation of the same contract,
chosen separately from the scheme name.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import QuantTuningConfig, TrainableParams

# Bounds on the scale, as in the official EfficientQAT quantizer (`clamp_ste(self.scale, 1e-4, 1e4)`).
# Without them a trainable scale can step to zero or below, where (code - z) * s is no grid.
SCALE_MIN, SCALE_MAX = 1e-4, 1e4


def _ste(value, x):
    """Straight-through estimator: the forward gives exactly `value`, the gradient is that of `x`.
    (The usual `x + (value - x).detach()` is not exactly `value` in floating point.)"""
    return value.detach() + (x - x.detach())


class UnsupportedSchemeError(RuntimeError):
    """A scheme cannot guarantee fake_quant == merge for a config, so it refuses it."""


@dataclass
class FakeQuantizeConfig:                   # same name as in torchao / torchtune
    dtype: str = "int4"
    group_size: int = 64
    is_symmetric: bool = False


class QuantScheme:
    """One quantized representation on one backend.

    Subclasses implement fake_quant / quantize / dequant / merge, and narrow `supports`
    to the configs for which merge provably matches fake_quant.
    `merge` takes integer codes and returns integer codes: it never dequantizes."""

    def __init__(self, fq: FakeQuantizeConfig, *, name: str, backend: str,
                 supports_adapter: bool = False):
        self.fq = fq
        self.name = name
        self.backend = backend
        self.supports_adapter = supports_adapter

    def supports(self, config: QuantTuningConfig) -> bool:
        trains_adapter = TrainableParams.ADAPTER in config.trainable_params
        return self.supports_adapter or not trains_adapter

    def assert_supported(self, config: QuantTuningConfig) -> None:
        if not self.supports(config):
            raise UnsupportedSchemeError(
                f"scheme {self.name!r} on backend {self.backend!r} cannot guarantee "
                f"fake_quant == fuse for this config (bits={config.bits}, "
                f"group_size={config.group_size}, trainable={[p.value for p in config.trainable_params]}). "
                f"Refusing rather than approximating.")

    def fake_quant(self, w, s, z):
        """Training: the quantized weight, with straight-through gradients."""
        raise NotImplementedError

    def quantize(self, w, s, z):
        """The integer codes of w."""
        raise NotImplementedError

    def dequant(self, wq, s, z):
        """w_hat = (codes - z) * s."""
        return (wq.to(s.dtype) - z) * s

    def merge(self, wq, s, z, adapter=None):
        """Fold the adapter into (codes, s, z); the result is still integer codes."""
        raise NotImplementedError


class _IntUniformScheme(QuantScheme):
    """Group-wise asymmetric integer quantization, shared by all backends:
        code  = clamp(round(w / s) + z, 0, 2**bits - 1)
        w_hat = (code - z) * s
    with one scale s and one integer zero-point z per group of `group_size` inputs.

    The adapter fold (`merge`) is written here once, so the backends cannot disagree on it.
    Backends implement init_qparams / fake_quant / quantize / dequant."""

    def __init__(self, fq: FakeQuantizeConfig, *, name: str = "int_uniform",
                 backend: str = "torch", supports_adapter: bool = True):
        super().__init__(fq, name=name, backend=backend, supports_adapter=supports_adapter)
        self.bits = int(fq.dtype.removeprefix("int"))
        self.group_size = fq.group_size

    @property
    def qmin(self) -> int:
        return 0

    @property
    def qmax(self) -> int:
        return (1 << self.bits) - 1

    @staticmethod
    def clamp_scale(s):
        """The scale that is really used: s clamped to [SCALE_MIN, SCALE_MAX], straight-through
        gradient. Every primitive and the merge use it, so training and export agree."""
        return _ste(s.clamp(SCALE_MIN, SCALE_MAX), s)

    def round_zero_point(self, z):
        """The zero-point that is really used: rounded to an integer in [qmin, qmax],
        straight-through gradient (official EfficientQAT: clamp_ste(round_ste(z), qmin, qmax)).
        Being an integer keeps the artifact GPTQ-packable; only an adapter fold in `merge`
        makes the stored zero-point fractional."""
        return _ste(z.round().clamp(self.qmin, self.qmax), z)

    def merge(self, wq, s, z, adapter=None):
        """QA-LoRA's adapter is constant inside each group, so it is exactly a per-group shift
        of the weight: delta. Moving the zero-point by -delta / s gives the same weight:
            (code - (z - delta / s)) * s == (code - z) * s + delta
        Codes and scale stay as they are, so the result is still quantized."""
        s = self.clamp_scale(s)
        z = self.round_zero_point(z)
        if adapter is None:
            return wq, s, z
        delta = adapter.folded_delta().to(s.dtype)      # (out, n_groups)
        return wq, s, z - delta / s
