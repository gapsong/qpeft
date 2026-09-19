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
from typing import Callable

import torch

from .config import QuantTuningConfig, TrainableParams


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


class ReferenceIntUniformScheme(QuantScheme):
    """Dependency-free reference implementation of the int_uniform contract.

    Weight-only, group-wise *asymmetric* affine quantization with a straight-
    through estimator for training. It exists so the (fake_quant, merge) pair is
    real and testable in pure PyTorch; the torchao/MLX backends are separate
    implementations of the SAME contract, measured at the SAME gate.

        code  = clamp(round(w / s + z), 0, 2**bits - 1)     # quantize (int artifact)
        w_hat = (code - z) * s                              # dequant
        fake_quant = w_hat, with STE gradient to {w, s, z}  # training surrogate

    Grouping is along the input dimension: `s` and `z` hold one entry per group of
    `group_size` input channels. `merge` keeps the integer codes and folds a
    group-structured adapter into the (float) zero-points -- int in, int out, the
    deliberate inverse of peft's dequantizing merge."""

    def __init__(self, fq: FakeQuantizeConfig, *, name: str = "int_uniform",
                 backend: str = "torch", supports_adapter: bool = True):
        super().__init__(fq, name=name, backend=backend, supports_adapter=supports_adapter)
        digits = "".join(c for c in str(fq.dtype) if c.isdigit())
        self.bits = int(digits) if digits else 4

    @property
    def qmin(self) -> int:
        return 0

    @property
    def qmax(self) -> int:
        return (1 << self.bits) - 1

    @staticmethod
    def _expand(p, in_features):
        # (out, n_groups) -> (out, in_features): one scale/zero-point per group
        return p.repeat_interleave(in_features // p.shape[-1], dim=-1)

    def init_qparams(self, weight, group_size: int):
        """RTN init: per-group scale and zero-point from the weight's min/max."""
        out, in_f = weight.shape
        w = weight.detach().reshape(out, in_f // group_size, group_size)
        wmin, wmax = w.min(dim=-1).values, w.max(dim=-1).values
        s = (wmax - wmin).clamp_min(1e-8) / self.qmax
        z = -wmin / s                                   # so (0 - z) * s == wmin
        return s, z

    def _codes_ste(self, w, s, z):
        se, ze = self._expand(s, w.shape[-1]), self._expand(z, w.shape[-1])
        q = w / se + ze
        q = q + (q.round() - q).detach()                # straight-through round
        return torch.clamp(q, self.qmin, self.qmax), se, ze

    def fake_quant(self, w, s, z):                      # STE surrogate used in TRAINING
        code, se, ze = self._codes_ste(w, s, z)
        return (code - ze) * se

    def quantize(self, w, s, z):                        # -> integer artifact (the codes)
        with torch.no_grad():
            code, _, _ = self._codes_ste(w, s, z)
        return code.round().to(torch.int32)

    def dequant(self, wq, s, z):                        # w_hat = (code - z) * s, grouped
        se, ze = self._expand(s, wq.shape[-1]), self._expand(z, wq.shape[-1])
        return (wq.to(se.dtype) - ze) * se

    def merge(self, wq, s, z, adapter=None):
        """int in, int out. No adapter -> identity. A group-structured adapter
        (QA-LoRA) folds EXACTLY into the zero-points; codes and scale are untouched,
        so the merged artifact stays quantized."""
        if adapter is None:
            return wq, s, z
        delta = adapter.folded_delta()                  # (out, n_groups) per-group weight shift
        z_new = z - delta / s                           # (code - z_new)*s == (code - z)*s + delta
        return wq, s, z_new


_SCHEMES: dict[str, Callable[[FakeQuantizeConfig, str], QuantScheme]] = {}


def register_scheme(name: str):             # ~ peft's peft_type -> tuner registry
    def deco(fn):
        _SCHEMES[name] = fn
        return fn
    return deco


def build_scheme(cfg: QuantTuningConfig) -> QuantScheme:
    fq = FakeQuantizeConfig(dtype=f"int{cfg.bits}", group_size=cfg.group_size)
    scheme = _SCHEMES[cfg.qat_scheme](fq, cfg.backend)
    scheme.assert_supported(cfg)            # refuse rather than approximate
    return scheme


@register_scheme("int_uniform")
def _int_uniform(fq: FakeQuantizeConfig, backend: str) -> QuantScheme:
    if backend in ("auto", "torch", "default", "reference"):
        return ReferenceIntUniformScheme(fq, backend="torch", supports_adapter=True)
    if backend in ("torchao", "torchao_cpu", "torchao_cuda"):
        try:
            from .schemes_torchao import TorchaoIntUniformScheme
        except ImportError as e:                    # torchao is an optional dependency
            raise NotImplementedError(
                f"backend {backend!r} needs torchao: pip install 'qpeft[torchao]'.") from e
        return TorchaoIntUniformScheme(fq, backend=backend, supports_adapter=True)
    # mlx and any other backend implement the same contract but are not built yet;
    # refuse rather than silently fall back.
    raise NotImplementedError(
        f"int_uniform backend {backend!r} is not built yet. Use backend='auto' "
        f"(pure-torch) or backend='torchao'; every backend implements the same "
        f"(fake_quant, merge) contract and is measured at the same gate.")
