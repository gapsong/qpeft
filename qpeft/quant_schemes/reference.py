"""Pure-torch reference backend for the int_uniform contract. Always available."""
from __future__ import annotations

import torch

from .base import _IntUniformScheme, _ste


class ReferenceIntUniformScheme(_IntUniformScheme):
    """Dependency-free reference implementation of the int_uniform contract.

    Weight-only, group-wise *asymmetric* affine quantization with a straight-
    through estimator for training. The torchao/MLX backends are separate
    implementations of the SAME contract, measured at the SAME gate.

        code  = clamp(round(w / s + z), 0, 2**bits - 1)     # quantize (int artifact)
        w_hat = (code - z) * s                              # dequant
        fake_quant = w_hat, with STE gradient to {w, s, z}  # training surrogate

    s is clamped to [SCALE_MIN, SCALE_MAX] and z is an integer in [0, 2**bits - 1]
    (both straight-through), as in the official EfficientQAT quantizer."""

    @staticmethod
    def _expand(p, in_features):
        # (out, n_groups) -> (out, in_features): one scale/zero-point per group
        return p.repeat_interleave(in_features // p.shape[-1], dim=-1)

    def init_qparams(self, weight, group_size: int):
        """RTN init: per-group scale and zero-point from the weight's min/max."""
        out, in_f = weight.shape
        w = weight.detach().reshape(out, in_f // group_size, group_size)
        wmin, wmax = w.min(dim=-1).values, w.max(dim=-1).values
        s = self.clamp_scale((wmax - wmin) / self.qmax)
        z = self.round_zero_point(-wmin / s)            # (0 - z) * s ~= wmin, z integer
        return s, z

    def _codes_ste(self, w, s, z):
        s, z = self.clamp_scale(s), self.round_zero_point(z)
        se, ze = self._expand(s, w.shape[-1]), self._expand(z, w.shape[-1])
        q = w / se + ze
        q = _ste(q.round(), q)                          # straight-through round
        return torch.clamp(q, self.qmin, self.qmax), se, ze

    def fake_quant(self, w, s, z):                      # STE surrogate used in TRAINING
        code, se, ze = self._codes_ste(w, s, z)
        return (code - ze) * se

    def quantize(self, w, s, z):                        # -> integer artifact (the codes)
        with torch.no_grad():
            code, _, _ = self._codes_ste(w, s, z)
        return code.round().to(torch.int32)

    def dequant(self, wq, s, z):                        # w_hat = (code - z) * s, grouped
        # z is NOT rounded here: after a QA-LoRA fold the stored zero-point is fractional.
        s = self.clamp_scale(s)
        se, ze = self._expand(s, wq.shape[-1]), self._expand(z, wq.shape[-1])
        return (wq.to(se.dtype) - ze) * se
