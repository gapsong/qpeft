"""Pure-torch backend of the int_uniform scheme. Always available."""
from __future__ import annotations

import torch

from .base import _IntUniformScheme, _ste


class ReferenceIntUniformScheme(_IntUniformScheme):
    """int_uniform in plain torch (formulas in _IntUniformScheme), with straight-through
    rounding for training. Rounding order as in the official EfficientQAT: round(w / s) + z."""

    @staticmethod
    def _expand(p, in_features):
        """(out, n_groups) -> (out, in_features): repeat each group's value for its inputs."""
        return p.repeat_interleave(in_features // p.shape[-1], dim=-1)

    def init_qparams(self, weight, group_size: int):
        """RTN init: per-group scale and zero-point from the group's min and max."""
        out_features, in_features = weight.shape
        groups = weight.detach().reshape(out_features, in_features // group_size, group_size)
        w_min, w_max = groups.min(dim=-1).values, groups.max(dim=-1).values
        s = self.clamp_scale((w_max - w_min) / self.qmax)
        z = self.round_zero_point(-w_min / s)           # so that code 0 lands on w_min
        return s, z

    def fake_quant(self, w, s, z):
        codes, s_full, z_full = self._codes_with_ste(w, s, z)
        return (codes - z_full) * s_full

    def quantize(self, w, s, z):
        with torch.no_grad():
            codes, _, _ = self._codes_with_ste(w, s, z)
        return codes.round().to(torch.int32)

    def dequant(self, wq, s, z):
        # z is not rounded here: after a QA-LoRA fold the stored zero-point is fractional.
        s = self.clamp_scale(s)
        s_full, z_full = self._expand(s, wq.shape[-1]), self._expand(z, wq.shape[-1])
        return (wq.to(s_full.dtype) - z_full) * s_full

    def _codes_with_ste(self, w, s, z):
        s, z = self.clamp_scale(s), self.round_zero_point(z)
        s_full, z_full = self._expand(s, w.shape[-1]), self._expand(z, w.shape[-1])
        q = w / s_full
        q = _ste(q.round(), q) + z_full
        return torch.clamp(q, self.qmin, self.qmax), s_full, z_full
