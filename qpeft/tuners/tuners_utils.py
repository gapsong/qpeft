"""Shared tuner machinery. ~ peft/tuners/tuners_utils.py (BaseTuner, BaseTunerLayer)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from ..config import QuantTuningConfig, TrainableParams
from ..schemes import QuantScheme, UnsupportedSchemeError


class AdapterLayer:                          # ~ peft BaseTunerLayer / torchtune AdapterModule
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None): ...
    def unmerge(self): ...


class QuantLinear(nn.Module, AdapterLayer):  # ~ peft lora.Linear / torchtune QATLoRALinear
    """A base linear held in a quantized representation, with an optional adapter.

    Which of {weight, scale, zero_point, adapter} carry gradients is decided by
    `config.trainable_params` -- that switch is what makes PEQA / EfficientQAT /
    QA-LoRA the same layer under different configs."""

    def __init__(self, base: nn.Linear, scheme: QuantScheme,
                 config: QuantTuningConfig, adapter: Optional[nn.Module] = None):
        super().__init__()
        if base.in_features % config.group_size != 0:
            raise ValueError(f"in_features {base.in_features} not divisible by group_size {config.group_size}")
        self.scheme, self.config, self.adapter = scheme, config, adapter
        self.merged = False
        tp = set(config.trainable_params)
        n_groups = base.in_features // config.group_size
        w = base.weight.data
        # scale/zero_point follow the base weight's dtype and device, so a
        # half-precision or GPU base (a typical HF model) survives injection.
        kw = dict(dtype=w.dtype, device=w.device)
        self.weight = nn.Parameter(w.clone(),
                                   requires_grad=TrainableParams.WEIGHT in tp)
        self.scale = nn.Parameter(torch.ones(base.out_features, n_groups, **kw),
                                  requires_grad=TrainableParams.SCALE in tp)
        self.zero_point = nn.Parameter(torch.zeros(base.out_features, n_groups, **kw),
                                       requires_grad=TrainableParams.ZERO_POINT in tp)
        # The bias is part of the base layer, not quantized: kept in full
        # precision and frozen, so a Linear(bias=True) survives injection unchanged.
        self.bias = (None if base.bias is None
                     else nn.Parameter(base.bias.data.clone(), requires_grad=False))
        self._init_qparams(config)

    def _init_qparams(self, config: QuantTuningConfig):
        """Place the starting quantization grid. RTN is the only init implemented;
        anything else is refused rather than left at the degenerate scale=1 grid."""
        if config.init_weights != "rtn":
            raise UnsupportedSchemeError(
                f"init_weights={config.init_weights!r} is not implemented; only 'rtn'. "
                f"Refusing rather than approximating with a degenerate scale=1 grid.")
        if not hasattr(self.scheme, "init_qparams"):
            raise UnsupportedSchemeError(
                f"scheme {self.scheme.name!r} cannot do RTN init (no init_qparams); "
                f"refusing rather than leaving scale=1.")
        s0, z0 = self.scheme.init_qparams(self.weight.data, config.group_size)
        with torch.no_grad():
            self.scale.copy_(s0)
            self.zero_point.copy_(z0)

    def forward(self, x):
        # After merge the base is a frozen integer artifact -> only dequant it.
        if self.merged:
            return F.linear(x, self.scheme.dequant(self.qweight, self.scale, self.zero_point), self.bias)
        # Training path: the STE fake_quant carries gradients to whichever of
        # {weight, scale, zero_point} are trainable; the adapter (if any) is added.
        w_hat = self.scheme.fake_quant(self.weight, self.scale, self.zero_point)
        y = F.linear(x, w_hat, self.bias)
        return y + self.adapter(x) if self.adapter is not None else y

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None):
        """Fold the adapter into the quantized weights. Unlike peft's merge, the
        result STAYS quantized: the base becomes integer codes and the adapter
        folds into the (float) zero-points."""
        if self.merged:
            return
        codes = self.scheme.quantize(self.weight, self.scale, self.zero_point)
        wq, s, z = self.scheme.merge(codes, self.scale.data, self.zero_point.data, self.adapter)
        self.register_buffer("qweight", wq)                      # int32 codes, the merged artifact
        with torch.no_grad():
            self.scale = nn.Parameter(s.clone(), requires_grad=False)
            self.zero_point = nn.Parameter(z.clone(), requires_grad=False)
        self.adapter = None
        self.merged = True


class BaseQuantTuner:                         # ~ peft BaseTuner / LoraModel
    def __init__(self, model: nn.Module, config: QuantTuningConfig):
        self.model, self.config = model, config
        self.inject_adapters()

    def inject_adapters(self):                # swap target nn.Linear -> QuantLinear via dispatch
        # Collect target names first, then resolve each against the LIVE tree at
        # swap time: replacing one module must not invalidate another's path.
        names = [name for name, module in self.model.named_modules() if self._is_target(name, module)]
        for name in names:
            parent, attr = self._resolve(name)
            module = getattr(parent, attr, None) if parent is not None else None
            if not isinstance(module, nn.Linear) or isinstance(module, QuantLinear):
                continue                      # path changed, or already injected
            setattr(parent, attr, self._create_new_module(module))

    def _create_new_module(self, target: nn.Linear):     # ~ LoraModel._create_new_module
        raise NotImplementedError("each method's model overrides this with its dispatch")

    def _is_target(self, name, module):
        return isinstance(module, nn.Linear) and (
            self.config.target_modules is None or any(t in name for t in self.config.target_modules))

    def _resolve(self, dotted):
        parent = self.model
        *path, attr = dotted.split(".")
        for p in path:
            parent = getattr(parent, p, None)
            if parent is None:
                return None, None             # path no longer exists in the live tree
        return parent, attr
