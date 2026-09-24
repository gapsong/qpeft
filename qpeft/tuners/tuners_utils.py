"""Shared tuner machinery. ~ peft/tuners/tuners_utils.py (BaseTuner, BaseTunerLayer)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from ..config import QuantTuningConfig, TrainableParams
from ..packing import pack_codes, unpack_codes
from ..quant_schemes import QuantScheme, UnsupportedSchemeError


class AdapterLayer:                          # ~ peft BaseTunerLayer / torchtune AdapterModule
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None): ...
    def unmerge(self): ...


class QuantLinear(nn.Module, AdapterLayer):  # ~ peft lora.Linear / torchtune QATLoRALinear
    """A base linear held in a quantized representation, with an optional adapter.

    Which of {weight, scale, zero_point, adapter} carry gradients is decided by
    `config.trainable_params` -- that switch is what makes PEQA / EfficientQAT /
    QA-LoRA the same layer under different configs.

    Precision: `compute_dtype` is the base weight's dtype. scale / zero_point are
    fp32 master copies, because an optimizer step of a typical QAT learning rate
    (1e-5 .. 2e-5) is below bf16's resolution and would silently do nothing.
    The forward always computes in the input's dtype on params cast to it, and
    codes and the merged artifact are made in `compute_dtype` -- so a half-
    precision training forward and its merged artifact see the same numbers.

    Frozen and merged codes are stored packed in `qweight` (GPTQ layout, see
    qpeft/packing.py); scale and zero_point stay separate float tensors, so a
    zero-point made fractional by an adapter fold is stored exactly."""

    def __init__(self, base: nn.Linear, scheme: QuantScheme,
                 config: QuantTuningConfig, adapter: Optional[nn.Module] = None):
        super().__init__()
        if base.in_features % config.group_size != 0:
            raise ValueError(f"in_features {base.in_features} not divisible by group_size {config.group_size}")
        if (base.in_features * config.bits) % 32 != 0:
            raise ValueError(f"in_features {base.in_features} x {config.bits} bits does not fill whole "
                             "int32 words; the packed codes need in_features * bits to be a multiple of 32.")
        self.scheme, self.config, self.adapter = scheme, config, adapter
        self.in_features, self.out_features = base.in_features, base.out_features
        self.merged = False
        self.codes_frozen = False     # True once the integer codes are fixed (qweight buffer)
        self.quant_enabled = True     # False only inside Block-AP to get the fp reference output
        tp = set(config.trainable_params)
        n_groups = base.in_features // config.group_size
        w = base.weight.data
        # scale/zero_point follow the base weight's device and are fp32 masters
        # (see the class docstring), so a half-precision or GPU base survives injection.
        self.compute_dtype = w.dtype
        kw = dict(dtype=torch.promote_types(w.dtype, torch.float32), device=w.device)
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
        # Weight not trainable => the integer codes can never legitimately move.
        # Fix them now (EfficientQAT E2E-QP, PEQA, QA-LoRA): training then moves
        # only scale / zero_point / adapter on top of FIXED codes, as in the papers.
        if TrainableParams.WEIGHT not in tp:
            self.freeze_codes()

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
        s0, z0 = self.scheme.init_qparams(self.weight.data.to(self.scale.dtype), config.group_size)
        with torch.no_grad():
            self.scale.copy_(s0)
            self.zero_point.copy_(z0)

    def forward(self, x):
        dt = x.dtype                  # params are cast to the input's dtype (see class docstring)
        s, z = self.scale.to(dt), self.zero_point.to(dt)
        bias = None if self.bias is None else self.bias.to(dt)
        if self.merged:
            # After merge the base is a frozen integer artifact -> only dequant it.
            return F.linear(x, self.scheme.dequant(self.codes, s, z), bias)
        if not self.quant_enabled:
            # fp reference path, used only by Block-AP to compute its targets.
            w_hat = self.weight.to(dt)
        elif self.codes_frozen:
            # Fixed integer codes; gradients reach scale / zero_point directly. The
            # zero-point is used rounded, exactly as fake_quant and merge use it.
            w_hat = self.scheme.dequant(self.codes, s, self.scheme.round_zero_point(z))
        else:
            # Training path: the STE fake_quant carries gradients to whichever of
            # {weight, scale, zero_point} are trainable.
            w_hat = self.scheme.fake_quant(self.weight.to(dt), s, z)
        y = F.linear(x, w_hat, bias)
        return y + self.adapter(x) if self.adapter is not None else y

    @property
    def codes(self) -> torch.Tensor:
        """The frozen / merged integer codes, (out_features, in_features) int32."""
        return unpack_codes(self.qweight, self.config.bits, self.in_features)

    # -- phase handling ------------------------------------------------------
    @torch.no_grad()
    def freeze_codes(self):
        """Fix the integer codes from the current (weight, scale, zero_point).
        Equivalent to the official EfficientQAT hand-over (quant_inplace + pack):
        from here on the codes are a packed buffer and never re-rounded, and the
        fp weight is dropped."""
        if self.codes_frozen or self.merged:
            return
        cd = self.compute_dtype
        self.register_buffer("qweight", pack_codes(self.scheme.quantize(
            self.weight.to(cd), self.scale.to(cd), self.zero_point.to(cd)), self.config.bits))
        self.weight = None
        self.codes_frozen = True

    def apply_config(self, config: QuantTuningConfig):
        """Switch this layer to a new phase config (e.g. Block-AP -> E2E-QP).

        Only the trainable set may change. Anything that would change the
        representation (bits, group size, contract, backend, method) is refused,
        and so is un-freezing codes: weights and codes would silently diverge."""
        if self.merged:
            raise UnsupportedSchemeError("layer is merged (an int artifact); it cannot change phase.")
        old = self.config
        for field in ("quant_tuning_type", "bits", "group_size", "qat_scheme", "backend"):
            if getattr(old, field) != getattr(config, field):
                raise UnsupportedSchemeError(
                    f"phase switch may only change the trainable set, not {field!r} "
                    f"({getattr(old, field)!r} -> {getattr(config, field)!r}). Refusing.")
        self.scheme.assert_supported(config)
        tp = set(config.trainable_params)
        if TrainableParams.ADAPTER in tp and self.adapter is None:
            raise UnsupportedSchemeError("config trains an adapter but this layer has none.")
        if TrainableParams.WEIGHT in tp and self.codes_frozen:
            raise UnsupportedSchemeError(
                "codes are frozen; making the weight trainable again would let weight and "
                "codes diverge. Refusing.")
        if self.weight is not None:
            self.weight.requires_grad_(TrainableParams.WEIGHT in tp)
        self.scale.requires_grad_(TrainableParams.SCALE in tp)
        self.zero_point.requires_grad_(TrainableParams.ZERO_POINT in tp)
        if self.adapter is not None:
            for p in self.adapter.parameters():
                p.requires_grad_(TrainableParams.ADAPTER in tp)
        if TrainableParams.WEIGHT not in tp:
            self.freeze_codes()
        self.config = config

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None):
        """Fold the adapter into the quantized weights. Unlike peft's merge, the
        result STAYS quantized: the base becomes integer codes and the adapter
        folds into the (float) zero-points."""
        if self.merged:
            return
        cd = self.compute_dtype       # the artifact is made in the base dtype
        s, z = self.scale.data.to(cd), self.zero_point.data.to(cd)
        codes = (self.codes if self.codes_frozen
                 else self.scheme.quantize(self.weight.data.to(cd), s, z))
        wq, s, z = self.scheme.merge(codes, s, z, self.adapter)
        # ~ peft safe_merge, but always on: never write a broken artifact. Checked
        # BEFORE anything is mutated, so a refused merge leaves the layer untouched.
        if not (torch.isfinite(s).all() and torch.isfinite(z).all()):
            raise ValueError("NaNs/Infs detected in the merged scale/zero_point; refusing to merge.")
        self.register_buffer("qweight", pack_codes(wq, self.config.bits))
        with torch.no_grad():
            self.scale = nn.Parameter(s.clone(), requires_grad=False)
            self.zero_point = nn.Parameter(z.clone(), requires_grad=False)
        self.weight = None            # merge stays int: the fp master copy is gone
        self.adapter = None
        self.codes_frozen = True
        self.merged = True

    def to_merged_skeleton(self):
        """Turn a freshly injected layer into an empty merged layer of the right
        shape, so a saved merged state_dict can be loaded into it (strict)."""
        device = self.scale.device
        self.register_buffer("qweight", torch.zeros(self.in_features * self.config.bits // 32, self.out_features,
                                                    dtype=torch.int32, device=device))
        self.weight = None
        self.adapter = None
        cd = self.compute_dtype       # a merged artifact stores scale / zero_point in the base dtype
        self.scale = nn.Parameter(self.scale.data.to(cd), requires_grad=False)
        self.zero_point = nn.Parameter(self.zero_point.data.to(cd), requires_grad=False)
        self.codes_frozen = True
        self.merged = True


class BaseQuantTuner:                         # ~ peft BaseTuner / LoraModel
    def __init__(self, model: nn.Module, config: QuantTuningConfig):
        self.model, self.config = model, config
        self.inject_adapters()

    def inject_adapters(self):                # swap target nn.Linear -> QuantLinear via dispatch
        # Already injected (e.g. Block-AP -> E2E-QP via efficient_qat_schedule):
        # switch the phase of the existing layers instead of silently skipping them.
        existing = [m for m in self.model.modules() if isinstance(m, QuantLinear)]
        if existing:
            for layer in existing:
                layer.apply_config(self.config)
            return
        # Collect target names first, then resolve each against the LIVE tree at
        # swap time: replacing one module must not invalidate another's path.
        names = [name for name, module in self.model.named_modules() if self._is_target(name, module)]
        for name in names:
            parent, attr = self._resolve(name)
            module = getattr(parent, attr, None) if parent is not None else None
            if not isinstance(module, nn.Linear) or isinstance(module, QuantLinear):
                continue                      # path changed, or already injected
            setattr(parent, attr, self._create_new_module(module))
        self._freeze_non_quant_params()

    def _freeze_non_quant_params(self):
        """~ peft _mark_only_adapters_as_trainable: everything outside the
        QuantLinear axes (embeddings, norms, lm_head, ...) is frozen, so only the
        config's trainable set can move."""
        own = set()
        for m in self.model.modules():
            if isinstance(m, QuantLinear):
                own.update(id(p) for p in m.parameters())
        for p in self.model.parameters():
            if id(p) not in own:
                p.requires_grad_(False)

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
