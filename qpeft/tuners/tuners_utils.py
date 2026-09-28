"""The quantized linear layer and the tuner that swaps it into a model.
Same roles as peft/tuners/tuners_utils.py (BaseTuner) and peft's lora.Linear."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..config import QuantTuningConfig, TrainableParams
from ..packing import pack_codes, unpack_codes
from ..quant_schemes import QuantScheme, UnsupportedSchemeError

WEIGHT, SCALE, ZERO_POINT, ADAPTER = (TrainableParams.WEIGHT, TrainableParams.SCALE,
                                      TrainableParams.ZERO_POINT, TrainableParams.ADAPTER)


class QuantLinear(nn.Module):
    """A linear layer whose weight is quantized, with an optional adapter.

    One layer serves every method; `config.trainable_params` decides what trains:
        EfficientQAT Block-AP   weight, scale, zero_point
        EfficientQAT E2E-QP     scale
        PEQA                    scale
        QA-LoRA                 adapter

    A layer goes through three states:
        training      fp weight + scale + zero_point, fake-quantized in forward
        codes frozen  integer codes in the packed `qweight` buffer, fp weight dropped
                      (from the start when the weight does not train)
        merged        adapter folded into the zero_point; an inference-only int artifact

    Precision: scale and zero_point are fp32 master copies, because a typical QAT step
    (lr 1e-5 .. 2e-5) is below bf16's resolution and would do nothing. `compute_dtype` is
    the base weight's dtype; the frozen codes and the merged artifact are made in it, so a
    half-precision training forward and its merged artifact see the same numbers.

    Packed codes use the GPTQ layout (qpeft/packing.py). scale and zero_point stay separate
    float tensors, so a zero_point made fractional by an adapter fold is stored exactly."""

    def __init__(self, base: nn.Linear, scheme: QuantScheme,
                 config: QuantTuningConfig, adapter: Optional[nn.Module] = None):
        super().__init__()
        _check_packable_shape(base.in_features, config)
        self.scheme = scheme
        self.config = config
        self.adapter = adapter
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.compute_dtype = base.weight.dtype
        self.merged = False
        self.codes_frozen = False
        self.quant_enabled = True     # False only inside Block-AP, for the fp reference output

        n_groups = base.in_features // config.group_size
        master = dict(dtype=torch.promote_types(base.weight.dtype, torch.float32), device=base.weight.device)
        self.weight = nn.Parameter(base.weight.data.clone())
        self.scale = nn.Parameter(torch.ones(base.out_features, n_groups, **master))
        self.zero_point = nn.Parameter(torch.zeros(base.out_features, n_groups, **master))
        # The bias is not quantized: kept as it is and frozen.
        self.bias = None if base.bias is None else nn.Parameter(base.bias.data.clone(), requires_grad=False)

        self._init_qparams(config)
        self._set_trainable(config)
        if WEIGHT not in config.trainable_params:
            self.freeze_codes()

    @property
    def codes(self) -> torch.Tensor:
        """The frozen / merged integer codes, (out_features, in_features) int32."""
        return unpack_codes(self.qweight, self.config.bits, self.in_features)

    def forward(self, x):
        y = F.linear(x, self._weight_in(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        if self.adapter is not None:
            y = y + self.adapter(x)
        return y

    def _weight_in(self, dtype):
        """The weight this layer computes with, in the input's dtype."""
        scale = self.scale.to(dtype)
        if self.merged:
            # zero_point is not rounded: an adapter fold made it fractional.
            return self.scheme.dequant(self.codes, scale, self.zero_point.to(dtype))
        if not self.quant_enabled:
            return self.weight.to(dtype)
        if self.codes_frozen:
            return self.scheme.dequant(self.codes, scale, self._zero_point_used().to(dtype))
        return self.scheme.fake_quant(self.weight.to(dtype), scale, self._zero_point_used().to(dtype))

    def _zero_point_used(self):
        """round(z) of the fp32 master: the integer that training, the frozen codes and the
        merge all use. Rounding after a cast to bf16 could pick another integer
        (7.49 -> bf16 7.5 -> 8), one full quant step off the frozen codes."""
        return self.scheme.round_zero_point(self.zero_point)

    @torch.no_grad()
    def freeze_codes(self):
        """Fix the integer codes from the current weight, scale and zero_point, and drop the
        fp weight. Same as the official EfficientQAT hand-over (quant_inplace + pack)."""
        if self.codes_frozen or self.merged:
            return
        dtype = self.compute_dtype
        codes = self.scheme.quantize(self.weight.to(dtype), self.scale.to(dtype), self._zero_point_used().to(dtype))
        self.register_buffer("qweight", pack_codes(codes, self.config.bits))
        self.weight = None
        self.codes_frozen = True

    def apply_config(self, config: QuantTuningConfig):
        """Switch to the next phase (e.g. Block-AP -> E2E-QP). Only the trainable set may change."""
        self._check_phase_switch(config)
        self._set_trainable(config)
        if WEIGHT not in config.trainable_params:
            self.freeze_codes()
        self.config = config

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None):
        """Fold the adapter into the zero_point. Unlike peft's merge, the result stays
        quantized: integer codes + scale + (now fractional) zero_point.
        `safe_merge` and `adapter_names` exist for peft compatibility; the NaN check always runs."""
        if self.merged:
            return
        dtype = self.compute_dtype
        scale, zero_point = self.scale.data.to(dtype), self.zero_point.data.to(dtype)
        codes = self.codes if self.codes_frozen else self.scheme.quantize(self.weight.data.to(dtype), scale, zero_point)
        codes, scale, zero_point = self.scheme.merge(codes, scale, zero_point, self.adapter)
        # Checked before anything changes, so a refused merge leaves the layer as it was.
        if not (torch.isfinite(scale).all() and torch.isfinite(zero_point).all()):
            raise ValueError("NaNs/Infs detected in the merged scale/zero_point; refusing to merge.")
        with torch.no_grad():
            self._become_merged(pack_codes(codes, self.config.bits), scale.clone(), zero_point.clone())

    def to_merged_skeleton(self):
        """Turn a freshly injected layer into an empty merged layer of the right shapes,
        so that a saved merged state_dict can be loaded into it (strict)."""
        qweight = torch.zeros(self.in_features * self.config.bits // 32, self.out_features,
                              dtype=torch.int32, device=self.scale.device)
        dtype = self.compute_dtype
        self._become_merged(qweight, self.scale.data.to(dtype), self.zero_point.data.to(dtype))

    def _become_merged(self, qweight, scale, zero_point):
        self.register_buffer("qweight", qweight)
        self.weight = None
        self.adapter = None
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.zero_point = nn.Parameter(zero_point, requires_grad=False)
        self.codes_frozen = True
        self.merged = True

    def _init_qparams(self, config: QuantTuningConfig):
        """RTN (round-to-nearest) start grid. Other inits are refused: without one,
        the grid would stay at the useless scale = 1."""
        if config.init_weights != "rtn":
            raise UnsupportedSchemeError(
                f"init_weights={config.init_weights!r} is not implemented; only 'rtn'. "
                f"Refusing rather than approximating with a degenerate scale=1 grid.")
        if not hasattr(self.scheme, "init_qparams"):
            raise UnsupportedSchemeError(
                f"scheme {self.scheme.name!r} cannot do RTN init (no init_qparams); "
                f"refusing rather than leaving scale=1.")
        scale, zero_point = self.scheme.init_qparams(self.weight.data.to(self.scale.dtype), config.group_size)
        with torch.no_grad():
            self.scale.copy_(scale)
            self.zero_point.copy_(zero_point)

    def _set_trainable(self, config: QuantTuningConfig):
        trainable = set(config.trainable_params)
        if self.weight is not None:
            self.weight.requires_grad_(WEIGHT in trainable)
        self.scale.requires_grad_(SCALE in trainable)
        self.zero_point.requires_grad_(ZERO_POINT in trainable)
        if self.adapter is not None:
            for p in self.adapter.parameters():
                p.requires_grad_(ADAPTER in trainable)

    def _check_phase_switch(self, config: QuantTuningConfig):
        """Refuse anything that would change the representation, and un-freezing the codes
        (the weight would then train while the frozen codes stay as they are)."""
        if self.merged:
            raise UnsupportedSchemeError("layer is merged (an int artifact); it cannot change phase.")
        for field in ("quant_tuning_type", "bits", "group_size", "qat_scheme", "backend"):
            old, new = getattr(self.config, field), getattr(config, field)
            if old != new:
                raise UnsupportedSchemeError(
                    f"phase switch may only change the trainable set, not {field!r} ({old!r} -> {new!r}). Refusing.")
        self.scheme.assert_supported(config)
        trainable = set(config.trainable_params)
        if ADAPTER in trainable and self.adapter is None:
            raise UnsupportedSchemeError("config trains an adapter but this layer has none.")
        if WEIGHT in trainable and self.codes_frozen:
            raise UnsupportedSchemeError(
                "codes are frozen; making the weight trainable again would let weight and "
                "codes diverge. Refusing.")


def _check_packable_shape(in_features: int, config: QuantTuningConfig):
    if in_features % config.group_size != 0:
        raise ValueError(f"in_features {in_features} not divisible by group_size {config.group_size}")
    if (in_features * config.bits) % 32 != 0:
        raise ValueError(f"in_features {in_features} x {config.bits} bits does not fill whole "
                         "int32 words; the packed codes need in_features * bits to be a multiple of 32.")


def quant_layers(model: nn.Module) -> list[QuantLinear]:
    return [m for m in model.modules() if isinstance(m, QuantLinear)]


class BaseQuantTuner:
    """Replaces the target nn.Linear layers of a model with QuantLinear (peft: BaseTuner).
    Each method's subclass says how to build the new layer (`_create_new_module`)."""

    def __init__(self, model: nn.Module, config: QuantTuningConfig):
        self.model = model
        self.config = config
        self.inject_adapters()

    def inject_adapters(self):
        existing = quant_layers(self.model)
        if existing:
            # The model is already quantized (the next phase of efficient_qat_schedule):
            # switch the existing layers to the new phase.
            for layer in existing:
                layer.apply_config(self.config)
            return

        targets = [name for name, module in self.model.named_modules() if self._is_target(name, module)]
        for name in targets:
            parent_name, _, attr = name.rpartition(".")
            parent = self.model.get_submodule(parent_name)
            setattr(parent, attr, self._create_new_module(getattr(parent, attr)))
        self._freeze_everything_else()

    def _create_new_module(self, target: nn.Linear) -> QuantLinear:
        raise NotImplementedError("each method's model builds its own QuantLinear")

    def _is_target(self, name, module):
        if not isinstance(module, nn.Linear):
            return False
        return self.config.target_modules is None or any(t in name for t in self.config.target_modules)

    def _freeze_everything_else(self):
        """Embeddings, norms, lm_head, ... do not train; only the QuantLinear parameters
        that the config makes trainable can move (peft: _mark_only_adapters_as_trainable)."""
        own = {id(p) for layer in quant_layers(self.model) for p in layer.parameters()}
        for p in self.model.parameters():
            if id(p) not in own:
                p.requires_grad_(False)
