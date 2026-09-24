"""The model wrapper that get_quant_model returns (peft: PeftModel)."""
from __future__ import annotations

import os

import torch
from torch import nn

from .config import QuantTuningConfig
from .mapping import QUANT_TUNING_TYPE_TO_TUNER_MAPPING
from .quant_schemes import UnsupportedSchemeError, build_scheme
from .tuners.tuners_utils import QuantLinear, quant_layers

WEIGHTS_NAME = "qpeft_model.pt"


class QuantModel(nn.Module):
    """Holds the HF model (`base`) with its target linears replaced by QuantLinear,
    and the qpeft config (`config`)."""

    def __init__(self, model: nn.Module, config: QuantTuningConfig):
        super().__init__()
        if isinstance(model, QuantModel):     # the next phase: get_quant_model(qmodel, next_config)
            model = model.base
        if config.quant_tuning_type not in QUANT_TUNING_TYPE_TO_TUNER_MAPPING:
            raise UnsupportedSchemeError(
                f"quant_tuning_type {config.quant_tuning_type!r} has no tuner yet; "
                f"available: {[t.value for t in QUANT_TUNING_TYPE_TO_TUNER_MAPPING]}. Refusing.")
        tuner = QUANT_TUNING_TYPE_TO_TUNER_MAPPING[config.quant_tuning_type]
        self.base = tuner(model, config).model
        self.config = config

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def quant_layers(self) -> list[QuantLinear]:
        return quant_layers(self.base)

    @property
    def is_merged(self) -> bool:
        layers = self.quant_layers()
        return bool(layers) and all(m.merged for m in layers)

    def set_phase(self, config: QuantTuningConfig):
        """Switch every layer to the next phase; only the trainable set may change
        (see QuantLinear.apply_config)."""
        for layer in self.quant_layers():
            layer.apply_config(config)
        self.config = config
        return self

    def merge_and_unload(self):
        """Like peft, but the result stays quantized (see QuantLinear.merge)."""
        for layer in self.quant_layers():
            layer.merge()
        return self.base

    def gradient_checkpointing_enable(self, **kwargs):
        if not hasattr(self.base, "gradient_checkpointing_enable"):
            raise ValueError(f"{type(self.base).__name__} does not support gradient checkpointing "
                             "(no gradient_checkpointing_enable); keep gradient_checkpointing=False.")
        self.base.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base, "gradient_checkpointing_disable"):
            self.base.gradient_checkpointing_disable()

    def save_pretrained(self, save_directory):
        """Write the merged, still-integer model: qpeft_config.json + qpeft_model.pt
        (docs/specs/save_load.md)."""
        if not self.is_merged:
            raise ValueError("model is not merged; call merge_and_unload() before save_pretrained().")
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, WEIGHTS_NAME))

    @classmethod
    def from_pretrained(cls, base_model: nn.Module, save_directory) -> "QuantModel":
        """Load a merged model. `base_model` gives the architecture (as in
        peft.PeftModel.from_pretrained); its weights are replaced."""
        config = QuantTuningConfig.from_pretrained(save_directory)
        build_scheme(config)                  # refuses an unknown scheme / backend before any work
        model = cls(base_model, config)
        for layer in model.quant_layers():
            layer.to_merged_skeleton()
        state = torch.load(os.path.join(save_directory, WEIGHTS_NAME), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)   # a shape mismatch raises; never a partial load
        for p in model.parameters():
            p.requires_grad_(False)
        return model
