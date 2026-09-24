"""~ peft/peft_model.py: the model wrapper returned to the user."""
from __future__ import annotations

import os

import torch
from torch import nn

from .config import QuantTuningConfig
from .mapping import QUANT_TUNING_TYPE_TO_TUNER_MAPPING
from .quant_schemes import UnsupportedSchemeError, build_scheme
from .tuners.tuners_utils import QuantLinear

WEIGHTS_NAME = "qpeft_model.pt"


class QuantModel(nn.Module):                  # ~ peft PeftModel
    def __init__(self, model: nn.Module, config: QuantTuningConfig):
        super().__init__()
        if isinstance(model, QuantModel):     # get_quant_model(qmodel, next_phase_cfg)
            model = model.base
        try:
            tuner = QUANT_TUNING_TYPE_TO_TUNER_MAPPING[config.quant_tuning_type]
        except KeyError:
            raise UnsupportedSchemeError(      # refuse loudly, don't KeyError (e.g. PEQA)
                f"quant_tuning_type {config.quant_tuning_type!r} has no tuner yet; "
                f"available: {[t.value for t in QUANT_TUNING_TYPE_TO_TUNER_MAPPING]}. Refusing.") from None
        self.base = tuner(model, config).model
        self.config = config

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    # -- gradient checkpointing (~ peft) ----------------------------------------
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if not hasattr(self.base, "gradient_checkpointing_enable"):
            raise ValueError(f"{type(self.base).__name__} does not support gradient checkpointing "
                             "(no gradient_checkpointing_enable); keep gradient_checkpointing=False.")
        self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base, "gradient_checkpointing_disable"):
            self.base.gradient_checkpointing_disable()

    # -- helpers ---------------------------------------------------------------
    def quant_layers(self) -> list[QuantLinear]:
        return [m for m in self.base.modules() if isinstance(m, QuantLinear)]

    @property
    def is_merged(self) -> bool:
        layers = self.quant_layers()
        return bool(layers) and all(m.merged for m in layers)

    def set_phase(self, config: QuantTuningConfig):
        """Switch every QuantLinear to a new phase config (only the trainable set
        may change; see QuantLinear.apply_config)."""
        for layer in self.quant_layers():
            layer.apply_config(config)
        self.config = config
        return self

    def merge_and_unload(self):               # ~ peft; here the result STAYS quantized
        for module in self.base.modules():
            if isinstance(module, QuantLinear):
                module.merge()
        return self.base

    # -- save / load (docs/specs/save_load.md) ---------------------------------
    def save_pretrained(self, save_directory):
        """Write the merged, still-integer model: qpeft_config.json + qpeft_model.pt."""
        if not self.is_merged:
            raise ValueError("model is not merged; call merge_and_unload() before save_pretrained().")
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, WEIGHTS_NAME))

    @classmethod
    def from_pretrained(cls, base_model: nn.Module, save_directory) -> "QuantModel":
        """Rebuild a merged QuantModel. `base_model` supplies the architecture
        (same pattern as peft.PeftModel.from_pretrained); its weights are replaced."""
        config = QuantTuningConfig.from_pretrained(save_directory)
        build_scheme(config)                  # refuse unknown contract / backend up front
        model = cls(base_model, config)
        for layer in model.quant_layers():
            layer.to_merged_skeleton()
        state = torch.load(os.path.join(save_directory, WEIGHTS_NAME),
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)   # shape mismatch -> RuntimeError, never partial
        for p in model.parameters():
            p.requires_grad_(False)
        return model
