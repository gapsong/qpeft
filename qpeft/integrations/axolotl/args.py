"""The `qpeft:` block of an axolotl YAML, and the axolotl settings a qpeft run refuses.

axolotl merges QPeftArgs into its own config model (`BasePlugin.get_input_args`), so the
validators here see the whole config. axolotl then hands plugins the validated config as a
plain dict; `QPeftBlock.model_validate(cfg.qpeft).quant_configs()` gives the qpeft configs back.
"""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from ...config import QuantTuningConfig, QuantTuningType
from ...quant_schemes import UnsupportedSchemeError, build_scheme

# The plugin trains with torch.optim.AdamW over param_groups; these axolotl names mean exactly that.
ADAMW_OPTIMIZERS = ("adamw_torch", "adamw_torch_fused")

_SHARDED = "sharded training is not supported: QuantLinear quantizes and merges full weights."
_PREQUANTIZED = ("qpeft quantizes the full-precision base itself; "
                 "a pre-quantized base has no grid that qpeft can merge into.")
_REFUSED_SETTINGS = {
    "fsdp": _SHARDED,
    "fsdp_config": _SHARDED,
    "deepspeed": _SHARDED,
    "rl": "RL trainers are not supported; qpeft runs supervised fine-tuning.",
    "load_in_4bit": _PREQUANTIZED,
    "load_in_8bit": _PREQUANTIZED,
    "gptq": _PREQUANTIZED,
    "relora": "ReLoRA merges LoRA into float weights during training.",
    "merge_lora": ("qpeft merges into the integer model itself after training "
                   "(see the `export` key of the block)."),
    "qat": "axolotl's QAT would fake-quantize a second time, on top of qpeft.",
}
_PARALLEL_SIZES = ("tensor_parallel_size", "context_parallel_size", "sequence_parallel_degree",
                   "expert_parallel_size", "dp_shard_size")


class BlockAPArgs(BaseModel):
    """EfficientQAT phase 1. Unset fields use the official EfficientQAT values."""
    model_config = ConfigDict(extra="forbid")

    epochs: Optional[PositiveInt] = None
    train_size: Optional[PositiveInt] = None          # calibration rows; all their activations stay in memory
    seqlen: Optional[PositiveInt] = None
    batch_size: Optional[PositiveInt] = None
    weight_lr: Optional[PositiveFloat] = None
    quant_lr: Optional[PositiveFloat] = None


class QPeftBlock(BaseModel):
    """The `qpeft:` block. An unset field takes the default of the method's qpeft config class,
    so qpeft stays the one source of truth for defaults and for which method has which field."""
    model_config = ConfigDict(extra="forbid")

    method: Literal["qa_lora", "peqa", "efficient_qat"]
    export: Literal["qpeft", "gptq"] = "qpeft"
    block_ap: Optional[BlockAPArgs] = None             # efficient_qat runs Block-AP even when unset
    bits: Optional[Literal[2, 3, 4, 8]] = None         # the widths GPTQ kernels serve
    group_size: Optional[PositiveInt] = None
    backend: Optional[str] = None
    target_modules: Optional[Union[list[str], str]] = None
    r: Optional[PositiveInt] = None
    lora_alpha: Optional[PositiveInt] = None
    lora_dropout: Optional[float] = Field(default=None, ge=0, lt=1)

    @model_validator(mode="after")
    def refuse_what_qpeft_cannot_merge_exactly(self):
        if self.block_ap is not None and self.method != "efficient_qat":
            raise ValueError(f"qpeft.block_ap is EfficientQAT's phase 1; method {self.method!r} has no Block-AP.")
        if self.export == "gptq" and self.method == "qa_lora":
            raise ValueError(
                "qpeft.export: gptq is refused for method qa_lora: the QA-LoRA merge makes the "
                "zero-points fractional, GPTQ stores integer zero-points, and rounding them would "
                "break fake_quant == merge. Use export: qpeft.")
        try:
            for config in self.quant_configs():
                build_scheme(config)
        except (UnsupportedSchemeError, NotImplementedError) as e:
            raise ValueError(str(e)) from e
        return self

    def quant_configs(self) -> list[QuantTuningConfig]:
        """The qpeft configs the run trains with, in order: Block-AP then E2E-QP for
        efficient_qat, one config otherwise."""
        fields = self.model_dump(exclude={"method", "export", "block_ap"}, exclude_none=True)
        fields["quant_tuning_type"] = QuantTuningType(self.method.upper())
        if self.method == "efficient_qat":
            return [QuantTuningConfig.from_dict({**fields, "phase": phase})
                    for phase in ("block_ap", "e2e_qp")]
        return [QuantTuningConfig.from_dict(fields)]


class QPeftArgs(BaseModel):
    """The input args of the qpeft axolotl plugin: the `qpeft:` block, and the refusal of
    every other axolotl setting that a qpeft run cannot honor."""

    qpeft: Optional[QPeftBlock] = None

    # The name must not clash with an axolotl validator: in the merged config class axolotl's
    # validators come first in the MRO, and one with the same name would hide this one.
    @model_validator(mode="before")
    @classmethod
    def refuse_axolotl_settings_qpeft_cannot_honor(cls, data):
        if not isinstance(data, dict):
            return data
        if data.get("adapter") != "qpeft" and data.get("qpeft") is None:
            return data             # not a qpeft run: the plugin loads on every axolotl run
        reasons = _refusals(data)
        if reasons:
            raise ValueError("adapter: qpeft refuses this config:\n" + "\n".join(f"  - {r}" for r in reasons))
        return data


def _refusals(data: dict) -> list[str]:
    if data.get("adapter") != "qpeft":
        return ["a `qpeft:` block needs `adapter: qpeft`; without it the block would be ignored."]
    if data.get("qpeft") is None:
        return ["`adapter: qpeft` needs a `qpeft:` block (at least `method:`)."]

    # A falsy value changes nothing, so it passes; axolotl itself sets lora_dropout: 0.0 for any adapter.
    reasons = [f"`{key}`: {why}" for key, why in _REFUSED_SETTINGS.items() if data.get(key)]
    reasons += [f"`{key}: {data[key]}`: parallelism that splits the model is not supported."
                for key in _PARALLEL_SIZES if (data.get(key) or 1) > 1]
    lora_keys = [f"`{key}`" for key, value in sorted(data.items()) if key.startswith(("lora_", "peft_")) and value]
    if lora_keys:
        reasons.append(", ".join(lora_keys) + ": peft LoRA settings do not apply; QA-LoRA takes "
                       "r / lora_alpha / lora_dropout inside the `qpeft:` block.")

    optimizer = data.get("optimizer")
    if optimizer is not None and optimizer not in ADAMW_OPTIMIZERS:
        reasons.append(f"`optimizer: {optimizer}`: qpeft trains with AdamW; use one of {list(ADAMW_OPTIMIZERS)}.")

    block = data["qpeft"]
    method = block.get("method") if isinstance(block, dict) else getattr(block, "method", None)
    world_size = _world_size(data)
    if method == "efficient_qat" and world_size > 1:
        reasons.append(f"method efficient_qat with world size {world_size}: Block-AP calibrates "
                       f"in one process; multi-process runs are not supported yet.")
    return reasons


def _world_size(data: dict) -> int:
    """As axolotl's own validators read it: capabilities.n_gpu is WORLD_SIZE at config time."""
    return int((data.get("capabilities") or {}).get("n_gpu") or data.get("world_size") or 1)
