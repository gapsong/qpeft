"""The config every method extends (peft: peft/config.py + utils/peft_types.py)."""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

CONFIG_NAME = "qpeft_config.json"


class QuantTuningType(str, Enum):
    EFFICIENT_QAT = "EFFICIENT_QAT"
    QA_LORA = "QA_LORA"
    PEQA = "PEQA"


class TrainableParams(str, Enum):
    WEIGHT = "weight"
    SCALE = "scale"
    ZERO_POINT = "zero_point"
    ADAPTER = "adapter"


@dataclass
class QuantTuningConfig:
    quant_tuning_type: QuantTuningType
    bits: int = 4
    group_size: int = 64
    qat_scheme: str = "int_uniform"
    backend: str = "auto"
    target_modules: Optional[Union[list[str], str]] = None   # as in peft, see BaseQuantTuner._is_target
    task_type: Optional[str] = None
    init_weights: str = "rtn"
    trainable_params: tuple[TrainableParams, ...] = (TrainableParams.SCALE,)

    def to_dict(self) -> dict:
        """JSON-ready: enums as their values, tuples as lists."""
        return {f.name: _json_value(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "QuantTuningConfig":
        """Build the right subclass (from quant_tuning_type). Unknown types or fields are refused,
        e.g. a config written by a newer qpeft. trainable_params is not read: each method
        derives it from its own fields."""
        from .mapping import QUANT_TUNING_TYPE_TO_CONFIG_MAPPING
        from .quant_schemes import UnsupportedSchemeError
        d = dict(d)
        try:
            target = QUANT_TUNING_TYPE_TO_CONFIG_MAPPING[QuantTuningType(d.pop("quant_tuning_type"))]
        except (KeyError, ValueError):
            raise UnsupportedSchemeError(
                f"saved config has an unknown quant_tuning_type; "
                f"known: {[t.value for t in QUANT_TUNING_TYPE_TO_CONFIG_MAPPING]}. Refusing.") from None
        if not issubclass(target, cls):
            raise UnsupportedSchemeError(
                f"saved config is a {target.__name__}, not a {cls.__name__}. Refusing.")
        d.pop("trainable_params", None)
        unknown = sorted(set(d) - {f.name for f in dataclasses.fields(target) if f.init})
        if unknown:
            raise UnsupportedSchemeError(
                f"saved config has fields {target.__name__} does not know: {unknown} "
                f"(written by a newer qpeft?). Refusing.")
        return target(**d)

    def save_pretrained(self, save_directory) -> None:
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, CONFIG_NAME), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory) -> "QuantTuningConfig":
        with open(os.path.join(save_directory, CONFIG_NAME)) as f:
            return cls.from_dict(json.load(f))


def _json_value(v):
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, (tuple, list)):
        return [_json_value(x) for x in v]
    return v
