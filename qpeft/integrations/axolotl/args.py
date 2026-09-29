"""The `qpeft:` block of an axolotl config."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class QpeftMethod(str, Enum):
    QA_LORA = "qa_lora"       # group-pooled LoRA; the merge folds it into the zero-points
    PEQA = "peqa"             # only the quantization scales train


class QpeftConfig(BaseModel):
    method: QpeftMethod = Field(default=QpeftMethod.QA_LORA, description="qa_lora or peqa")
    bits: int = Field(default=4, description="bit-width of the integer codes")
    group_size: int = Field(default=64, description="inputs per quantization group")
    backend: str = Field(default="auto", description="auto / torch or torchao")


class QpeftArgs(BaseModel):
    qpeft: QpeftConfig | None = Field(default=None, description="qpeft settings for `adapter: qpeft`")
