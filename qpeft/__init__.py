"""qpeft -- quantization-aware, PEFT-style tuning whose merge stays quantized."""
from .config import QuantTuningConfig, QuantTuningType, TrainableParams
from .schemes import (
    FakeQuantizeConfig, QuantScheme, UnsupportedSchemeError, build_scheme, register_scheme,
)
from .mapping import get_quant_model
from .peft_model import QuantModel
from .utils import check_merge_equivalence
from .tuners.efficient_qat import EfficientQATConfig, EfficientQATModel, efficient_qat_schedule
from .tuners.qa_lora import QALoraConfig, QALoraModel, ZeroPointFoldLoRA

__version__ = "0.0.1"
__all__ = [
    "QuantTuningConfig", "QuantTuningType", "TrainableParams",
    "FakeQuantizeConfig", "QuantScheme", "UnsupportedSchemeError", "build_scheme", "register_scheme",
    "get_quant_model", "QuantModel", "check_merge_equivalence",
    "EfficientQATConfig", "EfficientQATModel", "efficient_qat_schedule",
    "QALoraConfig", "QALoraModel", "ZeroPointFoldLoRA",
]
