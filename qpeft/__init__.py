"""qpeft -- quantization-aware, PEFT-style tuning whose merge stays quantized."""
from .config import QuantTuningConfig, QuantTuningType, TrainableParams
from .quant_schemes import (
    FakeQuantizeConfig, QuantScheme, UnsupportedSchemeError, build_scheme, register_scheme,
)
from .mapping import get_quant_model
from .peft_model import QuantModel
from .utils import (
    MergeMismatchError, check_layer_merge_equivalence, check_merge_equivalence, verify_quant_model,
)
from .training import param_groups
from .block_ap import run_block_ap
from .tuners.efficient_qat import EfficientQATConfig, EfficientQATModel, efficient_qat_schedule
from .tuners.peqa import PEQAConfig, PEQAModel
from .tuners.qa_lora import QALoraConfig, QALoraModel, ZeroPointFoldLoRA

__version__ = "0.0.1"
__all__ = [
    "QuantTuningConfig", "QuantTuningType", "TrainableParams",
    "FakeQuantizeConfig", "QuantScheme", "UnsupportedSchemeError", "build_scheme", "register_scheme",
    "get_quant_model", "QuantModel", "MergeMismatchError", "check_merge_equivalence",
    "check_layer_merge_equivalence", "verify_quant_model", "param_groups", "run_block_ap",
    "EfficientQATConfig", "EfficientQATModel", "efficient_qat_schedule",
    "PEQAConfig", "PEQAModel",
    "QALoraConfig", "QALoraModel", "ZeroPointFoldLoRA",
]


def __getattr__(name):
    # The trainer needs transformers; keep `import qpeft` torch-only.
    if name in ("QATTrainer", "QATTrainingArguments"):
        try:
            from . import trainer
        except ImportError as e:
            raise ImportError("QATTrainer needs transformers: pip install 'qpeft[train]'.") from e
        return getattr(trainer, name)
    raise AttributeError(f"module 'qpeft' has no attribute {name!r}")
