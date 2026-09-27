"""Taking over a layer that torchao already quantized (peft: lora/torchao.py). Not built yet."""
from __future__ import annotations

from ..tuners_utils import QuantLinear


def dispatch_torchao(target, config, *, get_apply_tensor_subclass):
    """When `target` is already backed by a torchao AffineQuantizedTensor, build a
    TorchaoQuantLinear so merge() folds into the tensor-subclass zero_point in place."""
    return TorchaoQuantLinear(target, config, get_apply_tensor_subclass=get_apply_tensor_subclass)


class TorchaoQuantLinear(QuantLinear):        # peft: TorchaoLoraLinear
    """Adopt an ALREADY torchao-quantized layer (target.weight is a torchao packed
    tensor) and fold the adapter into its zero-point in place.

    Still a stub, on purpose: this couples to torchao's packed tensor-subclass
    internals, which are in flux in torchao 0.18 (the int4 path needs an external
    kernel lib; the int8 `Int8Tensor` exposes `.qdata/.scale/.zero_point`). Doing it
    cleanly means pinning a torchao version and testing on the target hardware.
    The torchao *scheme* backend (`backend="torchao"`) already implements the same
    contract on torchao's STABLE primitives and passes the equivalence gate -- use
    that to quantize a plain HF model. This class is only for taking over weights
    torchao itself already packed."""

    def __init__(self, target, config, *, get_apply_tensor_subclass):
        raise NotImplementedError(
            "adopting a pre-packed torchao tensor subclass is not built yet; see the "
            "class docstring. Use backend='torchao' to quantize a plain HF model.")
