"""What every export of a merged qpeft model into a Hugging Face checkpoint shares: finding the
merged layers, their integer zero-points, the model a loader builds, and writing the files.
Each format (gptq.py, compressed_tensors.py) adds only its own tensors and quantization_config."""
from __future__ import annotations

import copy
import json
import os

import torch
from torch import nn

from ..quant_schemes import UnsupportedSchemeError
from ..tuners.tuners_utils import QuantLinear

QPEFT_KEYS = ("qweight", "scale", "zero_point", "bias")


def merged_quant_layers(model: nn.Module, format_name: str) -> tuple[nn.Module, dict[str, QuantLinear]]:
    """The Hugging Face model under a QuantModel (or the model itself) and its merged QuantLinears by name."""
    root = getattr(model, "base", model)
    if not (hasattr(root, "save_pretrained") and hasattr(root, "config")):
        raise UnsupportedSchemeError(
            f"{format_name} export writes a Hugging Face checkpoint; {type(root).__name__} is not a "
            "transformers PreTrainedModel.")
    layers = {name: m for name, m in root.named_modules() if isinstance(m, QuantLinear)}
    if not layers:
        raise ValueError("no QuantLinear found; nothing to export.")
    unmerged = sorted(name for name, layer in layers.items() if not layer.merged)
    if unmerged:
        raise UnsupportedSchemeError(
            f"{unmerged[0]}: {format_name} export needs a merged layer; call merge_and_unload() first.")
    return root, layers


def integer_zero_points(name: str, layer: QuantLinear, format_name: str) -> torch.Tensor:
    """The merged zero-points as (out, n_groups) int32, or a refusal if any is fractional."""
    zero_point = layer.zero_point.detach()
    if not torch.equal(zero_point, zero_point.round()):
        raise UnsupportedSchemeError(
            f"{name}: its zero-points are fractional (a QA-LoRA fold, z' = z - delta / s). {format_name} "
            "stores integer zero-points, and rounding them would break fake_quant == merge. Keep the "
            "qpeft artifact (save_pretrained) or run it on tinygemm (qpeft.kernels.to_tinygemm).")
    return zero_point.to(torch.int32)


def loader_skeleton(root: nn.Module) -> nn.Module:
    """The model a loader builds from config.json before it replaces any layer, on the meta
    device: no weights, and the user's model and config stay untouched."""
    with torch.device("meta"):
        return type(root)(copy.deepcopy(root.config))


def save_checkpoint(root: nn.Module, layer_tensors: dict[str, dict[str, torch.Tensor]],
                    save_directory, quantization_config: dict) -> None:
    """`root.save_pretrained` with each QuantLinear's qpeft tensors replaced by `layer_tensors[name]`,
    and `quantization_config` added to config.json."""
    state = root.state_dict()
    for name, tensors in layer_tensors.items():
        for key in QPEFT_KEYS:
            state.pop(f"{name}.{key}", None)
        state.update({f"{name}.{key}": value for key, value in tensors.items()})
    root.save_pretrained(save_directory, state_dict=state)

    path = os.path.join(save_directory, "config.json")
    with open(path) as f:
        config = json.load(f)
    config["quantization_config"] = quantization_config
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
