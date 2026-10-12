"""Write a merged qpeft model as a compressed-tensors checkpoint (the llm-compressor format), in the
asymmetric "pack-quantized" WNA16 layout that vLLM and transformers (with the compressed-tensors
library) load.

compressed-tensors stores signed codes and zero-points, q - 2**(bits-1) and z - 2**(bits-1), and its
packer adds 2**(bits-1) back before packing. So the packed words hold qpeft's unsigned q and z as they
are, and the dequant
    w = (q_signed - z_signed) * s = (q - z) * s
is qpeft's, bit for bit, in the scale's dtype. The export only transposes the codes, packs the
zero-points and keeps the scale in the model's dtype: no cast at all. It refuses every layer where a
loader would not read that back exactly:
    - a fractional zero-point (a QA-LoRA fold): the format stores integers
    - a quantized layer that the model built from config.json does not hold as a plain nn.Linear,
      or whose weight that model ties to another tensor (a tied lm_head)

Packing: every row is one bit stream, value k at bit k * bits, cut into int32 words (a value may
spill into the next word). That is qpeft's own qweight layout, and compressed-tensors' since 0.18.
compressed-tensors 0.17 and older pad 3 / 5 / 6 / 7-bit values to whole words instead; for 2, 4 and
8 bits the two layouts are the same.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..packing import pack_codes
from ..quant_schemes import UnsupportedSchemeError
from ..tuners.tuners_utils import QuantLinear
from ._checkpoint import integer_zero_points, loader_skeleton, merged_quant_layers, save_checkpoint

FORMAT_NAME = "compressed-tensors"
SUPPORTED_BITS = (2, 3, 4, 5, 6, 7, 8)        # vLLM's WNA16 scheme with zero-points takes these


def export_compressed_tensors(model: nn.Module, save_directory) -> None:
    """Write a merged QuantModel (or its Hugging Face base) to `save_directory` as a compressed-tensors
    checkpoint: the model's own files from `save_pretrained`, with pack-quantized tensors for every
    QuantLinear and a `quantization_config` in config.json. Every layer is checked before anything is
    written. The tokenizer is not part of the model; save it next to it yourself."""
    root, layers = merged_quant_layers(model, FORMAT_NAME)
    _check_loader_holds_plain_linears(root, layers)
    tensors = {name: _pack_quantized_tensors(name, layer) for name, layer in layers.items()}
    save_checkpoint(root, tensors, save_directory, _quantization_config(layers))


def _check_loader_holds_plain_linears(root: nn.Module, layers: dict[str, QuantLinear]) -> None:
    """A loader quantizes the modules that config.json names, in the model it builds from config.json.
    There each quantized layer must be a plain nn.Linear (a subclass, such as an MoE router, may have a
    forward of its own that a loader does not replace) with a weight of its own (a tied lm_head would
    load the embedding's float weight, or overwrite it)."""
    skeleton = loader_skeleton(root)
    parameter_names: dict[int, list[str]] = {}
    for name, parameter in skeleton.named_parameters(remove_duplicate=False):
        parameter_names.setdefault(id(parameter), []).append(name)

    for name in sorted(layers):
        module = skeleton.get_submodule(name)
        if type(module) is not nn.Linear:
            raise UnsupportedSchemeError(
                f"{name}: in the model a loader builds from config.json it is a {type(module).__name__}, "
                "not a plain nn.Linear, so the loader may not run it as a quantized linear. Quantize "
                "without it (target_modules), then export.")
        tied = sorted(set(parameter_names[id(module.weight)]) - {f"{name}.weight"})
        if tied:
            raise UnsupportedSchemeError(
                f"{name}: the model a loader builds from config.json ties its weight to {tied}, which "
                "stays float. Quantize without it (target_modules), or untie the weights, then export.")


def _pack_quantized_tensors(name: str, layer: QuantLinear) -> dict[str, torch.Tensor]:
    """The tensors of one pack-quantized layer: weight_packed (out, in * bits / 32) int32,
    weight_scale (out, n_groups) in the model's dtype, weight_zero_point
    (ceil(out * bits / 32), n_groups) int32, packed along the outputs, weight_shape [out, in] int64."""
    bits = layer.config.bits
    if bits not in SUPPORTED_BITS:
        raise UnsupportedSchemeError(f"{name}: {FORMAT_NAME} export takes bits {SUPPORTED_BITS}, got {bits}.")
    tensors = {
        "weight_packed": layer.qweight.t().contiguous(),
        "weight_scale": layer.scale.detach().contiguous(),
        "weight_zero_point": _pack_along_outputs(integer_zero_points(name, layer, FORMAT_NAME), bits),
        "weight_shape": torch.tensor([layer.out_features, layer.in_features], dtype=torch.int64),
    }
    if layer.bias is not None:
        tensors["bias"] = layer.bias.detach()
    return tensors


def _pack_along_outputs(zero_point: torch.Tensor, bits: int) -> torch.Tensor:
    """(out, n_groups) -> (ceil(out * bits / 32), n_groups) int32. The outputs are padded with zeros
    to whole 32-value runs first, as compressed-tensors does, and the words past the last value are cut."""
    out_features = zero_point.shape[0]
    padded = torch.nn.functional.pad(zero_point, (0, 0, 0, -out_features % 32))
    n_words = math.ceil(out_features * bits / 32)
    return pack_codes(padded.t(), bits)[:n_words].contiguous()


def _quantization_config(layers: dict[str, QuantLinear]) -> dict:
    """One config group per (bits, group_size), each targeting its layers by full name: a name matches
    exactly in compressed-tensors and in vLLM (also for vLLM's fused qkv_proj / gate_up_proj, whose
    shards it looks up by name), so no other module is ever read as quantized."""
    grids: dict[tuple[int, int], list[str]] = {}
    for name, layer in layers.items():
        grids.setdefault((layer.config.bits, layer.config.group_size), []).append(name)
    config_groups = {
        f"group_{i}": {
            "targets": names,
            "weights": {
                "num_bits": bits,
                "type": "int",
                "symmetric": False,
                "strategy": "group",
                "group_size": group_size,
                "dynamic": False,
                "actorder": None,
            },
            "input_activations": None,
            "output_activations": None,
            "format": "pack-quantized",
        }
        for i, ((bits, group_size), names) in enumerate(sorted(grids.items()))
    }
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": config_groups,
        "ignore": [],
        "kv_cache_scheme": None,
    }
