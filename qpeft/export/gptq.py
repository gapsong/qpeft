"""Write a merged qpeft model as a GPTQ checkpoint, the format transformers (through optimum and
gptqmodel) loads with `AutoModelForCausalLM.from_pretrained`.

GPTQ dequantizes      w = (q - z) * s      per group, with an integer zero-point z,
a merged QuantLinear  w = (q - z) * s      too, with the codes already in GPTQ's qweight layout
(qpeft/packing.py). So the export only transposes, casts and packs scale and zero-point, and it
refuses every layer where that is not exact:
    - a fractional zero-point (a QA-LoRA fold): GPTQ stores integers, and rounding would break
      fake_quant == merge
    - a scale that fp16, GPTQ's scale dtype, does not hold exactly (in general an fp32 model;
      bf16 and fp16 scales convert exactly, because the scale is clamped to [1e-4, 1e4])
    - a quantized layer that a GPTQ loader would not find, or a linear it would wrongly replace

The zero-point convention is the checkpoint_format:
    "gptq"     (v1, what GPTQ tooling reads by default) stores z - 1. Loaders undo it in
               different ways: gptqmodel adds 1 to whole packed words for 2 / 4 / 8 bits (a carry
               runs into the next field) and to each field mod 2**bits for 3 bits. They agree
               only for z >= 1, so z == 0 is refused.
    "gptq_v2"  stores z as it is: exact for every z, but only gptqmodel reads it.
"""
from __future__ import annotations

import json
import os
import re

import torch
from torch import nn

from ..packing import pack_codes
from ..quant_schemes import UnsupportedSchemeError
from ..tuners.tuners_utils import QuantLinear

CHECKPOINT_FORMATS = ("gptq", "gptq_v2")
SUPPORTED_BITS = (2, 3, 4, 8)          # transformers' GPTQConfig accepts only these
QPEFT_KEYS = ("qweight", "scale", "zero_point", "bias")
# The first numbered component of a module name is its decoder block:
# "model.layers.3.mlp.experts.7.up_proj" -> ("model.layers", "3", "mlp.experts.7.up_proj").
BLOCK_NAME = re.compile(r"(.+?)\.(\d+)\.(.+)")


def export_gptq(model: nn.Module, save_directory, checkpoint_format: str = "gptq") -> None:
    """Write a merged QuantModel (or its Hugging Face base) to `save_directory` as a GPTQ checkpoint:
    the model's own files from `save_pretrained`, with GPTQ tensors for every QuantLinear and a
    `quantization_config` in config.json. Every layer is checked before anything is written.
    The tokenizer is not part of the model; save it next to it yourself."""
    if checkpoint_format not in CHECKPOINT_FORMATS:
        raise UnsupportedSchemeError(
            f"checkpoint_format must be one of {CHECKPOINT_FORMATS}, got {checkpoint_format!r}.")
    root = getattr(model, "base", model)
    if not (hasattr(root, "save_pretrained") and hasattr(root, "config")):
        raise UnsupportedSchemeError(
            f"GPTQ export writes a Hugging Face checkpoint; {type(root).__name__} is not a "
            "transformers PreTrainedModel.")
    layers = {name: m for name, m in root.named_modules() if isinstance(m, QuantLinear)}
    if not layers:
        raise ValueError("no QuantLinear found; nothing to export.")

    bits, group_size = _shared_grid(layers)
    block_name, modules_in_block = _block_layout(root, layers)
    gptq = {name: _gptq_tensors(name, layer, checkpoint_format) for name, layer in layers.items()}

    state = root.state_dict()
    for name, tensors in gptq.items():
        for key in QPEFT_KEYS:
            state.pop(f"{name}.{key}", None)
        state.update({f"{name}.{key}": value for key, value in tensors.items()})
    root.save_pretrained(save_directory, state_dict=state)

    zero_points = torch.cat([layer.zero_point.detach().flatten() for layer in layers.values()])
    quantization_config = {
        "quant_method": "gptq",
        "checkpoint_format": checkpoint_format,
        "bits": bits,
        "group_size": group_size,
        "desc_act": False,
        "sym": bool((zero_points == 2 ** (bits - 1)).all()),
        "block_name_to_quantize": block_name,
    }
    if modules_in_block is not None:
        quantization_config["modules_in_block_to_quantize"] = [modules_in_block]
    _add_to_config_json(save_directory, quantization_config)


def _shared_grid(layers: dict[str, QuantLinear]) -> tuple[int, int]:
    """quantization_config has one bits and one group_size for the whole model."""
    grids = {(layer.config.bits, layer.config.group_size) for layer in layers.values()}
    if len(grids) != 1:
        raise UnsupportedSchemeError(
            f"GPTQ stores one bits / group_size for the whole model; the layers use {sorted(grids)}.")
    return grids.pop()


def _block_layout(root: nn.Module, layers: dict[str, QuantLinear]) -> tuple[str, list[str] | None]:
    """The decoder blocks the quantized layers live in, and the block-relative names a GPTQ loader
    must quantize (None: every linear in a block). The loader replaces exactly the linears inside
    `block_name` whose name ends with one of these names; anything else it would leave in float
    (a quantized lm_head) or fill with uninitialized GPTQ tensors (an unquantized q_proj)."""
    placement = {name: BLOCK_NAME.fullmatch(name) for name in layers}
    outside = sorted(name for name, match in placement.items() if match is None)
    if outside:
        raise UnsupportedSchemeError(
            f"GPTQ loaders quantize only the linears inside the decoder blocks; {outside} are "
            "outside them. Quantize without them (target_modules), then export.")
    containers = {match[1] for match in placement.values()}
    if len(containers) != 1:
        raise UnsupportedSchemeError(
            f"GPTQ quantizes the layers of one stack of decoder blocks; found {sorted(containers)}.")
    block_name = containers.pop()

    blocks = root.get_submodule(block_name)
    per_block = {str(i): set() for i in range(len(blocks))}
    for match in placement.values():
        per_block[match[2]].add(match[3])
    modules = per_block["0"]
    if any(names != modules for names in per_block.values()):
        raise UnsupportedSchemeError(
            "GPTQ quantizes the same linears in every decoder block; this model quantizes "
            "different ones per block. Use the same target_modules for every block.")

    in_blocks = {f"{block_name}.{name}": m for name, m in blocks.named_modules() if _loader_replaces(m)}
    quantized = {name for name, m in in_blocks.items() if isinstance(m, QuantLinear)}
    if quantized == set(in_blocks):
        return block_name, None
    names = sorted(modules)
    # optimum's convert_model keeps a layer when its full name ends with one of the names, plain str.endswith.
    replaced = {name for name in in_blocks if any(name.endswith(n) for n in names)}
    if replaced != quantized:
        raise UnsupportedSchemeError(
            f"a GPTQ loader picks the quantized linears by name suffix ({names}) and would also "
            f"replace {sorted(replaced - quantized)}, which are not quantized. Refusing.")
    return block_name, names


def _loader_replaces(module: nn.Module) -> bool:
    """The layer types optimum's get_layers finds inside the blocks; a QuantLinear was an nn.Linear.
    Conv1D is transformers' (GPT-2), matched by name so that qpeft does not import transformers."""
    return (isinstance(module, (QuantLinear, nn.Conv2d)) or type(module) is nn.Linear
            or type(module).__name__ == "Conv1D")


def _gptq_tensors(name: str, layer: QuantLinear, checkpoint_format: str) -> dict[str, torch.Tensor]:
    """The tensors a GPTQ QuantLinear loads: qweight (in * bits // 32, out) int32,
    qzeros (n_groups, out * bits // 32) int32, scales (n_groups, out) fp16, g_idx (in,) int32."""
    if not layer.merged:
        raise UnsupportedSchemeError(f"{name}: GPTQ export needs a merged layer; call merge_and_unload() first.")
    bits, group_size = layer.config.bits, layer.config.group_size
    if bits not in SUPPORTED_BITS:
        raise UnsupportedSchemeError(f"{name}: GPTQ loaders take bits {SUPPORTED_BITS}, got {bits}.")
    if (layer.out_features * bits) % 32:
        raise UnsupportedSchemeError(
            f"{name}: GPTQ packs the zero-points along the outputs; out_features * bits must be a "
            f"multiple of 32 (out_features={layer.out_features}, bits={bits}).")

    stored_zeros = _stored_zero_points(name, layer.zero_point.detach(), bits, checkpoint_format)
    tensors = {
        "qweight": layer.qweight,
        "qzeros": pack_codes(stored_zeros.t(), bits).t().contiguous(),
        "scales": _exact_fp16_scale(name, layer.scale.detach()).t().contiguous(),
        "g_idx": torch.arange(layer.in_features, dtype=torch.int32, device=layer.qweight.device) // group_size,
    }
    if layer.bias is not None:
        tensors["bias"] = layer.bias.detach()
    return tensors


def _stored_zero_points(name: str, zero_point: torch.Tensor, bits: int, checkpoint_format: str) -> torch.Tensor:
    """The integer zero-points as the checkpoint stores them, (out, n_groups) int32."""
    if not torch.equal(zero_point, zero_point.round()):
        raise UnsupportedSchemeError(
            f"{name}: its zero-points are fractional (a QA-LoRA fold, z' = z - delta / s). GPTQ stores "
            "integer zero-points, and rounding them would break fake_quant == merge. Keep the qpeft "
            "artifact (save_pretrained) or run it on tinygemm (qpeft.kernels.to_tinygemm).")
    if checkpoint_format == "gptq_v2":
        return zero_point.to(torch.int32)
    if bool((zero_point == 0).any()):
        raise UnsupportedSchemeError(
            f"{name}: {int((zero_point == 0).sum())} groups have zero-point 0, which the v1 GPTQ "
            "format (it stores z - 1) cannot hold in a way all loaders read alike. "
            "Use checkpoint_format='gptq_v2' (exact for every zero-point; read by gptqmodel only).")
    return zero_point.to(torch.int32) - 1


def _exact_fp16_scale(name: str, scale: torch.Tensor) -> torch.Tensor:
    half = scale.to(torch.float16)
    changed = half.to(scale.dtype) != scale
    if bool(changed.any()):
        raise UnsupportedSchemeError(
            f"{name}: {int(changed.sum())} of {scale.numel()} scales change when cast to fp16, "
            f"GPTQ's scale dtype (the merged scales are {scale.dtype}). Quantize and merge the model "
            "in bf16 or fp16, whose scales fp16 holds exactly.")
    return half


def _add_to_config_json(save_directory, quantization_config: dict) -> None:
    path = os.path.join(save_directory, "config.json")
    with open(path) as f:
        config = json.load(f)
    config["quantization_config"] = quantization_config
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
