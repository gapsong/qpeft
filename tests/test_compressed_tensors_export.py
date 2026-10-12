"""compressed-tensors export (qpeft/export/compressed_tensors.py) on a tiny random Llama, no download.

Oracle: compressed-tensors' pack-quantized dequant, applied to the tensors in the written file and
decoded here independently of qpeft.packing:
    q, z    the unsigned values in weight_packed / weight_zero_point (one dense bit stream per row),
            read as the signed q - 2**(bits-1), z - 2**(bits-1) that compressed-tensors unpacks to
    W       (q_signed.to(scale.dtype) - z_signed.to(scale.dtype)) * scale, per group of inputs
It must give the merged QuantLinear's weight and forward bit for bit.
That compressed-tensors itself and transformers read the file the same way is
tests/test_compressed_tensors_load.py."""
import json

import pytest

pytest.importorskip("transformers")

import torch                                                            # noqa: E402
import torch.nn.functional as F                                         # noqa: E402
from safetensors.torch import load_file                                 # noqa: E402

from export_models import (                                             # noqa: E402
    BLOCK_LINEARS, LlamaWithRouters, batches, llama, merged_efficient_qat, merged_peqa, quant_layers,
    unpack_bit_stream,
)
from qpeft import PEQAConfig, QALoraConfig, UnsupportedSchemeError, get_quant_model     # noqa: E402
from qpeft.export import export_compressed_tensors                      # noqa: E402
from qpeft.export.compressed_tensors import _pack_along_outputs         # noqa: E402
from qpeft.quant_schemes.registry import build_scheme                   # noqa: E402
from qpeft.tuners.tuners_utils import QuantLinear                       # noqa: E402

ALL_BITS = [2, 3, 4, 5, 6, 7, 8]


def ct_weight(tensors, prefix, group_size):
    """The (out, in) weight compressed-tensors computes from the written tensors."""
    out_features, in_features = tensors[f"{prefix}.weight_shape"].tolist()
    packed, scale = tensors[f"{prefix}.weight_packed"], tensors[f"{prefix}.weight_scale"]
    bits = packed.shape[1] * 32 // in_features
    offset = 1 << (bits - 1)
    codes = unpack_bit_stream(packed.t(), bits, in_features).t() - offset                    # (out, in)
    zeros = unpack_bit_stream(tensors[f"{prefix}.weight_zero_point"], bits, out_features) - offset
    grouped = codes.unflatten(-1, (in_features // group_size, group_size)).to(scale.dtype)
    weight = (grouped - zeros.to(scale.dtype)[..., None]) * scale[..., None]
    return weight.flatten(-2).contiguous()           # a strided weight takes another matmul path


def _export(model, path):
    export_compressed_tensors(model, path)
    with open(path / "config.json") as f:
        return load_file(path / "model.safetensors"), json.load(f)["quantization_config"]


def _assert_file_equals_merged_layers(model, tensors, group_size=64):
    x = torch.randn(3, 5, 256, generator=torch.Generator().manual_seed(0))
    for name, layer in quant_layers(model).items():
        dtype = layer.compute_dtype
        weight = ct_weight(tensors, name, group_size)
        assert weight.dtype == dtype
        assert torch.equal(weight, layer._weight_in(dtype)), name
        layer_x = x[..., :layer.in_features].to(dtype)
        with torch.no_grad():
            assert torch.equal(F.linear(layer_x, weight, layer.bias), layer(layer_x)), name


# --- 1. the written tensors equal the merged model --------------------------------------------

@pytest.mark.parametrize("bits", ALL_BITS)
@pytest.mark.parametrize("method", ["peqa", "efficient_qat"])
def test_ct_dequant_of_the_file_equals_the_merged_layers(method, bits, tmp_path):
    model = merged_peqa(bits) if method == "peqa" else merged_efficient_qat(bits)
    tensors, config = _export(model, tmp_path)
    assert config["config_groups"]["group_0"]["weights"]["num_bits"] == bits
    _assert_file_equals_merged_layers(model, tensors)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_tensor_layout_and_dtypes(dtype, tmp_path):
    model = merged_peqa(4, dtype)
    tensors, _ = _export(model, tmp_path)
    layer = model.base.model.layers[0].mlp.down_proj                     # in 256, out 128
    prefix = "model.layers.0.mlp.down_proj"
    assert torch.equal(tensors[f"{prefix}.weight_packed"], layer.qweight.t())
    assert tensors[f"{prefix}.weight_packed"].shape == (128, 256 * 4 // 32)
    assert torch.equal(tensors[f"{prefix}.weight_scale"], layer.scale)
    assert tensors[f"{prefix}.weight_scale"].dtype == dtype
    assert tensors[f"{prefix}.weight_zero_point"].shape == (128 * 4 // 32, 4)
    assert tensors[f"{prefix}.weight_zero_point"].dtype == torch.int32
    assert torch.equal(tensors[f"{prefix}.weight_shape"], torch.tensor([128, 256]))
    assert not any(k.endswith((".qweight", ".scale", ".zero_point")) for k in tensors)
    _assert_file_equals_merged_layers(model, tensors)


def test_quantization_config(tmp_path):
    model = merged_peqa(4)
    _, config = _export(model, tmp_path)
    assert config == {
        "quant_method": "compressed-tensors", "format": "pack-quantized", "quantization_status": "compressed",
        "ignore": [], "kv_cache_scheme": None,
        "config_groups": {"group_0": {
            "targets": list(quant_layers(model)),
            "weights": {"num_bits": 4, "type": "int", "symmetric": False, "strategy": "group",
                        "group_size": 64, "dynamic": False, "actorder": None},
            "input_activations": None, "output_activations": None, "format": "pack-quantized"}},
    }


def test_one_config_group_per_bits_and_group_size(tmp_path):
    """get_quant_model takes one config, so the 8-bit layers are injected by hand."""
    model = get_quant_model(llama(torch.bfloat16), PEQAConfig(bits=4, group_size=64,
                                                              target_modules=["q_proj", "k_proj", "v_proj"]))
    eight_bit = PEQAConfig(bits=8, group_size=128)
    for block in model.base.model.layers:
        block.mlp.up_proj = QuantLinear(block.mlp.up_proj, build_scheme(eight_bit), eight_bit)
        block.mlp.down_proj = QuantLinear(block.mlp.down_proj, build_scheme(eight_bit), eight_bit)
    model.merge_and_unload()
    tensors, config = _export(model, tmp_path)
    groups = config["config_groups"]
    assert [(g["weights"]["num_bits"], g["weights"]["group_size"]) for g in groups.values()] == [(4, 64), (8, 128)]
    assert groups["group_0"]["targets"] == [f"model.layers.{i}.self_attn.{p}_proj" for i in (0, 1) for p in "qkv"]
    assert groups["group_1"]["targets"] == [f"model.layers.{i}.mlp.{p}_proj" for i in (0, 1) for p in ("up", "down")]
    for group in groups.values():
        for name in group["targets"]:
            layer = model.base.get_submodule(name)
            weight = ct_weight(tensors, name, group["weights"]["group_size"])
            assert torch.equal(weight, layer._weight_in(torch.bfloat16)), name


def test_zero_points_on_the_edges_of_the_grid_are_exact(tmp_path):
    model = merged_peqa(4)
    layer = model.base.model.layers[1].self_attn.o_proj
    layer.zero_point.data[3, 0], layer.zero_point.data[4, 1] = 0, 15
    tensors, _ = _export(model, tmp_path)
    weight = ct_weight(tensors, "model.layers.1.self_attn.o_proj", 64)
    assert torch.equal(weight, layer._weight_in(torch.bfloat16))


@pytest.mark.parametrize("bits", [3, 5, 8])
def test_zero_points_pad_the_outputs_to_whole_words(bits):
    """20 outputs do not fill whole int32 words at these bits; the tail words are zero-padded."""
    zero_point = torch.randint(0, 1 << bits, (20, 3), generator=torch.Generator().manual_seed(bits), dtype=torch.int32)
    packed = _pack_along_outputs(zero_point, bits)
    assert packed.shape == (-(-20 * bits // 32), 3)
    assert torch.equal(unpack_bit_stream(packed, bits, 20), zero_point)


def test_an_untied_lm_head_is_exported(tmp_path):
    model = merged_peqa(4, targets=None)
    tensors, _ = _export(model, tmp_path)
    assert "lm_head.weight_packed" in tensors and "lm_head.weight" not in tensors
    _assert_file_equals_merged_layers(model, tensors)


def test_export_leaves_the_model_as_it_was(tmp_path):
    model = merged_peqa(4)
    ids = batches(1)[0]["input_ids"]
    with torch.no_grad():
        before = model(input_ids=ids).logits
    keys = set(model.state_dict())
    _export(model, tmp_path)
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, before)
    assert set(model.state_dict()) == keys


# --- 2. refusals: nothing is written --------------------------------------------------------

def _refused(model, tmp_path, match):
    out = tmp_path / "out"
    with pytest.raises(UnsupportedSchemeError, match=match):
        export_compressed_tensors(model, out)
    assert not out.exists()


def test_refuses_an_unmerged_model(tmp_path):
    model = get_quant_model(llama(torch.bfloat16), PEQAConfig(bits=4, group_size=64, target_modules=BLOCK_LINEARS))
    _refused(model, tmp_path, "merge_and_unload")


def test_refuses_qa_lora_fractional_zero_points(tmp_path):
    """Decision D2: a QA-LoRA fold makes z' = z - delta / s fractional; the format has integer zeros."""
    model = get_quant_model(llama(torch.bfloat16), QALoraConfig(bits=4, group_size=64, r=4,
                                                                target_modules=BLOCK_LINEARS))
    for layer in quant_layers(model).values():
        torch.nn.init.normal_(layer.adapter.B, std=0.1)
    model.merge_and_unload()
    _refused(model, tmp_path, "fractional")


def test_refuses_a_quantized_lm_head_tied_to_the_embedding(tmp_path):
    _refused(merged_peqa(4, targets=None, tie=True), tmp_path, r"lm_head.*ties its weight.*embed_tokens")


def test_refuses_a_quantized_linear_subclass(tmp_path):
    model = get_quant_model(llama(torch.bfloat16, model_class=LlamaWithRouters),
                            PEQAConfig(bits=4, group_size=64, target_modules=BLOCK_LINEARS + ["router"]))
    model.merge_and_unload()
    _refused(model, tmp_path, "router.*Router, not a plain nn.Linear")


def test_refuses_bits_outside_what_vllm_takes(tmp_path):
    _refused(merged_peqa(1, targets=["down_proj"]), tmp_path, "bits")


def test_refuses_a_model_that_is_not_a_hugging_face_model(tmp_path):
    model = get_quant_model(torch.nn.Sequential(torch.nn.Linear(128, 64).to(torch.bfloat16)),
                            PEQAConfig(bits=4, group_size=64, target_modules=["0"]))
    model.merge_and_unload()
    _refused(model, tmp_path, "Hugging Face")
