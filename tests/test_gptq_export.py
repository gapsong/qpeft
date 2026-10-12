"""GPTQ export (qpeft/export/gptq.py) on a tiny random Llama, no download.

Oracle: the GPTQ dequant formula, applied to the tensors in the written file and decoded here
independently of qpeft.packing:
    W[i, o] = (code[i, o] - zero[g, o]) * scale[g, o]    with g = g_idx[i],
    zero = stored + 1 for checkpoint_format "gptq" (v1), stored for "gptq_v2".
It must give the merged QuantLinear's weight and forward bit for bit.
That a real loader reads the file the same way is tests/test_gptq_load.py."""
import json

import pytest

pytest.importorskip("transformers")

import torch                                                            # noqa: E402
import torch.nn.functional as F                                         # noqa: E402
from safetensors.torch import load_file                                 # noqa: E402

from qpeft import (                                                     # noqa: E402
    EfficientQATConfig, PEQAConfig, QALoraConfig, UnsupportedSchemeError, get_quant_model, run_block_ap,
)
from transformers import LlamaConfig, LlamaForCausalLM                  # noqa: E402

from qpeft.export import export_gptq                                    # noqa: E402
from qpeft.tuners.tuners_utils import QuantLinear                       # noqa: E402

BLOCK_LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
VOCAB = 320


class _Router(torch.nn.Linear):
    """A subclass of nn.Linear, like an MoE router with its own forward."""


class _LlamaWithRouters(LlamaForCausalLM):
    """Every decoder block also holds a _Router, so the model a loader builds from config has it too."""
    def __init__(self, config):
        super().__init__(config)
        for block in self.model.layers:
            block.router = _Router(config.hidden_size, 8)


class _LlamaWithMyselfAttn(LlamaForCausalLM):
    """Every decoder block also holds a plain linear "myself_attn.o_proj"."""
    def __init__(self, config):
        super().__init__(config)
        for block in self.model.layers:
            block.myself_attn = torch.nn.Module()
            block.myself_attn.o_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)


def _llama(dtype, tie=False, model_class=LlamaForCausalLM):
    torch.manual_seed(0)
    return model_class(LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, vocab_size=VOCAB, max_position_embeddings=64,
        tie_word_embeddings=tie)).to(dtype).eval()


def _batches(n=2, seq=16):
    g = torch.Generator().manual_seed(0)
    return [{"input_ids": torch.randint(0, VOCAB, (2, seq), generator=g)} for _ in range(n)]


def _train_scales(model, steps=3):
    """A few optimizer steps on the scales, so the export sees trained, not RTN, scales."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for batch in _batches(steps):
        opt.zero_grad()
        model(input_ids=batch["input_ids"], labels=batch["input_ids"]).loss.backward()
        opt.step()


def _merged_peqa(bits, dtype=torch.bfloat16, targets=BLOCK_LINEARS, tie=False):
    model = get_quant_model(_llama(dtype, tie), PEQAConfig(bits=bits, group_size=64, target_modules=targets))
    _train_scales(model)
    model.merge_and_unload()
    return model


def _merged_efficient_qat(bits, dtype=torch.bfloat16):
    """Block-AP trains weight, scale and zero-point, so the zero-points are trained too."""
    block_ap, e2e_qp = (EfficientQATConfig(bits=bits, group_size=64, phase=phase, target_modules=BLOCK_LINEARS)
                        for phase in ("block_ap", "e2e_qp"))
    model = get_quant_model(_llama(dtype), block_ap)
    run_block_ap(model, _batches(), epochs=1, weight_lr=1e-3, quant_lr=1e-3, log=lambda _: None)
    model = get_quant_model(model, e2e_qp)
    _train_scales(model)
    model.merge_and_unload()
    return model


def _unpack_bit_stream(words, bits, n):
    """(n_words, cols) int32 -> (n, cols): value k is bits [k * bits, (k + 1) * bits) of each
    column's bit stream, lowest bit first (GPTQ's layout, also its 3-bit word spill)."""
    bit = (words.to(torch.int64)[:, None, :] >> torch.arange(32)[None, :, None]) & 1
    stream = bit.reshape(-1, words.shape[1])[: n * bits]
    return (stream.reshape(n, bits, -1) << torch.arange(bits)[None, :, None]).sum(1)


def _gptq_weight(tensors, prefix, bits, checkpoint_format, dtype):
    """The (out, in) weight a GPTQ loader computes from the written tensors."""
    qweight, qzeros = tensors[f"{prefix}.qweight"], tensors[f"{prefix}.qzeros"]
    scales, g_idx = tensors[f"{prefix}.scales"], tensors[f"{prefix}.g_idx"].long()
    in_features, out_features = qweight.shape[0] * 32 // bits, scales.shape[1]
    codes = _unpack_bit_stream(qweight, bits, in_features)                       # (in, out)
    zeros = _unpack_bit_stream(qzeros.t(), bits, out_features).t()               # (n_groups, out)
    if checkpoint_format == "gptq":
        zeros = zeros + 1
    weight = (codes - zeros[g_idx]).to(dtype) * scales.to(dtype)[g_idx]
    return weight.t().contiguous()


def _export(model, path, checkpoint_format="gptq"):
    export_gptq(model, path, checkpoint_format=checkpoint_format)
    with open(path / "config.json") as f:
        return load_file(path / "model.safetensors"), json.load(f)["quantization_config"]


def _quant_layers(model):
    return {name: m for name, m in model.base.named_modules() if isinstance(m, QuantLinear)}


# --- 1. the written tensors equal the merged model --------------------------------------------

@pytest.mark.parametrize("checkpoint_format", ["gptq", "gptq_v2"])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("method", ["peqa", "efficient_qat"])
def test_gptq_dequant_of_the_file_equals_the_merged_layers(method, bits, checkpoint_format, tmp_path):
    model = _merged_peqa(bits) if method == "peqa" else _merged_efficient_qat(bits)
    tensors, config = _export(model, tmp_path, checkpoint_format)
    assert config["bits"] == bits and config["checkpoint_format"] == checkpoint_format

    x = torch.randn(3, 5, 256).to(torch.bfloat16)
    for name, layer in _quant_layers(model).items():
        weight = _gptq_weight(tensors, name, bits, checkpoint_format, layer.compute_dtype)
        assert torch.equal(weight, layer._weight_in(layer.compute_dtype)), name
        layer_x = x[..., :layer.in_features]
        with torch.no_grad():
            assert torch.equal(F.linear(layer_x, weight), layer(layer_x)), name


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tensor_layout_and_dtypes(dtype, tmp_path):
    model = _merged_peqa(4, dtype)
    tensors, _ = _export(model, tmp_path)
    layer = model.base.model.layers[0].mlp.down_proj                     # in 256, out 128
    prefix = "model.layers.0.mlp.down_proj"
    assert torch.equal(tensors[f"{prefix}.qweight"], layer.qweight)
    assert tensors[f"{prefix}.qzeros"].shape == (4, 128 * 4 // 32) and tensors[f"{prefix}.qzeros"].dtype == torch.int32
    assert tensors[f"{prefix}.scales"].shape == (4, 128) and tensors[f"{prefix}.scales"].dtype == torch.float16
    assert torch.equal(tensors[f"{prefix}.g_idx"], torch.arange(256, dtype=torch.int32) // 64)
    assert not any(k.endswith((".scale", ".zero_point")) for k in tensors)


def test_quantization_config(tmp_path):
    _, config = _export(_merged_peqa(4), tmp_path)
    assert config == {"quant_method": "gptq", "checkpoint_format": "gptq", "bits": 4, "group_size": 64,
                      "desc_act": False, "sym": False, "block_name_to_quantize": "model.layers"}


def test_sym_only_when_every_zero_point_is_the_midpoint(tmp_path):
    model = _merged_peqa(4)
    for layer in _quant_layers(model).values():
        layer.zero_point.data.fill_(8)
    _, config = _export(model, tmp_path)
    assert config["sym"] is True


def test_a_subset_of_the_block_linears_names_them_for_the_loader(tmp_path):
    model = _merged_peqa(4, targets=["q_proj", "v_proj"])
    tensors, config = _export(model, tmp_path)
    assert config["modules_in_block_to_quantize"] == [["self_attn.q_proj", "self_attn.v_proj"]]
    assert "model.layers.0.self_attn.k_proj.weight" in tensors                  # stays float
    assert "model.layers.0.self_attn.q_proj.qweight" in tensors


def test_a_subclass_of_linear_in_the_blocks_is_not_one_the_loader_replaces(tmp_path):
    """optimum's get_layers takes nn.Linear by exact type, so it leaves a subclass (an MoE router)
    in float; quantizing every plain block linear is then the whole block, with no name list."""
    base = _llama(torch.bfloat16, model_class=_LlamaWithRouters)
    model = get_quant_model(base, PEQAConfig(bits=4, group_size=64, target_modules=BLOCK_LINEARS))
    model.merge_and_unload()
    assert type(base.model.layers[0].router) is _Router
    tensors, config = _export(model, tmp_path)
    assert "modules_in_block_to_quantize" not in config
    assert "model.layers.0.router.weight" in tensors                            # stays float


def test_tied_embeddings_are_written_once(tmp_path):
    tensors, _ = _export(_merged_peqa(4, tie=True), tmp_path)
    assert "model.embed_tokens.weight" in tensors and "lm_head.weight" not in tensors


def test_export_leaves_the_model_as_it_was(tmp_path):
    model = _merged_peqa(4)
    ids = _batches(1)[0]["input_ids"]
    with torch.no_grad():
        before = model(input_ids=ids).logits
    keys = set(model.state_dict())
    _export(model, tmp_path)
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, before)
    assert set(model.state_dict()) == keys


# --- 2. refusals: nothing is written --------------------------------------------------------

def _refused(model, tmp_path, match, checkpoint_format="gptq"):
    out = tmp_path / "out"
    with pytest.raises(UnsupportedSchemeError, match=match):
        export_gptq(model, out, checkpoint_format=checkpoint_format)
    assert not out.exists()


def test_refuses_an_unmerged_model(tmp_path):
    model = get_quant_model(_llama(torch.bfloat16), PEQAConfig(bits=4, group_size=64, target_modules=BLOCK_LINEARS))
    _refused(model, tmp_path, "merge_and_unload")


def test_refuses_qa_lora_fractional_zero_points(tmp_path):
    """Decision D2: a QA-LoRA fold makes z' = z - delta / s fractional; GPTQ has integer zeros."""
    model = get_quant_model(_llama(torch.bfloat16), QALoraConfig(bits=4, group_size=64, r=4,
                                                                 target_modules=BLOCK_LINEARS))
    for layer in _quant_layers(model).values():
        torch.nn.init.normal_(layer.adapter.B, std=0.1)
    model.merge_and_unload()
    _refused(model, tmp_path, "fractional")


def test_refuses_fp32_scales_that_fp16_does_not_hold(tmp_path):
    _refused(_merged_peqa(4, torch.float32), tmp_path, "change when cast to fp16")


def test_zero_point_zero_is_refused_in_v1_and_exact_in_v2(tmp_path):
    model = _merged_peqa(4)
    layer = model.base.model.layers[1].self_attn.o_proj
    layer.zero_point.data[3, 0] = 0
    _refused(model, tmp_path, "zero-point 0", checkpoint_format="gptq")
    tensors, _ = _export(model, tmp_path / "v2", checkpoint_format="gptq_v2")
    weight = _gptq_weight(tensors, "model.layers.1.self_attn.o_proj", 4, "gptq_v2", torch.bfloat16)
    assert torch.equal(weight, layer._weight_in(torch.bfloat16))


def test_refuses_a_quantized_layer_outside_the_blocks(tmp_path):
    _refused(_merged_peqa(4, targets=None), tmp_path, "lm_head")


def test_refuses_different_linears_per_block(tmp_path):
    _refused(_merged_peqa(4, targets=r"model\.layers\.0\..*_proj"), tmp_path, "every decoder block")


def test_refuses_when_the_loader_suffix_match_would_hit_an_unquantized_linear(tmp_path):
    """optimum keeps a block linear when its full name str.endswith a quantized name, without a dot
    boundary: "model.layers.0.myself_attn.o_proj" ends with "self_attn.o_proj"."""
    base = _llama(torch.bfloat16, model_class=_LlamaWithMyselfAttn)
    model = get_quant_model(base, PEQAConfig(bits=4, group_size=64, target_modules=["self_attn.o_proj"]))
    model.merge_and_unload()
    assert type(base.model.layers[0].myself_attn.o_proj) is torch.nn.Linear
    _refused(model, tmp_path, "myself_attn.o_proj")


def test_refuses_a_quantized_linear_subclass_the_loader_keeps_in_float(tmp_path):
    """optimum's get_layers takes nn.Linear by exact type, so a loader would keep the router in
    float and random-initialize it: the checkpoint holds only its GPTQ tensors."""
    model = get_quant_model(_llama(torch.bfloat16, model_class=_LlamaWithRouters),
                            PEQAConfig(bits=4, group_size=64, target_modules=BLOCK_LINEARS + ["router"]))
    model.merge_and_unload()
    assert isinstance(model.base.model.layers[0].router, QuantLinear)
    _refused(model, tmp_path, "router.*exact type")


def test_refuses_bits_a_gptq_loader_does_not_take(tmp_path):
    _refused(_merged_peqa(5, targets=["down_proj"]), tmp_path, "bits")


def test_refuses_a_model_that_is_not_a_hugging_face_model(tmp_path):
    model = get_quant_model(torch.nn.Sequential(torch.nn.Linear(128, 64).to(torch.bfloat16)),
                            PEQAConfig(bits=4, group_size=64, target_modules=["0"]))
    model.merge_and_unload()
    _refused(model, tmp_path, "Hugging Face")


def test_refuses_an_unknown_checkpoint_format(tmp_path):
    _refused(_merged_peqa(4), tmp_path, "checkpoint_format", checkpoint_format="awq")
