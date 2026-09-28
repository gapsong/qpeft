"""Half-precision training and the integer zero-point, as the official EfficientQAT
does them (quantize/quantizer.py, quantize/block_ap.py, main_e2e_qp.py).

Regression for a silent failure: with a bf16 model, an AdamW step of lr 1e-5 ..
2e-5 is below bf16's resolution, so Block-AP never moved the weights and E2E-QP
never moved the scale. Nothing crashed; the model just did not learn."""
import pytest
import torch
import torch.nn as nn

from qpeft import EfficientQATConfig, QALoraConfig, get_quant_model, param_groups, run_block_ap
from qpeft.tuners.tuners_utils import QuantLinear
from qpeft.utils import check_layer_merge_equivalence, verify_quant_model

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _llama_bf16(seed=0):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(seed)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, vocab_size=320, max_position_embeddings=64))
    return model.to(torch.bfloat16)


def _batches(n=8, seq=32, bs=2):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 320, (n, seq), generator=g)
    return [{"input_ids": ids[i:i + bs]} for i in range(0, n, bs)]


def _norms(model):
    return {n: p for n, p in model.named_parameters() if "norm" in n and n.endswith("weight")
            and ".layers." in n}


def test_block_ap_in_bf16_moves_weights_and_keeps_dtypes():
    """Official lrs on a bf16 model: the weights must move, and everything must
    come back in bf16 (scale/zero_point stay fp32 masters)."""
    model = get_quant_model(_llama_bf16(), EfficientQATConfig(bits=2, group_size=64,
                                                              target_modules=TARGETS))
    layers = [m for m in model.modules() if isinstance(m, QuantLinear)]
    w0 = [m.weight.detach().clone() for m in layers]

    run_block_ap(model, _batches(), epochs=2, weight_lr=2e-5, quant_lr=1e-4, log=lambda *_: None)

    for m, w in zip(layers, w0):
        assert m.weight.dtype == torch.bfloat16
        assert m.scale.dtype == torch.float32 and m.zero_point.dtype == torch.float32
        assert not torch.equal(m.weight, w), "a bf16 weight did not move in Block-AP"
    for n, p in _norms(model).items():
        assert p.dtype == torch.bfloat16 and not p.requires_grad, n


def test_block_ap_trains_norms_and_reduces_the_mean_reconstruction_error():
    """The block norms train with weight_lr (the official filter is "weight" in the
    name), and the log reports the mean MSE over all batches before -> after."""
    model = get_quant_model(_llama_bf16().float(), EfficientQATConfig(bits=2, group_size=64,
                                                                      target_modules=TARGETS))
    n0 = {n: p.detach().clone() for n, p in _norms(model).items()}
    logs = []
    run_block_ap(model, _batches(), epochs=4, weight_lr=1e-3, quant_lr=1e-3, log=logs.append)
    assert n0 and all(not torch.equal(p, n0[n]) for n, p in _norms(model).items()), \
        "a block norm was not trained"
    assert all(not p.requires_grad for p in _norms(model).values())
    assert len(logs) == 2 and all("mean mse" in line for line in logs)
    for line in logs:
        before, after = (float(v) for v in line.rsplit(" ", 3)[-3::2])
        assert after < before, line


def test_e2e_qp_in_bf16_moves_the_scale():
    model = get_quant_model(_llama_bf16(), EfficientQATConfig(bits=2, group_size=64, phase="e2e_qp",
                                                              target_modules=TARGETS))
    layers = [m for m in model.modules() if isinstance(m, QuantLinear)]
    s0 = [m.scale.detach().clone() for m in layers]
    opt = torch.optim.AdamW(param_groups(model, weight_lr=0.0, quant_lr=2e-5, adapter_lr=0.0))
    ids = _batches()[0]["input_ids"]
    model(input_ids=ids, labels=ids).loss.backward()
    opt.step()
    assert all(not torch.equal(m.scale, s) for m, s in zip(layers, s0)), \
        "an E2E-QP scale did not move at the official lr in a bf16 model"
    verify_quant_model(model)


def test_qa_lora_adapter_is_fp32_and_moves_in_a_bf16_model():
    base = nn.Sequential(nn.Linear(128, 64)).to(torch.bfloat16)
    model = get_quant_model(base, QALoraConfig(bits=4, group_size=32, r=8))
    layer = next(m for m in model.modules() if isinstance(m, QuantLinear))
    assert layer.adapter.A.dtype == torch.float32 and layer.adapter.B.dtype == torch.float32
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    y = model(x)
    assert y.dtype == torch.bfloat16
    opt = torch.optim.AdamW(param_groups(model, weight_lr=0.0, quant_lr=0.0, adapter_lr=2e-5))
    y.float().pow(2).mean().backward()
    opt.step()
    assert layer.adapter.B.abs().max() > 0, "the bf16 model's adapter did not move"


def test_zero_point_is_an_integer_in_training_and_in_the_artifact():
    """int_uniform has an integer zero-point domain (official: round_ste + clamp).
    The parameter may drift while training, but what training uses and what the
    artifact stores is round(z) in [0, qmax]."""
    torch.manual_seed(0)
    model = get_quant_model(nn.Sequential(nn.Linear(128, 64)),
                            EfficientQATConfig(bits=3, group_size=32))
    layer = next(m for m in model.modules() if isinstance(m, QuantLinear))
    assert torch.equal(layer.zero_point, layer.zero_point.round()), "RTN init z is not integer"
    with torch.no_grad():                   # a trained z drifts off the integers
        layer.zero_point.add_(torch.rand_like(layer.zero_point) - 0.5)
    x = torch.randn(8, 128)
    with torch.no_grad():
        before = model(x)
    model.merge_and_unload()
    z = layer.zero_point
    assert torch.equal(z, z.round()) and z.min() >= 0 and z.max() <= 2 ** 3 - 1
    with torch.no_grad():
        assert torch.equal(before, model(x)), "merge != fake_quant with a drifted zero-point"


def test_frozen_codes_forward_rounds_the_zero_point_like_the_freeze():
    """An fp32 model under bf16 inputs (autocast): after Block-AP the zero-point is fractional.
    freeze_codes rounds the fp32 master; the forward must use that same integer.
    z = 7.49 rounds to 7 in fp32, but bf16(7.49) = 7.5 would round to 8, one full quant step off."""
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(128, 64))
    model = get_quant_model(base, EfficientQATConfig(bits=4, group_size=32, phase="block_ap"))
    layer = next(m for m in model.modules() if isinstance(m, QuantLinear))
    with torch.no_grad():
        layer.zero_point.fill_(7.49)                        # as Block-AP can leave it
    model = get_quant_model(base, EfficientQATConfig(bits=4, group_size=32, phase="e2e_qp"))  # freezes

    x = torch.randn(8, 128, dtype=torch.bfloat16)
    s = layer.scale.detach().to(torch.bfloat16)
    expected = layer.scheme.dequant(layer.codes, s, torch.full_like(s, 7.0))
    with torch.no_grad():
        got = layer(x)
    assert torch.equal(got, torch.nn.functional.linear(x, expected, layer.bias.to(x.dtype))), \
        "the bf16 forward rounded the zero-point to another integer than the frozen codes"


def test_bf16_merge_rounds_the_zero_point_like_the_freeze():
    """A bf16 model: freeze_codes and the forward round the fp32 master z = 7.49 to 7.
    The merge must use the same 7, not round(bf16 7.5) = 8."""
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(128, 64)).to(torch.bfloat16)
    model = get_quant_model(base, EfficientQATConfig(bits=4, group_size=32, phase="block_ap"))
    layer = next(m for m in model.modules() if isinstance(m, QuantLinear))
    with torch.no_grad():
        layer.zero_point.fill_(7.49)                        # as Block-AP can leave it
    model = get_quant_model(base, EfficientQATConfig(bits=4, group_size=32, phase="e2e_qp"))  # freezes

    check_layer_merge_equivalence(layer)
    layer.merge()
    assert torch.equal(layer.zero_point, torch.full_like(layer.zero_point, 7.0))

