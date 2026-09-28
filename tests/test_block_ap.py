"""run_block_ap (qpeft/block_ap.py): EfficientQAT phase 1 on a tiny random Llama
(no download). The oracle is the official Block-AP contract: per block, the
quantized block learns to reproduce the fp block, and nothing outside the
blocks' weights / scale / zero_point / norms may change."""
import math

import pytest

pytest.importorskip("transformers")

import torch                                                    # noqa: E402

from qpeft import EfficientQATConfig, get_quant_model, run_block_ap  # noqa: E402
from qpeft.block_ap import _cosine, find_blocks                 # noqa: E402
from qpeft.tuners.tuners_utils import QuantLinear               # noqa: E402

BLOCK_LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _llama(dtype=torch.float32):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, vocab_size=320, max_position_embeddings=64)).to(dtype)


def _batches(n=4, seq=32):
    g = torch.Generator().manual_seed(0)
    return [{"input_ids": torch.randint(0, 320, (2, seq), generator=g)} for _ in range(n)]


def _qmodel(dtype=torch.float32, phase="block_ap"):
    return get_quant_model(_llama(dtype), EfficientQATConfig(
        bits=2, group_size=64, phase=phase, target_modules=BLOCK_LINEARS))


def _run(model, **kw):
    logs = []
    run_block_ap(model, _batches(), epochs=2, weight_lr=1e-3, quant_lr=1e-3, log=logs.append, **kw)
    return logs


def _mse_pairs(logs):
    pairs = []
    for line in logs:
        before, after = line.split("mean mse ")[1].split(" -> ")
        pairs.append((float(before), float(after)))
    return pairs


def test_every_block_reconstructs_better_after_training():
    model = _qmodel()
    pairs = _mse_pairs(_run(model))
    assert len(pairs) == len(find_blocks(model.base))
    for i, (before, after) in enumerate(pairs):
        assert after < before, f"block {i}: mse {before:.3e} -> {after:.3e} did not improve"


def test_only_block_weights_qparams_and_norms_change():
    model = _qmodel()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _run(model)
    after = dict(model.named_parameters())

    allowed_suffixes = (".weight", ".scale", ".zero_point")
    for name, p0 in before.items():
        changed = not torch.equal(p0, after[name])
        in_block = ".layers." in name
        if not in_block:
            assert not changed, f"{name} is outside the blocks but changed"
        elif changed:
            assert name.endswith(allowed_suffixes), f"{name} changed but is not trainable in Block-AP"
    for layer in (m for m in model.modules() if isinstance(m, QuantLinear)):
        assert layer.bias is None or not layer.bias.requires_grad
    norms = [n for n in before if ".layers." in n and "norm" in n]
    assert norms and any(not torch.equal(before[n], after[n]) for n in norms), \
        "the block norms should train with weight_lr, as in the official code"


def test_linears_that_are_not_quantized_stay_frozen():
    """With target_modules a subset, the other nn.Linear layers of a block stay full precision
    and frozen: Block-AP trains only QuantLinear parameters and the norms."""
    model = get_quant_model(_llama(), EfficientQATConfig(bits=2, group_size=64, target_modules=["q_proj"]))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _run(model)
    after = dict(model.named_parameters())

    untargeted = [n for n in before if ".layers." in n and "proj.weight" in n and "q_proj" not in n]
    assert untargeted
    for name in untargeted:
        assert torch.equal(before[name], after[name]), f"{name} is not quantized but Block-AP trained it"
    norms = [n for n in before if ".layers." in n and "norm" in n]
    assert any(not torch.equal(before[n], after[n]) for n in norms), "the norms should still train"


def test_dtypes_and_train_mode_are_restored():
    model = _qmodel(torch.bfloat16)
    model.train()
    dtypes = {n: p.dtype for n, p in model.named_parameters()}
    grads = {n: p.requires_grad for n, p in model.named_parameters()}
    _run(model)
    assert {n: p.dtype for n, p in model.named_parameters()} == dtypes
    assert {n: p.requires_grad for n, p in model.named_parameters()} == grads
    assert model.training


def test_refuses_a_model_that_is_not_in_block_ap_phase():
    with pytest.raises(ValueError, match="phase 'block_ap'"):
        _run(_qmodel(phase="e2e_qp"))


def test_refuses_without_calibration_batches():
    with pytest.raises(ValueError, match="no calibration batches"):
        run_block_ap(_qmodel(), [], log=lambda _: None)


def test_cosine_goes_from_lr_down_to_lr_over_min_lr_factor():
    assert _cosine(0, 100, 20) == pytest.approx(1.0)
    assert _cosine(50, 100, 20) == pytest.approx((1 + 1 / 20) / 2)
    assert _cosine(100, 100, 20) == pytest.approx(1 / 20)
    assert _cosine(150, 100, 20) == pytest.approx(1 / 20)      # clamped, never below the floor
    assert all(_cosine(s, 100, 20) >= _cosine(s + 1, 100, 20) for s in range(100))
    assert not math.isnan(_cosine(0, 0, 20))
