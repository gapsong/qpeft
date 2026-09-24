"""QA-LoRA end to end on a transformer: train -> merge must be LOSSLESS and the
integer weights must be exactly the ones the original model quantizes to.

Two tiers, same checks:
  * tiny random Llama   -- no download, runs whenever `transformers` is installed
  * real model (SOTA)   -- opt-in:  QPEFT_RUN_SLOW=1 pytest tests/test_qa_lora_real_model.py
                           model id: QPEFT_TEST_MODEL (default Qwen/Qwen3-0.6B)

Oracles (independent of the code under test):
  1. codes:  RTN-quantizing the ORIGINAL fp weights, computed BEFORE injection.
             QA-LoRA freezes {weight, scale, zero_point}, so after training and
             merging, qweight must equal these codes bit for bit.
  2. output: the trained, unmerged model (fake_quant + adapter). The merged model
             must reproduce its logits (same argmax, same loss).

Lesson from peft PR #2571 (QA-LoRA review by Benjamin Bossan): the loss sat at
0.0 and nobody noticed. So we also assert training really happened -- a zero
adapter would make the merge test pass trivially.
"""
import os

import pytest
import torch
import torch.nn as nn

transformers = pytest.importorskip("transformers")

from qpeft import QALoraConfig, get_quant_model                          # noqa: E402
from qpeft.quant_schemes import build_scheme                                   # noqa: E402
from qpeft.tuners.qa_lora.layer import ZeroPointFoldLoRA                 # noqa: E402
from qpeft.tuners.tuners_utils import QuantLinear                        # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TEXT = ("Quantization-aware training lets a model learn around the rounding error "
        "of its own integer grid, so that what is trained is what is deployed.")

# Merged logits may differ from unmerged only by float reassociation in
# z' = z - delta / s, accumulated over all layers. Tight on purpose; loosen only
# with a written reason, never to make a red run green.
LOGITS_ATOL = 1e-3
LOSS_RTOL = 1e-5


# --- model factories ------------------------------------------------------------

def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=320,
                      max_position_embeddings=64)
    model = LlamaForCausalLM(cfg).float().eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
    return model, input_ids


def _real_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = os.environ.get("QPEFT_TEST_MODEL", "Qwen/Qwen3-0.6B")
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    except Exception as e:                       # offline / gated / no disk
        pytest.skip(f"cannot load {model_id}: {e}")
    input_ids = tok([TEXT, TEXT[::-1]], return_tensors="pt", padding=True).input_ids
    return model.eval(), input_ids


MODELS = [
    pytest.param(_tiny_llama, id="tiny-llama"),
    pytest.param(_real_model, id="real-model", marks=pytest.mark.skipif(
        os.environ.get("QPEFT_RUN_SLOW") != "1", reason="set QPEFT_RUN_SLOW=1")),
]


# --- helpers --------------------------------------------------------------------

def _oracle_codes(model, cfg):
    """Oracle 1: RTN codes of the ORIGINAL fp weights, before qpeft touches them.
    Stored as uint8 (codes <= 2**bits - 1) to keep a 0.6B model affordable."""
    scheme = build_scheme(cfg)
    codes = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and any(t in name for t in TARGETS):
            s, z = scheme.init_qparams(m.weight.data, cfg.group_size)
            codes[name] = scheme.quantize(m.weight.data, s, z).to(torch.uint8)
    assert codes, "no target Linear found -- nothing would be tested"
    return codes


def _loss(model, input_ids):
    return model(input_ids=input_ids, labels=input_ids).loss


def _train_adapters(model, input_ids, steps=5, lr=1e-2):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = _loss(model, input_ids)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# --- the test -------------------------------------------------------------------

@pytest.mark.parametrize("make_model", MODELS)
def test_qa_lora_train_then_merge_is_lossless(make_model):
    base, input_ids = make_model()
    cfg = QALoraConfig(bits=4, group_size=64, r=8, target_modules=TARGETS)
    oracle = _oracle_codes(base, cfg)

    model = get_quant_model(base, cfg)
    adapters_before = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    # Only adapters train (QA-LoRA freezes the quantized base).
    assert adapters_before and all("adapter" in n for n in adapters_before)

    # --- training really happened (Bossan's loss == 0.0 lesson) ---------------
    losses = _train_adapters(model, input_ids)
    assert all(torch.isfinite(torch.tensor(losses))), losses
    assert losses[0] > 0.0, "loss is exactly 0 -- the adapter cannot be learning"
    assert losses[-1] < losses[0], f"loss did not go down: {losses}"
    moved = [n for n, p in model.named_parameters()
             if n in adapters_before and not torch.equal(p, adapters_before[n])]
    assert moved, "no adapter parameter changed -- the merge test would be trivial"
    assert any(m.B.abs().max() > 0 for m in model.modules()
               if isinstance(m, ZeroPointFoldLoRA)), "all B == 0 -> delta == 0"

    # --- oracle 2: the trained, unmerged model ------------------------------
    model.eval()
    with torch.no_grad():
        logits_before = model(input_ids=input_ids).logits
        loss_before = _loss(model, input_ids).item()
    scales_before = {n: m.scale.detach().clone()
                     for n, m in model.base.named_modules() if isinstance(m, QuantLinear)}

    merged = model.merge_and_unload()

    # --- "same weights": codes are the original model's RTN codes, bit for bit
    layers = {n: m for n, m in merged.named_modules() if isinstance(m, QuantLinear)}
    assert set(layers) == set(oracle), "merged layers != targeted layers"
    for name, layer in layers.items():
        assert layer.merged and layer.adapter is None
        assert layer.qweight.dtype == torch.int32
        assert torch.equal(layer.codes.to(torch.uint8), oracle[name]), \
            f"{name}: merge changed the integer codes"
        assert torch.equal(layer.scale, scales_before[name]), \
            f"{name}: merge changed the scale (only zero_point may move)"

    # --- "no loss": merged == trained --------------------------------------
    with torch.no_grad():
        logits_after = merged(input_ids=input_ids).logits
        loss_after = _loss(merged, input_ids).item()
    max_diff = (logits_before - logits_after).abs().max().item()
    assert max_diff <= LOGITS_ATOL, f"merged logits drift by {max_diff:.2e}"
    assert torch.equal(logits_before.argmax(-1), logits_after.argmax(-1)), \
        "merged model predicts different tokens"
    assert loss_after == pytest.approx(loss_before, rel=LOSS_RTOL)
