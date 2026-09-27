"""tinygemm end to end on a real model: train QA-LoRA -> merge -> save -> load -> to_tinygemm -> inference.

Opt-in (downloads SmolLM2-135M and WikiText-2, needs CUDA):
    QPEFT_RUN_SLOW=1 pytest tests/test_tinygemm_real_model.py

Oracle: the merged QuantModel on its reference forward (fp32, codes unpacked every call). The
tinygemm model holds the same codes, but runs every linear in bf16, so it may differ from the
oracle only by bf16 rounding of activations, scale and z_f.
"""
import copy
import math
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")

from qpeft import QALoraConfig, QuantModel, ZeroPointFoldLoRA, get_quant_model  # noqa: E402
from qpeft.kernels.tinygemm import TinyGemmLinear, to_tinygemm                  # noqa: E402
from qpeft.tuners.tuners_utils import QuantLinear                               # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("QPEFT_RUN_SLOW") != "1" or not torch.cuda.is_available(),
    reason="set QPEFT_RUN_SLOW=1 and run on a CUDA machine")

MODEL = "HuggingFaceTB/SmolLM2-135M"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SEQ, BATCH, STEPS, LR = 256, 4, 30, 2e-4
PPL_WINDOWS, LOGIT_WINDOWS, NEW_TOKENS = 32, 4, 40
PROMPT = "The history of the city begins in the"

# bf16 has an 8-bit mantissa; its rounding of activations through 30 layers stays near 1%.
LOGITS_REL_ERR = 3e-2
ARGMAX_AGREEMENT = 0.99
PPL_RTOL = 5e-3
# Greedy decoding is discontinuous at an argmax tie. Where the merged model's own top-1 and
# top-2 logits sit within bf16 noise, a faithful bf16 kernel may pick either token, and the two
# continuations then part ways for good -- this happens even bf16-merged vs bf16-tinygemm, so it
# is not the fp32/bf16 model precision, it is the tie. Measured on the test set the kernel changes
# the argmax only where that margin is <= 0.13 logits (logit std ~8, per-position error p99 ~0.5).
# A real kernel error moves logits by >10% and collapses the argmax agreement, so it would diverge
# at a *confident* step. We therefore let the two continuations diverge only at a near-tie in the
# merged model, with headroom over the measured 0.13 flip ceiling and far below confident margins.
GREEDY_TIE_MARGIN = 0.5


def _windows(tok, split, n):
    text = "\n\n".join(datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    return ids[: (ids.numel() // SEQ) * SEQ].view(-1, SEQ)[:n]


def _base():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()


@pytest.fixture(scope="module")
def setup():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
        base = _base().cuda()
        train = _windows(tok, "train", STEPS * BATCH).cuda()
        test = _windows(tok, "test", PPL_WINDOWS).cuda()
    except Exception as e:                       # offline / no disk
        pytest.skip(f"cannot load {MODEL} or WikiText-2: {e}")

    torch.manual_seed(0)
    model = get_quant_model(base, QALoraConfig(bits=4, group_size=64, r=16, lora_alpha=16,
                                               target_modules=TARGETS))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    for batch in train.split(BATCH):
        opt.zero_grad()
        model(input_ids=batch, labels=batch).loss.backward()
        opt.step()
    model.eval()
    assert all(m.B.abs().max() > 0 for m in model.modules() if isinstance(m, ZeroPointFoldLoRA)), \
        "an adapter is still zero; the fold would be trivial"

    model.merge_and_unload()
    kernel = to_tinygemm(copy.deepcopy(model))
    return tok, model, kernel, test


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits.float()


def _ppl(model, test):
    with torch.no_grad():
        losses = [model(input_ids=b, labels=b).loss.item() for b in test.split(8)]
    return math.exp(sum(losses) / len(losses))


def _greedy(model, tok):
    enc = tok(PROMPT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.base.generate(**enc, max_new_tokens=NEW_TOKENS, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
    return out[0, enc.input_ids.shape[1]:].tolist()


def test_every_target_runs_on_the_kernel(setup):
    _, model, kernel, _ = setup
    assert not any(isinstance(m, QuantLinear) for m in kernel.modules())
    n_quant = sum(isinstance(m, QuantLinear) for m in model.modules())
    assert n_quant == 7 * model.base.config.num_hidden_layers
    assert sum(isinstance(m, TinyGemmLinear) for m in kernel.modules()) == n_quant


def test_logits_match_the_merged_model(setup):
    _, model, kernel, test = setup
    ids = test[:LOGIT_WINDOWS]
    ref, got = _logits(model, ids), _logits(kernel, ids)
    rel = ((got - ref).norm() / ref.norm()).item()
    agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"\nlogits rel err {rel:.3e}, argmax agreement {agree:.4f} over {ids.numel()} tokens")
    assert rel < LOGITS_REL_ERR, f"logits rel err {rel:.3e}"
    assert agree >= ARGMAX_AGREEMENT, f"argmax agreement {agree:.4f} over {ids.numel()} tokens"


def test_perplexity_matches_the_merged_model(setup):
    _, model, kernel, test = setup
    ref, got = _ppl(model, test), _ppl(kernel, test)
    print(f"\nWikiText-2 ppl ({test.shape[0]} x {SEQ}): merged {ref:.4f}, tinygemm {got:.4f}")
    assert abs(got - ref) / ref < PPL_RTOL, f"ppl merged {ref:.4f} vs tinygemm {got:.4f}"


def _next_token_margin(model, ids):
    """The merged model's top1-top2 logit gap at the position that predicts the next token."""
    top2 = _logits(model, ids)[0, -1].topk(2).values
    return (top2[0] - top2[1]).item()


def test_greedy_generation_matches_the_merged_model(setup):
    tok, model, kernel, _ = setup
    ref, got = _greedy(model, tok), _greedy(kernel, tok)
    print(f"\ngreedy ({len(ref)} tokens): {tok.decode(ref)!r}")
    if ref == got:
        return
    first = next(i for i, (a, b) in enumerate(zip(ref, got)) if a != b)
    # They diverge: legitimate only at an argmax tie in the MERGED model itself. Both models chose
    # their first-th token from the same context (ref[:first] == got[:first]), so its margin decides.
    prefix = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
    shared = torch.cat([prefix, torch.tensor(ref[:first], device="cuda")[None]], dim=1)
    margin = _next_token_margin(model, shared)
    print(f"greedy diverges at new token {first}; merged top1-top2 margin there {margin:.3f}")
    assert margin < GREEDY_TIE_MARGIN, (
        f"generation diverges at new token {first} where the merged model is confident "
        f"(top1-top2 margin {margin:.3f} >= {GREEDY_TIE_MARGIN}) -- a kernel error, not a tie:\n"
        f"  merged:   {tok.decode(ref)!r}\n  tinygemm: {tok.decode(got)!r}")


def test_saved_and_loaded_model_runs_the_same_kernel(setup, tmp_path):
    _, model, kernel, test = setup
    model.save_pretrained(tmp_path)
    loaded = to_tinygemm(QuantModel.from_pretrained(_base(), tmp_path).cuda())
    ids = test[:LOGIT_WINDOWS]
    assert torch.equal(_logits(loaded, ids), _logits(kernel, ids))
