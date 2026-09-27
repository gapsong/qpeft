"""Fair comparison: PEQA vs EfficientQAT on the same model, data and E2E phase.

  RTN            min/max init, no training
  PEQA           RTN -> E2E (only the scales train)
  Block-AP only  RTN -> Block-AP (weights, scales, zero-points; block-wise MSE)
  EfficientQAT   RTN -> Block-AP -> E2E (exactly the same E2E loop as PEQA)

Training data: WikiText-2 train. Evaluation: WikiText-2 test (in domain) and C4 validation
(out of domain for the E2E phase). Every arm is merged and evaluated again. Run in the qpeft env.
"""
import argparse
import copy
import gzip
import json
import math
import random

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from qpeft import PEQAConfig, efficient_qat_schedule, get_quant_model, run_block_ap

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EVAL_SEQLEN, EVAL_WINDOWS = 1024, 64
E2E_SEQLEN, E2E_BATCH, E2E_STEPS = 512, 4, 300
CALIB_SEQLEN, CALIB_SAMPLES, CALIB_BATCH = 1024, 256, 2


def windows(ids, seqlen, n):
    return [ids[:, i * seqlen:(i + 1) * seqlen] for i in range(n)]


def c4_ids(tok):
    """C4 validation, first shard, documents shuffled with a fixed seed (as GPTQ / EfficientQAT do)."""
    path = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset")
    with gzip.open(path, "rt") as f:
        docs = [json.loads(line)["text"] for line in f]
    random.Random(0).shuffle(docs)
    return tok("\n\n".join(docs[:3000]), return_tensors="pt").input_ids


@torch.no_grad()
def ppl(model, eval_windows):
    nll = sum(model(input_ids=x.cuda(), labels=x.cuda()).loss.item() for x in eval_windows)
    return math.exp(nll / len(eval_windows))


def e2e(model, train_ids, lr, log):
    """The E2E phase, identical for PEQA and EfficientQAT: AdamW on the scales, cosine, no wd."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E2E_STEPS)
    g = torch.Generator().manual_seed(0)
    model.train()
    for step in range(E2E_STEPS):
        starts = torch.randint(0, train_ids.shape[1] - E2E_SEQLEN, (E2E_BATCH,), generator=g)
        x = torch.stack([train_ids[0, s:s + E2E_SEQLEN] for s in starts]).cuda()
        opt.zero_grad()
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == E2E_STEPS - 1:
            log(f"    e2e step {step:3d}  loss {loss.item():.4f}")
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--e2e-lr", type=float, required=True)
    ap.add_argument("--weight-lr", type=float, required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    log = print

    tok = AutoTokenizer.from_pretrained(a.model)
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_ids = tok("\n\n".join(wiki["train"]["text"]), return_tensors="pt").input_ids
    evals = {"wikitext2": windows(tok("\n\n".join(wiki["test"]["text"]), return_tensors="pt").input_ids,
                                  EVAL_SEQLEN, EVAL_WINDOWS),
             "c4": windows(c4_ids(tok), EVAL_SEQLEN, EVAL_WINDOWS)}
    g = torch.Generator().manual_seed(1)
    starts = torch.randint(0, train_ids.shape[1] - CALIB_SEQLEN, (CALIB_SAMPLES,), generator=g)
    calib = torch.stack([train_ids[0, s:s + CALIB_SEQLEN] for s in starts])
    calib_batches = [{"input_ids": calib[i:i + CALIB_BATCH].cuda()} for i in range(0, CALIB_SAMPLES, CALIB_BATCH)]

    results = {}

    def record(name, model):
        results[name] = {k: ppl(model, w) for k, w in evals.items()}
        log(f"  {name:28s} " + "  ".join(f"{k} {v:10.3f}" for k, v in results[name].items()))

    fp = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda().eval()
    log(f"== {a.model}  w{a.bits} g{a.group_size}  e2e lr {a.e2e_lr}  block-ap weight lr {a.weight_lr} ==")
    record("fp", fp)

    # RTN and PEQA
    peqa = get_quant_model(copy.deepcopy(fp), PEQAConfig(bits=a.bits, group_size=a.group_size, target_modules=TARGETS))
    record("RTN", peqa)
    log("  PEQA: e2e")
    e2e(peqa, train_ids, a.e2e_lr, log)
    record("PEQA", peqa)
    peqa.merge_and_unload()
    record("PEQA merged", peqa)
    del peqa

    # Block-AP only, then EfficientQAT
    block_ap_cfg, e2e_cfg = efficient_qat_schedule(bits=a.bits, group_size=a.group_size, target_modules=TARGETS)
    base = copy.deepcopy(fp)
    del fp
    eqat = get_quant_model(base, block_ap_cfg)
    log("  EfficientQAT: block-ap")
    run_block_ap(eqat, calib_batches, epochs=2, weight_lr=a.weight_lr, quant_lr=1e-4,
                 log=lambda s: log("    " + s))
    eqat.eval()
    record("Block-AP only", eqat)
    eqat = get_quant_model(base, e2e_cfg)       # codes freeze here: the E2E-QP phase
    log("  EfficientQAT: e2e")
    e2e(eqat, train_ids, a.e2e_lr, log)
    record("EfficientQAT (Block-AP+E2E)", eqat)
    eqat.merge_and_unload()
    record("EfficientQAT merged", eqat)

    log("SUMMARY " + json.dumps({"bits": a.bits, "results": results}))


if __name__ == "__main__":
    main()
