"""Perplexity of every arm after training, on WikiText-2 test and C4 validation.

  fp                  the bf16 base model, no training
  NF4                 bitsandbytes NF4 (as axolotl's QLoRA loads it), no training
  QLoRA (NF4 + LoRA)  the trained QLoRA adapter on the NF4 base: what was trained
  QLoRA merged bf16   the adapter merged into the bf16 base (peft's merge_and_unload): what
                      axolotl's merge gives; no longer 4-bit, and not the model that was trained
  RTN int4            qpeft's 4-bit start grid, no training
  qpeft merged int4   the integer artifact the plugin saved (codes + scale + zero-point),
                      once on the Triton kernel and once on the torch path;
                      then the artifact of every run in --extra-qpeft-runs (e.g. another lr)

Evaluation windows as in benchmarks/peqa_vs_official: 64 windows of 1024 tokens each.
Writes <runs>/eval.json.
"""
import argparse
import gzip
import json
import math
import random
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qpeft import QALoraConfig, QuantModel, get_quant_model
from qpeft.block_ap import block_linear_names
from qpeft.tuners.tuners_utils import QuantLinear

EVAL_SEQLEN, EVAL_WINDOWS = 1024, 64
DTYPE = torch.bfloat16


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
    model.eval()
    nll = sum(model(input_ids=x.cuda(), labels=x.cuda()).loss.item() for x in eval_windows)
    return math.exp(nll / len(eval_windows))


def bf16_base(model_id):
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE).cuda()


def nf4_base(model_id):
    """The same bitsandbytes settings axolotl uses for `adapter: qlora` + `load_in_4bit`."""
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=DTYPE, bnb_4bit_quant_storage=DTYPE)
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE, quantization_config=quant,
                                                device_map={"": 0})


def set_triton_kernel(use):
    QuantLinear.use_triton_kernel = use


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", required=True, help="directory with the qlora/ and qpeft_triton/ outputs")
    ap.add_argument("--extra-qpeft-runs", nargs="*", default=[])
    a = ap.parse_args()
    runs = Path(a.runs)

    tok = AutoTokenizer.from_pretrained(a.model)
    wiki_test = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1")["test"]["text"])
    evals = {"wikitext2": windows(tok(wiki_test, return_tensors="pt").input_ids, EVAL_SEQLEN, EVAL_WINDOWS),
             "c4": windows(c4_ids(tok), EVAL_SEQLEN, EVAL_WINDOWS)}
    results = {}

    def record(name, model):
        results[name] = {k: ppl(model, w) for k, w in evals.items()}
        print(f"{name:34s} " + "  ".join(f"{k} {v:8.3f}" for k, v in results[name].items()), flush=True)

    record("fp (bf16)", bf16_base(a.model))
    record("NF4, no training", nf4_base(a.model))
    record("QLoRA (NF4 + LoRA, unmerged)", PeftModel.from_pretrained(nf4_base(a.model), runs / "qlora"))
    record("QLoRA merged to bf16", PeftModel.from_pretrained(bf16_base(a.model), runs / "qlora").merge_and_unload())

    base = bf16_base(a.model)
    record("RTN int4 g64, no training", get_quant_model(base, QALoraConfig(
        bits=4, group_size=64, r=16, target_modules=block_linear_names(base))))
    merged = QuantModel.from_pretrained(bf16_base(a.model), runs / "qpeft_triton" / "qpeft")
    set_triton_kernel(True)
    record("qpeft QA-LoRA merged int4 (Triton)", merged)
    set_triton_kernel(False)
    record("qpeft QA-LoRA merged int4 (torch)", merged)
    set_triton_kernel(True)
    for run in a.extra_qpeft_runs:
        record(f"qpeft QA-LoRA merged int4 ({run})", QuantModel.from_pretrained(bf16_base(a.model), runs / run / "qpeft"))

    (runs / "eval.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
