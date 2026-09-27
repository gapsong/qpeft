"""Run qpeft's merged PEQA artifact in the official EfficientQAT int layer (int_linear_real.QuantLinear,
Triton dequant kernels), as its load_quantized_model would. Run in the eqat_official env with
PYTHONPATH=~/Documents/EfficientQAT."""
import argparse
import math

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from quantize.int_linear_real import QuantLinear

SEQLEN, EVAL_WINDOWS = 512, 40


@torch.no_grad()
def ppl(model, test_ids):
    nll = 0.0
    for i in range(EVAL_WINDOWS):
        x = test_ids[:, i * SEQLEN:(i + 1) * SEQLEN].cuda()
        nll += model(input_ids=x, labels=x).loss.float().item()
    return math.exp(nll / EVAL_WINDOWS)


def load(model_id, art, bits, group_size, dtype):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    for name, linear in list(model.named_modules()):
        if name not in art:
            continue
        q = QuantLinear(bits, group_size, linear.in_features, linear.out_features, linear.bias is not None)
        q.qweight.copy_(art[name]["qweight"])
        q.qzeros.copy_(art[name]["qzeros"])
        q.scales = nn.Parameter(art[name]["scales"].to(dtype), requires_grad=False)
        parent, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent), attr, q)
    assert sum(isinstance(m, QuantLinear) for m in model.modules()) == len(art)
    return model.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group-size", type=int, required=True)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    ref = torch.load(f"{a.dir}/official_layout.pt")
    test_ids = torch.load(f"{a.dir}/test_ids.pt")

    model = load(a.model, ref["layers"], a.bits, a.group_size, torch.float32)
    with torch.no_grad():
        logits = model(input_ids=test_ids[:, :SEQLEN].cuda()).logits.cpu()
    print(f"official QuantLinear fp32: ppl {ppl(model, test_ids):.3f} (qpeft merged {ref['ppl_merged']:.3f}), "
          f"logits max|diff| vs qpeft merged {(logits - ref['logits_merged']).abs().max().item():.3e}")
    del model
    model = load(a.model, ref["layers"], a.bits, a.group_size, torch.float16)
    print(f"official QuantLinear fp16 (its deployment dtype): ppl {ppl(model, test_ids):.3f}")


if __name__ == "__main__":
    main()
