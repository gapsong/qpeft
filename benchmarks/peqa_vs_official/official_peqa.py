"""The official side: PEQA as the official EfficientQAT code runs it, unchanged, on CUDA.

EfficientQAT with 0 Block-AP epochs is PEQA: min/max RTN init (UniformAffineQuantizer),
quant_inplace, pack into the real int QuantLinear (Triton kernels), then E2E-QP trains
only `scales` with AdamW, weight decay 0 (main_e2e_qp.py).

One change: the scales are cast from fp16 to fp32 after pack, so both sides train the same
fp32 master copy. Run in the eqat_official env with PYTHONPATH=~/Documents/EfficientQAT.
"""
import argparse
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.int_linear_real import QuantLinear
from quantize.quantizer import UniformAffineQuantizer
from quantize.triton_utils.kernels import dequant_dim0, dequant_dim1

with open(os.path.join(os.path.dirname(__file__), "text.txt")) as f:
    TEXT = f.read()


def peqa(model, bits, group_size):
    for layer in model.model.layers:
        for name, linear in list(layer.named_modules()):
            if not isinstance(linear, nn.Linear):
                continue
            quantizer = UniformAffineQuantizer(bits, group_size, weight=linear.weight)
            with torch.no_grad():
                linear.weight.copy_(quantizer(linear.weight))              # quant_inplace
            dim0 = linear.weight.shape[0]
            scales = quantizer.scale.clamp(1e-4, 1e4).detach().view(dim0, -1).t().contiguous()
            zeros = quantizer.zero_point.detach().round().view(dim0, -1).t().contiguous()
            q = QuantLinear(bits, group_size, linear.in_features, linear.out_features, linear.bias is not None)
            q.pack(linear.cpu(), scales.float().cpu(), zeros.float().cpu())
            q.scales = nn.Parameter(q.scales.float())
            q = q.cuda()
            parent, _, attr = name.rpartition(".")
            setattr(layer.get_submodule(parent) if parent else layer, attr, q)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = {n: m for n, m in model.named_modules() if isinstance(m, QuantLinear)}
    for m in layers.values():
        m.scales.requires_grad_(True)
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group-size", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    input_ids = tok([TEXT], return_tensors="pt").input_ids.cuda()
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda()
    layers = peqa(model, a.bits, a.group_size)

    def state():
        return {n: {"codes": dequant_dim0(m.qweight, m.bits, m.maxq, m.infeatures, m.outfeatures).t().int().cpu(),
                    "zeros": dequant_dim1(m.qzeros, m.bits, m.maxq, m.zeros_dim0, m.zeros_dim1).t().float().cpu(),
                    "scales": m.scales.detach().t().float().cpu()}
                for n, m in layers.items()}

    start = state()
    with torch.no_grad():
        logits0 = model(input_ids).logits.cpu()
    opt = torch.optim.AdamW([m.scales for m in layers.values()], lr=a.lr, weight_decay=0.0)
    losses = []
    for _ in range(a.steps):
        opt.zero_grad()
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        logits = model(input_ids).logits.cpu()
    torch.save({"input_ids": input_ids.cpu(), "start": start, "logits0": logits0,
                "end": state(), "losses": losses, "logits": logits}, a.out)
    print(f"official: {len(layers)} layers, loss {losses[0]:.5f} -> {losses[-1]:.5f}, saved {a.out}")


if __name__ == "__main__":
    main()
