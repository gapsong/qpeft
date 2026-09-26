"""End to end on the qpeft side: PEQA training on WikiText-2, merge, save / load, and export of
the merged artifact in the official EfficientQAT int layout (GPTQ qweight + packed qzeros + scales).
official_deploy.py then runs that artifact in the official QuantLinear. Run in the qpeft env."""
import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from qpeft import PEQAConfig, QuantModel, get_quant_model
from qpeft.packing import pack_codes
from qpeft.tuners.tuners_utils import QuantLinear

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SEQLEN, EVAL_WINDOWS = 512, 40


@torch.no_grad()
def ppl(model, test_ids):
    nll = 0.0
    for i in range(EVAL_WINDOWS):
        x = test_ids[:, i * SEQLEN:(i + 1) * SEQLEN].cuda()
        nll += model(input_ids=x, labels=x).loss.item()
    return float(torch.exp(torch.tensor(nll / EVAL_WINDOWS)))


def quant_layers(model):
    return {n.removeprefix("base."): m for n, m in model.named_modules() if isinstance(m, QuantLinear)}


def export_official(layers, bits):
    """The merged artifact in the official int_linear_real.QuantLinear layout. PEQA folds nothing,
    so the zero-point must still be an integer in [0, 2**bits - 1]; anything else is refused."""
    out = {}
    for n, m in layers.items():
        z = m.zero_point.detach()
        if not torch.equal(z, z.round()) or z.min() < 0 or z.max() > 2 ** bits - 1:
            raise ValueError(f"{n}: zero-point is not an integer code; the official layout cannot hold it")
        out[n] = {"qweight": m.qweight.cpu(),                                   # same GPTQ layout
                  # (groups, out), packed along out: pack_codes packs along dim 1 of its input.
                  "qzeros": pack_codes(z.t().to(torch.int32), bits).t().contiguous().cpu(),
                  "scales": m.scale.detach().t().contiguous().float().cpu()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group-size", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(a.model)
    data = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_ids = tok("\n\n".join(data["train"]["text"]), return_tensors="pt").input_ids
    test_ids = tok("\n\n".join(data["test"]["text"]), return_tensors="pt").input_ids
    torch.save(test_ids[:, :EVAL_WINDOWS * SEQLEN].clone(), f"{a.out}/test_ids.pt")

    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda()
    print(f"== {a.model}  w{a.bits} g{a.group_size}  lr {a.lr}  {a.steps} steps x {a.batch} x {SEQLEN} tokens ==")
    print(f"ppl fp                  {ppl(base, test_ids):.3f}")
    model = get_quant_model(base, PEQAConfig(bits=a.bits, group_size=a.group_size, target_modules=TARGETS))
    layers = quant_layers(model)
    codes0 = {n: m.codes.clone() for n, m in layers.items()}
    print(f"ppl RTN (PEQA start)    {ppl(model, test_ids):.3f}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(0)
    for step in range(a.steps):
        starts = torch.randint(0, train_ids.shape[1] - SEQLEN, (a.batch,), generator=g)
        x = torch.stack([train_ids[0, s:s + SEQLEN] for s in starts]).cuda()
        opt.zero_grad()
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == a.steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")
    trained = ppl(model, test_ids)
    print(f"ppl PEQA trained        {trained:.3f}")
    print(f"codes unchanged by training: {all(torch.equal(m.codes, codes0[n]) for n, m in layers.items())}")

    x = test_ids[:, :SEQLEN].cuda()
    with torch.no_grad():
        logits_train = model(input_ids=x).logits
    model.merge_and_unload()
    with torch.no_grad():
        logits_merged = model(input_ids=x).logits
    print(f"ppl merged              {ppl(model, test_ids):.3f}   "
          f"(logits max|diff| vs training {(logits_merged - logits_train).abs().max().item():.3e})")

    model.save_pretrained(f"{a.out}/qpeft")
    fresh = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda()
    loaded = QuantModel.from_pretrained(fresh, f"{a.out}/qpeft")
    with torch.no_grad():
        logits_loaded = loaded(input_ids=x).logits
    print(f"ppl saved + loaded      {ppl(loaded, test_ids):.3f}   "
          f"(bit-exact vs merged: {torch.equal(logits_loaded, logits_merged)})")

    torch.save({"layers": export_official(layers, a.bits), "logits_merged": logits_merged.cpu(),
                "ppl_merged": ppl(model, test_ids)}, f"{a.out}/official_layout.pt")
    print(f"exported {len(layers)} layers in the official layout -> {a.out}/official_layout.pt")


if __name__ == "__main__":
    main()
