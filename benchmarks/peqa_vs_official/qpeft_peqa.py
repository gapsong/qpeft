"""The qpeft side: PEQAConfig on the same model, input, optimizer and steps, compared against
the dump that official_peqa.py wrote. Run in the qpeft env."""
import argparse

import torch
from transformers import AutoModelForCausalLM

from qpeft import PEQAConfig, get_quant_model
from qpeft.tuners.tuners_utils import QuantLinear

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group-size", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--official", required=True)
    a = ap.parse_args()

    ref = torch.load(a.official)
    input_ids = ref["input_ids"].cuda()
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda()
    model = get_quant_model(base, PEQAConfig(bits=a.bits, group_size=a.group_size, target_modules=TARGETS))
    layers = {n.removeprefix("base."): m for n, m in model.named_modules() if isinstance(m, QuantLinear)}
    assert layers.keys() == ref["start"].keys(), "different layers quantized"

    def state():
        return {n: {"codes": m.codes.cpu(), "zeros": m.zero_point.detach().cpu(), "scales": m.scale.detach().cpu()}
                for n, m in layers.items()}

    def diff(a_state, b_state, key):
        return max((a_state[n][key].float() - b_state[n][key].float()).abs().max().item() for n in a_state)

    def count_diff(a_state, b_state, key):
        return sum((a_state[n][key] != b_state[n][key]).sum().item() for n in a_state)

    n_codes = sum(v["codes"].numel() for v in ref["start"].values())
    start = state()
    with torch.no_grad():
        logits0 = model(input_ids).logits.cpu()
    print(f"== {a.model}  w{a.bits} g{a.group_size}  {len(layers)} layers  lr {a.lr}  {a.steps} AdamW steps ==")
    print(f"start  codes differing       : {count_diff(start, ref['start'], 'codes')} of {n_codes}")
    print(f"start  zero-points differing : {count_diff(start, ref['start'], 'zeros')}")
    print(f"start  scale max|diff|       : {diff(start, ref['start'], 'scales'):.3e}")
    print(f"start  logits max|diff|      : {(logits0 - ref['logits0']).abs().max().item():.3e}")

    # The official pack() rounds the start scale to fp16. Start qpeft from exactly that scale,
    # so the training comparison below measures only the training.
    with torch.no_grad():
        for n, m in layers.items():
            m.scale.copy_(ref["start"][n]["scales"].to(m.scale.device))
        logits0 = model(input_ids).logits.cpu()
    print(f"start  (same start scale) logits max|diff|: {(logits0 - ref['logits0']).abs().max().item():.3e}")
    start = state()

    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) == len(layers), "PEQA must train one scale tensor per layer and nothing else"
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    losses = []
    for _ in range(a.steps):
        opt.zero_grad()
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        logits = model(input_ids).logits.cpu()
    end = state()

    moved = diff(end, start, "scales")
    print(f"loss   official {ref['losses'][0]:.5f} -> {ref['losses'][-1]:.5f}")
    print(f"loss   qpeft    {losses[0]:.5f} -> {losses[-1]:.5f}")
    print(f"loss   max|diff| per step    : {max(abs(x - y) for x, y in zip(losses, ref['losses'])):.3e}")
    print(f"end    scales moved up to    : {moved:.3e}")
    print(f"end    scale max|diff|       : {diff(end, ref['end'], 'scales'):.3e}")
    print(f"end    codes differing       : {count_diff(end, ref['end'], 'codes')}  (and vs start: {count_diff(end, start, 'codes')})")
    print(f"end    zero-points differing : {count_diff(end, ref['end'], 'zeros')}  (and vs start: {count_diff(end, start, 'zeros')})")
    print(f"end    logits max|diff|      : {(logits - ref['logits']).abs().max().item():.3e}  (logit range {ref['logits'].abs().max().item():.1f})")

    model.merge_and_unload()
    with torch.no_grad():
        merged = model(input_ids).logits.cpu()
    kept = all(torch.equal(layers[n].codes.cpu(), end[n]["codes"]) for n in layers)
    print(f"merge  codes kept = {kept}, logits max|diff| vs training = {(merged - logits).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
