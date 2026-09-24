"""EfficientQAT end to end: quantization-aware training over a 2-bit substrate,
then the spine check (fake_quant == merge) and a merge that stays integer.

Run:  python examples/train_efficient_qat.py

What this proves (pure-torch `int_uniform` reference backend):
  1. Block-AP: the STE fake_quant carries gradients -> weight/scale/zero_point train,
  2. the phase switch freezes the integer codes; E2E-QP trains only the scale,
  3. the layer-level merge check is green per QuantLinear,
  4. merge_and_unload() returns an INTEGER artifact whose output equals the trained one.
"""
import torch
import torch.nn as nn

from qpeft import efficient_qat_schedule, get_quant_model, verify_quant_model
from qpeft.tuners.tuners_utils import QuantLinear


def toy_task(in_f=256, out_f=128, n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    teacher = nn.Linear(in_f, out_f)
    x = torch.randn(n, in_f, generator=g)
    with torch.no_grad():
        y = teacher(x)
    return x, y


def train(model, x, y, steps=300, lr=5e-3):
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    loss_fn = nn.MSELoss()
    first = last = None
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    return first, last


def check_equivalence(model):
    """Layer-level spine check: training forward == merged forward, per QuantLinear.
    Covers frozen codes too (E2E-QP), which the scheme-level check cannot."""
    for name, err in verify_quant_model(model).items():
        print(f"    merge equivalence[{name or 'root'}] OK  max|delta|={err:.2e}")


def main():
    in_f, out_f = 256, 128
    x, y = toy_task(in_f, out_f)
    base = nn.Sequential(nn.Linear(in_f, out_f))

    # Phase 1, Block-AP: weight + scale + zero_point train over the 2-bit grid (STE).
    # (On a transformer, QATTrainer / run_block_ap do this block by block;
    #  see examples/qat_trainer.py. Here: one layer, end to end.)
    block_ap, e2e_qp = efficient_qat_schedule(bits=2, group_size=64)
    model = get_quant_model(base, block_ap)
    first, last = train(model, x, y)
    print(f"[block_ap] loss {first:.4f} -> {last:.4f}  ({100*(1-last/first):.0f}% down)")
    assert last < first * 0.8, "Block-AP did not reduce the loss"

    # Hand-over: same call as for phase 1. The existing QuantLinear switch phase,
    # the integer codes are FROZEN and only the scale stays trainable.
    model = get_quant_model(model, e2e_qp)
    q = next(m for m in model.modules() if isinstance(m, QuantLinear))
    trainable = {n for n in ("weight", "scale", "zero_point") if getattr(q, n).requires_grad}
    assert q.codes_frozen and trainable == {"scale"}, trainable
    codes = q.qweight.clone()

    # Phase 2, E2E-QP: only the step size trains, on top of fixed codes.
    first, last = train(model, x, y, steps=200, lr=1e-3)
    print(f"[e2e_qp]   loss {first:.4f} -> {last:.4f}  (scale only, codes frozen)")
    assert last <= first, "E2E-QP made the loss worse"
    assert torch.equal(q.qweight, codes), "E2E-QP moved the integer codes"

    # The scheme is only trustworthy once this is green.
    check_equivalence(model)

    # Merge: stays integer, and the merged output matches the trained output.
    with torch.no_grad():
        before = model(x)
    merged = model.merge_and_unload()
    with torch.no_grad():
        after = merged(x)

    assert q.qweight.dtype == torch.int32, "merged weights are not integer"
    assert q.weight is None, "fp master weight survived the merge"
    drift = (before - after).abs().max().item()
    print(f"[efficient_qat] merged stays int (dtype={q.qweight.dtype}); "
          f"output drift vs trained = {drift:.2e}")
    assert drift < 1e-4, "merged model disagrees with the trained model"
    print("[efficient_qat] OK")


if __name__ == "__main__":
    main()
