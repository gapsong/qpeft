"""EfficientQAT end to end: quantization-aware training over a 2-bit substrate,
then the spine check (fake_quant == merge) and a merge that stays integer.

Run:  python examples/train_efficient_qat.py

What this proves today (pure-torch `int_uniform` reference backend):
  1. the STE fake_quant carries gradients -> the model actually trains,
  2. check_merge_equivalence is green per QuantLinear (the scheme is trustworthy),
  3. merge_and_unload() returns an INTEGER artifact, and the merged model's
     output equals the trained (fake_quant) output.
"""
import torch
import torch.nn as nn

from qpeft import EfficientQATConfig, check_merge_equivalence, get_quant_model
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


def check_equivalence(model, in_f):
    """Run the spine test on every QuantLinear in the model."""
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear) and not m.merged:
            x = torch.randn(8, in_f)
            err = check_merge_equivalence(
                m.scheme, m.weight, m.scale, m.zero_point, m.adapter, x,
                codes=m.frozen_codes)
            print(f"    check_merge_equivalence[{name or 'root'}] OK  max|delta|={err:.2e}")


def main():
    in_f, out_f = 256, 128
    x, y = toy_task(in_f, out_f)

    # Block-AP phase: train weights + step size over the 2-bit grid (STE).
    cfg = EfficientQATConfig(bits=2, group_size=64, phase="block_ap")
    model = get_quant_model(nn.Sequential(nn.Linear(in_f, out_f)), cfg)

    first, last = train(model, x, y)
    print(f"[efficient_qat] loss {first:.4f} -> {last:.4f}  ({100*(1-last/first):.0f}% down)")
    assert last < first * 0.8, "training did not reduce the loss"

    # The scheme is only trustworthy once this is green.
    check_equivalence(model, in_f)

    # Merge: stays integer, and the merged output matches the trained output.
    with torch.no_grad():
        before = model(x)
    merged = model.merge_and_unload()
    with torch.no_grad():
        after = merged(x)

    q = next(m for m in merged.modules() if isinstance(m, QuantLinear))
    assert q.qweight.dtype == torch.int32, "merged weights are not integer"
    drift = (before - after).abs().max().item()
    print(f"[efficient_qat] merged stays int (dtype={q.qweight.dtype}); "
          f"output drift vs trained = {drift:.2e}")
    assert drift < 1e-4, "merged model disagrees with the trained model"
    print("[efficient_qat] OK")


if __name__ == "__main__":
    main()
