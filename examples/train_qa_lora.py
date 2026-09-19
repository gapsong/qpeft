"""QA-LoRA end to end: a frozen quantized base plus a trainable adapter, then a
merge that folds the adapter into the zero-points and STAYS integer.

Run:  python examples/train_qa_lora.py

Contrast with peft: peft trains an adapter too, but its merge_and_unload()
dequantizes to fp16. Here the adapter is pooled per quantization group, so on
merge it folds exactly into the per-group zero-points and the artifact remains
quantized -- check_merge_equivalence proves the fold is exact.
"""
import torch
import torch.nn as nn

from qpeft import QALoraConfig, check_merge_equivalence, get_quant_model
from qpeft.tuners.qa_lora.layer import ZeroPointFoldLoRA
from qpeft.tuners.tuners_utils import QuantLinear


def train(model, x, y, steps=400, lr=5e-3):
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


def main():
    torch.manual_seed(1)
    in_f, out_f = 256, 128
    x = torch.randn(512, in_f)

    cfg = QALoraConfig(bits=4, group_size=32, r=16)
    model = get_quant_model(nn.Sequential(nn.Linear(in_f, out_f)), cfg)

    # Only the adapter is trainable; the quantized base is frozen.
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all("adapter" in n for n in trainable), f"non-adapter params train: {trainable}"

    # The QA-LoRA adapter is pooled per quantization group, so it can only learn
    # a group-structured correction on top of the frozen quantized base -- that is
    # exactly the price of a merge that folds into the zero-points. So we set a
    # target the method can actually represent: base output + such a correction.
    teacher = ZeroPointFoldLoRA(in_f, out_f, r=4, alpha=16, group_size=cfg.group_size)
    with torch.no_grad():
        teacher.A.normal_(0, 1.0)
        teacher.B.normal_(0, 0.3)
        y = model(x) + teacher(x)               # frozen base + group-structured signal

    first, last = train(model, x, y)
    print(f"[qa_lora] loss {first:.4f} -> {last:.4f}  ({100*(1-last/first):.0f}% down); "
          f"trained only: {sorted(trainable)}")
    assert last < first * 0.8, "adapter training did not reduce the loss"

    # Spine test: the adapter folds into the zero-points exactly.
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear):
            err = check_merge_equivalence(
                m.scheme, m.weight, m.scale, m.zero_point, m.adapter, torch.randn(8, in_f))
            print(f"    check_merge_equivalence[{name or 'root'}] OK  max|delta|={err:.2e}")

    with torch.no_grad():
        before = model(x)
    merged = model.merge_and_unload()
    with torch.no_grad():
        after = merged(x)

    q = next(m for m in merged.modules() if isinstance(m, QuantLinear))
    assert q.qweight.dtype == torch.int32, "merged weights are not integer"
    assert q.adapter is None, "adapter was not folded away"
    rel = (before - after).abs().max().item() / before.abs().max().item()
    print(f"[qa_lora] merged stays int (dtype={q.qweight.dtype}, adapter folded); "
          f"relative output drift vs trained = {rel:.2e}")
    assert rel < 1e-4, "merged model disagrees with the trained model"
    print("[qa_lora] OK")


if __name__ == "__main__":
    main()
