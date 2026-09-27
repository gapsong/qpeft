"""Is the 2-bit gap chaos? Train qpeft PEQA twice from the official start scale; the second run
starts with the scales changed by a relative 1e-7. Run in the qpeft env."""
import sys
import torch
from transformers import AutoModelForCausalLM
from qpeft import PEQAConfig, get_quant_model
from qpeft.tuners.tuners_utils import QuantLinear

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
model_id, bits, lr, official = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
ref = torch.load(official)
x = ref["input_ids"].cuda()


def run(eps):
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).cuda()
    model = get_quant_model(base, PEQAConfig(bits=bits, group_size=64, target_modules=TARGETS))
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, QuantLinear):
                m.scale.copy_(ref["start"][n.removeprefix("base.")]["scales"].cuda() * (1 + eps))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


a, b = run(0.0), run(1e-7)
print(f"w{bits}: qpeft vs qpeft (scale x (1 + 1e-7)): loss max|diff| per step {max(abs(u - v) for u, v in zip(a, b)):.3e}")
print(f"w{bits}: qpeft vs official:                   loss max|diff| per step {max(abs(u - v) for u, v in zip(a, ref['losses'])):.3e}")
