"""Load a Hugging Face transformers model and inject QuantLinear into it.

Run:  python examples/hf_injection.py

qpeft's injection is the same mechanism peft uses: a transformers model is just an
nn.Module tree, so get_quant_model() walks it, matches the Linear layers named in
`target_modules`, and swaps each one in place for a QuantLinear (weight quantized,
bias kept). Everything else -- attention, norms, embeddings, the forward signature
-- is untouched, so the model still runs as a causal LM.
"""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from qpeft import EfficientQATConfig, get_quant_model
from qpeft.tuners.tuners_utils import QuantLinear

# A tiny Llama built from a config (random weights, no download). Dims are chosen
# divisible by group_size=64 so every targeted Linear can be grouped.
cfg = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, vocab_size=320,
                  max_position_embeddings=64)
hf_model = LlamaForCausalLM(cfg)

n_linear_before = sum(isinstance(m, torch.nn.Linear) for m in hf_model.modules())
input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
with torch.no_grad():
    ref = hf_model(input_ids=input_ids).logits

# Inject: same target-module names as peft/LoRA on Llama.
quant_cfg = EfficientQATConfig(
    bits=4, group_size=64, phase="block_ap",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"])
model = get_quant_model(hf_model, quant_cfg)

injected = [n for n, m in model.named_modules() if isinstance(m, QuantLinear)]
still_linear = [n for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear) and not isinstance(m, QuantLinear)]

print(f"nn.Linear in base model:        {n_linear_before}")
print(f"replaced with QuantLinear:      {len(injected)}")
print(f"  e.g. {injected[0]}")
print(f"left as plain nn.Linear:        {still_linear}   (lm_head not targeted)")

with torch.no_grad():
    out = model(input_ids=input_ids).logits          # same forward signature
print(f"forward still works: logits {tuple(out.shape)}, finite={torch.isfinite(out).all().item()}")
delta = (out - ref).abs().max().item()
print(f"quantization moved the logits (expected): max|delta|={delta:.3f}")
print("[hf_injection] OK")
