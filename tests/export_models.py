"""Tiny random Llama models, merged by qpeft, that the export tests write out (no download)."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from qpeft import EfficientQATConfig, PEQAConfig, get_quant_model, run_block_ap
from qpeft.tuners.tuners_utils import QuantLinear

BLOCK_LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
VOCAB = 320


class Router(torch.nn.Linear):
    """A subclass of nn.Linear, like an MoE router with its own forward."""


class LlamaWithRouters(LlamaForCausalLM):
    """Every decoder block also holds a Router, so the model a loader builds from config has it too."""
    def __init__(self, config):
        super().__init__(config)
        for block in self.model.layers:
            block.router = Router(config.hidden_size, 8)


def llama(dtype, tie=False, model_class=LlamaForCausalLM):
    """Built in `dtype` the way from_pretrained builds it, so the rotary inv_freq stays fp32:
    .to(dtype) would round it, and a loaded export would no longer match this model bit for bit."""
    torch.manual_seed(0)
    return model_class._from_config(LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, vocab_size=VOCAB, max_position_embeddings=64,
        tie_word_embeddings=tie), dtype=dtype).eval()


def batches(n=2, seq=16):
    g = torch.Generator().manual_seed(0)
    return [{"input_ids": torch.randint(0, VOCAB, (2, seq), generator=g)} for _ in range(n)]


def train_scales(model, steps=3):
    """A few optimizer steps on the scales, so the export sees trained, not RTN, scales."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for batch in batches(steps):
        opt.zero_grad()
        model(input_ids=batch["input_ids"], labels=batch["input_ids"]).loss.backward()
        opt.step()


def merged_peqa(bits, dtype=torch.bfloat16, targets=BLOCK_LINEARS, tie=False):
    model = get_quant_model(llama(dtype, tie), PEQAConfig(bits=bits, group_size=64, target_modules=targets))
    train_scales(model)
    model.merge_and_unload()
    return model


def merged_efficient_qat(bits, dtype=torch.bfloat16):
    """Block-AP trains weight, scale and zero-point, so the zero-points are trained too."""
    block_ap, e2e_qp = (EfficientQATConfig(bits=bits, group_size=64, phase=phase, target_modules=BLOCK_LINEARS)
                        for phase in ("block_ap", "e2e_qp"))
    model = get_quant_model(llama(dtype), block_ap)
    run_block_ap(model, batches(), epochs=1, weight_lr=1e-3, quant_lr=1e-3, log=lambda _: None)
    model = get_quant_model(model, e2e_qp)
    train_scales(model)
    model.merge_and_unload()
    return model


def unpack_bit_stream(words, bits, n):
    """(n_words, cols) int32 -> (n, cols): value k is bits [k * bits, (k + 1) * bits) of each
    column's bit stream, lowest bit first (GPTQ's qweight layout, also its 3-bit word spill;
    compressed-tensors' dense packing since 0.18)."""
    bit = (words.to(torch.int64)[:, None, :] >> torch.arange(32)[None, :, None]) & 1
    stream = bit.reshape(-1, words.shape[1])[: n * bits]
    return (stream.reshape(n, bits, -1) << torch.arange(bits)[None, :, None]).sum(1)


def quant_layers(model):
    return {name: m for name, m in model.base.named_modules() if isinstance(m, QuantLinear)}
