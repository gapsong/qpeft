"""Block-AP: EfficientQAT phase 1, block-wise reconstruction.

Mirrors the official quantize/block_ap.py:
  * the model is walked block by block (transformer layers)
  * for each block, the target is the output of the SAME block with its original
    fp weights; the quantized block is trained to reproduce it (MSE)
  * trainable: weight (weight_lr) + scale / zero_point (quant_lr), AdamW,
    cosine schedule down to lr / min_lr_factor. The block's other `*.weight`
    parameters (RMSNorm / LayerNorm) train with weight_lr too: the official
    filter is "weight in the name", so its results include trained norms
  * each block trains in fp32 and is cast back to its dtype afterwards
    (official: qlayer.float() ... qlayer.half()); in bf16 a 1e-5 step would not
    move the weights at all
  * the next block is fed with the (trained) quantized output, its target is the
    fp output chain

Deliberate differences, documented rather than hidden:
  * all calibration activations are kept in memory (official can offload to disk)
  * fp32 without autocast (official: autocast + GradScaler, CUDA only)
  * no validation split; the log reports the mean MSE over all calibration
    batches before and after training, in the block's real dtype
  * no quant_inplace here: the hand-over to E2E-QP freezes the codes via
    QuantModel.set_phase(...) -> QuantLinear.freeze_codes(), which is equivalent
"""
from __future__ import annotations

import inspect
import math

import torch
import torch.nn.functional as F
from torch import nn

from .tuners.tuners_utils import QuantLinear
from .training import param_groups


class _StopForward(Exception):
    pass


def find_blocks(model: nn.Module) -> nn.ModuleList:
    """The longest nn.ModuleList whose every entry contains a QuantLinear
    (for HF decoders: model.model.layers)."""
    best = None
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) > 0 and all(
                any(isinstance(x, QuantLinear) for x in child.modules()) for child in m):
            if best is None or len(m) > len(best):
                best = m
    if best is None:
        raise ValueError("Block-AP needs a model with an nn.ModuleList of blocks containing "
                         "QuantLinear (e.g. a Hugging Face decoder). None found.")
    return best


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _detach(v):
    if torch.is_tensor(v):
        return v.detach()
    if isinstance(v, tuple):
        return tuple(_detach(x) for x in v)
    return v


def _to_float32(v):
    """Floating tensors (also inside tuples, e.g. rotary cos/sin) -> fp32."""
    if torch.is_tensor(v):
        return v.float() if v.is_floating_point() else v
    if isinstance(v, tuple):
        return tuple(_to_float32(x) for x in v)
    return v


def _norm_weights(block):
    """The block's `*.weight` parameters outside QuantLinear (RMSNorm, LayerNorm)."""
    own = {id(p) for m in block.modules() if isinstance(m, QuantLinear) for p in m.parameters()}
    return [p for n, p in block.named_parameters()
            if id(p) not in own and n.rsplit(".", 1)[-1] == "weight"]


@torch.no_grad()
def _cast_params(params_to_dtype):
    for p, dtype in params_to_dtype.items():
        p.data = p.data.to(dtype)


@torch.no_grad()
def _capture_block_inputs(model, blocks, batches):
    """Run the model up to the first block and record exactly what it receives."""
    captured = []

    def hook(module, args, kwargs):
        kwargs = dict(kwargs)
        hidden = args[0] if args else kwargs.pop("hidden_states")
        captured.append((hidden.detach(), tuple(_detach(a) for a in args[1:]),
                         {k: _detach(v) for k, v in kwargs.items()}))
        raise _StopForward

    base = getattr(model, "base", model)
    extra = {"use_cache": False} if "use_cache" in inspect.signature(base.forward).parameters else {}
    handle = blocks[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for batch in batches:
            try:
                model(**batch, **extra)
            except _StopForward:
                pass
    finally:
        handle.remove()
    return captured


@torch.no_grad()
def _mean_mse(block, inps, targets, data):
    """Mean reconstruction MSE over all calibration batches, in the block's dtype."""
    errs = [F.mse_loss(_first(block(x, *a, **kw)).float(), t.float()).item()
            for x, t, (_, a, kw) in zip(inps, targets, data)]
    return sum(errs) / len(errs)


def _cosine(step, total, min_lr_factor):
    floor = 1.0 / min_lr_factor
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(step, total) / max(total, 1)))


def run_block_ap(model, batches, *, epochs: int = 2, weight_lr: float = 1e-5,
                 quant_lr: float = 1e-4, min_lr_factor: float = 20, weight_decay: float = 0.0,
                 log=print):
    """Train every block of an EfficientQAT (phase="block_ap") QuantModel.

    `batches`: iterable of dicts for model(**batch) (input_ids, attention_mask, ...),
    without labels. Returns the model (trained in place)."""
    layers = [m for m in model.modules() if isinstance(m, QuantLinear)]
    if not layers or not all(m.weight is not None and m.weight.requires_grad for m in layers):
        raise ValueError("run_block_ap needs an EfficientQAT model in phase 'block_ap' "
                         "(weight trainable on every QuantLinear).")
    was_training = model.training
    model.eval()                                  # no dropout: reconstruction is deterministic
    blocks = find_blocks(getattr(model, "base", model))
    data = _capture_block_inputs(model, blocks, list(batches))
    if not data:
        raise ValueError("no calibration batches for Block-AP")

    fp_inps = [d[0] for d in data]
    q_inps = [x.clone() for x in fp_inps]
    for i, block in enumerate(blocks):
        qls = [m for m in block.modules() if isinstance(m, QuantLinear)]

        for m in qls:                             # fp targets with the ORIGINAL weights
            m.quant_enabled = False
        with torch.no_grad():
            targets = [_first(block(x, *a, **kw)).detach() for x, (_, a, kw) in zip(fp_inps, data)]
        for m in qls:
            m.quant_enabled = True
        mse_before = _mean_mse(block, q_inps, targets, data)

        norms = _norm_weights(block)
        for p in norms:
            p.requires_grad_(True)
        dtypes = {p: p.dtype for p in block.parameters()}
        _cast_params({p: torch.float32 for p in dtypes})
        opt = torch.optim.AdamW(param_groups(block, weight_lr=weight_lr, quant_lr=quant_lr,
                                             adapter_lr=0.0, weight_decay=weight_decay,
                                             extra_weights=norms))
        total = epochs * len(q_inps)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s, t=total: _cosine(s, t, min_lr_factor))
        for _ in range(epochs):
            for x, t, (_, a, kw) in zip(q_inps, targets, data):
                out = _first(block(x.float(), *_to_float32(a), **{k: _to_float32(v) for k, v in kw.items()}))
                loss = F.mse_loss(out, t.float())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Block-AP block {i}: non-finite loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        del opt
        _cast_params(dtypes)                      # back to the block's dtype, as it will run
        for p in norms:
            p.requires_grad_(False)

        mse_after = _mean_mse(block, q_inps, targets, data)
        log(f"[block_ap] block {i}/{len(blocks) - 1}: mean mse {mse_before:.3e} -> {mse_after:.3e}")

        with torch.no_grad():
            q_inps = [_first(block(x, *a, **kw)).detach() for x, (_, a, kw) in zip(q_inps, data)]
        fp_inps = targets

    model.train(was_training)
    return model
