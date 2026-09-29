"""Block-AP: phase 1 of EfficientQAT, block-wise reconstruction.

Go through the transformer blocks one by one.
For each block, train its quantized version to reproduce the output of its
full-precision version (MSE loss).

Same as the official quantize/block_ap.py, except:
  * all calibration activations stay in memory (official can offload to disk)
  * plain fp32 training, no autocast + GradScaler
  * no validation split; the log shows the mean MSE before and after training
The hand-over to E2E-QP is QuantModel.set_phase(...), not the official quant_inplace.
"""
from __future__ import annotations

import inspect
import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from .training import param_groups
from .tuners.tuners_utils import QuantLinear, quant_layers


def run_block_ap(model, batches, *, epochs: int = 2, weight_lr: float = 1e-5,
                 quant_lr: float = 1e-4, min_lr_factor: float = 20, weight_decay: float = 0.0,
                 log=print):
    """Train every block of an EfficientQAT model in phase "block_ap", in place.

    `batches`: dicts for model(**batch) (input_ids, attention_mask, ...), without labels."""
    _check_phase_block_ap(model)
    was_training = model.training
    model.eval()                                  # no dropout: reconstruction is deterministic

    blocks = find_blocks(getattr(model, "base", model))
    fp_hidden, extras_per_block = _capture_inputs(model, blocks, batches)
    q_hidden = fp_hidden

    for i, (block, extras) in enumerate(zip(blocks, extras_per_block)):
        with _quantization_off(block):
            target = _run_block(block, fp_hidden, extras)
        mse_before = _mean_mse(block, q_hidden, extras, target)

        _train_block(block, q_hidden, extras, target, epochs=epochs, weight_lr=weight_lr,
                     quant_lr=quant_lr, min_lr_factor=min_lr_factor, weight_decay=weight_decay)

        mse_after = _mean_mse(block, q_hidden, extras, target)
        log(f"[block_ap] block {i}/{len(blocks) - 1}: mean mse {mse_before:.3e} -> {mse_after:.3e}")

        # The next quantized block sees what the quantized model really produces,
        # while its target stays the full-precision chain.
        q_hidden = _run_block(block, q_hidden, extras)
        fp_hidden = target

    model.train(was_training)
    return model


def _train_block(block, hidden, extras, target, *, epochs, weight_lr, quant_lr, min_lr_factor,
                 weight_decay):
    """AdamW on MSE(block(hidden), target), cosine schedule from lr down to lr / min_lr_factor.

    Trains the weights and the norm weights with weight_lr, scale / zero_point with quant_lr.
    The official code trains every parameter with "weight" in its name, which includes the
    RMSNorm / LayerNorm weights, so they are trained here too."""
    norms = _norm_weights(block)
    with _trainable(norms), _in_float32(block):
        groups = param_groups(block, weight_lr=weight_lr, quant_lr=quant_lr, adapter_lr=0.0,
                              weight_decay=weight_decay, extra_weights=norms)
        optimizer = torch.optim.AdamW(groups)
        total_steps = epochs * len(hidden)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: _cosine(step, total_steps, min_lr_factor))

        for _ in range(epochs):
            for h, extra, t in zip(hidden, extras, target):
                out = extra.call(block, h.float(), float32=True)
                loss = F.mse_loss(out, t.float())
                if not torch.isfinite(loss):
                    raise RuntimeError("Block-AP: non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()


def find_blocks(model: nn.Module, layer_type: type = QuantLinear) -> nn.ModuleList:
    """The transformer blocks: the longest nn.ModuleList whose every entry contains a
    `layer_type` (for HF decoders: model.model.layers)."""
    candidates = [m for m in model.modules()
                  if isinstance(m, nn.ModuleList) and len(m) > 0
                  and all(any(isinstance(x, layer_type) for x in child.modules()) for child in m)]
    if not candidates:
        raise ValueError(f"could not find the transformer blocks: no nn.ModuleList whose every entry "
                         f"contains a {layer_type.__name__}.")
    return max(candidates, key=len)


def block_linear_names(model: nn.Module) -> list[str]:
    """The paper's default targets: the names of every nn.Linear inside the transformer blocks
    (q/k/v/o_proj, gate/up/down_proj, ...), not lm_head. All blocks are read, not just the
    first: in a hybrid stack some linears exist only in some blocks."""
    blocks = find_blocks(model, layer_type=nn.Linear)
    names = set()
    for block in blocks:
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                names.add(name.split(".")[-1])
    return sorted(names)


@dataclass
class _BlockExtras:
    """What a block gets besides the hidden states: in HF, the attention mask, the rotary
    cos/sin, ... . They can differ per block (a sliding-window block gets another mask than a
    full-attention block), so they are captured for every block."""
    args: tuple
    kwargs: dict

    def call(self, block, hidden, float32=False):
        args, kwargs = self.args, self.kwargs
        if float32:
            args = _map_tensors(args, _float_to_float32)
            kwargs = {k: _map_tensors(v, _float_to_float32) for k, v in kwargs.items()}
        out = block(hidden, *args, **kwargs)
        return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def _run_block(block, hidden, extras):
    return [extra.call(block, h).detach() for h, extra in zip(hidden, extras)]


@torch.no_grad()
def _mean_mse(block, hidden, extras, target):
    """Mean MSE over all calibration batches, in the block's own dtype."""
    errors = [F.mse_loss(out.float(), t.float()).item()
              for out, t in zip(_run_block(block, hidden, extras), target)]
    return sum(errors) / len(errors)


class _StopForward(Exception):
    pass


class _InputRecorder:
    """A forward pre-hook on every block that records what the block receives.
    At the last block it stops the forward: Block-AP does not need the model's output."""

    def __init__(self, n_blocks):
        self.hidden = []                              # the first block's input, one per batch
        self.extras = [[] for _ in range(n_blocks)]   # per block: its extras, one per batch

    def record(self, index, module, args, kwargs):
        kwargs = dict(kwargs)
        hidden = args[0] if args else kwargs.pop("hidden_states")
        if index == 0:
            self.hidden.append(hidden.detach())
        # detach makes views, so blocks that share a mask do not copy it
        self.extras[index].append(_BlockExtras(
            args=_map_tensors(args[1:], torch.Tensor.detach),
            kwargs={k: _map_tensors(v, torch.Tensor.detach) for k, v in kwargs.items()}))
        if index == len(self.extras) - 1:
            raise _StopForward


@torch.no_grad()
def _capture_inputs(model, blocks, batches):
    """Run the model up to the last block and record exactly what each block receives.
    Returns (hidden states of the first block, one per batch,
             extras of every block, one list per block with one entry per batch)."""
    recorder = _InputRecorder(len(blocks))
    base = getattr(model, "base", model)
    no_cache = {"use_cache": False} if "use_cache" in inspect.signature(base.forward).parameters else {}
    handles = [block.register_forward_pre_hook(partial(recorder.record, i), with_kwargs=True)
               for i, block in enumerate(blocks)]
    try:
        for batch in batches:
            try:
                model(**batch, **no_cache)
            except _StopForward:
                pass
    finally:
        for handle in handles:
            handle.remove()

    if not recorder.hidden:
        raise ValueError("no calibration batches for Block-AP")
    if any(len(e) != len(recorder.hidden) for e in recorder.extras):
        raise ValueError("Block-AP: not every block ran for every calibration batch.")
    return recorder.hidden, recorder.extras


def _check_phase_block_ap(model):
    layers = quant_layers(model)
    if not layers or not all(m.weight is not None and m.weight.requires_grad for m in layers):
        raise ValueError("run_block_ap needs an EfficientQAT model in phase 'block_ap' "
                         "(weight trainable on every QuantLinear).")


def _norm_weights(block):
    """The block's norm weights (RMSNorm, LayerNorm): every module's own `weight` parameter,
    except in linear layers. A QuantLinear trains through its own groups, and an nn.Linear
    that is not a target stays full precision and frozen."""
    inside_quant_linear = {id(p) for m in quant_layers(block) for p in m.parameters()}
    norms = []
    for module in block.modules():
        if isinstance(module, nn.Linear):
            continue                                  # not a target: stays full precision, frozen
        weight = dict(module.named_parameters(recurse=False)).get("weight")
        if weight is not None and id(weight) not in inside_quant_linear:
            norms.append(weight)
    return norms


def _cosine(step, total, min_lr_factor):
    """LR multiplier: 1 at step 0, down on a cosine to 1 / min_lr_factor at step `total`."""
    floor = 1.0 / min_lr_factor
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(step, total) / max(total, 1)))


def _map_tensors(value, fn):
    """fn on a tensor, or on every tensor inside a tuple (rotary cos/sin come as a tuple)."""
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(v, fn) for v in value)
    return value


def _float_to_float32(t):
    return t.float() if t.is_floating_point() else t


@contextmanager
def _quantization_off(block):
    layers = quant_layers(block)
    for m in layers:
        m.quant_enabled = False
    try:
        yield
    finally:
        for m in layers:
            m.quant_enabled = True


@contextmanager
def _trainable(params):
    before = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    try:
        yield
    finally:
        for p, flag in zip(params, before):
            p.requires_grad_(flag)


@contextmanager
def _in_float32(block):
    """Train in fp32, then cast back to the dtypes the block runs in.
    In bf16, an AdamW step of lr 1e-5 is below the resolution and does not move the weights
    (the official code does the same: qlayer.float() ... qlayer.half())."""
    dtypes = {p: p.dtype for p in block.parameters()}
    with torch.no_grad():
        for p in dtypes:
            p.data = p.data.float()
    try:
        yield
    finally:
        with torch.no_grad():
            for p, dtype in dtypes.items():
                p.data = p.data.to(dtype)
