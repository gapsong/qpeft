"""What any Hugging Face Trainer needs from qpeft, whoever owns the training loop
(QATTrainer, or an integration that drives a Trainer it did not build).
Needs `transformers` (pip install "qpeft[train]").

    batches = block_ap_batches(trainer, train_size=4096, seqlen=2048, batch_size=2)
    run_block_ap(qmodel, batches, ...)
    trainer.add_callback(FirstStepMustMoveParams())
"""
from __future__ import annotations

import torch
from transformers import TrainerCallback

LABEL_KEYS = ("labels", "label", "label_ids")


def block_ap_batches(trainer, *, train_size: int, seqlen: int, batch_size: int) -> list[dict]:
    """Block-AP calibration batches from the trainer's training data, on the trainer's device:
    labels removed, sequences cut to `seqlen`, re-split into batches of at most `batch_size` rows,
    `train_size` rows in total (fewer if the data has fewer)."""
    batches, rows_taken = [], 0
    for batch in trainer.get_train_dataloader():
        batch = trainer._prepare_inputs(batch)
        batch = {k: _cut_sequence(v, seqlen) for k, v in batch.items() if k not in LABEL_KEYS}
        for start in range(0, _num_rows(batch), batch_size):
            size = min(batch_size, train_size - rows_taken)
            part = {k: (v[start:start + size] if torch.is_tensor(v) else v) for k, v in batch.items()}
            batches.append(part)
            rows_taken += _num_rows(part)
            if rows_taken >= train_size:
                return batches
    return batches


def _cut_sequence(value, seqlen):
    return value[:, :seqlen] if torch.is_tensor(value) and value.dim() == 2 else value


def _num_rows(batch):
    return next(v for v in batch.values() if torch.is_tensor(v)).shape[0]


class FirstStepMustMoveParams(TrainerCallback):
    """Stops training when the first optimizer step with lr > 0 moved no trainable parameter,
    or when training ends without such a step (a warmup starts at lr 0).
    Either way the model could not learn anything."""

    HINT = "check the learning rates and the trainable set."

    def __init__(self):
        self.snapshot = None        # trainable params before the first lr > 0 step; None once checked
        self.checking = False
        self.scale_before = None    # the fp16 GradScaler's scale before the step being checked

    def on_train_begin(self, args, state, control, model=None, **kw):
        self.snapshot = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def on_step_begin(self, args, state, control, optimizer=None, **kw):
        # The lr the coming optimizer step will use (the scheduler steps after it).
        self.checking = self.snapshot is not None and any(g["lr"] > 0 for g in optimizer.param_groups)
        if self.checking:
            self.scale_before = _grad_scale(optimizer)

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kw):
        if not self.checking:
            return
        # An fp16 overflow skipped this step; check the next one. Accelerate's step_was_skipped covers
        # DeepSpeed, but a fused optimizer (HF's default AdamW) never sets it; there the GradScaler
        # shows the skip by halving its scale.
        if getattr(optimizer, "step_was_skipped", False) or _grad_scale(optimizer) < self.scale_before:
            return
        moved = any(not torch.equal(p, self.snapshot[n])
                    for n, p in model.named_parameters() if n in self.snapshot)
        self.snapshot, self.checking = None, False
        if not moved:
            raise RuntimeError(f"no trainable parameter changed in the first step with lr > 0 -- {self.HINT}")

    def on_train_end(self, args, state, control, **kw):
        if self.snapshot is not None:
            raise RuntimeError(f"training ended without an optimizer step at lr > 0 -- {self.HINT}")


def _grad_scale(optimizer):
    """The scale of the fp16 GradScaler on accelerate's optimizer wrapper; 1.0 without one."""
    scaler = getattr(optimizer, "scaler", None)
    return scaler.get_scale() if scaler is not None else 1.0
