"""QATTrainer: EfficientQAT with the ease of use of peft + Trainer.
See docs/specs/qat_trainer.md. Needs `transformers` (pip install "qpeft[train]").

    model   = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    trainer = QATTrainer(model=model,
                         quant_config=EfficientQATConfig(bits=2, group_size=64),
                         args=QATTrainingArguments(output_dir="out"),
                         train_dataset=ds, data_collator=collator)
    trainer.train()          # Block-AP -> codes frozen -> E2E-QP, merge check at the end
    trainer.save_model()     # merge -> int artifact in output_dir

Only EfficientQAT. QA-LoRA and other methods are refused.
"""
from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from transformers import Trainer, TrainerCallback, TrainingArguments

from .block_ap import run_block_ap
from .mapping import get_quant_model
from .peft_model import QuantModel
from .quant_schemes import UnsupportedSchemeError
from .training import param_groups
from .tuners.efficient_qat import EfficientQATConfig
from .utils import verify_quant_model


@dataclass
class QATTrainingArguments(TrainingArguments):
    """TrainingArguments + the official EfficientQAT defaults (main_block_ap.py,
    main_e2e_qp.py, examples/e2e_qp/*.sh). `None` means "resolve from the model's
    bit-width"."""
    quant_lr: float = field(default=1e-4, metadata={"help": "scale / zero_point lr (Block-AP)"})
    weight_lr: Optional[float] = field(default=None, metadata={"help": "1e-5, 2e-5 if bits == 2"})
    e2e_lr: Optional[float] = field(default=None, metadata={"help": "E2E-QP scale lr: 1e-5, 2e-5 if bits == 2"})
    block_ap_epochs: int = 2
    block_ap_train_size: int = 4096
    block_ap_seqlen: int = 2048
    block_ap_batch_size: int = 2                  # official --batch_size (Block-AP)
    block_ap_min_lr_factor: float = 20
    # E2E-QP, as the official scripts run it:
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    lr_scheduler_type: str = "cosine"
    warmup_steps: float = 0.03                    # < 1 = fraction of the steps (warmup_ratio=0.03)
    max_grad_norm: float = 0.3
    save_strategy: str = "no"                     # a mid-training checkpoint is not the int artifact


def _block_linear_names(model: nn.Module) -> list[str]:
    """Paper default target: every nn.Linear inside the transformer blocks
    (q/k/v/o_proj, gate/up/down_proj, ...), not lm_head."""
    best = None
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) > 0 and all(
                any(isinstance(x, nn.Linear) for x in child.modules()) for child in m):
            if best is None or len(m) > len(best):
                best = m
    if best is None:
        raise ValueError("could not find the transformer blocks; pass "
                         "EfficientQATConfig(target_modules=[...]) explicitly.")
    return sorted({n.split(".")[-1] for n, x in best[0].named_modules() if isinstance(x, nn.Linear)})


def _prepare_model(model, quant_config):
    """HF model (+ optional EfficientQATConfig) -> QuantModel. A QuantModel built
    with get_quant_model is accepted as is. Everything else is refused."""
    if isinstance(model, QuantModel):
        if quant_config is not None:
            raise ValueError("model is already a QuantModel; pass quant_config only with a plain model.")
    elif isinstance(model, nn.Module):
        cfg = quant_config if quant_config is not None else EfficientQATConfig()
        if not isinstance(cfg, EfficientQATConfig):
            raise UnsupportedSchemeError(
                f"QATTrainer only supports EfficientQATConfig, got {type(cfg).__name__}. Refusing.")
        if cfg.target_modules is None:
            cfg = dataclasses.replace(cfg, target_modules=_block_linear_names(model))
        model = get_quant_model(model, cfg)
    else:
        raise TypeError(f"QATTrainer needs a model, got {type(model).__name__}.")

    if not isinstance(model.config, EfficientQATConfig):
        raise UnsupportedSchemeError(
            f"QATTrainer only supports EfficientQAT, got {type(model.config).__name__}. Refusing.")
    if not model.quant_layers():
        raise ValueError(f"no layer was quantized: target_modules {model.config.target_modules!r} "
                         "matched nothing.")
    if any(m.merged for m in model.quant_layers()):
        raise ValueError("model is merged (an int artifact) and cannot be trained.")
    return model


class _GuardCallback(TrainerCallback):
    """At least one trainable parameter must move in the first optimizer step
    that has a non-zero learning rate (a warmup starts at lr 0). If training
    ends without such a step, nothing could have been learned either."""

    _MSG = "check the learning rates and the trainable set."

    def __init__(self, trainer):
        self.trainer, self.snapshot, self.checking = trainer, None, False

    def on_train_begin(self, args, state, control, **kw):
        self.snapshot = {n: p.detach().clone() for n, p in self.trainer.model.named_parameters()
                         if p.requires_grad}

    def on_step_begin(self, args, state, control, **kw):
        # The lr the coming optimizer step will use (the scheduler steps after it).
        self.checking = self.snapshot is not None and any(
            g["lr"] > 0 for g in self.trainer.optimizer.param_groups)

    def on_step_end(self, args, state, control, **kw):
        if not self.checking:
            return
        moved = any(not torch.equal(p, self.snapshot[n])
                    for n, p in self.trainer.model.named_parameters() if n in self.snapshot)
        self.snapshot, self.checking = None, False
        if not moved:
            raise RuntimeError(f"no trainable parameter changed in the first step with lr > 0 -- {self._MSG}")

    def on_train_end(self, args, state, control, **kw):
        if self.snapshot is not None:
            raise RuntimeError(f"training ended without an optimizer step at lr > 0 -- {self._MSG}")


class QATTrainer(Trainer):
    def __init__(self, model=None, args: Optional[QATTrainingArguments] = None,
                 quant_config: Optional[EfficientQATConfig] = None, **kwargs):
        model = _prepare_model(model, quant_config)
        if args is None:
            args = QATTrainingArguments(output_dir="qpeft_output")
        if not isinstance(args, QATTrainingArguments):
            raise TypeError("QATTrainer needs QATTrainingArguments (it carries the qpeft learning rates).")
        if getattr(args.save_strategy, "value", args.save_strategy) != "no":
            raise ValueError("mid-training checkpoints are not supported (they are not the int "
                             "artifact); keep save_strategy='no' and call trainer.save_model() at the end.")
        if args.push_to_hub:
            raise ValueError(f"push_to_hub is not supported; {self._SAVE_MSG}")
        bits = model.config.bits
        if args.weight_lr is None:
            args.weight_lr = 2e-5 if bits == 2 else 1e-5
        if args.e2e_lr is None:
            args.e2e_lr = 2e-5 if bits == 2 else 1e-5
        super().__init__(model=model, args=args, **kwargs)
        base = model.base
        # Trainer sets use_cache on model.config, which is the qpeft config here.
        vars(model.config).pop("use_cache", None)
        if getattr(base, "config", None) is not None:
            base.config.use_cache = self.args.use_cache
        # QuantModel.forward is (*args, **kwargs): inspect the wrapped HF model instead.
        self.model_accepts_loss_kwargs = getattr(
            base, "accepts_loss_kwargs",
            any(p.kind == inspect.Parameter.VAR_KEYWORD
                for p in inspect.signature(base.forward).parameters.values()))
        self.add_callback(_GuardCallback(self))
        self._log_setup()

    # -- setup / logging -----------------------------------------------------------
    def _log_setup(self):
        a, c = self.args, self.model.config
        print(f"[QATTrainer] {type(c).__name__} bits={c.bits} group_size={c.group_size} "
              f"backend={c.backend} layers={len(self.model.quant_layers())}")
        print(f"[QATTrainer] targets={c.target_modules}")
        print(f"[QATTrainer] Block-AP: epochs={a.block_ap_epochs} weight_lr={a.weight_lr} "
              f"quant_lr={a.quant_lr} samples<={a.block_ap_train_size} seqlen<={a.block_ap_seqlen}; "
              f"E2E-QP: lr={a.e2e_lr} max_grad_norm={a.max_grad_norm}")

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            params = inspect.signature(self.model.base.forward).parameters
            self._signature_columns = list(params) + list({"label", "label_ids", *self.label_names})

    # -- optimizer: one group per parameter kind -----------------------------------
    def create_optimizer(self):
        if self.optimizer is None:
            a, c = self.args, self.model.config
            quant_lr = a.e2e_lr if c.phase == "e2e_qp" else a.quant_lr
            groups = param_groups(self.model, weight_lr=a.weight_lr, quant_lr=quant_lr,
                                  adapter_lr=0.0, weight_decay=a.weight_decay)
            for g in groups:
                g.pop("name")
            self.optimizer = torch.optim.AdamW(groups)
        return self.optimizer

    # -- loss guard (peft PR #2571: loss sat at 0.0 unnoticed) ---------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        loss = out[0] if return_outputs else out
        value = loss.detach().float()
        if not torch.isfinite(value) or value.item() == 0.0:
            raise RuntimeError(f"training loss is {value.item()} -- nothing can be learned. "
                               "Check group_size, the data and the labels.")
        return out

    # -- phases --------------------------------------------------------------------
    def _block_ap_batches(self):
        """Calibration batches from train_dataset: truncated to block_ap_seqlen,
        re-split to block_ap_batch_size rows, at most block_ap_train_size rows."""
        a, batches, n = self.args, [], 0
        for batch in self.get_train_dataloader():
            batch = self._prepare_inputs(batch)
            batch = {k: (v[:, :a.block_ap_seqlen] if torch.is_tensor(v) and v.dim() == 2 else v)
                     for k, v in batch.items() if k not in ("labels", "label", "label_ids")}
            rows = next(v for v in batch.values() if torch.is_tensor(v)).shape[0]
            for start in range(0, rows, a.block_ap_batch_size):
                take = min(a.block_ap_batch_size, rows - start, a.block_ap_train_size - n)
                batches.append({k: (v[start:start + take] if torch.is_tensor(v) else v)
                                for k, v in batch.items()})
                n += take
                if n >= a.block_ap_train_size:
                    return batches
        return batches

    def train(self, *args, **kwargs):
        c, a = self.model.config, self.args
        if c.phase == "block_ap":
            run_block_ap(self.model, self._block_ap_batches(), epochs=a.block_ap_epochs,
                         weight_lr=a.weight_lr, quant_lr=a.quant_lr,
                         min_lr_factor=a.block_ap_min_lr_factor, weight_decay=a.weight_decay)
            # Hand-over: freeze the codes, train only the scale (official E2E-QP).
            self.model.set_phase(dataclasses.replace(c, phase="e2e_qp"))
            print("[QATTrainer] Block-AP done -> codes frozen -> E2E-QP (scale only)")
        self.optimizer, self.lr_scheduler = None, None
        result = super().train(*args, **kwargs)
        errs = verify_quant_model(self.model)          # red -> raises, never hands back a model
        print(f"[QATTrainer] merge equivalence OK on {len(errs)} layers, "
              f"max|delta|={max(errs.values()):.2e}")
        return result

    # -- saving --------------------------------------------------------------------
    _SAVE_MSG = ("only the merged int artifact is saved: train, then call trainer.save_model(output_dir) "
                 "and upload that directory yourself.")

    def save_model(self, output_dir=None, _internal_call=False):
        """Merge (irreversible) and write the int artifact: qpeft_config.json +
        qpeft_model.pt. Load with QuantModel.from_pretrained(base_model, output_dir)."""
        if _internal_call:
            raise RuntimeError("Trainer-internal saves (mid-training checkpoints, push_to_hub, "
                               f"hyperparameter search) are not supported; {self._SAVE_MSG}")
        output_dir = output_dir or self.args.output_dir
        self.model.merge_and_unload()
        if self.args.should_save:
            self.model.save_pretrained(output_dir)

    def push_to_hub(self, *args, **kwargs):
        raise RuntimeError(f"push_to_hub is not supported; {self._SAVE_MSG}")
