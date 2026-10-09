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
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from transformers.utils import can_return_loss, find_labels

from .block_ap import find_blocks, run_block_ap
from .hf_trainer import FirstStepMustMoveParams, block_ap_batches
from .mapping import get_quant_model
from .peft_model import QuantModel
from .quant_schemes import UnsupportedSchemeError
from .training import param_groups
from .tuners.efficient_qat import EfficientQATConfig
from .utils import verify_quant_model

SAVE_HINT = ("only the merged int artifact is saved: train, then call trainer.save_model(output_dir) "
             "and upload that directory yourself.")


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


class QATTrainer(Trainer):
    def __init__(self, model=None, args: Optional[QATTrainingArguments] = None,
                 quant_config: Optional[EfficientQATConfig] = None, **kwargs):
        model = _prepare_model(model, quant_config)
        args = args if args is not None else QATTrainingArguments(output_dir="qpeft_output")
        _check_args(args)
        _fill_bit_dependent_lrs(args, model.config.bits)
        super().__init__(model=model, args=args, **kwargs)
        self._point_trainer_at_the_hf_model()
        self.add_callback(FirstStepMustMoveParams())
        self._log_setup()

    def train(self, *args, **kwargs):
        """Block-AP (if the model is in that phase), then E2E-QP with the normal Trainer loop,
        then the merge check on every layer."""
        config = self.model.config
        if config.phase == "block_ap":
            self._run_block_ap()
            self.model.set_phase(dataclasses.replace(config, phase="e2e_qp"))   # freezes the codes
            print("[QATTrainer] Block-AP done -> codes frozen -> E2E-QP (scale only)")
        self.optimizer, self.lr_scheduler = None, None
        result = super().train(*args, **kwargs)
        errors = verify_quant_model(self.model)          # raises if a layer's merge would differ
        print(f"[QATTrainer] merge equivalence OK on {len(errors)} layers, "
              f"max|delta|={max(errors.values()):.2e}")
        return result

    def save_model(self, output_dir=None, _internal_call=False):
        """Merge (irreversible) and write the int artifact: qpeft_config.json +
        qpeft_model.pt. Load with QuantModel.from_pretrained(base_model, output_dir)."""
        if _internal_call:
            raise RuntimeError("Trainer-internal saves (mid-training checkpoints, push_to_hub, "
                               f"hyperparameter search) are not supported; {SAVE_HINT}")
        self.model.merge_and_unload()
        if self.args.should_save:
            self.model.save_pretrained(output_dir or self.args.output_dir)

    def push_to_hub(self, *args, **kwargs):
        raise RuntimeError(f"push_to_hub is not supported; {SAVE_HINT}")

    def create_optimizer(self, model=None):
        """AdamW with one group per parameter kind, each with its own lr (official EfficientQAT).
        `model`: as in HF, the model to optimize when it is not self.model (a wrapped model
        when HF delays optimizer creation)."""
        if self.optimizer is None:
            a = self.args
            model = self.model if model is None else model
            quant_lr = a.e2e_lr if self.model.config.phase == "e2e_qp" else a.quant_lr
            groups = param_groups(model, weight_lr=a.weight_lr, quant_lr=quant_lr,
                                  adapter_lr=0.0, weight_decay=a.weight_decay)
            for g in groups:
                g.pop("name")
            self.optimizer = torch.optim.AdamW(groups)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """The normal loss, but a loss of 0 or NaN stops training: nothing can be learned from it
        (peft PR #2571: a loss stuck at 0.0 went unnoticed). A batch whose labels are all -100
        (a masked prompt, cut off) has loss 0 legitimately, so it is let through."""
        out = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        loss = (out[0] if return_outputs else out).detach().float().item()
        labels = inputs.get("labels")
        has_labelled_tokens = labels is None or bool((labels != -100).any())
        if math.isnan(loss) or math.isinf(loss) or (loss == 0.0 and has_labelled_tokens):
            raise RuntimeError(f"training loss is {loss} -- nothing can be learned. "
                               "Check group_size, the data and the labels.")
        return out

    def _run_block_ap(self):
        a = self.args
        batches = block_ap_batches(self, train_size=a.block_ap_train_size, seqlen=a.block_ap_seqlen,
                                   batch_size=a.block_ap_batch_size)
        run_block_ap(self.model, batches, epochs=a.block_ap_epochs,
                     weight_lr=a.weight_lr, quant_lr=a.quant_lr,
                     min_lr_factor=a.block_ap_min_lr_factor, weight_decay=a.weight_decay)

    def _point_trainer_at_the_hf_model(self):
        """Trainer inspects `self.model` for a few things. Here that is the QuantModel wrapper,
        so point those checks at the wrapped HF model instead."""
        base = self.model.base
        # Trainer writes use_cache onto model.config, which is the qpeft config here.
        vars(self.model.config).pop("use_cache", None)
        if getattr(base, "config", None) is not None:
            base.config.use_cache = self.args.use_cache
        # QuantModel.forward takes (*args, **kwargs), which says nothing about the HF model.
        self.model_accepts_loss_kwargs = getattr(
            base, "accepts_loss_kwargs",
            any(p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(base.forward).parameters.values()))
        # The same for the label names and "can the model return a loss"; without them
        # evaluate() and predict() compute no loss.
        if self.args.label_names is None:
            self.label_names = find_labels(base.__class__)
        self.can_return_loss = can_return_loss(base.__class__)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            params = inspect.signature(self.model.base.forward).parameters
            self._signature_columns = list(params) + list({"label", "label_ids", *self.label_names})

    def _log_setup(self):
        a, c = self.args, self.model.config
        print(f"[QATTrainer] {type(c).__name__} bits={c.bits} group_size={c.group_size} "
              f"backend={c.backend} layers={len(self.model.quant_layers())}")
        print(f"[QATTrainer] targets={c.target_modules}")
        print(f"[QATTrainer] Block-AP: epochs={a.block_ap_epochs} weight_lr={a.weight_lr} "
              f"quant_lr={a.quant_lr} samples<={a.block_ap_train_size} seqlen<={a.block_ap_seqlen}; "
              f"E2E-QP: lr={a.e2e_lr} max_grad_norm={a.max_grad_norm}")


def _prepare_model(model, quant_config) -> QuantModel:
    """Plain HF model (+ optional EfficientQATConfig) -> QuantModel.
    A QuantModel made with get_quant_model is taken as it is. Everything else is refused."""
    if isinstance(model, QuantModel):
        if quant_config is not None:
            raise ValueError("model is already a QuantModel; pass quant_config only with a plain model.")
    elif isinstance(model, nn.Module):
        config = quant_config if quant_config is not None else EfficientQATConfig()
        if not isinstance(config, EfficientQATConfig):
            raise UnsupportedSchemeError(
                f"QATTrainer only supports EfficientQATConfig, got {type(config).__name__}. Refusing.")
        if config.target_modules is None:
            config = dataclasses.replace(config, target_modules=_block_linear_names(model))
        model = get_quant_model(model, config)
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


def _block_linear_names(model: nn.Module) -> list[str]:
    """The paper's default targets: every nn.Linear inside the transformer blocks
    (q/k/v/o_proj, gate/up/down_proj, ...), not lm_head. All blocks are read, not just the
    first: in a hybrid stack some linears exist only in some blocks."""
    try:
        blocks = find_blocks(model, layer_type=nn.Linear)
    except ValueError as e:
        raise ValueError(f"{e} Pass EfficientQATConfig(target_modules=[...]) explicitly.") from None
    names = set()
    for block in blocks:
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                names.add(name.split(".")[-1])
    return sorted(names)


def _check_args(args):
    if not isinstance(args, QATTrainingArguments):
        raise TypeError("QATTrainer needs QATTrainingArguments (it carries the qpeft learning rates).")
    if getattr(args.save_strategy, "value", args.save_strategy) != "no":
        raise ValueError("mid-training checkpoints are not supported (they are not the int "
                         "artifact); keep save_strategy='no' and call trainer.save_model() at the end.")
    if args.push_to_hub:
        raise ValueError(f"push_to_hub is not supported; {SAVE_HINT}")
    if args.learning_rate != _default_of(TrainingArguments, "learning_rate"):
        raise ValueError("QATTrainer does not use learning_rate; it trains with its own lrs: "
                         "e2e_lr (E2E-QP scales), weight_lr and quant_lr (Block-AP).")


def _default_of(dataclass_type, field_name):
    return next(f.default for f in dataclasses.fields(dataclass_type) if f.name == field_name)


def _fill_bit_dependent_lrs(args, bits):
    """Official EfficientQAT: 2-bit models train with twice the lr."""
    if args.weight_lr is None:
        args.weight_lr = 2e-5 if bits == 2 else 1e-5
    if args.e2e_lr is None:
        args.e2e_lr = 2e-5 if bits == 2 else 1e-5
