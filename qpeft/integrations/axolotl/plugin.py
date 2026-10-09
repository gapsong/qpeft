"""The axolotl plugin: `adapter: qpeft` trains a qpeft method in an axolotl run and leaves the
merged integer artifact (qpeft_config.json + qpeft_model.pt) in `output_dir`.

    QuantModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_model), output_dir)

The `axolotl.plugins` entry point makes axolotl load this plugin on every run, so every hook
does nothing unless the config says `adapter: qpeft`. Needs axolotl (pip install "qpeft[axolotl]").
"""
from __future__ import annotations

import os
from pathlib import Path

from axolotl.integrations.base import AdapterCapabilities, BasePlugin
from axolotl.utils.logging import get_logger
from transformers import Trainer

from ...hf_trainer import FirstStepMustMoveParams
from ...mapping import get_quant_model
from ...quant_schemes import UnsupportedSchemeError
from ...training import param_groups
from ...utils import verify_quant_model
from .args import QPeftBlock

ADAPTER = "qpeft"
PLUGIN_METHODS = ("qa_lora", "peqa")

# axolotl drops INFO lines from loggers whose name does not start with "axolotl".
LOG = get_logger("axolotl.integrations.qpeft")


class QPeftPlugin(BasePlugin):
    """One qpeft run: load_adapter quantizes the model in place, post_trainer_create gives the
    trainer qpeft's optimizer, post_train checks fake_quant == merge, merges and saves."""

    def __init__(self):
        super().__init__()
        self.qmodel = None          # the QuantModel of the current run, from load_adapter
        self.output_dir = None
        self.trainer = None

    def register(self, cfg):
        """Called with the raw config before axolotl validates it: the earliest point to refuse."""
        if cfg.get("adapter") != ADAPTER:
            return
        method = (cfg.get("qpeft") or {}).get("method")
        if method == "efficient_qat":
            raise UnsupportedSchemeError(
                "adapter: qpeft does not run method efficient_qat in axolotl yet: its Block-AP phase "
                f"is not wired into the plugin. Supported here: {list(PLUGIN_METHODS)}. "
                "For EfficientQAT use qpeft.QATTrainer (examples/qat_trainer.py).")

    def get_input_args(self):
        return "qpeft.integrations.axolotl.args.QPeftArgs"

    def get_adapter_capabilities(self):
        return [AdapterCapabilities(name=ADAPTER)]

    def load_adapter(self, model, cfg, inference=False, config_only=False):
        if cfg.adapter != ADAPTER:
            return None
        if inference or config_only:
            raise UnsupportedSchemeError(
                "adapter: qpeft only trains; loading for inference or as a bare adapter config "
                "is not supported. Load the artifact with qpeft.QuantModel.from_pretrained.")
        (config,) = QPeftBlock.model_validate(dict(cfg.qpeft)).quant_configs()
        self.qmodel = get_quant_model(model, config)       # swaps QuantLinear into `model` in place
        self.output_dir = Path(cfg.output_dir).resolve()
        self.trainer = None
        self._guard_final_saves(model)
        LOG.info(f"qpeft: {config.quant_tuning_type.value} on {len(self.qmodel.quant_layers())} layers, "
                 f"bits={config.bits}, group_size={config.group_size}")
        return model, None

    def post_trainer_create(self, cfg, trainer):
        """axolotl never calls the plugin create_optimizer hook, but HF's Trainer keeps an
        optimizer that is already set, so the optimizer is set here."""
        if cfg.adapter != ADAPTER:
            return
        lr = trainer.args.learning_rate     # each method trains one kind of parameter
        groups = param_groups(trainer.model, weight_lr=lr, quant_lr=lr, adapter_lr=lr,
                              weight_decay=trainer.args.weight_decay)
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(trainer.args, trainer.model)
        trainer.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        trainer.add_callback(FirstStepMustMoveParams())
        self.trainer = trainer

    def post_train(self, cfg, model):
        """axolotl calls this twice, before and after its own final save; the second call finds
        the model merged and does nothing."""
        if cfg.adapter != ADAPTER or self.qmodel.is_merged:
            return
        errors = verify_quant_model(self.qmodel)        # raises MergeMismatchError: nothing is saved
        self.qmodel.merge_and_unload()
        if self.trainer.args.should_save:
            self.qmodel.save_pretrained(self.output_dir)
        LOG.info(f"qpeft: fake_quant == merge on {len(errors)} layers (max |delta| "
                 f"{max(errors.values()):.2e}); integer artifact saved to {self.output_dir}")

    def _guard_final_saves(self, model):
        """Until axolotl lets a plugin own the final save: axolotl's final save and its Ctrl-C save
        call model.save_pretrained(output_dir), which writes the QuantLinear state as
        model.safetensors, and transformers loads that file with only a warning and random weights.
        So a save into output_dir writes no weights: there the qpeft artifact is the model.
        Trainer checkpoints (output_dir/checkpoint-N) are saved as usual, for resuming."""
        hf_save_pretrained = model.save_pretrained

        def save_pretrained(save_directory, *args, **kwargs):
            is_checkpoint = Path(save_directory).resolve() != self.output_dir
            if is_checkpoint and not self.qmodel.is_merged:
                return hf_save_pretrained(save_directory, *args, **kwargs)
            if self.qmodel.is_merged:
                LOG.info(f"qpeft: no transformers weights written to {save_directory}; "
                         f"the model is the qpeft artifact in {self.output_dir}")
            else:
                # This runs in axolotl's Ctrl-C handler, which can interrupt axolotl's stdout tee while
                # it holds a non-reentrant lock: logging here can deadlock, os.write cannot.
                os.write(2, (f"[qpeft] training stopped before the merge, so no model was written to "
                             f"{save_directory}. Resume from the last checkpoint to finish the run.\n").encode())

        model.save_pretrained = save_pretrained
