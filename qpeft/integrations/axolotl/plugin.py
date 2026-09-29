"""qpeft as an axolotl plugin: QA-LoRA and PEQA whose merge stays quantized.

    plugins:
      - qpeft.integrations.axolotl.QpeftPlugin
    adapter: qpeft
    qpeft:
      method: qa_lora            # or peqa
      bits: 4
      group_size: 64
    lora_r: 16                   # QA-LoRA reads the usual LoRA fields
    lora_alpha: 32
    lora_dropout: 0.05
    lora_target_linear: true     # or lora_target_modules: [q_proj, v_proj, ...]

The base model is loaded in full precision; qpeft quantizes the targeted linears itself.
After training the model is checked (fake_quant == merge), merged into the integer artifact
and written with QuantModel.save_pretrained to <output_dir>/qpeft.
"""
from __future__ import annotations

import os
from pathlib import Path

from axolotl.integrations.base import AdapterCapabilities, BasePlugin
from axolotl.utils.logging import get_logger

from ...block_ap import block_linear_names
from ...mapping import get_quant_model
from ...tuners.peqa import PEQAConfig
from ...tuners.qa_lora import QALoraConfig
from ...utils import verify_quant_model
from .args import QpeftConfig, QpeftMethod

LOG = get_logger("axolotl.integrations.qpeft")      # axolotl shows only its own loggers

ADAPTER = "qpeft"
ARTIFACT_DIR = "qpeft"


class QpeftPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.quant_model = None
        self.artifact_saved = False

    def get_input_args(self) -> str:
        return "qpeft.integrations.axolotl.args.QpeftArgs"

    def get_adapter_capabilities(self) -> list[AdapterCapabilities]:
        return [AdapterCapabilities(name=ADAPTER)]

    def load_adapter(self, model, cfg, inference=False, config_only=False):
        """Quantize the targeted linears of the HF model in place (QuantLinear) and return the
        same HF model, so the rest of axolotl sees an ordinary PreTrainedModel."""
        if cfg.adapter != ADAPTER:
            return None
        check_supported(cfg)
        if config_only:
            return None, None
        quant_config = quant_config_from(cfg, model)
        self.quant_model = get_quant_model(model, quant_config)
        if not self.quant_model.quant_layers():
            raise ValueError(f"adapter: qpeft quantized no layer: target_modules "
                             f"{quant_config.target_modules!r} matched nothing.")
        n_layers = len(self.quant_model.quant_layers())
        LOG.info(f"qpeft: {quant_config.quant_tuning_type.value} on {n_layers} linears, "
                 f"bits={quant_config.bits} group_size={quant_config.group_size}")
        return model, None

    def post_train(self, cfg, model):
        """Check that the merged integer model computes what was trained, merge, and save it.
        axolotl calls post_train twice (after trainer.train() and at the end of train()),
        so the second call does nothing."""
        if self.quant_model is None or self.artifact_saved:
            return
        errors = verify_quant_model(self.quant_model)
        LOG.info(f"qpeft: merge equivalence OK on {len(errors)} layers, max|delta|={max(errors.values()):.2e}")
        self.quant_model.merge_and_unload()
        artifact = Path(cfg.output_dir) / ARTIFACT_DIR
        self.quant_model.save_pretrained(artifact)
        self.artifact_saved = True
        LOG.info(f"qpeft: integer artifact saved to {artifact} "
                 "(load with qpeft.QuantModel.from_pretrained(base_model, path))")


def check_supported(cfg):
    """Refuse what the plugin does not do, instead of silently training something else."""
    if cfg.load_in_4bit or cfg.load_in_8bit or cfg.gptq:
        raise ValueError("adapter: qpeft quantizes the full-precision model itself; "
                         "remove load_in_4bit / load_in_8bit / gptq.")
    if cfg.fsdp_config or cfg.fsdp or cfg.deepspeed:
        raise ValueError("adapter: qpeft is not tested with FSDP or DeepSpeed yet; train on one GPU.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("adapter: qpeft is not tested on multiple GPUs yet; train on one GPU.")
    if cfg.relora:
        raise ValueError("adapter: qpeft does not support ReLoRA.")
    if cfg.lora_modules_to_save:
        raise ValueError("adapter: qpeft trains only the quantization parameters or the adapter; "
                         "remove lora_modules_to_save.")
    if qpeft_config(cfg).method == QpeftMethod.PEQA and (cfg.weight_decay or 0) > 0:
        raise ValueError("qpeft method peqa trains the quantization scales; weight_decay would "
                         "shrink them. Set weight_decay: 0.")


def qpeft_config(cfg) -> QpeftConfig:
    block = cfg.qpeft
    if block is None:
        return QpeftConfig()
    if isinstance(block, QpeftConfig):
        return block
    return QpeftConfig(**dict(block))


def quant_config_from(cfg, model):
    q = qpeft_config(cfg)
    common = dict(bits=q.bits, group_size=q.group_size, backend=q.backend,
                  target_modules=target_modules_from(cfg, model))
    if q.method == QpeftMethod.QA_LORA:
        return QALoraConfig(**common, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                            lora_dropout=cfg.lora_dropout or 0.0)
    return PEQAConfig(**common)


def target_modules_from(cfg, model):
    """axolotl's lora_target_linear / "all-linear" means every linear of the decoder blocks
    (not lm_head), the same default as QATTrainer; otherwise the listed names, matched as peft."""
    targets = cfg.lora_target_modules
    if cfg.lora_target_linear or targets == "all-linear":
        return block_linear_names(model)
    if not targets:
        raise ValueError("adapter: qpeft needs lora_target_modules or lora_target_linear: true.")
    return list(targets) if not isinstance(targets, str) else targets
