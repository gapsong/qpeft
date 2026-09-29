"""An axolotl plugin that measures a training run the same way in every arm:
tokens/s over the steps after WARMUP_STEPS, peak CUDA memory (allocated by this process) during
training, and the loss log.
Writes <output_dir>/measure.json.

QPEFT_TORCH_PATH=1 switches qpeft's Triton kernel off (the torch path), for the kernel comparison."""
import json
import os
import time
from pathlib import Path

import torch
from axolotl.integrations.base import BasePlugin
from transformers import TrainerCallback

WARMUP_STEPS = 10


class MeasurePlugin(BasePlugin):
    def post_model_load(self, cfg, model):
        if os.environ.get("QPEFT_TORCH_PATH") == "1":
            from qpeft.tuners.tuners_utils import QuantLinear
            QuantLinear.use_triton_kernel = False

    def add_callbacks_post_trainer(self, cfg, trainer):
        tokens_per_step = cfg.micro_batch_size * cfg.gradient_accumulation_steps * cfg.sequence_len
        return [MeasureCallback(Path(cfg.output_dir) / "measure.json", tokens_per_step)]


class MeasureCallback(TrainerCallback):
    def __init__(self, path, tokens_per_step):
        self.path = path
        self.tokens_per_step = tokens_per_step
        self.start = None
        self.peak_bytes = 0

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        # Read every step: axolotl resets the peak statistics each time it logs.
        self.peak_bytes = max(self.peak_bytes, torch.cuda.max_memory_allocated())
        if state.global_step == WARMUP_STEPS:
            torch.cuda.synchronize()
            self.start = time.perf_counter()

    def on_train_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        seconds = time.perf_counter() - self.start
        timed_steps = state.global_step - WARMUP_STEPS
        result = {
            "tokens_per_s": timed_steps * self.tokens_per_step / seconds,
            "timed_steps": timed_steps,
            "peak_memory_gib": self.peak_bytes / 2 ** 30,
            "torch_path": os.environ.get("QPEFT_TORCH_PATH") == "1",
            "losses": [(entry["step"], entry["loss"]) for entry in state.log_history if "loss" in entry],
        }
        self.path.write_text(json.dumps(result, indent=1))
