"""Basic usage. Runs the construction path end to end (no training) so you can
see the two methods share one interface. Training: train_efficient_qat.py,
train_qa_lora.py and qat_trainer.py."""
import copy

import torch.nn as nn

from qpeft import QALoraConfig, efficient_qat_schedule, get_quant_model


def demo():
    base = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 512))

    # --- EfficientQAT: two PEFT-style configs run in order (not a "Recipe") ---
    # The second get_quant_model call switches the SAME layers to the next phase:
    # the integer codes are frozen and only the scale stays trainable.
    model = copy.deepcopy(base)
    for cfg in efficient_qat_schedule(bits=2, group_size=64):
        model = get_quant_model(model, cfg)
        trainable = [p.value for p in cfg.trainable_params]
        print(f"[efficient_qat] phase={cfg.phase:<9} trainable={trainable}")

    # --- QA-LoRA: one config; the adapter folds into the zero-points on merge ---
    model = get_quant_model(copy.deepcopy(base), QALoraConfig(bits=4, group_size=32, r=64))
    print("[qa_lora]       built; merge_and_unload() returns an int model (stays quantized)")


if __name__ == "__main__":
    demo()
