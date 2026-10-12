"""A test-only axolotl plugin for tests/test_axolotl_e2e.py: at the end of training, before
qpeft merges, it records the logits of the trained model on fixed token ids, so that a fresh
process can compare the saved integer artifact with the model that was really trained."""
import os

import torch
from axolotl.integrations.base import BasePlugin
from transformers import TrainerCallback

PROBE_FILE = "probe.pt"
PROBE_IDS = torch.arange(100, 132)[None]


class _RecordTrainedLogits(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_train_end(self, args, state, control, model=None, **kw):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits = model(input_ids=PROBE_IDS.to(model.device)).logits
        model.train(was_training)
        torch.save({"input_ids": PROBE_IDS, "logits": logits.float().cpu(), "device": str(model.device),
                    "dtype": str(model.dtype).removeprefix("torch.")},
                   os.path.join(self.output_dir, PROBE_FILE))


class ProbePlugin(BasePlugin):
    def post_trainer_create(self, cfg, trainer):
        trainer.add_callback(_RecordTrainedLogits(cfg.output_dir))


def compare_in_fresh_process(output_dir, base_model):
    """Reload the artifact the way a user would and compare with the recorded trained logits.
    Run as a script, so nothing of the training process is reused."""
    from transformers import AutoModelForCausalLM

    from qpeft import QuantModel

    probe = torch.load(os.path.join(output_dir, PROBE_FILE))
    dtype = getattr(torch, probe["dtype"])
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    model = QuantModel.from_pretrained(base, output_dir).to(probe["device"]).eval()
    with torch.no_grad():
        logits = model(input_ids=probe["input_ids"].to(probe["device"])).logits.float().cpu()
    trained = probe["logits"]
    try:
        AutoModelForCausalLM.from_pretrained(output_dir)
        plain_load = "loaded"
    except OSError as e:
        plain_load = f"refused: {str(e).splitlines()[0]}"
    return {"max_diff": (logits - trained).abs().max().item(),
            "max_abs_logit": trained.abs().max().item(),
            "argmax_agree": (logits.argmax(-1) == trained.argmax(-1)).float().mean().item(),
            "plain_transformers_load": plain_load}


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(compare_in_fresh_process(sys.argv[1], sys.argv[2])))
