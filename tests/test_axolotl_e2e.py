"""The axolotl plugin through the real `axolotl train` CLI, on a tiny model
(downloads axolotl-ai-co/tiny-qwen3-129m). Opt-in, about 4 minutes on Apple Silicon:

    QPEFT_RUN_SLOW=1 pytest tests/test_axolotl_e2e.py

axolotl trains on MPS when there is one; HF_DEACTIVATE_ASYNC_LOAD=1 avoids a transformers
crash while loading weights to MPS (transformers #48029).
"""
import json
import os
import random
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("QPEFT_RUN_SLOW") != "1", reason="set QPEFT_RUN_SLOW=1")
pytest.importorskip("axolotl")

import torch  # noqa: E402
import yaml   # noqa: E402

TESTS = Path(__file__).parent
BASE_MODEL = "axolotl-ai-co/tiny-qwen3-129m"
ENV = {**os.environ, "HF_DEACTIVATE_ASYNC_LOAD": "1", "AXOLOTL_DO_NOT_TRACK": "1", "DO_NOT_TRACK": "1",
       "WANDB_DISABLED": "true", "PYTHONPATH": os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")])}
METHODS = {
    "qa_lora": ({"method": "qa_lora", "r": 8, "lora_alpha": 16}, 1e-3),
    "peqa": ({"method": "peqa"}, 1e-4),
}
TRAINED_KIND = {"qa_lora": "adapter", "peqa": "quant"}      # the param_groups name that trains


def _config(tmp_path, method, output_dir, **settings):
    block, learning_rate = METHODS[method]
    data = tmp_path / "data.jsonl"
    if not data.exists():
        words = "the quick brown fox jumps over lazy dog adapter scale zero point group merge".split()
        rng = random.Random(0)
        data.write_text("".join(json.dumps({"text": " ".join(rng.choices(words, k=120))}) + "\n"
                                for _ in range(64)))
    config = {
        "base_model": BASE_MODEL,
        "plugins": ["axolotl_probe.ProbePlugin"],
        "adapter": "qpeft",
        "qpeft": {**block, "bits": 4, "group_size": 64,
                  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        "datasets": [{"path": str(data), "ds_type": "json", "type": "completion"}],
        "dataset_prepared_path": str(tmp_path / "prepared"),
        "val_set_size": 0,
        "sequence_len": 128,
        "micro_batch_size": 2,
        "max_steps": 6,             # fp16 AMP skips the first ~3 steps while its loss scale settles
        "learning_rate": learning_rate,
        "optimizer": "adamw_torch",
        "lr_scheduler": "cosine",
        "save_strategy": "no",
        "logging_steps": 1,
        "output_dir": str(output_dir),
        **settings,
    }
    path = output_dir.with_suffix(".yml")
    path.write_text(yaml.safe_dump(config))
    return path


def _train_command(config_path):
    return [sys.executable, "-m", "axolotl.cli.train", str(config_path)]


def _train(config_path):
    run = subprocess.run(_train_command(config_path), env=ENV, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
    return run


def _losses(checkpoint):
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    return {entry["step"]: entry["loss"] for entry in state["log_history"] if "loss" in entry}


@pytest.mark.parametrize("method", METHODS)
def test_the_artifact_reloads_in_a_fresh_process_as_trained(method, tmp_path):
    out = tmp_path / "out"
    _train(_config(tmp_path, method, out))

    assert not (out / "model.safetensors").exists()
    compare = subprocess.run([sys.executable, str(TESTS / "axolotl_probe.py"), str(out), BASE_MODEL],
                             env=ENV, capture_output=True, text=True, timeout=300)
    assert compare.returncode == 0, compare.stderr[-4000:]
    result = json.loads(compare.stdout.splitlines()[-1])
    print(method, result)
    assert result["argmax_agree"] == 1.0
    # As qpeft's own merge check in half precision: the QA-LoRA fold z - delta / s is computed in fp16.
    assert result["max_diff"] <= 1e-2 * max(result["max_abs_logit"], 1.0)
    if method == "peqa":
        assert result["max_diff"] == 0.0      # nothing to fold: the artifact is bit-exact
    assert result["plain_transformers_load"].startswith("refused: Error no file named model.safetensors")


@pytest.mark.parametrize("method", METHODS)
def test_resume_from_a_checkpoint_gives_the_same_losses(method, tmp_path):
    checkpoints = {"max_steps": 8, "save_strategy": "steps", "save_steps": 4}
    full = tmp_path / "full"
    _train(_config(tmp_path, method, full, **checkpoints))
    resumed = tmp_path / "resumed"
    shutil.copytree(full / "checkpoint-4", resumed / "checkpoint-4")
    _train(_config(tmp_path, method, resumed, resume_from_checkpoint=str(resumed / "checkpoint-4"), **checkpoints))

    full_losses, resumed_losses = _losses(full / "checkpoint-8"), _losses(resumed / "checkpoint-8")
    print(method, full_losses, resumed_losses)
    resumed_steps = range(5, 9)
    assert [resumed_losses[step] for step in resumed_steps] == [full_losses[step] for step in resumed_steps]
    # The checkpointed optimizer is qpeft's: its groups carry param_groups' names.
    optimizer = torch.load(full / "checkpoint-8" / "optimizer.pt", weights_only=True)
    assert [group["name"] for group in optimizer["param_groups"]] == [TRAINED_KIND[method]]


def test_ctrl_c_leaves_no_model_file_that_loads_with_random_weights(tmp_path):
    out = tmp_path / "out"
    config = _config(tmp_path, "peqa", out, max_steps=200)
    with subprocess.Popen(_train_command(config), env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True) as run:
        log = []
        for line in run.stdout:
            log.append(line)
            if "'loss'" in line:        # right after a log line: often while axolotl's stdout tee holds its lock
                run.send_signal(signal.SIGINT)
                break
        try:
            log.append(run.communicate(timeout=120)[0])
        except subprocess.TimeoutExpired:
            run.kill()
            pytest.fail("the run hung after Ctrl-C")
    log = "".join(log)
    assert run.returncode == 0, log[-4000:]
    assert "'loss'" in log, log[-4000:]
    assert "training stopped before the merge, so no model was written" in log, log[-4000:]
    assert not (out / "model.safetensors").exists()
    assert not (out / "qpeft_model.pt").exists()
