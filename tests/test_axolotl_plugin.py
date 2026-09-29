"""The axolotl plugin (qpeft/integrations/axolotl): its hooks called the way axolotl calls them,
on a tiny random Llama, plus one real `axolotl train` run (slow: downloads SmolLM2-135M).
Needs axolotl (Python >= 3.12); skipped where it is not installed."""
import os
import subprocess
import sys

import pytest

pytest.importorskip("axolotl")

import torch                                                    # noqa: E402
from axolotl.utils.dict import DictDefault                      # noqa: E402

from qpeft import QuantModel, UnsupportedSchemeError            # noqa: E402
from qpeft.integrations.axolotl import QpeftPlugin              # noqa: E402
from qpeft.tuners.tuners_utils import QuantLinear               # noqa: E402

BLOCK_LINEARS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def _llama(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, vocab_size=320, max_position_embeddings=64))


def _cfg(tmp_path, **kw):
    base = dict(adapter="qpeft", qpeft={"method": "qa_lora", "bits": 4, "group_size": 64},
                lora_r=8, lora_alpha=16, lora_dropout=0.0, lora_target_linear=True,
                output_dir=str(tmp_path), weight_decay=0.0)
    base.update(kw)
    return DictDefault(base)


def _quantized_leaf_names(model):
    return {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, QuantLinear)}


# --- load_adapter -----------------------------------------------------------------

def test_load_adapter_quantizes_the_block_linears_in_place(tmp_path):
    model = _llama()
    returned, peft_config = QpeftPlugin().load_adapter(model, _cfg(tmp_path))
    assert returned is model and peft_config is None       # axolotl keeps an ordinary HF model
    assert _quantized_leaf_names(model) == BLOCK_LINEARS     # lm_head stays full precision
    trainable = {n.split(".")[-1] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"A", "B"}                           # only the QA-LoRA adapter


def test_peqa_trains_only_the_scales(tmp_path):
    model = _llama()
    QpeftPlugin().load_adapter(model, _cfg(tmp_path, qpeft={"method": "peqa", "bits": 4, "group_size": 64}))
    trainable = {n.split(".")[-1] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"scale"}


def test_listed_targets_match_like_peft(tmp_path):
    model = _llama()
    QpeftPlugin().load_adapter(model, _cfg(tmp_path, lora_target_linear=None, lora_target_modules=["q_proj", "v_proj"]))
    assert _quantized_leaf_names(model) == {"q_proj", "v_proj"}


def test_other_adapters_are_left_to_axolotl(tmp_path):
    model = _llama()
    assert QpeftPlugin().load_adapter(model, _cfg(tmp_path, adapter="lora")) is None
    assert not _quantized_leaf_names(model)


def test_config_only_touches_nothing(tmp_path):
    model = _llama()
    assert QpeftPlugin().load_adapter(model, _cfg(tmp_path), config_only=True) == (None, None)
    assert not _quantized_leaf_names(model)


@pytest.mark.parametrize("kw, match", [
    (dict(load_in_4bit=True), "full-precision"),
    (dict(gptq=True), "full-precision"),
    (dict(deepspeed="zero2.json"), "FSDP or DeepSpeed"),
    (dict(fsdp_config={"fsdp_version": 2}), "FSDP or DeepSpeed"),
    (dict(relora=True), "ReLoRA"),
    (dict(lora_modules_to_save=["embed_tokens"]), "lora_modules_to_save"),
    (dict(qpeft={"method": "peqa"}, weight_decay=0.01), "weight_decay"),
    (dict(lora_target_linear=None, lora_target_modules=None), "lora_target_modules"),
], ids=["4bit", "gptq", "deepspeed", "fsdp", "relora", "modules_to_save", "peqa_weight_decay", "no_targets"])
def test_unsupported_settings_are_refused(tmp_path, kw, match):
    model = _llama()
    with pytest.raises(ValueError, match=match):
        QpeftPlugin().load_adapter(model, _cfg(tmp_path, **kw))
    assert not _quantized_leaf_names(model), "a refused config must not change the model"


def test_a_scheme_refusal_reaches_the_user(tmp_path):
    with pytest.raises(UnsupportedSchemeError):
        QpeftPlugin().load_adapter(_llama(), _cfg(tmp_path, qpeft={"method": "qa_lora", "backend": "does_not_exist"}))


# --- post_train -------------------------------------------------------------------

def test_post_train_merges_once_and_saves_a_loadable_int_artifact(tmp_path):
    plugin, model, cfg = QpeftPlugin(), _llama(), _cfg(tmp_path)
    plugin.load_adapter(model, cfg)
    with torch.no_grad():
        for layer in (m for m in model.modules() if isinstance(m, QuantLinear)):
            layer.adapter.B.normal_(0, 0.1)                  # a trained-looking adapter
    x = torch.randint(0, 320, (2, 16))
    with torch.no_grad():
        trained = model(x).logits

    plugin.post_train(cfg, model)
    plugin.post_train(cfg, model)                            # axolotl calls it twice
    assert all(m.merged for m in model.modules() if isinstance(m, QuantLinear))

    loaded = QuantModel.from_pretrained(_llama(seed=1), tmp_path / "qpeft")
    with torch.no_grad():
        assert torch.allclose(loaded(x).logits, trained, atol=1e-4)


def test_post_train_does_nothing_without_qpeft(tmp_path):
    QpeftPlugin().post_train(_cfg(tmp_path, adapter="lora"), _llama())
    assert not (tmp_path / "qpeft").exists()


# --- a real `axolotl train` run -----------------------------------------------------

@pytest.mark.skipif(os.environ.get("QPEFT_RUN_SLOW") != "1", reason="downloads; set QPEFT_RUN_SLOW=1")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="axolotl's bf16 run needs a GPU")
@pytest.mark.parametrize("method", ["qa_lora", "peqa"])
def test_axolotl_train_writes_the_int_artifact(tmp_path, method):
    import json
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"text": f"qpeft keeps the integer model, sample {i}. " * 8})
                              for i in range(64)))
    lr = "0.0002" if method == "qa_lora" else "0.00002"
    config = tmp_path / "config.yml"
    config.write_text(f"""
base_model: HuggingFaceTB/SmolLM2-135M
plugins:
  - qpeft.integrations.axolotl.QpeftPlugin
adapter: qpeft
qpeft:
  method: {method}
  bits: 4
  group_size: 64
lora_r: 8
lora_alpha: 16
lora_target_linear: true
datasets:
  - path: {data}
    ds_type: json
    type: completion
val_set_size: 0
sequence_len: 128
micro_batch_size: 2
max_steps: 4
learning_rate: {lr}
weight_decay: 0.0
optimizer: adamw_torch
bf16: true
save_strategy: "no"
output_dir: {tmp_path / "out"}
""")
    run = subprocess.run([sys.executable, "-m", "axolotl.cli.train", str(config)],
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "qpeft: merge equivalence OK" in run.stdout + run.stderr
    assert (tmp_path / "out" / "qpeft" / "qpeft_config.json").exists()
    assert (tmp_path / "out" / "qpeft" / "qpeft_model.pt").exists()
