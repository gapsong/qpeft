"""The axolotl plugin's hooks (qpeft/integrations/axolotl/plugin.py), called as axolotl calls
them, on a tiny random Llama and a plain Hugging Face Trainer. No download, no axolotl CLI;
tests/test_axolotl_e2e.py runs the real `axolotl train`."""
import subprocess
import sys
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

pytest.importorskip("axolotl")

import torch                                                     # noqa: E402
from axolotl.utils.dict import DictDefault                       # noqa: E402
from torch.utils.data import Dataset                             # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments   # noqa: E402

from qpeft import QuantModel, UnsupportedSchemeError             # noqa: E402
from qpeft.hf_trainer import FirstStepMustMoveParams             # noqa: E402
from qpeft.integrations.axolotl.args import QPeftArgs            # noqa: E402
from qpeft.integrations.axolotl.plugin import QPeftPlugin        # noqa: E402
from qpeft.tuners.tuners_utils import QuantLinear                # noqa: E402

ROOT = Path(__file__).parent.parent
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BLOCKS = {
    "qa_lora": {"method": "qa_lora", "bits": 4, "group_size": 32, "r": 4, "lora_alpha": 8, "target_modules": TARGETS},
    "peqa": {"method": "peqa", "bits": 4, "group_size": 32, "target_modules": TARGETS},
}


class _Data(Dataset):
    def __init__(self, n=8, seq=16, vocab=320):
        self.ids = torch.randint(0, vocab, (n, seq), generator=torch.Generator().manual_seed(0))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.ids[i]}


def _collate(rows):
    return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}


def _llama():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=4, vocab_size=320, max_position_embeddings=64))


def _cfg(tmp_path, block="qa_lora", **settings):
    block = BLOCKS[block] if isinstance(block, str) else block
    return DictDefault({"adapter": "qpeft", "qpeft": block, "output_dir": str(tmp_path), **settings})


def _trainer(tmp_path, model, **args):
    args = {"output_dir": str(tmp_path), "per_device_train_batch_size": 4, "max_steps": 2, "learning_rate": 1e-2,
            "report_to": [], "dataloader_pin_memory": False, "save_strategy": "no", "use_cpu": True, **args}
    return Trainer(model=model, args=TrainingArguments(**args), train_dataset=_Data(), data_collator=_collate)


def _loaded(tmp_path, block="qa_lora"):
    plugin, model, cfg = QPeftPlugin(), _llama(), _cfg(tmp_path, block)
    assert plugin.load_adapter(model, cfg) == (model, None)
    return plugin, model, cfg


def _logits(model):
    with torch.no_grad():
        return model.eval()(input_ids=torch.arange(10, 26)[None]).logits


# --- a run without adapter: qpeft -------------------------------------------------

def test_every_hook_does_nothing_without_adapter_qpeft(tmp_path):
    """axolotl loads the plugin on every run through its entry point."""
    plugin, model, cfg = QPeftPlugin(), _llama(), DictDefault({"adapter": "lora", "output_dir": str(tmp_path)})
    trainer = _trainer(tmp_path, model)
    plugin.register(cfg)
    assert plugin.load_adapter(model, cfg) is None
    plugin.post_trainer_create(cfg, trainer)
    plugin.post_train(cfg, model)
    assert trainer.optimizer is None
    assert not any(isinstance(c, FirstStepMustMoveParams) for c in trainer.callback_handler.callbacks)
    assert not any(isinstance(m, QuantLinear) for m in model.modules())
    assert list(tmp_path.iterdir()) == []


def test_the_entry_point_and_the_input_args_resolve_to_the_plugin():
    tomllib = pytest.importorskip("tomllib")
    points = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["entry-points"]["axolotl.plugins"]
    plugin = EntryPoint("qpeft", points["qpeft"], "axolotl.plugins").load()
    assert plugin is QPeftPlugin
    module, _, name = QPeftPlugin().get_input_args().rpartition(".")
    assert getattr(__import__(module, fromlist=[name]), name) is QPeftArgs
    assert [c.name for c in QPeftPlugin().get_adapter_capabilities()] == ["qpeft"]


def test_importing_the_plugin_adds_only_qpeft_modules():
    """The entry point imports the plugin on every axolotl run: it must not pull in torchao,
    datasets or anything else that axolotl did not load already (its plugin base and the
    pydantic config model it validates every run with)."""
    script = ("import sys, axolotl.integrations.base, axolotl.utils.schemas.config\n"
              "before = set(sys.modules)\n"
              "import qpeft.integrations.axolotl.plugin\n"
              "print(sorted(m for m in set(sys.modules) - before if m.split('.')[0] != 'qpeft'))")
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-3000:]
    assert run.stdout.splitlines()[-1] == "[]"


# --- refusals ---------------------------------------------------------------------

def test_efficient_qat_is_refused_when_axolotl_registers_the_plugin(tmp_path):
    plugin = QPeftPlugin()
    with pytest.raises(UnsupportedSchemeError, match="does not run method efficient_qat in axolotl yet"):
        plugin.register({"adapter": "qpeft", "qpeft": {"method": "efficient_qat"}})
    for block in BLOCKS.values():
        plugin.register({"adapter": "qpeft", "qpeft": block})


@pytest.mark.parametrize("method", ["qa_lora", "peqa"])
def test_an_export_other_than_the_qpeft_artifact_is_refused_when_axolotl_registers_the_plugin(method):
    with pytest.raises(UnsupportedSchemeError, match="does not run qpeft.export: gptq in axolotl yet"):
        QPeftPlugin().register({"adapter": "qpeft", "qpeft": {**BLOCKS[method], "export": "gptq"}})
    QPeftPlugin().register({"adapter": "qpeft", "qpeft": {**BLOCKS[method], "export": "qpeft"}})


@pytest.mark.parametrize("mode", [{"inference": True}, {"config_only": True}])
def test_load_adapter_refuses_inference_and_a_bare_config(tmp_path, mode):
    with pytest.raises(UnsupportedSchemeError, match="only trains"):
        QPeftPlugin().load_adapter(_llama(), _cfg(tmp_path), **mode)


def test_a_group_size_that_does_not_divide_in_features_is_refused_for_every_layer(tmp_path):
    model = _llama()
    with pytest.raises(ValueError, match=r"(?s)7 of 7 target layers.*mlp.down_proj: in_features 128 not divisible"):
        QPeftPlugin().load_adapter(model, _cfg(tmp_path, {**BLOCKS["peqa"], "group_size": 96}))
    assert not any(isinstance(m, QuantLinear) for m in model.modules())


def test_targets_that_match_no_layer_are_refused(tmp_path):
    with pytest.raises(ValueError, match=r"target_modules \['nope_proj'\] matched no nn.Linear"):
        QPeftPlugin().load_adapter(_llama(), _cfg(tmp_path, {**BLOCKS["peqa"], "target_modules": ["nope_proj"]}))


# --- a qpeft run ------------------------------------------------------------------

def test_load_adapter_quantizes_the_model_in_place(tmp_path):
    plugin, model, _ = _loaded(tmp_path)
    assert plugin.qmodel.base is model
    assert sum(isinstance(m, QuantLinear) for m in model.modules()) == len(TARGETS)


@pytest.mark.parametrize("block, kind", [("qa_lora", "adapter"), ("peqa", "quant")])
@pytest.mark.parametrize("optim, fused", [("adamw_torch", None), ("adamw_torch_fused", True)])
def test_the_trainer_trains_with_qpeft_param_groups(tmp_path, block, kind, optim, fused):
    plugin, model, cfg = _loaded(tmp_path, block)
    trainer = _trainer(tmp_path, model, optim=optim, weight_decay=0.01, adam_beta2=0.95)
    plugin.post_trainer_create(cfg, trainer)

    optimizer = trainer.optimizer
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["fused"] is fused
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert [(g["name"], g["lr"], g["weight_decay"]) for g in optimizer.param_groups] == [(kind, 1e-2, 0.01)]
    assert any(isinstance(c, FirstStepMustMoveParams) for c in trainer.callback_handler.callbacks)
    trainer.train()
    assert getattr(trainer.optimizer, "optimizer", trainer.optimizer) is optimizer    # accelerate wraps it


def test_a_learning_rate_of_zero_stops_the_run(tmp_path):
    plugin, model, cfg = _loaded(tmp_path)
    trainer = _trainer(tmp_path, model, learning_rate=0.0)
    plugin.post_trainer_create(cfg, trainer)
    with pytest.raises(RuntimeError, match="without an optimizer step at lr > 0"):
        trainer.train()


@pytest.mark.parametrize("block", BLOCKS)
def test_post_train_merges_and_saves_once_and_the_artifact_computes_what_was_trained(tmp_path, block):
    plugin, model, cfg = _loaded(tmp_path, block)
    trainer = _trainer(tmp_path, model)
    plugin.post_trainer_create(cfg, trainer)
    trainer.train()
    trained = _logits(model)

    plugin.post_train(cfg, model)
    saved = (tmp_path / "qpeft_model.pt").stat().st_mtime_ns
    plugin.post_train(cfg, model)                    # axolotl calls it a second time

    assert plugin.qmodel.is_merged
    assert (tmp_path / "qpeft_model.pt").stat().st_mtime_ns == saved
    reloaded = _logits(QuantModel.from_pretrained(_llama(), tmp_path))
    assert (reloaded - trained).abs().max().item() < 1e-4


# --- the guard against axolotl's own model saves ----------------------------------

def test_a_save_into_output_dir_before_the_merge_writes_nothing_and_says_so(tmp_path, capfd):
    """axolotl's Ctrl-C handler: model.save_pretrained(output_dir) while training."""
    _, model, _ = _loaded(tmp_path)
    model.save_pretrained(str(tmp_path))
    assert list(tmp_path.iterdir()) == []
    assert "training stopped before the merge, so no model was written" in capfd.readouterr().err


def test_trainer_checkpoints_are_saved_as_usual_for_resuming(tmp_path):
    plugin, model, _ = _loaded(tmp_path)
    checkpoint = tmp_path / "checkpoint-2"
    model.save_pretrained(checkpoint, state_dict=model.state_dict())
    assert (checkpoint / "model.safetensors").exists()


def test_a_merged_model_is_never_saved_as_transformers_weights(tmp_path):
    """axolotl's final save runs after post_train merged; any other save of the merged model
    would write the same file that loads with random weights."""
    plugin, model, cfg = _loaded(tmp_path)
    plugin.post_trainer_create(cfg, _trainer(tmp_path, model))
    plugin.post_train(cfg, model)
    model.save_pretrained(str(tmp_path))
    model.save_pretrained(tmp_path / "elsewhere")
    assert not (tmp_path / "model.safetensors").exists()
    assert not (tmp_path / "elsewhere").exists()
    assert (tmp_path / "qpeft_model.pt").exists()
