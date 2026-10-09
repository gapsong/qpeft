"""The `qpeft:` block of an axolotl YAML (qpeft/integrations/axolotl/args.py):
valid blocks become the same qpeft configs, and every setting qpeft cannot honor is refused
at config time, before any model or data is loaded.
"""
import dataclasses
import typing

import pytest

pydantic = pytest.importorskip("pydantic")

from qpeft import EfficientQATConfig, PEQAConfig, QALoraConfig, QuantTuningType  # noqa: E402
from qpeft.integrations.axolotl.args import QPeftArgs, QPeftBlock  # noqa: E402

QA_LORA = {"method": "qa_lora", "bits": 4, "group_size": 64, "r": 16, "lora_alpha": 32,
           "target_modules": ["q_proj", "v_proj"]}


def validate(**config):
    return QPeftArgs.model_validate(config)


def quant_configs(**config):
    return validate(**config).qpeft.quant_configs()


# --- valid blocks ---------------------------------------------------------------

def test_qa_lora_block_becomes_its_qpeft_config():
    assert quant_configs(adapter="qpeft", qpeft=QA_LORA) == [
        QALoraConfig(bits=4, group_size=64, r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])]


def test_unset_fields_keep_the_qpeft_defaults_of_the_method():
    assert quant_configs(adapter="qpeft", qpeft={"method": "peqa"}) == [PEQAConfig()]
    assert quant_configs(adapter="qpeft", qpeft={"method": "qa_lora"}) == [QALoraConfig()]


def test_efficient_qat_block_runs_block_ap_then_e2e_qp():
    block = {"method": "efficient_qat", "bits": 2, "group_size": 64, "export": "gptq",
             "block_ap": {"epochs": 2, "train_size": 4096, "seqlen": 2048, "batch_size": 2,
                          "weight_lr": 2e-5, "quant_lr": 1e-4}}
    args = validate(adapter="qpeft", qpeft=block, optimizer="adamw_torch_fused")
    assert args.qpeft.quant_configs() == [EfficientQATConfig(bits=2, group_size=64, phase="block_ap"),
                                          EfficientQATConfig(bits=2, group_size=64, phase="e2e_qp")]
    assert args.qpeft.block_ap.train_size == 4096


def test_efficient_qat_runs_block_ap_without_a_block_ap_block():
    phases = [c.phase for c in quant_configs(adapter="qpeft", qpeft={"method": "efficient_qat"})]
    assert phases == ["block_ap", "e2e_qp"]


def test_the_block_survives_axolotls_dump_to_a_dict():
    """axolotl gives plugins model_dump(exclude_none=True), not the pydantic object."""
    args = validate(adapter="qpeft", qpeft={"method": "efficient_qat", "bits": 3})
    dumped = args.model_dump(exclude_none=True)["qpeft"]
    assert QPeftBlock.model_validate(dumped).quant_configs() == args.qpeft.quant_configs()


def test_a_run_without_qpeft_is_not_touched():
    """The plugin loads on every axolotl run, so other adapters must pass through as they are."""
    args = validate(adapter="lora", lora_r=8, fsdp_config={"fsdp_version": 2}, optimizer="sgd")
    assert args.qpeft is None


@pytest.mark.parametrize("setting", [
    {"lora_dropout": 0.0},                  # axolotl sets this itself for every adapter
    {"load_in_4bit": False},
    {"tensor_parallel_size": 1},
    {"optimizer": "adamw_torch"},
    {"capabilities": {"n_gpu": 1}},
])
def test_settings_that_change_nothing_pass(setting):
    validate(adapter="qpeft", qpeft={"method": "efficient_qat"}, **setting)


def test_qa_lora_and_peqa_may_run_on_several_processes():
    for method in ("qa_lora", "peqa"):
        validate(adapter="qpeft", qpeft={"method": method}, capabilities={"n_gpu": 2})


# --- refusals ---------------------------------------------------------------------

def refused(match, **config):
    with pytest.raises(pydantic.ValidationError, match=match):
        validate(**config)


@pytest.mark.parametrize("block, match", [
    ({"method": "foo"}, "'qa_lora', 'peqa' or 'efficient_qat'"),
    ({**QA_LORA, "rank": 8}, "rank\n.*Extra inputs are not permitted"),
    ({"method": "efficient_qat", "phase": "e2e_qp"}, "phase\n.*Extra inputs are not permitted"),
    ({"method": "peqa", "bits": 1}, "Input should be 2, 3, 4 or 8"),
    ({"method": "peqa", "bits": 9}, "Input should be 2, 3, 4 or 8"),
    ({"method": "peqa", "group_size": 0}, "greater than 0"),
    ({"method": "peqa", "r": 16}, r"PEQAConfig does not know: \['r'\]"),
    ({"method": "qa_lora", "block_ap": {"epochs": 1}}, "method 'qa_lora' has no Block-AP"),
    ({"method": "efficient_qat", "block_ap": {"epoch": 1}}, "epoch\n.*Extra inputs are not permitted"),
    ({"method": "qa_lora", "export": "gptq"}, "export: gptq is refused for method qa_lora"),
    ({"method": "peqa", "export": "gguf"}, "'qpeft' or 'gptq'"),
    ({"method": "peqa", "backend": "nope"}, "unknown backend 'nope'"),
    ({"method": "peqa", "backend": "mlx"}, "'mlx' backend is not built yet"),
])
def test_block_refusal(block, match):
    refused(match, adapter="qpeft", qpeft=block)


def test_torchao_refuses_block_ap_at_config_time():
    pytest.importorskip("torchao")
    refused("trainable=.*zero_point.*Refusing", adapter="qpeft",
            qpeft={"method": "efficient_qat", "backend": "torchao"})


@pytest.mark.parametrize("setting, match", [
    ({"fsdp": ["full_shard"]}, "`fsdp`: sharded training"),
    ({"fsdp_config": {"fsdp_version": 2}}, "`fsdp_config`: sharded training"),
    ({"deepspeed": "zero3.json"}, "`deepspeed`: sharded training"),
    ({"tensor_parallel_size": 2}, "`tensor_parallel_size: 2`"),
    ({"context_parallel_size": 2}, "`context_parallel_size: 2`"),
    ({"sequence_parallel_degree": 2}, "`sequence_parallel_degree: 2`"),
    ({"expert_parallel_size": 2}, "`expert_parallel_size: 2`"),
    ({"dp_shard_size": 2}, "`dp_shard_size: 2`"),
    ({"rl": "dpo"}, "`rl`: RL trainers"),
    ({"load_in_4bit": True}, "`load_in_4bit`: qpeft quantizes"),
    ({"load_in_8bit": True}, "`load_in_8bit`: qpeft quantizes"),
    ({"gptq": True}, "`gptq`: qpeft quantizes"),
    ({"relora": True}, "`relora`: ReLoRA"),
    ({"merge_lora": True}, "`merge_lora`: qpeft merges"),
    ({"qat": {"weight_dtype": "int4"}}, "`qat`: axolotl's QAT"),
    ({"optimizer": "sgd"}, "`optimizer: sgd`: qpeft trains with AdamW"),
    ({"optimizer": "adamw_bnb_8bit"}, "`optimizer: adamw_bnb_8bit`"),
    ({"lora_r": 8, "lora_target_linear": True}, "`lora_r`, `lora_target_linear`: peft LoRA"),
    ({"peft_use_dora": True}, "`peft_use_dora`: peft LoRA"),
    ({"capabilities": {"n_gpu": 2}}, "efficient_qat with world size 2"),
    ({"world_size": 4}, "efficient_qat with world size 4"),
])
def test_axolotl_setting_refusal(setting, match):
    refused(match, adapter="qpeft", qpeft={"method": "efficient_qat"}, **setting)


def test_adapter_qpeft_without_a_block_is_refused():
    refused("needs a `qpeft:` block", adapter="qpeft")


def test_a_block_without_adapter_qpeft_is_refused():
    refused("needs `adapter: qpeft`", adapter="lora", qpeft={"method": "peqa"})
    refused("needs `adapter: qpeft`", qpeft={"method": "peqa"})


def test_every_refusal_is_reported_at_once():
    refused("(?s)`deepspeed`.*`load_in_4bit`.*`optimizer: sgd`", adapter="qpeft",
            qpeft={"method": "peqa"}, deepspeed="zero3.json", load_in_4bit=True, optimizer="sgd")


# --- the block stays in step with qpeft -------------------------------------------

def test_every_method_of_qpeft_is_a_method_of_the_block():
    methods = typing.get_args(QPeftBlock.model_fields["method"].annotation)
    assert sorted(methods) == sorted(t.value.lower() for t in QuantTuningType)


def test_every_qpeft_field_of_the_block_is_a_field_of_a_qpeft_config():
    qpeft_fields = {f.name for c in (QALoraConfig, PEQAConfig, EfficientQATConfig)
                    for f in dataclasses.fields(c)}
    assert set(QPeftBlock.model_fields) - {"method", "export", "block_ap"} <= qpeft_fields


# --- inside axolotl's own config model --------------------------------------------

def test_inside_axolotls_config_model(monkeypatch):
    """The real merge: axolotl's validators run first and add values of their own."""
    pytest.importorskip("axolotl")
    from axolotl.integrations.base import PluginManager
    from axolotl.utils.schemas.config import AxolotlInputConfig

    # Registering the adapter name is the plugin's job, which is not under test here.
    monkeypatch.setattr(PluginManager.get_instance(), "supports_adapter", lambda name: name == "qpeft")
    # As axolotl.integrations.config.merge_input_args builds it:
    config_class = type("AxolotlInputConfig", (AxolotlInputConfig, QPeftArgs), {})
    run = {"base_model": "axolotl-ai-co/tiny-qwen3-129m", "datasets": [{"path": "d", "type": "alpaca"}],
           "micro_batch_size": 1, "num_epochs": 1, "learning_rate": 1e-4, "sequence_len": 128,
           "output_dir": "out", "adapter": "qpeft"}

    cfg = config_class(**run, qpeft=QA_LORA)
    dumped = cfg.model_dump(exclude_none=True)["qpeft"]
    assert QPeftBlock.model_validate(dumped).quant_configs() == validate(**run, qpeft=QA_LORA).qpeft.quant_configs()

    with pytest.raises(pydantic.ValidationError, match="lora_r"):
        config_class(**run, qpeft=QA_LORA, lora_r=8)
    with pytest.raises(pydantic.ValidationError, match="Extra inputs are not permitted"):
        config_class(**run, qpeft={**QA_LORA, "rank": 8})
