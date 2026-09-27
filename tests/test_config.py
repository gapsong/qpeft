"""Config validation. ~ peft/tests/test_config.py: a method config must set the
right trainable axis and refuse nonsense up front.
"""
import pytest

from qpeft import (
    EfficientQATConfig, QALoraConfig, QuantTuningConfig, TrainableParams, efficient_qat_schedule,
)
from qpeft.quant_schemes import UnsupportedSchemeError, build_scheme


def test_efficient_qat_phases_set_trainable_params():
    assert set(EfficientQATConfig(phase="block_ap").trainable_params) == {
        TrainableParams.WEIGHT, TrainableParams.SCALE, TrainableParams.ZERO_POINT}
    assert set(EfficientQATConfig(phase="e2e_qp").trainable_params) == {TrainableParams.SCALE}


def test_efficient_qat_unknown_phase_raises():
    with pytest.raises(ValueError):
        EfficientQATConfig(phase="does_not_exist")


def test_qa_lora_trains_only_adapter():
    assert set(QALoraConfig().trainable_params) == {TrainableParams.ADAPTER}


def test_unbuilt_backend_refuses_rather_than_approximates():
    with pytest.raises((NotImplementedError, UnsupportedSchemeError)):
        build_scheme(EfficientQATConfig(bits=4, group_size=64, backend="mlx"))


def test_unknown_qat_scheme_refuses_with_scheme_error():
    """An unknown qat_scheme is a loud refusal, not a bare KeyError."""
    with pytest.raises(UnsupportedSchemeError):
        build_scheme(EfficientQATConfig(bits=4, group_size=64, qat_scheme="does_not_exist"))


def test_backend_names_are_providers():
    """'auto' and 'torch' both select the pure-torch reference; unknown provider refuses."""
    for backend in ("auto", "torch"):
        s = build_scheme(EfficientQATConfig(bits=4, group_size=64, phase="e2e_qp", backend=backend))
        assert type(s).__name__ == "ReferenceIntUniformScheme"
    with pytest.raises(UnsupportedSchemeError):        # a device suffix is no longer a backend
        build_scheme(EfficientQATConfig(bits=4, group_size=64, backend="torchao_cuda"))


def test_efficient_qat_defaults_match_official_code():
    """OpenGVLab/EfficientQAT main_block_ap.py: --wbits 4, --group_size 128."""
    for phase in ("block_ap", "e2e_qp"):
        cfg = EfficientQATConfig(phase=phase)
        assert (cfg.bits, cfg.group_size) == (4, 128)
    assert all((c.bits, c.group_size) == (4, 128) for c in efficient_qat_schedule())


# --- serialization (QuantTuningConfig.to_dict / from_dict, ~ peft PeftConfigMixin) --

@pytest.mark.parametrize("cfg", [
    EfficientQATConfig(bits=2, group_size=64, phase="e2e_qp", target_modules=["q_proj"]),
    QALoraConfig(bits=4, group_size=32, r=8, backend="torch"),
])
def test_config_roundtrips_through_dict_and_disk(cfg, tmp_path):
    assert QuantTuningConfig.from_dict(cfg.to_dict()) == cfg
    cfg.save_pretrained(tmp_path)
    assert QuantTuningConfig.from_pretrained(tmp_path) == cfg
    assert type(cfg).from_pretrained(tmp_path) == cfg


def test_config_from_dict_refuses_an_unknown_field():
    d = EfficientQATConfig().to_dict()
    d["field_from_a_newer_qpeft"] = 1
    with pytest.raises(UnsupportedSchemeError, match="field_from_a_newer_qpeft"):
        QuantTuningConfig.from_dict(d)


def test_config_from_dict_refuses_a_different_method():
    with pytest.raises(UnsupportedSchemeError, match="QALoraConfig"):
        EfficientQATConfig.from_dict(QALoraConfig().to_dict())
