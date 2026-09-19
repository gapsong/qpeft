"""Config validation. ~ peft/tests/test_config.py: a method config must set the
right trainable axis and refuse nonsense up front.
"""
import pytest

from qpeft import EfficientQATConfig, QALoraConfig, TrainableParams
from qpeft.schemes import UnsupportedSchemeError, build_scheme


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
