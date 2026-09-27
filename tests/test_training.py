"""param_groups (qpeft/training.py): one optimizer group per parameter KIND,
each with its own learning rate, as in the official EfficientQAT
(weight_lr for the weights, quant_lr for scale / zero_point)."""
import pytest
import torch
import torch.nn as nn

from qpeft import EfficientQATConfig, QALoraConfig, get_quant_model, param_groups
from qpeft.tuners.tuners_utils import QuantLinear

LRS = dict(weight_lr=1e-5, quant_lr=1e-4, adapter_lr=2e-4)


def _base():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64), nn.Linear(64, 64))


def _layers(model):
    return [m for m in model.modules() if isinstance(m, QuantLinear)]


def _ids(params):
    return {id(p) for p in params}


def _by_name(groups):
    return {g["name"]: g for g in groups}


def test_block_ap_groups_weights_and_qparams_with_their_own_lr():
    model = get_quant_model(_base(), EfficientQATConfig(bits=4, group_size=64, phase="block_ap"))
    groups = _by_name(param_groups(model, **LRS))

    assert set(groups) == {"weight", "quant"}
    assert groups["weight"]["lr"] == LRS["weight_lr"]
    assert groups["quant"]["lr"] == LRS["quant_lr"]
    layers = _layers(model)
    assert _ids(groups["weight"]["params"]) == _ids(m.weight for m in layers)
    assert _ids(groups["quant"]["params"]) == _ids(
        p for m in layers for p in (m.scale, m.zero_point))


def test_e2e_qp_groups_only_the_scale():
    model = get_quant_model(_base(), EfficientQATConfig(bits=4, group_size=64, phase="e2e_qp"))
    groups = _by_name(param_groups(model, **LRS))

    assert set(groups) == {"quant"}
    assert _ids(groups["quant"]["params"]) == _ids(m.scale for m in _layers(model))


def test_qa_lora_groups_only_the_adapter():
    model = get_quant_model(_base(), QALoraConfig(bits=4, group_size=64, r=8))
    groups = _by_name(param_groups(model, **LRS))

    assert set(groups) == {"adapter"}
    assert groups["adapter"]["lr"] == LRS["adapter_lr"]
    assert _ids(groups["adapter"]["params"]) == _ids(
        p for m in _layers(model) for p in m.adapter.parameters())


def test_every_trainable_parameter_is_in_exactly_one_group():
    """AdamW refuses a parameter in two groups; a missing one would silently not train."""
    model = get_quant_model(_base(), EfficientQATConfig(bits=4, group_size=64, phase="block_ap"))
    norm = model.base[1]
    for p in norm.parameters():
        p.requires_grad_(True)
    groups = param_groups(model, **LRS, extra_weights=list(norm.parameters()))

    grouped = [id(p) for g in groups for p in g["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == _ids(p for p in model.parameters() if p.requires_grad)
    torch.optim.AdamW(groups)


def test_extra_weights_join_the_weight_group():
    """Block-AP's norm weights train with weight_lr, as in the official code."""
    model = get_quant_model(_base(), EfficientQATConfig(bits=4, group_size=64, phase="block_ap"))
    norm_w = model.base[1].weight
    norm_w.requires_grad_(True)
    groups = _by_name(param_groups(model, **LRS, extra_weights=[norm_w]))

    assert id(norm_w) in _ids(groups["weight"]["params"])


def test_trainable_parameter_outside_quant_linear_refuses():
    model = get_quant_model(_base(), EfficientQATConfig(bits=4, group_size=64, phase="e2e_qp"))
    model.base[1].weight.requires_grad_(True)          # a norm, not passed as extra_weights
    with pytest.raises(ValueError, match="outside QuantLinear"):
        param_groups(model, **LRS)


def test_nothing_trainable_refuses():
    model = get_quant_model(_base(), EfficientQATConfig(bits=4, group_size=64, phase="e2e_qp"))
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="nothing is trainable"):
        param_groups(model, **LRS)


def test_weight_decay_reaches_every_group():
    model = get_quant_model(_base(), EfficientQATConfig(bits=4, group_size=64, phase="block_ap"))
    groups = param_groups(model, **LRS, weight_decay=0.1)
    assert all(g["weight_decay"] == 0.1 for g in groups)
