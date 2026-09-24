"""EfficientQAT phases: during training, ONLY what the phase allows may change.

Reference: the official pipeline (OpenGVLab/EfficientQAT).
  Block-AP  (main_block_ap.py, quantize/block_ap.py)
            trainable: weight (weight_lr) + scale, zero_point (quant_lr)
            after each block: quant_inplace -> weight := fake_quant(weight), then
            packed to real int (int_linear_real.QuantLinear)
  E2E-QP    (main_e2e_qp.py)
            loads the PACKED int model; only `scales` get requires_grad.
            => weight codes AND zero_point are frozen, codes do not move.

These tests encode the paper's contract; qpeft meets it (codes are an int32
buffer from the phase switch on, see QuantLinear.freeze_codes).
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qpeft import EfficientQATConfig, efficient_qat_schedule, get_quant_model
from qpeft.tuners.tuners_utils import QuantLinear

TRAINABLE = {
    "block_ap": {"weight", "scale", "zero_point"},
    "e2e_qp": {"scale"},
}


def _task(seed=0, in_f=128, out_f=64, n=256):
    g = torch.Generator().manual_seed(seed)
    teacher = nn.Linear(in_f, out_f)
    x = torch.randn(n, in_f, generator=g)
    with torch.no_grad():
        y = teacher(x) + 0.1 * torch.randn(n, out_f, generator=g)
    return x, y


def _base(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 64))


def _layers(model):
    layers = [m for m in model.modules() if isinstance(m, QuantLinear)]
    assert layers, "no QuantLinear -- nothing tested"
    return layers


def _snapshot(model):
    def grab(t):
        return None if t is None else t.detach().clone()
    return [{k: grab(getattr(layer, k)) for k in ("weight", "scale", "zero_point", "bias")}
            for layer in _layers(model)]


def _train(model, steps=20, lr=1e-2):
    x, y = _task()
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "nothing is trainable"
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(model(x), y).backward()
        opt.step()


def _assert_only_allowed_changed(before, after, allowed):
    for i, (b, a) in enumerate(zip(before, after)):
        for name in ("weight", "scale", "zero_point", "bias"):
            if b[name] is None:
                continue
            changed = not torch.equal(b[name], a[name])
            if name in allowed:
                assert changed, f"layer {i}: {name} should train but did not change"
            else:
                assert not changed, f"layer {i}: {name} is frozen in this phase but changed"


# --- 1. requires_grad matches the phase ------------------------------------------

@pytest.mark.parametrize("phase", TRAINABLE)
def test_requires_grad_matches_phase(phase):
    model = get_quant_model(_base(), EfficientQATConfig(bits=4, group_size=64, phase=phase))
    for layer in _layers(model):
        trainable = {n for n in ("weight", "scale", "zero_point")
                     if getattr(layer, n).requires_grad}
        assert trainable == TRAINABLE[phase]
        assert layer.bias is None or not layer.bias.requires_grad


# --- 2. training changes exactly the allowed set ---------------------------------

@pytest.mark.parametrize("phase", TRAINABLE)
def test_training_changes_only_allowed_params(phase):
    model = get_quant_model(_base(), EfficientQATConfig(bits=4, group_size=64, phase=phase))
    before = _snapshot(model)
    _train(model)
    _assert_only_allowed_changed(before, _snapshot(model), TRAINABLE[phase])


# --- 3. E2E-QP: the integer codes are frozen (paper contract) ----------------------

def test_e2e_qp_codes_are_frozen_when_scale_moves():
    """Official E2E-QP trains `scales` on an already PACKED int model, so the codes
    cannot change. Here we move the scale deterministically (x1.5, as training
    could) and require the layer to still compute with the ORIGINAL codes:
        y == x @ ((codes0 - z) * s_new)^T + b
    Goes red if the forward re-rounds the weight (codes = round(w / s + z)) instead
    of using the frozen codes."""
    model = get_quant_model(_base(), EfficientQATConfig(bits=4, group_size=64, phase="e2e_qp"))
    layer = _layers(model)[0]
    codes0 = layer.scheme.quantize(layer.weight, layer.scale, layer.zero_point)

    with torch.no_grad():
        layer.scale.mul_(1.5)
        x = torch.randn(8, layer.weight.shape[1])
        expected = F.linear(x, layer.scheme.dequant(codes0, layer.scale, layer.zero_point), layer.bias)
        got = layer(x)
    assert torch.allclose(got, expected, atol=1e-6), \
        "E2E-QP re-rounded the weights: codes are not frozen"


# --- 4. the schedule hands Block-AP results over to E2E-QP -------------------------

def test_schedule_switches_trainable_set_and_keeps_phase1_results():
    """The README flow:  for cfg in efficient_qat_schedule(...): model = get_quant_model(base, cfg)
    After the switch, (a) ONLY scale may be trainable and (b) weight/scale/zero_point
    must be the ones Block-AP produced, not re-initialized.
    Goes red if phase 2 skips the already injected QuantLinear and keeps Block-AP's
    requires_grad (weight + zero_point would stay trainable)."""
    base = _base()
    phase1_cfg, phase2_cfg = efficient_qat_schedule(bits=4, group_size=64)

    model = get_quant_model(base, phase1_cfg)
    _train(model)
    after_block_ap = _snapshot(model)

    model = get_quant_model(base, phase2_cfg)
    for layer in _layers(model):
        trainable = {n for n in ("weight", "scale", "zero_point") if getattr(layer, n).requires_grad}
        assert trainable == TRAINABLE["e2e_qp"], f"after switch, trainable = {trainable}"
    for b, a in zip(after_block_ap, _snapshot(model)):
        for name in ("weight", "scale", "zero_point"):
            assert torch.equal(b[name], a[name]), f"{name} was re-initialized at the phase switch"

    before = _snapshot(model)
    _train(model)
    _assert_only_allowed_changed(before, _snapshot(model), TRAINABLE["e2e_qp"])
