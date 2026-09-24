"""The merged artifact: exact data format + peft's merge checks, applied to qpeft.

peft (tests/testing_common.py, PeftCommonTester) checks for every method:
  _test_merge_layers            merged logits == unmerged logits (atol=rtol=1e-4),
                                no tuner layer left behind
  _test_merge_layers_fp16       merging in half precision just works
  _test_merge_layers_is_idempotent
  _test_merge_layers_nan        NaNs in the merged weights are refused
  _test_save_pretrained         every state_dict key survives save -> load
qpeft adds what peft does not have: after the merge the layer IS an integer artifact.

Merged QuantLinear format (the contract these tests pin):
  qweight     int32, shape (out, in), values in [0, 2**bits - 1]
  scale       base dtype, shape (out, in // group_size), finite, > 0
  zero_point  base dtype, shape (out, in // group_size), finite
  bias        unchanged (value and dtype), or None
  weight      None  (no fp master copy)      adapter  None
  merged      True, nothing requires grad, state_dict keys = exactly the above
"""
import copy

import pytest
import torch
import torch.nn as nn

from qpeft import EfficientQATConfig, QALoraConfig, QuantModel, get_quant_model
from qpeft.tuners.qa_lora.layer import ZeroPointFoldLoRA
from qpeft.tuners.tuners_utils import QuantLinear

IN, HID, OUT = 128, 64, 32
TARGETS = ["0", "2"]                       # nn.Sequential names; "4" (the head) stays fp

METHODS = {
    "eqat_block_ap": lambda bits, gs: EfficientQATConfig(bits=bits, group_size=gs, phase="block_ap",
                                                        target_modules=TARGETS),
    "eqat_e2e_qp": lambda bits, gs: EfficientQATConfig(bits=bits, group_size=gs, phase="e2e_qp",
                                                      target_modules=TARGETS),
    "qa_lora": lambda bits, gs: QALoraConfig(bits=bits, group_size=gs, r=8, target_modules=TARGETS),
}


def _base(dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(IN, HID, bias=True), nn.ReLU(),
                         nn.Linear(HID, HID, bias=False), nn.ReLU(),
                         nn.Linear(HID, OUT)).to(dtype)


def _trained(method, bits=4, gs=32, dtype=torch.float32):
    """A QuantModel that moved away from its init, so the merge is non-trivial."""
    model = get_quant_model(_base(dtype), METHODS[method](bits, gs))
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, ZeroPointFoldLoRA):
                m.B.copy_(torch.randn_like(m.B) * 0.1)
    if dtype == torch.float32:                  # a few real steps on whatever is trainable
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        for _ in range(3):
            opt.zero_grad()
            model(torch.randn(16, IN)).pow(2).mean().backward()
            opt.step()
    return model


def _assert_merged_format(layer, bits, gs, dtype, bias_before):
    out_f, in_f = layer.qweight.shape
    assert layer.merged
    assert layer.weight is None, "fp master weight survived the merge"
    assert layer.adapter is None, "adapter survived the merge"

    assert layer.qweight.dtype == torch.int32
    assert int(layer.qweight.min()) >= 0 and int(layer.qweight.max()) <= 2 ** bits - 1

    for name in ("scale", "zero_point"):
        t = getattr(layer, name)
        assert t.shape == (out_f, in_f // gs), f"{name} shape {tuple(t.shape)}"
        assert t.dtype == dtype, f"{name} dtype {t.dtype} != {dtype}"
        assert torch.isfinite(t).all(), f"{name} not finite"
        assert not t.requires_grad
    assert (layer.scale > 0).all()

    if bias_before is None:
        assert layer.bias is None
    else:
        assert layer.bias.dtype == dtype and torch.equal(layer.bias, bias_before)

    expected = {"qweight", "scale", "zero_point"} | ({"bias"} if bias_before is not None else set())
    assert set(layer.state_dict()) == expected, f"state_dict keys {sorted(layer.state_dict())}"


# --- format + peft _test_merge_layers ----------------------------------------------

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_merge_layers_fp32(method, bits):
    gs = 32
    model = _trained(method, bits=bits, gs=gs)
    biases = {n: (None if m.bias is None else m.bias.detach().clone())
              for n, m in model.base.named_modules() if isinstance(m, QuantLinear)}
    x = torch.randn(8, IN)
    with torch.no_grad():
        before = model(x)
    merged = model.merge_and_unload()
    with torch.no_grad():
        after = merged(x)

    assert torch.allclose(before, after, atol=1e-4, rtol=1e-4), \
        f"max|delta|={(before - after).abs().max().item():.2e}"
    layers = {n: m for n, m in merged.named_modules() if isinstance(m, QuantLinear)}
    assert set(layers) == set(biases) == set(TARGETS)
    for n, layer in layers.items():
        _assert_merged_format(layer, bits, gs, torch.float32, biases[n])
        assert not any(p.requires_grad for p in layer.parameters())


@pytest.mark.parametrize("method", METHODS)
def test_merge_layers_fp16(method):
    """peft: 'This should simply work'. Plus: the format keeps the base dtype."""
    model = _trained(method, dtype=torch.float16)
    x = torch.randn(4, IN, dtype=torch.float16)
    with torch.no_grad():
        before = model(x)
    merged = model.merge_and_unload()
    with torch.no_grad():
        after = merged(x)
    assert torch.isfinite(after).all()
    assert torch.allclose(before.float(), after.float(), atol=1e-2, rtol=1e-2)
    for layer in (m for m in merged.modules() if isinstance(m, QuantLinear)):
        assert layer.qweight.dtype == torch.int32
        assert layer.scale.dtype == layer.zero_point.dtype == torch.float16


# --- only the targets change -------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_non_target_layers_untouched(method):
    base = _base()
    head_before = copy.deepcopy(base[4])
    merged = get_quant_model(base, METHODS[method](4, 32)).merge_and_unload()
    head = merged[4]
    assert type(head) is nn.Linear
    assert torch.equal(head.weight, head_before.weight) and torch.equal(head.bias, head_before.bias)


# --- idempotent / NaN (peft) -------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_merge_is_idempotent(method):
    merged = _trained(method).merge_and_unload()
    x = torch.randn(4, IN)
    with torch.no_grad():
        y0 = merged(x)
    state0 = {k: v.clone() for k, v in merged.state_dict().items()}
    for m in merged.modules():
        if isinstance(m, QuantLinear):
            m.merge()
    with torch.no_grad():
        y1 = merged(x)
    assert torch.equal(y0, y1)
    assert all(torch.equal(state0[k], v) for k, v in merged.state_dict().items())


def test_merge_with_nan_adapter_refuses_and_leaves_layer_untouched():
    model = _trained("qa_lora")
    layer = next(m for m in model.modules() if isinstance(m, QuantLinear))
    with torch.no_grad():
        layer.adapter.B[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        layer.merge()
    assert not layer.merged and layer.weight is not None and layer.adapter is not None


# --- save -> load: every key equal (peft _test_save_pretrained) ----------------------

@pytest.mark.parametrize("method", METHODS)
def test_save_load_every_key_equal(tmp_path, method):
    model = _trained(method)
    model.merge_and_unload()
    model.save_pretrained(tmp_path)
    loaded = QuantModel.from_pretrained(_base(seed=99), tmp_path)

    saved, got = model.state_dict(), loaded.state_dict()
    assert set(saved) == set(got)
    for k in saved:
        assert saved[k].dtype == got[k].dtype, k
        assert torch.equal(saved[k], got[k]), k
    for layer in (m for m in loaded.modules() if isinstance(m, QuantLinear)):
        assert layer.merged and layer.weight is None and layer.qweight.dtype == torch.int32
