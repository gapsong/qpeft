"""Save / load of a merged QuantModel -- see docs/specs/save_load.md.

Written BEFORE the implementation (red first). The implementing agent must not
modify, weaken, skip, or delete these tests; if the spec changes, the owner
changes the tests first.

Oracle: the in-memory merged model. A loaded model must compute bit-identically
to it (torch.equal, not allclose) and stay integer.
"""
import json
import sys

import pytest
import torch
import torch.nn as nn

from qpeft import (
    EfficientQATConfig, QALoraConfig, QuantModel, UnsupportedSchemeError,
    ZeroPointFoldLoRA, get_quant_model,
)
from qpeft.tuners.tuners_utils import QuantLinear

CONFIG_FILE = "qpeft_config.json"

# Factories, not instances: each test gets a fresh config object.
CONFIGS = {
    "qa_lora_4bit": lambda: QALoraConfig(bits=4, group_size=32, r=8),
    "eqat_block_ap_2bit": lambda: EfficientQATConfig(bits=2, group_size=32, phase="block_ap"),
    "eqat_e2e_qp_4bit": lambda: EfficientQATConfig(bits=4, group_size=32, phase="e2e_qp"),
}


def _base(seed: int, out_last: int = 32) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, out_last))


def _merged(cfg, seed: int = 0) -> QuantModel:
    """A merged QuantModel with a NON-trivial adapter, so the fold is really exercised."""
    model = get_quant_model(_base(seed), cfg)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, ZeroPointFoldLoRA):
                m.B.copy_(torch.randn_like(m.B) * 0.1)
    model.merge_and_unload()
    return model


def _quant_layers(model):
    layers = [m for m in model.modules() if isinstance(m, QuantLinear)]
    assert layers, "no QuantLinear found -- nothing was tested"
    return layers


# --- the oracle ---------------------------------------------------------------

@pytest.mark.parametrize("make_cfg", CONFIGS.values(), ids=CONFIGS.keys())
def test_roundtrip_is_bit_exact(tmp_path, make_cfg):
    model = _merged(make_cfg())
    x = torch.randn(4, 128)
    with torch.no_grad():
        before = model(x)

    model.save_pretrained(tmp_path)
    # Different seed on purpose: if loading silently re-quantized the fresh
    # base instead of restoring the saved tensors, this test must go red.
    loaded = QuantModel.from_pretrained(_base(seed=123), tmp_path)

    with torch.no_grad():
        after = loaded(x)
    assert torch.equal(before, after)


# --- merge stays int (in memory and on disk) ----------------------------------

@pytest.mark.parametrize("make_cfg", CONFIGS.values(), ids=CONFIGS.keys())
def test_merge_drops_float_weight(make_cfg):
    """Prerequisite: after merge the float weight must be gone, not just unused."""
    for layer in _quant_layers(_merged(make_cfg())):
        assert getattr(layer, "weight", None) is None
        assert not any(name.endswith("weight") and t.is_floating_point()
                       for name, t in layer.state_dict().items())


def test_loaded_model_is_merged_integer_and_frozen(tmp_path):
    model = _merged(CONFIGS["qa_lora_4bit"]())
    model.save_pretrained(tmp_path)
    loaded = QuantModel.from_pretrained(_base(seed=123), tmp_path)

    for orig, new in zip(_quant_layers(model), _quant_layers(loaded)):
        assert new.merged
        assert new.adapter is None
        assert new.qweight.dtype == torch.int32
        assert getattr(new, "weight", None) is None
        assert torch.equal(orig.bias, new.bias)
    assert not any(p.requires_grad for p in loaded.parameters())


# --- refuse instead of approximate --------------------------------------------

def test_saving_unmerged_model_refuses(tmp_path):
    model = get_quant_model(_base(0), CONFIGS["qa_lora_4bit"]())   # NOT merged
    with pytest.raises(ValueError):
        model.save_pretrained(tmp_path)


@pytest.mark.parametrize("field,value", [("backend", "does-not-exist"),
                                         ("qat_scheme", "does-not-exist")])
def test_unknown_contract_in_saved_config_refuses(tmp_path, field, value):
    _merged(CONFIGS["qa_lora_4bit"]()).save_pretrained(tmp_path)
    path = tmp_path / CONFIG_FILE
    cfg = json.loads(path.read_text())
    cfg[field] = value
    path.write_text(json.dumps(cfg))

    with pytest.raises(UnsupportedSchemeError):
        QuantModel.from_pretrained(_base(seed=123), tmp_path)


def test_torchao_backend_without_torchao_refuses(tmp_path, monkeypatch):
    _merged(CONFIGS["qa_lora_4bit"]()).save_pretrained(tmp_path)
    path = tmp_path / CONFIG_FILE
    cfg = json.loads(path.read_text())
    cfg["backend"] = "torchao"
    path.write_text(json.dumps(cfg))
    for name in [n for n in sys.modules if n == "torchao" or n.startswith("torchao.")] + ["torchao"]:
        monkeypatch.setitem(sys.modules, name, None)                # simulate "not installed"
    monkeypatch.delitem(sys.modules, "qpeft.quant_schemes.torchao", raising=False)

    with pytest.raises(NotImplementedError, match="needs torchao"):
        QuantModel.from_pretrained(_base(seed=123), tmp_path)


def test_shape_mismatch_refuses_instead_of_partial_load(tmp_path):
    _merged(CONFIGS["qa_lora_4bit"]()).save_pretrained(tmp_path)
    wrong_base = _base(seed=123, out_last=16)                       # 32 -> 16
    with pytest.raises((RuntimeError, ValueError)):
        QuantModel.from_pretrained(wrong_base, tmp_path)


# --- dtypes survive -----------------------------------------------------------

def test_fp16_dtypes_survive_roundtrip(tmp_path):
    model = get_quant_model(_base(0).half(), CONFIGS["eqat_e2e_qp_4bit"]())
    model.merge_and_unload()
    model.save_pretrained(tmp_path)
    loaded = QuantModel.from_pretrained(_base(seed=123).half(), tmp_path)

    for layer in _quant_layers(loaded):
        assert layer.qweight.dtype == torch.int32
        assert layer.scale.dtype == torch.float16
        assert layer.zero_point.dtype == torch.float16
        assert layer.bias.dtype == torch.float16
