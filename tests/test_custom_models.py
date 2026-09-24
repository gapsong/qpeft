"""Behaviour on small hand-built models. ~ peft/tests/test_custom_models.py:
only the trainable axis updates, merge is idempotent, and a biased base survives
injection.
"""
import pytest
import torch
import torch.nn as nn

from qpeft import EfficientQATConfig, QALoraConfig, UnsupportedSchemeError, get_quant_model
from qpeft.tuners.tuners_utils import QuantLinear

IN, OUT = 128, 64


def _model(cfg, bias=False):
    return get_quant_model(nn.Sequential(nn.Linear(IN, OUT, bias=bias)), cfg)


def _leaves(model, trainable_only):
    out = set()
    for n, p in model.named_parameters():
        if trainable_only and not p.requires_grad:
            continue
        out.add("adapter" if "adapter" in n else n.split(".")[-1])
    return out


@pytest.mark.parametrize("cfg,expected", [
    (EfficientQATConfig(bits=3, group_size=64, phase="block_ap"),
     {"weight", "scale", "zero_point"}),
    (EfficientQATConfig(bits=3, group_size=64, phase="e2e_qp"), {"scale"}),
    (QALoraConfig(bits=4, group_size=64, r=8), {"adapter"}),
])
def test_only_trainable_params_are_updated(cfg, expected):
    """The trainable set is exactly the method's axis, and a step moves only it."""
    torch.manual_seed(0)
    model = _model(cfg)
    assert _leaves(model, trainable_only=True) == expected

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=1e-2)
    opt.zero_grad()
    model(torch.randn(8, IN)).pow(2).mean().backward()
    opt.step()

    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert changed, "nothing was updated"
    assert not (changed & frozen), f"a frozen param changed: {changed & frozen}"


def test_merge_layers_is_idempotent():
    """Merging an already-merged model is a no-op with identical output."""
    torch.manual_seed(0)
    model = _model(QALoraConfig(bits=4, group_size=64, r=8))
    x = torch.randn(4, IN)
    merged = model.merge_and_unload()
    with torch.no_grad():
        y1 = merged(x)
    for m in merged.modules():                      # merge again
        if isinstance(m, QuantLinear):
            m.merge()
    with torch.no_grad():
        y2 = merged(x)
    assert torch.equal(y1, y2)


def test_bias_survives_injection():
    """A Linear(bias=True) (common in HF models) keeps its bias through injection."""
    torch.manual_seed(0)
    base = nn.Linear(IN, OUT, bias=True)
    ref_bias = base.bias.detach().clone()
    model = get_quant_model(nn.Sequential(base),
                            EfficientQATConfig(bits=4, group_size=64, phase="block_ap"))
    q = next(m for m in model.modules() if isinstance(m, QuantLinear))
    assert q.bias is not None
    assert torch.equal(q.bias, ref_bias)
    assert not q.bias.requires_grad                 # bias is part of the frozen base
    assert torch.isfinite(model(torch.randn(2, IN))).all()


def test_bias_none_when_base_has_no_bias():
    model = _model(EfficientQATConfig(bits=4, group_size=64, phase="block_ap"), bias=False)
    q = next(m for m in model.modules() if isinstance(m, QuantLinear))
    assert q.bias is None


def test_fp16_base_survives_injection_and_forward():
    """A half-precision base (typical HF model) injects and runs without a dtype crash.
    scale / zero_point are fp32 masters while training (a half-precision optimizer
    step would not move them); the forward and the merged artifact stay fp16."""
    torch.manual_seed(0)
    base = nn.Linear(IN, OUT, bias=True).half()
    model = get_quant_model(nn.Sequential(base),
                            EfficientQATConfig(bits=4, group_size=64, phase="block_ap"))
    q = next(m for m in model.modules() if isinstance(m, QuantLinear))
    assert q.compute_dtype == torch.float16 and q.weight.dtype == torch.float16
    assert q.scale.dtype == torch.float32 and q.zero_point.dtype == torch.float32
    y = model(torch.randn(2, IN, dtype=torch.float16))
    assert y.dtype == torch.float16 and torch.isfinite(y).all()
    model.merge_and_unload()
    assert q.scale.dtype == torch.float16 and q.zero_point.dtype == torch.float16


def test_non_rtn_init_refuses_rather_than_approximates():
    """A non-RTN init must refuse, not silently leave a degenerate scale=1 grid."""
    with pytest.raises(UnsupportedSchemeError):
        get_quant_model(nn.Sequential(nn.Linear(IN, OUT)),
                        EfficientQATConfig(bits=4, group_size=64, init_weights="loftq"))
