"""The spine gate: fake_quant (training) must equal merge (export).

A scheme is only trustworthy once these pass. This locks the int_uniform
reference backend; every other backend (torchao, mlx) must clear the same test.
"""
import copy

import pytest
import torch
import torch.nn as nn

from qpeft import (
    EfficientQATConfig, QALoraConfig, check_merge_equivalence, get_quant_model,
)
from qpeft.quant_schemes import UnsupportedSchemeError, build_scheme
from qpeft.tuners.qa_lora.layer import ZeroPointFoldLoRA
from qpeft.tuners.tuners_utils import QuantLinear


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("group_size", [32, 64])
def test_efficient_qat_merge_is_exact(bits, group_size):
    """No adapter: the merge is an identity on the codes, so equivalence is exact."""
    torch.manual_seed(0)
    cfg = EfficientQATConfig(bits=bits, group_size=group_size, phase="block_ap")
    scheme = build_scheme(cfg)
    w = torch.randn(64, 128) * 0.02
    s, z = scheme.init_qparams(w, group_size)
    x = torch.randn(8, 128)
    err = check_merge_equivalence(scheme, w, s, z, adapter=None, x=x)
    assert err == 0.0


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 64])
def test_qa_lora_adapter_folds_into_zero_point(bits, group_size):
    """The pooled adapter folds into the per-group zero-points within tolerance."""
    torch.manual_seed(0)
    scheme = build_scheme(QALoraConfig(bits=bits, group_size=group_size, r=8))
    w = torch.randn(64, 128) * 0.02
    s, z = scheme.init_qparams(w, group_size)
    adapter = ZeroPointFoldLoRA(128, 64, r=8, alpha=16, group_size=group_size)
    with torch.no_grad():                       # a non-trivial delta to fold
        adapter.B.copy_(torch.randn_like(adapter.B) * 0.1)
    err = check_merge_equivalence(scheme, w, s, z, adapter=adapter, x=torch.randn(8, 128))
    assert err < 1e-4


def test_merge_and_unload_stays_integer():
    """merge_and_unload() returns an int artifact whose output matches training."""
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(128, 64))
    cfg = QALoraConfig(bits=4, group_size=64, r=8)
    model = get_quant_model(copy.deepcopy(base), cfg)
    x = torch.randn(4, 128)
    with torch.no_grad():
        before = model(x)
    merged = model.merge_and_unload()
    q = next(m for m in merged.modules() if isinstance(m, QuantLinear))
    assert q.qweight.dtype == torch.int32
    assert q.adapter is None
    with torch.no_grad():
        after = merged(x)
    assert torch.allclose(before, after, atol=1e-4)


def test_unimplemented_backend_refuses():
    """Refuse rather than approximate: an unbuilt backend must not silently fall back."""
    with pytest.raises((NotImplementedError, UnsupportedSchemeError)):
        build_scheme(EfficientQATConfig(bits=4, group_size=64, backend="mlx"))
