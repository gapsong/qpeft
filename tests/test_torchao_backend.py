"""The torchao backend at the SAME gate as the pure-torch reference.

Skipped when torchao is not installed. This backend binds to torchao's STABLE
affine primitives (quant_primitives), not its in-flux packed tensor subclasses.
"""
import pytest

pytest.importorskip("torchao")

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402

from qpeft import (                                             # noqa: E402
    EfficientQATConfig, QALoraConfig, UnsupportedSchemeError,
    check_merge_equivalence, get_quant_model,
)
from qpeft.quant_schemes import build_scheme                          # noqa: E402
from qpeft.tuners.qa_lora.layer import ZeroPointFoldLoRA        # noqa: E402
from qpeft.tuners.tuners_utils import QuantLinear               # noqa: E402


def test_backend_selects_torchao_scheme():
    s = build_scheme(EfficientQATConfig(bits=4, group_size=64, phase="e2e_qp", backend="torchao"))
    assert type(s).__name__ == "TorchaoIntUniformScheme"


@pytest.mark.parametrize("bits", [4, 8])
def test_no_adapter_merge_is_exact(bits):
    torch.manual_seed(0)
    s = build_scheme(EfficientQATConfig(bits=bits, group_size=64, phase="e2e_qp", backend="torchao"))
    w = torch.randn(64, 128) * 0.02
    scale, z = s.init_qparams(w, 64)
    err = check_merge_equivalence(s, w, scale, z, adapter=None, x=torch.randn(8, 128))
    assert err == 0.0


def test_qa_lora_adapter_folds_into_zero_point():
    torch.manual_seed(0)
    s = build_scheme(QALoraConfig(bits=4, group_size=64, r=8, backend="torchao"))
    w = torch.randn(64, 128) * 0.02
    scale, z = s.init_qparams(w, 64)
    adapter = ZeroPointFoldLoRA(128, 64, r=8, alpha=16, group_size=64)
    with torch.no_grad():
        adapter.B.normal_(0, 0.1)
    err = check_merge_equivalence(s, w, scale, z, adapter=adapter, x=torch.randn(8, 128))
    assert err < 1e-4


def test_refuses_trainable_zero_point():
    """torchao's integer zero-point domain cannot train the zero-point -> refuse."""
    with pytest.raises(UnsupportedSchemeError):
        build_scheme(EfficientQATConfig(bits=4, group_size=64, phase="block_ap", backend="torchao"))


def test_training_over_torchao_then_int_merge():
    torch.manual_seed(0)
    cfg = QALoraConfig(bits=4, group_size=64, r=16, backend="torchao")
    model = get_quant_model(nn.Sequential(nn.Linear(128, 64)), cfg)
    x = torch.randn(64, 128)
    teacher = ZeroPointFoldLoRA(128, 64, r=4, alpha=16, group_size=64)
    with torch.no_grad():
        teacher.A.normal_(0, 1.0)
        teacher.B.normal_(0, 0.3)
        y = model(x) + teacher(x)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=5e-3)
    first = last = None
    for i in range(300):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        first = loss.item() if i == 0 else first
        last = loss.item()
    assert last < first * 0.5
    merged = model.merge_and_unload()
    q = next(m for m in merged.modules() if isinstance(m, QuantLinear))
    assert q.qweight.dtype == torch.int32
