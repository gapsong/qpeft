"""PEQA against its paper: Kim et al., "Memory-Efficient Fine-Tuning of Compressed Large
Language Models via sub-4-bit Integer Quantization" (NeurIPS 2023, arXiv 2305.14152).

There is no official code, so the reference is the paper's equations, written out here
independently of qpeft:
  Eq. 1  w_bar = clamp(round(w0 / s0) + z0, 0, 2**b - 1) - z0      (frozen integers)
         w_hat0 = s0 * w_bar                                        (RTN init of s0, z0)
  Eq. 2  w_hat = (s0 + delta_s) * w_bar                             (only delta_s trains)
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qpeft import PEQAConfig, QuantModel, QuantTuningConfig, TrainableParams, get_quant_model
from qpeft.quant_schemes.base import SCALE_MIN
from qpeft.tuners.tuners_utils import QuantLinear

IN, OUT, BITS, GROUP = 128, 64, 4, 32


def _backends():
    out = ["auto"]
    try:
        import torchao  # noqa: F401
        out.append("torchao")
    except ImportError:
        pass
    return out


class PaperPEQALinear(nn.Module):
    """PEQA as the paper writes it (Eq. 1 and 2), group-wise, RTN (min/max) init."""

    def __init__(self, base: nn.Linear, bits: int, group_size: int):
        super().__init__()
        qmax = 2 ** bits - 1
        w0 = base.weight.detach()
        groups = w0.reshape(w0.shape[0], -1, group_size)
        w_min, w_max = groups.amin(-1, keepdim=True), groups.amax(-1, keepdim=True)
        s0 = (w_max - w_min) / qmax
        z0 = torch.clamp(torch.round(-w_min / s0), 0, qmax)
        w_bar = torch.clamp(torch.round(groups / s0) + z0, 0, qmax) - z0
        self.register_buffer("w_bar", w_bar)                       # frozen integers
        self.s = nn.Parameter(s0.clone())                          # s0 + delta_s
        self.register_buffer("bias", base.bias.detach().clone())

    def weight(self):
        return (self.s * self.w_bar).reshape(self.w_bar.shape[0], -1)

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


def _base(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(IN, OUT))


def _qpeft_layer(model) -> QuantLinear:
    return next(m for m in model.modules() if isinstance(m, QuantLinear))


def _dequant(layer: QuantLinear) -> torch.Tensor:
    return layer.scheme.dequant(layer.codes, layer.scale, layer.scheme.round_zero_point(layer.zero_point))


# --- the config ----------------------------------------------------------------

def test_config_trains_only_the_scale():
    assert PEQAConfig().trainable_params == (TrainableParams.SCALE,)


def test_config_defaults_follow_the_paper():
    """Table 13, the grouped setting (LLaMA2): 4 bits, group size 256."""
    cfg = PEQAConfig()
    assert (cfg.bits, cfg.group_size) == (4, 256)


def test_config_roundtrip(tmp_path):
    cfg = PEQAConfig(bits=3, group_size=64, target_modules=["q_proj"])
    assert QuantTuningConfig.from_dict(cfg.to_dict()) == cfg
    cfg.save_pretrained(tmp_path)
    assert QuantTuningConfig.from_pretrained(tmp_path) == cfg


# --- against the paper -----------------------------------------------------------

@pytest.mark.parametrize("backend", _backends())
def test_init_matches_paper_eq1(backend):
    base = _base()
    paper = PaperPEQALinear(base[0], BITS, GROUP)
    layer = _qpeft_layer(get_quant_model(base, PEQAConfig(bits=BITS, group_size=GROUP, backend=backend)))

    assert layer.weight is None and layer.codes_frozen, "PEQA keeps no fp weight"
    assert torch.equal(layer.scale.detach(), paper.s.detach().squeeze(-1))
    assert torch.allclose(_dequant(layer), paper.weight(), atol=1e-6)


@pytest.mark.parametrize("backend", _backends())
def test_training_matches_paper_eq2(backend):
    """Same data, same optimizer: qpeft and the paper formula take the same steps.
    The target is a nearby fp model and the lr is small, as in fine-tuning (paper Table 12/13:
    lr 6e-6 .. 1e-3). The scale then stays positive; below SCALE_MIN qpeft clamps it and the
    paper does not, so that regime is not compared here."""
    base = _base()
    paper = PaperPEQALinear(base[0], BITS, GROUP)
    model = get_quant_model(base, PEQAConfig(bits=BITS, group_size=GROUP, backend=backend))
    layer = _qpeft_layer(model)
    codes0, z0, scale0 = layer.codes, layer.zero_point.detach().clone(), layer.scale.detach().clone()

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert [id(p) for p in trainable] == [id(layer.scale)], "PEQA trains the scale and nothing else"

    g = torch.Generator().manual_seed(1)
    teacher = _base(seed=0)[0]
    with torch.no_grad():
        teacher.weight.mul_(1 + 0.1 * torch.randn(teacher.weight.shape, generator=g))
        x = torch.randn(256, IN, generator=g)
        y = teacher(x)
    opt_q = torch.optim.Adam(trainable, lr=1e-4)
    opt_p = torch.optim.Adam(paper.parameters(), lr=1e-4)
    for _ in range(50):
        for opt, net in ((opt_q, model), (opt_p, paper)):
            opt.zero_grad()
            F.mse_loss(net(x), y).backward()
            opt.step()

    assert not torch.equal(layer.scale.detach(), scale0), "the scale did not train"
    assert (paper.s > SCALE_MIN).all(), "the test left the regime where qpeft and the paper agree"
    assert torch.allclose(layer.scale.detach(), paper.s.detach().squeeze(-1), atol=1e-6)
    assert torch.equal(layer.codes, codes0), "PEQA must never re-assign the integer codes"
    assert torch.equal(layer.zero_point.detach(), z0), "PEQA keeps the zero-point frozen"
    with torch.no_grad():
        assert torch.allclose(model(x), paper(x), atol=1e-5)


@pytest.mark.parametrize("backend", _backends())
def test_merge_stays_int_and_equals_training(backend, tmp_path):
    base = _base()
    model = get_quant_model(base, PEQAConfig(bits=BITS, group_size=GROUP, backend=backend))
    layer = _qpeft_layer(model)
    with torch.no_grad():
        layer.scale.mul_(1.1)                     # stands in for a trained delta_s
        x = torch.randn(8, IN)
        before = model(x)
    codes0 = layer.codes

    model.merge_and_unload()
    assert layer.merged and layer.qweight.dtype == torch.int32
    assert torch.equal(layer.codes, codes0)
    with torch.no_grad():
        assert torch.equal(model(x), before)

    model.save_pretrained(tmp_path)
    loaded = QuantModel.from_pretrained(_base(), tmp_path)
    assert isinstance(loaded.config, PEQAConfig)
    with torch.no_grad():
        assert torch.equal(loaded(x), before)
