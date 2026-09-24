"""Straight-through gradients of the int_uniform primitives equal the official EfficientQAT
quantizer's (clamp_ste / round_ste): gradient 1 through every clamp and round."""
import torch

from qpeft.quant_schemes.base import FakeQuantizeConfig
from qpeft.quant_schemes.reference import ReferenceIntUniformScheme


def _scheme(bits=4, group_size=8):
    return ReferenceIntUniformScheme(FakeQuantizeConfig(dtype=f"int{bits}", group_size=group_size))


def test_clamp_scale_passes_gradient_one():
    s = torch.tensor([0.5, 2.0, 1e-6, 1e6], dtype=torch.float64, requires_grad=True)
    _scheme().clamp_scale(s).sum().backward()
    assert torch.equal(s.grad, torch.ones_like(s))           # inside and outside the bounds (STE)


def test_round_zero_point_passes_gradient_one():
    z = torch.tensor([0.4, 3.6, -2.0, 99.0], dtype=torch.float64, requires_grad=True)
    _scheme().round_zero_point(z).sum().backward()
    assert torch.equal(z.grad, torch.ones_like(z))


def test_fake_quant_gradients_match_the_official_ste_formula():
    """Official UniformAffineQuantizer.fake_quant, written out:
        code = clamp(round(w / s) + z, qmin, qmax);  w_hat = (code - z) * s
    with round_ste / clamp_ste, i.e. d w_hat / d s = (code - z) - [in range] * w / s,
    d w_hat / d z = -[clamped], d w_hat / d w = [in range]."""
    torch.manual_seed(0)
    bits, g, out, inf = 4, 8, 6, 32
    sch = _scheme(bits, g)
    w = (torch.randn(out, inf, dtype=torch.float64) * 0.1).requires_grad_(True)
    s0, z0 = sch.init_qparams(w.detach(), g)
    s = (s0 * 0.8).requires_grad_(True)                      # shrink the grid so some codes clamp
    z = z0.clone().requires_grad_(True)
    G = torch.randn(out, inf, dtype=torch.float64)
    (sch.fake_quant(w, s, z) * G).sum().backward()

    with torch.no_grad():
        se, ze = s.repeat_interleave(g, -1), z.round().repeat_interleave(g, -1)
        r = (w / se).round() + ze
        inr = (r >= sch.qmin) & (r <= sch.qmax)
        code = r.clamp(sch.qmin, sch.qmax)
        ds = ((code - ze) - torch.where(inr, w / se, torch.zeros_like(w))) * G
        dz = torch.where(inr, torch.zeros_like(w), -torch.ones_like(w)) * G * se
        dw = torch.where(inr, G, torch.zeros_like(G))
    assert inr.logical_not().any(), "the test needs clamped codes"
    torch.testing.assert_close(s.grad, ds.view(out, -1, g).sum(-1))
    torch.testing.assert_close(z.grad, dz.view(out, -1, g).sum(-1))
    torch.testing.assert_close(w.grad, dw)


def test_fp16_codes_round_before_adding_the_zero_point():
    torch.manual_seed(0)
    bits, g = 4, 16
    sch = _scheme(bits, g)
    w = (torch.randn(256, 512) * 0.05).half()
    s, z = sch.init_qparams(w.float(), g)
    s, z = s.half(), z.half()
    se, ze = s.repeat_interleave(g, -1), z.repeat_interleave(g, -1)
    official = ((w / se).round() + ze).clamp(sch.qmin, sch.qmax)
    assert torch.equal(sch.quantize(w, s, z).to(official.dtype), official)
    assert torch.equal(sch.fake_quant(w, s, z), (official - ze) * se)
