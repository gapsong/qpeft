"""tinygemm export: the zero-point conversion, the int4 pair packing, refusals, and (CUDA) one layer
on the kernel against the merged QuantLinear."""
import pytest
import torch

from qpeft import EfficientQATConfig, QALoraConfig, UnsupportedSchemeError, get_quant_model
from qpeft.kernels.tinygemm import TinyGemmLinear, check_supported, pack_int4_pairs, tinygemm_zero, to_tinygemm
from qpeft.tuners.tuners_utils import QuantLinear

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinygemm is a CUDA kernel")


def _merged(cfg, in_f=256, out_f=64, bias=True, adapter_scale=0.1, seed=0):
    torch.manual_seed(seed)
    model = get_quant_model(torch.nn.Sequential(torch.nn.Linear(in_f, out_f, bias=bias)), cfg)
    layer = model.base[0]
    if layer.adapter is not None:                                   # a trained-looking adapter
        with torch.no_grad():
            layer.adapter.B.normal_(0, adapter_scale)
    model.merge_and_unload()
    return model, model.base[0]


def _qalora(gs=64, bits=4):
    return QALoraConfig(bits=bits, group_size=gs, r=8, target_modules=["0"])


# --- 1. the conversion ------------------------------------------------------------------

@pytest.mark.parametrize("fractional", [False, True])
def test_tinygemm_zero_gives_the_same_weight(fractional):
    g = torch.Generator().manual_seed(0)
    q = torch.randint(0, 16, (32, 8), generator=g).double()
    s = torch.rand(32, 8, generator=g, dtype=torch.float64) + 0.01
    z = torch.randint(0, 16, (32, 8), generator=g).double()
    if fractional:                                                  # a QA-LoRA fold
        z = z + torch.randn(32, 8, generator=g, dtype=torch.float64)
    torch.testing.assert_close((q - 8) * s + tinygemm_zero(s, z), (q - z) * s, rtol=0, atol=1e-12)


def test_int4_pairs_hold_even_columns_in_the_high_nibble():
    codes = torch.randint(0, 16, (8, 64), dtype=torch.int32)
    packed = pack_int4_pairs(codes).to(torch.int32)
    assert packed.dtype == torch.int32 and packed.shape == (8, 32)
    assert torch.equal(packed >> 4, codes[:, ::2]) and torch.equal(packed & 0xF, codes[:, 1::2])


# --- 3. refusals (no compute before the error) --------------------------------------------

def test_refuses_an_unmerged_layer():
    model = get_quant_model(torch.nn.Sequential(torch.nn.Linear(256, 64)), _qalora())
    with pytest.raises(UnsupportedSchemeError, match="merged"):
        check_supported(model.base[0])


@pytest.mark.parametrize("cfg, match", [
    (_qalora(bits=3), "int4 only"),
    (_qalora(gs=16), "group sizes"),
])
def test_refuses_what_the_kernel_cannot_run(cfg, match):
    _, layer = _merged(cfg)
    with pytest.raises(UnsupportedSchemeError, match=match):
        check_supported(layer)


def test_refuses_out_features_off_the_kernel_tiles():
    _, layer = _merged(_qalora(), out_f=60)
    with pytest.raises(UnsupportedSchemeError, match="out_features"):
        check_supported(layer)


def test_to_tinygemm_changes_nothing_when_one_layer_is_refused():
    model = get_quant_model(torch.nn.Sequential(torch.nn.Linear(256, 64), torch.nn.ReLU(),
                                                torch.nn.Linear(64, 60)),
                            QALoraConfig(bits=4, group_size=64, r=8, target_modules=["0", "2"]))
    model.merge_and_unload()
    with pytest.raises(UnsupportedSchemeError):
        to_tinygemm(model)
    assert all(isinstance(model.base[i], QuantLinear) for i in (0, 2))


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA refusal")
def test_refuses_without_cuda():
    _, layer = _merged(_qalora())
    with pytest.raises(UnsupportedSchemeError, match="CUDA"):
        check_supported(layer)


# --- 2. one layer on the kernel -------------------------------------------------------------

def _rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@cuda
@pytest.mark.parametrize("gs", [32, 64, 128, 256])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("method", ["qa_lora", "eqat"])
def test_kernel_matches_the_merged_layer(gs, bias, method):
    cfg = (_qalora(gs) if method == "qa_lora"
           else EfficientQATConfig(bits=4, group_size=gs, phase="e2e_qp", target_modules=["0"]))
    _, layer = _merged(cfg, in_f=512, out_f=128, bias=bias)
    layer = layer.cuda()
    kernel = TinyGemmLinear(layer)
    for shape in [(1, 512), (5, 512), (2, 7, 512)]:
        x = torch.randn(*shape, device="cuda")
        with torch.no_grad():
            ref, got = layer(x), kernel(x)
        assert got.shape == ref.shape and got.dtype == ref.dtype
        assert _rel_err(got, ref) < 1e-2, f"{shape}: rel err {_rel_err(got, ref):.2e}"


@cuda
def test_wrong_zero_point_is_caught():
    """Negative control for the tolerance above: the qpeft zero-point passed as-is (no z_f
    conversion) must fail it."""
    _, layer = _merged(_qalora(), in_f=512, out_f=128)
    layer = layer.cuda()
    kernel = TinyGemmLinear(layer)
    s = layer.scale.float()
    kernel.scales_and_zeros.copy_(torch.stack([s, layer.zero_point.float()], -1).transpose(0, 1).to(torch.bfloat16))
    x = torch.randn(5, 512, device="cuda")
    with torch.no_grad():
        assert _rel_err(kernel(x), layer(x)) > 1e-1
