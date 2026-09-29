"""The Triton kernel for frozen codes (qpeft/kernels/dequant.py) must give exactly the torch
path's numbers: torch.equal, not allclose, for the weight, the output and every gradient.
Every comparison also asserts that the kernel really ran, so a silent fallback to the torch
path cannot pass. CI has no GPU; there the torch path is covered by the rest of the suite."""
import random
import time

import pytest
import torch
from torch import nn

from qpeft import EfficientQATConfig, PEQAConfig, QALoraConfig, get_quant_model, verify_quant_model
from qpeft.kernels import dequant
from qpeft.tuners.tuners_utils import quant_layers

pytestmark = pytest.mark.skipif(not dequant.TRITON_AVAILABLE, reason="needs CUDA and triton")

DEVICE = "cuda"


@pytest.fixture
def kernel_calls(monkeypatch):
    """Counts the kernel launches of dequant.dequantize."""
    calls = []
    original = dequant.dequantize

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(dequant, "dequantize", counting)
    return calls


def make_layer(method, bits, group_size, in_features, out_features, dtype, bias=True, seed=0):
    torch.manual_seed(seed)
    base = nn.Sequential(nn.Linear(in_features, out_features, bias=bias)).to(DEVICE, dtype)
    common = dict(bits=bits, group_size=group_size)
    config = {"peqa": PEQAConfig(**common),
              "e2e_qp": EfficientQATConfig(phase="e2e_qp", **common),
              "qa_lora": QALoraConfig(r=8, **common)}[method]
    model = get_quant_model(base, config)
    layer = quant_layers(model)[0]
    if layer.adapter is not None:
        with torch.no_grad():                       # B starts at 0; make the adapter do something
            layer.adapter.B.normal_(0, 0.02)
    codes = layer.codes
    assert codes.min() == 0 and codes.max() == 2 ** bits - 1, "RTN init puts codes on 0 and qmax"
    return model, layer


def forward_backward(layer, x, grad_y, use_kernel):
    """Output, input gradient and the gradient of every trainable parameter."""
    layer.use_triton_kernel = use_kernel
    layer.zero_grad(set_to_none=True)
    x = x.detach().requires_grad_(True)
    y = layer(x)
    y.backward(grad_y)
    grads = {name: p.grad.clone() for name, p in layer.named_parameters() if p.requires_grad}
    return y.detach(), x.grad.clone(), grads


def assert_kernel_matches_torch(layer, x, kernel_calls, message=""):
    grad_y = torch.randn(*x.shape[:-1], layer.out_features, device=DEVICE, dtype=x.dtype)
    y_ref, dx_ref, grads_ref = forward_backward(layer, x, grad_y, use_kernel=False)
    assert not kernel_calls, "the torch path must not launch the kernel"
    y, dx, grads = forward_backward(layer, x, grad_y, use_kernel=True)
    assert kernel_calls, "the kernel did not run (silent fallback to torch)"
    kernel_calls.clear()
    assert torch.equal(y, y_ref), message
    assert torch.equal(dx, dx_ref), message
    assert grads.keys() == grads_ref.keys() and grads, message
    for name in grads:
        assert torch.equal(grads[name], grads_ref[name]), f"{name} {message}"


# --- the dense weight -------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dequantized_weight_is_bit_identical(bits, group_size, dtype):
    _, layer = make_layer("peqa", bits, group_size, 384, 97, dtype)
    scale = layer.scheme.clamp_scale(layer.scale.detach().to(dtype))
    zero_point = layer._zero_point_used().detach().to(dtype)
    expected = layer.scheme.dequant(layer.codes, scale, zero_point)
    weight = dequant.dequantize(layer.qweight, scale, zero_point, bits, group_size)
    assert weight.shape == (97, 384) and weight.is_contiguous()
    assert torch.equal(weight, expected)


# --- forward and backward through the layer ---------------------------------------------

BATCH_SHAPES = [(5,), (2, 3, 7), (3, 11)]


def make_input(batch_shape, in_features, dtype, contiguous=True):
    if contiguous:
        return torch.randn(*batch_shape, in_features, device=DEVICE, dtype=dtype)
    # every other column of a wider tensor: same shape, not contiguous
    return torch.randn(*batch_shape, 2 * in_features, device=DEVICE, dtype=dtype)[..., ::2]


@pytest.mark.parametrize("method", ["peqa", "e2e_qp", "qa_lora"])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_output_and_gradients_are_bit_identical(method, bits, dtype, batch_shape, kernel_calls):
    _, layer = make_layer(method, bits, 64, 256, 129, dtype)
    assert_kernel_matches_torch(layer, make_input(batch_shape, 256, dtype), kernel_calls)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("group_size", [32, 128])
def test_non_contiguous_input(bias, group_size, kernel_calls):
    _, layer = make_layer("peqa", 4, group_size, 256, 63, torch.bfloat16, bias=bias)
    x = make_input((3, 5), 256, torch.bfloat16, contiguous=False)
    assert not x.is_contiguous()
    assert_kernel_matches_torch(layer, x, kernel_calls)


@pytest.mark.parametrize("method", ["peqa", "qa_lora"])
def test_model_sized_layer(method, kernel_calls):
    """Many tokens and a wide input: cuBLAS may pick other algorithms than for small shapes."""
    _, layer = make_layer(method, 4, 128, 2048, 1024, torch.bfloat16)
    assert_kernel_matches_torch(layer, make_input((4, 256), 2048, torch.bfloat16), kernel_calls)


def test_shapes_drawn_at_test_time(kernel_calls):
    """Shapes nobody knew when the kernel was written; the seed is printed on failure."""
    seed = time.time_ns() % 2 ** 32
    rng = random.Random(seed)
    for _ in range(20):
        bits, group_size = rng.choice([2, 4, 8]), rng.choice([32, 64, 128])
        in_features = group_size * rng.randint(1, 12)
        out_features = rng.randint(1, 300)
        batch_shape = tuple(rng.randint(1, 9) for _ in range(rng.randint(1, 3)))
        method = rng.choice(["peqa", "e2e_qp", "qa_lora"])
        dtype = rng.choice([torch.bfloat16, torch.float16])
        message = (f"seed={seed}: {method} bits={bits} group_size={group_size} "
                   f"in={in_features} out={out_features} batch={batch_shape} {dtype}")
        _, layer = make_layer(method, bits, group_size, in_features, out_features, dtype, seed=rng.randint(0, 999))
        x = make_input(batch_shape, in_features, dtype, contiguous=rng.random() < 0.7)
        assert_kernel_matches_torch(layer, x, kernel_calls, message)


# --- merged layers, autocast, the merge gate ---------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_merged_layer_with_fractional_zero_point(dtype, kernel_calls):
    model, layer = make_layer("qa_lora", 4, 64, 256, 129, dtype)
    model.merge_and_unload()
    assert layer.merged and not torch.equal(layer.zero_point, layer.zero_point.round())
    x = make_input((2, 9), 256, dtype)
    layer.use_triton_kernel = False
    expected = layer(x)
    layer.use_triton_kernel = True
    with torch.no_grad():
        y = layer(x)
    assert kernel_calls and torch.equal(y, expected)


def test_merge_gate_stays_green_with_the_kernel(kernel_calls):
    model, _ = make_layer("qa_lora", 4, 64, 256, 129, torch.bfloat16)
    verify_quant_model(model)
    assert kernel_calls


def test_autocast_to_the_input_dtype_uses_the_kernel(kernel_calls):
    _, layer = make_layer("peqa", 4, 64, 256, 129, torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert_kernel_matches_torch(layer, make_input((2, 7), 256, torch.bfloat16), kernel_calls)


def test_autocast_to_another_dtype_takes_the_torch_path(kernel_calls):
    _, layer = make_layer("peqa", 4, 64, 256, 129, torch.bfloat16)
    x = make_input((2, 7), 256, torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        layer(x)
    assert not kernel_calls


def test_three_bit_codes_take_the_torch_path(kernel_calls):
    _, layer = make_layer("peqa", 3, 32, 256, 64, torch.bfloat16)
    layer(make_input((2,), 256, torch.bfloat16))
    assert not kernel_calls
