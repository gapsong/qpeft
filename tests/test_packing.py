"""Code packing: the GPTQ qweight layout, checked against the AutoGPTQ pack loop."""
import numpy as np
import pytest
import torch

from qpeft.packing import pack_codes, unpack_codes


def _autogptq_pack(codes, bits):
    """AutoGPTQ QuantLinear.pack (qweight part), for codes of shape (out, in)."""
    intweight = codes.t().contiguous().numpy().astype(np.uint32)
    qweight = np.zeros((intweight.shape[0] // 32 * bits, intweight.shape[1]), dtype=np.uint32)
    i = row = 0
    while row < qweight.shape[0]:
        if bits in (2, 4, 8):
            for j in range(i, i + (32 // bits)):
                qweight[row] |= intweight[j] << (bits * (j - i))
            i += 32 // bits
            row += 1
        elif bits == 3:
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i))
            i += 10
            qweight[row] |= intweight[i] << 30
            row += 1
            qweight[row] |= (intweight[i] >> 2) & 1
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 1)
            i += 10
            qweight[row] |= intweight[i] << 31
            row += 1
            qweight[row] |= (intweight[i] >> 1) & 0x3
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 2)
            i += 10
            row += 1
    return torch.from_numpy(qweight.astype(np.int32))


def _codes(bits, out=24, inf=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2 ** bits, (out, inf), generator=g, dtype=torch.int32)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_pack_matches_autogptq(bits):
    codes = _codes(bits)
    assert torch.equal(pack_codes(codes, bits), _autogptq_pack(codes, bits))


@pytest.mark.parametrize("bits", range(1, 9))
def test_unpack_inverts_pack(bits):
    codes = _codes(bits, inf=128)
    packed = pack_codes(codes, bits)
    assert packed.dtype == torch.int32 and packed.shape == (128 * bits // 32, 24)
    assert torch.equal(unpack_codes(packed, bits, 128), codes)


def test_extreme_codes_survive_the_sign_bit():
    codes = torch.full((4, 64), 15, dtype=torch.int32)             # all bits set -> negative int32 words
    assert torch.equal(unpack_codes(pack_codes(codes, 4), 4, 64), codes)


def test_refuses_what_does_not_fill_whole_words():
    with pytest.raises(ValueError, match="multiple of 32"):
        pack_codes(_codes(3, inf=48), 3)
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        pack_codes(torch.full((2, 32), 16, dtype=torch.int32), 4)


def test_layer_refuses_shapes_that_cannot_be_packed():
    from qpeft import EfficientQATConfig, get_quant_model
    model = torch.nn.Sequential(torch.nn.Linear(48, 8))            # 48 x 3 bits = 144, not a multiple of 32
    with pytest.raises(ValueError, match="multiple of 32"):
        get_quant_model(model, EfficientQATConfig(bits=3, group_size=16, target_modules=["0"]))


def test_merged_codes_take_bits_per_weight():
    from qpeft import QALoraConfig, get_quant_model
    model = get_quant_model(torch.nn.Sequential(torch.nn.Linear(128, 64)),
                            QALoraConfig(bits=4, group_size=32, r=4, target_modules=["0"]))
    model.merge_and_unload()
    layer = model.base[0]
    assert layer.qweight.numel() * 32 == 128 * 64 * 4
    assert layer.zero_point.dtype == torch.float32                   # separate float zero-points
