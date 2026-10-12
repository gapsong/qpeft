"""compressed-tensors itself and transformers read qpeft's compressed-tensors export: on a tiny random
Llama, on CPU, no download.

Needs compressed-tensors >= 0.18 (its dense packing of 3 / 5 / 6 / 7-bit values); skipped without it:
    pip install -e ".[dev]" "compressed-tensors>=0.18"
    pytest tests/test_compressed_tensors_load.py

Oracle: the merged qpeft model.
    - compressed-tensors' own packer, fed qpeft's codes and zero-points the way llm-compressor feeds
      it, writes the same words as the export.
    - transformers' AutoModelForCausalLM.from_pretrained decompresses every layer to qpeft's weight bit
      for bit, and so computes the same logits bit for bit."""
import pytest

ct = pytest.importorskip("compressed_tensors")
pytest.importorskip("transformers")

import torch                                                            # noqa: E402
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32   # noqa: E402
from packaging.version import Version                                   # noqa: E402
from safetensors.torch import load_file                                 # noqa: E402
from transformers import AutoModelForCausalLM                           # noqa: E402

from export_models import VOCAB, merged_efficient_qat, merged_peqa, quant_layers   # noqa: E402
from qpeft.export import export_compressed_tensors                      # noqa: E402

if Version(ct.__version__) < Version("0.18"):
    pytest.skip("compressed-tensors < 0.18 pads 3 / 5 / 6 / 7-bit values to whole words", allow_module_level=True)

ALL_BITS = [2, 3, 4, 5, 6, 7, 8]


@pytest.mark.parametrize("bits", ALL_BITS)
def test_compressed_tensors_packs_the_same_words(bits, tmp_path):
    model = merged_peqa(bits)
    export_compressed_tensors(model, tmp_path)
    tensors = load_file(tmp_path / "model.safetensors")
    offset = 1 << (bits - 1)
    for name, layer in quant_layers(model).items():
        signed_codes = (layer.codes - offset).to(torch.int8)
        signed_zeros = (layer.zero_point.to(torch.int32) - offset).to(torch.int8)
        assert torch.equal(tensors[f"{name}.weight_packed"], pack_to_int32(signed_codes, bits)), name
        assert torch.equal(tensors[f"{name}.weight_zero_point"], pack_to_int32(signed_zeros, bits, packed_dim=0)), name


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bits", ALL_BITS)
@pytest.mark.parametrize("method", ["peqa", "efficient_qat"])
def test_transformers_loads_the_qpeft_model_bit_for_bit(method, bits, dtype, tmp_path):
    model = merged_peqa(bits, dtype) if method == "peqa" else merged_efficient_qat(bits, dtype)
    export_compressed_tensors(model, tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=dtype, device_map="cpu").eval()

    ids = torch.randint(0, VOCAB, (4, 32), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected, got = model(input_ids=ids).logits, loaded(input_ids=ids).logits   # decompresses on first call
    assert torch.equal(got, expected)

    loaded_modules = dict(loaded.named_modules())
    for name, layer in quant_layers(model).items():
        assert torch.equal(loaded_modules[name].weight, layer._weight_in(dtype)), name
