"""A real GPTQ loader reads qpeft's GPTQ export: transformers' AutoModelForCausalLM.from_pretrained
(through optimum and gptqmodel) on a tiny random Llama, on CPU, no download.

Needs gptqmodel and optimum; skipped without them. Install them in a venv WITHOUT axolotl
(axolotl pins optimum==1.16.2, which cannot import optimum.gptq on transformers 5):
    pip install -e ".[dev]" gptqmodel optimum requests pillow
    pytest tests/test_gptq_load.py

Oracle: the merged qpeft model.
    - Per layer, gptqmodel's torch kernel dequantizes in fp16; its weight must equal qpeft's dequant
      in fp16 bit for bit (bf16 or fp16 model).
    - Logits: the merged model upcast to fp32 computes the exact model, since its fp16 / bf16 weights
      are exact in fp32. The loaded model may be no farther from it than qpeft's own half-precision
      forward is, up to a factor for the different rounding order; and it picks the same argmax.
The loader's default CPU kernel for 4 bits (TorchAtenLinear) rounds the scales to bf16 itself,
which is within bf16 noise for a bf16 model but 8x fp16 noise for an fp16 model; so it is
checked on a bf16 model."""
import copy

import pytest

pytest.importorskip("gptqmodel")
pytest.importorskip("optimum.gptq")

import torch                                                            # noqa: E402
from transformers import AutoModelForCausalLM, GPTQConfig               # noqa: E402

from qpeft import PEQAConfig, get_quant_model                           # noqa: E402
from qpeft.export import export_gptq                                    # noqa: E402
from qpeft.tuners.tuners_utils import QuantLinear                       # noqa: E402

pytestmark = [
    # Choosing the kernel with GPTQConfig(backend=...) is how transformers documents it, and it warns.
    pytest.mark.filterwarnings("ignore:You passed `quantization_config`:UserWarning"),
    # gptqmodel leaves its offload TemporaryDirectory to the garbage collector.
    pytest.mark.filterwarnings("ignore:Implicitly cleaning up <TemporaryDirectory:ResourceWarning"),
]
BLOCK_LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
VOCAB = 320
# Measured: loaded vs exact is at most 1.12x qpeft's own fp16 error (torch kernel), 1.39x its
# bf16 error (default kernel, bf16 model).
ROUNDING_ORDER_FACTOR = 2.0


def _merged(bits, dtype, targets=BLOCK_LINEARS):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    base = LlamaForCausalLM(LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, vocab_size=VOCAB, max_position_embeddings=64)).to(dtype).eval()
    model = get_quant_model(base, PEQAConfig(bits=bits, group_size=64, target_modules=targets))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):                                                  # trained, not RTN, scales
        opt.zero_grad()
        ids = torch.randint(0, VOCAB, (2, 16))
        model(input_ids=ids, labels=ids).loss.backward()
        opt.step()
    model.merge_and_unload()
    return model


def _load(path, bits, dtype, backend):
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, device_map="cpu",
        quantization_config=GPTQConfig(bits=bits, backend=backend)).eval()


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits.float()


def _ids():
    return torch.randint(0, VOCAB, (4, 32), generator=torch.Generator().manual_seed(1))


def _assert_as_close_as_qpeft_itself(model, loaded, same_argmax=True):
    ids = _ids()
    exact = _logits(copy.deepcopy(model).float(), ids)
    ref, got = _logits(model, ids), _logits(loaded, ids)
    if same_argmax:
        assert torch.equal(got.argmax(-1), ref.argmax(-1))
    own_error, loaded_error = (ref - exact).abs().max(), (got - exact).abs().max()
    assert loaded_error <= ROUNDING_ORDER_FACTOR * own_error, (
        f"loaded model is {loaded_error:.3e} from the exact model, qpeft's own forward {own_error:.3e}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("checkpoint_format", ["gptq", "gptq_v2"])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_loader_dequantizes_the_qpeft_weights_bit_for_bit(bits, checkpoint_format, dtype, tmp_path):
    model = _merged(bits, dtype)
    export_gptq(model, tmp_path, checkpoint_format=checkpoint_format)
    loaded = _load(tmp_path, bits, dtype, backend="torch")

    layers = {name: m for name, m in model.base.named_modules() if isinstance(m, QuantLinear)}
    loaded_modules = dict(loaded.named_modules())
    for name, layer in layers.items():
        weight = loaded_modules[name].dequantize_weight().t()                   # (out, in) fp16
        expected = layer.scheme.dequant(layer.codes, layer.scale.half(), layer.zero_point.half())
        assert torch.equal(weight, expected), name

    if dtype == torch.float16:                       # the torch kernel computes bf16 models in fp16
        _assert_as_close_as_qpeft_itself(model, loaded)


def test_a_subset_of_the_block_linears_loads(tmp_path):
    model = _merged(4, torch.float16, targets=["q_proj", "v_proj"])
    export_gptq(model, tmp_path)
    loaded = _load(tmp_path, 4, torch.float16, backend="torch")
    attention = loaded.model.layers[0].self_attn
    assert type(attention.k_proj) is torch.nn.Linear and hasattr(attention.q_proj, "qweight")
    _assert_as_close_as_qpeft_itself(model, loaded)


def test_the_default_cpu_kernel_on_a_bf16_model(tmp_path):
    model = _merged(4, torch.bfloat16)
    export_gptq(model, tmp_path)
    loaded = _load(tmp_path, 4, torch.bfloat16, backend="auto")
    assert type(loaded.model.layers[0].mlp.up_proj).__name__ == "TorchAtenLinear"
    # No argmax check: bf16 noise flips near-ties; qpeft's own bf16 forward picks another token
    # than the exact model at 1 of these 128 positions.
    _assert_as_close_as_qpeft_itself(model, loaded, same_argmax=False)
