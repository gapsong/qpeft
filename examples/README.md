# examples

Runnable, self-checking examples, the qpeft analog of peft's `examples/`.
Each script trains a tiny model and asserts its own success, so a clean exit means it worked.

```bash
pip install -e .                          # runtime is just torch
python examples/quickstart.py             # construction path, no training
python examples/train_efficient_qat.py    # QAT over a 2-bit substrate + merge
python examples/train_qa_lora.py          # adapter training + zero-point fold merge
python examples/hf_injection.py           # inject QuantLinear into a Hugging Face model
```

## What runs today

All scripts run against the dependency-free reference backend (`qat_scheme="int_uniform"`, `backend="auto"`, pure PyTorch).

- **`train_efficient_qat.py`**: quantization-aware training over a 2-bit grid (STE fake_quant), then `check_merge_equivalence` is green per `QuantLinear`, then `merge_and_unload()` returns an **integer artifact** whose output equals the trained output exactly.
- **`train_qa_lora.py`**: a frozen quantized base plus a trainable adapter.
  The adapter is pooled per quantization group, folds **exactly** into the zero-points on merge, and stays int (the deliberate opposite of peft's dequantizing merge).
- **`hf_injection.py`**: loads a small transformers model and swaps its targeted `nn.Linear` layers for `QuantLinear` in place, then runs a forward pass.

The same contract also runs on the torchao backend (`backend="torchao"`), measured at the same gate; see `tests/test_torchao_backend.py`.
The spine test behind all of this is `tests/test_merge_equivalence.py` (`pytest`).

## What is still missing

- The `mlx` backend is not built yet; it refuses at model-build time (`backend="mlx"` raises) rather than approximating.
  It will implement the same `int_uniform` contract and be measured at the same gate.
- The **ternary** scheme is deliberately not in yet (see `docs/DESIGN.md`).
- EfficientQAT's phase transition (Block-AP to E2E-QP with frozen int codes) is simplified in the skeleton; the example shows the Block-AP phase.
