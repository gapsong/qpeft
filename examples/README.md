# examples

Runnable, self-checking examples, the qpeft analog of peft's `examples/`.
Each script trains a tiny model and asserts its own success, so a clean exit means it worked.

```bash
pip install -e .                          # runtime is just torch
python examples/quickstart.py             # construction path, no training
python examples/train_efficient_qat.py    # QAT over a 2-bit substrate + merge
python examples/train_qa_lora.py          # adapter training + zero-point fold merge
python examples/hf_injection.py           # inject QuantLinear into a Hugging Face model

pip install -e ".[train]"                 # transformers + accelerate
python examples/qat_trainer.py            # EfficientQAT via QATTrainer on Qwen3-0.6B (GPU recommended)
```

## What runs today

All scripts run against the dependency-free reference backend (`qat_scheme="int_uniform"`, `backend="auto"`, pure PyTorch).

- **`train_efficient_qat.py`**: both phases. Block-AP trains weight, scale and zero-point over a 2-bit grid (STE fake_quant); the phase switch (`get_quant_model(model, e2e_qp_config)`) **freezes the integer codes**; E2E-QP trains only the scale. Then the layer-level merge check is green and `merge_and_unload()` returns an **integer artifact** whose output equals the trained output.
- **`qat_trainer.py`**: EfficientQAT the peft way. The plain HF model goes into `QATTrainer(...)`; `train()` runs Block-AP block by block -> frozen codes -> E2E-QP; `save_model()` merges and writes the int model; `QuantModel.from_pretrained` loads it back.
- **`train_qa_lora.py`**: a frozen quantized base plus a trainable adapter.
  The adapter is pooled per quantization group, folds **exactly** into the zero-points on merge, and stays int (the deliberate opposite of peft's dequantizing merge).
- **`hf_injection.py`**: loads a small transformers model and swaps its targeted `nn.Linear` layers for `QuantLinear` in place, then runs a forward pass.

The same contract also runs on the torchao backend (`backend="torchao"`), measured at the same gate; see `tests/test_torchao_backend.py`.
The spine test behind all of this is `tests/test_merge_equivalence.py` (`pytest`).

## What is still missing

- The `mlx` backend is not built yet; it refuses at model-build time (`backend="mlx"` raises) rather than approximating.
  It will implement the same `int_uniform` contract and be measured at the same gate.
- The **ternary** scheme is deliberately not in yet (see `docs/DESIGN.md`).
- Block-AP keeps all calibration activations in memory (the official code can offload to disk).
- Resuming a `QATTrainer` run from a mid-training checkpoint is not supported (`save_strategy="no"` by default).
