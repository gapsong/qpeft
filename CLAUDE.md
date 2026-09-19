# qpeft

Quantization-aware, PEFT-style tuning whose **merge stays quantized**.
Full problem statement & approach comparison: @docs/DESIGN.md
Overview & usage: @README.md

## Invariants (do not violate)

1. **fake_quant (training) == merge/fuse (export).**
   `check_merge_equivalence` must be green before a scheme is trusted.
   A fake_quant that does not match the fuse is worse than none.
2. **Refuse instead of approximate.**
   An unsupported combination of (scheme, backend, config) raises `UnsupportedSchemeError` at model-build time, never silently approximating.
3. **merge stays int.**
   `merge(wq, s, z, adapter) -> (wq', s', z')`, never dequantizing.
   This is the deliberate inverse of peft's `merge_and_unload`.
4. **The trainable set is the first axis.**
   `{weight, scale, zero_point, adapter}`.
   Methods (EfficientQAT, QA-LoRA, PEQA, ...) are configs over it, not new subsystems.
5. **backend = implementation, qat_scheme = contract.**
   A new backend is a `QuantScheme` subclass that implements four primitives and narrows `supports()`.

## Naming convention (mirrors peft/torchtune)

`<Method>Config`, `<Method>Model(BaseQuantTuner)`, `QuantLinear`,
`merge` / `merge_and_unload` / `unmerge`, `dispatch_default` / `dispatch_torchao`,
`TrainableParams`.
Do not deviate from this without a reason.

## Structure

- `qpeft/config.py` `schemes.py` `schemes_torchao.py` `mapping.py` `peft_model.py` `utils.py`
- `qpeft/tuners/tuners_utils.py` (BaseQuantTuner, QuantLinear)
- `qpeft/tuners/{efficient_qat,qa_lora}/` (config, model, layer[, torchao])

## Commands

- `pip install -e ".[dev]"` - editable install (the torchao backend: `.[torchao]`)
- `python examples/quickstart.py` - construction path for both methods
- `pytest` - the test suite (merge equivalence, backends, injection, config)

## Status & current milestone

The `int_uniform` contract is implemented against TWO backends, and both are green at the same gate (`check_merge_equivalence`: EfficientQAT without an adapter is exact, the QA-LoRA fold is < 1e-4).
- `backend="auto"` -> `ReferenceIntUniformScheme` (pure torch, grouped affine + STE, RTN init).
  Always available.
- `backend="torchao"` -> `TorchaoIntUniformScheme` (`qpeft/schemes_torchao.py`, on torchao's STABLE `quant_primitives`, lazy import).
  Refuses a trainable `zero_point` (INT zero-point domain), which is `supports()` in action.

Tests: `tests/test_merge_equivalence.py` + `test_torchao_backend.py` (skips without torchao) + `test_custom_models/tuners_utils/initialization/config.py`.
`examples/train_*.py` and `examples/hf_injection.py` show training, integer merge, and HF injection.

**Next step (choose one):**
(a) `TorchaoQuantLinear` = adopt a layer that torchao ALREADY packed (case b), blocked by torchao's in-flux tensor-subclass API (int4 needs a kernel lib, int8 exposes `Int8Tensor.qdata`); needs a pinned torchao version + target hardware.
(b) A save/load layer (`PeftModel`-like).
(c) The `mlx` backend.
Do not implement a stub without delivering the equivalence test in the same step.

## Planned long-term (do NOT build now)

Ternary (1.58-bit, {-1, 0, +1}, group-wise FP16 scale, symmetric, no zero-point) is a planned future scheme, inside the substrate, so later a registry entry `qat_scheme="ternary"` plus an export layer (GGUF Q2_0_g128 / MLX 2-bit), not a redesign.
Do not add a stub or an `export()` hook for it now; dead placeholders only rot.

To keep the path open, do NOT bake in three assumptions: `bits` is not the sole source of truth about the representation (the `qat_scheme` string is the contract); `zero_point` is not present everywhere (ternary ignores it); `merge` always returns int, never fp16.
All three already hold; just do not violate them.
