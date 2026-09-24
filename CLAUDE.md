# qpeft

Quantization-aware, PEFT-style tuning whose **merge stays quantized**.
Full problem statement & approach comparison: @docs/DESIGN.md
Overview & usage: @README.md
Open work: `TASKS.md` (local, not in git).

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
6. **No stub without its equivalence test in the same step.**

## Naming convention (mirrors peft/torchtune)

`<Method>Config`, `<Method>Model(BaseQuantTuner)`, `QuantLinear`,
`merge` / `merge_and_unload` / `unmerge`, `dispatch_default` / `dispatch_torchao`,
`TrainableParams`.
Do not deviate from this without a reason.

## Commands

- `pip install -e ".[dev]"` - editable install (the torchao backend: `.[torchao]`)
- `python examples/quickstart.py` - construction path for both methods
- `pytest` - the test suite (`QPEFT_RUN_SLOW=1` also runs the tests that download a model)
