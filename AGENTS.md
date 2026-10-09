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

## Design principle: deep modules (Ousterhout)

Small, simple interfaces that hide a lot of complexity: pull complexity down into the module, not up to its callers.
No shallow wrappers, no pass-through methods, no internals leaking across module boundaries.
Define errors out of existence where you can, but never at the cost of invariant 2: refuse, do not approximate.

## Naming convention (mirrors peft/torchtune)

`<Method>Config`, `<Method>Model(BaseQuantTuner)`, `QuantLinear`,
`merge` / `merge_and_unload` / `unmerge`, `dispatch_default` / `dispatch_torchao`,
`TrainableParams`.
Do not deviate from this without a reason.

## Commands

- `pip install -e ".[dev]"` - editable install (the torchao backend: `.[torchao]`)
- `python examples/quickstart.py` - construction path for both methods
- `pytest` - the test suite (`QPEFT_RUN_SLOW=1` also runs the tests that download a model)
- `tests/test_gptq_load.py` (a real GPTQ load) runs only with `gptqmodel` and `optimum` installed; install them in a venv **without** axolotl (axolotl pins `optimum==1.16.2`, which cannot import `optimum.gptq` on transformers 5)
- `QPEFT_RUN_SLOW=1 pytest tests/test_axolotl_e2e.py` - the axolotl plugin through the real `axolotl train` CLI (needs `.[axolotl]`; on Apple Silicon it trains on MPS)

## GPTQ export facts (verified in loader source, October 2026)

- GPTQ v1 (`checkpoint_format: gptq`) stores `z - 1`.
  gptqmodel undoes it with a whole-word add for 2/4/8 bits (a carry runs into the next field) and per field mod `2**bits` for 3 bits, so only `z >= 1` reads the same everywhere; `export_gptq` refuses `z == 0` in v1.
  `gptq_v2` stores `z` and is read by gptqmodel only.
- transformers 5 loads GPTQ only through optimum + gptqmodel; gptqmodel runs on CPU (macOS too).
  Its default 4-bit CPU kernel `TorchAtenLinear` rounds the scales to bf16; `GPTQConfig(backend="torch")` is the exact path.
- vLLM main accepts GPTQ only with `sym=True` at 4 or 8 bits, so qpeft's asymmetric exports are refused there.

## axolotl integration (`qpeft/integrations/axolotl/`)

Facts about axolotl (verified on main `d17227d`, 2026-10) that the plugin is built around:

- The `axolotl.plugins` entry point loads `QPeftPlugin` on every axolotl run, so every hook must do nothing unless `adapter: qpeft`.
  Keep `integrations/axolotl/__init__.py` and `args.py` free of axolotl imports.
- axolotl never calls a plugin's `create_optimizer`; the plugin sets `trainer.optimizer` in `post_trainer_create`, and HF's Trainer keeps an optimizer that is already set.
- axolotl calls `post_train` twice (before and after its own final save), so `post_train` must be idempotent.
- axolotl's final save and its Ctrl-C save call `model.save_pretrained(output_dir)`; transformers loads the resulting `model.safetensors` with only a warning and random weights.
  Until axolotl has a plugin save hook, the plugin replaces `save_pretrained` on the model instance: saves into `output_dir` write no transformers weights, Trainer checkpoints still do (resume needs them).
- axolotl drops INFO log lines of loggers whose name does not start with `axolotl`; log under `axolotl.integrations.qpeft`.
- Never log from code that axolotl's Ctrl-C handler calls: the handler can interrupt axolotl's stdout tee while it holds a non-reentrant lock, and a log call then deadlocks.
  Write such messages with `os.write(2, ...)`.
- Off-GPU axolotl trains an fp16 model with fp16 AMP; the loss scaler skips the first ~3 steps (inf gradients) while it settles.
  A run of 3 steps or fewer trains nothing, and `FirstStepMustMoveParams` stops it.
- On MPS set `HF_DEACTIVATE_ASYNC_LOAD=1`, or transformers can crash while loading weights (transformers #48029).
