# Spec: QATTrainer

Status: **verified** 2026-09-24.
Open decisions were resolved with the proposed defaults -- confirm or change them.
Autonomy stage: **S1-S2** for the phase logic, the refusal gates and the final
equivalence check (a silent error ships a model that differs from the trained
one); S3 is fine for logging / argument plumbing.

## Goal

EfficientQAT with the ease of use of peft + Trainer. The user hands over the
**plain Hugging Face model**; the result is a real integer model.

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
trainer = QATTrainer(
    model=model,
    quant_config=EfficientQATConfig(bits=2, group_size=64),   # optional, default = paper (4 bit, g128)
    train_dataset=dataset,
    args=QATTrainingArguments(output_dir="qwen3-0.6b-w2g64"),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()        # Block-AP -> codes frozen -> E2E-QP, merge check at the end
trainer.save_model()   # merge -> int artifact
```

Full script: `examples/qat_trainer.py`.

## Scope: EfficientQAT only

| Input | Expected |
|---|---|
| plain `nn.Module` / HF model (+ optional `EfficientQATConfig`) | wrapped via `get_quant_model`; `target_modules=None` -> every `nn.Linear` inside the transformer blocks (not `lm_head`, as in the paper) |
| `QuantModel` from `get_quant_model(..., EfficientQATConfig)` | accepted as is (`quant_config` must then be `None`) |
| any other config (`QALoraConfig`, ...), as `quant_config` or inside a `QuantModel` | `UnsupportedSchemeError` |
| targets match nothing | `ValueError` |
| already merged `QuantModel` | `ValueError` |
| `save_strategy != "no"` | `ValueError` (a mid-training checkpoint is not the int artifact) |

## Behavior

`QATTrainer(Trainer)` — subclass of `transformers.Trainer` (pinned to 5.7.x: it uses Trainer internals), so logging, bf16,
gradient accumulation, checkpoints and multi-GPU come from HF.

1. **Phases.** `phase="block_ap"` (default): Block-AP loop -> phase switch -> E2E-QP via
   `super().train()`. `phase="e2e_qp"`: only E2E-QP.
2. **Phase switch (EfficientQAT).** Freeze the integer codes (see
   `tests/test_efficient_qat_phases.py::test_e2e_qp_codes_are_frozen_when_scale_moves`),
   then only `scale` is trainable. Block-AP results are carried over, never re-initialized.
3. **Optimizer.** One param group per parameter kind, from `QATTrainingArguments`:
   `weight_lr`, `quant_lr` (scale, zero_point), `e2e_lr` in E2E-QP. The HF
   `learning_rate` is not used for qpeft parameters (documented, not silent).
4. **Guards during training.**
   - trainable set == the set the phase allows (checked at every phase start)
   - loss is finite and not exactly 0.0 (peft PR #2571 lesson)
   - after the first optimizer step at least one trainable parameter moved
5. **End of `train()`.** Run `check_merge_equivalence` on every `QuantLinear`.
   Red -> raise, never hand back a model.
6. **Saving.** `trainer.save_model()` merges (irreversible) and calls
   `QuantModel.save_pretrained` (see `docs/specs/save_load.md`). Load with
   `QuantModel.from_pretrained(base_model, output_dir)`.

## QATTrainingArguments (defaults = official EfficientQAT code)

Subclass of `transformers.TrainingArguments`.

| Field | Default | Source |
|---|---|---|
| `quant_lr` | 1e-4 | main_block_ap.py |
| `weight_lr` | 1e-5 (2e-5 if bits == 2) | main_block_ap.py / w2g64.sh |
| `e2e_lr` | 1e-5 (2e-5 if bits == 2) | e2e_qp scripts |
| `block_ap_epochs` | 2 | main_block_ap.py |
| `block_ap_train_size` | 4096 | main_block_ap.py |
| `block_ap_seqlen` | 2048 | main_block_ap.py |
| `block_ap_batch_size` | 2 | main_block_ap.py `--batch_size` |
| `block_ap_min_lr_factor` | 20 (cosine to lr/20) | main_block_ap.py |
| `per_device_train_batch_size` (E2E-QP) | 4 | e2e_qp scripts |
| `gradient_accumulation_steps` (E2E-QP) | 8 | e2e_qp scripts |
| `lr_scheduler_type` (E2E-QP) | cosine | main_e2e_qp.py |
| `warmup_steps` (E2E-QP) | 0.03 (a fraction: warmup_ratio=0.03) | main_e2e_qp.py |
| `max_grad_norm` (E2E-QP) | 0.3 | e2e_qp scripts |

Precision follows the official code: Block-AP trains each block in fp32 and casts
it back (`qlayer.float()` ... `qlayer.half()`), and E2E-QP trains fp32 scales
(main_e2e_qp.py casts all non-int parameters to fp32). In qpeft, scale and
zero_point are always fp32 masters; the forward and the artifact use the base
dtype, so `fake_quant == merge` stays exact in bf16/fp16. Block-AP also trains
the block norms (see the README, "EfficientQAT compared to the official implementation").

The warmup starts at lr 0, so the "something moved" guard checks the first
optimizer step with lr > 0, and fails at the end if there was none.

Bit-dependent defaults are resolved from the model's config at `__init__`
and logged, so the user sees what actually runs.

## Prerequisites (build these first, each with its own tests)

1. Frozen codes at the phase switch - done, verified 2026-09-24.
2. `save_pretrained` / `from_pretrained` - done, verified 2026-09-24 (`docs/specs/save_load.md`).
3. Param-group helper (`param_groups`) - done, verified 2026-09-24 (`tests/test_training.py`).
4. Block-AP loop (`run_block_ap`) - done, verified 2026-09-24 (`tests/test_block_ap.py`).
5. Only then: `QATTrainer` as a thin layer on top.

## Out of scope

Loading an already-quantized model (GPTQ/AWQ/torchao-packed, CLAUDE.md step (a)),
multi-adapter, DPO/RLHF trainers, MLX backend, ternary.

## Open decisions (owner decides, not the agent)

- [x] Block-AP calibration data: **reuses `train_dataset`** (first `block_ap_train_size`
      samples, truncated to `block_ap_seqlen`). A separate `calib_dataset` is not built.
- [x] `train()` does **not** merge; `trainer.save_model()` does (merge + save).
      `train()` runs the layer-level merge check at the end.
- [x] EfficientQAT only; QA-LoRA keeps working via get_quant_model + your own loop / HF Trainer.
- [x] The trainer takes the plain HF model; `save_model()` merges and saves.
- [x] Error types: `TypeError` / `ValueError` as proposed.

## Acceptance criteria / tests to write first

- [x] Each row of the refusal table has a test (no compute happens before the error)
- [x] EfficientQAT: after `train()`, Block-AP changed {weight, scale, zero_point}
      (the weight via the frozen codes, the fp weight is dropped at the hand-over),
      E2E-QP changed only {scale}, codes frozen in E2E-QP
- [x] QA-LoRA: out of scope for QATTrainer (refused); covered by `get_quant_model` tests
- [x] Optimizer has the right param groups and learning rates per phase
- [x] Loss == 0.0 or no parameter movement -> training aborts with a clear error
- [x] After `train()`, `check_merge_equivalence` is green on every layer
- [x] `save_model()` produces the files from `save_load.md`, loadable and bit-exact
      (bit-exact in `tests/test_save_load.py`; the trainer test compares a GPU/MPS run
      with a CPU load, so it checks within 1e-3 and the same argmax)
- [x] Tiny random Llama end to end (no download) in the default suite;
      real model behind `QPEFT_RUN_SLOW=1`
