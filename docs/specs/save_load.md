# Spec: save / load a merged QuantModel

Status: **verified** 2026-09-24 (proposed defaults taken: refuse unmerged save, torch.save + weights_only load, ValueError).
Autonomy stage: **S1-S2** — a silent error here ships a model that computes
differently from the one that was trained, which is exactly the failure qpeft
exists to prevent. Every line of the implementation gets a human review.

## Goal

Persist a merged `QuantModel` and load it back so that it computes
**bit-identically** and **stays integer** — on disk as well as in memory.
This is next step (b) from `CLAUDE.md`.

## Behavior

- `model.save_pretrained(save_directory)` on a `QuantModel` whose layers are merged
  (i.e. after `merge_and_unload()`) writes:
  - `qpeft_config.json` — the full `QuantTuningConfig` (incl. `quant_tuning_type`,
    `qat_scheme`, `backend`, `bits`, `group_size`, method-specific fields).
  - the tensors of every `QuantLinear`: `qweight` (int32), `scale`, `zero_point`,
    `bias` — plus all non-quantized parameters of the base model.
- `QuantModel.from_pretrained(base_model, save_directory)` returns a `QuantModel`
  in the merged state. `base_model` supplies the architecture (same pattern as
  `peft.PeftModel.from_pretrained(base_model, ...)`).
- The scheme is rebuilt through `build_scheme(config)`, so the existing refusal
  path (`UnsupportedSchemeError`) applies to loaded configs too.
- After loading, nothing is trainable (`requires_grad=False` everywhere).

## Required change to merge (prerequisite)

After `QuantLinear.merge()` the float `weight` Parameter must be gone.
Today it survives the merge, so the state_dict carries the full-precision weight
next to `qweight` — which violates invariant 3 ("merge stays int") the moment it
is written to disk.

## Edge cases

| Case | Expected |
|---|---|
| Save a model that is not merged | refuse with `ValueError` ("call merge_and_unload() first") |
| Saved config names an unknown `backend` | `UnsupportedSchemeError` at load |
| Saved config names an unknown `qat_scheme` | `UnsupportedSchemeError` at load |
| Saved with `backend="torchao"`, torchao not installed | the existing `NotImplementedError` from `build_scheme` |
| `base_model` has different shapes than the saved one | error (strict load), never a partial load |
| fp16 base model | `scale` / `zero_point` / `bias` stay fp16, `qweight` stays int32 |
| `Linear(bias=True)` | bias round-trips unchanged |

## Out of scope

HF Hub upload, safetensors-only format, GGUF / MLX export, ternary,
`TorchaoQuantLinear`, saving an *unmerged* (still-training) model.

## Decisions (owner confirmed the proposed defaults, 2026-09-24)

- [x] **Unmerged save:** refuse (proposed, tested below) or save adapter + qparams
      to resume training later?
- [x] **File format:** `torch.save` + `torch.load(weights_only=True)` (proposed,
      no new dependency) or `safetensors` as an optional extra?
- [x] **Error type for unmerged save:** `ValueError` (proposed) or `UnsupportedSchemeError`?

If you change one of these, change the matching test in `tests/test_save_load.py` first.

## Acceptance criteria

- [x] Output before save == output after load, **bit-exact** (`torch.equal`, not `allclose`)
- [x] Loaded `qweight` is `int32`, adapter is `None`, layer reports `merged`
- [x] No float `weight` on any merged `QuantLinear` (in memory and after load)
- [x] Every edge case above has a test
- [x] Existing suite stays green; invariants in `CLAUDE.md` untouched
- [x] Test-the-test: drop `zero_point` from the saved state on purpose →
      the round-trip test must go red

## Instructions for the implementing agent

Implement this spec. `tests/test_save_load.py` is given — do **not** modify,
weaken, skip, or delete any test in it. Keep all invariants in `CLAUDE.md`.
Done means: `pytest` fully green. If a decision is not covered by this spec,
stop and ask instead of choosing.
