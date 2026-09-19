# Design and scope

Why qpeft exists, which concrete failure modes it is built against, and where each interface decision lives in the code.
This is the "why"; the README is the "what".

## The problem: one root, three symptoms

**Root.**
The `fake_quant` used during training and the `fuse`/`save` used at export are separate code paths.
If they drift even slightly, you train a model that does not exist at deployment.

Three existing approaches show three reactions to the same root.

- **torchao + torchtune** compose QAT with LoRA and reach a quantized model with minimal quality loss, but the capability is stuck inside a **CUDA and torchtune recipe**.
  It is not an extractable primitive, it is a framework flow.
- **unsloth-zoo (the MLX PR)** runs straight into the root: `merged_4bit` does not leave the quantized base clean, so `fuse != fake_quant`.
  The only safe reaction the author has is to **refuse** the affected schemes: "a fake-quant that does not match fuse() is worse than none".
  Correct, but ad hoc and per backend.
- **peft** sidesteps it: `merge_and_unload()` **dequantizes** to fp16.
  The problem is avoided, but the quantized artifact is lost at merge time.

**Conclusion.**
Nobody owns the pair `(fake_quant, fuse)` as a *contract*.
So it is re-fought per backend, and the general case (an adapter that folds into the quantized artifact and stays quantized) falls through the cracks.

## The three approaches compared

| | trainable | merge artifact | fake_quant <-> fuse | backend / framework | extractable primitive? |
|---|---|---|---|---|---|
| **peft (QLoRA)** | adapter | **dequantized (fp16)** | n/a (no QAT merge) | delegated to bnb/torchao | no |
| **torchtune + torchao** | base fake-quant + LoRA | quantized | matched internally | **CUDA + torchtune recipe** | **no (framework-coupled)** |
| **unsloth-zoo (MLX)** | LoRA over QuantizedLinear | `merged_4bit` | **refused when !=** | **MLX / Apple Silicon** | no (backend-specific) |
| **qpeft** | `{weight, scale, zero_point, adapter}` selectable | **stays int (by signature)** | **enforced + tested** | backend behind the scheme | **yes (contract-first)** |

## What qpeft does differently, and where it lives in the interface

### 1. (fake_quant, fuse) is ONE object
`fake_quant()` and `merge()` are methods on the same `QuantScheme`.
You cannot define the training quantization without placing the matching export fuse right next to it.
That prevents drift structurally instead of documenting against it.
-> `qpeft/schemes.py::QuantScheme`

### 2. Equivalence is a test gate
The insight "a mismatch is worse than none" becomes a testable invariant: the training path (`fake_quant`) must numerically equal the merged path (`merge` -> `dequant`).
That is the milestone at which a scheme becomes "valid".
-> `qpeft/utils.py::check_merge_equivalence`

### 3. Refuse instead of approximate, as a first-class mechanism
What the MLX author did by hand per scheme is here an interface contract: a scheme declares via `supports(config)` which configs its fuse is guaranteed to match its fake_quant, and `assert_supported` **refuses** loudly at model-build time when it cannot.
No silent approximation.
-> `QuantScheme.supports` / `assert_supported` / `UnsupportedSchemeError`, called in `build_scheme`

### 4. Backend is implementation, not contract
`qat_scheme=` names the *contract* (the representation), `backend=` names the *implementation* (`auto`/`torch`, `torchao`, `mlx`).
A backend names a *provider* (who implements the primitives), never a device: the device is orthogonal and follows the model's tensors.
So torch, torchao and MLX are implementations of the same `QuantScheme` interface, all secured by the same equivalence gate.
That is the decoupling torchtune lacks.
-> `QuantTuningConfig.backend` + `build_scheme` (registry per contract, backend chosen in the factory)

### 5. Merge stays quantized, by signature
`merge(wq, s, z, adapter) -> (wq', s', z')`: int in, int out.
Unlike peft's dequantizing merge, the quantized artifact is the result, not an intermediate step that is thrown away.
-> `QuantScheme.merge` / `QuantModel.merge_and_unload`

## What follows from this

A new method (EfficientQAT, QA-LoRA, PEQA, L4Q, ...) is a config over the axes plus at most one new operation (the zero-point fold).
A new backend is a `QuantScheme` subclass that implements four primitives and narrows `supports()`, and it is automatically measured at the same gate.
The general case that used to fall through the cracks is thereby the normal case, not the exception.

## Planned for later: ternary (not in yet)

Ternary (1.58-bit, {-1, 0, +1} with a group-wise FP16 scale, symmetric, no zero-point) is a *planned future scheme*, not to be confused with the items in the "What it deliberately is NOT" section of the README.
Those (codebook / vector quant, SBC) break the substrate; ternary sits *inside* it: grouped, uniform, weight-only.
So it will later be a registry entry (`qat_scheme="ternary"`) plus an export layer into a served format (GGUF Q2_0_g128, MLX 2-bit), not a redesign.
This representation already ships in the wild: PrismML's Bonsai is exactly this, including "packed weights, never expanded to FP16", which maps one-to-one onto the `merge` contract.

Currently out of scope.
To keep the path open without writing a single line of ternary-specific code, three *omitted* mistakes suffice: (1) do not make `bits` the sole source of truth about the representation (the `qat_scheme` string is the contract, so do not scatter `if bits == 4` logic); (2) do not assume `zero_point` is present everywhere (ternary is symmetric and ignores it, which the trainable-set switch and `supports()` already allow); (3) `merge` stays "int in, int out", never fp16.
All three are already built this way; they just must not be violated.
