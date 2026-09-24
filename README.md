# qpeft

**Quantization-aware, PEFT-style tuning whose merge stays quantized.**

`qpeft` trains over a quantized substrate.
Unlike `peft`, its `merge_and_unload()` returns an *integer* model instead of dequantizing back to fp16.
It leans on torchao for low-level primitives (affine quant, packed dtypes, kernels) and owns the one seam neither `peft` nor `unsloth` owns: **a QAT-trained adapter/parameter set that folds into the quantized weights and stays quantized, with a correctness guarantee.**

---

## Why this repo exists

The quantized-training space has a gap that nobody yet owns as *one* feature.

- **peft** freezes the base and trains an adapter.
  Its `merge_and_unload()` **dequantizes** the model, so the quantized artifact is lost at merge time.
  Quantization itself is delegated to bitsandbytes/torchao, and the "merge stays quantized" case is only half covered (torchao merges cleanly only for `int8_weight_only` + LoRA; AQLM/AWQ cannot merge at all).
- **unsloth** is tuned for fast, memory-light QLoRA/LoRA, but on the same frozen-base-plus-separate-adapter model.
  Real QAT there is backend-specific and in progress (the MLX QAT PR), not a backend-agnostic contract.

`qpeft` closes exactly that seam: training over a quantized substrate whose merge stays quantized, as a first-class contract with a correctness test.

## The core principle

Two ideas carry everything.

1. **The trainable set is the first axis.**
   Not "adapter vs. frozen", but a choice from `{weight, scale, zero_point, adapter}` (`TrainableParams`).
   This makes PEQA (`scale` only), EfficientQAT (`weight + scale + zero_point`, then `scale`) and QA-LoRA (`adapter` folds into `zero_point`) **configurations over one substrate**, not separate subsystems.
2. **`fake_quant` must match `merge`.**
   The STE surrogate used in training and the exact fold used at export must agree numerically.
   A fake-quant that does not match the fuse is worse than none.
   `check_merge_equivalence` turns that into a testable invariant, and it is the spine of the library.

## What it does beyond peft

- **Train quantization parameters.**
  `peft` cannot even express "train the quant scale"; here it is `trainable_params=(SCALE,)`.
- **Merge stays int.**
  `QuantModel.merge_and_unload()` returns a quantized model, not fp16, which is the deliberate opposite of `peft`.
- **The method zoo is just configs.**
  EfficientQAT, QA-LoRA, PEQA, L4Q, LoftQ-init and so on are each a point on the axes.
  A new paper is a config plus at most one operation, not a new integration.
- **Correctness as an invariant.**
  `check_merge_equivalence` has no equivalent in `peft`.

## What it does beyond unsloth

- A **backend-agnostic contract** instead of a single backend PR.
- **QAT + adapter as one parameterized recipe**, the combination torchtune/torchao offer only partially and only per backend.
- Not just speed on the frozen-base model, but the quantized training artifact itself as the target.

## Relation to torchao

`qpeft` does not replace torchao; it composes with it.
It ships a dependency-free pure-torch reference backend so the contract is real and testable without any extra install, and a torchao backend (`backend="torchao"`) that implements the *same* contract on torchao's stable affine primitives (`quant_primitives`).
Both backends are measured at the same equivalence gate.
`qpeft` owns only the seam torchao leaves open: folding the adapter into the `zero_point`, plus the multi-phase QAT schedule, with a test that proves both paths agree.

torchao is a backend, not the foundation.
Its affine primitives take the zero-point as an `int32` tensor, so no gradient can reach it, and the torchao backend refuses a trainable `zero_point`.
EfficientQAT's Block-AP trains the zero-point, so it needs the reference backend.
The reference backend is also the proof that the contract holds independently of torchao.

## What it deliberately is NOT

A coherent slice, not a do-everything wrapper: **weight-only, grouped, uniform-int, decoder LLMs.**
Explicitly out of scope: codebook / vector quant (AQLM, QuIP#) and SBC-style stochastic binary codecs.
Those are a different substrate and a different inference operator; they belong in a sibling project, not here.

## The `int_uniform` representation

Weight-only, group-wise, asymmetric affine quantization.
Per group of `group_size` input columns there is one scale `s` and one zero-point `z`.

```
code  = clamp(round(w / s + z), 0, 2**bits - 1)     # the integer codes (int32)
w_hat = (code - z) * s                              # dequant
```

- **Scale:** clamped to `[1e-4, 1e4]`, as in the official EfficientQAT quantizer.
  A trainable scale can otherwise step to zero or below.
- **Zero-point:** an integer in `[0, 2**bits - 1]` (`round_zero_point`, the official `clamp_ste(round_ste(z))`).
  The trainable parameter may drift between integers, but training, the frozen codes and the merged artifact all use the rounded value.
  So the artifact has the same shape as a GPTQ-style checkpoint (int codes, scale, int zero-point).
- **QA-LoRA:** the adapter average-pools its input per group (as the official `nn.AvgPool1d(group_size)`), so its weight delta is constant inside a group.
  `merge` folds that delta into the zero-point: `z' = z - delta / s`.
  Codes and scale do not change; only the merged zero-point becomes fractional.

Both backends implement this same representation, and both are measured at the same gate.

## Precision (bf16 / fp16 models)

An AdamW step at a typical QAT learning rate (1e-5 to 2e-5) is below bf16's resolution: in a bf16 tensor it changes nothing.
So `qpeft` keeps fp32 master copies of everything it trains, like the official EfficientQAT code:

- `scale`, `zero_point` and the QA-LoRA adapter are always fp32.
- Block-AP casts each block to fp32 for training and back to the model's dtype afterwards.
- The forward computes in the input's dtype on parameters cast to it, and the codes and the merged artifact are made in the model's dtype (`QuantLinear.compute_dtype`).

So a bf16 training forward and its merged bf16 artifact see the same numbers, and the merge check stays exact for EfficientQAT.

## EfficientQAT compared to the official implementation

Reference: [OpenGVLab/EfficientQAT](https://github.com/OpenGVLab/EfficientQAT).

The same as the official code:

- Min/max (RTN) init, asymmetric, integer zero-point, scale clamped to `[1e-4, 1e4]`.
- Block-AP: block by block, MSE against the fp block output; the input comes from the quantized chain, the target from the fp chain.
- Block-AP trains weight (`weight_lr`), scale and zero-point (`quant_lr`) and the block norms (`*.weight`, the official name filter), with AdamW, `wd=0`, 2 epochs, cosine down to `lr / 20`, batch size 2, in fp32.
- E2E-QP: codes frozen, only the scale trains; batch 4 x gradient accumulation 8, cosine with 3 % warmup, `max_grad_norm=0.3`.
- Learning rates: `quant_lr=1e-4`, `weight_lr` and the E2E-QP lr `1e-5` (`2e-5` at 2 bits).

`QATTrainingArguments` carries these values as defaults.

Deliberate differences:

- Any Hugging Face decoder and any device (the official code supports the Llama family on CUDA).
- fp32 without autocast (official: autocast + GradScaler, CUDA only).
- Block-AP calibrates on `train_dataset` (official: its own RedPajama sample) and keeps all activations in memory (official can offload to disk).
- No validation split; the Block-AP log reports the mean MSE over all calibration batches before and after each block.
- The hand-over to E2E-QP freezes the integer codes directly and checks `fake_quant == merge` at the end.
  The official code casts to fp16 after training (`qlayer.half()`) and packs without such a check.

## Structure (mirrors peft)

```
qpeft/
  config.py            # QuantTuningType, TrainableParams, QuantTuningConfig   (~ PeftType / PeftConfig)
                       #   to_dict / from_dict / save_pretrained / from_pretrained
  quant_schemes/
    base.py            # QuantScheme contract, shared int_uniform machinery
    reference.py       # ReferenceIntUniformScheme: pure torch, always available
    torchao.py         # TorchaoIntUniformScheme: same contract on torchao primitives (optional)
    registry.py        # qat_scheme -> scheme, build_scheme
  mapping.py           # get_quant_model + registries                         (~ get_peft_model)
  peft_model.py        # QuantModel.merge_and_unload()                        (~ PeftModel)
  utils.py             # check_merge_equivalence, verify_quant_model          (the spine test)
  training.py          # param_groups: one optimizer group per parameter kind
  block_ap.py          # run_block_ap: EfficientQAT phase 1
  trainer.py           # QATTrainer, QATTrainingArguments                     (needs .[train])
  tuners/
    tuners_utils.py    # BaseQuantTuner, AdapterLayer, QuantLinear            (~ BaseTuner / lora.Linear)
    efficient_qat/{config,model,layer}.py
    qa_lora/{config,model,layer,torchao}.py
examples/              # quickstart, train_efficient_qat, train_qa_lora, hf_injection, qat_trainer
tests/                 # merge equivalence, backends, injection, config, save/load, phases, precision, trainer
pyproject.toml
```

## Design and scope

For why the repo exists, the three comparison approaches, and where each interface decision lives in the code, see [`docs/DESIGN.md`](docs/DESIGN.md).

## Basic usage

```bash
pip install -e .                 # runtime is just torch
pip install -e ".[torchao]"      # + the torchao backend
pip install -e ".[train]"        # + QATTrainer (transformers 5.7.x)
pip install -e ".[dev]"          # + tests and examples
python examples/quickstart.py
```

```python
import torch.nn as nn
from qpeft import get_quant_model, QALoraConfig, efficient_qat_schedule

base = nn.Sequential(nn.Linear(512, 512))

# EfficientQAT: two phases (Block-AP then E2E-QP), one config each.
for cfg in efficient_qat_schedule(bits=2, group_size=64):
    model = get_quant_model(base, cfg)
    # ... train this phase ...

# QA-LoRA: one config; the adapter folds into the zero-points on merge.
model = get_quant_model(base, QALoraConfig(bits=4, group_size=32, r=64))
# ... train ...
quantized = model.merge_and_unload()          # stays integer

# Same contract on the torchao backend:
model = get_quant_model(base, QALoraConfig(bits=4, group_size=64, r=16, backend="torchao"))
```

EfficientQAT with the `QATTrainer` (needs `pip install -e ".[train]"`), same shape as peft + `Trainer`,
but you hand over the plain Hugging Face model and get an integer model back:

```python
from qpeft import EfficientQATConfig, QATTrainer, QATTrainingArguments

trainer = QATTrainer(model=model,                                     # plain HF model
                     quant_config=EfficientQATConfig(bits=2, group_size=64),
                     args=QATTrainingArguments(output_dir="out"),     # paper defaults
                     train_dataset=dataset, data_collator=collator)
trainer.train()          # Block-AP -> codes frozen -> E2E-QP; merge check at the end
trainer.save_model()     # int artifact; load with QuantModel.from_pretrained(base, "out")
```

`QATTrainer` supports EfficientQAT only (see `examples/qat_trainer.py`).
QA-LoRA runs with `get_quant_model` and your own loop or a plain HF `Trainer`; `param_groups` builds the optimizer groups.

Save and load: only a merged model can be saved, as the integer artifact.

```python
model.merge_and_unload()
model.save_pretrained("out")                        # qpeft_config.json + qpeft_model.pt
loaded = QuantModel.from_pretrained(base, "out")    # base supplies the architecture

cfg = QuantTuningConfig.from_pretrained("out")      # the right config class, or a clear refusal
```

Run the spine test per layer before trusting a scheme:

```python
from qpeft import check_merge_equivalence
check_merge_equivalence(scheme, w, s, z, adapter, x)   # fake_quant (train) == merge (export)
```

Load a Hugging Face model and inject `QuantLinear` into it (`examples/hf_injection.py`):

```python
from transformers import AutoModelForCausalLM
from qpeft import get_quant_model, EfficientQATConfig

hf = AutoModelForCausalLM.from_pretrained("...")
model = get_quant_model(hf, EfficientQATConfig(
    bits=4, group_size=64, phase="e2e_qp",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
```

## Status

The `int_uniform` contract is implemented against two backends, and both are green at the same gate (`check_merge_equivalence`: EfficientQAT is exact, also in bf16; the QA-LoRA fold is < 1e-4 in fp32 and within 1e-2 relative in half precision, where the fold itself is computed in half).
`backend="auto"` selects the pure-torch `ReferenceIntUniformScheme`, which is always available.
`backend="torchao"` selects `TorchaoIntUniformScheme`, built on torchao's stable `quant_primitives` and imported lazily so torchao stays optional.
`examples/train_*.py` show real training plus an integer merge, `examples/hf_injection.py` shows injection into a Hugging Face model, and `examples/qat_trainer.py` runs both EfficientQAT phases on Qwen3-0.6B.
The test suite (`pytest`, plus `QPEFT_RUN_SLOW=1` for the tests that download a model) covers merge equivalence, both backends, injection, initialization, config, save/load, the EfficientQAT phases, half precision and the trainer.

Not measured yet: model quality.
There is no perplexity benchmark against RTN or the official EfficientQAT numbers yet, so green tests prove correctness, not quality.

Not supported:

- Loading an already-quantized checkpoint (GPTQ, AWQ, torchao-packed).
  `TorchaoQuantLinear` is a stub, blocked by torchao's in-flux tensor-subclass API.
- Exporting to GPTQ or other serving formats.
  The artifact has the right shape for it (int codes, scale, integer zero-point), but no exporter is built.
- The `mlx` backend and the planned ternary scheme.
