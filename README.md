<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/qpeft-logo-dark.svg">
    <img alt="qpeft" src="docs/assets/qpeft-logo-light.svg" width="520">
  </picture>
</h1>

<h3 align="center">Quantization-aware, PEFT-style tuning whose merge stays quantized.</h3>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.10-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%E2%89%A52.4-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="torchao" src="https://img.shields.io/badge/backend-torch%20%7C%20torchao-6366f1">
  <img alt="merge" src="https://img.shields.io/badge/merge-stays%20int-ec4899">
  <img alt="status" src="https://img.shields.io/badge/status-alpha-lightgrey">
</p>

<p align="center">
  <a href="#basic-usage">Quickstart</a> ·
  <a href="docs/DESIGN.md">Design</a> ·
  <a href="#structure-mirrors-peft">Structure</a> ·
  <a href="#status">Status</a>
</p>

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

## What it deliberately is NOT

A coherent slice, not a do-everything wrapper: **weight-only, grouped, uniform-int, decoder LLMs.**
Explicitly out of scope: codebook / vector quant (AQLM, QuIP#) and SBC-style stochastic binary codecs.
Those are a different substrate and a different inference operator; they belong in a sibling project, not here.

## Structure (mirrors peft)

```
qpeft/
  config.py            # QuantTuningType, TrainableParams, QuantTuningConfig   (~ PeftType / PeftConfig)
  schemes.py           # QuantScheme contract, ReferenceIntUniformScheme, registry
  schemes_torchao.py   # TorchaoIntUniformScheme: same contract on torchao primitives (optional)
  mapping.py           # get_quant_model + registries                         (~ get_peft_model)
  peft_model.py        # QuantModel.merge_and_unload()                        (~ PeftModel)
  utils.py             # check_merge_equivalence                              (the spine test)
  tuners/
    tuners_utils.py    # BaseQuantTuner, AdapterLayer, QuantLinear            (~ BaseTuner / lora.Linear)
    efficient_qat/{config,model,layer}.py
    qa_lora/{config,model,layer,torchao}.py
examples/              # quickstart, train_efficient_qat, train_qa_lora, hf_injection
tests/                 # merge-equivalence, backends, injection, config
pyproject.toml
```

## Design and scope

For why the repo exists, the three comparison approaches, and where each interface decision lives in the code, see [`docs/DESIGN.md`](docs/DESIGN.md).

## Basic usage

```bash
pip install -e .            # runtime is just torch; the torchao backend: pip install -e ".[torchao]"
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

The `int_uniform` contract is implemented against two backends, and both are green at the same gate (`check_merge_equivalence`: EfficientQAT without an adapter is exact, the QA-LoRA fold is < 1e-4).
`backend="auto"` selects the pure-torch `ReferenceIntUniformScheme`, which is always available.
`backend="torchao"` selects `TorchaoIntUniformScheme`, built on torchao's stable `quant_primitives` and imported lazily so torchao stays optional.
`examples/train_*.py` show real training plus an integer merge, and `examples/hf_injection.py` shows injection into a Hugging Face model.
The test suite (`pytest`) covers merge equivalence, both backends, injection, initialization and config validation.

Still stubbed: `TorchaoQuantLinear` (adopting a tensor that torchao itself already packed, blocked by torchao's in-flux tensor-subclass API), the `mlx` backend, and the planned ternary scheme.
