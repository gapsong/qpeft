# PEQA against the official code

There is no official PEQA code.
The closest real implementation is the official EfficientQAT code ([OpenGVLab/EfficientQAT](https://github.com/OpenGVLab/EfficientQAT)) with 0 Block-AP epochs.
It does exactly what PEQA does: min/max RTN init, pack into the real int `QuantLinear` (Triton kernels), then E2E-QP trains only the scales.
These scripts run qpeft's `PEQAConfig` against it, unchanged, on CUDA.

Run on one GPU with two conda envs (the official one: torch 2.2, transformers 4.40, triton; and one with qpeft):

```bash
EQAT_DIR=~/Documents/EfficientQAT bash benchmarks/peqa_vs_official/run.sh
```

The official env's transformers 4.40 does not know Qwen3, so the model is SmolLM2-360M (Llama architecture).
Group size is 64 everywhere.

## 1. Training step by step (`official_peqa.py`, `qpeft_peqa.py`)

The official side dumps its codes, zero-points, scales, losses and logits; qpeft then trains with the same input, optimizer (AdamW, wd 0) and steps.
The official `pack()` rounds the start scale to fp16, so qpeft starts from that exact scale.

| 20 AdamW steps | 4 bits | 2 bits |
|---|---|---|
| codes differing (of 314M) | 0 | 0 |
| zero-points differing | 0 | 0 |
| loss max diff per step | 1.4e-6 | 1.9 |
| codes changed by training | no | no |

At 2 bits the model is so fragile that training is chaotic: `chaos_check.py` trains qpeft twice, the second time with the scales changed by a relative 1e-7, and the losses already differ by 0.57 per step.

## 2. End to end with merge (`qpeft_peqa_e2e.py`, `official_deploy.py`)

qpeft trains PEQA for 300 steps on WikiText-2, merges, saves and loads, and exports the merged artifact in the official int layout.
Its `qweight` is the same GPTQ layout bit for bit; the zero-points are packed into `qzeros`.
`official_deploy.py` loads that artifact into the official `QuantLinear` and evaluates it.

| WikiText-2 perplexity | 4 bits | 2 bits |
|---|---|---|
| fp | 14.98 | 14.98 |
| RTN (PEQA start) | 17.49 | 234,639 |
| PEQA trained | 12.57 | 2,494 |
| merged | 12.57 (bit-exact) | 2,494 (bit-exact) |
| saved + loaded | 12.57 (bit-exact) | 2,494 (bit-exact) |
| official `QuantLinear`, fp32 | 12.57 | 2,494 |
| official `QuantLinear`, fp16 | 12.57 | 2,508 |

PEQA folds nothing into the zero-point, so it stays an integer and the artifact fits GPTQ-style formats without rounding.

## 3. PEQA vs EfficientQAT (`peqa_vs_efficientqat.py`)

Same model, same data, and the same E2E loop (300 steps, 4 x 512 tokens) for PEQA and for EfficientQAT.
Block-AP calibrates on 256 x 1024 tokens of WikiText-2 train, 2 epochs.
C4 validation is out of domain for the E2E phase.

| 4 bits | WikiText-2 | C4 |
|---|---|---|
| fp | 13.12 | 17.09 |
| RTN | 15.27 | 19.85 |
| PEQA | 12.22 | 19.31 |
| Block-AP only | 14.06 | 18.67 |
| EfficientQAT | 11.75 | 18.69 |

| 2 bits | WikiText-2 | C4 |
|---|---|---|
| RTN | 328,916 | 376,202 |
| PEQA | 2,793 | 5,409 |
| Block-AP only | 529 | 1,257 |
| EfficientQAT | 123 | 484 |

EfficientQAT is PEQA with a better start: Block-AP moves the codes, PEQA cannot.
At 4 bits the gap is small, and most of PEQA's WikiText gain is fitting that domain (C4 only moves from 19.85 to 19.31).
At 2 bits, training only the scales cannot repair RTN codes.
Every merged model is bit-exact with its training forward.

## Notes

- The official Triton dequant kernel sometimes fails with "illegal memory access" on its first call in a fresh process; a rerun passes.
- The 2-bit numbers are far from the official EfficientQAT results, which use far more data (4096 Block-AP samples, a full E2E epoch).
