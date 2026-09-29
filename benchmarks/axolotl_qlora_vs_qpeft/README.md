# axolotl QLoRA against qpeft QA-LoRA

The same axolotl run twice: once as axolotl's QLoRA (bitsandbytes NF4 base + LoRA), once as qpeft's QA-LoRA through the plugin (`adapter: qpeft`, 4-bit integer codes, group size 64), with qpeft on its Triton kernel and on the torch path.

```bash
bash benchmarks/axolotl_qlora_vs_qpeft/run.sh      # env with qpeft[axolotl], bitsandbytes, peft; one CUDA GPU
```

## Setup

- Model: SmolLM2-360M (bf16). Not Qwen3-0.6B: the GPU was shared with another job that held 9 to 13 GB.
- Data: WikiText-2 train, pre-tokenized into 5016 windows of exactly 512 tokens (`prepare_data.py`), so every step is 8 x 512 = 4096 tokens in every arm.
- Same for all arms (`common.yml`): 300 steps, micro batch 8, no gradient accumulation, AdamW, lr 2e-4, cosine, 10 warmup steps, weight decay 0, bf16, seed 0, LoRA r 16, alpha 32, dropout 0.05, every linear of the decoder blocks.
- QLoRA (`qlora.yml`): NF4, double quant, blocksize 64, bf16 storage (axolotl's defaults for `load_in_4bit`).
- qpeft (`qpeft_qa_lora.yml`): QA-LoRA, 4 bits, group size 64. `QPEFT_TORCH_PATH=1` turns the Triton kernel off.
- `measure_plugin.py` measures every arm the same way: tokens/s over steps 11 to 300, peak memory allocated by the training process (read every step, because axolotl resets the peak statistics each time it logs).
- Perplexity (`evaluate.py`): 64 windows of 1024 tokens of WikiText-2 test and of C4 validation, as in `benchmarks/peqa_vs_official`.

Raw numbers: `results/`.

## Results (RTX 4090, torch 2.13 + CUDA 13, axolotl 0.19, bitsandbytes 0.50.2, peft 0.20)

### Training

| arm | tokens/s [unconfirmed: GPU shared] | peak memory | final loss (step 300) |
|---|---|---|---|
| axolotl QLoRA (NF4 + LoRA) | 25,054 | 7.08 GiB | 2.585 |
| qpeft QA-LoRA, Triton kernel | 23,363 | 6.18 GiB | 2.889 |
| qpeft QA-LoRA, torch path | 20,400 | 6.76 GiB | 2.889 |

tokens/s is [unconfirmed]: another job used the same GPU at 90 to 97 % utilization during all runs.
axolotl's own `tokens/train_per_sec_per_gpu` log agrees with it within about 3 %, and a repeat of both qpeft arms gave 23,319 and 20,413 tokens/s.
Peak memory is per process and not affected by the other job.

The Triton kernel makes qpeft's training 15 % faster and needs 0.58 GiB less memory than the torch path.
QLoRA is still 7 % faster.

Triton and torch runs are not bit-identical end to end, but neither are two torch runs: a repeat of the torch run differs from the first in 211 of 739 saved tensors, the same as Triton against torch (211); the losses agree to about 4 digits.
The training itself is not deterministic (nondeterministic GPU ops outside qpeft).
Layer by layer, the kernel is bit-identical to the torch path (`tests/test_triton_dequant.py`).

### Perplexity

| model | 4-bit? | WikiText-2 | C4 |
|---|---|---|---|
| fp (bf16), no training | no | 13.126 | 17.106 |
| NF4, no training | yes | 14.825 | 19.064 |
| QLoRA: NF4 + LoRA, unmerged (the model that was trained) | base yes, adapter bf16 | 11.322 | 21.146 |
| QLoRA merged into the bf16 base (what a peft / axolotl merge gives) | **no** | 10.905 | 19.608 |
| RTN int4 g64, no training (qpeft's start) | yes | 15.277 | 19.839 |
| qpeft QA-LoRA merged int4, Triton kernel | **yes** | 15.096 | 19.775 |
| qpeft QA-LoRA merged int4, torch path | **yes** | 15.096 | 19.775 |

- The qpeft artifact stays 4-bit (integer codes + per-group scale and zero-point), and the Triton and torch paths give the same perplexity to the last digit.
- The QLoRA merge is not 4-bit any more, and it is not the model that was trained (the adapter was trained on the NF4 weights, the merge adds it to the bf16 weights).
- At lr 2e-4, QA-LoRA barely trains: WikiText-2 15.28 -> 15.10, while QLoRA goes 14.83 -> 11.32.
  The QA-LoRA adapter sees the per-group average of its input, which is small, so its gradients are small (grad norm about 0.002 against QLoRA's 0.15).
- QLoRA gets worse on C4 (19.06 -> 21.15): 300 steps on WikiText-2 fit that one domain.

### Higher learning rates: the merge check refuses

To see whether QA-LoRA just needs a larger lr, two more runs (`qpeft_qa_lora_lr1e-3.yml`, `qpeft_qa_lora_lr2e-3.yml`):

| lr | final loss | plugin at the end (max delta of the first failing layer) |
|---|---|---|
| 2e-4 | 2.889 | merge check OK, artifact saved |
| 1e-3 | 2.667 | `MergeMismatchError`: max delta 0.375 > tolerance 0.348, nothing saved |
| 2e-3 | 2.619 | `MergeMismatchError`: max delta 0.0625 > tolerance 0.053, nothing saved |
| 2e-3, torch path | 2.619 | `MergeMismatchError`: max delta 0.25 > tolerance 0.221, nothing saved |

With a higher lr QA-LoRA's loss comes close to QLoRA's (2.62 against 2.59), but the merged bf16 model then differs from the trained one by more than the tolerance (1 % of the largest output), so the plugin refuses to write it.
It fails on the torch path too, so it is not the kernel.
The max delta is the first failing layer's.

It is **not mainly** the bf16 storage of the folded zero-point `z' = z - delta / s`. `merge_check_isolation.py` builds one bf16 QA-LoRA layer (960 x 960, g64, r16) with a growing adapter and compares the training forward with the merged output (max delta, seed 0):

| adapter weights (std of A and B) | tolerance | trained vs merged, z' bf16 | trained vs merged, z' fp32 | trained vs exact fp32 | merged (z' fp32) vs exact |
|---|---|---|---|---|---|
| 0.05 | 0.021 | 0.0117 | 0.0078 | 0.0093 | 0.0080 |
| 0.2 | 0.034 | 0.0234 | 0.0156 | 0.0176 | 0.0081 |
| 0.4 | 0.086 | 0.0625 | 0.0625 | 0.0446 | 0.0456 |
| 0.8 | 0.413 | 0.2500 | 0.1875 | 0.1980 | 0.0936 |

Keeping `z'` in fp32 makes the mismatch a bit smaller at some sizes (0.25 -> 0.19) and not at all at others (0.0625 both).
The training forward itself is about as far from the exact result as trained and merged are from each other: it computes the adapter branch and the sum in bf16.
The merged model with z' in fp32 is closer to the exact result than the training forward at most sizes (0.4: about equal).
The failing deltas are one to two bf16 rounding steps of the largest outputs (a bf16 step is 0.125 between 16 and 32, 0.25 between 32 and 64), and the tolerance of 1 % of the largest output is only about 2.5 steps.
So the check refuses rounding noise once the adapter makes the outputs large; how to change the check (compare both sides to an fp32 reference, count bf16 steps, or run the adapter branch in fp32) is the owner's decision and not changed here.
The refusal also discards the trained adapter: the plugin checks before it saves anything.
