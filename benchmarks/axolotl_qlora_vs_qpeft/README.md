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

| lr | final loss | plugin at the end |
|---|---|---|
| 2e-4 | 2.889 | merge check OK, artifact saved |
| 1e-3 | 2.667 | `MergeMismatchError`: max delta 0.375 > tolerance 0.348, nothing saved |
| 2e-3 | 2.619 | `MergeMismatchError`: max delta 0.0625 > tolerance 0.053, nothing saved |
| 2e-3, torch path | 2.619 | `MergeMismatchError`: max delta 0.25 > tolerance 0.221, nothing saved |

With a higher lr QA-LoRA's loss comes close to QLoRA's (2.62 against 2.59), but the merged bf16 model then differs from the trained one by more than the 1 % tolerance, so the plugin refuses to write it.
It fails on the torch path too, so it is not the kernel.
The cause is the known bf16 fold: the merged zero-point `z' = z - delta / s` is stored in the model dtype (bf16), and with a larger adapter the rounding of `z'` grows.
Keeping `z'` in fp32 would change the save format (`docs/specs/save_load.md`); that is the owner's decision, so it is not changed here.
The refusal also discards the trained adapter: the plugin checks before it saves anything.
