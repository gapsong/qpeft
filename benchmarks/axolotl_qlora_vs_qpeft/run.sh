#!/bin/bash
# axolotl QLoRA vs qpeft QA-LoRA (plugin, Triton kernel and torch path) on one CUDA GPU.
# Needs an env with qpeft[axolotl] (axolotl >= 0.19), bitsandbytes and peft.
#   OUT   where configs, runs and results go (default: <repo>/runs/axolotl_qlora_vs_qpeft)
set -eu
B=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$B/../../runs/axolotl_qlora_vs_qpeft}
MODEL=HuggingFaceTB/SmolLM2-360M
mkdir -p "$OUT"
export PYTHONPATH=$B${PYTHONPATH:+:$PYTHONPATH}      # measure_plugin

python "$B/prepare_data.py" --model $MODEL --seq-len 512 --out "$OUT/wiki_train_512.jsonl"

train() {   # name arm.yml [env]
  python "$B/make_config.py" "$B/common.yml" "$B/$2" "$OUT/wiki_train_512.jsonl" "$OUT/$1" "$OUT/$1.yml"
  env ${3:-} python -m axolotl.cli.train "$OUT/$1.yml" > "$OUT/$1.log" 2>&1
  python "$B/summarize.py" "$OUT/$1/measure.json"
}
train qlora qlora.yml
train qpeft_triton qpeft_qa_lora.yml
train qpeft_torch qpeft_qa_lora.yml QPEFT_TORCH_PATH=1

python "$B/evaluate.py" --model $MODEL --runs "$OUT" | tee "$OUT/eval.log"
python "$B/compare_artifacts.py" "$OUT/qpeft_triton/qpeft" "$OUT/qpeft_torch/qpeft"
