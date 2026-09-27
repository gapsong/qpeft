#!/bin/bash
# PEQA in qpeft against the official EfficientQAT code, on one CUDA GPU.
# Needs two conda envs: the official one (torch 2.2, transformers 4.40, triton) and one with qpeft.
#   EQAT_DIR    clone of https://github.com/OpenGVLab/EfficientQAT
#   QPEFT_DIR   this repo
#   OUT         where the dumps and artifacts go
set -u
EQAT_DIR=${EQAT_DIR:-~/Documents/EfficientQAT}
QPEFT_DIR=${QPEFT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
EQAT_ENV=${EQAT_ENV:-eqat_official}
QPEFT_ENV=${QPEFT_ENV:-qpeft}
OUT=${OUT:-$QPEFT_DIR/runs/peqa_vs_official}
MODEL=${MODEL:-HuggingFaceTB/SmolLM2-360M}
B=$QPEFT_DIR/benchmarks/peqa_vs_official
mkdir -p "$OUT"
source ~/miniforge3/etc/profile.d/conda.sh
official() { conda activate "$EQAT_ENV"; PYTHONPATH=$EQAT_DIR python -u "$@"; }
qpeft()    { conda activate "$QPEFT_ENV"; PYTHONPATH=$QPEFT_DIR python -u "$@"; }

# 1. Training step by step: same codes, same steps as the official code.
for s in "4 1e-5" "2 2e-5"; do set -- $s
  echo "######## step by step: w$1 g64"
  official $B/official_peqa.py --model $MODEL --bits $1 --group-size 64 --lr $2 --out $OUT/official_w$1g64.pt
  qpeft $B/qpeft_peqa.py --model $MODEL --bits $1 --group-size 64 --lr $2 --official $OUT/official_w$1g64.pt
  qpeft $B/chaos_check.py $MODEL $1 $2 $OUT/official_w$1g64.pt
done

# 2. End to end: train, merge, save / load, run the merged artifact in the official int layer.
for s in "4 2e-5" "2 5e-5"; do set -- $s
  echo "######## end to end with merge: w$1 g64"
  qpeft $B/qpeft_peqa_e2e.py --model $MODEL --bits $1 --group-size 64 --lr $2 --out $OUT/e2e_w$1g64
  official $B/official_deploy.py --model $MODEL --bits $1 --group-size 64 --dir $OUT/e2e_w$1g64
done

# 3. Quality: PEQA vs Block-AP vs EfficientQAT, same data and E2E phase.
echo "######## PEQA vs EfficientQAT"
qpeft $B/peqa_vs_efficientqat.py --model $MODEL --bits 4 --e2e-lr 2e-5 --weight-lr 1e-5
qpeft $B/peqa_vs_efficientqat.py --model $MODEL --bits 2 --e2e-lr 5e-5 --weight-lr 2e-5
