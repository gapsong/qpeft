"""WikiText-2 train as pre-tokenized windows of exactly SEQ_LEN tokens (axolotl takes a dataset
with input_ids / attention_mask / labels as it is), so every training step sees the same number
of tokens in every arm and tokens/s is steps * batch * SEQ_LEN / time."""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1")["train"]["text"])
    ids = tok(text).input_ids
    n_windows = len(ids) // a.seq_len
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for i in range(n_windows):
            window = ids[i * a.seq_len:(i + 1) * a.seq_len]
            f.write(json.dumps({"input_ids": window, "attention_mask": [1] * a.seq_len, "labels": window}) + "\n")
    print(f"{n_windows} windows of {a.seq_len} tokens -> {a.out}")


if __name__ == "__main__":
    main()
