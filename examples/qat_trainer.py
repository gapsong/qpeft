"""EfficientQAT with the QATTrainer -- same shape as peft + Trainer.

Run:  pip install -e ".[train]" datasets
      python examples/qat_trainer.py        (a GPU is strongly recommended)
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from qpeft import EfficientQATConfig, QATTrainer, QATTrainingArguments, QuantModel

model_id = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("Abirate/english_quotes", split="train")
dataset = dataset.map(lambda s: tokenizer(s["quote"]), batched=True,
                      remove_columns=dataset.column_names)

trainer = QATTrainer(
    model=model,                                              # the plain Hugging Face model
    quant_config=EfficientQATConfig(bits=2, group_size=64),   # targets: all Linear in the blocks
    train_dataset=dataset,
    args=QATTrainingArguments(                                # EfficientQAT paper defaults ...
        output_dir="qwen3-0.6b-w2g64",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        block_ap_train_size=256,                              # ... shortened for a quick run
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()        # Block-AP -> codes frozen -> E2E-QP, merge check at the end
trainer.save_model()   # merge -> integer model in output_dir

# Later: load the integer model back (the base model supplies the architecture).
base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
qmodel = QuantModel.from_pretrained(base, "qwen3-0.6b-w2g64")
