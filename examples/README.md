# examples

Runnable, self-checking examples - the qpeft analog of peft's `examples/`.
Each script trains a tiny model and asserts its own success, so a clean exit
means it worked.

```bash
pip install -e .            # runtime is just torch
python examples/quickstart.py            # construction path, no training
python examples/train_efficient_qat.py   # QAT over a 2-bit substrate + merge
python examples/train_qa_lora.py         # adapter training + zero-point fold merge
```

## Was heute läuft

Alle Skripte laufen gegen den dependency-freien Referenz-Backend
(`qat_scheme="int_uniform"`, `backend="auto"`, reines PyTorch):

- **`train_efficient_qat.py`** - quantization-aware Training über einem 2-bit
  Gitter (STE fake_quant), danach `check_merge_equivalence` grün pro
  `QuantLinear`, danach `merge_and_unload()` -> **int-Artefakt**, dessen Ausgabe
  exakt der trainierten Ausgabe entspricht.
- **`train_qa_lora.py`** - frozen quantisierte Base + trainierbarer Adapter. Der
  Adapter ist pro Quantisierungs-Gruppe gepoolt, faltet beim Merge **exakt** in
  die Zero-Points und bleibt int (der bewusste Gegensatz zu pefts
  dequantisierendem Merge).

Der Spine-Test dahinter ist `tests/test_merge_equivalence.py` (`pytest`).

## Was noch fehlt

- **Backends** `torchao_cuda` / `mlx` sind noch nicht gebaut - sie verweigern
  beim Model-Bau (`backend="mlx"` -> Fehler), statt still anzunähern. Sie werden
  denselben `int_uniform`-Vertrag implementieren und am selben Gate gemessen.
- Der **ternär**-Scheme ist bewusst noch nicht drin (siehe `docs/DESIGN.md`).
- EfficientQATs Phasen-Übergang (Block-AP -> E2E-QP mit eingefrorenen int-Codes)
  ist im Skelett vereinfacht; das Beispiel zeigt die Block-AP-Phase.
