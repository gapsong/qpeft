# qpeft

**Quantization-aware, PEFT-style tuning whose merge stays quantized.**

`qpeft` trains over a quantized substrate and, unlike `peft`, its
`merge_and_unload()` returns an *integer* model rather than dequantizing back to
fp16. It borrows torchao for the low-level primitives (fake-quant, packed dtypes,
kernels) and owns the one seam neither `peft` nor `unsloth` owns: **a QAT-trained
adapter/parameter set that folds into the quantized weights and stays quantized,
with a correctness guarantee.**

---

## Warum dieses Repo existiert

Der quantisierte Trainings-Space hat ein Zuständigkeitsloch, das bisher niemand
als *ein* Feature besitzt:

- **peft** friert die Base ein und trainiert Adapter. Sein `merge_and_unload()`
  **dequantisiert** das Modell — das quantisierte Artefakt geht beim Mergen
  verloren. Quantisierung selbst ist an bitsandbytes/torchao delegiert; der
  „Merge bleibt quantisiert"-Fall ist nur halb abgedeckt (torchao mergt sauber
  nur `int8_weight_only` + LoRA; AQLM/AWQ können gar nicht mergen).
- **unsloth** ist auf schnelles, speichersparsames QLoRA/LoRA optimiert — aber
  auf demselben Frozen-Base-plus-getrennter-Adapter-Modell. Echtes QAT ist dort
  backend-spezifisch und in Arbeit (der MLX-QAT-PR), kein backend-agnostischer
  Vertrag.

`qpeft` schließt genau diese Naht: Training über einem quantisierten Substrat,
dessen Merge quantisiert bleibt — als First-Class-Contract mit Correctness-Test.

## Das Kernprinzip

Zwei Ideen tragen alles:

1. **Die trainierbare Menge ist die erste Achse.** Nicht „Adapter vs. Frozen",
   sondern eine Auswahl aus `{weight, scale, zero_point, adapter}`
   (`TrainableParams`). Damit sind PEQA (nur `scale`), EfficientQAT
   (`weight+scale+zero_point`, dann `scale`) und QA-LoRA (`adapter` faltet in
   `zero_point`) **Konfigurationen über einem Substrat**, keine getrennten
   Subsysteme.
2. **`fake_quant` muss zum `merge` passen.** Der STE-Surrogat im Training und die
   exakte Faltung beim Export müssen numerisch übereinstimmen — ein Fake-Quant,
   der nicht zum Fuse passt, ist schlechter als keiner. `check_merge_equivalence`
   macht das zur testbaren Invariante. Das ist der Spine der Lib.

## Was es über peft hinaus kann

- **Quantisierungsparameter trainieren.** `peft` kann „trainiere den Quant-Scale"
  gar nicht ausdrücken; hier ist es `trainable_params=(SCALE,)`.
- **Merge bleibt int.** `QuantModel.merge_and_unload()` liefert ein quantisiertes
  Modell, nicht fp16 — die entgegengesetzte Semantik zu `peft`, bewusst.
- **Der Methoden-Zoo sind Configs.** EfficientQAT, QA-LoRA, PEQA, L4Q, LoftQ-Init
  … jeweils ein Punkt auf den Achsen. Ein neues Paper = eine Config plus evtl.
  eine Operation, keine neue Integration.
- **Correctness als Invariante.** `check_merge_equivalence` gibt es in `peft` so
  nicht.

## Was es über unsloth hinaus kann

- **Backend-agnostischer Vertrag** statt eines einzelnen Backend-PRs.
- **QAT + Adapter als ein parametrisiertes Rezept** — die Kombination, die
  torchtune/torchao nur teilweise und nur pro Backend anbieten.
- **Nicht nur Speed auf dem Frozen-Base-Modell**, sondern das quantisierte
  Trainings-Artefakt selbst als Ziel.

## Verhältnis zu torchao

`qpeft` ersetzt torchao nicht, es hängt sich dran: der `fake_quant` bindet an
torchaos QAT-`FakeQuantizer`, `quantize`/`dequant` an den `AffineQuantizedTensor`
(Datentyp + tinygemm/Marlin-Kernel + `torch.compile`/FSDP), Export an
`quantize_`. `qpeft` besitzt nur die Naht, die torchao offen lässt: die Faltung
des Adapters in den `zero_point` plus die mehrphasige QAT-Abfolge — mit einem
Test, der beweist, dass beide Pfade übereinstimmen.

## Was es bewusst NICHT ist

Kohärente Scheibe statt Do-Everything-Wrapper: **weight-only, gruppiert,
uniform-int, Decoder-LLMs.** Ausdrücklich draußen — Codebook/Vektor-Quant
(AQLM, QuIP#) und SBC-artige stochastische Binär-Codecs. Anderes Substrat,
anderer Inferenzoperator; das gehört in ein Schwesterprojekt, nicht hierher.

## Struktur (spiegelt peft)

```
qpeft/
  config.py                     # QuantTuningType, TrainableParams, QuantTuningConfig  (~ PeftType/PeftConfig)
  schemes.py                    # FakeQuantizeConfig, QuantScheme, Registry           (qat_scheme= a la unsloth)
  mapping.py                    # get_quant_model + Registries                        (~ get_peft_model)
  peft_model.py                 # QuantModel.merge_and_unload()                       (~ PeftModel)
  utils.py                      # check_merge_equivalence                             (der Spine-Test)
  tuners/
    tuners_utils.py             # BaseQuantTuner, AdapterLayer, QuantLinear           (~ BaseTuner/BaseTunerLayer/lora.Linear)
    efficient_qat/{config,model,layer}.py
    qa_lora/{config,model,layer,torchao}.py   # torchao.py ~ peft lora/torchao.py
examples/quickstart.py
pyproject.toml
```

## Design & Abgrenzung

Warum das Repo existiert, die drei Vergleichs-Ansätze und wo jede
Interface-Entscheidung im Code steht: siehe [`docs/DESIGN.md`](docs/DESIGN.md).

## Basic usage

```bash
pip install -e .            # Laufzeit: torch; echte Schemes: pip install -e ".[torchao]"
python examples/quickstart.py
```

```python
import torch.nn as nn
from qpeft import get_quant_model, QALoraConfig, efficient_qat_schedule

base = nn.Sequential(nn.Linear(512, 512))

# EfficientQAT: zwei Phasen (Block-AP -> E2E-QP), je eine Config
for cfg in efficient_qat_schedule(bits=2, group_size=64):
    model = get_quant_model(base, cfg)
    # ... train phase ...

# QA-LoRA: eine Config; Adapter faltet in die Zero-Points
model = get_quant_model(base, QALoraConfig(bits=4, group_size=32, r=64))
# ... train ...
quantized = model.merge_and_unload()          # bleibt int
```

Den Spine-Test pro Layer laufen lassen, bevor man einem Scheme traut:

```python
from qpeft import check_merge_equivalence
check_merge_equivalence(scheme, w, s, z, adapter, x)   # fake_quant (train) == merge (export)
```

## Status

Der `int_uniform`-Vertrag ist gegen ZWEI Backends ausimplementiert und beide sind
am selben Gate grün (`check_merge_equivalence`: EfficientQAT ohne Adapter exakt,
QA-LoRA-Fold < 1e-4): `backend="auto"` (pure-torch `ReferenceIntUniformScheme`,
immer da) und `backend="torchao"` (`TorchaoIntUniformScheme` auf torchaos stabilen
`quant_primitives`). `examples/train_*.py` zeigen echtes Training + int-Merge,
`examples/hf_injection.py` das Injizieren in ein Hugging-Face-Modell. Weiter Stub:
`TorchaoQuantLinear` (ein von torchao SCHON gepacktes Layer übernehmen - blockiert
durch torchaos in-flux Tensor-Subclass-API), der `mlx`-Backend und der geplante
ternär-Scheme.
