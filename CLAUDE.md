# qpeft

Quantization-aware, PEFT-style tuning whose **merge stays quantized**.
Full problem statement & approach comparison: @docs/DESIGN.md
Overview & usage: @README.md

## Invarianten (nicht verletzen)

1. **fake_quant (Training) == merge/fuse (Export).** `check_merge_equivalence`
   muss grün sein, bevor einem Scheme getraut wird. Ein fake_quant, der nicht
   zum fuse passt, ist schlechter als keiner.
2. **Refuse statt approximate.** Eine nicht unterstützte Kombination aus
   (scheme, backend, config) wirft `UnsupportedSchemeError` beim Model-Bau -
   niemals still annähern.
3. **merge bleibt int.** `merge(wq, s, z, adapter) -> (wq', s', z')`, niemals
   dequantisieren. Das ist die bewusste Umkehr von pefts `merge_and_unload`.
4. **Die trainierbare Menge ist die erste Achse.** `{weight, scale, zero_point,
   adapter}`. Verfahren (EfficientQAT, QA-LoRA, PEQA, ...) sind Configs darüber,
   keine neuen Subsysteme.
5. **backend = Implementierung, qat_scheme = Vertrag.** Ein neues Backend ist
   eine `QuantScheme`-Subklasse, die vier Primitive implementiert und
   `supports()` einschränkt.

## Namenskonvention (spiegelt peft/torchtune)

`<Method>Config`, `<Method>Model(BaseQuantTuner)`, `QuantLinear`,
`merge` / `merge_and_unload` / `unmerge`, `dispatch_default` / `dispatch_torchao`,
`TrainableParams`. Nicht davon abweichen ohne Grund.

## Struktur

- `qpeft/config.py` `schemes.py` `mapping.py` `peft_model.py` `utils.py`
- `qpeft/tuners/tuners_utils.py` (BaseQuantTuner, QuantLinear)
- `qpeft/tuners/{efficient_qat,qa_lora}/` (config, model, layer[, torchao])

## Commands

- `pip install -e ".[dev]"` - Editable-Install (echte Schemes: `.[torchao]`)
- `python examples/quickstart.py` - Konstruktionspfad beider Methoden
- `pytest` - Tests (sobald vorhanden)

## Status & aktueller Meilenstein

Der `int_uniform`-Vertrag ist für ZWEI Backends ausimplementiert und beide sind
am selben Gate grün (`check_merge_equivalence`: EfficientQAT ohne Adapter exakt,
QA-LoRA-Fold < 1e-4):
- `backend="auto"` -> `ReferenceIntUniformScheme` (pure torch, grouped affine +
  STE, RTN-Init). Immer verfügbar.
- `backend="torchao"` -> `TorchaoIntUniformScheme` (`qpeft/schemes_torchao.py`,
  auf torchaos STABILEN Primitiven `quant_primitives`; lazy import). Verweigert
  trainierbaren `zero_point` (INT zero-point domain) - `supports()` in Aktion.

Tests: `tests/test_merge_equivalence.py` + `test_torchao_backend.py` (skippt ohne
torchao) + `test_custom_models/tuners_utils/initialization/config.py`.
`examples/train_*.py` und `examples/hf_injection.py` zeigen Training, int-Merge
und HF-Injection.

**Nächster Schritt (Auswahl):** (a) `TorchaoQuantLinear` = ein SCHON von torchao
gepacktes Layer übernehmen (Fall b) - blockiert durch torchaos in-flux
Tensor-Subclass-API (int4 braucht Kernel-Lib, int8 `Int8Tensor.qdata`); braucht
gepinnte torchao-Version + Ziel-Hardware. (b) Save/Load-Schicht (`PeftModel`-artig).
(c) `mlx`-Backend. Stubs nicht implementieren, ohne im selben Schritt den
Äquivalenztest mitzuliefern.

## Langfristig vorgesehen (jetzt NICHT bauen)

Ternär (1.58-bit, {-1, 0, +1}, group-weise FP16-Scale, symmetrisch, kein
Zero-Point) ist ein geplanter zukünftiger Scheme - innerhalb des Substrats, also
später ein Registry-Eintrag `qat_scheme="ternary"` plus eine Export-Schicht
(GGUF Q2_0_g128 / MLX 2-bit), kein Redesign. Jetzt keinen Stub und keinen
`export()`-Haken dafür anlegen; tote Platzhalter veralten nur.

Damit der Pfad offen bleibt, drei Annahmen NICHT einziehen: `bits` ist nicht die
alleinige Wahrheit über die Repräsentation (der `qat_scheme`-String ist der
Vertrag); `zero_point` ist nicht überall vorhanden (ternär ignoriert ihn);
`merge` gibt immer int zurück, nie fp16. Alle drei sind bereits so - nur nicht
verletzen.
