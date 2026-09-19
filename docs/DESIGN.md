# Design & Abgrenzung

Warum qpeft existiert, gegen welche konkreten Fehlermodi es gebaut ist, und wo
jede Interface-Entscheidung im Code steht. Dies ist das „warum", die README ist
das „was".

## Das Problem: eine Wurzel, drei Symptome

**Wurzel.** Der `fake_quant`, der im Training benutzt wird, und der `fuse`/`save`,
der beim Export benutzt wird, sind getrennte Code-Pfade. Wenn sie auch nur leicht
auseinanderlaufen, trainierst du ein Modell, das im Deployment nicht existiert.

Drei bestehende Ansätze zeigen drei Reaktionen auf dieselbe Wurzel:

- **torchao + torchtune** komponieren QAT mit LoRA und erreichen ein quantisiertes
  Modell mit minimalem Qualitätsverlust - aber die Fähigkeit steckt in einem
  **CUDA- und torchtune-Rezept** fest. Es ist keine herauslösbare Primitive,
  sondern ein Framework-Flow.
- **unsloth-zoo (MLX-PR)** läuft frontal in die Wurzel: `merged_4bit` lässt die
  quantisierte Base nicht sauber, also `fuse ≠ fake_quant`. Die einzig sichere
  Reaktion des Autors ist, die betroffenen Schemata zu **verweigern** - „a
  fake-quant that does not match fuse() is worse than none". Korrekt, aber
  ad hoc und pro Backend.
- **peft** weicht aus: `merge_and_unload()` **dequantisiert** zu fp16. Das Problem
  ist umgangen, aber das quantisierte Artefakt ist beim Mergen verloren.

**Fazit.** Niemand besitzt das Paar `(fake_quant, fuse)` als *Vertrag*. Deshalb
wird es pro Backend neu ausgefochten, und der allgemeine Fall - ein Adapter, der
in das quantisierte Artefakt faltet und quantisiert bleibt - fällt durch die
Ritzen.

## Die drei Ansätze im Vergleich

| | trainierbar | Merge-Artefakt | fake_quant ↔ fuse | Backend / Framework | herauslösbare Primitive? |
|---|---|---|---|---|---|
| **peft (QLoRA)** | Adapter | **dequantisiert (fp16)** | n/a (kein QAT-Merge) | an bnb/torchao delegiert | nein |
| **torchtune + torchao** | Base fake-quant + LoRA | quantisiert | intern gematcht | **CUDA + torchtune-Rezept** | **nein (framework-gekoppelt)** |
| **unsloth-zoo (MLX)** | LoRA über QuantizedLinear | `merged_4bit` | **verweigert bei ≠** | **MLX / Apple Silicon** | nein (backend-spezifisch) |
| **qpeft** | `{weight, scale, zero_point, adapter}` wählbar | **bleibt int (per Signatur)** | **erzwungen + getestet** | Backend hinter dem Scheme | **ja (contract-first)** |

## Was qpeft anders macht - und wo es im Interface steht

### 1. (fake_quant, fuse) ist EIN Objekt
`fake_quant()` und `merge()` sind Methoden derselben `QuantScheme`. Man kann die
Trainings-Quantisierung nicht definieren, ohne den passenden Export-Fuse
danebenzulegen. Das verhindert das Auseinanderdriften strukturell, statt es zu
dokumentieren. → `qpeft/schemes.py::QuantScheme`

### 2. Äquivalenz ist ein Test-Gate
Die Einsicht „mismatch ist schlechter als keiner" wird zur testbaren Invariante:
der Trainingspfad (`fake_quant`) muss numerisch dem gemergten Pfad (`merge` →
`dequant`) entsprechen. Das ist der Meilenstein, an dem ein Scheme „gültig" wird.
→ `qpeft/utils.py::check_merge_equivalence`

### 3. Refuse statt Approximate - als First-Class-Mechanismus
Was der MLX-Autor von Hand pro Scheme tat, ist hier ein Interface-Vertrag: ein
Scheme deklariert über `supports(config)`, für welche Configs sein Fuse den
Fake-Quant garantiert trifft, und `assert_supported` **verweigert** beim
Model-Bau lautstark, wenn nicht. Keine stille Annäherung.
→ `QuantScheme.supports` / `assert_supported` / `UnsupportedSchemeError`,
   aufgerufen in `build_scheme`

### 4. Backend ist Implementierung, nicht Vertrag
`qat_scheme=` benennt den *Vertrag* (die Repräsentation), `backend=` benennt die
*Implementierung* (torchao_cuda, mlx, ...). Damit sind CUDA/torchao und MLX zwei
Implementierungen desselben `QuantScheme`-Interfaces, beide durch dasselbe
Äquivalenz-Gate abgesichert. Das ist die Entkopplung, die torchtune fehlt.
→ `QuantTuningConfig.backend` + `build_scheme` (Registry pro Vertrag, Backend im Factory)

### 5. Merge bleibt quantisiert - per Signatur
`merge(wq, s, z, adapter) -> (wq', s', z')`: int rein, int raus. Anders als pefts
dequantisierender Merge ist das quantisierte Artefakt das Ergebnis, nicht ein
Zwischenschritt, der weggeworfen wird.
→ `QuantScheme.merge` / `QuantModel.merge_and_unload`

## Was daraus folgt

Ein neues Verfahren (EfficientQAT, QA-LoRA, PEQA, L4Q, ...) ist eine Config über
den Achsen plus höchstens eine neue Operation (der Zero-Point-Fold). Ein neues
Backend ist eine `QuantScheme`-Subklasse, die vier Primitive implementiert und
`supports()` einschränkt - und automatisch am selben Gate gemessen wird. Der
allgemeine Fall, der bisher durch die Ritzen fiel, ist damit der Normalfall,
nicht die Ausnahme.

## Vorgesehen für später: ternär (noch nicht drin)

Ternär (1.58-bit, {-1, 0, +1} mit group-weiser FP16-Scale, symmetrisch, kein
Zero-Point) ist ein *vorgesehener zukünftiger Scheme* - nicht mit den Dingen im
Abschnitt "Was es bewusst NICHT ist" (README) zu verwechseln. Jene
(Codebook/Vektor-Quant, SBC) sprengen das Substrat; ternär liegt *innerhalb*
davon: gruppiert, uniform, gewichts-only. Es wird deshalb später ein
Registry-Eintrag (`qat_scheme="ternary"`) plus eine Export-Schicht in ein
serviertes Format (GGUF Q2_0_g128, MLX 2-bit), kein Redesign. Ausgeliefert ist
diese Repräsentation bereits - PrismMLs Bonsai ist genau das, inklusive "packed
weights, never expanded to FP16", was 1:1 dem `merge`-Contract entspricht.

Aktuell außerhalb des Scopes. Damit der Pfad offen bleibt, ohne dass eine Zeile
ternär-spezifischer Code entsteht, reichen drei *unterlassene* Fehler: (1) `bits`
nicht zur alleinigen Wahrheit über die Repräsentation machen - der
`qat_scheme`-String ist der Vertrag, keine `if bits == 4`-Logik streuen;
(2) `zero_point` nicht als überall vorhanden voraussetzen - ternär ist
symmetrisch und ignoriert ihn (der trainable-set-Switch und `supports()` erlauben
das bereits); (3) `merge` bleibt "int rein, int raus", nie fp16. Alle drei sind
schon so gebaut - sie sind nur nicht zu verletzen.
