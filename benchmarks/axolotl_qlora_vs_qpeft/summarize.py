"""One line per training run from its measure.json."""
import json
import sys
from pathlib import Path

for path in sys.argv[1:]:
    m = json.loads(Path(path).read_text())
    print(f"{Path(path).parent.name:14s} {m['tokens_per_s']:8.0f} tok/s  peak {m['peak_memory_gib']:.2f} GiB  "
          f"final loss {m['losses'][-1][1]}")
