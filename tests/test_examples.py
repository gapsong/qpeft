"""Examples must stay clean: each script asserts its own success, so running it
is the test. qat_trainer.py downloads a model + dataset -> only with QPEFT_RUN_SLOW=1."""
import os
import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _run(name, monkeypatch):
    monkeypatch.setattr(sys, "argv", [name])
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")


@pytest.mark.parametrize("name", ["quickstart.py", "train_efficient_qat.py", "train_qa_lora.py"])
def test_example_runs(name, monkeypatch):
    _run(name, monkeypatch)


def test_hf_injection_example_runs(monkeypatch):
    pytest.importorskip("transformers")
    _run("hf_injection.py", monkeypatch)


@pytest.mark.skipif(os.environ.get("QPEFT_RUN_SLOW") != "1", reason="downloads; set QPEFT_RUN_SLOW=1")
def test_qat_trainer_example_runs(monkeypatch, tmp_path):
    pytest.importorskip("datasets")
    monkeypatch.chdir(tmp_path)                        # it writes its output_dir to cwd
    _run("qat_trainer.py", monkeypatch)


def test_every_example_is_covered():
    covered = {"quickstart.py", "train_efficient_qat.py", "train_qa_lora.py",
               "hf_injection.py", "qat_trainer.py"}
    assert {p.name for p in EXAMPLES.glob("*.py")} == covered, "new example without a test"
