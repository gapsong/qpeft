"""QATTrainer (docs/specs/qat_trainer.md): EfficientQAT only, takes the plain HF
model. Needs transformers; tiny random Llama, no download."""
import pytest

pytest.importorskip("transformers")

import torch                                                    # noqa: E402
from torch.utils.data import Dataset                            # noqa: E402

from qpeft import (                                             # noqa: E402
    EfficientQATConfig, QALoraConfig, QATTrainer, QATTrainingArguments, QuantModel,
    UnsupportedSchemeError, get_quant_model,
)
from qpeft.tuners.tuners_utils import QuantLinear               # noqa: E402

BLOCK_LINEARS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


class _Data(Dataset):
    def __init__(self, n=16, seq=32, vocab=320):
        g = torch.Generator().manual_seed(0)
        self.ids = torch.randint(0, vocab, (n, seq), generator=g)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.ids[i]}


def _collate(rows):
    return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}


def _llama(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, vocab_size=320, max_position_embeddings=64))


def _args(tmp_path, **kw):
    base = dict(output_dir=str(tmp_path), per_device_train_batch_size=4, num_train_epochs=1,
                report_to=[], dataloader_pin_memory=False,           # MPS cannot pin memory
                block_ap_epochs=1, block_ap_train_size=8,
                weight_lr=1e-3, quant_lr=1e-3, e2e_lr=1e-3,
                # 16 samples: with the official accumulation (8) + warmup the only
                # optimizer step would run at lr 0 and nothing could be learned.
                gradient_accumulation_steps=1, warmup_steps=0)
    base.update(kw)
    return QATTrainingArguments(**base)


def _trainer(model, tmp_path, quant_config=None, **kw):
    return QATTrainer(model=model, quant_config=quant_config, args=_args(tmp_path, **kw),
                      train_dataset=_Data(), data_collator=_collate)


def _cfg(**kw):
    return EfficientQATConfig(bits=4, group_size=64, **kw)


def _snap(qmodel):
    out = {}
    for name, m in qmodel.base.named_modules():
        if isinstance(m, QuantLinear):
            out[f"{name}.scale"] = m.scale.detach().clone()
            out[f"{name}.zero_point"] = m.zero_point.detach().clone()
            out[f"{name}.codes"] = (m.codes if m.codes_frozen
                                    else m.scheme.quantize(m.weight, m.scale, m.zero_point))
    return out


def _changed_kinds(before, after):
    return {k.split(".")[-1] for k in before if not torch.equal(before[k], after[k])}


# --- the plain HF model is enough --------------------------------------------------

def test_accepts_plain_hf_model_and_targets_block_linears(tmp_path):
    trainer = _trainer(_llama(), tmp_path, quant_config=_cfg())
    assert isinstance(trainer.model, QuantModel)
    assert set(trainer.model.config.target_modules) == BLOCK_LINEARS
    names = {n.split(".")[-1] for n, m in trainer.model.base.named_modules()
             if isinstance(m, QuantLinear)}
    assert names == BLOCK_LINEARS                       # lm_head stays fp, as in the paper


def test_default_targets_come_from_every_block(tmp_path):
    """Hybrid stacks (Qwen3-Next, Jamba, dense-then-MoE) have linears that only some blocks
    have. Taking the names from block 0 alone would leave those fp and say nothing."""
    model = _llama()
    model.model.layers[1].mlp.extra_proj = torch.nn.Linear(128, 128)   # only in block 1
    trainer = _trainer(model, tmp_path, quant_config=_cfg())
    assert set(trainer.model.config.target_modules) == BLOCK_LINEARS | {"extra_proj"}
    assert isinstance(trainer.model.base.model.layers[1].mlp.extra_proj, QuantLinear)


def test_default_config_is_paper_default(tmp_path):
    """No quant_config -> EfficientQATConfig() = bits 4, group_size 128."""
    from transformers import LlamaConfig, LlamaForCausalLM
    model = LlamaForCausalLM(LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=4, vocab_size=320))
    trainer = _trainer(model, tmp_path)
    assert (trainer.model.config.bits, trainer.model.config.group_size) == (4, 128)


# --- refusals ----------------------------------------------------------------------

def test_refuses_qa_lora_config(tmp_path):
    with pytest.raises(UnsupportedSchemeError):
        _trainer(_llama(), tmp_path, quant_config=QALoraConfig(bits=4, group_size=64, r=8))


def test_refuses_qa_lora_quant_model(tmp_path):
    qmodel = get_quant_model(_llama(), QALoraConfig(bits=4, group_size=64, r=8,
                                                    target_modules=sorted(BLOCK_LINEARS)))
    with pytest.raises(UnsupportedSchemeError):
        _trainer(qmodel, tmp_path)


def test_refuses_targets_that_match_nothing(tmp_path):
    with pytest.raises(ValueError):
        _trainer(_llama(), tmp_path, quant_config=_cfg(target_modules=["does_not_exist"]))


def test_refuses_merged_model(tmp_path):
    qmodel = get_quant_model(_llama(), _cfg(target_modules=sorted(BLOCK_LINEARS)))
    qmodel.merge_and_unload()
    with pytest.raises(ValueError):
        _trainer(qmodel, tmp_path)


def test_refuses_mid_training_checkpoints(tmp_path):
    with pytest.raises(ValueError):
        _trainer(_llama(), tmp_path, quant_config=_cfg(), save_strategy="steps")


# --- phases ------------------------------------------------------------------------

def test_runs_both_phases_with_the_right_trainable_sets(tmp_path):
    trainer = _trainer(_llama(), tmp_path, quant_config=_cfg())
    before = _snap(trainer.model)
    w0 = {id(m): m.weight.detach().clone() for m in trainer.model.quant_layers()}

    # Record the state right when E2E-QP starts (after Block-AP + hand-over).
    import qpeft.trainer as T
    after_block_ap, grid = {}, {}
    orig = T.Trainer.train

    def spy(self, *a, **kw):
        after_block_ap.update(_snap(self.model))
        grid.update({id(m): (m.scale.detach().clone(), m.zero_point.detach().clone())
                     for m in self.model.quant_layers()})
        layers = self.model.quant_layers()
        assert all(m.codes_frozen and m.weight is None for m in layers)
        assert {n for m in layers for n in ("scale", "zero_point") if getattr(m, n).requires_grad} == {"scale"}
        return orig(self, *a, **kw)

    T.Trainer.train = spy
    try:
        trainer.train()
    finally:
        T.Trainer.train = orig

    assert _changed_kinds(before, after_block_ap) == {"codes", "scale", "zero_point"}
    # the new grid alone would also change the codes; the weight moved too:
    assert any(not torch.equal(m.codes, m.scheme.quantize(w0[id(m)], *grid[id(m)]))
               for m in trainer.model.quant_layers())
    assert _changed_kinds(after_block_ap, _snap(trainer.model)) == {"scale"}


def test_bit_dependent_defaults_resolve(tmp_path):
    trainer = QATTrainer(model=_llama(), quant_config=EfficientQATConfig(bits=2, group_size=64),
                         args=QATTrainingArguments(output_dir=str(tmp_path), report_to=[]),
                         train_dataset=_Data(), data_collator=_collate)
    assert (trainer.args.weight_lr, trainer.args.e2e_lr, trainer.args.quant_lr) == (2e-5, 2e-5, 1e-4)


# --- guard -------------------------------------------------------------------------

def test_zero_learning_rate_is_caught(tmp_path):
    """E2E-QP with lr 0: nothing moves -> abort (peft PR #2571 lesson)."""
    qmodel = get_quant_model(_llama(), _cfg(phase="e2e_qp", target_modules=sorted(BLOCK_LINEARS)))
    with pytest.raises(RuntimeError):
        _trainer(qmodel, tmp_path, e2e_lr=0.0).train()


# --- end to end --------------------------------------------------------------------

def test_train_save_load_is_consistent(tmp_path):
    trainer = _trainer(_llama(), tmp_path, quant_config=_cfg())
    trainer.train()
    x = _Data()[0]["input_ids"][None]
    device = next(trainer.model.parameters()).device    # the Trainer may move the model (cuda / mps)
    with torch.no_grad():
        before = trainer.model(input_ids=x.to(device)).logits.cpu()

    trainer.save_model(str(tmp_path / "final"))          # merges, writes the int artifact
    assert trainer.model.is_merged
    loaded = QuantModel.from_pretrained(_llama(seed=1), tmp_path / "final")
    with torch.no_grad():
        after = loaded(input_ids=x).logits
    assert (before - after).abs().max().item() <= 1e-3
    assert torch.equal(before.argmax(-1), after.argmax(-1))


# --- construction edge cases -------------------------------------------------------

def test_accepts_efficient_qat_quant_model_as_is(tmp_path):
    qmodel = get_quant_model(_llama(), _cfg(target_modules=sorted(BLOCK_LINEARS)))
    assert _trainer(qmodel, tmp_path).model is qmodel


def test_refuses_quant_model_plus_quant_config(tmp_path):
    qmodel = get_quant_model(_llama(), _cfg(target_modules=sorted(BLOCK_LINEARS)))
    with pytest.raises(ValueError, match="already a QuantModel"):
        _trainer(qmodel, tmp_path, quant_config=_cfg())


# --- optimizer: one group per kind, E2E-QP uses e2e_lr -----------------------------

def test_e2e_qp_optimizer_trains_only_the_scales_at_e2e_lr(tmp_path):
    qmodel = get_quant_model(_llama(), _cfg(phase="e2e_qp", target_modules=sorted(BLOCK_LINEARS)))
    trainer = _trainer(qmodel, tmp_path, e2e_lr=3e-4, quant_lr=7e-4)
    groups = trainer.create_optimizer().param_groups
    assert [g["lr"] for g in groups] == [3e-4]
    assert {id(p) for p in groups[0]["params"]} == {id(m.scale) for m in qmodel.quant_layers()}


def test_block_ap_hands_over_to_an_e2e_qp_optimizer(tmp_path):
    trainer = _trainer(_llama(), tmp_path, quant_config=_cfg(), e2e_lr=3e-4, quant_lr=7e-4)
    trainer.train()
    groups = trainer.optimizer.param_groups
    assert [g["initial_lr"] for g in groups] == [3e-4]
    assert {id(p) for p in groups[0]["params"]} == {id(m.scale) for m in trainer.model.quant_layers()}


# --- loss guard (peft PR #2571) ----------------------------------------------------

def test_zero_loss_is_caught(tmp_path, monkeypatch):
    import qpeft.trainer as T
    trainer = _trainer(_llama(), tmp_path, quant_config=_cfg(phase="e2e_qp"))
    monkeypatch.setattr(T.Trainer, "compute_loss",
                        lambda self, model, inputs, return_outputs=False, **kw: torch.zeros((), requires_grad=True))
    with pytest.raises(RuntimeError, match="training loss is 0.0"):
        trainer.compute_loss(trainer.model, _collate([_Data()[0]]))


# --- Block-AP calibration batches ---------------------------------------------------

def test_block_ap_batches_hold_exactly_block_ap_train_size_rows(tmp_path):
    """16 samples in loader batches of 3 (3,3,3,3,3,1), re-split to 2 rows each.
    The uneven tail of a loader batch must not be counted as a full Block-AP batch."""
    trainer = _trainer(_llama(), tmp_path, quant_config=_cfg(), per_device_train_batch_size=3,
                       block_ap_batch_size=2, block_ap_train_size=5)
    batches = trainer._block_ap_batches()
    assert sum(b["input_ids"].shape[0] for b in batches) == 5
    assert all(b["input_ids"].shape[0] <= 2 for b in batches)
    assert all("labels" not in b for b in batches)


# --- Trainer integration: checkpointing, KV cache, saving, imports ---------------------

def test_gradient_checkpointing_trains_both_phases(tmp_path):
    trainer = _trainer(_llama(), tmp_path, quant_config=_cfg(), gradient_checkpointing=True)
    trainer.train()
    assert trainer.model.base.is_gradient_checkpointing


def test_gradient_checkpointing_is_refused_for_a_model_without_it():
    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(128, 128)

        def forward(self, x):
            return self.q_proj(x)

    qmodel = get_quant_model(Plain(), _cfg(target_modules=["q_proj"]))
    with pytest.raises(ValueError, match="gradient checkpointing"):
        qmodel.gradient_checkpointing_enable()


def test_use_cache_goes_to_the_hf_model_not_the_qpeft_config(tmp_path):
    model = _llama()
    assert model.config.use_cache                                # HF default: KV cache on
    trainer = _trainer(model, tmp_path, quant_config=_cfg())     # TrainingArguments.use_cache = False
    assert trainer.model.base.config.use_cache is False
    assert "use_cache" not in vars(trainer.model.config)


def test_learning_rate_is_refused_with_a_pointer_to_e2e_lr(tmp_path):
    """QATTrainer sets its own per-group lrs (e2e_lr, weight_lr, quant_lr). A changed
    learning_rate would be silently ignored, so it is refused."""
    with pytest.raises(ValueError, match="e2e_lr"):
        _trainer(_llama(), tmp_path, quant_config=_cfg(), learning_rate=3e-4)


def test_evaluate_reports_the_loss(tmp_path):
    """Trainer finds the label names in model.forward's signature. QuantModel.forward is
    (*args, **kwargs), so they must come from the wrapped HF model, or eval has no loss."""
    trainer = _trainer(_llama(), tmp_path, quant_config=_cfg(phase="e2e_qp"))
    assert "eval_loss" in trainer.evaluate(eval_dataset=_Data(n=4))


def test_push_to_hub_is_refused_with_a_pointer_to_save_model(tmp_path):
    trainer = _trainer(_llama(), tmp_path, quant_config=_cfg())
    with pytest.raises(RuntimeError, match="save_model"):
        trainer.push_to_hub()
    with pytest.raises(RuntimeError, match="save_model"):
        trainer.save_model(_internal_call=True)                   # e.g. hyperparameter search
    assert not trainer.model.is_merged


def test_save_model_merges_everywhere_but_writes_only_on_the_main_process(tmp_path, monkeypatch):
    trainer = _trainer(_llama(), tmp_path, quant_config=_cfg())
    monkeypatch.setattr(type(trainer.args), "should_save", property(lambda self: False))
    out = tmp_path / "rank1"
    trainer.save_model(str(out))
    assert trainer.model.is_merged
    assert not out.exists() or not any(out.iterdir())


def test_star_import_works_without_transformers():
    import subprocess
    import sys
    code = "import sys; sys.modules['transformers'] = None; from qpeft import *; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_refuses_push_to_hub_argument(tmp_path):
    """Trainer.__init__ would create a Hub repo that save_model() never fills."""
    with pytest.raises(ValueError, match="push_to_hub"):
        _trainer(_llama(), tmp_path, quant_config=_cfg(), push_to_hub=True, hub_model_id="x/y")
