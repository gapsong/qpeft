"""qpeft.hf_trainer on a plain Hugging Face Trainer, without QATTrainer: the pieces an
integration (axolotl) shares with QATTrainer. Needs transformers; tiny random Llama, no download."""
from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

import torch                                                    # noqa: E402
from torch.utils.data import Dataset                            # noqa: E402
from transformers import Trainer, TrainingArguments             # noqa: E402

from qpeft.hf_trainer import FirstStepMustMoveParams, block_ap_batches   # noqa: E402


class _Data(Dataset):
    def __init__(self, n=16, seq=32, vocab=320):
        g = torch.Generator().manual_seed(0)
        self.ids = torch.randint(0, vocab, (n, seq), generator=g)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": torch.ones_like(self.ids[i]),
                "labels": self.ids[i], "label": torch.tensor(0), "label_ids": self.ids[i]}


def _collate(rows):
    return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}


def _llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=4, vocab_size=320, max_position_embeddings=64))


def _trainer(tmp_path, model=None, data=None, **kw):
    args = dict(output_dir=str(tmp_path), per_device_train_batch_size=3, num_train_epochs=1,
                report_to=[], dataloader_pin_memory=False, save_strategy="no", warmup_steps=0)
    args.update(kw)
    return Trainer(model=model if model is not None else _llama(), args=TrainingArguments(**args),
                   train_dataset=data if data is not None else _Data(), data_collator=_collate)


# --- block_ap_batches --------------------------------------------------------------

def test_block_ap_batches_hold_exactly_train_size_rows(tmp_path):
    """16 samples in loader batches of 3 (3,3,3,3,3,1), re-split to 2 rows each.
    The uneven tail of a loader batch must not be counted as a full Block-AP batch."""
    batches = block_ap_batches(_trainer(tmp_path), train_size=5, seqlen=32, batch_size=2)
    assert [b["input_ids"].shape[0] for b in batches] == [2, 1, 2]


def test_block_ap_batches_take_all_rows_when_the_data_is_short(tmp_path):
    batches = block_ap_batches(_trainer(tmp_path), train_size=100, seqlen=32, batch_size=2)
    assert sum(b["input_ids"].shape[0] for b in batches) == 16


def test_block_ap_batches_drop_labels_and_cut_sequences(tmp_path):
    batches = block_ap_batches(_trainer(tmp_path), train_size=4, seqlen=8, batch_size=2)
    assert all(set(b) == {"input_ids", "attention_mask"} for b in batches)
    assert all(v.shape[1] == 8 for b in batches for v in b.values())


def test_block_ap_batches_are_on_the_trainers_device(tmp_path):
    trainer = _trainer(tmp_path)
    batches = block_ap_batches(trainer, train_size=4, seqlen=32, batch_size=2)
    assert all(v.device.type == trainer.args.device.type for b in batches for v in b.values())


def test_block_ap_batches_hold_the_training_rows(tmp_path):
    """The Trainer shuffles, so compare sets."""
    batches = block_ap_batches(_trainer(tmp_path), train_size=16, seqlen=32, batch_size=2)
    got = {tuple(row.tolist()) for b in batches for row in b["input_ids"].cpu()}
    assert got == {tuple(row.tolist()) for row in _Data().ids}


# --- FirstStepMustMoveParams: the four events, called directly ---------------------

def _step(callback, model, optimizer, move):
    callback.on_step_begin(None, None, None, optimizer=optimizer)
    if move:
        with torch.no_grad():
            model.weight.add_(1.0)
    callback.on_step_end(None, None, None, model=model, optimizer=optimizer)


def _optimizer(lr, scaler=None, skipped=False):
    return SimpleNamespace(param_groups=[{"lr": lr}], scaler=scaler, step_was_skipped=skipped)


class _GradScaler:
    def __init__(self):
        self.scale = 65536.0

    def get_scale(self):
        return self.scale


def test_first_step_that_moves_passes():
    model, callback = torch.nn.Linear(2, 2), FirstStepMustMoveParams()
    callback.on_train_begin(None, None, None, model=model)
    _step(callback, model, _optimizer(1e-3), move=True)
    callback.on_train_end(None, None, None)


def test_first_step_that_moves_nothing_is_refused():
    model, callback = torch.nn.Linear(2, 2), FirstStepMustMoveParams()
    callback.on_train_begin(None, None, None, model=model)
    with pytest.raises(RuntimeError, match="no trainable parameter changed"):
        _step(callback, model, _optimizer(1e-3), move=False)


def test_frozen_parameters_do_not_count_as_moved():
    model, callback = torch.nn.Linear(2, 2), FirstStepMustMoveParams()
    model.weight.requires_grad_(False)
    callback.on_train_begin(None, None, None, model=model)
    with pytest.raises(RuntimeError, match="no trainable parameter changed"):
        _step(callback, model, _optimizer(1e-3), move=True)    # moves only the frozen weight


def test_warmup_steps_at_lr_zero_are_not_checked():
    model, callback = torch.nn.Linear(2, 2), FirstStepMustMoveParams()
    callback.on_train_begin(None, None, None, model=model)
    _step(callback, model, _optimizer(0.0), move=False)
    _step(callback, model, _optimizer(1e-3), move=True)
    callback.on_train_end(None, None, None)


def test_a_step_skipped_by_the_grad_scaler_is_not_checked():
    model, callback = torch.nn.Linear(2, 2), FirstStepMustMoveParams()
    scaler = _GradScaler()
    optimizer = _optimizer(1e-3, scaler)
    callback.on_train_begin(None, None, None, model=model)
    callback.on_step_begin(None, None, None, optimizer=optimizer)
    scaler.scale /= 2                                           # overflow: the step is skipped
    callback.on_step_end(None, None, None, model=model, optimizer=optimizer)
    _step(callback, model, optimizer, move=True)
    callback.on_train_end(None, None, None)


def test_a_step_skipped_without_a_grad_scaler_is_not_checked():
    """DeepSpeed's optimizer wrapper has no GradScaler and reports the skip in step_was_skipped."""
    model, callback = torch.nn.Linear(2, 2), FirstStepMustMoveParams()
    callback.on_train_begin(None, None, None, model=model)
    _step(callback, model, _optimizer(1e-3, skipped=True), move=False)
    _step(callback, model, _optimizer(1e-3), move=True)
    callback.on_train_end(None, None, None)


def test_a_step_the_grad_scaler_did_not_skip_is_checked():
    model, callback = torch.nn.Linear(2, 2), FirstStepMustMoveParams()
    callback.on_train_begin(None, None, None, model=model)
    with pytest.raises(RuntimeError, match="no trainable parameter changed"):
        _step(callback, model, _optimizer(1e-3, _GradScaler()), move=False)


def test_only_the_first_real_step_is_checked():
    model, callback = torch.nn.Linear(2, 2), FirstStepMustMoveParams()
    callback.on_train_begin(None, None, None, model=model)
    _step(callback, model, _optimizer(1e-3), move=True)
    _step(callback, model, _optimizer(1e-3), move=False)       # a later step may move nothing
    callback.on_train_end(None, None, None)


def test_training_that_ends_without_a_step_at_lr_above_zero_is_refused():
    model, callback = torch.nn.Linear(2, 2), FirstStepMustMoveParams()
    callback.on_train_begin(None, None, None, model=model)
    _step(callback, model, _optimizer(0.0), move=False)
    with pytest.raises(RuntimeError, match="without an optimizer step at lr > 0"):
        callback.on_train_end(None, None, None)


# --- FirstStepMustMoveParams inside a real Trainer loop -----------------------------

def test_a_real_trainer_run_that_learns_passes(tmp_path):
    trainer = _trainer(tmp_path, learning_rate=1e-3, max_steps=2)
    trainer.add_callback(FirstStepMustMoveParams())
    trainer.train()


def test_a_real_trainer_run_at_lr_zero_is_stopped(tmp_path):
    trainer = _trainer(tmp_path, learning_rate=0.0, max_steps=2)
    trainer.add_callback(FirstStepMustMoveParams())
    with pytest.raises(RuntimeError, match="without an optimizer step at lr > 0"):
        trainer.train()


fp16_amp = pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()),
                              reason="fp16 AMP needs CUDA or MPS")


def _overflow_the_first_backward(model):
    """An inf gradient in the first backward: the fp16 GradScaler skips that optimizer step."""
    calls = {"n": 0}

    def hook(grad):
        calls["n"] += 1
        return torch.full_like(grad, float("inf")) if calls["n"] == 1 else grad

    next(p for p in model.parameters() if p.requires_grad).register_hook(hook)


@fp16_amp
@pytest.mark.parametrize("optim", ["adamw_torch_fused", "adamw_torch"])
def test_a_real_fp16_step_skipped_by_the_grad_scaler_is_not_a_dead_run(tmp_path, optim):
    """A fused AdamW never sets accelerate's step_was_skipped, so the skip must be seen another way."""
    trainer = _trainer(tmp_path, learning_rate=1e-3, max_steps=3, fp16=True, optim=optim)
    _overflow_the_first_backward(trainer.model)
    trainer.add_callback(FirstStepMustMoveParams())
    trainer.train()


def test_a_real_trainer_run_that_moves_nothing_is_stopped(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, learning_rate=1e-3, max_steps=2)
    trainer.add_callback(FirstStepMustMoveParams())
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, *a, **kw: None)
    with pytest.raises(RuntimeError, match="no trainable parameter changed"):
        trainer.train()
