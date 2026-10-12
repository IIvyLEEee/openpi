import dataclasses

import filelock
import pytest
import torch

from openpi.training.config import DataConfig
from openpi.training.config import get_config
from scripts import train_pytorch
from scripts.train_pytorch import load_checkpoint
from scripts.train_pytorch import save_checkpoint


def test_last_completed_step_is_saved_and_resumed(tmp_path):
    config = dataclasses.replace(
        get_config("pi05_libero_async"),
        exp_name="checkpoint_test",
        checkpoint_base_dir=str(tmp_path),
        num_train_steps=2,
        save_interval=1000,
        wandb_enabled=False,
    )
    config.checkpoint_dir.mkdir(parents=True)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(2):
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    save_checkpoint(model, optimizer, 1, config, is_main=True, data_config=DataConfig())
    assert not (config.checkpoint_dir / "1").exists()
    save_checkpoint(model, optimizer, 2, config, is_main=True, data_config=DataConfig())
    assert (config.checkpoint_dir / "2" / "metadata.pt").exists()
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.1)
    step = load_checkpoint(restored, restored_optimizer, config.checkpoint_dir, torch.device("cpu"))
    assert step == 2
    for left, right in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right)
    assert restored_optimizer.param_groups[0]["lr"] == 0.01
    assert all(state["step"].item() == 2 for state in restored_optimizer.state.values())


def test_checkpoint_retention_preserves_periodic_and_incomplete_directories(tmp_path):
    config = dataclasses.replace(
        get_config("pi05_libero_async"),
        exp_name="retention_test",
        checkpoint_base_dir=str(tmp_path),
        num_train_steps=5,
        save_interval=1,
        keep_period=2,
        pytorch_keep_last_checkpoints=1,
        wandb_enabled=False,
    )
    config.checkpoint_dir.mkdir(parents=True)
    incomplete = config.checkpoint_dir / "0"
    incomplete.mkdir()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    for step in range(1, 6):
        save_checkpoint(model, optimizer, step, config, is_main=True, data_config=DataConfig())
    assert {path.name for path in config.checkpoint_dir.iterdir()} == {"0", "2", "4", "5"}
    assert (config.checkpoint_dir / "5" / "metadata.pt").is_file()
    with pytest.raises(ValueError, match="must be positive"):
        dataclasses.replace(config, pytorch_keep_last_checkpoints=0)


def test_shared_save_lock_is_held_and_released_on_error(tmp_path, monkeypatch):
    config = dataclasses.replace(
        get_config("pi05_libero_async"),
        pytorch_checkpoint_lock_path=str(tmp_path / "shared" / ".save.lock"),
        num_train_steps=1,
    )

    def failing_writer(*_args):
        with pytest.raises(filelock.Timeout):
            filelock.FileLock(config.pytorch_checkpoint_lock_path).acquire(timeout=0)
        raise RuntimeError("write failed")

    monkeypatch.setattr(train_pytorch, "_save_checkpoint", failing_writer)
    with pytest.raises(RuntimeError, match="write failed"):
        save_checkpoint(None, None, 1, config, is_main=True, data_config=DataConfig())
    with filelock.FileLock(config.pytorch_checkpoint_lock_path, timeout=0):
        pass
