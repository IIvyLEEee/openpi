"""Exercise the actual pinned LeRobot dataset and openpi transform/collation path."""

import dataclasses

from lerobot.common.datasets import lerobot_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.models.model import ModelType
from openpi.models.pi0_config import Pi0Config
from openpi.policies.libero_policy import LiberoInputs
from openpi.shared.normalize import NormStats
from openpi.training.config import DataConfig
from openpi.training.data_loader import DataLoaderImpl
from openpi.training.data_loader import TorchDataLoader
from openpi.training.data_loader import create_torch_dataset
from openpi.training.data_loader import transform_dataset


class IntegerStateTokenizer:
    def tokenize(self, _prompt, state):
        return np.asarray([round(float(state[0]))], dtype=np.int32), np.asarray([True])


def test_lerobot_delay_pipeline_and_torch_batch(tmp_path, monkeypatch):
    root = tmp_path / "local" / "async_test"
    features = {
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        "image": {"dtype": "image", "shape": (16, 16, 3), "names": ["h", "w", "c"]},
        "wrist_image": {"dtype": "image", "shape": (16, 16, 3), "names": ["h", "w", "c"]},
    }
    writer = LeRobotDataset.create("local/async_test", fps=10, root=root, robot_type="panda", features=features)
    for episode in range(2):
        for frame in range(4):
            value = episode * 4 + frame
            writer.add_frame(
                {
                    "state": np.full(8, value, dtype=np.float32),
                    "actions": np.full(7, value + 1, dtype=np.float32),
                    "image": np.full((16, 16, 3), value * 10, dtype=np.uint8),
                    "wrist_image": np.full((16, 16, 3), value * 10, dtype=np.uint8),
                    "task": "task",
                }
            )
        writer.save_episode()
    dataset = LeRobotDataset("local/async_test", root=root, delta_timestamps={"actions": [0.0, 0.1, 0.2]})
    state_stats = NormStats(mean=np.zeros(8), std=np.ones(8))
    action_stats = NormStats(mean=np.zeros(7), std=np.ones(7))
    config = DataConfig(
        repo_id="local/async_test",
        prefix_delay_max=2,
        norm_stats={"state": state_stats, "actions": action_stats},
        repack_transforms=transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/state": "state",
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        ),
        data_transforms=transforms.Group(inputs=[LiberoInputs(ModelType.PI05)]),
        model_transforms=transforms.Group(
            inputs=[
                transforms.TokenizePrompt(IntegerStateTokenizer(), discrete_state_input=True),
                transforms.PadStatesAndActions(32),
            ]
        ),
    )
    monkeypatch.setattr(np.random, "randint", lambda _: 2)
    delayed = transform_dataset(dataset, config)
    item = delayed[6]
    assert int(item["prefix_age"]) == 2
    assert item["tokenized_prompt"][0] == 4
    np.testing.assert_allclose(item["state"][:8], 6, atol=1e-5)
    expected = np.broadcast_to(np.asarray([[7], [8], [8]]), (3, 7))
    np.testing.assert_allclose(item["actions"][:, :7], expected, atol=1e-5)
    assert int(delayed[4]["prefix_age"]) == 0
    loader = DataLoaderImpl(config, TorchDataLoader(delayed, 2, num_batches=1, framework="pytorch"))
    observation, actions = next(iter(loader))
    assert observation.state.shape == (2, 32)
    assert observation.prefix_age.dtype == torch.int32
    assert observation.tokenized_prompt.dtype == torch.int32
    assert actions.shape == (2, 3, 32)
    assert transform_dataset(dataset, dataclasses.replace(config, prefix_delay_max=0))[6].get("prefix_age") is None
    monkeypatch.setattr(lerobot_dataset, "HF_LEROBOT_HOME", tmp_path)
    with pytest.raises(ValueError, match="episode prefix"):
        create_torch_dataset(dataclasses.replace(config, episodes=[1]), 3, Pi0Config(pi05=True))
    selected = create_torch_dataset(
        dataclasses.replace(config, episodes=[0]), 3, Pi0Config(pi05=True, action_horizon=3)
    )
    assert len(selected) == 4
    assert int(selected[0]["episode_index"]) == 0
    selected_delayed = transform_dataset(selected, config)[2]
    assert int(selected_delayed["prefix_age"]) == 2
    assert selected_delayed["tokenized_prompt"][0] == 0
    np.testing.assert_allclose(selected_delayed["state"][:8], 2, atol=1e-5)
