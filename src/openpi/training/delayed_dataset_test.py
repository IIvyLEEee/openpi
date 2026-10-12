import numpy as np
import pytest

from openpi import transforms
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared.normalize import NormStats
from openpi.training.delayed_dataset import DelayedObservationDataset


class StateTokenizer:
    def tokenize(self, prompt, state):
        return np.asarray(state), np.ones_like(state, dtype=bool)


def dataset():
    return [
        {
            "episode_index": i // 4,
            "frame_index": i % 4,
            "task_index": i // 4,
            "state": np.asarray([float(i)]),
            "actions": np.asarray([[float(i + 1)]]),
            "image": {"base": np.asarray([i])},
            "image_mask": {"base": np.True_},
            "prompt": "task",
        }
        for i in range(8)
    ]


def transform():
    return transforms.compose(
        [
            transforms.Normalize({"state": NormStats(mean=np.asarray([1.0]), std=np.asarray([2.0]))}),
            transforms.TokenizePrompt(StateTokenizer(), discrete_state_input=True),
            transforms.PadStatesAndActions(4),
        ]
    )


def test_old_prefix_current_normalized_state_and_targets(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda _: 2)
    result = DelayedObservationDataset(dataset(), transform(), 5)[6]
    assert int(result["prefix_age"]) == 2
    assert result["image"]["base"][0] == 4
    assert result["tokenized_prompt"][0] == pytest.approx(1.5, abs=1e-5)
    assert result["state"][0] == pytest.approx(2.5, abs=1e-5)
    assert result["state"].shape == (4,)
    assert result["actions"][0, 0] == 7


def test_episode_start_and_task_switch(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda _: 5)
    raw = dataset()
    delayed = DelayedObservationDataset(raw, transform(), 5)
    assert int(delayed[4]["prefix_age"]) == 0
    # At frame 1 the actual delay is 1, never the sampled 5.
    assert int(delayed[5]["prefix_age"]) == 1
    raw[6]["task_index"] = 9
    assert int(delayed[6]["prefix_age"]) == 0


def test_rejects_noncontiguous_metadata(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda _: 1)
    raw = dataset()
    raw[5]["episode_index"] = 0
    with pytest.raises(ValueError, match="episode"):
        DelayedObservationDataset(raw, transform(), 5)[6]


def test_strict_tokenizer_does_not_truncate():
    class Pieces:
        def encode(self, *_args, **_kwargs):
            return list(range(20))

    tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
    tokenizer._max_len = 4  # noqa: SLF001
    tokenizer._truncate = False  # noqa: SLF001
    tokenizer._tokenizer = Pieces()  # noqa: SLF001
    with pytest.raises(ValueError, match="max_token_len"):
        tokenizer.tokenize("task", np.zeros(8))
