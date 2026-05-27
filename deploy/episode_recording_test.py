import numpy as np

from deploy.umi.real_world.episode_recording import recorded_action_step_count
from deploy.umi.real_world.episode_recording import should_finalize_episode


def test_recorded_action_step_count_returns_zero_without_actions():
    assert recorded_action_step_count(np.array([], dtype=np.float64), end_time=10.0) == 0


def test_recorded_action_step_count_counts_actions_at_or_before_end_time():
    timestamps = np.array([10.0, 10.1, 10.2, 10.3], dtype=np.float64)

    assert recorded_action_step_count(timestamps, end_time=10.2) == 3


def test_should_finalize_episode_is_false_before_start_episode_succeeds():
    assert not should_finalize_episode(obs_accumulator=None, action_accumulator=None)
