import numpy as np

from deploy.umi.real_world.episode_recording import recorded_action_step_count


def test_recorded_action_step_count_returns_zero_without_actions():
    assert recorded_action_step_count(np.array([], dtype=np.float64), end_time=10.0) == 0


def test_recorded_action_step_count_counts_actions_at_or_before_end_time():
    timestamps = np.array([10.0, 10.1, 10.2, 10.3], dtype=np.float64)

    assert recorded_action_step_count(timestamps, end_time=10.2) == 3
