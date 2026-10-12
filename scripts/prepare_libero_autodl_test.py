import numpy as np

from openpi.shared.normalize import RunningStats
from scripts.prepare_libero_autodl import accumulate_episode


def test_stats_preserve_future_action_padding_and_episode_boundaries():
    stats = {key: RunningStats() for key in ("state", "actions")}
    accumulate_episode(stats, np.asarray([[0.0], [2.0]]), np.asarray([[1.0], [2.0]]), 3)
    accumulate_episode(stats, np.asarray([[4.0]]), np.asarray([[10.0]]), 3)
    np.testing.assert_allclose(stats["state"].get_statistics().mean, [2.0])
    # Chunks are [1,2,2], [2,2,2], [10,10,10]; no action crosses an episode boundary.
    np.testing.assert_allclose(stats["actions"].get_statistics().mean, [41.0 / 9.0])
