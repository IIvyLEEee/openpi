# ruff: noqa: E402, SLF001
import pathlib
import sys

import numpy as np

_OPENPI_CLIENT_SRC = pathlib.Path(__file__).resolve().parents[1] / "packages" / "openpi-client" / "src"
if str(_OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

from deploy import inference_real


def test_future_action_schedule_can_start_at_async_chunk_boundary(monkeypatch):
    monkeypatch.setattr(inference_real.time, "time", lambda: 100.0)
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)

    scheduled_actions, timestamps = inference_real._future_action_schedule(
        actions=actions,
        obs_timestamp=90.0,
        eval_start_time=0.0,
        frequency=10.0,
        action_exec_latency=0.01,
        start_timestamp=101.0,
    )

    np.testing.assert_array_equal(scheduled_actions, actions)
    np.testing.assert_allclose(timestamps, np.array([101.0, 101.1, 101.2], dtype=np.float64))
