# ruff: noqa: E402, SLF001
import pathlib
import sys

import numpy as np
import pytest

_OPENPI_CLIENT_SRC = pathlib.Path(__file__).resolve().parents[1] / "packages" / "openpi-client" / "src"
if str(_OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

from deploy import inference_real


def test_record_episode_is_enabled_by_default():
    assert inference_real.Args().record_episode is True


def test_inference_latency_scale_defaults_to_no_slowdown():
    assert inference_real.Args().inference_latency_scale == 1.0


def test_validate_runtime_args_rejects_latency_scale_below_one():
    with pytest.raises(ValueError, match="inference-latency-scale"):
        inference_real._validate_runtime_args(inference_real.Args(inference_latency_scale=0.5))


def test_validate_runtime_args_rejects_nonfinite_latency_scale():
    with pytest.raises(ValueError, match="inference-latency-scale"):
        inference_real._validate_runtime_args(inference_real.Args(inference_latency_scale=float("inf")))


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


def test_strict_sync_schedule_uses_first_actions_from_inference_return_time(monkeypatch):
    monkeypatch.setattr(inference_real.time, "time", lambda: 200.0)
    actions = np.arange(35, dtype=np.float32).reshape(5, 7)
    args = inference_real.Args(
        frequency=10.0,
        steps_per_inference=3,
        strict_sync=True,
        strict_sync_start_delay=0.25,
    )

    scheduled_actions, timestamps, action_indices = inference_real._schedule_actions_for_execution(
        actions=actions,
        obs_timestamp=100.0,
        eval_start_time=0.0,
        inference_return_timestamp=200.0,
        args=args,
    )

    np.testing.assert_array_equal(scheduled_actions, actions[:3])
    np.testing.assert_allclose(timestamps, np.array([200.25, 200.35, 200.45], dtype=np.float64))
    np.testing.assert_array_equal(action_indices, np.array([0, 1, 2]))


def test_strict_sync_schedule_uses_max_scheduled_actions_when_set():
    actions = np.arange(35, dtype=np.float32).reshape(5, 7)
    args = inference_real.Args(
        frequency=20.0,
        steps_per_inference=4,
        max_scheduled_actions=2,
        strict_sync=True,
        strict_sync_start_delay=0.1,
    )

    scheduled_actions, timestamps, action_indices = inference_real._schedule_actions_for_execution(
        actions=actions,
        obs_timestamp=100.0,
        eval_start_time=0.0,
        inference_return_timestamp=200.0,
        args=args,
    )

    np.testing.assert_array_equal(scheduled_actions, actions[:2])
    np.testing.assert_allclose(timestamps, np.array([200.1, 200.15], dtype=np.float64))
    np.testing.assert_array_equal(action_indices, np.array([0, 1]))


def test_strict_sync_waits_until_rescheduled_chunk_finishes():
    timestamps = np.array([200.25, 200.35, 200.45], dtype=np.float64)

    wait_target = inference_real._next_cycle_wait_monotonic(
        loop_start_monotonic=10.0,
        eval_start_time=100.0,
        iter_idx=3,
        dt=0.1,
        timestamps=timestamps,
        strict_sync=True,
    )

    assert wait_target == pytest.approx(110.55)


def test_validate_runtime_args_rejects_strict_sync_with_async_inference():
    with pytest.raises(ValueError, match="strict-sync"):
        inference_real._validate_runtime_args(inference_real.Args(strict_sync=True, async_inference=True))


def test_validate_runtime_args_rejects_negative_strict_sync_start_delay():
    with pytest.raises(ValueError, match="strict-sync-start-delay"):
        inference_real._validate_runtime_args(inference_real.Args(strict_sync=True, strict_sync_start_delay=-0.1))
