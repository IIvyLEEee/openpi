import threading

import numpy as np
import pytest

from deploy.async_inference import AsyncInferenceWorker
from deploy.async_inference import apply_future_state


def _policy_obs() -> dict:
    return {
        "observation/image": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": "move the block",
    }


def test_async_inference_worker_runs_request_in_background_and_reports_latency():
    inference_started = threading.Event()
    release_inference = threading.Event()

    def infer_fn(obs: dict) -> dict:
        inference_started.set()
        assert obs["prompt"] == "move the block"
        assert release_inference.wait(timeout=2.0)
        return {"actions": np.ones((2, 7), dtype=np.float32)}

    worker = AsyncInferenceWorker(infer_fn)
    try:
        request = worker.submit(_policy_obs(), obs_timestamp=10.5)

        assert request.request_id == 0
        assert inference_started.wait(timeout=1.0)
        assert worker.has_pending
        assert not worker.done()

        release_inference.set()
        result = worker.result(timeout=1.0)
    finally:
        worker.close()

    assert result.request_id == request.request_id
    assert result.obs_timestamp == 10.5
    assert result.policy_call_ms >= 0.0
    np.testing.assert_array_equal(result.response["actions"], np.ones((2, 7), dtype=np.float32))
    assert not worker.has_pending


def test_async_inference_worker_rejects_second_pending_request():
    release_inference = threading.Event()

    def infer_fn(obs: dict) -> dict:
        assert release_inference.wait(timeout=2.0)
        return {"actions": np.zeros((1, 7), dtype=np.float32)}

    worker = AsyncInferenceWorker(infer_fn)
    try:
        worker.submit(_policy_obs(), obs_timestamp=1.0)
        with pytest.raises(RuntimeError, match="pending"):
            worker.submit(_policy_obs(), obs_timestamp=2.0)

        release_inference.set()
        worker.result(timeout=1.0)
    finally:
        worker.close()


def test_apply_future_state_copies_observation_and_does_not_mutate_input():
    obs = _policy_obs()
    future_state = np.arange(7, dtype=np.float32)

    updated = apply_future_state(obs, future_state)

    np.testing.assert_array_equal(updated["observation/state"], future_state)
    np.testing.assert_array_equal(obs["observation/state"], np.zeros(7, dtype=np.float32))
    assert updated is not obs
