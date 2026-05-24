from collections.abc import Callable
from concurrent import futures
import copy
import dataclasses
import time
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class AsyncInferenceRequest:
    request_id: int
    wall_time: float
    submitted_monotonic: float
    obs_timestamp: float


@dataclasses.dataclass(frozen=True)
class AsyncInferenceResult:
    request_id: int
    wall_time: float
    obs_timestamp: float
    policy_call_ms: float
    response: dict[str, Any]


def _copy_policy_obs(policy_obs: dict[str, Any]) -> dict[str, Any]:
    copied = {}
    for key, value in policy_obs.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = copy.deepcopy(value)
    return copied


def apply_future_state(policy_obs: dict[str, Any], future_state: np.ndarray) -> dict[str, Any]:
    updated = _copy_policy_obs(policy_obs)
    updated["observation/state"] = np.asarray(future_state, dtype=np.float32).copy()
    return updated


class AsyncInferenceWorker:
    def __init__(self, infer_fn: Callable[[dict[str, Any]], dict[str, Any]]):
        self._infer_fn = infer_fn
        self._executor = futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-async-infer")
        self._pending: futures.Future[AsyncInferenceResult] | None = None
        self._pending_request: AsyncInferenceRequest | None = None
        self._next_request_id = 0

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def done(self) -> bool:
        return self._pending is not None and self._pending.done()

    def submit(self, policy_obs: dict[str, Any], *, obs_timestamp: float) -> AsyncInferenceRequest:
        if self._pending is not None:
            raise RuntimeError("Cannot submit async inference while a request is pending.")

        request = AsyncInferenceRequest(
            request_id=self._next_request_id,
            wall_time=time.time(),
            submitted_monotonic=time.perf_counter(),
            obs_timestamp=float(obs_timestamp),
        )
        self._next_request_id += 1
        obs_copy = _copy_policy_obs(policy_obs)
        self._pending_request = request
        self._pending = self._executor.submit(self._run, request, obs_copy)
        return request

    def result(self, timeout: float | None = None) -> AsyncInferenceResult:
        if self._pending is None:
            raise RuntimeError("No async inference request is pending.")
        pending = self._pending
        try:
            return pending.result(timeout=timeout)
        finally:
            self._pending = None
            self._pending_request = None

    def close(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def _run(self, request: AsyncInferenceRequest, policy_obs: dict[str, Any]) -> AsyncInferenceResult:
        infer_start = time.perf_counter()
        response = self._infer_fn(policy_obs)
        policy_call_ms = 1000 * (time.perf_counter() - infer_start)
        return AsyncInferenceResult(
            request_id=request.request_id,
            wall_time=request.wall_time,
            obs_timestamp=request.obs_timestamp,
            policy_call_ms=policy_call_ms,
            response=response,
        )
