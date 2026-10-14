"""Two independent workers with observation-step alignment and generation-safe handoff."""

import copy
import dataclasses
import threading
import time
from typing import Any, Protocol

import numpy as np


class DualSystemBackend(Protocol):
    max_prefix_age: int

    def encode(self, observation: dict) -> Any: ...

    def sample(self, observation: dict, condition: Any, prefix_age: int) -> np.ndarray: ...


class PrefixExpiredError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ObservationSnapshot:
    observation: dict
    step: int
    captured_at: float
    generation: int


@dataclasses.dataclass(frozen=True)
class PrefixPacket:
    condition: Any
    observation_step: int
    captured_at: float
    cache_id: int
    generation: int
    task: str
    encode_seconds: float


@dataclasses.dataclass(frozen=True)
class ActionPrediction:
    actions: np.ndarray
    observation_step: int
    captured_at: float
    generation: int
    cache_id: int
    prefix_age: int
    sample_seconds: float


class AsyncDualSystemRunner:
    """System 1 samples whole chunks; System 2 refreshes a prefix in parallel.

    The caller advances the robot at its fixed control period and publishes each
    observation. A returning chunk is indexed from its observation step, not from
    its arrival time. Prefix staleness uses environment steps, matching training.
    """

    def __init__(self, backend: DualSystemBackend, *, execution_horizon: int):
        if execution_horizon < 1 or backend.max_prefix_age < 0:
            raise ValueError("Invalid execution horizon or maximum prefix age")
        self.backend = backend
        self.execution_horizon = execution_horizon
        self._cv = threading.Condition()
        self._generation = 0
        self._task = ""
        self._latest = None
        self._prefix = None
        self._prediction = None
        self._cache_id = 0
        self._error = None
        self._stop = False
        self._workers = [
            threading.Thread(target=self._vlm_loop, name="pi05-system2", daemon=True),
            threading.Thread(target=self._expert_loop, name="pi05-system1", daemon=True),
        ]
        for worker in self._workers:
            worker.start()

    def reset(self, task: str):
        with self._cv:
            self._generation += 1
            self._task = task
            self._latest = self._prefix = self._prediction = self._error = None
            self._cv.notify_all()

    def publish(self, observation: dict, *, step: int, captured_at: float | None = None):
        # Own the arrays: environments commonly reuse their observation buffers.
        observation = copy.deepcopy(observation)
        with self._cv:
            self._raise_worker_error()
            if self._stop:
                raise RuntimeError("Runner is closed")
            if step < 0 or (self._latest is not None and step <= self._latest.step):
                raise ValueError("Observation steps must increase; call reset between episodes")
            if observation.get("prompt", self._task) != self._task:
                raise ValueError("Call reset before changing the task")
            self._latest = ObservationSnapshot(
                observation, step, time.monotonic() if captured_at is None else captured_at, self._generation
            )
            self._cv.notify_all()

    def wait_ready(self, timeout: float = 30.0):
        """Prime both systems once before advancing the environment."""
        with self._cv:
            ready = self._cv.wait_for(lambda: self._prediction is not None or self._error or self._stop, timeout)
            self._raise_worker_error()
            if not ready or self._prediction is None:
                raise TimeoutError("No initial action prediction available")

    def action_at(self, step: int) -> tuple[np.ndarray | None, dict]:
        with self._cv:
            self._raise_worker_error()
            prefix, prediction = self._prefix, self._prediction
            prefix_expired = prefix is not None and step - prefix.observation_step > self.backend.max_prefix_age
            if prediction is None:
                if prefix_expired:
                    raise PrefixExpiredError("Cached prefix exceeds the training age range; wait for a fresh prefix")
                return None, {}
            # Training bounds the prefix/state gap at sampling. A valid chunk also
            # predicts future actions; executing those positions must not reapply
            # the sampling-age bound as a shorter action lifetime.
            used_prefix_step = prediction.observation_step - prediction.prefix_age
            offset = step - prediction.observation_step
            info = {
                "cache_id_used": prediction.cache_id,
                "state_step": prediction.observation_step,
                "action_offset": offset,
                "prefix_age_at_sample": prediction.prefix_age,
                "prefix_age_at_execute": step - used_prefix_step,
                "sample_ms": prediction.sample_seconds * 1000,
                "state_to_execute_ms": (time.monotonic() - prediction.captured_at) * 1000,
                "encode_ms": prefix.encode_seconds * 1000 if prefix is not None else None,
            }
            if offset < 0 or offset >= min(len(prediction.actions), self.execution_horizon):
                if prefix_expired:
                    raise PrefixExpiredError("Cached prefix exceeds the training age range; wait for a fresh prefix")
                return None, info
            return prediction.actions[offset].copy(), info

    def _raise_worker_error(self):
        if self._error is not None:
            raise RuntimeError("Asynchronous policy worker failed") from self._error

    def _fail(self, error, generation):
        with self._cv:
            if generation == self._generation and not self._stop:
                self._error = error
                self._cv.notify_all()

    def _vlm_loop(self):
        seen = None
        while True:
            with self._cv:
                self._cv.wait_for(
                    lambda seen=seen: (
                        self._stop
                        or (
                            self._error is None
                            and self._latest is not None
                            and (self._latest.generation, self._latest.step) != seen
                        )
                    )
                )
                if self._stop:
                    return
                snapshot, task = self._latest, self._task
                seen = (snapshot.generation, snapshot.step)
            start = time.monotonic()
            try:
                condition = self.backend.encode(snapshot.observation)
            except Exception as error:
                self._fail(error, snapshot.generation)
                continue
            with self._cv:
                if snapshot.generation == self._generation and not self._stop:
                    self._cache_id += 1
                    self._prefix = PrefixPacket(
                        condition,
                        snapshot.step,
                        snapshot.captured_at,
                        self._cache_id,
                        snapshot.generation,
                        task,
                        time.monotonic() - start,
                    )
                    self._cv.notify_all()

    def _expert_loop(self):
        seen = None
        while True:
            with self._cv:
                self._cv.wait_for(
                    lambda seen=seen: (
                        self._stop
                        or (
                            self._error is None
                            and self._latest is not None
                            and self._prefix is not None
                            and (self._latest.generation, self._latest.step, self._prefix.cache_id) != seen
                            and 0 <= self._latest.step - self._prefix.observation_step <= self.backend.max_prefix_age
                        )
                    )
                )
                if self._stop:
                    return
                snapshot, prefix = self._latest, self._prefix
                seen = (snapshot.generation, snapshot.step, prefix.cache_id)
            age = snapshot.step - prefix.observation_step
            start = time.monotonic()
            try:
                actions = self.backend.sample(snapshot.observation, prefix.condition, age)
                actions = np.array(actions, copy=True)
                if actions.ndim != 2 or not np.isfinite(actions).all():
                    raise ValueError("Expert must return finite [horizon, action_dim] actions")
            except Exception as error:
                self._fail(error, snapshot.generation)
                continue
            with self._cv:
                if snapshot.generation == self._generation and not self._stop:
                    self._prediction = ActionPrediction(
                        actions,
                        snapshot.step,
                        snapshot.captured_at,
                        snapshot.generation,
                        prefix.cache_id,
                        age,
                        time.monotonic() - start,
                    )
                    self._cv.notify_all()

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        for worker in self._workers:
            worker.join(timeout=30)
            if worker.is_alive():
                raise TimeoutError(f"{worker.name} did not finish its in-flight inference")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
