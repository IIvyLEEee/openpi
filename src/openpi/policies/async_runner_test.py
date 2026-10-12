import threading

import numpy as np
import pytest

from openpi.policies.async_runner import AsyncDualSystemRunner
from openpi.policies.async_runner import PrefixExpiredError


class Backend:
    max_prefix_age = 5

    def __init__(self):
        self.release = threading.Event()
        self.blocked = threading.Event()
        self.sampled = threading.Event()
        self.encodes = 0
        self.samples = []

    def encode(self, observation):
        self.encodes += 1
        if self.encodes == 2:
            self.blocked.set()
            if not self.release.wait(5):
                raise TimeoutError("test gate")
        return observation["prompt"], observation["state"]

    def sample(self, observation, condition, prefix_age):
        assert condition[0] == observation["prompt"]
        self.samples.append((observation["state"], condition[1], prefix_age))
        if observation["state"] == 1:
            self.sampled.set()
        return np.arange(observation["state"], observation["state"] + 4, dtype=float)[:, None]


def test_expert_continues_with_fresh_state_while_vlm_blocked():
    backend = Backend()
    with AsyncDualSystemRunner(backend, execution_horizon=4) as runner:
        try:
            runner.reset("task")
            runner.publish({"prompt": "task", "state": 0}, step=0)
            runner.wait_ready(2)
            runner.publish({"prompt": "task", "state": 1}, step=1)
            assert backend.blocked.wait(2)
            assert backend.sampled.wait(2)
            # Synchronize publication of the completed prediction, not only backend entry.
            with runner._cv:  # noqa: SLF001
                assert runner._cv.wait_for(lambda: runner._prediction.observation_step == 1, timeout=2)  # noqa: SLF001
            assert (1, 0, 1) in backend.samples
            action, info = runner.action_at(2)
            assert action[0] == 2  # offset 1, not the chunk's first action (1).
            assert info["action_offset"] == 1
            with pytest.raises(PrefixExpiredError):
                runner.action_at(6)
        finally:
            backend.release.set()


def test_reset_discards_old_inflight_prefix():
    backend = Backend()
    with AsyncDualSystemRunner(backend, execution_horizon=4) as runner:
        try:
            runner.reset("old")
            runner.publish({"prompt": "old", "state": 0}, step=0)
            runner.wait_ready(2)
            runner.publish({"prompt": "old", "state": 1}, step=1)
            assert backend.blocked.wait(2)
            runner.reset("new")
            runner.publish({"prompt": "new", "state": 10}, step=0)
            backend.release.set()
            runner.wait_ready(2)
            action, _ = runner.action_at(0)
            assert action[0] == 10
        finally:
            backend.release.set()


def test_worker_errors_are_reported():
    class FailingBackend(Backend):
        def encode(self, observation):
            raise ValueError("broken encoder")

    with AsyncDualSystemRunner(FailingBackend(), execution_horizon=4) as runner:
        runner.reset("task")
        runner.publish({"prompt": "task", "state": 0}, step=0)
        with pytest.raises(RuntimeError, match="worker failed") as caught:
            runner.wait_ready(2)
        assert isinstance(caught.value.__cause__, ValueError)


def test_reset_discards_old_inflight_actions():
    class BlockingExpert(Backend):
        def encode(self, observation):
            return observation["prompt"], observation["state"]

        def sample(self, observation, condition, prefix_age):
            if observation["state"] == 1:
                self.blocked.set()
                if not self.release.wait(5):
                    raise TimeoutError("test gate")
            return super().sample(observation, condition, prefix_age)

    backend = BlockingExpert()
    with AsyncDualSystemRunner(backend, execution_horizon=4) as runner:
        try:
            runner.reset("old")
            runner.publish({"prompt": "old", "state": 0}, step=0)
            runner.wait_ready(2)
            runner.publish({"prompt": "old", "state": 1}, step=1)
            assert backend.blocked.wait(2)
            runner.reset("new")
            runner.publish({"prompt": "new", "state": 10}, step=0)
            backend.release.set()
            runner.wait_ready(2)
            action, _ = runner.action_at(0)
            assert action[0] == 10
        finally:
            backend.release.set()
