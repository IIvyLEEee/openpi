import json
import pathlib
from typing import Any

import numpy as np


def _array_or_none(value: np.ndarray | None) -> list[float] | None:
    if value is None:
        return None
    return np.asarray(value).reshape(-1).tolist()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_inference_record(
    *,
    iteration: int,
    loop_step: int,
    wall_time: float,
    obs_timestamp: float,
    inference_latency_ms: float,
    actions: np.ndarray,
    scheduled_actions: np.ndarray,
    scheduled_timestamps: np.ndarray,
    state: np.ndarray,
    no_execute: bool,
    steps_per_inference: int | None,
    async_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float32)
    scheduled_actions = np.asarray(scheduled_actions, dtype=np.float32)
    scheduled_timestamps = np.asarray(scheduled_timestamps, dtype=np.float64)
    state = np.asarray(state, dtype=np.float32)

    model_action_count = int(actions.shape[0])
    scheduled_action_count = int(scheduled_actions.shape[0])
    executed_action_count = 0 if no_execute else scheduled_action_count
    action_dim = int(actions.shape[1]) if actions.ndim == 2 else None

    first_action = actions[0] if model_action_count > 0 else None
    last_action = actions[-1] if model_action_count > 0 else None

    record = {
        "iteration": int(iteration),
        "loop_step": int(loop_step),
        "wall_time": float(wall_time),
        "obs_timestamp": float(obs_timestamp),
        "inference_latency_ms": float(inference_latency_ms),
        "action_dim": action_dim,
        "model_action_count": model_action_count,
        "scheduled_action_count": scheduled_action_count,
        "executed_action_count": executed_action_count,
        "dropped_action_count": max(0, model_action_count - scheduled_action_count),
        "no_execute": bool(no_execute),
        "steps_per_inference": steps_per_inference,
        "state": _array_or_none(state),
        "first_action": _array_or_none(first_action),
        "last_action": _array_or_none(last_action),
        "scheduled_timestamps": scheduled_timestamps.reshape(-1).tolist(),
    }
    if async_metrics:
        record.update({key: _json_safe(value) for key, value in async_metrics.items()})
    return record


class InferenceTelemetryRecorder:
    def __init__(self, path: pathlib.Path | str, *, append: bool = True):
        self.path = pathlib.Path(path).expanduser()
        self.append = append
        self._file = None

    def __enter__(self) -> "InferenceTelemetryRecorder":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.append else "w"
        self._file = self.path.open(mode, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, record: dict[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError("InferenceTelemetryRecorder must be used as a context manager.")
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()
