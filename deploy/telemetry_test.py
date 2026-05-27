import json

import numpy as np

from deploy.telemetry import InferenceTelemetryRecorder
from deploy.telemetry import build_inference_record


def test_build_inference_record_counts_model_scheduled_and_executed_actions():
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    scheduled_actions = actions[:2]
    state = np.arange(7, dtype=np.float32)

    record = build_inference_record(
        iteration=4,
        loop_step=12,
        wall_time=100.5,
        obs_timestamp=99.5,
        inference_latency_ms=123.456,
        actions=actions,
        scheduled_actions=scheduled_actions,
        scheduled_timestamps=np.array([101.0, 101.1], dtype=np.float64),
        state=state,
        no_execute=False,
        steps_per_inference=6,
        run_id="20260528_120000",
    )

    assert record["run_id"] == "20260528_120000"
    assert record["iteration"] == 4
    assert record["loop_step"] == 12
    assert record["inference_latency_ms"] == 123.456
    assert record["action_dim"] == 7
    assert record["model_action_count"] == 3
    assert record["scheduled_action_count"] == 2
    assert record["executed_action_count"] == 2
    assert record["dropped_action_count"] == 1
    assert record["state"] == state.tolist()
    assert record["first_action"] == actions[0].tolist()
    assert record["last_action"] == actions[-1].tolist()
    assert record["scheduled_timestamps"] == [101.0, 101.1]


def test_build_inference_record_sets_executed_count_to_zero_when_no_execute():
    actions = np.ones((2, 7), dtype=np.float32)

    record = build_inference_record(
        iteration=0,
        loop_step=0,
        wall_time=100.0,
        obs_timestamp=99.0,
        inference_latency_ms=10.0,
        actions=actions,
        scheduled_actions=actions,
        scheduled_timestamps=np.array([100.1, 100.2], dtype=np.float64),
        state=np.zeros(7, dtype=np.float32),
        no_execute=True,
        steps_per_inference=None,
    )

    assert record["scheduled_action_count"] == 2
    assert record["executed_action_count"] == 0
    assert record["no_execute"] is True


def test_build_inference_record_includes_async_metrics_when_present():
    actions = np.ones((2, 7), dtype=np.float32)

    record = build_inference_record(
        iteration=0,
        loop_step=0,
        wall_time=100.0,
        obs_timestamp=99.0,
        inference_latency_ms=10.0,
        actions=actions,
        scheduled_actions=actions,
        scheduled_timestamps=np.array([100.1, 100.2], dtype=np.float64),
        state=np.zeros(7, dtype=np.float32),
        no_execute=False,
        steps_per_inference=2,
        async_metrics={
            "async_mode": True,
            "async_request_id": 5,
            "async_overlap_steps": 3,
            "async_overlap_window_ms": 300.0,
            "async_policy_call_ms": 125.0,
            "async_chunk_boundary_wait_ms": 20.0,
            "async_hidden_inference_ms": 105.0,
            "async_future_state_applied": True,
            "async_target_start_timestamp": 101.0,
        },
    )

    assert record["async_mode"] is True
    assert record["async_request_id"] == 5
    assert record["async_overlap_steps"] == 3
    assert record["async_overlap_window_ms"] == 300.0
    assert record["async_policy_call_ms"] == 125.0
    assert record["async_chunk_boundary_wait_ms"] == 20.0
    assert record["async_hidden_inference_ms"] == 105.0
    assert record["async_future_state_applied"] is True
    assert record["async_target_start_timestamp"] == 101.0


def test_inference_telemetry_recorder_writes_jsonl_and_creates_parent_dir(tmp_path):
    telemetry_path = tmp_path / "nested" / "inference.jsonl"
    record = {
        "iteration": 1,
        "inference_latency_ms": 20.0,
        "model_action_count": 3,
        "scheduled_action_count": 2,
        "executed_action_count": 2,
    }

    with InferenceTelemetryRecorder(telemetry_path) as recorder:
        recorder.write(record)

    assert telemetry_path.exists()
    lines = telemetry_path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [record]
