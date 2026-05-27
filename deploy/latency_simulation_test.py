import pytest

from deploy import latency_simulation


def test_scaled_infer_sleeps_extra_time_proportional_to_observed_latency():
    perf_values = iter([10.0, 10.2])
    sleep_calls = []

    def infer_fn(policy_obs):
        assert policy_obs == {"prompt": "pick"}
        return {"actions": [1, 2, 3]}

    result = latency_simulation.scaled_infer(
        infer_fn,
        {"prompt": "pick"},
        latency_scale=3.0,
        perf_counter=lambda: next(perf_values),
        sleep=sleep_calls.append,
    )

    assert result == {"actions": [1, 2, 3]}
    assert sleep_calls == pytest.approx([0.4])


def test_wrap_infer_returns_original_function_when_scale_is_one():
    def infer_fn(policy_obs):
        return policy_obs

    assert latency_simulation.wrap_infer(infer_fn, latency_scale=1.0) is infer_fn
