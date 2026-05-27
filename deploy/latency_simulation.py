from collections.abc import Callable
import time
from typing import Any


def scaled_infer(
    infer_fn: Callable[[dict[str, Any]], dict[str, Any]],
    policy_obs: dict[str, Any],
    *,
    latency_scale: float,
    perf_counter: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    start = perf_counter()
    result = infer_fn(policy_obs)
    elapsed = perf_counter() - start
    extra_delay = elapsed * (latency_scale - 1.0)
    if extra_delay > 0:
        sleep(extra_delay)
    return result


def wrap_infer(
    infer_fn: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    latency_scale: float,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if latency_scale == 1.0:
        return infer_fn

    def wrapped(policy_obs: dict[str, Any]) -> dict[str, Any]:
        return scaled_infer(infer_fn, policy_obs, latency_scale=latency_scale)

    return wrapped
