import numpy as np


def recorded_action_step_count(action_timestamps: np.ndarray, *, end_time: float) -> int:
    action_timestamps = np.asarray(action_timestamps, dtype=np.float64)
    if action_timestamps.size == 0:
        return 0
    valid = np.nonzero(action_timestamps <= end_time)[0]
    if valid.size == 0:
        return 0
    return int(valid[-1] + 1)


def should_finalize_episode(*, obs_accumulator, action_accumulator) -> bool:
    return obs_accumulator is not None or action_accumulator is not None
