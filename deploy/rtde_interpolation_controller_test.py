# ruff: noqa: E402, SLF001
import sys
import types

import numpy as np


class _DummyRTDEControlInterface:
    pass


class _DummyRTDEReceiveInterface:
    pass


sys.modules.setdefault(
    "rtde_control",
    types.SimpleNamespace(RTDEControlInterface=_DummyRTDEControlInterface),
)
sys.modules.setdefault(
    "rtde_receive",
    types.SimpleNamespace(RTDEReceiveInterface=_DummyRTDEReceiveInterface),
)
sys.modules.setdefault(
    "deploy.umi_data.pose_util",
    types.SimpleNamespace(
        mat_to_pose=lambda mat: mat,
        pose_to_mat=lambda pose: pose,
    ),
)
sys.modules.setdefault(
    "deploy.umi.common.pose_trajectory_interpolator",
    types.SimpleNamespace(PoseTrajectoryInterpolator=object),
)
_shared_memory_pkg = types.ModuleType("deploy.umi.shared_memory")
_shared_memory_pkg.__path__ = []
sys.modules.setdefault("deploy.umi.shared_memory", _shared_memory_pkg)
sys.modules.setdefault(
    "deploy.umi.shared_memory.shared_memory_queue",
    types.SimpleNamespace(Empty=Exception, SharedMemoryQueue=object),
)
sys.modules.setdefault(
    "deploy.umi.shared_memory.shared_memory_ring_buffer",
    types.SimpleNamespace(SharedMemoryRingBuffer=object),
)

from deploy.umi.real_world import rtde_interpolation_controller


class _RecordingInterpolator:
    def __init__(self):
        self.drive_to_waypoint_calls = []

    def drive_to_waypoint(
        self,
        *,
        pose,
        time,
        curr_time,
        max_pos_speed,
        max_rot_speed,
    ):
        self.drive_to_waypoint_calls.append(
            {
                "pose": pose,
                "time": time,
                "curr_time": curr_time,
                "max_pos_speed": max_pos_speed,
                "max_rot_speed": max_rot_speed,
            }
        )
        return "next-interpolator"


def test_servol_command_converts_unified_pose_before_driving_waypoint():
    pose_interp = _RecordingInterpolator()
    target_pose = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03], dtype=np.float64)
    converted_pose = target_pose + np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float64)

    next_interp, last_waypoint_time = rtde_interpolation_controller._apply_servol_command(
        command={"target_pose": target_pose, "duration": 0.25},
        pose_interp=pose_interp,
        t_now=12.0,
        dt=0.002,
        max_pos_speed=0.4,
        max_rot_speed=0.5,
        pose_to_robot_frame=lambda pose: pose + np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float64),
    )

    assert next_interp == "next-interpolator"
    assert last_waypoint_time == 12.252
    assert len(pose_interp.drive_to_waypoint_calls) == 1
    call = pose_interp.drive_to_waypoint_calls[0]
    np.testing.assert_allclose(call["pose"], converted_pose)
    assert call["time"] == 12.252
    assert call["curr_time"] == 12.002
    assert call["max_pos_speed"] == 0.4
    assert call["max_rot_speed"] == 0.5
