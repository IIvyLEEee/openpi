import numpy as np

DEFAULT_SINGLE_UR5E_JOINTS = np.array([0, -90, -90, -90, 90, 0], dtype=np.float64) / 180 * np.pi
DEFAULT_BIMANUAL_UR5E_JOINTS = np.array(
    [
        [-2.35, -1.57, -1.57, -1.57, 1.57, 0.0],
        [2.35, -1.57, 1.57, -1.57, -1.57, 0.0],
    ],
    dtype=np.float64,
)
DEFAULT_LAUNCH_TIMEOUT = 3
DEFAULT_INIT_JOINTS_LAUNCH_TIMEOUT = 30


def _validate_joint_vector(joints):
    joints = np.array(joints, dtype=np.float64)
    if joints.shape != (6,):
        raise ValueError(f"Expected init joint vector shape (6,), got shape {joints.shape}")
    return joints


def resolve_robot_init_joints(init_joints, robot_count):
    if robot_count <= 0:
        raise ValueError(f"Expected positive robot_count, got {robot_count}")

    if isinstance(init_joints, bool):
        if not init_joints:
            return [None] * robot_count
        if robot_count == 1:
            return [DEFAULT_SINGLE_UR5E_JOINTS.copy()]
        if robot_count == 2:
            return [joints.copy() for joints in DEFAULT_BIMANUAL_UR5E_JOINTS]
        raise ValueError(f"Boolean init_joints=True only has defaults for robot_count 1 or 2, got {robot_count}")

    if init_joints is None:
        return [None] * robot_count

    if robot_count == 1:
        init_joints_array = np.array(init_joints, dtype=np.float64)
        if init_joints_array.shape == (6,):
            return [init_joints_array]

    if len(init_joints) != robot_count:
        raise ValueError(f"Expected init_joints length to match robot_count {robot_count}, got {len(init_joints)}")

    return [_validate_joint_vector(joints) for joints in init_joints]


def resolve_robot_launch_timeout(robot_config, joints_init):
    if "launch_timeout" in robot_config:
        return float(robot_config["launch_timeout"])
    if joints_init is not None:
        return DEFAULT_INIT_JOINTS_LAUNCH_TIMEOUT
    return DEFAULT_LAUNCH_TIMEOUT
