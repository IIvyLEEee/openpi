import numpy as np
import pytest

from deploy.umi.real_world.robot_init import DEFAULT_SINGLE_UR5E_JOINTS
from deploy.umi.real_world.robot_init import resolve_robot_init_joints
from deploy.umi.real_world.robot_init import resolve_robot_launch_timeout


def test_single_ur5e_bool_init_uses_umi_default_joints():
    j_inits = resolve_robot_init_joints(init_joints=True, robot_count=1)

    assert len(j_inits) == 1
    np.testing.assert_allclose(j_inits[0], DEFAULT_SINGLE_UR5E_JOINTS)


def test_false_init_returns_none_per_robot():
    assert resolve_robot_init_joints(init_joints=False, robot_count=2) == [None, None]


def test_explicit_init_joints_must_match_robot_count():
    with pytest.raises(ValueError, match="robot_count"):
        resolve_robot_init_joints(init_joints=[np.zeros(6)], robot_count=2)


def test_explicit_init_joints_are_validated_as_six_joint_vectors():
    with pytest.raises(ValueError, match="shape"):
        resolve_robot_init_joints(init_joints=[np.zeros(7)], robot_count=1)


def test_launch_timeout_is_longer_when_init_joints_are_used():
    assert resolve_robot_launch_timeout(robot_config={}, joints_init=np.zeros(6)) == 30


def test_launch_timeout_uses_config_override_when_present():
    assert resolve_robot_launch_timeout(robot_config={"launch_timeout": 45}, joints_init=np.zeros(6)) == 45
