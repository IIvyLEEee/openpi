import numpy as np
import pytest

from deploy.plot_umi_trajectory import extract_trajectory_data
from deploy.plot_umi_trajectory import make_colored_trajectory_segments


def test_extract_trajectory_data_reads_robot_position_and_gripper_width():
    episode = {
        "timestamp": np.array([10.0, 10.1, 10.2], dtype=np.float64),
        "robot0_eef_pos": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.1, 0.2, 0.0],
            ],
            dtype=np.float32,
        ),
        "robot0_gripper_width": np.array([[0.02], [0.04], [0.08]], dtype=np.float32),
    }

    trajectory = extract_trajectory_data(episode, robot_idx=0)

    np.testing.assert_allclose(trajectory.timestamps, episode["timestamp"])
    np.testing.assert_allclose(trajectory.positions, episode["robot0_eef_pos"])
    np.testing.assert_allclose(trajectory.gripper_widths, [0.02, 0.04, 0.08])


def test_extract_trajectory_data_reports_missing_keys():
    with pytest.raises(KeyError, match="robot0_eef_pos"):
        extract_trajectory_data({"timestamp": np.array([1.0])}, robot_idx=0)


def test_make_colored_trajectory_segments_uses_average_segment_gripper_width():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    gripper_widths = np.array([0.1, 0.2, 0.4], dtype=np.float32)

    segments, segment_gripper_widths = make_colored_trajectory_segments(positions, gripper_widths)

    assert segments.shape == (2, 2, 3)
    np.testing.assert_allclose(segments[0], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(segments[1], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    np.testing.assert_allclose(segment_gripper_widths, [0.15, 0.3])
