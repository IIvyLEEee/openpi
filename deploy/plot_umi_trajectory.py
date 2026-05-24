import argparse
import dataclasses
import pathlib

import numpy as np


@dataclasses.dataclass(frozen=True)
class TrajectoryData:
    timestamps: np.ndarray
    positions: np.ndarray
    gripper_widths: np.ndarray


def extract_trajectory_data(episode: dict[str, np.ndarray], *, robot_idx: int = 0) -> TrajectoryData:
    pos_key = f"robot{robot_idx}_eef_pos"
    gripper_key = f"robot{robot_idx}_gripper_width"
    required_keys = ("timestamp", pos_key, gripper_key)
    for key in required_keys:
        if key not in episode:
            raise KeyError(f"Episode is missing required key {key!r}")

    timestamps = np.asarray(episode["timestamp"], dtype=np.float64).reshape(-1)
    positions = np.asarray(episode[pos_key], dtype=np.float64)
    gripper_widths = np.asarray(episode[gripper_key], dtype=np.float64).reshape(-1)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected {pos_key!r} shape (T, 3), got {positions.shape}")
    if len(timestamps) != len(positions) or len(gripper_widths) != len(positions):
        raise ValueError(
            "Expected timestamp, EEF position, and gripper width arrays to have the same first dimension; "
            f"got {len(timestamps)}, {len(positions)}, {len(gripper_widths)}"
        )

    return TrajectoryData(timestamps=timestamps, positions=positions, gripper_widths=gripper_widths)


def make_colored_trajectory_segments(
    positions: np.ndarray,
    gripper_widths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float64)
    gripper_widths = np.asarray(gripper_widths, dtype=np.float64).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected positions shape (T, 3), got {positions.shape}")
    if len(positions) != len(gripper_widths):
        raise ValueError(f"Expected one gripper width per position, got {len(gripper_widths)} and {len(positions)}")
    if len(positions) < 2:
        raise ValueError("Need at least two trajectory points to build colored segments.")

    segments = np.stack([positions[:-1], positions[1:]], axis=1)
    segment_gripper_widths = 0.5 * (gripper_widths[:-1] + gripper_widths[1:])
    return segments, segment_gripper_widths


def load_replay_episode(replay_buffer_path: pathlib.Path | str, *, episode_idx: int = -1) -> dict[str, np.ndarray]:
    from deploy.umi_data.common.replay_buffer import ReplayBuffer

    replay_buffer_path = pathlib.Path(replay_buffer_path).expanduser()
    buffer = ReplayBuffer.copy_from_path(str(replay_buffer_path), store=None)
    return buffer.get_episode(episode_idx, copy=True)


def _set_equal_3d_limits(ax, positions: np.ndarray) -> None:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    centers = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1e-3
    half = 0.5 * span
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)


def plot_trajectory(
    trajectory: TrajectoryData,
    output_path: pathlib.Path | str,
    *,
    title: str | None = None,
    cmap: str = "viridis",
) -> pathlib.Path:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    output_path = pathlib.Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments, segment_gripper_widths = make_colored_trajectory_segments(
        trajectory.positions,
        trajectory.gripper_widths,
    )

    fig = plt.figure(figsize=(11, 8))
    ax_traj = fig.add_subplot(2, 1, 1, projection="3d")
    norm = plt.Normalize(vmin=float(np.min(trajectory.gripper_widths)), vmax=float(np.max(trajectory.gripper_widths)))
    collection = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=2.5)
    collection.set_array(segment_gripper_widths)
    ax_traj.add_collection3d(collection)
    ax_traj.scatter(*trajectory.positions[0], color="black", s=35, label="start")
    ax_traj.scatter(*trajectory.positions[-1], color="red", s=35, label="end")
    _set_equal_3d_limits(ax_traj, trajectory.positions)
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.set_zlabel("z (m)")
    ax_traj.set_title(title or "UR5e end-effector trajectory")
    ax_traj.legend(loc="upper right")
    colorbar = fig.colorbar(collection, ax=ax_traj, pad=0.08)
    colorbar.set_label("gripper width (m)")

    ax_width = fig.add_subplot(2, 1, 2)
    elapsed = trajectory.timestamps - trajectory.timestamps[0]
    ax_width.plot(elapsed, trajectory.gripper_widths, color="black", linewidth=1.5)
    ax_width.set_xlabel("time (s)")
    ax_width.set_ylabel("gripper width (m)")
    ax_width.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _default_output_path(replay_buffer_path: pathlib.Path, episode_idx: int) -> pathlib.Path:
    suffix = "last" if episode_idx == -1 else str(episode_idx)
    return replay_buffer_path.parent / f"trajectory_episode_{suffix}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot UR5e EEF trajectory colored by WSG50 gripper width.")
    parser.add_argument(
        "--replay-buffer", type=pathlib.Path, default=pathlib.Path("data/umi_real_inference/replay_buffer.zarr")
    )
    parser.add_argument("--episode-idx", type=int, default=-1)
    parser.add_argument("--robot-idx", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    episode = load_replay_episode(args.replay_buffer, episode_idx=args.episode_idx)
    trajectory = extract_trajectory_data(episode, robot_idx=args.robot_idx)
    output_path = args.output or _default_output_path(args.replay_buffer, args.episode_idx)
    saved_path = plot_trajectory(trajectory, output_path, title=args.title)
    print(saved_path)


if __name__ == "__main__":
    main()
