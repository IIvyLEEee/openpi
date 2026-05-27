import dataclasses
import json
import pathlib
import time
from typing import Any


@dataclasses.dataclass(frozen=True)
class RunPaths:
    output_dir: pathlib.Path
    telemetry_path: pathlib.Path
    metadata_path: pathlib.Path
    replay_buffer_path: pathlib.Path
    videos_dir: pathlib.Path


def resolve_run_id(args) -> str:
    if getattr(args, "run_id", None):
        return args.run_id
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


def resolve_run_paths(args, *, run_id: str) -> RunPaths:
    output_dir = pathlib.Path(args.output_dir).expanduser()
    if getattr(args, "timestamp_outputs", True):
        output_dir = output_dir / "runs" / run_id

    telemetry_path = getattr(args, "telemetry_path", None)
    if telemetry_path is None:
        telemetry_path = output_dir / "telemetry" / "inference.jsonl"
    else:
        telemetry_path = pathlib.Path(telemetry_path).expanduser()

    return RunPaths(
        output_dir=output_dir,
        telemetry_path=telemetry_path,
        metadata_path=output_dir / "run_metadata.json",
        replay_buffer_path=output_dir / "replay_buffer.zarr",
        videos_dir=output_dir / "videos",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, pathlib.Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {key: _json_safe(val) for key, val in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _runtime_metadata(args) -> dict[str, Any]:
    keys = [
        "policy_server_host",
        "policy_server_port",
        "robot_config",
        "prompt",
        "frequency",
        "steps_per_inference",
        "max_scheduled_actions",
        "max_duration",
        "camera_obs_latency",
        "action_exec_latency",
        "inference_latency_scale",
        "max_pos_speed",
        "max_rot_speed",
        "dry_run",
        "observe_only",
        "no_execute",
        "show_camera",
        "record_episode",
        "async_inference",
        "inference_overlap_steps",
        "async_future_state",
        "init_joints",
        "timestamp_outputs",
    ]
    return {key: _json_safe(getattr(args, key)) for key in keys if hasattr(args, key)}


def build_run_metadata(
    *,
    run_id: str,
    args,
    run_paths: RunPaths,
    robot_config_path: pathlib.Path | None,
    robot_config: dict[str, Any] | None,
    server_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())),
        "output_dir": str(run_paths.output_dir),
        "telemetry_path": str(run_paths.telemetry_path),
        "metadata_path": str(run_paths.metadata_path),
        "replay_buffer_path": str(run_paths.replay_buffer_path),
        "videos_dir": str(run_paths.videos_dir),
        "robot_config_path": str(robot_config_path) if robot_config_path is not None else None,
        "robot_config": _json_safe(robot_config),
        "runtime": _runtime_metadata(args),
        "policy_server_metadata": _json_safe(server_metadata or {}),
    }


def write_run_metadata(path: pathlib.Path | str, metadata: dict[str, Any]) -> None:
    path = pathlib.Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
