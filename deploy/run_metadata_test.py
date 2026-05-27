import dataclasses
import pathlib
import time

from deploy.run_metadata import build_run_metadata
from deploy.run_metadata import resolve_run_id
from deploy.run_metadata import resolve_run_paths


@dataclasses.dataclass
class _Args:
    output_dir: pathlib.Path = pathlib.Path("data/umi_real_inference")
    telemetry_path: pathlib.Path | None = None
    timestamp_outputs: bool = True
    run_id: str | None = None
    prompt: str = "pick"
    frequency: float = 10.0
    steps_per_inference: int | None = 6
    max_scheduled_actions: int | None = None
    async_inference: bool = False
    inference_overlap_steps: int = 0
    async_future_state: bool = True
    record_episode: bool = True
    no_execute: bool = False
    observe_only: bool = False
    init_joints: bool = False


def test_resolve_run_id_uses_local_timestamp_when_missing(monkeypatch):
    monkeypatch.setattr("deploy.run_metadata.time.time", lambda: 1_700_000_000.0)

    assert resolve_run_id(_Args()) == time.strftime("%Y%m%d_%H%M%S", time.localtime(1_700_000_000.0))


def test_resolve_run_paths_uses_timestamped_run_directory_by_default():
    args = _Args(run_id="20260528_120000")

    paths = resolve_run_paths(args, run_id="20260528_120000")

    assert paths.output_dir == pathlib.Path("data/umi_real_inference/runs/20260528_120000")
    assert paths.telemetry_path == paths.output_dir / "telemetry" / "inference.jsonl"
    assert paths.metadata_path == paths.output_dir / "run_metadata.json"
    assert paths.replay_buffer_path == paths.output_dir / "replay_buffer.zarr"
    assert paths.videos_dir == paths.output_dir / "videos"


def test_build_run_metadata_records_paths_runtime_and_policy_steps(tmp_path):
    args = _Args(run_id="run42", output_dir=tmp_path)
    paths = resolve_run_paths(args, run_id="run42")

    metadata = build_run_metadata(
        run_id="run42",
        args=args,
        run_paths=paths,
        robot_config_path=pathlib.Path("deploy/configs/umi_ur5e_wsg50.yaml"),
        robot_config={"robots": [{"robot_type": "ur5e"}]},
        server_metadata={"serve_policy": {"num_steps": 12, "log_denoise_steps": True}},
    )

    assert metadata["run_id"] == "run42"
    assert metadata["output_dir"] == str(paths.output_dir)
    assert metadata["telemetry_path"] == str(paths.telemetry_path)
    assert metadata["replay_buffer_path"] == str(paths.replay_buffer_path)
    assert metadata["videos_dir"] == str(paths.videos_dir)
    assert metadata["runtime"]["prompt"] == "pick"
    assert metadata["runtime"]["steps_per_inference"] == 6
    assert metadata["policy_server_metadata"]["serve_policy"]["num_steps"] == 12
