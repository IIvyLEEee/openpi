"""Convert the UMI conveyor zarr dataset to LeRobot format."""

import pathlib
import shutil
import sys
import zipfile

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro
import zarr


DEFAULT_ZARR_ZIP = pathlib.Path(
    "/data/archive_liyixuan23/umi_for_train/example_pick_night/mixdataset.zarr.zip"
)
DEFAULT_EXTRACT_DIR = pathlib.Path(
    "/data/archive_liyixuan23/finetune_pi/openpi_zarr_cache/umi_conveyor/mixdataset.zarr"
)
DEFAULT_OUTPUT_ROOT = pathlib.Path(
    "/data/archive_liyixuan23/finetune_pi/openpi_lerobot/liyixuan23/umi_conveyor"
)
DEFAULT_PROMPT = "pick up the red block on the conveyor belt. put it in the blue box."


def _register_umi_codecs() -> None:
    codec_dir = pathlib.Path("/data/archive_liyixuan23/umi_for_train/diffusion_policy/codecs")
    if codec_dir.exists():
        sys.path.insert(0, str(codec_dir))
    try:
        from imagecodecs_numcodecs import register_codecs
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import UMI JPEG-XL codec registration. "
            "Install imagecodecs and check /data/archive_liyixuan23/umi_for_train/diffusion_policy/codecs."
        ) from exc
    register_codecs()


def _extract_zarr(zarr_zip: pathlib.Path, extract_dir: pathlib.Path, overwrite: bool) -> pathlib.Path:
    if extract_dir.exists():
        if overwrite:
            shutil.rmtree(extract_dir)
        else:
            return extract_dir

    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zarr_zip) as archive:
        archive.extractall(extract_dir)
    return extract_dir


def _make_dataset(output_root: pathlib.Path, fps: int, overwrite: bool) -> LeRobotDataset:
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output dataset already exists: {output_root}")

    return LeRobotDataset.create(
        repo_id="liyixuan23/umi_conveyor",
        root=output_root,
        robot_type="ur5e",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=0,
    )


def _state_at(root: zarr.Group, index: int) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(root["data/robot0_eef_pos"][index], dtype=np.float32),
            np.asarray(root["data/robot0_eef_rot_axis_angle"][index], dtype=np.float32),
            np.asarray(root["data/robot0_gripper_width"][index], dtype=np.float32).reshape(1),
        ]
    )


def main(
    zarr_zip: pathlib.Path = DEFAULT_ZARR_ZIP,
    extract_dir: pathlib.Path = DEFAULT_EXTRACT_DIR,
    output_root: pathlib.Path = DEFAULT_OUTPUT_ROOT,
    *,
    prompt: str = DEFAULT_PROMPT,
    fps: int = 10,
    max_episodes: int | None = None,
    overwrite: bool = False,
    overwrite_extract: bool = False,
) -> None:
    _register_umi_codecs()

    zarr_dir = _extract_zarr(zarr_zip, extract_dir, overwrite_extract)
    root = zarr.open_group(str(zarr_dir), mode="r")

    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if max_episodes is not None:
        episode_ends = episode_ends[:max_episodes]

    total_frames = int(episode_ends[-1])
    image_shape = tuple(root["data/camera0_rgb"].shape[1:])
    if image_shape != (224, 224, 3):
        raise ValueError(f"Expected camera0_rgb shape (*, 224, 224, 3), got (*, {image_shape})")

    dataset = _make_dataset(output_root, fps=fps, overwrite=overwrite)

    start = 0
    for episode_idx, end in enumerate(tqdm.tqdm(episode_ends, desc="Converting episodes")):
        end = int(end)
        for index in tqdm.trange(start, end, desc=f"episode {episode_idx}", leave=False):
            state = _state_at(root, index)
            dataset.add_frame(
                {
                    "image": np.asarray(root["data/camera0_rgb"][index], dtype=np.uint8),
                    "state": state,
                    "actions": state,
                    "task": prompt,
                }
            )
        dataset.save_episode()
        start = end

    print(f"Wrote {len(episode_ends)} episodes / {total_frames} frames to {output_root}")
    print("Use HF_LEROBOT_HOME=/data/archive_liyixuan23/finetune_pi/openpi_lerobot when training.")


if __name__ == "__main__":
    tyro.cli(main)
