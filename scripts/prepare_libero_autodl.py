"""Download the complete LIBERO dataset and compute stats without decoding RGB images."""

import json
import os
from pathlib import Path
import time
from typing import Literal

from huggingface_hub import HfApi
from huggingface_hub import snapshot_download
import numpy as np
import pyarrow.parquet as pq
import tyro

from openpi.shared import normalize
from openpi.training.config import get_config


def accumulate_episode(stats, state, actions, horizon):
    """Match LeRobot future-action chunks, including repetition at the episode end."""
    if len(state) != len(actions) or not len(state):
        raise ValueError("Episode state/actions must have equal nonzero lengths")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Episode contains non-finite state/actions")
    indices = np.minimum(np.arange(len(actions))[:, None] + np.arange(horizon)[None, :], len(actions) - 1)
    stats["state"].update(state)
    stats["actions"].update(actions[indices])


def main(stage: Literal["download", "stats", "all"] = "all", workers: int = 8):
    config = get_config("pi05_libero_async")
    repo = config.data.repo_id
    root = Path(os.environ["HF_LEROBOT_HOME"]) / repo
    if stage in ("download", "all"):
        info = HfApi().dataset_info(repo, revision="v2.0", files_metadata=True)
        print(f"Downloading {repo}, revision {info.sha}, to {root}", flush=True)
        for attempt in range(5):
            try:
                snapshot_download(
                    repo,
                    repo_type="dataset",
                    revision=info.sha,
                    local_dir=root,
                    allow_patterns=["meta/*", "data/*"],
                    max_workers=workers,
                )
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                print(f"Download retry {attempt + 1}: {type(exc).__name__}", flush=True)
                time.sleep(2 ** (attempt + 1))
        (root / "download_revision.json").write_text(json.dumps({"repo": repo, "revision": info.sha}, indent=2))
    if stage == "download":
        return

    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    frames = 0
    for index, episode in enumerate(episodes):
        episode_id = episode["episode_index"]
        path = root / info["data_path"].format(
            episode_chunk=episode_id // info["chunks_size"], episode_index=episode_id
        )
        # Column projection skips embedded JPEG bytes. LIBERO's native state/actions need no numeric transform.
        table = pq.read_table(path, columns=["state", "actions"])
        state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        if len(state) != episode["length"] or state.shape[1:] != (8,) or actions.shape[1:] != (7,):
            raise ValueError(f"LIBERO episode metadata/shape mismatch: {path}")
        accumulate_episode(stats, state, actions, config.model.action_horizon)
        frames += len(state)
        if index % 100 == 0:
            print(f"Stats: {index + 1}/{len(episodes)} episodes, {frames} frames", flush=True)
    if len(episodes) != info["total_episodes"] or frames != info["total_frames"]:
        raise ValueError("Full dataset counts do not match metadata")
    output = config.assets_dirs / repo
    normalize.save(output, {key: value.get_statistics() for key, value in stats.items()})
    report = {
        "repo": repo,
        "episodes": len(episodes),
        "frames": frames,
        "tasks": info["total_tasks"],
        "action_horizon": config.model.action_horizon,
        "includes_all_frames": True,
        "stats_path": str(output),
    }
    (output / "dataset_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    tyro.cli(main)
