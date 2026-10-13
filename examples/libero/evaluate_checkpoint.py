"""Evaluate a local PyTorch checkpoint using the standard synchronous LIBERO protocol."""

import collections
import dataclasses
import json
import logging
import pathlib
import time

import imageio
from libero.libero import benchmark
from main import _get_libero_env
from main import _quat2axisangle
import numpy as np
from openpi_client import image_tools
import torch
import tyro

from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = dict(zip(SUITES, (220, 280, 300, 520), strict=True))


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    output_dir: str
    config_name: str = "pi05_libero_async"
    device: str = "cuda:0"
    suites: tuple[str, ...] = SUITES
    trials_per_task: int = 5
    seed: int = 7
    flow_steps: int = 10
    replan_steps: int = 5
    settle_steps: int = 10
    save_videos: bool = True


def summarize(records, expected):
    suites = {}
    for record in records:
        result = suites.setdefault(record["suite"], {"episodes": 0, "successes": 0})
        result["episodes"] += 1
        result["successes"] += int(record["success"])
    for value in suites.values():
        value["success_rate"] = value["successes"] / value["episodes"]
    successes = sum(int(record["success"]) for record in records)
    return {
        "completed_episodes": len(records),
        "expected_episodes": expected,
        "complete": len(records) == expected,
        "successes": successes,
        "success_rate": successes / len(records) if records else None,
        "per_suite": suites,
    }


def observation(obs, prompt):
    return {
        "observation/image": image_tools.resize_with_pad(
            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224
        ),
        "observation/wrist_image": image_tools.resize_with_pad(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224
        ),
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"])
        ),
        "prompt": prompt,
    }


def main(args: Args):
    if args.trials_per_task <= 0 or any(suite not in SUITES for suite in args.suites):
        raise ValueError("Use a positive trial count and the four supported LIBERO suites")
    config = get_config(args.config_name)
    if not 0 < args.replan_steps <= config.model.action_horizon:
        raise ValueError("Invalid action execution horizon")
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specification = {**dataclasses.asdict(args), "protocol": "synchronous_fresh_observation"}
    specification["suites"] = list(args.suites)
    spec_path = output / "protocol.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != specification:
        raise ValueError("Existing results use a different evaluation protocol")
    spec_path.write_text(json.dumps(specification, indent=2))
    records_path = output / "episodes.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    completed = {(r["suite"], r["task_id"], r["trial"]) for r in records}
    suites = {name: benchmark.get_benchmark_dict()[name]() for name in args.suites}
    expected = sum(suite.n_tasks * args.trials_per_task for suite in suites.values())
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = create_trained_policy(
        config, args.checkpoint_dir, pytorch_device=args.device, sample_kwargs={"num_steps": args.flow_steps}
    )
    logging.info("Checkpoint loaded; evaluating %d episodes", expected)
    saved_videos = set()
    with records_path.open("a") as metrics:
        for suite_index, (suite_name, suite) in enumerate(suites.items()):
            for task_id in range(suite.n_tasks):
                if all((suite_name, task_id, trial) in completed for trial in range(args.trials_per_task)):
                    continue
                task = suite.get_task(task_id)
                initial_states = suite.get_task_init_states(task_id)
                if len(initial_states) < args.trials_per_task:
                    raise ValueError("Insufficient official initial states")
                env, prompt = _get_libero_env(task, resolution=256, seed=args.seed)
                try:
                    for trial in range(args.trials_per_task):
                        if (suite_name, task_id, trial) in completed:
                            continue
                        env.reset()
                        obs = env.set_init_state(initial_states[trial])
                        for _ in range(args.settle_steps):
                            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                        episode_seed = args.seed + 10000 * suite_index + 100 * task_id + trial
                        np.random.seed(episode_seed)
                        torch.manual_seed(episode_seed)
                        actions = collections.deque()
                        frames, latencies = [], []
                        success = False
                        control_steps = 0
                        start = time.monotonic()
                        for _ in range(MAX_STEPS[suite_name]):
                            control_steps += 1
                            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                            if not actions:
                                request_start = time.monotonic()
                                chunk = np.asarray(policy.infer(observation(obs, prompt))["actions"])
                                latencies.append(time.monotonic() - request_start)
                                if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < args.replan_steps:
                                    raise ValueError(f"Invalid policy action shape: {chunk.shape}")
                                if not np.isfinite(chunk).all():
                                    raise ValueError("Policy returned non-finite actions")
                                actions.extend(chunk[: args.replan_steps])
                            obs, _, done, _ = env.step(actions.popleft().tolist())
                            if done:
                                success = True
                                break
                        record = {
                            "suite": suite_name,
                            "task_id": task_id,
                            "task": prompt,
                            "trial": trial,
                            "initial_state_index": trial,
                            "seed": episode_seed,
                            "success": success,
                            "control_steps": control_steps,
                            "wall_seconds": time.monotonic() - start,
                            "mean_inference_seconds": float(np.mean(latencies)),
                        }
                        metrics.write(json.dumps(record) + "\n")
                        metrics.flush()
                        records.append(record)
                        (output / "summary.json").write_text(json.dumps(summarize(records, expected), indent=2))
                        video_key = suite_name, task_id, success
                        if args.save_videos and video_key not in saved_videos:
                            imageio.mimwrite(
                                output
                                / f"{suite_name}_task{task_id}_trial{trial}_{'success' if success else 'failure'}.mp4",
                                frames,
                                fps=10,
                            )
                            saved_videos.add(video_key)
                        logging.info(
                            "%s task=%d trial=%d success=%s total=%d/%d successes=%d",
                            suite_name,
                            task_id,
                            trial,
                            success,
                            len(records),
                            expected,
                            sum(int(r["success"]) for r in records),
                        )
                finally:
                    env.close()
    print(json.dumps(summarize(records, expected), indent=2), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
