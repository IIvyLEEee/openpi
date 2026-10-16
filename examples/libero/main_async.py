"""Local asynchronous dual-system evaluation, with a fixed simulation-step/control clock."""

import dataclasses
import json
import logging
import pathlib
import time

from evaluate_checkpoint import observation as _observation
from evaluate_checkpoint import summarize
import imageio
from libero.libero import benchmark
from main import _get_libero_env
import numpy as np
import torch
import tyro

from openpi.policies.async_runner import AsyncDualSystemRunner
from openpi.policies.async_runner import PrefixExpiredError
from openpi.policies.policy_config import create_async_libero_policy
from openpi.training.config import get_config


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    config_name: str = "pi05_libero_async"
    vlm_device: str = "cuda:0"
    expert_device: str = "cuda:1"
    num_steps: int = 10
    rate_hz: float = 10.0  # Wall-clock ticks; each tick advances one original LIBERO step.
    execution_horizon: int | None = None  # Default: all positions in the predicted chunk.
    replan_steps: int | None = None  # Legacy alias for the execution cap.
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 10
    num_steps_wait: int = 10
    video_out_path: str = "data/libero/async"
    seed: int = 7
    save_videos: bool = True


def timing_summary(records):
    steps = [step for record in records for step in record["timing"]]
    actions = [step for step in steps if not step["no_action_ready"]]
    ages = [step["prefix_age_at_execute"] for step in actions]
    sample_ages = [step["prefix_age_at_sample"] for step in actions]
    return {
        "control_steps": len(steps),
        "no_action_fraction": sum(step["no_action_ready"] for step in steps) / len(steps) if steps else None,
        "expired_prefix_fraction": sum(step["expired_prefix"] for step in steps) / len(steps) if steps else None,
        "prefix_age_at_execute_mean": float(np.mean(ages)) if ages else None,
        "prefix_age_at_execute_p95": float(np.percentile(ages, 95)) if ages else None,
        "prefix_age_at_execute_max": max(ages) if ages else None,
        "prefix_age_at_sample_mean": float(np.mean(sample_ages)) if sample_ages else None,
        "prefix_age_at_sample_max": max(sample_ages) if sample_ages else None,
        "executed_action_sample_ms_mean": float(np.mean([step["sample_ms"] for step in actions])) if actions else None,
        "state_to_execute_ms_mean": float(np.mean([step["state_to_execute_ms"] for step in actions]))
        if actions
        else None,
        "control_lateness_ms_p95": float(np.percentile([step["control_lateness_ms"] for step in steps], 95))
        if steps
        else None,
    }


def eval_libero(args: Args):
    if args.rate_hz <= 0 or args.num_trials_per_task <= 0:
        raise ValueError("rate_hz and num_trials_per_task must be positive")
    config = get_config(args.config_name)
    if args.execution_horizon is not None and args.replan_steps is not None:
        raise ValueError("Use execution_horizon, or the legacy replan_steps alias, not both")
    execution_horizon = args.execution_horizon if args.execution_horizon is not None else args.replan_steps
    if execution_horizon is None:
        execution_horizon = config.model.action_horizon
    if not 0 < execution_horizon <= config.model.action_horizon:
        raise ValueError("execution_horizon exceeds the model action horizon")
    output = pathlib.Path(args.video_out_path)
    output.mkdir(parents=True, exist_ok=True)
    specification = {
        **dataclasses.asdict(args),
        "protocol": "async_two_workers",
        "injected_delay": False,
        "prefix_age_bound": "sampling",
        "simulation_control_freq": 20,
        "resolved_execution_horizon": execution_horizon,
    }
    spec_path = output / "protocol.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != specification:
        raise ValueError("Existing results use a different evaluation protocol")
    spec_path.write_text(json.dumps(specification, indent=2))
    records_path = output / "episodes.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    completed = {(record["task_id"], record["trial"]) for record in records}
    np.random.seed(args.seed)
    backend = create_async_libero_policy(
        config,
        args.checkpoint_dir,
        vlm_device=args.vlm_device,
        expert_device=args.expert_device,
        num_steps=args.num_steps,
    )
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[args.task_suite_name]
    expected = suite.n_tasks * args.num_trials_per_task
    saved_videos = set()
    with records_path.open("a") as metrics:
        for task_id in range(suite.n_tasks):
            if all((task_id, trial) in completed for trial in range(args.num_trials_per_task)):
                continue
            task = suite.get_task(task_id)
            env, prompt = _get_libero_env(task, resolution=256, seed=args.seed)
            initial_states = suite.get_task_init_states(task_id)
            try:
                for episode in range(args.num_trials_per_task):
                    if (task_id, episode) in completed:
                        continue
                    env.reset()
                    obs = env.set_init_state(initial_states[episode])
                    for _ in range(args.num_steps_wait):
                        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                    episode_seed = args.seed + 100 * task_id + episode
                    np.random.seed(episode_seed)
                    torch.manual_seed(episode_seed)
                    done = False
                    frames, timing = [], []
                    gripper = -1.0
                    start = time.monotonic()
                    # Join in-flight workers before reseeding the next episode.
                    with AsyncDualSystemRunner(backend, execution_horizon=execution_horizon) as runner:
                        runner.reset(prompt)
                        runner.publish(_observation(obs, prompt), step=0)
                        runner.wait_ready()
                        startup_seconds = time.monotonic() - start
                        tick = time.monotonic()
                        for step in range(max_steps):
                            deadline = tick + step / args.rate_hz
                            time.sleep(max(0.0, deadline - time.monotonic()))
                            lateness_ms = max(0.0, time.monotonic() - deadline) * 1000
                            if step:
                                runner.publish(_observation(obs, prompt), step=step)
                            expired_prefix = False
                            try:
                                action, info = runner.action_at(step)
                            except PrefixExpiredError:
                                action, info = None, {}
                                expired_prefix = True
                            info["control_step"] = step
                            info["no_action_ready"] = action is None
                            info["expired_prefix"] = expired_prefix
                            info["control_lateness_ms"] = lateness_ms
                            timing.append(info)
                            # No extrapolation of expired delta commands: hold pose, retain gripper command.
                            if action is None:
                                action = np.asarray([0.0] * 6 + [gripper])
                            gripper = float(action[-1])
                            if action.shape != (7,) or not np.isfinite(action).all():
                                raise ValueError("Asynchronous policy returned an invalid action")
                            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                            obs, _, done, _ = env.step(action.tolist())
                            if done:
                                break
                    record = {
                        "suite": args.task_suite_name,
                        "task_id": task_id,
                        "task": prompt,
                        "trial": episode,
                        "initial_state_index": episode,
                        "seed": episode_seed,
                        "episode": episode,
                        "success": bool(done),
                        "error": None,
                        "control_steps": len(timing),
                        "wall_seconds": time.monotonic() - start,
                        "startup_seconds": startup_seconds,
                        "rate_hz": args.rate_hz,
                        "num_steps": args.num_steps,
                        "execution_horizon": execution_horizon,
                        "timing": timing,
                    }
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    records.append(record)
                    summary = {**summarize(records, expected), "timing": timing_summary(records)}
                    (output / "summary.json").write_text(json.dumps(summary, indent=2))
                    video_key = task_id, bool(done)
                    if args.save_videos and frames and video_key not in saved_videos:
                        imageio.mimwrite(
                            output / f"task{task_id}_episode{episode}_{'success' if done else 'failure'}.mp4",
                            frames,
                            fps=args.rate_hz,
                        )
                        saved_videos.add(video_key)
                    logging.info(
                        "Task %d trial %d success=%s: %d/%d successes, %d/%d episodes",
                        task_id,
                        episode,
                        done,
                        summary["successes"],
                        len(records),
                        len(records),
                        expected,
                    )
            finally:
                env.close()
    print(json.dumps({**summarize(records, expected), "timing": timing_summary(records)}, indent=2), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(eval_libero)
