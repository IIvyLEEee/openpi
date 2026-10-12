"""Local asynchronous dual-system evaluation, with a fixed simulation-step/control clock."""

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
import tyro

from openpi.policies.async_runner import AsyncDualSystemRunner
from openpi.policies.policy_config import create_async_libero_policy
from openpi.training.config import get_config


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    config_name: str = "pi05_libero_async"
    vlm_device: str = "cuda:0"
    expert_device: str = "cuda:1"
    num_steps: int = 10
    rate_hz: float = 10.0  # Match the FPS of the openpi LIBERO conversion.
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 10
    num_steps_wait: int = 10
    video_out_path: str = "data/libero/async"
    seed: int = 7


def _observation(obs, prompt):
    return {
        # Same 180-degree image rotation and state representation as main.py.
        "observation/image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "observation/wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"])
        ),
        "prompt": prompt,
    }


def eval_libero(args: Args):
    if args.rate_hz <= 0:
        raise ValueError("rate_hz must be positive and match training FPS")
    config = get_config(args.config_name)
    if args.replan_steps > config.model.action_horizon:
        raise ValueError("replan_steps exceeds the model action horizon")
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
    output = pathlib.Path(args.video_out_path)
    output.mkdir(parents=True, exist_ok=True)
    successes = episodes = 0
    with (
        AsyncDualSystemRunner(backend, execution_horizon=args.replan_steps) as runner,
        (output / "metrics.jsonl").open("w") as metrics,
    ):
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            env, prompt = _get_libero_env(task, resolution=256, seed=args.seed)
            initial_states = suite.get_task_init_states(task_id)
            try:
                for episode in range(args.num_trials_per_task):
                    env.reset()
                    obs = env.set_init_state(initial_states[episode])
                    for _ in range(args.num_steps_wait):
                        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                    runner.reset(prompt)
                    runner.publish(_observation(obs, prompt), step=0)
                    runner.wait_ready()
                    done = False
                    frames, timing = [], []
                    error = None
                    gripper = -1.0
                    tick = time.monotonic()
                    try:
                        for step in range(max_steps):
                            if step:
                                runner.publish(_observation(obs, prompt), step=step)
                            action, info = runner.action_at(step)
                            info["control_step"] = step
                            info["no_action_ready"] = action is None
                            timing.append(info)
                            # No extrapolation of expired delta commands: hold pose, retain gripper command.
                            if action is None:
                                action = np.asarray([0.0] * 6 + [gripper])
                            gripper = float(action[-1])
                            frames.append(_observation(obs, prompt)["observation/image"])
                            obs, _, done, _ = env.step(action.tolist())
                            if done:
                                break
                            tick += 1.0 / args.rate_hz
                            time.sleep(max(0.0, tick - time.monotonic()))
                    except Exception as caught:
                        error = str(caught)
                        logging.exception("Asynchronous episode failed")
                    episodes += 1
                    successes += int(done)
                    record = {
                        "task_id": task_id,
                        "episode": episode,
                        "success": bool(done),
                        "error": error,
                        "rate_hz": args.rate_hz,
                        "num_steps": args.num_steps,
                        "replan_steps": args.replan_steps,
                        "timing": timing,
                    }
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    if frames:
                        imageio.mimwrite(
                            output / f"task{task_id}_episode{episode}_{'success' if done else 'failure'}.mp4",
                            frames,
                            fps=args.rate_hz,
                        )
                    logging.info("Successes: %d/%d", successes, episodes)
            finally:
                env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
