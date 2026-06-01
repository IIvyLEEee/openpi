# ruff: noqa: E402, PLC0415, SIM117
import dataclasses
import logging
import math
from multiprocessing.managers import SharedMemoryManager
import os
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from deploy import async_inference as _async_inference
from deploy import latency_simulation as _latency_simulation
from deploy import run_metadata as _run_metadata
from deploy import telemetry as _telemetry

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    policy_server_host: str = "localhost"
    policy_server_port: int = 8000
    policy_server_api_key: str | None = None
    robot_config: pathlib.Path = pathlib.Path("deploy/configs/umi_ur5e_wsg50.yaml")
    prompt: str = "pick up the red block on the conveyor belt. put it in the blue box."
    output_dir: pathlib.Path = pathlib.Path("data/umi_real_inference")
    frequency: float = 10.0
    steps_per_inference: int | None = 6
    max_scheduled_actions: int | None = None
    max_duration: float = 120.0
    camera_obs_latency: float = 0.17
    action_exec_latency: float = 0.01
    inference_latency_scale: float = 1.0
    max_pos_speed: float = 2.0
    max_rot_speed: float = 6.0
    dry_run: bool = False
    observe_only: bool = False
    no_execute: bool = False
    show_camera: bool = True
    record_episode: bool = True
    telemetry_path: pathlib.Path | None = None
    async_inference: bool = False
    inference_overlap_steps: int = 0
    async_future_state: bool = True
    strict_sync: bool = False
    strict_sync_start_delay: float = 0.25
    strict_async: bool = False
    strict_async_overlap_steps: int = 0
    init_joints: bool = False
    run_id: str | None = None
    timestamp_outputs: bool = True


@dataclasses.dataclass(frozen=True)
class _StrictAsyncLaunchTiming:
    target_start_timestamp: float
    launch_timestamp: float
    launch_monotonic: float
    overlap_window_ms: float


@dataclasses.dataclass
class _PendingAsyncRequest:
    request: _async_inference.AsyncInferenceRequest
    target_start_timestamp: float
    launch_timestamp: float | None
    overlap_steps: int
    overlap_window_ms: float
    future_state_applied: bool
    state: np.ndarray


def _load_robot_config(path: pathlib.Path) -> dict:
    import yaml

    with path.expanduser().open("r") as f:
        config = yaml.safe_load(f)
    for key in ("cameras", "robots", "grippers"):
        if key not in config:
            raise ValueError(f"Missing {key!r} in {path}")
    if len(config["robots"]) != 1 or len(config["grippers"]) != 1:
        raise ValueError("This deployment only supports one UR5e and one WSG50 gripper.")
    return config


def _policy_observation_from_env_obs(obs: dict, prompt: str) -> dict:
    image = _apply_gripper_mask(np.asarray(obs["camera0_rgb"][-1], dtype=np.uint8))
    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"][-1], dtype=np.float32),
            np.asarray(obs["robot0_eef_rot_axis_angle"][-1], dtype=np.float32),
            np.asarray(obs["robot0_gripper_width"][-1], dtype=np.float32).reshape(1),
        ]
    ).astype(np.float32)
    if state.shape != (7,):
        raise ValueError(f"Expected 7D UMI state, got {state.shape}")
    return {
        "observation/image": image,
        "observation/state": state,
        "prompt": prompt,
    }


def _fake_policy_observation(prompt: str) -> dict:
    return {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": prompt,
    }


def _gripper_mask_polygons() -> np.ndarray:
    # Canonical polygons copied from the original UMI gripper mask logic.
    left_pts = [
        [1352, 1730],
        [1100, 1700],
        [650, 1500],
        [0, 1350],
        [0, 2028],
        [1352, 2704],
    ]
    img_shape = np.array([2028, 2704])
    left_coords = (np.asarray(left_pts) - img_shape[::-1] * 0.5) / img_shape[0]
    right_coords = left_coords.copy()
    right_coords[:, 0] *= -1
    return np.stack([left_coords, right_coords])


def _apply_gripper_mask(image: np.ndarray) -> np.ndarray:
    import cv2

    masked = np.ascontiguousarray(image.copy())
    for coords in _gripper_mask_polygons():
        pts = coords * masked.shape[0] + np.array(masked.shape[1::-1]) * 0.5
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(masked, [pts], color=(0, 0, 0), lineType=cv2.LINE_AA)
    return masked


def _normalize_action_chunk(result: dict) -> np.ndarray:
    if "actions" not in result:
        raise KeyError(f"Policy response does not contain 'actions': {result.keys()}")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected action chunk shape (T, 7), got {actions.shape}")
    return actions


def _apply_safety_filters(actions: np.ndarray, robot_config: dict) -> np.ndarray:
    from deploy.collision_utils import solve_table_collision

    filtered = np.asarray(actions, dtype=np.float32).copy()
    robots_config = robot_config["robots"]
    grippers_config = robot_config["grippers"]
    for target_pose in filtered:
        solve_table_collision(
            ee_pose=target_pose[:6],
            gripper_width=float(target_pose[6]),
            height_threshold=robots_config[0]["height_threshold"],
            finger_thickness=grippers_config[0]["finger_thickness"],
        )
    return filtered


def _future_action_schedule(
    actions: np.ndarray,
    obs_timestamp: float,
    eval_start_time: float,
    frequency: float,
    action_exec_latency: float,
    start_timestamp: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    scheduled_actions, timestamps, _ = _future_action_schedule_with_indices(
        actions=actions,
        obs_timestamp=obs_timestamp,
        eval_start_time=eval_start_time,
        frequency=frequency,
        action_exec_latency=action_exec_latency,
        start_timestamp=start_timestamp,
    )
    return scheduled_actions, timestamps


def _future_action_schedule_with_indices(
    actions: np.ndarray,
    obs_timestamp: float,
    eval_start_time: float,
    frequency: float,
    action_exec_latency: float,
    start_timestamp: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = 1.0 / frequency
    schedule_start = obs_timestamp if start_timestamp is None else start_timestamp
    timestamps = np.arange(len(actions), dtype=np.float64) * dt + schedule_start
    action_indices = np.arange(len(actions), dtype=np.int64)

    curr_time = time.time()
    is_new = timestamps > (curr_time + action_exec_latency)
    if np.any(is_new):
        return actions[is_new], timestamps[is_new], action_indices[is_new]

    next_step_idx = int(np.ceil((curr_time - eval_start_time) / dt))
    return (
        actions[[-1]],
        np.array([eval_start_time + next_step_idx * dt], dtype=np.float64),
        np.array([len(actions) - 1], dtype=np.int64),
    )


def _scheduled_action_limit(action_count: int, args: Args) -> int:
    max_scheduled_actions = args.max_scheduled_actions
    if max_scheduled_actions is None:
        max_scheduled_actions = args.steps_per_inference
    if max_scheduled_actions is None:
        return action_count
    return min(action_count, max_scheduled_actions)


def _strict_sync_action_schedule(
    actions: np.ndarray,
    inference_return_timestamp: float,
    frequency: float,
    start_delay: float,
    args: Args,
    start_timestamp: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_count = _scheduled_action_limit(len(actions), args)
    scheduled_actions = actions[:action_count]
    earliest_start = inference_return_timestamp + start_delay
    schedule_start = earliest_start if start_timestamp is None else max(float(start_timestamp), earliest_start)
    timestamps = schedule_start + np.arange(action_count, dtype=np.float64) / frequency
    action_indices = np.arange(action_count, dtype=np.int64)
    return scheduled_actions, timestamps, action_indices


def _schedule_actions_for_execution(
    *,
    actions: np.ndarray,
    obs_timestamp: float,
    eval_start_time: float,
    inference_return_timestamp: float,
    args: Args,
    start_timestamp: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.strict_sync or args.strict_async:
        return _strict_sync_action_schedule(
            actions=actions,
            inference_return_timestamp=inference_return_timestamp,
            frequency=args.frequency,
            start_delay=args.strict_sync_start_delay,
            args=args,
            start_timestamp=start_timestamp if args.strict_async else None,
        )

    scheduled_actions, timestamps, action_indices = _future_action_schedule_with_indices(
        actions=actions,
        obs_timestamp=obs_timestamp,
        eval_start_time=eval_start_time,
        frequency=args.frequency,
        action_exec_latency=args.action_exec_latency,
        start_timestamp=start_timestamp,
    )
    action_count = _scheduled_action_limit(len(scheduled_actions), args)
    return scheduled_actions[:action_count], timestamps[:action_count], action_indices[:action_count]


def _truncate_scheduled_actions(
    scheduled_actions: np.ndarray,
    timestamps: np.ndarray,
    args: Args,
) -> tuple[np.ndarray, np.ndarray]:
    action_count = _scheduled_action_limit(len(scheduled_actions), args)
    return scheduled_actions[:action_count], timestamps[:action_count]


def _next_cycle_wait_monotonic(
    *,
    loop_start_monotonic: float,
    eval_start_time: float,
    iter_idx: int,
    dt: float,
    timestamps: np.ndarray,
    strict_scheduled: bool,
) -> float:
    if strict_scheduled and len(timestamps) > 0:
        chunk_end_timestamp = float(timestamps[-1] + dt)
        return loop_start_monotonic + max(0.0, chunk_end_timestamp - eval_start_time)
    return loop_start_monotonic + iter_idx * dt


def _strict_async_launch_timing(
    *,
    timestamps: np.ndarray,
    dt: float,
    overlap_steps: int,
    loop_start_monotonic: float,
    eval_start_time: float,
) -> _StrictAsyncLaunchTiming:
    target_start_timestamp = float(timestamps[-1] + dt)
    launch_timestamp = target_start_timestamp - overlap_steps * dt
    return _StrictAsyncLaunchTiming(
        target_start_timestamp=target_start_timestamp,
        launch_timestamp=launch_timestamp,
        launch_monotonic=loop_start_monotonic + max(0.0, launch_timestamp - eval_start_time),
        overlap_window_ms=1000 * overlap_steps * dt,
    )


def _validate_runtime_args(args: Args) -> None:
    if args.frequency <= 0:
        raise ValueError("--frequency must be positive.")
    if args.steps_per_inference is not None and args.steps_per_inference <= 0:
        raise ValueError("--steps-per-inference must be positive when set.")
    if args.max_scheduled_actions is not None and args.max_scheduled_actions <= 0:
        raise ValueError("--max-scheduled-actions must be positive when set.")
    if not math.isfinite(args.inference_latency_scale) or args.inference_latency_scale < 1.0:
        raise ValueError("--inference-latency-scale must be >= 1.0.")
    if args.strict_sync and args.async_inference:
        raise ValueError("--strict-sync cannot be combined with --async-inference.")
    if args.strict_async and args.async_inference:
        raise ValueError("--strict-async cannot be combined with --async-inference.")
    if args.strict_async and args.strict_sync:
        raise ValueError("--strict-async cannot be combined with --strict-sync.")
    if args.async_inference and args.inference_overlap_steps <= 0:
        raise ValueError("--async-inference requires --inference-overlap-steps to be positive.")
    if not math.isfinite(args.strict_sync_start_delay) or args.strict_sync_start_delay < 0:
        raise ValueError("--strict-sync-start-delay must be finite and non-negative.")
    if args.strict_async:
        configured_action_count = args.max_scheduled_actions
        if configured_action_count is None:
            configured_action_count = args.steps_per_inference
        if configured_action_count is None:
            raise ValueError("--strict-async requires --steps-per-inference or --max-scheduled-actions.")
        if args.strict_async_overlap_steps <= 0:
            raise ValueError("--strict-async-overlap-steps must be positive.")
        if args.strict_async_overlap_steps > configured_action_count:
            raise ValueError("--strict-async-overlap-steps cannot exceed the fixed scheduled action count.")


def _show_camera(obs: dict, policy_image: np.ndarray | None = None) -> bool:
    import cv2

    image = np.ascontiguousarray(np.asarray(obs["camera0_rgb"][-1]).copy())
    if policy_image is not None:
        image = np.concatenate([image, policy_image], axis=1)
    cv2.putText(
        image,
        "raw | policy" if policy_image is not None else "camera0_rgb",
        (10, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        thickness=1,
        color=(255, 255, 255),
    )
    cv2.imshow("umi_ur5e_camera", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return cv2.waitKey(1) == 27


def _connect_policy(args: Args):
    from openpi_client import websocket_client_policy

    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_server_host,
        port=args.policy_server_port,
        api_key=args.policy_server_api_key,
    )
    logger.info("Server metadata: %s", policy.get_server_metadata())
    return policy


def _run_dry(args: Args) -> None:
    _validate_runtime_args(args)
    run_id = _run_metadata.resolve_run_id(args)
    run_paths = _run_metadata.resolve_run_paths(args, run_id=run_id)
    policy = _connect_policy(args)
    policy_infer = _latency_simulation.wrap_infer(policy.infer, latency_scale=args.inference_latency_scale)
    server_metadata = policy.get_server_metadata()
    _run_metadata.write_run_metadata(
        run_paths.metadata_path,
        _run_metadata.build_run_metadata(
            run_id=run_id,
            args=args,
            run_paths=run_paths,
            robot_config_path=None,
            robot_config=None,
            server_metadata=server_metadata,
        ),
    )
    policy_obs = _fake_policy_observation(args.prompt)
    infer_start = time.perf_counter()
    wall_time = time.time()
    result = policy_infer(policy_obs)
    infer_ms = 1000 * (time.perf_counter() - infer_start)
    actions = _normalize_action_chunk(result)
    telemetry_path = run_paths.telemetry_path
    with _telemetry.InferenceTelemetryRecorder(telemetry_path) as telemetry_recorder:
        telemetry_recorder.write(
            _telemetry.build_inference_record(
                iteration=0,
                loop_step=0,
                wall_time=wall_time,
                obs_timestamp=wall_time,
                inference_latency_ms=infer_ms,
                actions=actions,
                scheduled_actions=np.empty((0, actions.shape[1]), dtype=np.float32),
                scheduled_timestamps=np.empty((0,), dtype=np.float64),
                state=policy_obs["observation/state"],
                no_execute=True,
                steps_per_inference=args.steps_per_inference,
                run_id=run_id,
            )
        )
    logger.info(
        "Dry-run action chunk shape: %s, inference %.1f ms, telemetry=%s", actions.shape, infer_ms, telemetry_path
    )


def _run_real(args: Args) -> None:
    from deploy.umi.common.precise_sleep import precise_wait
    from deploy.umi.real_world.bimanual_umi_env import BimanualUmiEnv

    _validate_runtime_args(args)
    run_id = _run_metadata.resolve_run_id(args)
    run_paths = _run_metadata.resolve_run_paths(args, run_id=run_id)
    robot_config = _load_robot_config(args.robot_config)
    cameras_config = robot_config["cameras"]
    robots_config = robot_config["robots"]
    grippers_config = robot_config["grippers"]
    run_paths.output_dir.parent.mkdir(parents=True, exist_ok=True)

    policy = None if args.observe_only else _connect_policy(args)
    server_metadata = policy.get_server_metadata() if policy is not None else {}
    policy_infer = (
        _latency_simulation.wrap_infer(policy.infer, latency_scale=args.inference_latency_scale)
        if policy is not None
        else None
    )
    _run_metadata.write_run_metadata(
        run_paths.metadata_path,
        _run_metadata.build_run_metadata(
            run_id=run_id,
            args=args,
            run_paths=run_paths,
            robot_config_path=args.robot_config,
            robot_config=robot_config,
            server_metadata=server_metadata,
        ),
    )
    show_camera = args.show_camera
    if show_camera and not os.environ.get("DISPLAY"):
        logger.warning("DISPLAY is not set; disabling camera window.")
        show_camera = False

    dt = 1.0 / args.frequency
    telemetry_path = run_paths.telemetry_path
    with SharedMemoryManager() as shm_manager:
        with _telemetry.InferenceTelemetryRecorder(telemetry_path) as telemetry_recorder:
            with BimanualUmiEnv(
                output_dir=run_paths.output_dir,
                cameras_config=cameras_config,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=args.frequency,
                camera_obs_latency=args.camera_obs_latency,
                camera_obs_horizon=1,
                robot_obs_horizon=1,
                gripper_obs_horizon=1,
                max_pos_speed=args.max_pos_speed,
                max_rot_speed=args.max_rot_speed,
                init_joints=args.init_joints,
                shm_manager=shm_manager,
            ) as env:
                logger.info("Waiting for camera and robot buffers.")
                time.sleep(1.0)
                eval_start_time = time.time()
                if args.record_episode:
                    env.start_episode(start_time=eval_start_time)
                    logger.info("Recording replay buffer under %s", run_paths.replay_buffer_path)
                    logger.info("Recording camera videos under %s", run_paths.videos_dir)
                logger.info("Writing run metadata to %s", run_paths.metadata_path)
                logger.info("Writing inference telemetry to %s", telemetry_path)
                loop_start = time.monotonic()
                iter_idx = 0
                inference_idx = 0
                async_worker = (
                    _async_inference.AsyncInferenceWorker(policy_infer)
                    if policy_infer is not None and (args.async_inference or args.strict_async)
                    else None
                )
                pending_async: _PendingAsyncRequest | None = None

                try:
                    while True:
                        schedule_start_timestamp = None
                        async_metrics = {"async_mode": bool(async_worker)}

                        if pending_async is not None:
                            if async_worker is None:
                                raise RuntimeError("Pending async request exists without an async worker.")
                            wait_start = time.perf_counter()
                            async_result = async_worker.result()
                            inference_return_timestamp = time.time()
                            chunk_boundary_wait_ms = 1000 * (time.perf_counter() - wait_start)
                            result = async_result.response
                            wall_time = async_result.wall_time
                            obs_timestamp = async_result.obs_timestamp
                            state = pending_async.state
                            infer_ms = async_result.policy_call_ms
                            schedule_start_timestamp = pending_async.target_start_timestamp
                            async_metrics.update(
                                {
                                    "async_request_id": pending_async.request.request_id,
                                    "async_overlap_steps": pending_async.overlap_steps,
                                    "async_overlap_window_ms": pending_async.overlap_window_ms,
                                    "async_policy_call_ms": async_result.policy_call_ms,
                                    "async_chunk_boundary_wait_ms": chunk_boundary_wait_ms,
                                    "async_hidden_inference_ms": max(
                                        0.0, async_result.policy_call_ms - chunk_boundary_wait_ms
                                    ),
                                    "async_future_state_applied": pending_async.future_state_applied,
                                    "async_target_start_timestamp": pending_async.target_start_timestamp,
                                    "async_launch_timestamp": pending_async.launch_timestamp,
                                }
                            )
                            if args.strict_async:
                                async_metrics.update(
                                    {
                                        "strict_async_mode": True,
                                        "strict_async_overlap_steps": pending_async.overlap_steps,
                                        "strict_async_overlap_window_ms": pending_async.overlap_window_ms,
                                        "strict_async_target_start_timestamp": pending_async.target_start_timestamp,
                                        "strict_async_launch_timestamp": pending_async.launch_timestamp,
                                        "strict_async_boundary_wait_ms": chunk_boundary_wait_ms,
                                    }
                                )
                            pending_async = None
                        else:
                            obs = env.get_obs()

                            policy_obs = _policy_observation_from_env_obs(obs, args.prompt)
                            if show_camera:
                                try:
                                    if _show_camera(obs, policy_obs["observation/image"]):
                                        logger.info("ESC pressed; stopping.")
                                        break
                                except Exception as exc:
                                    logger.warning("cv2.imshow failed (%s); disabling camera window.", exc)
                                    show_camera = False
                            logger.info(
                                "obs state=%s image_shape=%s",
                                np.array2string(policy_obs["observation/state"], precision=4),
                                policy_obs["observation/image"].shape,
                            )

                            if args.observe_only:
                                iter_idx += 1
                                t_cycle_end = loop_start + iter_idx * dt
                                if time.time() - eval_start_time > args.max_duration:
                                    logger.info("Max duration reached.")
                                    break
                                precise_wait(t_cycle_end)
                                continue

                            if policy_infer is None:
                                raise RuntimeError("Policy is not connected.")
                            infer_start = time.perf_counter()
                            wall_time = time.time()
                            result = policy_infer(policy_obs)
                            inference_return_timestamp = time.time()
                            infer_ms = 1000 * (time.perf_counter() - infer_start)
                            obs_timestamp = float(obs["timestamp"][-1])
                            state = policy_obs["observation/state"]
                            if async_worker is None:
                                async_metrics = {"async_mode": False}

                        actions = _normalize_action_chunk(result)
                        actions = _apply_safety_filters(actions, robot_config)

                        scheduled_actions, timestamps, scheduled_action_indices = _schedule_actions_for_execution(
                            actions=actions,
                            obs_timestamp=obs_timestamp,
                            eval_start_time=eval_start_time,
                            inference_return_timestamp=inference_return_timestamp,
                            args=args,
                            start_timestamp=schedule_start_timestamp,
                        )
                        if args.strict_async:
                            schedule_mode = "strict_async"
                        elif args.strict_sync:
                            schedule_mode = "strict_sync"
                        else:
                            schedule_mode = "timestamp_filter"

                        telemetry_recorder.write(
                            _telemetry.build_inference_record(
                                iteration=inference_idx,
                                loop_step=iter_idx,
                                wall_time=wall_time,
                                obs_timestamp=obs_timestamp,
                                inference_latency_ms=infer_ms,
                                actions=actions,
                                scheduled_actions=scheduled_actions,
                                scheduled_timestamps=timestamps,
                                scheduled_action_indices=scheduled_action_indices,
                                state=state,
                                no_execute=args.no_execute,
                                steps_per_inference=args.steps_per_inference,
                                async_metrics=async_metrics,
                                schedule_mode=schedule_mode,
                                run_id=run_id,
                            )
                        )
                        inference_idx += 1

                        logger.info(
                            "inference %.1f ms, got %d actions, scheduled %d%s, mode=%s, first_action_index=%s, "
                            "first_delta=%s, last_delta=%s",
                            infer_ms,
                            len(actions),
                            len(scheduled_actions),
                            " (no-execute)" if args.no_execute else "",
                            schedule_mode,
                            int(scheduled_action_indices[0]) if len(scheduled_action_indices) > 0 else None,
                            np.array2string(actions[0, :6] - state[:6], precision=4),
                            np.array2string(actions[-1, :6] - state[:6], precision=4),
                        )
                        if not args.no_execute:
                            env.exec_actions(
                                actions=scheduled_actions,
                                timestamps=timestamps,
                                compensate_latency=True,
                            )

                        if time.time() - eval_start_time > args.max_duration:
                            logger.info("Max duration reached.")
                            break

                        if args.max_scheduled_actions is not None:
                            step_advance = args.max_scheduled_actions
                        elif args.steps_per_inference is not None:
                            step_advance = args.steps_per_inference
                        else:
                            step_advance = len(actions)
                        iter_idx += max(1, step_advance)
                        t_cycle_end = _next_cycle_wait_monotonic(
                            loop_start_monotonic=loop_start,
                            eval_start_time=eval_start_time,
                            iter_idx=iter_idx,
                            dt=dt,
                            timestamps=timestamps,
                            strict_scheduled=args.strict_sync or args.strict_async,
                        )

                        if async_worker is not None and len(scheduled_actions) > 0:
                            if args.strict_async:
                                overlap_steps = args.strict_async_overlap_steps
                                launch_timing = _strict_async_launch_timing(
                                    timestamps=timestamps,
                                    dt=dt,
                                    overlap_steps=overlap_steps,
                                    loop_start_monotonic=loop_start,
                                    eval_start_time=eval_start_time,
                                )
                            else:
                                overlap_steps = min(args.inference_overlap_steps, len(scheduled_actions))
                                launch_timing = _strict_async_launch_timing(
                                    timestamps=timestamps,
                                    dt=dt,
                                    overlap_steps=overlap_steps,
                                    loop_start_monotonic=loop_start,
                                    eval_start_time=eval_start_time,
                                )
                            precise_wait(launch_timing.launch_monotonic)

                            launch_obs = env.get_obs()
                            launch_policy_obs = _policy_observation_from_env_obs(launch_obs, args.prompt)
                            future_state_applied = False
                            if args.async_inference and args.async_future_state:
                                launch_policy_obs = _async_inference.apply_future_state(
                                    launch_policy_obs, scheduled_actions[-1]
                                )
                                future_state_applied = True

                            request = async_worker.submit(
                                launch_policy_obs,
                                obs_timestamp=float(launch_obs["timestamp"][-1]),
                            )
                            pending_async = _PendingAsyncRequest(
                                request=request,
                                target_start_timestamp=launch_timing.target_start_timestamp,
                                launch_timestamp=launch_timing.launch_timestamp,
                                overlap_steps=overlap_steps,
                                overlap_window_ms=launch_timing.overlap_window_ms,
                                future_state_applied=future_state_applied,
                                state=launch_policy_obs["observation/state"],
                            )
                            logger.info(
                                "launched async inference request %d with overlap_steps=%d, "
                                "target_start=%.6f, future_state=%s",
                                request.request_id,
                                overlap_steps,
                                launch_timing.target_start_timestamp,
                                future_state_applied,
                            )

                        precise_wait(t_cycle_end)
                finally:
                    if async_worker is not None:
                        async_worker.close()


def main(args: Args) -> None:
    if args.dry_run:
        _run_dry(args)
    else:
        _run_real(args)


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
