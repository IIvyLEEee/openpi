# UMI RTDE Controller Migration Log

## Scope

Migrate the relevant behavior from `/home/wyj24/project/realworld-pi/umi_for_train` into the OpenPI branch for UR5e real-world runs used by both `pi0` and `pi0.5` policies.

## Source Reading

`umi_for_train/umi/real_world/rtde_interpolation_controller.py` lines 221-247 start the RTDE control and receive interfaces, set TCP and payload parameters, optionally move the robot to `joints_init`, then enter a 500 Hz UR5e control loop. The source controller stores and commands poses in the robot base frame.

The source `SERVOL` command path updates the existing `PoseTrajectoryInterpolator` by calling `drive_to_waypoint()` with `curr_time = t_now + dt` and `t_insert = curr_time + duration`. This avoids restarting interpolation from lagging measured TCP pose, which would introduce discontinuities.

## Target Differences

The target OpenPI controller already adds `unified_tx`, converting received TCP poses into the UMI unified frame and scheduled waypoints back into the robot base frame. However, the `SERVOL` command path is disabled by an assertion because that coordinate conversion was not implemented there.

`deploy/umi/real_world/bimanual_umi_env.py` is single-UR5e oriented for this branch, but its boolean `init_joints=True` path still used a fixed two-arm joint list. The UMI source uses `[0, -90, -90, -90, 90, 0]` degrees as the default UR5e init pose.

## Migration Plan

1. Re-enable `SERVOL` by preserving the source interpolation update behavior while converting unified-frame targets back into the robot base frame before `drive_to_waypoint()`.
2. Add a small tested helper for UR5e init joint selection so one-arm pi0/pi0.5 experiments use the UMI default, while two-arm explicit behavior remains possible.
3. Document that the migrated controller path is model-agnostic and therefore applies to both pi0 and pi0.5 real robot deployments.

## Implementation

`deploy/umi/real_world/rtde_interpolation_controller.py` now has `_apply_servol_command()`, a direct migration of the UMI `SERVOL` interpolation update. The run loop passes `pose_to_robot_frame=lambda pose: self.convert_pose_to_unified_frame(pose, backward=True)`, so `SERVOL` and `SCHEDULE_WAYPOINT` both accept unified-frame poses at the public controller boundary and drive robot-base waypoints internally.

`deploy/umi/real_world/robot_init.py` centralizes init joint selection. Boolean `init_joints=True` with one robot now resolves to UMI's single-UR5e default. Two robots keep the branch's existing two-arm presets. Explicit joint vectors are validated as 6D vectors.

`deploy/inference_real.py` exposes `--init-joints` and passes it into `BimanualUmiEnv`; the default remains `False` to avoid unexpected motion during smoke tests.

## Verification

- `python -m pytest deploy/rtde_interpolation_controller_test.py deploy/robot_init_test.py deploy/async_inference_test.py deploy/telemetry_test.py deploy/inference_real_test.py deploy/plot_umi_trajectory_test.py`
- `ruff check deploy/umi/real_world/robot_init.py deploy/rtde_interpolation_controller_test.py deploy/robot_init_test.py deploy/inference_real.py`
- `ruff check --select E9,F63,F7,F82 deploy/umi/real_world/rtde_interpolation_controller.py deploy/umi/real_world/bimanual_umi_env.py deploy/umi/real_world/robot_init.py deploy/inference_real.py deploy/rtde_interpolation_controller_test.py deploy/robot_init_test.py`
