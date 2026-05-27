# UMI UR5e WSG50 Deployment

This directory contains a self-contained real-robot listener for a single UMI UR5e setup with one UVC/GoPro camera and one WSG50 gripper.

For the full Chinese usage guide covering sync/async inference, telemetry, and trajectory visualization, see [`docs/ur5e-real-deployment-usage.md`](../docs/ur5e-real-deployment-usage.md).

Start the OpenPI policy server:

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  --lora-merge=auto \
  policy:checkpoint \
  --policy.config=pi0_umi_ur5e_pick_place_lora \
  --policy.dir=checkpoints/openpi-ur5e-lora/pi0_pick_place/29999
```

For the conveyor checkpoint, use the matching conveyor config:

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  --lora-merge=auto \
  policy:checkpoint \
  --policy.config=pi0_umi_ur5e_conveyor_lora \
  --policy.dir=checkpoints/openpi-ur5e-lora/pi0_conveyor/29999
```

For pi0.5, keep the same real-robot client and switch only the policy server config/checkpoint, for example `--policy.config=pi05_umi_ur5e_pick_place_lora`.

Use `--lora-merge=auto|on|off` on the policy server to control whether JAX LoRA checkpoints are folded into non-LoRA base weights at load time. `auto` is the default.

Run a server-only smoke test:

```bash
uv run --group umi deploy/inference_real.py \
  --policy-server-host=localhost \
  --policy-server-port=8000 \
  --dry-run
```

Observe hardware without policy inference:

```bash
uv run --group umi deploy/inference_real.py \
  --robot-config=deploy/configs/umi_ur5e_wsg50.yaml \
  --observe-only \
  --max-duration=10
```

By default this opens an OpenCV window showing `raw | policy`: the raw camera frame and the frame sent to OpenPI. The listener applies the UMI gripper mask before sending the image to OpenPI. Press `Esc` in the window to stop. Use `--no-show-camera` to disable the window.

The RTDE controller keeps UMI's interpolated `servoL` behavior and converts targets between the UMI unified frame and the robot base frame when `unified_tx` is configured. Add `--init-joints` if you want startup to move a single UR5e to UMI's default `[0, -90, -90, -90, 90, 0]` degree joint pose before running.

Run policy inference without executing actions:

```bash
uv run --group umi deploy/inference_real.py \
  --policy-server-host=<server-ip> \
  --policy-server-port=8000 \
  --robot-config=deploy/configs/umi_ur5e_wsg50.yaml \
  --prompt="<task instruction>" \
  --no-execute \
  --steps-per-inference=6 \
  --max-duration=10
```

Run async policy inference without executing actions:

```bash
uv run --group umi deploy/inference_real.py \
  --policy-server-host=<server-ip> \
  --policy-server-port=8000 \
  --robot-config=deploy/configs/umi_ur5e_wsg50.yaml \
  --prompt="<task instruction>" \
  --no-execute \
  --async-inference \
  --inference-overlap-steps=3 \
  --steps-per-inference=6 \
  --max-duration=20
```

Inference telemetry is written as JSONL by default:

```text
data/umi_real_inference/runs/<run_id>/telemetry/inference.jsonl
```

Each line is one policy call and includes:

- `inference_latency_ms`: wall-clock policy request latency for that call.
- `model_action_count`: number of actions returned by the model chunk.
- `scheduled_action_count`: number of future actions kept after latency filtering and `--steps-per-inference` / `--max-scheduled-actions` truncation.
- `executed_action_count`: number of scheduled actions actually sent to the robot; this is `0` when `--no-execute` is used.
- `dropped_action_count`: `model_action_count - scheduled_action_count`.
- `first_action`, `last_action`, `state`, and `scheduled_timestamps` for quick debugging.
- Async runs additionally include `async_policy_call_ms`, `async_chunk_boundary_wait_ms`, `async_hidden_inference_ms`, `async_overlap_steps`, and `async_future_state_applied`.
- `run_id`: the timestamped run identifier shared with videos, replay buffer, and metadata.

Use `--telemetry-path=<path>` to write the JSONL file somewhere else.

Each run also writes:

```text
data/umi_real_inference/runs/<run_id>/run_metadata.json
```

This records runtime arguments, robot config, policy server metadata, and denoising settings such as `serve_policy.num_steps` when the policy server is launched from this branch. Use `--run-id=<name>` to choose the run id, or `--no-timestamp-outputs` to keep writing directly under `--output-dir`.

Run real execution:

```bash
LOG=deploy/logs/inference_real_$(date +%Y%m%d_%H%M%S).log
echo "logging to $LOG"
uv run --group umi deploy/inference_real.py \
  --policy-server-host=localhost \
  --policy-server-port=8000 \
  --robot-config=deploy/configs/umi_ur5e_wsg50.yaml \
  --steps-per-inference=8 \
  --record-episode \
  --prompt="Pick up the red block and put it in the green box." 2>&1 | tee "$LOG"
```

Episode recording is enabled by default. Use `--no-record-episode` if you only want telemetry and metadata.

The UMI replay buffer is written under:

```text
data/umi_real_inference/runs/<run_id>/replay_buffer.zarr
```

It also saves raw UVC/fisheye camera video and per-frame timestamp sidecars:

```text
data/umi_real_inference/runs/<run_id>/videos/<episode_id>/<camera_idx>.mp4
data/umi_real_inference/runs/<run_id>/videos/<episode_id>/<camera_idx>.timestamps.jsonl
```

The replay buffer stores the executed action timeline, end-effector position, joint state, and WSG50 gripper width. Plot the latest episode trajectory with color mapped to gripper width:

```bash
RUN=data/umi_real_inference/runs/<run_id>
uv run --group umi --group dev deploy/plot_umi_trajectory.py \
  --replay-buffer=$RUN/replay_buffer.zarr \
  --episode-idx=-1 \
  --output=$RUN/trajectory_episode_last.png
```

The upper plot is the 3D UR5e end-effector trajectory; darker/lighter colors follow the selected matplotlib colormap and represent gripper width in meters. The lower plot shows gripper width over time.

For quick telemetry summaries:

```bash
python - <<'PY'
import json
import statistics

path = "data/umi_real_inference/runs/<run_id>/telemetry/inference.jsonl"
records = [json.loads(line) for line in open(path)]
latencies = [r["inference_latency_ms"] for r in records]
print("num_inferences", len(records))
print("latency_ms_mean", statistics.fmean(latencies))
print("latency_ms_max", max(latencies))
print("model_action_counts", sorted(set(r["model_action_count"] for r in records)))
print("scheduled_action_counts", sorted(set(r["scheduled_action_count"] for r in records)))
print("executed_action_counts", sorted(set(r["executed_action_count"] for r in records)))
PY
```
