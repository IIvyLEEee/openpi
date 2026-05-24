# UMI UR5e WSG50 Deployment

This directory contains a self-contained real-robot listener for a single UMI UR5e setup with one UVC/GoPro camera and one WSG50 gripper.

Start the OpenPI policy server:

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi0_umi_ur5e_pick_place_lora \
  --policy.dir=checkpoints/openpi-ur5e-lora/pi0_pick_place/29999
```

For the conveyor checkpoint, use the matching conveyor config:

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi0_umi_ur5e_conveyor_lora \
  --policy.dir=checkpoints/openpi-ur5e-lora/pi0_conveyor/29999
```

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

Inference telemetry is written as JSONL by default:

```text
data/umi_real_inference/telemetry/inference.jsonl
```

Each line is one policy call and includes:

- `inference_latency_ms`: wall-clock policy request latency for that call.
- `model_action_count`: number of actions returned by the model chunk.
- `scheduled_action_count`: number of future actions kept after latency filtering and `--steps-per-inference` / `--max-scheduled-actions` truncation.
- `executed_action_count`: number of scheduled actions actually sent to the robot; this is `0` when `--no-execute` is used.
- `dropped_action_count`: `model_action_count - scheduled_action_count`.
- `first_action`, `last_action`, `state`, and `scheduled_timestamps` for quick debugging.

Use `--telemetry-path=<path>` to write the JSONL file somewhere else.

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

`--record-episode` enables the UMI replay buffer under:

```text
data/umi_real_inference/replay_buffer.zarr
```

That replay buffer stores the executed action timeline, end-effector position, joint state, and WSG50 gripper width. Plot the latest episode trajectory with color mapped to gripper width:

```bash
uv run --group umi --group dev deploy/plot_umi_trajectory.py \
  --replay-buffer=data/umi_real_inference/replay_buffer.zarr \
  --episode-idx=-1 \
  --output=data/umi_real_inference/trajectory_episode_last.png
```

The upper plot is the 3D UR5e end-effector trajectory; darker/lighter colors follow the selected matplotlib colormap and represent gripper width in meters. The lower plot shows gripper width over time.

For quick telemetry summaries:

```bash
python - <<'PY'
import json
import statistics

path = "data/umi_real_inference/telemetry/inference.jsonl"
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
