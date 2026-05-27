# UR5e 真机部署使用说明

本文档说明这个分支中 UR5e + WSG50 真机部署相关改动的使用方法，包括同步推理、异步推理、推理遥测、动作 chunk 统计、轨迹和夹爪状态可视化。

## 1. 启动策略服务器

先在一端启动 OpenPI policy server。普通 pick-place checkpoint 示例：

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  --lora-merge=auto \
  policy:checkpoint \
  --policy.config=pi0_umi_ur5e_pick_place_lora \
  --policy.dir=checkpoints/openpi-ur5e-lora/pi0_pick_place/29999
```

conveyor checkpoint 示例：

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  --lora-merge=auto \
  policy:checkpoint \
  --policy.config=pi0_umi_ur5e_conveyor_lora \
  --policy.dir=checkpoints/openpi-ur5e-lora/pi0_conveyor/29999
```

pi0.5 使用同一套 UR5e 真机客户端，只需要把 policy config 和 checkpoint 换成 pi0.5：

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  --lora-merge=auto \
  policy:checkpoint \
  --policy.config=pi05_umi_ur5e_pick_place_lora \
  --policy.dir=checkpoints/openpi-ur5e-lora/pi05_pick_place/29999
```

LoRA merge 开关在 policy server 端：

- `--lora-merge=auto`: 默认值。如果 checkpoint 和 config 都包含可合并的 LoRA 参数，加载时自动折叠到非 LoRA 主权重；否则保持原行为。
- `--lora-merge=on`: 强制合并。若不是 JAX LoRA checkpoint，或没有可合并的 LoRA 参数，会报错。
- `--lora-merge=off`: 不合并。推理时继续走 base 分支 + LoRA 分支实时相加。

先做一次 server-only smoke test：

```bash
uv run --group umi deploy/inference_real.py \
  --policy-server-host=localhost \
  --policy-server-port=8000 \
  --dry-run
```

`--dry-run` 只请求一次模型，不连接真机。

## 1.1. UR5e RTDE 控制器行为

当前真机客户端使用 UMI 的 RTDE 插值控制方式：控制器进程连接 UR5e 后会设置 TCP offset / payload，并以 UR5e 的 500Hz 循环持续向机器人发送 `servoL`。每个 action chunk 会通过 `schedule_waypoint()` 排入 `PoseTrajectoryInterpolator`，避免从滞后的实测 TCP pose 重新起轨导致抖动。

如果 `deploy/configs/umi_ur5e_wsg50.yaml` 中配置了 `unified_tx`，观测到的 `ActualTCPPose` / `TargetTCPPose` 会从 robot base 转到 UMI unified frame；下发的 `schedule_waypoint()` 和 `servoL()` 目标会先从 unified frame 转回 robot base。这个路径与模型无关，因此 pi0 和 pi0.5 共用。

默认不会移动到初始化关节位。如果需要在启动环境时先移动到 UMI 默认单臂 UR5e 姿态 `[0, -90, -90, -90, 90, 0]` 度，可以给 `deploy/inference_real.py` 加：

```bash
--init-joints
```

注意：`--init-joints` 在 `--observe-only` 和 `--no-execute` 下也会生效，因为它发生在环境启动阶段。

## 2. 观察硬件但不推理

```bash
uv run --group umi deploy/inference_real.py \
  --robot-config=deploy/configs/umi_ur5e_wsg50.yaml \
  --observe-only \
  --max-duration=10
```

默认会打开 OpenCV 窗口，左边是原始相机图，右边是送入 OpenPI 的图。脚本会在送入模型前应用 UMI gripper mask。没有显示器或不需要窗口时加：

```bash
--no-show-camera
```

## 3. 同步推理运行

先用 `--no-execute` 检查模型输出和遥测，不给机械臂下发动作：

```bash
uv run --group umi deploy/inference_real.py \
  --policy-server-host=<server-ip> \
  --policy-server-port=8000 \
  --robot-config=deploy/configs/umi_ur5e_wsg50.yaml \
  --prompt="Pick up the red block and put it in the green box." \
  --no-execute \
  --steps-per-inference=6 \
  --max-duration=10
```

确认输出正常后再真机执行：

```bash
LOG=deploy/logs/inference_real_$(date +%Y%m%d_%H%M%S).log
uv run --group umi deploy/inference_real.py \
  --policy-server-host=<server-ip> \
  --policy-server-port=8000 \
  --robot-config=deploy/configs/umi_ur5e_wsg50.yaml \
  --prompt="Pick up the red block and put it in the green box." \
  --steps-per-inference=6 \
  --record-episode \
  --max-duration=60 2>&1 | tee "$LOG"
```

`--record-episode` 现在默认开启；上面的参数可以保留，也可以省略。若只想保存 telemetry / metadata，不想保存 replay buffer 和相机视频，使用 `--no-record-episode`。

## 4. 异步推理运行

异步模式让机械臂执行当前 action chunk 时，在后台提前请求下一 action chunk。首个 chunk 仍然同步等待；第二个 chunk 开始才能隐藏推理延迟。

推荐先用小 overlap 和 `--no-execute` 做测试：

```bash
uv run --group umi deploy/inference_real.py \
  --policy-server-host=<server-ip> \
  --policy-server-port=8000 \
  --robot-config=deploy/configs/umi_ur5e_wsg50.yaml \
  --prompt="Pick up the red block and put it in the green box." \
  --no-execute \
  --async-inference \
  --inference-overlap-steps=3 \
  --steps-per-inference=6 \
  --max-duration=20
```

真机异步执行：

```bash
LOG=deploy/logs/inference_real_async_$(date +%Y%m%d_%H%M%S).log
uv run --group umi deploy/inference_real.py \
  --policy-server-host=<server-ip> \
  --policy-server-port=8000 \
  --robot-config=deploy/configs/umi_ur5e_wsg50.yaml \
  --prompt="Pick up the red block and put it in the green box." \
  --async-inference \
  --inference-overlap-steps=3 \
  --steps-per-inference=6 \
  --record-episode \
  --max-duration=60 2>&1 | tee "$LOG"
```

参数含义：

- `--async-inference`: 开启后台推理。
- `--inference-overlap-steps=N`: 当前已排程 chunk 还剩 `N` 个控制步时，发起下一次推理。
- `--async-future-state`: 默认开启，把下一次推理的 `observation/state` 替换为当前 chunk 边界附近的目标 TCP pose + gripper width。
- `--no-async-future-state`: 关闭 future-state 替换，用发起推理时的真实观测 state 做对照实验。

overlap 对应的时间窗口为：

```text
overlap_window_seconds = inference_overlap_steps / frequency
```

例如默认 `frequency=10Hz`，`--inference-overlap-steps=3` 表示最多隐藏约 `300ms` 推理时间。`inference_overlap_steps` 不应大于实际排程动作数；代码会按当前 `scheduled_action_count` 做 clamp。

## 5. 遥测文件和指标

默认每次运行会生成一个带本地时间戳的 run id，并把该次运行的输出放在：

```text
data/umi_real_inference/runs/<YYYYMMDD_HHMMSS>/
```

可以用 `--run-id=<name>` 手动指定 run id；如果要恢复旧行为、不创建时间戳子目录，可以加 `--no-timestamp-outputs`。

默认遥测路径：

```text
data/umi_real_inference/runs/<run_id>/telemetry/inference.jsonl
```

可用 `--telemetry-path=<path>` 改路径。每一行是一条 JSON，对应一次模型 action chunk，并包含同一个 `run_id`。

每次运行还会写：

```text
data/umi_real_inference/runs/<run_id>/run_metadata.json
```

其中包含 prompt、frequency、`steps_per_inference`、async 设置、是否 `--record-episode`、机器人配置、policy server metadata。policy server 如果用当前分支的 `scripts/serve_policy.py` 启动，metadata 里会包含 `serve_policy.num_steps` 和 `serve_policy.log_denoise_steps`，用于记录当前去噪步数设置。

基础字段：

- `inference_latency_ms`: 模型请求耗时。异步模式下它仍然是原始 policy call 时间，不会因为 async 变小。
- `model_action_count`: 模型返回的动作数。
- `scheduled_action_count`: 过滤过期动作并按 `steps_per_inference` / `max_scheduled_actions` 截断后，实际排程的动作数。
- `executed_action_count`: 实际下发给 robot env 的动作数；`--no-execute` 时为 `0`。
- `dropped_action_count`: `model_action_count - scheduled_action_count`。
- `state`, `first_action`, `last_action`, `scheduled_timestamps`: 快速检查模型输出和排程时间。

异步字段：

- `async_mode`: 该 run 是否处于 async 模式。
- `async_request_id`: 后台请求编号。
- `async_overlap_steps`: 本次请求使用的 overlap 步数。
- `async_overlap_window_ms`: overlap 对应的理论时间窗口。
- `async_policy_call_ms`: 后台 policy call 原始耗时。
- `async_chunk_boundary_wait_ms`: 到达 chunk 边界后，为等待后台结果实际阻塞的时间。
- `async_hidden_inference_ms`: 被当前 chunk 执行过程隐藏掉的推理时间。
- `async_future_state_applied`: 是否应用了 future-state 替换。
- `async_target_start_timestamp`: 下一 chunk 第一个动作的目标开始时间。

判断 async 是否真的降低 per-chunk delay，重点看：

```text
async_chunk_boundary_wait_ms << async_policy_call_ms
async_hidden_inference_ms 接近 async_policy_call_ms
```

原始 `inference_latency_ms` 不应该明显下降；下降的是 chunk 边界可见等待时间。

快速统计脚本：

```bash
python - <<'PY'
import json
import statistics

path = "data/umi_real_inference/runs/<run_id>/telemetry/inference.jsonl"
records = [json.loads(line) for line in open(path)]

lat = [r["inference_latency_ms"] for r in records]
wait = [r["async_chunk_boundary_wait_ms"] for r in records if "async_chunk_boundary_wait_ms" in r]
hidden = [r["async_hidden_inference_ms"] for r in records if "async_hidden_inference_ms" in r]

print("num_records", len(records))
print("raw_inference_ms_mean", statistics.fmean(lat) if lat else None)
print("raw_inference_ms_max", max(lat) if lat else None)
print("async_boundary_wait_ms_mean", statistics.fmean(wait) if wait else None)
print("async_boundary_wait_ms_max", max(wait) if wait else None)
print("async_hidden_inference_ms_mean", statistics.fmean(hidden) if hidden else None)
print("scheduled_action_counts", sorted(set(r["scheduled_action_count"] for r in records)))
print("executed_action_counts", sorted(set(r["executed_action_count"] for r in records)))
PY
```

## 6. 轨迹和夹爪可视化

真机运行默认会记录 UMI replay buffer：

```text
data/umi_real_inference/runs/<run_id>/replay_buffer.zarr
```

同时保存每个相机的原始 UVC/鱼眼视频：

```text
data/umi_real_inference/runs/<run_id>/videos/<episode_id>/<camera_idx>.mp4
data/umi_real_inference/runs/<run_id>/videos/<episode_id>/<camera_idx>.timestamps.jsonl
```

`.timestamps.jsonl` 每行对应一帧，包含 `frame_idx`、`timestamp`、`camera_capture_timestamp` 和 `camera_receive_timestamp`，方便后处理时和机器人轨迹/动作 chunk 对齐。

如果加 `--no-record-episode`，仍会保存 telemetry 和 `run_metadata.json`，但不会保存 replay buffer 轨迹和相机视频。

记录停止时机：

- 按 OpenCV 窗口里的 `Esc` 后，主循环退出，环境 context manager 会调用 `env.stop()`，随后 `end_episode()` 停止相机录像并写入 zarr。
- 达到 `--max-duration` 后同样会走 `end_episode()`。
- 正常 `Ctrl+C` 通常也会触发 context manager 清理；直接 kill 进程或断电则可能留下未 finalize 的视频/zarr。
- `--no-execute` 或 `--observe-only` 下没有实际 action 时间轴，因此会保存视频、metadata、telemetry；replay buffer 轨迹 episode 可能不会写入，因为当前 zarr 以已排程/执行 action 时间戳为主时间轴。

运行结束后用对应 run 目录绘图：

```bash
RUN=data/umi_real_inference/runs/<run_id>
uv run --group umi --group dev deploy/plot_umi_trajectory.py \
  --replay-buffer=$RUN/replay_buffer.zarr \
  --episode-idx=-1 \
  --output=$RUN/trajectory_episode_last.png
```

为了让实验输出更容易对照，建议同时保留：

```text
run_metadata.json
telemetry/inference.jsonl
replay_buffer.zarr
videos/<episode_id>/*.mp4
trajectory_episode_last.png
```

图上半部分是 UR5e end-effector 3D 轨迹，颜色随 `robot0_gripper_width` 变化；下半部分是夹爪宽度随时间变化曲线。

## 7. 建议实验顺序

1. `--dry-run` 确认 policy server 可用。
2. `--observe-only` 确认相机、UR5e、WSG50 buffer 正常。
3. 同步 `--no-execute` 跑 10 到 20 秒，看 `inference_latency_ms` 和 action chunk 数。
4. 同步真机短跑，确认安全滤波和任务 prompt 正常。
5. 异步 `--no-execute --async-inference --inference-overlap-steps=2` 起步。
6. 异步真机从 `overlap=2` 或 `3` 起步，逐步增加，观察 `async_chunk_boundary_wait_ms`、轨迹平滑性和任务成功率。
7. 如果大 overlap 下动作不稳，先对比 `--no-async-future-state`，再考虑按 VLASH 思路做 delay-offset fine-tuning。

## 8. 注意事项

- async 降低的是 chunk 边界等待，不是模型本身推理耗时。
- 第一个 chunk 必然同步等待，统计 async 效果时应重点看带 `async_request_id` 的后续记录。
- 如果 `async_policy_call_ms > async_overlap_window_ms`，边界仍会有剩余等待。
- 如果 policy server 或 websocket client 仍在同一线程阻塞运行，就不会得到真正 overlap；当前实现使用单请求后台 worker。
- 真机测试前先使用 `--no-execute`，并保持急停和 workspace 限制可用。
