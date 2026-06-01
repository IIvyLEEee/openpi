# Real-Robot Timing Analysis: Why Current Sync Is Not the Paper's Clean Sync

本文档分析 `deploy/inference_real.py` 这套 UR5e + WSG50 真机部署路径和论文
`The_Speedup_Paradox__Rethinking_Inference_Optimization_in_Embodied_Tasks.pdf` 中理想化同步设置之间的差异。

结论先行：当前框架在 `--async-inference=false` 时确实没有显式后台推理，但它仍然不是论文中那种干净的同步执行。它采用的是时间戳排程式同步：模型输出一个完整 action chunk，客户端根据真实 wall-clock 时间丢弃已经过期的动作，再截断为本轮要下发的动作子集，并交给独立 500Hz RTDE 控制器插值执行。因此，单纯放大 `T_inf` 会改变实际执行的 action index、有效 `n`、甚至低层控制器实际接受的 waypoint 数。

## 1. 论文中的理想同步假设

论文第 3 节的静态任务模型可以抽象为：

```text
T_chunk = T_inf + n * T_act
T_task = N * T_chunk
rho = T_inf / T_act
```

对应的关键假设是：

- 对同一个静态任务、同一个模型、同一个轻量化设置，只改变推理延迟 `T_inf`。
- `T_act` 固定。
- 每个 chunk 执行的动作数 `n` 固定。
- 推理变慢本身不改变模型输出被执行的动作序列。
- 静态环境在推理期间不发生任务相关变化。

论文第 6 节 limitations 也明确承认：为了隔离闭环效应，分析抽象掉低层执行细节；对于静态任务，还假设只改变 inference latency 不改变轨迹。这一点和当前真机框架有直接冲突。

## 2. 当前真机框架的实际执行链路

### 2.1 模型输出 chunk 长度是 30

UMI UR5e 的 pi0/pi0.5 配置中，模型 action horizon 是 30。例如 pi0.5 pick-place：

- `src/openpi/training/config.py:1114-1118`

```python
TrainConfig(
    name="pi05_umi_ur5e_pick_place_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=7,
        action_horizon=30,
```

因此模型每次推理返回的是 30 个动作，而不是直接返回本轮实际全部执行的动作数。

### 2.2 真机客户端只计划执行 chunk 的一个子集

运行参数在 `deploy/inference_real.py` 中定义：

- `deploy/inference_real.py:25-50`

```python
frequency: float = 10.0
steps_per_inference: int | None = 6
max_scheduled_actions: int | None = None
action_exec_latency: float = 0.01
async_inference: bool = False
inference_overlap_steps: int = 0
```

这意味着默认控制节拍是 10 Hz，即 `T_act = 0.1s`。但每次模型返回 30 个动作后，客户端通常只会保留 `steps_per_inference` 个动作。实际实验 run metadata 中常见设置是 `steps_per_inference=8`。

截断逻辑：

- `deploy/inference_real.py:181-192`

```python
max_scheduled_actions = args.max_scheduled_actions
if max_scheduled_actions is None:
    max_scheduled_actions = args.steps_per_inference
if max_scheduled_actions is not None:
    scheduled_actions = scheduled_actions[:max_scheduled_actions]
    timestamps = timestamps[:max_scheduled_actions]
```

因此，论文公式里的 `n` 不应直接取 `action_horizon=30`，而应至少取本轮 runtime 的 `scheduled_action_count`，进一步还要考虑低层控制器是否真正接受这些 waypoint。

### 2.3 同步路径会根据 wall-clock 丢弃过期动作

同步推理路径：

- `deploy/inference_real.py:417-421`

```python
infer_start = time.perf_counter()
wall_time = time.time()
result = policy_infer(policy_obs)
infer_ms = 1000 * (time.perf_counter() - infer_start)
obs_timestamp = float(obs["timestamp"][-1])
```

推理结束后，客户端不是从 `actions[0]` 开始盲目执行，而是调用 `_future_action_schedule()`：

- `deploy/inference_real.py:160-178`

```python
dt = 1.0 / frequency
schedule_start = obs_timestamp if start_timestamp is None else start_timestamp
timestamps = np.arange(len(actions), dtype=np.float64) * dt + schedule_start

curr_time = time.time()
is_new = timestamps > (curr_time + action_exec_latency)
if np.any(is_new):
    return actions[is_new], timestamps[is_new]

next_step_idx = int(np.ceil((curr_time - eval_start_time) / dt))
return actions[[-1]], np.array([eval_start_time + next_step_idx * dt], dtype=np.float64)
```

这个逻辑的含义是：

```text
first_retained_index ~= ceil((now_after_infer + action_exec_latency - obs_timestamp) / dt)
```

所以当你只放大 `T_inf` 时，实际执行的动作不是同一个 chunk 的同一个前缀，而是从更靠后的 action index 开始执行。如果 `T_inf` 足够大，前面大量动作都会被过滤掉；极端情况下只 fallback 到最后一个动作。

这已经违反了论文理想同步中的“只改变 `T_inf`，执行动作序列不变”的假设。

### 2.4 `env.exec_actions()` 还会再次过滤未来时间戳

下发动作：

- `deploy/inference_real.py:471-476`

```python
env.exec_actions(
    actions=scheduled_actions,
    timestamps=timestamps,
    compensate_latency=True,
)
```

`env.exec_actions()` 中再次过滤：

- `deploy/umi/real_world/bimanual_umi_env.py:420-431`

```python
receive_time = time.time()
is_new = timestamps > receive_time
new_actions = actions[is_new]
new_timestamps = timestamps[is_new]
```

然后对 UR5e 和 gripper 分别做 action latency compensation：

- `deploy/umi/real_world/bimanual_umi_env.py:441-446`

```python
r_latency = rc["robot_action_latency"] if compensate_latency else 0.0
g_latency = gc["gripper_action_latency"] if compensate_latency else 0.0
robot.schedule_waypoint(pose=r_actions, target_time=new_timestamps[i] - r_latency)
gripper.schedule_waypoint(pos=g_actions, target_time=new_timestamps[i] - g_latency)
```

配置文件中：

- `deploy/configs/umi_ur5e_wsg50.yaml:15-17`
- `deploy/configs/umi_ur5e_wsg50.yaml:31-32`

```yaml
robot_obs_latency: 0.0001
robot_action_latency: 0.1
gripper_obs_latency: 0.01
gripper_action_latency: 0.1
```

注意：`_future_action_schedule()` 默认只看 `action_exec_latency=0.01`，但低层实际下发时会减去 `robot_action_latency=0.1`。这会导致一些看似未来的 waypoint，在进入 500Hz 控制器后已经变成过去时间。

### 2.5 低层 RTDE 控制器可能忽略已经过期的 waypoint

UR5e 控制器以独立进程运行，并以 500Hz 持续发送 `servoL`：

- `deploy/umi/real_world/bimanual_umi_env.py:149-153`
- `deploy/umi/real_world/rtde_interpolation_controller.py:299-330`

```python
frequency=500
...
dt = 1.0 / self.frequency
...
pose_command = pose_interp(t_now)
assert rtde_c.servoL(
    pose_command,
    vel,
    acc,
    dt,
    self.lookahead_time,
    self.gain,
)
```

收到 `SCHEDULE_WAYPOINT` 后，控制器把 wall-clock target time 转为 monotonic time：

- `deploy/umi/real_world/rtde_interpolation_controller.py:391-407`

```python
target_time = float(command["target_time"])
target_time = time.monotonic() - time.time() + target_time
curr_time = t_now + dt
pose_interp = pose_interp.schedule_waypoint(
    pose=target_pose,
    time=target_time,
    max_pos_speed=self.max_pos_speed,
    max_rot_speed=self.max_rot_speed,
    curr_time=curr_time,
    last_waypoint_time=last_waypoint_time,
)
last_waypoint_time = target_time
```

而 `PoseTrajectoryInterpolator.schedule_waypoint()` 会直接忽略早于当前时间的 waypoint：

- `deploy/umi/common/pose_trajectory_interpolator.py:122-126`

```python
if curr_time is not None:
    if time <= curr_time:
        # if insert time is earlier than current time
        # no effect should be done to the interpolator
        return self
```

因此 telemetry 里的 `executed_action_count` 当前更准确地说是“发给 env 的动作数”，不一定等于底层 RTDE 控制器真正接受并插入轨迹的 waypoint 数。

### 2.6 主循环按配置步数推进，而不是按实际执行步数推进

每轮动作下发后，主循环推进逻辑如下：

- `deploy/inference_real.py:482-489`

```python
if args.max_scheduled_actions is not None:
    step_advance = args.max_scheduled_actions
elif args.steps_per_inference is not None:
    step_advance = args.steps_per_inference
else:
    step_advance = len(actions)
iter_idx += max(1, step_advance)
t_cycle_end = loop_start + iter_idx * dt
```

也就是说，即使因为延迟过滤或底层时间过期导致实际动作少于 `steps_per_inference`，主循环仍然按配置的 chunk 长度推进。这会进一步让真实 `T_chunk`、有效 `n`、动作序列和理论设定偏离。

## 3. 显式异步和默认同步的区别

显式异步只有 `--async-inference` 打开时才启用：

- `deploy/inference_real.py:349-352`

```python
async_worker = (
    _async_inference.AsyncInferenceWorker(policy_infer)
    if policy_infer is not None and args.async_inference
    else None
)
```

异步 worker 是一个单请求后台线程：

- `deploy/async_inference.py:44-91`

```python
self._executor = futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-async-infer")
...
self._pending = self._executor.submit(self._run, request, obs_copy)
...
response = self._infer_fn(policy_obs)
```

异步启动时机：

- `deploy/inference_real.py:491-518`

```python
overlap_steps = min(args.inference_overlap_steps, len(scheduled_actions))
target_start_timestamp = float(timestamps[-1] + dt)
launch_timestamp = target_start_timestamp - overlap_steps * dt
...
request = async_worker.submit(
    launch_policy_obs,
    obs_timestamp=float(launch_obs["timestamp"][-1]),
)
```

默认 `async_inference=false` 时没有后台请求，但仍然存在第 2 节描述的时间戳过滤、动作截断、latency compensation、控制器插值等机制。因此“默认同步”不是“论文干净同步”。

## 4. `--inference-latency-scale` 实际模拟了什么

延迟缩放逻辑：

- `deploy/latency_simulation.py:6-20`

```python
start = perf_counter()
result = infer_fn(policy_obs)
elapsed = perf_counter() - start
extra_delay = elapsed * (latency_scale - 1.0)
if extra_delay > 0:
    sleep(extra_delay)
return result
```

它模拟的是客户端看到的 policy call wall-clock 变慢。它不会改变模型输出本身，但会改变推理结束时的 `time.time()`，从而触发 `_future_action_schedule()` 丢弃更多前缀动作。

所以在当前实现里，`--inference-latency-scale` 并不是论文里“只改变 `T_inf` 而动作执行保持同一 chunk”的纯净干预。它同时改变了被执行动作的时间切片。

## 5. 已有 telemetry 的支持证据

已有 pick-place telemetry 显示：

- `model_action_count=30`
- `scheduled_action_count` 通常是 8
- `dropped_action_count=22`
- 首个 scheduled timestamp 往往相当于 chunk 内第 4 或第 5 个动作

例如：

- `../data/extracted_20260528_02_analysis/20260528_021236_p2_10step_30s/telemetry/inference.jsonl`
- `../data/extracted_20260528_02_analysis/20260528_020556_p2_1step_34s/telemetry/inference.jsonl`
- `../data/umi_step_p3_trials_analysis/raw/umi_10step_p3/runs/20260528_191727/telemetry/inference.jsonl`

汇总结果也显示，1/7/10 denoise steps 的 `T_chunk` 基本接近，但 `N` 明显不同：

- `../data/umi_step_p3_trials_analysis/plots/tisd_chunk_metrics_trimmed_1_7_10_summary.csv`

```text
denoise_steps,T_chunk_mean_s,N_mean,T_task_mean_s
1,0.7912,57.4,45.42
7,0.7873,49.2,38.74
10,0.7894,47.2,37.26
```

这和代码机制一致：4070 上去噪步数减少只小幅改变可见 chunk 周期，但动作质量下降导致 `N` 上升；同时，当前框架中的 `T_inf` 变化会先被 action timestamp/slack 机制吸收，而不是简单线性反映到 `T_chunk = T_inf + nT_act`。

## 6. 对实验假设的影响

如果把论文变量映射到当前框架，建议这样理解：

```text
T_act = 1 / args.frequency
model_n = action_horizon = 30
configured_n = steps_per_inference 或 max_scheduled_actions
scheduled_n = scheduled_action_count
controller_n = RTDE 控制器实际接受的 waypoint 数
T_inf = inference_latency_ms / 1000
T_chunk_visible = 相邻 telemetry 记录或相邻 chunk 起始时间间隔
```

论文理想同步需要 `configured_n == scheduled_n == controller_n`，并且每次都从同一个 action index 开始执行。当前框架不保证这一点。

因此，单纯放大 `T_inf` 会造成：

1. 首个执行 action index 后移。
2. `scheduled_action_count` 可能下降。
3. 低层 controller 可能忽略补偿后已过期的 waypoint。
4. 主循环仍按配置步数推进，导致 loop time 和真实动作执行数不完全一致。
5. 执行动作序列发生变化，可能表现为跳过早期平滑过渡、直接追 chunk 后段目标，从而过冲。

## 7. 如何实现论文里的干净同步

目标是让实验满足：

```text
每轮：
1. 取观测 obs_t
2. 阻塞等待 policy infer 完成，得到完整 actions[0:n]
3. 从 actions[0] 开始，按固定 dt 顺序执行固定 n 个动作
4. 执行完这 n 个动作后，再取下一次观测
5. 不丢动作、不按 wall-clock 跳过前缀、不后台 overlap
```

### 7.1 最小侵入方案：新增 strict sync 模式

建议在 `deploy/inference_real.py` 增加参数：

```python
strict_sync: bool = False
strict_sync_actions: int | None = None
```

启用后修改三个地方。

第一，禁用 async：

```python
if args.strict_sync and args.async_inference:
    raise ValueError("--strict-sync cannot be combined with --async-inference.")
```

第二，替换 `_future_action_schedule()` 的时间戳逻辑。严格同步不再用 `obs_timestamp` 给动作定时，也不根据 `time.time()` 丢弃动作，而是从当前推理结束后的一个安全起点开始排固定 `n` 个动作：

```python
def _strict_sync_action_schedule(
    actions: np.ndarray,
    *,
    frequency: float,
    action_count: int,
    action_exec_latency: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / frequency
    selected = actions[:action_count]
    start = time.time() + action_exec_latency + dt
    timestamps = start + np.arange(len(selected), dtype=np.float64) * dt
    return selected, timestamps
```

这保证每次都从 `actions[0]` 开始，而不是根据推理耗时跳到 `actions[k0]`。

第三，主循环推进按实际严格同步动作数：

```python
iter_idx += len(scheduled_actions)
t_cycle_end = time.monotonic() + len(scheduled_actions) * dt
precise_wait(t_cycle_end)
```

不要再用配置的 `steps_per_inference` 推进，否则当实际动作数变化时仍会引入偏差。

### 7.2 更干净的方案：阻塞式逐步执行

如果想最接近论文公式，可以完全避免 timestamp queue 的“未来排程”语义，改成阻塞执行：

```python
for action in actions[:n]:
    target_time = time.time() + dt
    env.exec_actions(
        actions=action[None, :],
        timestamps=np.array([target_time], dtype=np.float64),
        compensate_latency=False,
    )
    precise_wait(time.monotonic() + dt)
```

这个方案的优点是概念最干净：infer 完成后就是固定执行 `n` 个动作。缺点是低层 UR5e 仍是 500Hz 插值 servoL，单步下发会更容易受 Python 调度和 websocket/RTDE jitter 影响，运动可能不如提前排多个 waypoint 平滑。

### 7.3 推荐实验用方案：固定 chunk 起点，保留 waypoint 插值

实际真机更推荐 7.1，而不是 7.2：

- 仍然一次性给 RTDE 控制器排多个未来 waypoint，保持轨迹平滑。
- 但严格从 `actions[0:n]` 执行，不再丢弃前缀。
- 每次推理都阻塞，chunk 执行期间不发起下一次推理。
- `T_chunk` 可以定义为 `inference_latency + n/frequency`，与论文更一致。

为了避免第一个 waypoint 因 latency compensation 过期，strict sync 下建议：

```python
compensate_latency=False
start = time.time() + 2 * dt
```

或者如果必须保留 `compensate_latency=True`，则：

```python
start = time.time() + robot_action_latency + 2 * dt
```

并且把 `action_exec_latency` 设置成不小于 `robot_action_latency`，否则 `_future_action_schedule()` 和低层 compensation 的时间基准不一致。

## 8. strict sync 需要新增的 telemetry

为了验证 strict sync 是否真的成立，建议新增字段：

```text
strict_sync_mode
model_action_count
configured_action_count
scheduled_action_count
first_scheduled_action_idx
last_scheduled_action_idx
schedule_start_source
infer_done_wall_time
first_timestamp_lead_ms
post_robot_compensation_lead_ms
controller_accepted_waypoint_count
loop_lag_ms
```

判定条件：

```text
first_scheduled_action_idx == 0
scheduled_action_count == configured_action_count
controller_accepted_waypoint_count == scheduled_action_count
async_mode == false
```

满足这些条件后，才可以说实验中的 `rho = T_inf / T_act` 变化基本没有改变 `T_exec` 或有效 `n`。

## 9. 建议的对照实验

建议做三组对照：

1. 当前默认同步：
   - `async_inference=false`
   - 使用现有 `_future_action_schedule()`
   - 记录前缀丢弃和 schedule lead

2. strict sync：
   - `strict_sync=true`
   - 固定执行 `actions[0:n]`
   - 不丢弃、不 overlap

3. 显式 async：
   - `async_inference=true`
   - sweep `inference_overlap_steps`
   - 观察 `async_chunk_boundary_wait_ms` 和 `async_hidden_inference_ms`

如果目标是验证论文静态任务中的理想假设，应优先使用第 2 组；如果目标是解释当前真机实际现象，则第 1 组的 telemetry 更关键。

## 10. LIBERO 这类静态模拟器更像哪一种机制

以本仓库的 LIBERO 默认评测为例，它更接近论文中的干净同步假设，而不是当前 UR5e 真机部署机制。

LIBERO eval 的主循环是：

```text
action_plan 为空
-> 阻塞等待 policy inference
-> 取 action_chunk[:replan_steps]
-> 每个 simulation step 执行一个 action
-> action_plan 用完后再重新 inference
```

代码证据：

- `examples/libero/main.py:21-30` 定义默认 `replan_steps=5`。
- `examples/libero/main.py:127-148` 只有 `action_plan` 为空时才调用 `client.infer()`，并固定取 `action_chunk[:args.replan_steps]`。
- `examples/libero/main.py:150-153` 每次从 `action_plan` 弹出一个 action，然后调用一次 `env.step(action.tolist())`。
- `src/openpi/training/config.py:781-783` 中 `pi05_libero` 使用 `action_horizon=10`。

对应代码片段：

```python
if not action_plan:
    action_chunk = client.infer(element)["actions"]
    assert len(action_chunk) >= args.replan_steps
    action_plan.extend(action_chunk[: args.replan_steps])

action = action_plan.popleft()
obs, reward, done, info = env.step(action.tolist())
```

这和真机部署有三个关键差异：

1. 没有基于 `time.time()` 的动作前缀丢弃。
2. 没有 `robot_action_latency` / `gripper_action_latency` compensation。
3. 没有独立 500Hz 控制器在后台持续插值执行已排程 waypoint。

因此，如果在 LIBERO 中只增加 policy inference 的 wall-clock 耗时，但不让模拟器在等待推理时额外 `env.step()`，那么模拟器状态不会在 inference 期间前进，执行的动作 index 也不会因为耗时变长而后移。此时论文变量可以较干净地映射为：

```text
T_act = one simulator env.step duration in logical time
n = replan_steps
T_inf = wall-clock policy inference latency
T_chunk = T_inf + replan_steps * T_act
```

注意这里的 `n` 更应取 `replan_steps=5`，不是模型的 `action_horizon=10`。因为 eval 只执行 chunk 的前 5 个动作，剩余动作不会进入本轮模拟器执行。

所以，LIBERO 默认 eval 是“阻塞推理 + 固定前缀 chunk 执行”的 receding-horizon 同步。它不完全等价于论文中执行完整 action horizon 的形式，但比当前真机部署更接近论文假设。

## 11. 为什么真机不默认采用严格同步

真机和 LIBERO / 论文假设拉开距离，主要不是理论选择，而是机器人系统工程约束。论文假设适合做可控变量分析；真机部署必须优先保证连续控制、时序鲁棒性和硬件安全。

### 11.1 真机控制不能在推理期间“暂停世界”

LIBERO 中，`client.infer()` 阻塞时，除非代码显式调用 `env.step()`，模拟器状态不会继续演化。但真机不同：重力、物体滑动、夹爪闭合、机器人伺服控制、相机采集都在真实时间里继续发生。

如果采用最朴素的严格同步：

```text
观测 -> 等待 inference -> 开始执行 actions[0:n]
```

那么 `actions[0]` 对应的是推理前的观测状态，但实际执行时环境已经过去了 `T_inf`。在静态 pick-place 中这个问题可能不严重；在接触、抓取、移动物体或 conveyor 任务中，这会直接增加 observation-execution mismatch。

当前真机代码用 `_future_action_schedule()` 丢弃已经过期的动作，本质上是在承认：这些动作的目标时间已经错过，继续执行它们未必合理。

代码证据：

- `deploy/inference_real.py:160-178` 只保留 `timestamps > current_time + action_exec_latency` 的动作。

### 11.2 UR5e 需要连续 servoL，而不是一段段停顿式动作

UR5e 控制器进程以 500Hz 连续调用 `servoL`：

- `deploy/umi/real_world/rtde_interpolation_controller.py:299-330`

这意味着低层控制必须持续给机器人目标位姿。真机部署把多个 waypoint 一次性排进 `PoseTrajectoryInterpolator`，让控制器在 500Hz 循环里平滑插值：

- `deploy/umi/real_world/rtde_interpolation_controller.py:391-407`
- `deploy/umi/common/pose_trajectory_interpolator.py:105-185`

如果严格同步实现成“每个 action 执行完再发下一个 action”，高层 Python / websocket / OS 调度 jitter 会更直接地传到机器人控制上，表现为停顿、抖动、速度不连续，甚至接触不稳定。提前排程多个未来 waypoint 可以把高层 jitter 和低层 servo 解耦。

### 11.3 latency compensation 是为了让目标时间对齐硬件执行，而不是论文建模

真机中 robot 和 gripper 都有执行延迟：

- `deploy/configs/umi_ur5e_wsg50.yaml:15-17`
- `deploy/configs/umi_ur5e_wsg50.yaml:31-32`

```yaml
robot_action_latency: 0.1
gripper_action_latency: 0.1
```

下发时会减去这些 latency：

- `deploy/umi/real_world/bimanual_umi_env.py:441-446`

```python
robot.schedule_waypoint(pose=r_actions, target_time=new_timestamps[i] - r_latency)
gripper.schedule_waypoint(pos=g_actions, target_time=new_timestamps[i] - g_latency)
```

这对真实执行有意义：命令到达和硬件响应都有延迟，目标时间需要提前补偿。但这也让系统不再是论文里简单的 `T_inf + nT_act`。它引入了“目标时间是否仍在未来”的判定。如果补偿后已经过期，底层插值器会忽略该 waypoint：

- `deploy/umi/common/pose_trajectory_interpolator.py:122-126`

```python
if time <= curr_time:
    return self
```

### 11.4 时间戳排程可以容忍推理耗时波动

真实推理耗时不是常数。GPU warmup、websocket 传输、图像编码、Python 调度、CUDA kernel 抖动都会让 `policy.infer()` wall-clock 有波动。

如果严格同步要求每次都完整执行 `actions[0:n]`，那么推理偶发变慢时，机器人会在 chunk 边界等待更久，形成明显停顿。当前框架用时间戳过滤和排程，把“错过的动作”丢掉，把剩下的未来动作继续接到控制器时间轴上，牺牲了理论干净性，换取了实际控制连续性。

这也是为什么已有 telemetry 里有时会出现 `scheduled_action_count=1` 的 fallback：系统宁愿少排一些仍然未来的动作，也不强行执行已经错过时间窗口的动作。

### 11.5 真机默认目标是任务执行稳定，不是变量隔离实验

论文里的严格同步适合回答：

```text
只改变 T_inf，理论上的 T_chunk/T_task 如何变化？
```

真机默认部署更关心：

```text
在真实硬件延迟、控制频率、网络抖动和环境继续演化的情况下，机器人能否平滑、安全地完成任务？
```

这两个目标不同。为了任务执行稳定，真机系统引入了：

- future timestamp scheduling
- stale action dropping
- latency compensation
- interpolation controller
- optional async overlap
- optional future-state substitution

这些机制都会破坏论文中“只改变 `T_inf`，其他执行条件不变”的干净假设，但它们是实际部署常见且合理的工程选择。

## 12. 什么时候应该使用严格同步或严格异步

严格同步和严格异步不是没有价值，而是用途不同。它们都应该作为实验模式，而不是覆盖
真机默认的 practical timestamp scheduler。

适合使用严格同步的场景：

- 复现实验论文的变量隔离假设。
- 测量 `T_inf`、`T_act`、`n`、`N` 的干净关系。
- 比较不同 denoising steps / quantization / pruning 的理论速度收益。
- 证明真机默认机制中的 action-prefix dropping 是否是过冲来源。

适合使用严格异步的场景：

- 复现论文中固定 \(n'\) overlap 的异步推理模型。
- 测量
  `T_cycle = T_inf + n*T_act - min(T_inf, n_prime*T_act)`
  这种理论式的 chunk 周期。
- 比较不同 overlap steps 在真机上的边界等待和隐藏推理时间。
- 在保持 action index 固定为 `0..n-1` 的前提下，研究推理和执行重叠能否改善
  `T_task`。

不适合直接作为默认真机部署的场景：

- conveyor、动态抓取、接触敏感任务。
- 推理耗时明显大于 `T_act` 或波动很大。
- 需要高频连续 servo 平滑性的任务。
- 网络远程推理或 policy server 和 robot 分离部署。

因此，推荐做法不是把真机默认机制全部改成严格同步，而是在 `deploy/inference_real.py` 中提供一个显式实验开关，例如：

```text
--strict-sync
--strict-async --strict-async-overlap-steps <n_prime>
```

当前实现中，`--strict-sync` 会在推理返回后把 chunk 从 action index 0 重新锚定到未来；
`--strict-async` 会在当前 chunk 执行到第 `n - n_prime` 个动作时启动下一轮推理，并把
下一 chunk 贴到计划边界，若推理超时则顺延到 `t_return + strict_sync_start_delay`。
两种模式都不会因为 wall-clock 过期而跳过 action chunk 前缀。

默认仍保留当前时间戳排程机制；需要验证论文假设时，显式打开 strict sync 或
strict async，并在 telemetry 中记录：

```text
first_scheduled_action_idx == 0
scheduled_action_count == configured_action_count
controller_accepted_waypoint_count == scheduled_action_count
schedule_mode in {"strict_sync", "strict_async"}
prefix_dropped_action_count == 0
strict_async_overlap_steps == configured_n_prime  # only for strict async
```

这样可以同时保留工程部署能力和论文变量隔离能力。
