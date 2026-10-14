# 双状态 π0.5：18000-step Spatial 异步 rollout

2026-10-14：使用同一对 18k checkpoint，每模型 Spatial 10 tasks × 5 trials = 50 episodes。

| 训练路线 | 同步 rollout | 双 worker 异步 rollout | 无动作 tick |
|---|---:|---:|---:|
| 直接延迟训练 | 45/50 (90.0%) | 25/50 (50.0%) | 43.1% |
| 同步训练 | 42/50 (84.0%) | 39/50 (78.0%) | 29.8% |

两模型依次使用同一对空闲 RTX 6000 Ada：System2 / VLM 在物理 GPU 2，System1 / action expert 在 GPU 3。两个 worker 独立运行；System2 更新视觉、任务与离散 proprio 的 prefix KV，System1 用最新连续 proprio 与最近可用 KV 生成动作。远程 A800 训练未中断。

壁钟控制为 10 ticks/s，每 tick 推进一次原始 LIBERO env.step；保留 control_freq=20，因此仿真时间约以壁钟的一半速度推进。这不是 10 Hz 仿真环境。每 episode 开头等待首次有效 prefix 与动作，然后启动时钟；之后不等待两个 worker。

复用同步 eval 的图像旋转/resize、state 拼接、官方初始状态 0–4、逐 episode seed 7+100×task_id+trial、settle 10、flow steps 10、动作 horizon 10、最多 220 控制步，均为 eager PyTorch（未启用 torch.compile）。仅使用 chunk 的前 5 个位置，按当前步减采样 state 步索引动作；已过去的位置跳过。无动作时零 EEF 增量并保留夹爪指令。相同随机种子不保证异步调度下的采样次数/噪声序列相同。

这里 replan_steps=5 指采样 state 起点的前 5 个位置，不是预测返回后再执行 5 个新动作。例如返回时 offset=3，则本协议只允许位置 3、4；位置 5–9 被执行范围截断，后续预测尚未返回时会产生 hold。因此结果包含这一执行范围限制的影响。

年龄上限 5 约束采样时当前 proprio 与 prefix 的步数差，未来动作执行年龄等于采样年龄加 offset，允许超过 5。修正了 runner 将采样年龄上限重复当作执行 TTL 的问题，并通过 5 项异步 runner 测试；修正前的中断诊断已单独保留，排除出本比较。没有额外注入延迟，也没有 streaming/action-buffer conditioning 或 inpainting。

| 时序指标 | 直接延迟训练 | 同步训练 |
|---|---:|---:|
| 采样 prefix 年龄均值 / tick | 1.09 | 1.02 |
| 执行 prefix 年龄均值 / tick | 4.67 | 4.55 |
| 执行 prefix 年龄 P95 / tick | 6.0 | 5.0 |
| 执行 prefix 年龄最大 / tick | 7 | 7 |
| expert 采样耗时 / ms（按已执行动作加权） | 242.2 | 229.2 |
| state 到动作执行 / ms（按已执行动作加权） | 371.6 | 364.6 |
| 控制 tick 迟到 P95 / ms | 0.10 | 0.09 |
| expert 采样耗时均值 / ms（按已执行预测去重） | 251.4 | 235.5 |
| expert 采样耗时 P95 / ms（按已执行预测去重） | 313.2 | 286.6 |

采样耗时包含 KV 转移（发生更新时）、flow 推理及动作后处理；只统计产生了实际执行动作的预测。按预测去重的耗时、年龄与 offset 分布另存 comparison.json。无动作比例是 tick 比例，不能解释为失败 episode 比例。

这是当前硬件、10 Hz 壁钟调度和前 5 位置限制下的异步结果。成功率同时受训练、实际推理延迟、动作缺口与切换影响；每任务仅 5 次，不能单独据此认定训练路线优劣，也不能与固定 cache-age 或 streaming 实验混为一谈。

训练随机错开 prefix 与当前 state 的时间索引，监督仍为当前 state 起点的未来动作；没有在训练中实时运行两个 worker 或显式模拟动作断档/执行缓冲。未来动作索引本身属于训练 horizon，但异步过程中实际执行的旧计划或 hold 可能偏离这些动作所对应的轨迹。

| task id | 直接延迟训练 | 同步训练 |
|---|---:|---:|
| 0 | 3/5 | 5/5 |
| 1 | 2/5 | 3/5 |
| 2 | 5/5 | 5/5 |
| 3 | 3/5 | 5/5 |
| 4 | 0/5 | 0/5 |
| 5 | 0/5 | 5/5 |
| 6 | 4/5 | 5/5 |
| 7 | 4/5 | 5/5 |
| 8 | 2/5 | 2/5 |
| 9 | 2/5 | 4/5 |

两组均完整 50 episodes，退出码 0；逐 episode 任务/初始状态/种子与同步 18k 完全匹配，权重相同，成功计数及逐 tick 年龄/offset 检查通过。

W&B：[直接延迟](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/pujwqnr1)；[同步训练](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/4wrhdq1y)。仅上传指标与 JSON，不上传视频。

结果/视频/协议：`eval/results/20261014_spatial_18000_async/<run>/18000/`；完整对照与校验：`eval/results/20261014_spatial_18000_async/comparison.json`；修正前诊断：`eval/results/20261014_spatial_18000_async_diagnostic_pre_ttl_fix/`。

同步对照见 [18k Spatial 同步结果](pi05_libero_spatial_ckpt18000_20261014.md)，模型 SHA256 与同步评估一致。
