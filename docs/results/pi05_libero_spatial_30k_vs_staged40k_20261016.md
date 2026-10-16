# π0.5：30k 与同步 30k＋延迟 10k 的 Spatial 对照

2026-10-16。本轮每模型分别完成同步、真实双 worker 异步各 50 episodes（10 tasks × 5 trials）。

| 训练路线 | 累计训练步数 | 同步 rollout | 异步 rollout | 异步无动作 tick |
| --- | ---: | ---: | ---: | ---: |
| 同步30k+延迟10k | 40000 | 42/50（84%） | 43/50（86%） | 0.00% |
| 直接延迟30k | 30000 | 44/50（88%） | 43/50（86%） | 0.00% |

本轮两条路线的异步成功次数相同；同步相差 2 个 episode。纯同步 30k 基线尚缺，当前结果不能判断追加延迟 10k 相对于其起点是否改善。

**缺项：纯同步训练 30k 尚未评估。** 本机没有该 checkpoint，AutoDL 原 SSH 端口拒绝连接。历史纯同步结果是 18k，未将其冒充 30k；恢复连接后需要补齐同协议两种 rollout。

## 协议与历史记录

此前 6k 跑过四个 suite，最近一次 18k 仅跑 Spatial；此前不存在 30k 的成功率记录。本轮沿用最近范围，只跑 Spatial，并重新评估本机可用的 30k 基线。

所有模型使用相同官方初始状态 0–4、逐 episode seed 7+100×task_id+trial、settle 10、最多 220 控制步、10 步 flow sampling（NFE=10）、eager PyTorch。图像旋转/resize、proprio 拼接及动作变换复用已有 eval。任务、初始状态与 seed 逐项匹配历史 18k 记录。

同步 rollout 每次使用新观测，prefix age=0，预测 10 个动作并执行前 5 个。异步采用独立 VLM/expert worker，10 Hz 壁钟发布观测，expert 读取最新连续 proprio、最近完成的 prefix KV 及实际年龄；采样年龄限制 0–5，无额外注入延迟。模型推理后按已过去的控制步跳过动作位置，允许使用完整 10 位置窗口。窗口用尽且没有有效新动作时，输出零 EEF 增量并保留夹爪指令。没有 streaming、action-buffer conditioning 或 inpainting。

**异步窗口与 18k 历史评估不同。** 历史异步仅允许前 5 位置，本轮允许全部 10 位置；所有本轮模型统一使用新协议。历史 direct/sync 18k 异步为 50%/78%，不能与本轮直接作为相同协议的训练进展比较。历史同步为 90%/84%，同步协议未变。

LIBERO 环境仍为 control_freq=20，每个 10 Hz 壁钟 tick 推进一个原始环境步，约 0.5 倍仿真实时时速。壁钟调度频率与仿真控制频率需要分开解释。相同 episode seed 不能保证异步调度下调用次数及噪声序列相同。

## 实际异步时序

| 训练路线 | 采样 prefix 年龄均值 | 执行 prefix 年龄均值 / 最大值 | 已执行预测的采样耗时均值 / P95 | 用到 offset≥5 的 tick |
| --- | ---: | ---: | ---: | ---: |
| 同步30k+延迟10k | 1.08 | 5.28 / 11 | 247.2 / 334.5 ms | 2139 |
| 直接延迟30k | 1.02 | 5.06 / 10 | 239.8 / 304.9 ms | 1892 |

预测耗时按每 episode 的 `(state_step, cache_id_used)` 去重，仅统计至少执行过一个动作的预测，不包括被完全丢弃的预测。年龄均值按实际执行动作加权；原始 `encode_ms` 表示当时最新发布 prefix 的耗时，不保证对应当前动作使用的 prefix。采样年龄与执行年龄分开检查，后者等于前者加动作 offset，可超过 5（最大合法值 14）。

## 逐任务成功数

| Spatial task ID | 同步30k+延迟10k 同步 / 异步 | 直接延迟30k 同步 / 异步 |
| --- | ---: | ---: |
| 0 | 5/5 / 5/5 | 5/5 / 5/5 |
| 1 | 5/5 / 5/5 | 5/5 / 4/5 |
| 2 | 5/5 / 5/5 | 5/5 / 5/5 |
| 3 | 5/5 / 5/5 | 4/5 / 5/5 |
| 4 | 2/5 / 2/5 | 3/5 / 3/5 |
| 5 | 4/5 / 4/5 | 4/5 / 4/5 |
| 6 | 5/5 / 5/5 | 5/5 / 5/5 |
| 7 | 5/5 / 5/5 | 5/5 / 4/5 |
| 8 | 2/5 / 2/5 | 4/5 / 3/5 |
| 9 | 4/5 / 5/5 | 4/5 / 5/5 |

## Checkpoint 与运行环境

本机四张 RTX 6000 Ada 中，启动前确认 GPU 0、2、3 空闲：GPU 0 跑同步队列，GPU 2 跑 VLM、GPU 3 跑 expert；GPU 1 的其他任务未使用。两种协议的队列并行运行，具体重叠时间见各 `hardware.json`，耗时不能视为完全独占整机的硬件基准。环境使用既有 `.venv-libero-eval`，未安装新包；LIBERO 固定源码 f78abd68ee283de9f9be3c8f7e2a9ad60246e95c、robosuite 1.4.1、MuJoCo 2.3.7、Torch 2.7.1，EGL 渲染。

各模型 global batch=128，均为 VLM 与 action expert 全量微调；模型 815 tensors / 7,473,251,472 bytes。直接延迟 30k 与同步 30k 的数据延迟分别为 0–5 / 0；追加阶段从同步 30k 严格加载模型，以新 optimizer、warmup 500、decay 10k 训练 10k，累计 40k。直接延迟路线原 decay=40k，在 30k 停止；同步路线 decay=30k，追加阶段重新 warmup。因此训练量和 LR schedule 不同，不能仅据成功率差值认定延迟训练的独立因果效果。每任务 5 次的样本也较少。

模型全文件 SHA256 在传输时已与服务器比对，完成后通过原子重命名发布；本次再次核对 verified manifest、文件大小、metadata step/source/batch/delay 与 norm stats SHA，逐 episode 检查覆盖、身份、退出码、成功计数，逐 tick 检查年龄、offset 和有效时延。所有模型使用相同全量 norm stats：`ccf0c0d5ca98df82a7b2933afb5441141a5d2f06b494d0f9106f5aad8492d05e`。

- 同步30k+延迟10k：stage step 10000 / cumulative 40000，SHA256 `5c12f8a3e2116880a7d85c74549aadbcd4c7f3ab2cc24f49bfc2360746091f36`；本地 `/home/liyixuan23/mlsys/openpi/eval/checkpoints/20261016_sync30k_delay10k/dual_state_sync30k_delay10k_b128_4gpu/10000`。
- 直接延迟30k：stage step 30000 / cumulative 30000，SHA256 `891686019757955285cba85b254fdac5a3fa8f9381647aea4ade57065779ad9c`；本地 `/home/liyixuan23/mlsys/openpi/eval/checkpoints/20261016_direct_30000/dual_state_direct_delay_40k_b128/30000`。

## 结果与 W&B

本地完整指标、逐 episode/tick 记录、协议、硬件信息及校验比较：`eval/results/20261016_spatial_30k_vs_staged40k/`。`comparison.json` 包含成功率、逐任务结果和时序统计；视频仅保存在本地，W&B 仅上传指标及 JSON。

- 同步30k+延迟10k：[同步](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/riqe8kue)、[异步](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/9l8a0sfi)。
- 直接延迟30k：[同步](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/50fpeycc)、[异步](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/uya1rph1)。

完整窗口变更通过 6 项 async runner 测试、Ruff 与格式检查；没有改变模型权重或采样算法。W&B 完成状态、episode/成功计数、退出码及无视频上传情况另存 `wandb_verified.json`。

同步/异步 CLI 和完整窗口说明见 [运行说明](../pi05_async.md#异步-libero-运行)；历史 [18k 同步](pi05_libero_spatial_ckpt18000_20261014.md)、[18k 异步（前 5 位置）](pi05_libero_spatial_ckpt18000_async_20261014.md)。
