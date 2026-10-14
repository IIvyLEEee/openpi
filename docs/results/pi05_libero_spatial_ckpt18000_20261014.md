# 双状态 π0.5：18000-step Spatial 同步 rollout

2026-10-14：两路线分别固定第 18,000 步的完整 checkpoint，远程训练继续；本地空闲 GPU 评估。

复用 `examples/libero/evaluate_checkpoint.py` 与 `pi05_libero_async`，同步 fresh observation / prefix age 0。Spatial 10 tasks × 5 trials，官方初始状态 0–4，seed 7，10 flow steps，每次生成 10 个动作、执行前 5 个，settle 10，最多 220 控制步，统一关闭 torch.compile。

| 训练路线 | 6k Spatial | 18k Spatial |
|---|---:|---:|
| 直接延迟训练 | 40/50 (80.0%) | 45/50 (90.0%) |
| 同步训练 | 37/50 (74.0%) | 42/50 (84.0%) |

历史官方 π0.5-LIBERO @ 30k 的同一套 Spatial eval 为 50/50（100%）。本轮两组与历史 6k/官方的任务、初始状态和逐 episode 随机种子逐项匹配，主要评估参数一致；norm stats 与各自 6k 完全相同。两组均完整 50 episodes / 退出码 0，成功计数与汇总一致。

两模型下载后与服务器端 SHA256 一致，metadata 均为 global_step 18000、batch 128。直接延迟路线训练时 prefix delay 最大 5，同步路线为 0；同步路线尚未进入原定同步 30k 后的延迟阶段。

这是同步基础评估，每任务只有 5 次，不能替代双 worker 异步 rollout 或用于证明训练路线的最终优劣。

W&B：[直接延迟](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/1ma0gnqk)；[同步训练](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/3gvyaa4c)。仅上传指标和 JSON，视频留本地。

结果、视频与协议：`eval/results/20261014_spatial_18000/<run>/18000/`；对照与权重 SHA256：`eval/results/20261014_spatial_18000/comparison.json`。

模型 SHA256：

- `dual_state_direct_delay_40k_b128`：`e903f7a44cb918a13d9aaea149354e608691d8d32979762866724819b928c052`。
- `dual_state_sync_30k_b128`：`a0c89f3523121442b17bba99686f0652a1eb46c38ba3df3c1636b3cd0cbbfb62`。
