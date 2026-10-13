# 官方 π0.5-LIBERO checkpoint：Spatial 同流程校准

2026-10-13：官方 `gs://openpi-assets/checkpoints/pi05_libero/` checkpoint（官方 README 标注微调 30k），本地空闲 GPU 3。

使用与双状态模型相同的 `examples/libero/evaluate_checkpoint.py`；配置使用官方 `pi05_libero`，加载官方自带 norm stats，不加入连续 state projection 或 age embedding。

Spatial 10 tasks × 5 trials，官方初始状态 0–4，seed 7 / 相同逐 episode 种子，flow steps 10，action horizon 10，replan steps 5，settle steps 10，最多 220 控制步；同一 LIBERO commit `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`、图像处理、状态表示、动作接口和失败判定。

| 模型 | Spatial 成功数 | 成功率 |
|---|---:|---:|
| 官方 π0.5-LIBERO @ 30k | 50/50 | 100.0% |
| 双状态，直接延迟训练 @ 6k | 40/50 | 80.0% |
| 双状态，同步训练 @ 6k | 37/50 | 74.0% |

两组历史结果与官方本轮的任务、初始状态、随机种子和评估参数逐项核对通过；本轮完整 50 episodes、退出码 0，成功计数与汇总一致。

这是同步 rollout、每任务仅 5 次的校准；官方 README 的 Spatial 98.8% 是另一轮报告值。即使官方模型在本流程中取得高成功率，也不能据此排除双状态模型路径的问题，或把差距全部归因于训练步数；两者的结构、训练步数及训练配置均有区别。

权重获取与计算：官方 GCS 固定 object generation，共 16 文件 / 12,439,085,481 bytes；官方 CRC32C 全部通过，提供 MD5 的文件也通过 MD5。用仓库官方 converter 在 CPU 转换为 float32 Safetensors，并复制官方 assets；推理 backbone 按官方配置使用 bfloat16，projection 等保持原 PyTorch 流程。为与既有双状态配置一致，评估统一关闭 torch.compile（不改变模型结构或训练权重）。

转换后模型：14,467,165,872 bytes / 812 tensors。SHA256：`a10e3b07de63cd85b745b207bb14fa9304c032d23af10b0deae4314d02c12839`。

W&B：[官方 checkpoint 的 Spatial 评估](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/gb4qm9zx)。按用户要求，W&B 仅上传指标和 JSON，视频保留本地。

本地明细、协议、provenance、Spatial 对照与视频：`eval/results/20261013/official_pi05_libero_30k/spatial/`。官方源文件清单与 generation/校验值：`eval/official_pi05_libero/gcs_listing.json`。
