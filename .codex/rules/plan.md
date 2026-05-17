# UMI Conveyor 微调 pi0/pi05 执行计划

## Source

- 2026-05-17 00:52:11 CST: 从已关闭 session `019e30c9-f5c3-7772-99fb-541bf97b70d6` 的 rollout 文件恢复最后一版 `<proposed_plan>`。
- 原 plan 中部分路径是 `/data/archive_liyixuan23/openpi_lerobot`；按用户后续规则和当前实际输出，统一修正为 `/data/archive_liyixuan23/finetune_pi/...`。

## Summary

- 使用 `/data/archive_liyixuan23/umi_for_train/example_pick_night/mixdataset.zarr.zip`，已确认 196 episodes、147180 帧。
- 训练动作空间为 7 维 EEF 位姿+夹爪：`eef_pos(3) + eef_rot_axis_angle(3) + gripper_width(1)`。
- 统一 prompt：`pick up the red block on the conveyor belt. put it in the blue box.`
- pi0 和 pi05 都先跑 LoRA 微调；pi05 使用 `Pi0Config(pi05=True)`。

## Implementation

- 新增 UMI zarr 到 LeRobot 转换脚本：
  - 先解压 zip 到本地 zarr 目录，避免直接从超大 zip 读导致索引卡住。
  - 读取 `camera0_rgb`、`robot0_eef_pos`、`robot0_eef_rot_axis_angle`、`robot0_gripper_width`、`meta/episode_ends`。
  - 写出 LeRobot dataset 到 `/data/archive_liyixuan23/finetune_pi/openpi_lerobot/liyixuan23/umi_conveyor`，repo_id 为 `liyixuan23/umi_conveyor`。
  - `state` 和 `actions` 都存 7 维绝对 EEF target，`task` 存固定 prompt。
- 新增 UR5e/UMI transforms：
  - 单路 `camera0_rgb` 映射到 `base_0_rgb`。
  - 两个 wrist slots 用零图填充并 mask 为 false。
  - 对前 6 维 EEF pose action 做 delta transform，夹爪维度保持绝对值。
- 新增两个 TrainConfig：
  - `pi0_umi_ur5e_lora`
  - `pi05_umi_ur5e_lora`
  - pi0 使用 `Pi0Config(..._lora variants, action_dim=7, action_horizon=30)`。
  - pi05 使用 `Pi0Config(pi05=True, ..._lora variants, action_dim=7, action_horizon=30)`。
  - 权重分别从 `gs://openpi-assets/checkpoints/pi0_base/params` 和 `gs://openpi-assets/checkpoints/pi05_base/params` 加载。
  - 数据 norm stats 使用本数据集重新计算，不复用官方 UR5e stats。
- 环境变量：
  - `UV_CACHE_DIR=/tmp/uv-cache`
  - `HF_LEROBOT_HOME=/data/archive_liyixuan23/finetune_pi/openpi_lerobot`
  - `OPENPI_DATA_HOME=/data/archive_liyixuan23/finetune_pi/openpi_cache`
  - 训练时使用 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`。

## Workflow

- 转换前先做 2-episode smoke conversion，验证字段和 LeRobot dataset 可读。
- 全量转换 196 episodes。
- 分别计算 norm stats：
  - `uv run scripts/compute_norm_stats.py pi0_umi_ur5e_lora`
  - `uv run scripts/compute_norm_stats.py pi05_umi_ur5e_lora`
- 分别跑 1-step smoke training：
  - `uv run scripts/train.py pi0_umi_ur5e_lora --exp-name=smoke --overwrite --num-train-steps=1`
  - `uv run scripts/train.py pi05_umi_ur5e_lora --exp-name=smoke --overwrite --num-train-steps=1`
- smoke 通过后给出正式训练命令，checkpoint 写入 `checkpoints/pi0_umi_ur5e_lora/...` 和 `checkpoints/pi05_umi_ur5e_lora/...`。

## Test Plan

- 验证转换后 episode 数为 196，总帧数为 147180。
- 验证 LeRobot 单帧包含 `image`、`state`、`actions`、`task`，其中图像为 `(224, 224, 3)` uint8 或 LeRobot 读取后的 CHW float tensor，state/action 为 7 维。
- 验证 pi0/pi05 dataloader 各能成功取 sample/batch，并完成 prompt tokenization、image resize、state/action padding。
- 验证 pi0/pi05 各完成 1 个训练 step，并成功写 checkpoint。

## Assumptions

- UMI 数据里的 EEF pose 坐标系就是后续 UR5e 执行端期望的坐标系，本次不做坐标变换。
- `gripper_width` 直接作为夹爪动作维度。
- 如果 gcloud 权重本地缓存不存在，由 openpi 原生 `gs://` 下载逻辑拉取，不使用 hf-mirror。

## Current Execution State

- 已完成全量转换完整性验证：196 parquet files，147180 rows，`episode_000195.parquet` 完整可读。
- 已验证 `pi0_umi_ur5e_lora` 和 `pi05_umi_ur5e_lora` 可创建 LeRobot dataset 并 transform 单样本。
- 已生成并验证 UMI norm stats：
  - `assets/pi0_umi_ur5e_lora/liyixuan23/umi_conveyor/norm_stats.json`
  - `assets/pi05_umi_ur5e_lora/liyixuan23/umi_conveyor/norm_stats.json`
- 已新增 `ShapeCompatibleCheckpointWeightLoader`，用于 UMI 7D action/head 与官方 32D base checkpoint 的形状兼容加载；仅 UMI pi0/pi05 config 使用该 loader。
- 已完成 `pi0_umi_ur5e_lora` 1-step smoke training，checkpoint 为 `checkpoints/pi0_umi_ur5e_lora/smoke/0`，日志 `loss=2.0420`、`grad_norm=24.4392`。
- 已完成 `pi05_umi_ur5e_lora` 1-step smoke training，checkpoint 为 `checkpoints/pi05_umi_ur5e_lora/smoke/0`，日志 `loss=1.1999`、`grad_norm=2.2743`。
- 大 checkpoint 缓存已放在 `/data/archive_liyixuan23/finetune_pi/openpi_cache/openpi-assets/checkpoints/`，其中 `pi0_base` 和 `pi05_base` 各约 12G。
- 当前缺口：尚未启动正式长训练；smoke 阶段已通过，可以按 GPU 空闲情况用 tmux 启动正式训练。
