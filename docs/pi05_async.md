# π0.5 双 proprio 与异步推理

本实现保留 π0.5 的多步 flow matching。VLM 读取图像、任务和离散 state 快照，action expert 额外直接读取当前连续 proprio；两者通过完整逐层 prefix KV 连接。没有引入 streaming、逐位置 noise level、动作 inpainting 或快视觉 encoder。

## 配置与接口

实验配置为 `pi05_libero_async`，新增参数如下：

| 参数 | 作用 |
|---|---|
| `model.continuous_state_input` | 在 expert suffix 中加入连续 state token，默认关闭 |
| `model.prefix_delay_max` | age embedding 覆盖的最大条件年龄，单位为训练/控制 tick |
| `data.prefix_delay_max` | 训练时随机采样过去图像/state 的最大延迟；设为 0 即同步训练 |
| `pytorch_allow_new_parameters` | 仅允许 base checkpoint 缺少新增 state projection 和 delay embedding |
| `pytorch_vlm_lr_scale` | VLM/视觉参数相对 expert 的学习率倍率，实验默认 0.1 |

LIBERO 使用原有 8 维 state，经现有 norm stats 归一化，再 padding 到 model action_dim。tokenizer 读取 padding 前的慢 state，expert 读取 padding 后的快 state。实验配置启用 strict token length 检查，超长任务直接报错。

PyTorch 模型提供：

```python
# observation 是已经过输入 transforms 的 Observation。
condition = model.encode_prefix(observation)  # System 2
actions = model.sample_actions_from_prefix(
    current_normalized_state,
    condition,
    prefix_age=age_in_ticks,  # [B] 的整数 Tensor
    num_steps=10,
)  # System 1，不做图像处理或 tokenization
system2, system1 = model.split_for_inference("cuda:0", "cuda:1")
```

`PrefixCondition` 包含 K/V tensors 和有效 mask，可通过 `.to(device)` 转移。K 已做 RoPE；采样不会修改 prefix，整次去噪固定使用同一份条件和 state。旧 `sample_actions()` 通过这两个接口组合运行。

新选项目前只支持 PyTorch；JAX 构建会明确报错。原 π0/π0.5 配置默认不启用这些选项。

## 准备与微调

使用项目原有 PyTorch 环境，并按 README 安装 `transformers_replace`。准备 LeRobot LIBERO 数据和已转换的 π0.5 PyTorch base checkpoint；本实现不会自动下载模型权重。训练需要实际的 GPU 环境。

当前机器已创建 Conda 环境 `openpi` 并应用 patch。`conda activate openpi` 后，可将下面训练命令的 `uv run` 替换为 `python`，测试则使用 `python -m pytest`，直接使用该环境。

在 `openpi/` 目录下计算归一化统计：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero_async
```

第一阶段：同步双 proprio 微调。模型仍保留 age embedding，但年龄为 0。

```bash
uv run scripts/train_pytorch.py pi05_libero_async \
  --exp-name dual_state_sync \
  --pytorch-weight-path /path/to/pi05_base_pytorch \
  --data.prefix-delay-max 0
```

第二阶段：从第一阶段的完整 checkpoint 继续新的延迟微调实验。这里是新实验初始化，不使用 `--resume`，不沿用旧实验的 optimizer/step。

```bash
uv run scripts/train_pytorch.py pi05_libero_async \
  --exp-name dual_state_async \
  --pytorch-weight-path checkpoints/pi05_libero_async/dual_state_sync/STEP \
  --no-pytorch-allow-new-parameters \
  --data.prefix-delay-max 5
```

延迟采样发生在同一 episode 内：prefix 使用 `images[t-k]` 和 `state[t-k]`，expert 使用 `state[t]`，监督仍为 `actions[t:t+H]`。episode 开头缩短延迟并记录实际年龄；任务切换不沿用旧任务条件。两路分别做 transforms，使慢 state 进入文本、快 state 进入连续投影，使用同一套 norm stats。

训练继续一次 prefix + suffix 联合 forward；flow matching loss 可以更新两系统，VLM 使用较小学习率。这里没有采用 KI 的 stop-gradient 或 FAST 辅助目标。旧 checkpoint 只允许明确新增参数缺失，其他结构不匹配仍报错；推理与实验 resume 要求完整 checkpoint。

## 异步 LIBERO 运行

使用项目原有 LIBERO 环境；也可以把已检出的
`../pi-r2-flow/learning/Isaac-GR00T/external_dependencies/LIBERO`
加入 `PYTHONPATH` 并安装其依赖。需要仿真环境和视频编码依赖。

```bash
uv run examples/libero/main_async.py \
  --args.checkpoint-dir checkpoints/pi05_libero_async/dual_state_async/STEP \
  --args.vlm-device cuda:0 \
  --args.expert-device cuda:1 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 10 \
  --args.rate-hz 10 \
  --args.replan-steps 5
```

参数前缀 `args.` 与 `main.py` 的 `Args` CLI 一致。控制频率须匹配数据 FPS；openpi 的转换脚本写入 10 FPS，因此示例默认 10 Hz。

两个 worker 分别持有 VLM 或 expert 模块，可放在不同 GPU；使用同一 GPU 时会争用算力，不能据此假定有加速。首次观测会等待有效 cache 和动作，之后 expert 不等待下一次 VLM 更新。KV 每次更新只向 expert device 转移一次。

主循环按仿真步数发布观测并对齐新 chunk：从观测步到执行步已经过去几步，就跳过新 chunk 的相应位置。无可用动作时输出零 EEF 增量并保持夹爪指令，不重复过期位移。缓存超出训练年龄范围会显式报错并结束该 episode。reset/task 切换通过 generation 丢弃旧的在途结果。

输出目录包含视频和 `metrics.jsonl`，记录成功率、worker 延迟、所用 cache_id、条件年龄、动作 offset 和没有动作就绪的 tick。条件年龄按环境步计算；wall-clock timing 另行记录，不混作训练延迟标签。

## 验证

```bash
uv run pytest \
  src/openpi/models_pytorch/pi0_async_test.py \
  src/openpi/training/delayed_dataset_test.py \
  src/openpi/training/delayed_loader_test.py \
  src/openpi/policies/async_runner_test.py
```

这些测试使用缩小后的真实 SigLIP/Gemma，验证联合/缓存 forward 等价、cache 不变、两系统与新增模块的梯度、checkpoint allowlist、历史观测边界，以及 VLM 阻塞时 expert 继续、动作时间对齐和 reset 隔离。

任务性能需要用同任务、种子、控制周期和 execution horizon 比较原 π0.5、双 proprio 同步、双 proprio 异步及延迟微调后异步。CPU 测试通过不代表已经获得 LIBERO 成功率或 GPU 加速结果。

## 已完成的 GPU 启动验证（2026-10-12）

复用了旧项目的 π0.5 base JAX 权重，并转换到 `checkpoints/base/pi05_base_pytorch`。使用真实 LIBERO episode 0（214 帧），统计放在独立的 `assets_smoke` 中。`--data.episodes 0` 只读取这个 episode；当前锁定的 LeRobot 版本只支持 `[0, ..., N-1]` 这样的连续 episode 子集。

在 `openpi/` 目录下：

```bash
conda activate openpi
export JAX_PLATFORMS=cpu
export HF_HOME="$PWD/.cache/huggingface"
export HF_LEROBOT_HOME="$PWD/data/lerobot"
export OPENPI_DATA_HOME="$PWD/.cache/openpi"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2

CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py pi05_libero_async \
  --exp-name YOUR_NEW_SYNC_RUN \
  --pytorch-weight-path checkpoints/base/pi05_base_pytorch \
  --assets-base-dir ./assets_smoke \
  --data.episodes 0 --data.prefix-delay-max 0 \
  --batch-size 2 --num-workers 0 --num-train-steps 2 \
  --log-interval 1 --save-interval 1000 --lr-schedule.warmup-steps 1 \
  --no-wandb-enabled

CUDA_VISIBLE_DEVICES=2 python scripts/train_pytorch.py pi05_libero_async \
  --exp-name YOUR_NEW_DELAY_RUN \
  --pytorch-weight-path checkpoints/pi05_libero_async/gpu_smoke_sync_20261012/2 \
  --no-pytorch-allow-new-parameters \
  --assets-base-dir ./assets_smoke \
  --data.episodes 0 --data.prefix-delay-max 5 \
  --batch-size 1 --num-workers 0 --num-train-steps 2 \
  --log-interval 1 --save-interval 1000 --lr-schedule.warmup-steps 1 \
  --no-wandb-enabled
```

两个阶段均完成反向传播、optimizer 更新和最后一步 checkpoint 保存；延迟阶段严格加载同步阶段的完整模型。VLM/expert 的代表参数均有实际变化，新增 state projection 更新，age 3、4 的 embedding 行更新。

| 阶段 | GPU / batch | loss（两步） | 峰值 allocated 显存 |
|---|---|---|---|
| 同步双 proprio | 0、2 / 总 batch 2 | 0.0996、0.0615 | 31.95 GB/卡 |
| 随机延迟 0–5 | 2 / batch 1 | 0.0995、0.1535 | 31.21 GB |

日志为 `logs/gpu_smoke_sync_20261012.log`、`logs/gpu_smoke_delay_20261012.log`；参数变化证据为 `logs/gpu_smoke_verification_20261012.json`。完整 checkpoint 分别位于 `checkpoints/pi05_libero_async/gpu_smoke_sync_20261012/2` 和 `checkpoints/pi05_libero_async/gpu_smoke_delay_20261012/2`。

这次还修复了训练器最后一步保存的计数偏差，并验证保存后恢复的是已完成步数。短程训练没有测量任务成功率。全量微调需关闭上述 offline 选项、取消 `--data.episodes` 限制、重新计算全量 norm stats，并使用正常 warmup；不能使用这里的 smoke stats 作为全量统计。

## AutoDL 四卡 A800（2026-10-12）

独立环境为 `~/.conda/envs/openpi`：Python 3.11.17、PyTorch 2.7.1+cu126、Transformers 4.53.2，并应用五个官方 replacement 文件。原服务器环境未修改。四张 A800-SXM4-80GB 通过 NVLink 连接；实际 full fine-tuning 校准如下，未使用 LoRA。

| global batch / 每卡 | prefix delay / workers 每进程 | 峰值 allocated | 训练步耗时 |
|---|---|---|---|
| 64 / 16 | 0 / 0 | rank 0：34.94 GB | 4.9、5.9 秒 |
| 128 / 32 | 0–5 / 0 | rank 0：41.42 GB | 18.3、17.1 秒 |
| 256 / 64 | 0–5 / 4 | 四卡：54.39–54.42 GB | 43.4 秒启动，随后 10.2、10.1 秒 |

batch 256 的最终峰值 reserved 为 56.87–58.99 GB。三个校准都完成真实 backward/optimizer 更新；256 使用 episode 0–3，其余使用 episode 0，均使用 smoke stats。由于 delay 和 workers 不同，这些耗时不能用作只改变 batch 的性能比较，也不能用于判断任务成功率。[W&B batch 256 记录](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/gwuttmtu)。

建议 global batch 256，每卡 64；同步阶段 30,000 steps、warmup 1,000，延迟阶段从同步 checkpoint 初始化新实验，10,000 steps、warmup 500。expert peak LR 为 5e-5，VLM 为 5e-6。step 数是初始实验预算，应结合 LIBERO rollout 评估选择 checkpoint。

代码位于 `~/sharefs/mlsys/openpi`。raw 数据与 checkpoints 放在 sharefs；base 权重、依赖缓存和 Arrow cache 放在 `/root/autodl-tmp/$USER`。下载完整数据（1693 episodes、273465 frames、40 tasks）并生成全量 stats：

```bash
cd ~/sharefs/mlsys/openpi
source scripts/autodl_env.sh
python scripts/prepare_libero_autodl.py
```

本机测速中，官方 CDN 直连比 AutoDL 转发更快，环境脚本将 `us.aws.cdn.hf.co` 加入 `no_proxy`。如官方 API 超时，可在启动下载进程前设置 `HF_ENDPOINT=https://hf-mirror.com`，并将 `hf-mirror.com` 加入 `no_proxy` 与 `NO_PROXY`。已核对镜像返回相同不可变 commit `9dfa69510ea9e1613fc54112bc706444b686a231`；下载脚本仍固定该数据版本。

W&B 凭据存放在 `~/.config/openpi/wandb.env`，由环境脚本加载。正式训练命令：

```bash
bash scripts/train_libero_autodl.sh sync

# STEP 替换为已保存的同步 checkpoint 步数。
INIT_CHECKPOINT="$PWD/checkpoints/pi05_libero_async/dual_state_sync_a800/STEP" \
  bash scripts/train_libero_autodl.sh delay

# 中断后恢复同一个同步实验。
bash scripts/train_libero_autodl.sh sync --resume
```

launcher 默认保存间隔 1,000 steps，只保留最新完整 checkpoint；未额外保留周期 checkpoint，适配个人存储额度。长程实验启动前，必须确认全量 stats 与数据已准备完成；这里的 batch 校准不等于已开始 30,000 steps 正式训练。
