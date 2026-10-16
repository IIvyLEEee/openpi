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

参数前缀 `args.` 与 `main.py` 的 `Args` CLI 一致。默认每秒 10 个壁钟 tick，每个 tick 推进一次原有 LIBERO `env.step`。没有更改环境的 `control_freq=20`，因此一秒壁钟时间约推进 0.5 秒仿真时间；10 Hz 不是新的仿真控制频率。延迟标签以数据/环境步数计，不用壁钟秒数代替。

两个 worker 分别持有 VLM 或 expert 模块，可放在不同 GPU；使用同一 GPU 时会争用算力，不能据此假定有加速。首次观测会等待有效 cache 和动作，之后 expert 不等待下一次 VLM 更新。KV 每次更新只向 expert device 转移一次。

主循环按仿真步数发布观测并对齐新 chunk：从观测步到执行步已经过去几步，就跳过新 chunk 的相应位置。`replan_steps=5` 限制使用 chunk 的前 5 个位置；推理期间已过去的位置不再执行。无可用动作时输出零 EEF 增量并保持夹爪指令，不重复过期位移。

训练年龄上限约束采样时 `state_step - prefix_step`，expert 仅在这个差值为 0–5 时采样。训练监督包含后续动作，因此已经生成的 chunk 可以在有效位置内继续执行；执行时的 prefix 年龄等于采样年龄加动作 offset，可以超过 5。若 chunk 已用尽且 prefix 过期，runner 抛出 `PrefixExpiredError`，评估主循环保持姿态，等待 worker 更新；模型或基础设施错误直接中止评估。reset/task 切换通过 generation 丢弃旧的在途结果，每个 episode 结束后等待在途 worker 退出再重设随机种子。

输出目录包含本地视频、`protocol.json`、`episodes.jsonl` 和 `summary.json`。使用与同步评估相同的图像/state 处理和逐 episode seed；记录成功率、worker 延迟、所用 cache_id、采样/执行年龄、动作 offset 和没有动作就绪的 tick，支持同协议断点续跑。没有额外注入延迟，也没有 streaming/action-buffer conditioning；实际延迟由两个 worker 的运行产生。W&B 可仅上传指标和 JSON，视频保留本地。

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

全量准备已完成：1693 个 Parquet 共 34,938,730,091 bytes，逐文件与官方固定 commit 的 LFS SHA256 一致；全量 norm stats 覆盖 273465 帧，LeRobot Arrow cache 已构建。使用全量数据与全量 stats 的四卡启动验证完成三步（loss 0.1116、0.1011、0.1147），最后保存模型、optimizer、metadata 和相同 norm stats。核对 checkpoint 后，VLM 有 597 个参数的 optimizer 状态、expert/新增模块有 211 个；两组 moment 均有限且存在非零值，代表参数有实际变化。正常 warmup 1000 / decay 30000 保留。

[全量启动验证 W&B](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/ci4ploeh)。远程证据位于 `logs/libero_integrity.json`、`logs/full_data_preparation.log` 和 `logs/full_data_checkpoint_verification_3.json`。本地保存于 `logs/autodl_a800_20261012/`。该运行用于工程验证，step 数不代表模型已收敛。

随后严格恢复模型、optimizer 和 metadata，从已完成的第 3 步继续至第 4 步，并成功保存及再次核验；W&B 同一 run 的 step/checkpoint_step 均为 4。retention 已移除第 3 步目录，最新完整 checkpoint 为 `checkpoints/pi05_libero_async/full_data_startup_a800/4`。恢复日志与最终核验分别为 `logs/full_data_resume.log`、`logs/full_data_checkpoint_verification_4.json`。验证结束后四张 GPU 均已释放；正式同步实验使用上面的默认 launcher，从 base 权重启动新的 `dual_state_sync_a800` 实验。

## 两组各两卡的 40k 对比实验

按用户指定方案，两个路线同时运行，均使用全部 40 个任务、global batch 128（每卡 64）、seed 42、VLM/expert full fine-tuning，以及同一套全量 norm stats。

| 路线 | GPU | 实验名 | steps / warmup | prefix delay |
|---|---|---|---|---|
| 直接延迟 | 0、1 | `dual_state_direct_delay_40k_b128` | 40000 / 1000 | 0–5 ticks |
| 先同步 | 2、3 | `dual_state_sync_30k_b128` | 30000 / 1000 | 0 |
| 接延迟 | 2、3 | `dual_state_sync30k_delay10k_b128` | 10000 / 500 | 0–5 ticks |

前两项均从原 base 初始化新增参数。第三项自动严格加载同步 30000-step 模型，重新初始化 optimizer 和 LR schedule；各阶段的 cosine decay 覆盖该阶段的总 steps，expert / VLM peak LR 仍为 5e-5 / 5e-6。两个路线包含不同 LR schedule，比较结果应按这套实际训练方案解释。

```bash
# 分别在两个持久 shell / tmux session 启动。
bash scripts/train_libero_compare_autodl.sh direct
bash scripts/train_libero_compare_autodl.sh staged
```

`staged` 成功完成同步阶段后才开始延迟阶段；失败则停止，不用不完整权重继续。重启同一路线会恢复已有完整 checkpoint，已完成阶段根据 marker 跳过。日志位于 `logs/libero_compare_40k_b128/`，W&B group 为 `pi05_async_40k_b128_compare`。每个路线使用独立的 torchrun rendezvous；启动锁阻止重复运行同一路线。

checkpoint 仍每 1000 steps 保存，只留各活动阶段最新完整 checkpoint；共享 `pytorch_checkpoint_lock_path` 使写入及旧文件清理串行执行，避免两个实验同时写入临时 checkpoint 超出个人额度。同步 30k 完成后保留模型、metadata 和 norm stats，移除该已完成阶段的 optimizer；延迟阶段使用新的 optimizer。四步启动验证的模型/metadata/stats 归档到 `~/openpi-validation/full_data_startup_a800/4`，其短程 optimizer 已清理，为正式实验腾出存储空间。

两路线已于 2026-10-12 在 tmux session `pi05-direct-40k`、`pi05-staged-40k` 启动，并完成多步真实优化，峰值 allocated 约 54.4 GB/卡、预热后约 9.8 秒/步。W&B 服务端确认两 run 均为 running、同一 comparison group，直接延迟的条件年龄最大 5，同步为 0。[直接延迟 40k](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/0ghenb07)；[同步 30k](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/wr08liy0)。阶段二成功启动后会创建独立 W&B run。实际配置与启动核验分别保存在 `logs/libero_compare_40k_b128/experiment_manifest.json`、`wandb_startup_verified.json`。

## 同步 30k 后的四卡延迟阶段（2026-10-16）

用户将直接延迟路线的停止点改为 30k，并要求同步路线满 30k 后使用全部四张 A800 继续 10k 延迟微调。直接路线停止时计算日志到 30586，保留完整的 30000-step checkpoint（含远程 optimizer）；同步路线已经完成 30000。原自动启动的两卡延迟阶段运行到 337，尚无 checkpoint，已中止并保留日志和 W&B 历史；新的四卡阶段从同步 30k 重新初始化，不接续这 337 步。

```bash
bash scripts/train_libero_compare_autodl.sh delay4
```

该入口选择 GPU 0–3 / 4 个 DDP ranks，global batch 仍为 128，因此每卡 batch 从 64 改为 32。实验名为 `dual_state_sync30k_delay10k_b128_4gpu`，从 `checkpoints/pi05_libero_async/dual_state_sync_30k_b128/30000` 严格加载模型；重新初始化 optimizer，训练 10000 steps，delay 0–5、warmup 500、cosine decay 10000，expert / VLM peak LR 保持 5e-5 / 5e-6。不因卡数翻倍调整 LR。完整 checkpoint 可断点恢复，成功结束时写入 `training_complete`，共享保存锁与 retention 规则保持原样。

直接延迟的 30k 模型、metadata 和 norm stats 已完整传至本机 `eval/checkpoints/20261016_direct_30000/dual_state_direct_delay_40k_b128/30000/`。模型 7,473,251,472 bytes / 815 tensors，SHA256 与服务器一致：`891686019757955285cba85b254fdac5a3fa8f9381647aea4ade57065779ad9c`；metadata 的 step/batch/delay 及 stats 校验通过，传输分片已清理。服务器原 30k checkpoint 和 optimizer 保留；仅模型与评估所需配套文件传回本机。

确认传输完成后已启动四卡任务，tmux 为 `pi05-delay4-10k`，[W&B run 为 zd7d4bze](https://wandb.ai/iivyleeee-tsinghua-university/pi05-async-libero/runs/zd7d4bze)。启动核验时已到第 788/10000 步，4 个 ranks 的 WORLD_SIZE=4，模型与 DataLoader 的实际 batch 均为每卡 32 / global 128；loss 与梯度有限，prefix_age_max=5，约 5.06 秒/步，四卡利用率均为 100%、显存约 42 GB/卡。10k 训练仍在进行中。切换记录与核验位于远程 `logs/libero_compare_40k_b128/continuation_4gpu_20261016.json`、`four_gpu_startup_verified_20261016.json`；本地证据为 `eval/30000_transfer/four_gpu_wandb_verified.json`、`four_gpu_status.json` 及 checkpoint 根目录下的 `verified_manifest.json`。

## 本地 checkpoint rollout

`examples/libero/evaluate_checkpoint.py` 直接加载 PyTorch checkpoint 与其全量 norm stats，使用 OpenPI 固定的 LIBERO commit `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。评估依赖单独安装到继承 OpenPI 的隔离环境，版本见 `examples/libero/requirements_eval.txt`；不修改训练环境。PyTorch 2.7 读取可信的官方 `.pruned_init` 文件时需要 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。

```bash
git submodule update --init third_party/libero
python -m venv --system-site-packages .venv-libero-eval
.venv-libero-eval/bin/python -m pip install -r examples/libero/requirements_eval.txt
# LIBERO_CONFIG_PATH 中的 config.yaml 应指向本地 LIBERO 的 bddl_files/init_states/assets。
CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cpu \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  PYTHONPATH=third_party/libero:$PYTHONPATH \
  .venv-libero-eval/bin/python examples/libero/evaluate_checkpoint.py \
    --args.checkpoint-dir /ABSOLUTE/CHECKPOINT/6000 \
    --args.output-dir eval/results/YOUR_RUN/6000
```

默认协议：四个 suite 共 40 个任务，每任务使用官方初始状态 0–4，各 5 次，seed 7；10 个 flow steps，每次生成 10 个动作、执行前 5 个再获取新观测；最大控制步数 Spatial/Object/Goal/Long 分别为 220/280/300/520。这是同步 fresh-observation rollout，prefix age 为 0，不衡量异步部署表现。两 checkpoint 使用相同初始状态与逐 episode 随机种子。

结果逐 episode 保存至 `episodes.jsonl`，汇总保存在 `summary.json`；`complete=true` 才表示完整评估。相同目录支持断点续跑，协议不一致时拒绝混写；仿真/模型异常直接中止，不混入正常失败样本。每个任务保存首个成功与首个失败视频。

2026-10-13：取用户请求时两路线最新完整的 6000-step checkpoint，本地空闲 GPU 2、3 分别评估 direct-delay 与同步路线，各 200 episodes。远程训练继续运行。评估单独记录到 W&B group `pi05_libero_ckpt6000_rollout_20261013`，不写入正在训练的 run。

本轮已完成：direct-delay 128/200（64.0%），同步 131/200（65.5%）。任务覆盖、初始状态、种子与成功计数核对通过，两组退出码均为 0；各 suite 的结果见 [评估报告](results/pi05_libero_ckpt6000_20261013.md)。按用户要求，视频仅保存在本地，W&B 只同步指标与 JSON 结果。后续需要在相同条件下比较两种 checkpoint 的真实双 worker 异步 rollout，以及受控 cache delay；本轮结果不能替代异步评估。

同日使用官方 `pi05_libero` checkpoint 进行 Spatial 校准：复用同一评估入口、初始状态与随机种子，50/50（100%），退出码 0；我们的 direct/sync 6000-step checkpoint 分别为 40/50（80%）、37/50（74%）。评估统一关闭 torch.compile，与既有双状态配置一致。官方源文件经过 MD5/CRC32C 校验，使用官方 converter 转换并复制其 norm stats；详见 [官方 Spatial 校准报告](results/pi05_libero_official_spatial_20261013.md)。当前流程能评估 stock 模型的高成功率，但不能据此排除双状态模型路径的问题；训练步数、结构及训练配置差异仍需进一步验证。
