# UMI Conveyor Fine-Tuning Start Guide

This guide starts from a fresh machine that has the repository code and the raw UMI zarr zip, but none of the ignored data directories such as `assets/`, `checkpoints/`, `data/`, `.hf_cache/`, `.openpi_cache/`, `logs/`, or anything under `/data/archive_liyixuan23/finetune_pi`.

The target workflow is:

1. Prepare cache and output directories.
2. Convert raw UMI zarr zip to LeRobot parquet.
3. Compute norm stats for `pi0_umi_ur5e_lora` and `pi05_umi_ur5e_lora`.
4. Run 1-step smoke training.
5. Start formal LoRA fine-tuning.

## 0. Assumptions

Default paths used below:

```bash
export OPENPI_ROOT=/home/liyixuan23/openpi
export DATA_ROOT=/data/archive_liyixuan23/finetune_pi
export UMI_ROOT=/data/archive_liyixuan23/umi_for_train
export RAW_ZARR_ZIP=$UMI_ROOT/example_pick_night/mixdataset.zarr.zip
```

The raw UMI zarr zip is assumed to already exist at:

```text
/data/archive_liyixuan23/umi_for_train/example_pick_night/mixdataset.zarr.zip
```

This repository version must include the UMI additions:

```bash
test -f examples/ur5/convert_umi_zarr_to_lerobot.py
test -f src/openpi/policies/umi_ur5e_policy.py
rg -n "pi0_umi_ur5e_lora|pi05_umi_ur5e_lora|ShapeCompatibleCheckpointWeightLoader" src/openpi/training
```

The converter uses the UMI JPEG-XL codec registration file that is vendored in this repo:

```text
examples/ur5/codecs/imagecodecs_numcodecs.py
```

The converter also has a legacy fallback to `/data/archive_liyixuan23/umi_for_train/diffusion_policy/codecs`, but a fresh machine should not need that external UMI tree as long as the vendored file exists.

## 1. Environment Setup

Install dependencies from the repo:

```bash
cd $OPENPI_ROOT
uv sync
```

Create all data/cache/output directories on the data disk:

```bash
mkdir -p \
  $DATA_ROOT/openpi_zarr_cache/umi_conveyor \
  $DATA_ROOT/openpi_lerobot \
  $DATA_ROOT/openpi_cache \
  $DATA_ROOT/hf_cache/datasets \
  $DATA_ROOT/checkpoints \
  logs
```

Use these environment variables for all conversion/stat/training commands:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
export HF_HOME=$DATA_ROOT/hf_cache
export HF_DATASETS_CACHE=$DATA_ROOT/hf_cache/datasets
export HF_LEROBOT_HOME=$DATA_ROOT/openpi_lerobot
export OPENPI_DATA_HOME=$DATA_ROOT/openpi_cache
```

Optional convenience symlinks from the repo:

```bash
ln -sfn $DATA_ROOT/hf_cache $OPENPI_ROOT/.hf_cache
ln -sfn $DATA_ROOT/openpi_cache $OPENPI_ROOT/.openpi_cache
mkdir -p $OPENPI_ROOT/data
ln -sfn $DATA_ROOT/checkpoints $OPENPI_ROOT/data/outputs
```

Check GPU/JAX visibility:

```bash
nvidia-smi
uv run python - <<'PY'
import jax
print(jax.__version__)
print(jax.devices())
print(jax.default_backend())
PY
```

For training on shared machines, choose idle GPUs explicitly with `CUDA_VISIBLE_DEVICES`. In the current workflow, avoid GPU 0 unless the machine policy says otherwise.

## 2. Convert Raw Zarr To LeRobot

Expected raw dataset facts:

```text
episodes: 196
frames: 147180
action/state dim: 7
state/action layout: eef_pos(3), eef_rot_axis_angle(3), gripper_width(1)
prompt: pick up the red block on the conveyor belt. put it in the blue box.
```

Run conversion:

```bash
cd $OPENPI_ROOT

uv run examples/ur5/convert_umi_zarr_to_lerobot.py \
  --zarr-zip $RAW_ZARR_ZIP \
  --extract-dir $DATA_ROOT/openpi_zarr_cache/umi_conveyor/mixdataset.zarr \
  --output-root $DATA_ROOT/openpi_lerobot/liyixuan23/umi_conveyor \
  --overwrite \
  --overwrite-extract
```

Expected output:

```text
$DATA_ROOT/openpi_lerobot/liyixuan23/umi_conveyor
```

Validate conversion:

```bash
uv run python - <<'PY'
import json
import pathlib
import pyarrow.parquet as pq

root = pathlib.Path("/data/archive_liyixuan23/finetune_pi/openpi_lerobot/liyixuan23/umi_conveyor")
info = json.loads((root / "meta/info.json").read_text())
parquets = sorted((root / "data/chunk-000").glob("episode_*.parquet"))
rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parquets)
last = pq.read_table(root / "data/chunk-000/episode_000195.parquet")

print("episodes", info["total_episodes"])
print("frames", info["total_frames"])
print("parquet_files", len(parquets))
print("parquet_rows", rows)
print("episode_000195_rows", last.num_rows)
PY
```

Expected validation:

```text
episodes 196
frames 147180
parquet_files 196
parquet_rows 147180
episode_000195_rows 838
```

## 3. Compute Norm Stats

`assets/` is gitignored and will be empty on a fresh machine, so norm stats must be generated before training.

Compute pi0 norm stats:

```bash
cd $OPENPI_ROOT

UV_CACHE_DIR=/tmp/uv-cache \
HF_HOME=$DATA_ROOT/hf_cache \
HF_DATASETS_CACHE=$DATA_ROOT/hf_cache/datasets \
HF_LEROBOT_HOME=$DATA_ROOT/openpi_lerobot \
OPENPI_DATA_HOME=$DATA_ROOT/openpi_cache \
uv run scripts/compute_norm_stats.py --config-name pi0_umi_ur5e_lora
```

For this dataset, pi0 and pi05 use the same repack/data transforms for `state` and `actions`, so the stats can be copied:

```bash
mkdir -p assets/pi05_umi_ur5e_lora/liyixuan23/umi_conveyor
cp assets/pi0_umi_ur5e_lora/liyixuan23/umi_conveyor/norm_stats.json \
  assets/pi05_umi_ur5e_lora/liyixuan23/umi_conveyor/norm_stats.json
```

Alternatively, compute pi05 separately:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
HF_HOME=$DATA_ROOT/hf_cache \
HF_DATASETS_CACHE=$DATA_ROOT/hf_cache/datasets \
HF_LEROBOT_HOME=$DATA_ROOT/openpi_lerobot \
OPENPI_DATA_HOME=$DATA_ROOT/openpi_cache \
uv run scripts/compute_norm_stats.py --config-name pi05_umi_ur5e_lora
```

Validate stats exist:

```bash
test -f assets/pi0_umi_ur5e_lora/liyixuan23/umi_conveyor/norm_stats.json
test -f assets/pi05_umi_ur5e_lora/liyixuan23/umi_conveyor/norm_stats.json
```

## 4. Smoke Training

Base checkpoints download automatically from `gs://openpi-assets` and are cached under:

```text
$DATA_ROOT/openpi_cache/openpi-assets/checkpoints
```

Run `nvidia-smi` first and choose idle GPUs. The examples below use GPU 1.

pi0 1-step smoke:

```bash
tmux new-session -d -s pi0_umi_smoke \
  "cd $OPENPI_ROOT && \
  CUDA_VISIBLE_DEVICES=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  UV_CACHE_DIR=/tmp/uv-cache \
  HF_HOME=$DATA_ROOT/hf_cache \
  HF_DATASETS_CACHE=$DATA_ROOT/hf_cache/datasets \
  HF_LEROBOT_HOME=$DATA_ROOT/openpi_lerobot \
  OPENPI_DATA_HOME=$DATA_ROOT/openpi_cache \
  uv run scripts/train.py pi0_umi_ur5e_lora \
    --exp-name smoke \
    --overwrite \
    --num-train-steps 1 \
    --batch-size 1 \
    --num-workers 0 \
    --checkpoint-base-dir $DATA_ROOT/checkpoints \
    > logs/pi0_umi_smoke.log 2>&1"
```

pi05 1-step smoke:

```bash
tmux new-session -d -s pi05_umi_smoke \
  "cd $OPENPI_ROOT && \
  CUDA_VISIBLE_DEVICES=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  UV_CACHE_DIR=/tmp/uv-cache \
  HF_HOME=$DATA_ROOT/hf_cache \
  HF_DATASETS_CACHE=$DATA_ROOT/hf_cache/datasets \
  HF_LEROBOT_HOME=$DATA_ROOT/openpi_lerobot \
  OPENPI_DATA_HOME=$DATA_ROOT/openpi_cache \
  uv run scripts/train.py pi05_umi_ur5e_lora \
    --exp-name smoke \
    --overwrite \
    --num-train-steps 1 \
    --batch-size 1 \
    --num-workers 0 \
    --checkpoint-base-dir $DATA_ROOT/checkpoints \
    > logs/pi05_umi_smoke.log 2>&1"
```

Monitor:

```bash
tmux ls
tail -f logs/pi0_umi_smoke.log
tail -f logs/pi05_umi_smoke.log
```

Expected behavior:

```text
ShapeCompatibleCheckpointWeightLoader skips only incompatible 32D->7D action/state head params.
Step 0 prints loss, grad_norm, param_norm.
Smoke checkpoints are written under $DATA_ROOT/checkpoints.
```

## 5. Formal Training

Current UMI configs:

```text
config: pi0_umi_ur5e_lora / pi05_umi_ur5e_lora
num_train_steps: 30000
global batch_size: 16
num_workers: 4
save_interval: 1000
keep_period: 5000
wandb_enabled: false by default, enable with --wandb-enabled
```

`batch_size` is global. With two GPUs and `--batch-size 16`, each GPU gets 8 samples.

The dataset has 147180 frames, so `30000` steps at global batch 16 is about:

```text
147180 / 16 = 9199 steps per epoch
30000 / 9199 = 3.26 epochs
```

Recommended pi0 two-GPU data-parallel run:

```bash
tmux new-session -d -s pi0_umi_2gpu \
  "cd $OPENPI_ROOT && \
  CUDA_VISIBLE_DEVICES=1,2 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  WANDB_MODE=online \
  UV_CACHE_DIR=/tmp/uv-cache \
  HF_HOME=$DATA_ROOT/hf_cache \
  HF_DATASETS_CACHE=$DATA_ROOT/hf_cache/datasets \
  HF_LEROBOT_HOME=$DATA_ROOT/openpi_lerobot \
  OPENPI_DATA_HOME=$DATA_ROOT/openpi_cache \
  uv run scripts/train.py pi0_umi_ur5e_lora \
    --exp-name pi0_umi_2gpu_lora \
    --overwrite \
    --wandb-enabled \
    --checkpoint-base-dir $DATA_ROOT/checkpoints \
    --batch-size 16 \
    --num-workers 4 \
    > logs/pi0_umi_2gpu.log 2>&1"
```

Recommended pi05 command, after pi0 is healthy:

```bash
tmux new-session -d -s pi05_umi_2gpu \
  "cd $OPENPI_ROOT && \
  CUDA_VISIBLE_DEVICES=1,2 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  WANDB_MODE=online \
  UV_CACHE_DIR=/tmp/uv-cache \
  HF_HOME=$DATA_ROOT/hf_cache \
  HF_DATASETS_CACHE=$DATA_ROOT/hf_cache/datasets \
  HF_LEROBOT_HOME=$DATA_ROOT/openpi_lerobot \
  OPENPI_DATA_HOME=$DATA_ROOT/openpi_cache \
  uv run scripts/train.py pi05_umi_ur5e_lora \
    --exp-name pi05_umi_2gpu_lora \
    --overwrite \
    --wandb-enabled \
    --checkpoint-base-dir $DATA_ROOT/checkpoints \
    --batch-size 16 \
    --num-workers 4 \
    > logs/pi05_umi_2gpu.log 2>&1"
```

Monitor training:

```bash
tmux ls
tail -f logs/pi0_umi_2gpu.log
nvidia-smi
du -sh $DATA_ROOT/checkpoints
```

Resume an interrupted run:

```bash
tmux new-session -d -s pi0_umi_2gpu_resume \
  "cd $OPENPI_ROOT && \
  CUDA_VISIBLE_DEVICES=1,2 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  WANDB_MODE=online \
  UV_CACHE_DIR=/tmp/uv-cache \
  HF_HOME=$DATA_ROOT/hf_cache \
  HF_DATASETS_CACHE=$DATA_ROOT/hf_cache/datasets \
  HF_LEROBOT_HOME=$DATA_ROOT/openpi_lerobot \
  OPENPI_DATA_HOME=$DATA_ROOT/openpi_cache \
  uv run scripts/train.py pi0_umi_ur5e_lora \
    --exp-name pi0_umi_2gpu_lora \
    --resume \
    --wandb-enabled \
    --checkpoint-base-dir $DATA_ROOT/checkpoints \
    --batch-size 16 \
    --num-workers 4 \
    > logs/pi0_umi_2gpu_resume.log 2>&1"
```

## 6. Storage Estimate

Measured sizes from the current machine:

```text
raw zarr zip:                    3.4G
extracted zarr cache:             3.6G
converted LeRobot dataset:        8.8G
HF Arrow cache:                   8.8G
openpi base checkpoint cache:
  pi0_base only:                 ~12G
  pi0_base + pi05_base:          ~23G
single formal train checkpoint:  8.8-8.9G
current pi0 30k checkpoint set:  ~54G
```

Why the pi0 checkpoint set is about 54G:

```text
save_interval=1000
keep_period=5000
max_to_keep keeps the latest checkpoint
kept checkpoints are roughly: 5000, 10000, 15000, 20000, 25000, latest
6 checkpoints * ~8.9G = ~53G
```

Recommended data disk sizes:

```text
100G: minimal pi0-only run, with careful cleanup and limited checkpoint retention.
150G: practical minimum for pi0 + pi05 or one full run plus caches.
200G: recommended default, avoids mid-run cleanup and leaves room for retries.
300G+: recommended if keeping both pi0 and pi05 full checkpoint histories.
```

Cleanup options after validation:

```bash
# Remove extracted zarr cache after LeRobot conversion is verified.
rm -rf $DATA_ROOT/openpi_zarr_cache/umi_conveyor/mixdataset.zarr

# Keep raw zip if you want reproducibility; otherwise archive it elsewhere.
du -sh $RAW_ZARR_ZIP

# Remove failed/smoke checkpoints after formal training is healthy.
du -sh $DATA_ROOT/checkpoints/*/*/smoke 2>/dev/null
```

Do not delete:

```text
$DATA_ROOT/openpi_lerobot/liyixuan23/umi_conveyor
$DATA_ROOT/openpi_cache/openpi-assets/checkpoints/pi0_base
assets/pi0_umi_ur5e_lora/liyixuan23/umi_conveyor/norm_stats.json
assets/pi05_umi_ur5e_lora/liyixuan23/umi_conveyor/norm_stats.json
```

## 7. Common Failure Points

Missing codec:

```text
ModuleNotFoundError: imagecodecs_numcodecs
```

Fix by making sure the repo contains:

```text
examples/ur5/codecs/imagecodecs_numcodecs.py
```

Missing norm stats:

```text
Make sure to run scripts/compute_norm_stats.py --config-name=<your-config>.
```

Fix by running the norm stats commands in section 3.

Wrong cache path:

```text
Large base checkpoints appear under .openpi_cache or ~/.cache/openpi.
```

Fix by exporting:

```bash
export OPENPI_DATA_HOME=$DATA_ROOT/openpi_cache
```

Batch not divisible by visible GPU count:

```text
Batch size <N> must be divisible by the number of devices <M>.
```

Fix by changing `--batch-size` or `CUDA_VISIBLE_DEVICES`.

OOM with larger batch:

```text
Current measured two-GPU data-parallel run uses about 44.3G/49.1G per GPU at global batch 16.
```

Keep `--batch-size 16` unless you have verified a larger batch with a short benchmark.
