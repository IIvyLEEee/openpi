#!/usr/bin/env bash
# Source from a shell, or through train_libero_autodl.sh. Credentials stay outside the repository.
export OPENPI_REPO_ROOT="${OPENPI_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export OPENPI_ENV_PREFIX="${OPENPI_ENV_PREFIX:-$HOME/.conda/envs/openpi}"
export OPENPI_LOCAL_ROOT="${OPENPI_LOCAL_ROOT:-/root/autodl-tmp/$USER}"
export PATH="$OPENPI_ENV_PREFIX/bin:$HOME/.local/bin:$PATH"

if [[ -f /etc/network_turbo ]]; then
    source /etc/network_turbo >/dev/null
fi
# Use forwarding for API traffic; the official dataset CDN is faster over a direct connection.
export no_proxy="${no_proxy:+$no_proxy,}pypi.org,files.pythonhosted.org,us.aws.cdn.hf.co"
export NO_PROXY="$no_proxy"
if [[ -f "$HOME/.config/openpi/wandb.env" ]]; then
    source "$HOME/.config/openpi/wandb.env"
fi

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$OPENPI_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$OPENPI_LOCAL_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_LEROBOT_HOME="$OPENPI_REPO_ROOT/data/lerobot"
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_XET=1
export OPENPI_DATA_HOME="$OPENPI_LOCAL_ROOT/cache/openpi"
export UV_CACHE_DIR="$OPENPI_LOCAL_ROOT/cache/uv"
export UV_LINK_MODE=copy
export WANDB_DIR="$OPENPI_REPO_ROOT/wandb"
export WANDB_CACHE_DIR="$OPENPI_LOCAL_ROOT/cache/wandb"
export WANDB_DATA_DIR="$OPENPI_LOCAL_ROOT/cache/wandb-data"
export WANDB_PROJECT="${WANDB_PROJECT:-pi05-async-libero}"
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME" "$OPENPI_DATA_HOME" "$WANDB_DIR"
