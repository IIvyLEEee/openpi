#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/autodl_env.sh"
cd "$OPENPI_REPO_ROOT"

phase="${1:-sync}"
if (( $# )); then shift; fi
case "$phase" in
    sync)
        init_checkpoint="${INIT_CHECKPOINT:-$OPENPI_LOCAL_ROOT/models/pi05_base_pytorch}"
        steps="${TRAIN_STEPS:-30000}"
        warmup="${WARMUP_STEPS:-1000}"
        delay=0
        init_flags=(--pytorch-allow-new-parameters)
        ;;
    delay)
        : "${INIT_CHECKPOINT:?Set INIT_CHECKPOINT to the complete synchronous checkpoint directory}"
        init_checkpoint="$INIT_CHECKPOINT"
        steps="${TRAIN_STEPS:-10000}"
        warmup="${WARMUP_STEPS:-500}"
        delay=5
        init_flags=(--no-pytorch-allow-new-parameters)
        ;;
    *) printf 'Usage: %s {sync|delay} [extra training arguments]\n' "$0" >&2; exit 2 ;;
esac

: "${WANDB_API_KEY:?Configure W&B in ~/.config/openpi/wandb.env or export WANDB_API_KEY}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
exec "$OPENPI_ENV_PREFIX/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-4}" \
    scripts/train_pytorch.py pi05_libero_async \
    --exp-name "${EXP_NAME:-dual_state_${phase}_a800}" \
    --project-name "$WANDB_PROJECT" \
    --pytorch-weight-path "$init_checkpoint" "${init_flags[@]}" \
    --batch-size "${BATCH_SIZE:-64}" --num-workers "${NUM_WORKERS:-4}" \
    --num-train-steps "$steps" --data.prefix-delay-max "$delay" \
    --lr-schedule.warmup-steps "$warmup" --lr-schedule.decay-steps "$steps" \
    --log-interval 10 --save-interval 1000 \
    --pytorch-keep-last-checkpoints 1 --keep-period None --wandb-enabled "$@"
