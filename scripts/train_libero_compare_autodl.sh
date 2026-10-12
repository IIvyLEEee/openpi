#!/usr/bin/env bash
# Two independent two-GPU experiments. Rerunning resumes the active phase.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/autodl_env.sh"
cd "$OPENPI_REPO_ROOT"
route="${1:?Usage: train_libero_compare_autodl.sh direct|staged}"
case "$route" in
    direct) export CUDA_VISIBLE_DEVICES=0,1 ;;
    staged) export CUDA_VISIBLE_DEVICES=2,3 ;;
    *) printf 'Unknown route: %s\n' "$route" >&2; exit 2 ;;
esac
export NPROC_PER_NODE=2 BATCH_SIZE=128 NUM_WORKERS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_RUN_GROUP=pi05_async_40k_b128_compare
export WANDB_TAGS=dual-state,full-finetune,batch128,"$route"
export CHECKPOINT_LOCK_PATH="$OPENPI_REPO_ROOT/checkpoints/.save.lock"
log_dir="$OPENPI_REPO_ROOT/logs/libero_compare_40k_b128"
mkdir -p "$log_dir"
exec 9>"$log_dir/$route.launch.lock"
flock -n 9 || { printf '%s is already running\n' "$route" >&2; exit 1; }

run_phase() {
    local phase="$1" experiment="$2" steps="$3"
    local checkpoint_dir="$OPENPI_REPO_ROOT/checkpoints/pi05_libero_async/$experiment"
    if [[ -f "$checkpoint_dir/training_complete" ]]; then
        printf 'Already completed %s\n' "$experiment"
        return
    fi
    local resume_flags=()
    if [[ -f "$checkpoint_dir/wandb_id.txt" ]]; then
        if python - "$checkpoint_dir" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
complete = any(p.name.isdigit() and all((p / f).is_file() for f in
    ('model.safetensors', 'optimizer.pt', 'metadata.pt')) for p in root.iterdir() if p.is_dir())
sys.exit(0 if complete else 1)
PY
        then resume_flags=(--resume)
        else resume_flags=(--overwrite)
        fi
    fi
    printf 'Starting %s: GPU %s, global batch 128, %s steps\n' "$experiment" "$CUDA_VISIBLE_DEVICES" "$steps"
    EXP_NAME="$experiment" TRAIN_STEPS="$steps" bash scripts/train_libero_autodl.sh "$phase" \
        "${resume_flags[@]}" >> "$log_dir/$experiment.log" 2>&1
    test -f "$checkpoint_dir/$steps/model.safetensors"
    test -f "$checkpoint_dir/$steps/optimizer.pt"
    printf '%s\n' "$steps" > "$checkpoint_dir/training_complete"
}

if [[ "$route" == direct ]]; then
    run_phase direct-delay dual_state_direct_delay_40k_b128 40000
else
    run_phase sync dual_state_sync_30k_b128 30000
    export INIT_CHECKPOINT="$OPENPI_REPO_ROOT/checkpoints/pi05_libero_async/dual_state_sync_30k_b128/30000"
    # Stage two initializes model weights with a new optimizer. Keep the synchronous
    # model/metadata/stats for evaluation, and reclaim its unnecessary optimizer.
    python - "$INIT_CHECKPOINT" <<'PY'
from pathlib import Path
import sys
import torch
root = Path(sys.argv[1])
metadata = torch.load(root / 'metadata.pt', map_location='cpu', weights_only=False)
assert metadata['global_step'] == 30000
assert (root / 'model.safetensors').is_file()
assert (root / 'assets/physical-intelligence/libero/norm_stats.json').is_file()
(root / 'optimizer.pt').unlink(missing_ok=True)
print('Retained synchronous model/metadata/stats; removed completed-phase optimizer', flush=True)
PY
    run_phase delay dual_state_sync30k_delay10k_b128 10000
fi
printf 'Completed %s route\n' "$route"
