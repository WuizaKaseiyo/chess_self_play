#!/bin/bash
# Chain 4-phase budget curriculum training (easy → hard).
# Each phase resumes from previous phase's merged HF ckpt.
#
# Usage:
#   export WANDB_API_KEY=...
#   bash submit_chesslesson_curriculum.sh

set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY first}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${REPO_DIR}"

BASE_MODEL="${BASE_MODEL:-$HOME/models/Qwen2.5-7B-Instruct}"
MERGED_OUT_DIR="${MERGED_OUT_DIR:-$HOME/models/chesslesson_curriculum}"

# Phase config: (PHASE, MAX_BUDGET, WALLTIME, EPOCHS)
# Phase 1 (108 task, ≤1-move): shortest rollout (1 turn) → fastest
# Phase 4 (170 task, all):     longest rollout (up to 8 turn) → slowest
declare -A WALLTIME=(
    [1]="03:00:00"
    [2]="04:00:00"
    [3]="06:00:00"
    [4]="06:00:00"
)
declare -A MAX_BUDGET=(
    [1]="1"
    [2]="2"
    [3]="4"
    [4]="null"
)
declare -A EPOCHS=(
    [1]="10"
    [2]="10"
    [3]="10"
    [4]="10"
)

prev_job=""
prev_merged="${BASE_MODEL}"

for phase in 1 2 3 4; do
    # Resume from previous phase's merged ckpt, or base model for phase 1
    model_path="${prev_merged}"

    # Build sbatch command
    export_args="ALL,WANDB_API_KEY=${WANDB_API_KEY}"
    export_args="${export_args},PHASE=${phase}"
    export_args="${export_args},MAX_BUDGET=${MAX_BUDGET[$phase]}"
    export_args="${export_args},MODEL_PATH=${model_path}"
    export_args="${export_args},TOTAL_EPOCHS=${EPOCHS[$phase]}"
    export_args="${export_args},MERGED_OUT_DIR=${MERGED_OUT_DIR}"

    dep_flag=""
    if [ -n "${prev_job}" ]; then
        dep_flag="--dependency=afterok:${prev_job}"
    fi

    echo "=== Phase ${phase}: max_budget=${MAX_BUDGET[$phase]}, model_path=${model_path} ==="
    job_id=$(sbatch --parsable \
        --time="${WALLTIME[$phase]}" \
        ${dep_flag} \
        --export="${export_args}" \
        sbatch_chesslesson_curriculum_phase.slurm)
    echo "  → submitted job ${job_id} ${dep_flag}"

    prev_job="${job_id}"
    prev_merged="${MERGED_OUT_DIR}/phase${phase}"
done

echo ""
echo "=== Chain submitted ==="
squeue -u "$USER" -o "%.10i %.20j %.2t %.10M %R" | head
echo ""
echo "Final merged ckpt will be at: ${MERGED_OUT_DIR}/phase4"
