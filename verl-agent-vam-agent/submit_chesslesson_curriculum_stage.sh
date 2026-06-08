#!/bin/bash
# Stage-based curriculum (easy ability → hard ability), each phase teaches a specific skill.
#   Phase 1 (60 task):   coord-only        — learn the board / coord system / format
#   Phase 2 (94 task):   + 6 piece stages  — learn each piece's movement rules
#   Phase 3 (133 task):  + tactics stages  — learn capture / check / mate
#   Phase 4 (170 task):  + advanced rules  — castling / enpassant / stalemate / value / check2 / fork
#
# Each phase resumes from the previous phase's merged HF checkpoint.

set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY first}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${REPO_DIR}"

BASE_MODEL="${BASE_MODEL:-$HOME/models/Qwen2.5-7B-Instruct}"
MERGED_OUT_DIR="${MERGED_OUT_DIR:-$HOME/models/chesslesson_curriculum_stage}"

P2_STAGES="rook,bishop,queen,king,knight,pawn"
P3_STAGES="${P2_STAGES},capture,protection,combat,check1,outOfCheck,checkmate1"
# P4 = no filter = all stages

declare -A WALLTIME=([1]="03:30:00" [2]="04:30:00" [3]="06:30:00" [4]="07:30:00")
declare -A EPOCHS=([1]="10" [2]="10" [3]="10" [4]="10")

# Phase config: (coord_only, allowed_stages)
declare -A COORD_ONLY=([1]="True" [2]="False" [3]="False" [4]="False")
declare -A STAGES=([1]="null" [2]="${P2_STAGES}" [3]="${P3_STAGES}" [4]="null")

prev_job=""
prev_merged="${BASE_MODEL}"

for phase in 1 2 3 4; do
    model_path="${prev_merged}"

    export_args="ALL,WANDB_API_KEY=${WANDB_API_KEY}"
    export_args="${export_args},PHASE=${phase}"
    export_args="${export_args},MAX_BUDGET=null"
    export_args="${export_args},ALLOWED_STAGES=${STAGES[$phase]}"
    export_args="${export_args},COORD_ONLY=${COORD_ONLY[$phase]}"
    export_args="${export_args},MODEL_PATH=${model_path}"
    export_args="${export_args},TOTAL_EPOCHS=${EPOCHS[$phase]}"
    export_args="${export_args},MERGED_OUT_DIR=${MERGED_OUT_DIR}"
    export_args="${export_args},WANDB_PROJECT=chess_self_play_chesslesson_curriculum_stage"

    dep_flag=""
    if [ -n "${prev_job}" ]; then
        dep_flag="--dependency=afterok:${prev_job}"
    fi

    echo "=== Phase ${phase}: coord_only=${COORD_ONLY[$phase]}  stages=${STAGES[$phase]:0:60}... ==="
    echo "    model=${model_path}"
    job_id=$(sbatch --parsable \
        --time="${WALLTIME[$phase]}" \
        ${dep_flag} \
        --export="${export_args}" \
        sbatch_chesslesson_curriculum_phase.slurm)
    echo "  → job ${job_id} ${dep_flag}"

    prev_job="${job_id}"
    prev_merged="${MERGED_OUT_DIR}/phase${phase}"
done

echo ""
echo "=== Chain submitted ==="
squeue -u "$USER" -o "%.10i %.30j %.2t %.10M %R" | head
echo ""
echo "Final merged ckpt at: ${MERGED_OUT_DIR}/phase4"
