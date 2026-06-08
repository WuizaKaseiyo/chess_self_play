#!/bin/bash
# Submit Chess-R1 hold-out evals for all phases + base anchor.
# Phase X eval depends on Phase X training finishing (use --dependency).
#
# Mapping:
#   Phase 2 (piece-movement skill)  → stage2_fundamental hold-out
#   Phase 3 (tactics skill)          → stage1_mate hold-out
#   Phase 4 (full)                   → stage1_mate + stage2_fundamental (both)

set -euo pipefail

REPO=/home/y50047367/chess_self_play/verl-agent-vam-agent
DATA=$HOME/chess/chess-rl-C224/data
BASE=$HOME/models/Qwen2.5-7B-Instruct
N=${N_SAMPLES:-200}

cd "$REPO"

submit() {
    local name=$1 model=$2 parquet=$3 out=$4 dep=$5
    local dep_flag=""
    [ -n "$dep" ] && dep_flag="--dependency=afterok:$dep"
    local job_id
    job_id=$(sbatch --parsable \
        $dep_flag \
        --export=ALL,MODEL=$model,PARQUET=$parquet,OUT_PATH=$out,N_SAMPLES=$N \
        sbatch_eval_chessr1_holdout.slurm)
    echo "  $name → job $job_id ${dep_flag}"
    echo "$job_id"
}

echo "=== Phase 2 chain ==="
PH2_TRAIN=14564
J_BASE_FUND=$(submit "base on stage2_fund (anchor for phase 2)" "$BASE" \
    "$DATA/chess_puzzles_stage2_fundamental/train_0.parquet" \
    "eval_results/holdout_chessr1_base_stage2_fundamental.json" "")
J_PH2_FUND=$(submit "phase2 on stage2_fund" \
    "$HOME/models/chesslesson_curriculum_stage/phase2" \
    "$DATA/chess_puzzles_stage2_fundamental/train_0.parquet" \
    "eval_results/holdout_chessr1_phase2_stage2_fundamental.json" "$PH2_TRAIN")

echo "=== Phase 3 chain ==="
PH3_TRAIN=14565
J_BASE_MATE=$(submit "base on stage1_mate (anchor for phase 3)" "$BASE" \
    "$DATA/chess_puzzles_stage1_mate/train_0.parquet" \
    "eval_results/holdout_chessr1_base_stage1_mate.json" "")
J_PH3_MATE=$(submit "phase3 on stage1_mate" \
    "$HOME/models/chesslesson_curriculum_stage/phase3" \
    "$DATA/chess_puzzles_stage1_mate/train_0.parquet" \
    "eval_results/holdout_chessr1_phase3_stage1_mate.json" "$PH3_TRAIN")

echo "=== Phase 4 chain (eval on both mate + fundamental) ==="
PH4_TRAIN=14566
J_PH4_MATE=$(submit "phase4 on stage1_mate" \
    "$HOME/models/chesslesson_curriculum_stage/phase4" \
    "$DATA/chess_puzzles_stage1_mate/train_0.parquet" \
    "eval_results/holdout_chessr1_phase4_stage1_mate.json" "$PH4_TRAIN")
J_PH4_FUND=$(submit "phase4 on stage2_fund" \
    "$HOME/models/chesslesson_curriculum_stage/phase4" \
    "$DATA/chess_puzzles_stage2_fundamental/train_0.parquet" \
    "eval_results/holdout_chessr1_phase4_stage2_fundamental.json" "$PH4_TRAIN")

echo ""
echo "=== queue after ==="
squeue -u "$USER" -o "%.10i %.32j %.2t %.10M %R" | head -15
