#!/usr/bin/env bash

set -euo pipefail

# Baseline-aligned launcher for Qwen2.5-3B with optional Pass@k.
# Source of truth for baseline envelope is launch_baselines.sh / run 82fpo6l0.
#
# Default mode is a quick alignment smoke:
# - total_training_steps=1
# - val_before_train=False
# Full mode keeps baseline values:
# - total_training_steps=null
# - val_before_train=True
#
# Usage:
#   ./launch_baseline_passk_qwen3b.sh
#   SMOKE=0 ./launch_baseline_passk_qwen3b.sh
#   WAIT=0 ./launch_baseline_passk_qwen3b.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SMOKE="${SMOKE:-1}"
WAIT="${WAIT:-1}"

# Pass@k controls (default to requested baseline variant).
CHESS_PASS_K_TRAINING="${CHESS_PASS_K_TRAINING:-True}"
CHESS_PASS_K="${CHESS_PASS_K:-4}"

# Diversity auxiliary advantage controls (default disabled for baseline parity).
CHESS_DIVERSITY_ENABLE="${CHESS_DIVERSITY_ENABLE:-False}"
CHESS_DIVERSITY_METHOD="${CHESS_DIVERSITY_METHOD:-none}"
CHESS_DIVERSITY_LAMBDA="${CHESS_DIVERSITY_LAMBDA:-0.0}"
CHESS_DIVERSITY_DISTINCT_K="${CHESS_DIVERSITY_DISTINCT_K:-4}"
CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE="${CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE:-True}"

# HF upload controls (off for smoke by default; opt-in for full runs).
CHESS_RL_HF_UPLOAD_ENABLE="${CHESS_RL_HF_UPLOAD_ENABLE:-False}"
CHESS_RL_HF_CKPT_REPO_ID="${CHESS_RL_HF_CKPT_REPO_ID:-Gabr1e11/a_lot_of_models}"
CHESS_RL_HF_TOKEN_FILE="${CHESS_RL_HF_TOKEN_FILE:-${HOME}/.hf_token}"

# Baseline model + generation envelope from 82fpo6l0.
MODEL_PATH="${MODEL_PATH:-/projects/a5l/ziyan/models/Qwen/Qwen2.5-3B-Instruct}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2000}"
ROLLOUT_N="${ROLLOUT_N:-16}"  # intentionally differs from 82fpo6l0 n=8
# One-seed reproducibility control (dataloader seed).
CHESS_RL_DATA_SEED="${CHESS_RL_DATA_SEED:-null}"

if [ "${SMOKE}" = "1" ]; then
  TOTAL_TRAINING_STEPS="1"
  VAL_BEFORE_TRAIN="False"
  RUN_KIND="smoke"
else
  TOTAL_TRAINING_STEPS="null"
  VAL_BEFORE_TRAIN="True"
  RUN_KIND="full"
fi

SBATCH_WAIT_ARGS=()
if [ "${WAIT}" = "1" ]; then
  SBATCH_WAIT_ARGS+=(--wait)
fi

RUN_TAG="cr1_passk_qwen3b_${RUN_KIND}_$(date +%Y%m%d_%H%M%S)"
JOB_NAME="cr1-baseline-passk-qwen3b-${RUN_KIND}"

echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] JOB_NAME=${JOB_NAME}"
echo "[INFO] SMOKE=${SMOKE} WAIT=${WAIT}"
echo "[INFO] PASS_K_TRAINING=${CHESS_PASS_K_TRAINING} PASS_K=${CHESS_PASS_K}"
echo "[INFO] DIVERSITY_ENABLE=${CHESS_DIVERSITY_ENABLE} METHOD=${CHESS_DIVERSITY_METHOD} LAMBDA=${CHESS_DIVERSITY_LAMBDA}"
echo "[INFO] CHESS_RL_DATA_SEED=${CHESS_RL_DATA_SEED}"

CHESS_RL_VAL_FILES='[data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet,data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet]' \
sbatch "${SBATCH_WAIT_ARGS[@]}" \
  --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=144 \
  --partition=workq --account=brics.a5l --job-name="${JOB_NAME}" \
  --export=ALL,CHESS_RL_VERL_BASE_DIR=/projects/a5l/ziyan/chess_rl_outputs/${RUN_TAG},CHESS_DATA_DIR=data/chess_puzzles_chessr1_aligned_sharded_baseline,CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH=recipe/chess/prompt_templates/original_chessr1_prompt.jinja,USE_HARD_DATASET=False,CHESS_REWARD_FN=expected_score_wdl_vs_best,CHESS_RL_TRAIN_BATCH_SIZE=128,TOTAL_EPOCHS=1,CHESS_ALLOWED_MOVE_ELIM_ENABLE=False,FILTER_GROUPS_ENABLE=False,CHESS_RL_TENSOR_MODEL_PARALLEL_SIZE=1,CHESS_RL_PPO_MICRO_BATCH_SIZE=8,CHESS_RL_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE=16,CHESS_RL_REF_LOGPROB_MICRO_BATCH_SIZE=16,TRAINER_TEST_FREQ=40,MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH},MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH},ROLLOUT_N=${ROLLOUT_N},MODEL_PATH=${MODEL_PATH},CHESS_RL_WANDB_MODE=online,CHESS_RL_TRAINER_SAVE_FREQ=80,CHESS_RL_FULL_EVAL_FREQ=80,CHESS_RL_USE_KL_LOSS=True,CHESS_PASS_K_TRAINING=${CHESS_PASS_K_TRAINING},CHESS_PASS_K=${CHESS_PASS_K},CHESS_DIVERSITY_ENABLE=${CHESS_DIVERSITY_ENABLE},CHESS_DIVERSITY_METHOD=${CHESS_DIVERSITY_METHOD},CHESS_DIVERSITY_LAMBDA=${CHESS_DIVERSITY_LAMBDA},CHESS_DIVERSITY_DISTINCT_K=${CHESS_DIVERSITY_DISTINCT_K},CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE=${CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE},CHESS_RL_DATA_SEED=${CHESS_RL_DATA_SEED},CHESS_RL_TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS},VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN},FORCED_PREFIX_ENABLE=False,CHESS_RL_MODEL_ONLY_CKPT=True,CHESS_RL_HF_UPLOAD_ENABLE=${CHESS_RL_HF_UPLOAD_ENABLE},CHESS_RL_HF_CKPT_REPO_ID=${CHESS_RL_HF_CKPT_REPO_ID},CHESS_RL_HF_TOKEN_FILE=${CHESS_RL_HF_TOKEN_FILE} \
  ./sbatch_train_chess_gh200.slurm
