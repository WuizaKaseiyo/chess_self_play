#!/usr/bin/env bash

set -euo pipefail

# Core comparison launcher:
# - baseline (Pass@k + no diversity)
# - baseline + OBE-Batch
# - baseline + GAPO
# - baseline + Distinct@k analytic
#
# Fairness constraints:
# - Fixed baseline envelope copied from launch_baseline_passk_qwen3b.sh
# - CHESS_ALLOWED_MOVE_ELIM_ENABLE is hard-disabled
# - one fixed seed across all variants
# - only diversity-advantage knobs change across variants
#
# Usage:
#   ./launch_diversity_core_comparison_qwen3b.sh
#   VARIANT=baseline ./launch_diversity_core_comparison_qwen3b.sh
#   VARIANT=obe ./launch_diversity_core_comparison_qwen3b.sh
#   SMOKE=1 ./launch_diversity_core_comparison_qwen3b.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VARIANT="${VARIANT:-all}"   # baseline|obe|gapo|distinct|all
WAIT="${WAIT:-1}"
SMOKE="${SMOKE:-0}"

# Fixed one-seed setup for fair single-seed comparison.
CHESS_RL_DATA_SEED="${CHESS_RL_DATA_SEED:-3407}"

# Baseline envelope (kept aligned with launch_baseline_passk_qwen3b.sh).
MODEL_PATH="${MODEL_PATH:-/projects/a5l/ziyan/models/Qwen/Qwen2.5-3B-Instruct}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2000}"
ROLLOUT_N="${ROLLOUT_N:-16}"

CHESS_DATA_DIR="${CHESS_DATA_DIR:-data/chess_puzzles_chessr1_aligned_sharded_baseline}"
CHESS_RL_VAL_FILES="${CHESS_RL_VAL_FILES:-[data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet,data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet]}"
CHESS_REWARD_FN="${CHESS_REWARD_FN:-expected_score_wdl_vs_best}"

# Keep Pass@k baseline path constant across variants.
CHESS_PASS_K_TRAINING="${CHESS_PASS_K_TRAINING:-True}"
CHESS_PASS_K="${CHESS_PASS_K:-4}"

# Diversity defaults for enabled variants.
CHESS_DIVERSITY_LAMBDA="${CHESS_DIVERSITY_LAMBDA:-0.25}"
CHESS_DIVERSITY_DISTINCT_K="${CHESS_DIVERSITY_DISTINCT_K:-4}"
CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE="${CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE:-True}"

# Optional HF upload controls.
CHESS_RL_HF_UPLOAD_ENABLE="${CHESS_RL_HF_UPLOAD_ENABLE:-False}"
CHESS_RL_HF_CKPT_REPO_ID="${CHESS_RL_HF_CKPT_REPO_ID:-Gabr1e11/a_lot_of_models}"
CHESS_RL_HF_TOKEN_FILE="${CHESS_RL_HF_TOKEN_FILE:-${HOME}/.hf_token}"

if [ "${SMOKE}" = "1" ]; then
  TOTAL_TRAINING_STEPS="1"
  VAL_BEFORE_TRAIN="False"
  TRAINER_TEST_FREQ="-1"
  CHESS_RL_TRAINER_SAVE_FREQ="-1"
  CHESS_RL_FULL_EVAL_FREQ="-1"
  RUN_KIND="smoke"
else
  TOTAL_TRAINING_STEPS="null"
  VAL_BEFORE_TRAIN="True"
  TRAINER_TEST_FREQ="40"
  CHESS_RL_TRAINER_SAVE_FREQ="80"
  CHESS_RL_FULL_EVAL_FREQ="80"
  RUN_KIND="full"
fi

SBATCH_WAIT_ARGS=()
if [ "${WAIT}" = "1" ]; then
  SBATCH_WAIT_ARGS+=(--wait)
fi

submit_one() {
  local label="$1"
  local diversity_enable="$2"
  local diversity_method="$3"
  local diversity_lambda="$4"
  local diversity_distinct_k="$5"
  local include_base="$6"

  local run_tag="cr1_diversity_core_${label}_${RUN_KIND}_$(date +%Y%m%d_%H%M%S)"
  # Keep "baseline" token in job names so train_chess.sh applies baseline prompt guardrails
  # for the baseline Chess-R1 dataset used in this core comparison.
  local job_name="cr1-baseline-div-core-${label}-${RUN_KIND}"

  echo "[INFO] submit ${label}"
  echo "[INFO]   RUN_TAG=${run_tag}"
  echo "[INFO]   JOB_NAME=${job_name}"
  echo "[INFO]   DATA_SEED=${CHESS_RL_DATA_SEED}"
  echo "[INFO]   DIVERSITY_ENABLE=${diversity_enable}"
  echo "[INFO]   DIVERSITY_METHOD=${diversity_method}"
  echo "[INFO]   DIVERSITY_LAMBDA=${diversity_lambda}"
  echo "[INFO]   DIVERSITY_DISTINCT_K=${diversity_distinct_k}"
  echo "[INFO]   DIVERSITY_INCLUDE_BASE_ADVANTAGE=${include_base}"

  CHESS_RL_VAL_FILES="${CHESS_RL_VAL_FILES}" \
  sbatch "${SBATCH_WAIT_ARGS[@]}" \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=144 \
    --partition=workq --account=brics.a5l --job-name="${job_name}" \
    --export=ALL,CHESS_RL_VERL_BASE_DIR=/projects/a5l/ziyan/chess_rl_outputs/${run_tag},CHESS_DATA_DIR=${CHESS_DATA_DIR},CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH=recipe/chess/prompt_templates/original_chessr1_prompt.jinja,USE_HARD_DATASET=False,CHESS_REWARD_FN=${CHESS_REWARD_FN},CHESS_RL_TRAIN_BATCH_SIZE=128,TOTAL_EPOCHS=1,CHESS_ALLOWED_MOVE_ELIM_ENABLE=False,FILTER_GROUPS_ENABLE=False,CHESS_RL_TENSOR_MODEL_PARALLEL_SIZE=1,CHESS_RL_PPO_MICRO_BATCH_SIZE=8,CHESS_RL_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE=16,CHESS_RL_REF_LOGPROB_MICRO_BATCH_SIZE=16,TRAINER_TEST_FREQ=${TRAINER_TEST_FREQ},MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH},MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH},ROLLOUT_N=${ROLLOUT_N},MODEL_PATH=${MODEL_PATH},CHESS_RL_WANDB_MODE=online,CHESS_RL_TRAINER_SAVE_FREQ=${CHESS_RL_TRAINER_SAVE_FREQ},CHESS_RL_FULL_EVAL_FREQ=${CHESS_RL_FULL_EVAL_FREQ},CHESS_RL_USE_KL_LOSS=True,CHESS_PASS_K_TRAINING=${CHESS_PASS_K_TRAINING},CHESS_PASS_K=${CHESS_PASS_K},CHESS_DIVERSITY_ENABLE=${diversity_enable},CHESS_DIVERSITY_METHOD=${diversity_method},CHESS_DIVERSITY_LAMBDA=${diversity_lambda},CHESS_DIVERSITY_DISTINCT_K=${diversity_distinct_k},CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE=${include_base},CHESS_RL_DATA_SEED=${CHESS_RL_DATA_SEED},CHESS_RL_TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS},VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN},FORCED_PREFIX_ENABLE=False,CHESS_RL_MODEL_ONLY_CKPT=True,CHESS_RL_HF_UPLOAD_ENABLE=${CHESS_RL_HF_UPLOAD_ENABLE},CHESS_RL_HF_CKPT_REPO_ID=${CHESS_RL_HF_CKPT_REPO_ID},CHESS_RL_HF_TOKEN_FILE=${CHESS_RL_HF_TOKEN_FILE} \
    ./sbatch_train_chess_gh200.slurm
}

if [ "${VARIANT}" = "baseline" ]; then
  submit_one baseline False none 0.0 "${CHESS_DIVERSITY_DISTINCT_K}" True
  exit 0
fi

if [ "${VARIANT}" = "obe" ]; then
  submit_one obe True obe_batch "${CHESS_DIVERSITY_LAMBDA}" "${CHESS_DIVERSITY_DISTINCT_K}" "${CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE}"
  exit 0
fi

if [ "${VARIANT}" = "gapo" ]; then
  submit_one gapo True gapo "${CHESS_DIVERSITY_LAMBDA}" "${CHESS_DIVERSITY_DISTINCT_K}" "${CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE}"
  exit 0
fi

if [ "${VARIANT}" = "distinct" ]; then
  submit_one distinct True distinct_k "${CHESS_DIVERSITY_LAMBDA}" "${CHESS_DIVERSITY_DISTINCT_K}" "${CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE}"
  exit 0
fi

if [ "${VARIANT}" = "all" ]; then
  submit_one baseline False none 0.0 "${CHESS_DIVERSITY_DISTINCT_K}" True
  submit_one obe True obe_batch "${CHESS_DIVERSITY_LAMBDA}" "${CHESS_DIVERSITY_DISTINCT_K}" "${CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE}"
  submit_one gapo True gapo "${CHESS_DIVERSITY_LAMBDA}" "${CHESS_DIVERSITY_DISTINCT_K}" "${CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE}"
  submit_one distinct True distinct_k "${CHESS_DIVERSITY_LAMBDA}" "${CHESS_DIVERSITY_DISTINCT_K}" "${CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE}"
  exit 0
fi

echo "Unknown VARIANT='${VARIANT}'. Expected baseline|obe|gapo|distinct|all." >&2
exit 1
