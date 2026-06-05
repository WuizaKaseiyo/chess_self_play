#!/usr/bin/env bash

set -euo pipefail

# Smoke launcher for diversity-auxiliary GRPO variants on GH200.
# Runs a minimal 1-step, no-validation/no-eval training job per requested variant.
#
# Usage:
#   ./launch_diversity_smoke_qwen3b.sh
#   VARIANT=obe ./launch_diversity_smoke_qwen3b.sh
#   VARIANT=gapo ./launch_diversity_smoke_qwen3b.sh
#   VARIANT=distinct ./launch_diversity_smoke_qwen3b.sh
#   VARIANT=all RUN_BASELINE_CONTROL=1 ./launch_diversity_smoke_qwen3b.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VARIANT="${VARIANT:-all}"  # baseline|obe|gapo|distinct|all
RUN_BASELINE_CONTROL="${RUN_BASELINE_CONTROL:-1}"

MODEL_PATH="${MODEL_PATH:-/projects/a5l/ziyan/models/Qwen/Qwen2.5-3B-Instruct}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2000}"
ROLLOUT_N="${ROLLOUT_N:-16}"

CHESS_DATA_DIR="${CHESS_DATA_DIR:-data/chess_puzzles_chessr1_aligned_sharded_ours}"
CHESS_REWARD_FN="${CHESS_REWARD_FN:-expected_score_wdl_vs_best}"

CHESS_ALLOWED_MOVE_ELIM_ENABLE="${CHESS_ALLOWED_MOVE_ELIM_ENABLE:-False}"
CHESS_ALLOWED_MOVE_ELIM_UID_MODE="${CHESS_ALLOWED_MOVE_ELIM_UID_MODE:-per_round}"
CHESS_ALLOWED_MOVE_ELIM_R_MAX_START="${CHESS_ALLOWED_MOVE_ELIM_R_MAX_START:-2}"
CHESS_ALLOWED_MOVE_ELIM_R_MAX_END="${CHESS_ALLOWED_MOVE_ELIM_R_MAX_END:-2}"
CHESS_ALLOWED_MOVE_ELIM_ANNEAL_FRAC="${CHESS_ALLOWED_MOVE_ELIM_ANNEAL_FRAC:-1.0}"

CHESS_DIVERSITY_LAMBDA="${CHESS_DIVERSITY_LAMBDA:-0.25}"
CHESS_DIVERSITY_DISTINCT_K="${CHESS_DIVERSITY_DISTINCT_K:-4}"

CHESS_RL_HF_UPLOAD_ENABLE="${CHESS_RL_HF_UPLOAD_ENABLE:-False}"
CHESS_RL_HF_CKPT_REPO_ID="${CHESS_RL_HF_CKPT_REPO_ID:-Gabr1e11/a_lot_of_models}"
CHESS_RL_HF_TOKEN_FILE="${CHESS_RL_HF_TOKEN_FILE:-${HOME}/.hf_token}"

SMOKE_TOTAL_STEPS=1
SMOKE_VAL_BEFORE_TRAIN=False
SMOKE_TEST_FREQ=-1
SMOKE_FULL_EVAL_FREQ=-1
SMOKE_SAVE_FREQ=-1

submit_one() {
  local label="$1"
  local diversity_enable="$2"
  local diversity_method="$3"
  local diversity_lambda="$4"
  local diversity_distinct_k="$5"
  local include_base="$6"

  local run_tag
  run_tag="cr1_diversity_${label}_smoke_$(date +%Y%m%d_%H%M%S)"
  local job_name
  if [ "${label}" = "baseline" ]; then
    # Avoid triggering train_chess.sh baseline-dataset name guard for control runs.
    job_name="cr1-div-control-smoke"
  else
    job_name="cr1-div-${label}-smoke"
  fi

  echo "[INFO] submitting ${label}: RUN_TAG=${run_tag} JOB_NAME=${job_name}"

  CHESS_RL_VAL_FILES='[data/chess_puzzles_chessr1_aligned_sharded_ours/test.parquet,data/chess_puzzles_chessr1_aligned_sharded_ours/test_shuffled_legal_moves.parquet]' \
  sbatch --wait \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=144 \
    --partition=workq --account=brics.a5l --job-name="${job_name}" \
    --export=ALL,CHESS_RL_VERL_BASE_DIR=/projects/a5l/ziyan/chess_rl_outputs/${run_tag},CHESS_DATA_DIR=${CHESS_DATA_DIR},CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH=recipe/chess/prompt_templates/select_prompt.jinja,USE_HARD_DATASET=False,CHESS_REWARD_FN=${CHESS_REWARD_FN},CHESS_RL_TRAIN_BATCH_SIZE=128,TOTAL_EPOCHS=1,CHESS_ALLOWED_MOVE_ELIM_ENABLE=${CHESS_ALLOWED_MOVE_ELIM_ENABLE},CHESS_ALLOWED_MOVE_ELIM_UID_MODE=${CHESS_ALLOWED_MOVE_ELIM_UID_MODE},CHESS_ALLOWED_MOVE_ELIM_R_MAX_START=${CHESS_ALLOWED_MOVE_ELIM_R_MAX_START},CHESS_ALLOWED_MOVE_ELIM_R_MAX_END=${CHESS_ALLOWED_MOVE_ELIM_R_MAX_END},CHESS_ALLOWED_MOVE_ELIM_ANNEAL_FRAC=${CHESS_ALLOWED_MOVE_ELIM_ANNEAL_FRAC},FILTER_GROUPS_ENABLE=False,CHESS_RL_TENSOR_MODEL_PARALLEL_SIZE=1,CHESS_RL_PPO_MICRO_BATCH_SIZE=8,CHESS_RL_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE=16,CHESS_RL_REF_LOGPROB_MICRO_BATCH_SIZE=16,TRAINER_TEST_FREQ=${SMOKE_TEST_FREQ},MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH},MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH},ROLLOUT_N=${ROLLOUT_N},MODEL_PATH=${MODEL_PATH},CHESS_RL_WANDB_MODE=online,CHESS_RL_TRAINER_SAVE_FREQ=${SMOKE_SAVE_FREQ},CHESS_RL_FULL_EVAL_FREQ=${SMOKE_FULL_EVAL_FREQ},CHESS_RL_USE_KL_LOSS=True,CHESS_PASS_K_TRAINING=False,CHESS_PASS_K=1,CHESS_DIVERSITY_ENABLE=${diversity_enable},CHESS_DIVERSITY_METHOD=${diversity_method},CHESS_DIVERSITY_LAMBDA=${diversity_lambda},CHESS_DIVERSITY_DISTINCT_K=${diversity_distinct_k},CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE=${include_base},CHESS_RL_TOTAL_TRAINING_STEPS=${SMOKE_TOTAL_STEPS},VAL_BEFORE_TRAIN=${SMOKE_VAL_BEFORE_TRAIN},FORCED_PREFIX_ENABLE=False,CHESS_RL_MODEL_ONLY_CKPT=True,CHESS_RL_HF_UPLOAD_ENABLE=${CHESS_RL_HF_UPLOAD_ENABLE},CHESS_RL_HF_CKPT_REPO_ID=${CHESS_RL_HF_CKPT_REPO_ID},CHESS_RL_HF_TOKEN_FILE=${CHESS_RL_HF_TOKEN_FILE} \
    ./sbatch_train_chess_gh200.slurm
}

if [ "${VARIANT}" = "baseline" ]; then
  submit_one baseline False none 0.0 "${CHESS_DIVERSITY_DISTINCT_K}" True
  exit 0
fi

if [ "${VARIANT}" = "obe" ]; then
  submit_one obe True obe_batch "${CHESS_DIVERSITY_LAMBDA}" "${CHESS_DIVERSITY_DISTINCT_K}" True
  exit 0
fi

if [ "${VARIANT}" = "gapo" ]; then
  submit_one gapo True gapo "${CHESS_DIVERSITY_LAMBDA}" "${CHESS_DIVERSITY_DISTINCT_K}" True
  exit 0
fi

if [ "${VARIANT}" = "distinct" ]; then
  submit_one distinct True distinct_k "${CHESS_DIVERSITY_LAMBDA}" "${CHESS_DIVERSITY_DISTINCT_K}" True
  exit 0
fi

if [ "${VARIANT}" = "all" ]; then
  if [ "${RUN_BASELINE_CONTROL}" = "1" ]; then
    submit_one baseline False none 0.0 "${CHESS_DIVERSITY_DISTINCT_K}" True
  fi
  submit_one obe True obe_batch "${CHESS_DIVERSITY_LAMBDA}" "${CHESS_DIVERSITY_DISTINCT_K}" True
  submit_one gapo True gapo "${CHESS_DIVERSITY_LAMBDA}" "${CHESS_DIVERSITY_DISTINCT_K}" True
  submit_one distinct True distinct_k "${CHESS_DIVERSITY_LAMBDA}" "${CHESS_DIVERSITY_DISTINCT_K}" True
  exit 0
fi

echo "Unknown VARIANT='${VARIANT}'. Expected baseline|obe|gapo|distinct|all." >&2
exit 1
