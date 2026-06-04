#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SBATCH_SCRIPT="${SBATCH_SCRIPT:-./sbatch_train_chess_gh200_passk.slurm}"

# Run control:
# - RUN_MODE=smoke: one 1-step smoke submission for n=8,r=4 with VAL_BEFORE_TRAIN=True.
# - RUN_MODE=full: submit full runs for requested setting(s).
RUN_MODE="${RUN_MODE:-full}"                  # smoke | full
SETTINGS="${SETTINGS:-both}"                  # both | n8_r4 | n2_r16

# Cluster submission defaults (override via env as needed).
NODES="${NODES:-4}"
NTASKS="${NTASKS:-4}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-1}"
GRES="${GRES:-gpu:4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-144}"
PARTITION="${PARTITION:-workq}"
ACCOUNT="${ACCOUNT:-brics.a5l}"

# Ours-method wiring.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
CHESS_DATA_DIR="${CHESS_DATA_DIR:-data/chess_puzzles_chessr1_aligned_sharded_ours_small_legal}"
# Default eval includes both canonical and shuffled-legal splits.
# This is passed via command environment (not embedded in --export kvs), so
# commas in the Hydra list are preserved safely.
VAL_FILES_DEFAULT="[${CHESS_DATA_DIR}/test.parquet,${CHESS_DATA_DIR}/test_shuffled_legal_moves.parquet]"
CHESS_RL_VAL_FILES="${CHESS_RL_VAL_FILES:-${VAL_FILES_DEFAULT}}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-recipe/chess/prompt_templates/select_prompt_small_legal.jinja}"
REWARD_FN="${REWARD_FN:-expected_score_wdl_vs_best}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

# Vanilla-GRPO controls (no pass@k/diversity optimization behavior).
PASS_K_TRAINING="${PASS_K_TRAINING:-False}"
ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL="${ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL:-False}"
ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL="${ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL:-False}"
DIVERSITY_ENABLE="${DIVERSITY_ENABLE:-False}"
DIVERSITY_METHOD="${DIVERSITY_METHOD:-none}"
DIVERSITY_LAMBDA="${DIVERSITY_LAMBDA:-0.0}"
DIVERSITY_INCLUDE_BASE_ADVANTAGE="${DIVERSITY_INCLUDE_BASE_ADVANTAGE:-True}"

# Keep checkpoint/full-eval lightweight by default; override in full runs as needed.
TRAINER_SAVE_FREQ_DEFAULT="${TRAINER_SAVE_FREQ_DEFAULT:--1}"
FULL_EVAL_FREQ_DEFAULT="${FULL_EVAL_FREQ_DEFAULT:--1}"

# Optional Hugging Face upload pass-through (same convention as prior launcher).
CHESS_RL_HF_UPLOAD_ENABLE="${CHESS_RL_HF_UPLOAD_ENABLE:-False}"
CHESS_RL_HF_CKPT_REPO_ID="${CHESS_RL_HF_CKPT_REPO_ID:-}"
CHESS_RL_HF_TOKEN_FILE="${CHESS_RL_HF_TOKEN_FILE:-${HOME}/.hf_token}"

submit_run() {
  local setting_name="$1"
  local rollout_n="$2"
  local r_max="$3"
  local total_steps="$4"
  local trainer_save_freq="$5"
  local full_eval_freq="$6"
  local job_suffix="$7"

  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)"
  local out_root="/projects/a5l/ziyan/chess_rl_outputs/ours_smalllegal_${setting_name}_${job_suffix}_${run_id}"
  local job_name="ours-${setting_name}-${job_suffix}"

  echo "[INFO] Submitting ${job_name}"
  echo "[INFO] MODEL_PATH=${MODEL_PATH}"
  echo "[INFO] CHESS_DATA_DIR=${CHESS_DATA_DIR}"
  echo "[INFO] CHESS_RL_VAL_FILES=${CHESS_RL_VAL_FILES}"
  echo "[INFO] PROMPT_TEMPLATE=${PROMPT_TEMPLATE}"
  echo "[INFO] ROLLOUT_N=${rollout_n} R_MAX=${r_max} TOTAL_STEPS=${total_steps}"

  CHESS_RL_VAL_FILES="${CHESS_RL_VAL_FILES}" sbatch --wait \
    --nodes="${NODES}" --ntasks="${NTASKS}" --ntasks-per-node="${NTASKS_PER_NODE}" \
    --gres="${GRES}" --cpus-per-task="${CPUS_PER_TASK}" \
    --partition="${PARTITION}" --account="${ACCOUNT}" \
    --job-name="${job_name}" \
    --export=ALL,\
CHESS_RL_VERL_BASE_DIR="${out_root}",\
CHESS_RL_TOTAL_TRAINING_STEPS="${total_steps}",\
VAL_BEFORE_TRAIN=True,\
CHESS_RL_TRAINER_SAVE_FREQ="${trainer_save_freq}",\
CHESS_RL_FULL_EVAL_FREQ="${full_eval_freq}",\
CHESS_DATA_DIR="${CHESS_DATA_DIR}",\
CHESS_SELF_PLAY_ENABLE=False,\
CHESS_ALLOWED_MOVE_ELIM_ENABLE=True,\
CHESS_ALLOWED_MOVE_ELIM_TEMPLATE_PATH="${PROMPT_TEMPLATE}",\
CHESS_ALLOWED_MOVE_ELIM_UID_MODE=per_prompt,\
CHESS_ALLOWED_MOVE_ELIM_R_MAX_START="${r_max}",\
CHESS_ALLOWED_MOVE_ELIM_R_MAX_END="${r_max}",\
CHESS_ALLOWED_MOVE_ELIM_ANNEAL_FRAC=1.0,\
CHESS_ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB=True,\
CHESS_ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI=True,\
CHESS_ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN=0.0,\
CHESS_ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_DUMP_MAX_GROUPS=-1,\
CHESS_ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL="${ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL}",\
CHESS_ALLOWED_MOVE_ELIM_PASS_K_K=4,\
CHESS_ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL="${ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL}",\
CHESS_PASS_K_TRAINING="${PASS_K_TRAINING}",\
CHESS_DIVERSITY_ENABLE="${DIVERSITY_ENABLE}",\
CHESS_DIVERSITY_METHOD="${DIVERSITY_METHOD}",\
CHESS_DIVERSITY_LAMBDA="${DIVERSITY_LAMBDA}",\
CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE="${DIVERSITY_INCLUDE_BASE_ADVANTAGE}",\
CHESS_REWARD_FN="${REWARD_FN}",\
CHESS_RL_TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}",\
TOTAL_EPOCHS="${TOTAL_EPOCHS}",\
TRAINER_TEST_FREQ=40,\
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}",\
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}",\
ROLLOUT_N="${rollout_n}",\
MODEL_PATH="${MODEL_PATH}",\
CHESS_RL_WANDB_MODE=online,\
CHESS_RL_USE_KL_LOSS=True,\
CHESS_RL_TRAINER_RESUME_MODE=disable,\
CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE}",\
CHESS_RL_PPO_MICRO_BATCH_SIZE=8,\
CHESS_RL_REF_LOGPROB_MICRO_BATCH_SIZE=16,\
CHESS_RL_HF_UPLOAD_ENABLE="${CHESS_RL_HF_UPLOAD_ENABLE}",\
CHESS_RL_HF_CKPT_REPO_ID="${CHESS_RL_HF_CKPT_REPO_ID}",\
CHESS_RL_HF_TOKEN_FILE="${CHESS_RL_HF_TOKEN_FILE}" \
    "${SBATCH_SCRIPT}"
}

if [[ "${RUN_MODE}" != "smoke" && "${RUN_MODE}" != "full" ]]; then
  echo "[ERROR] RUN_MODE must be 'smoke' or 'full' (got '${RUN_MODE}')." >&2
  exit 1
fi

if [[ "${SETTINGS}" != "both" && "${SETTINGS}" != "n8_r4" && "${SETTINGS}" != "n2_r16" ]]; then
  echo "[ERROR] SETTINGS must be one of: both, n8_r4, n2_r16 (got '${SETTINGS}')." >&2
  exit 1
fi

if [[ "${RUN_MODE}" == "smoke" ]]; then
  # Single smoke run for wiring validation.
  submit_run "n8_r4" "8" "4" "1" "-1" "-1" "smoke"
  exit 0
fi

# Full runs.
TOTAL_STEPS_N8_R4="${TOTAL_STEPS_N8_R4:-800}"
TOTAL_STEPS_N2_R16="${TOTAL_STEPS_N2_R16:-800}"
TRAINER_SAVE_FREQ_N8_R4="${TRAINER_SAVE_FREQ_N8_R4:-${TRAINER_SAVE_FREQ_DEFAULT}}"
TRAINER_SAVE_FREQ_N2_R16="${TRAINER_SAVE_FREQ_N2_R16:-${TRAINER_SAVE_FREQ_DEFAULT}}"
FULL_EVAL_FREQ_N8_R4="${FULL_EVAL_FREQ_N8_R4:-${FULL_EVAL_FREQ_DEFAULT}}"
FULL_EVAL_FREQ_N2_R16="${FULL_EVAL_FREQ_N2_R16:-${FULL_EVAL_FREQ_DEFAULT}}"

if [[ "${SETTINGS}" == "both" || "${SETTINGS}" == "n8_r4" ]]; then
  submit_run "n8_r4" "8" "4" "${TOTAL_STEPS_N8_R4}" "${TRAINER_SAVE_FREQ_N8_R4}" "${FULL_EVAL_FREQ_N8_R4}" "full"
fi

if [[ "${SETTINGS}" == "both" || "${SETTINGS}" == "n2_r16" ]]; then
  submit_run "n2_r16" "2" "16" "${TOTAL_STEPS_N2_R16}" "${TRAINER_SAVE_FREQ_N2_R16}" "${FULL_EVAL_FREQ_N2_R16}" "full"
fi
