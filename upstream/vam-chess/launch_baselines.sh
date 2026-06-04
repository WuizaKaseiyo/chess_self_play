#!/usr/bin/env bash

set -euo pipefail

# Optional Hugging Face checkpoint upload (opt-in).
# - `CHESS_RL_TRAINER_SAVE_FREQ` controls *local* checkpoint saves.
# - HF uploads are token-gated in `verl/trainer/ppo/ray_trainer.py`, but the GH200 Slurm launcher
#   additionally requires `CHESS_RL_HF_UPLOAD_ENABLE=True` because it unsets HF token env vars otherwise.
CHESS_RL_HF_UPLOAD_ENABLE="${CHESS_RL_HF_UPLOAD_ENABLE:-False}"
CHESS_RL_HF_CKPT_REPO_ID="${CHESS_RL_HF_CKPT_REPO_ID:-}"
CHESS_RL_HF_TOKEN_FILE="${CHESS_RL_HF_TOKEN_FILE:-${HOME}/.hf_token}"
HF_EXPORT_KVS="CHESS_RL_HF_UPLOAD_ENABLE=${CHESS_RL_HF_UPLOAD_ENABLE},CHESS_RL_HF_CKPT_REPO_ID=${CHESS_RL_HF_CKPT_REPO_ID},CHESS_RL_HF_TOKEN_FILE=${CHESS_RL_HF_TOKEN_FILE}"

# Optional Pass@k GRPO baseline variant (default-off).
CHESS_PASS_K_TRAINING="${CHESS_PASS_K_TRAINING:-False}"
CHESS_PASS_K="${CHESS_PASS_K:-1}"
PASSK_EXPORT_KVS="CHESS_PASS_K_TRAINING=${CHESS_PASS_K_TRAINING},CHESS_PASS_K=${CHESS_PASS_K}"

# Optional one-seed reproducibility (dataloader seed).
CHESS_RL_DATA_SEED="${CHESS_RL_DATA_SEED:-null}"
SEED_EXPORT_KVS="CHESS_RL_DATA_SEED=${CHESS_RL_DATA_SEED}"

# Model sweep list: "model_path|max_response_length|tensor_parallel_level|ppo_micro_batch_size|ref_micro_batch_size"
MODEL_SWEEP=(
  "/projects/a5l/ziyan/models/Qwen/Qwen2.5-7B-Instruct|2000|1|8|16"
  "/projects/a5l/ziyan/models/Qwen/Qwen2.5-3B-Instruct|2000|1|16|32"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "[INFO] Submitting chess-rl jobs from: ${SCRIPT_DIR}"
echo "[INFO] CHESS_RL_HF_UPLOAD_ENABLE=${CHESS_RL_HF_UPLOAD_ENABLE}"
echo "[INFO] CHESS_PASS_K_TRAINING=${CHESS_PASS_K_TRAINING}"
echo "[INFO] CHESS_RL_DATA_SEED=${CHESS_RL_DATA_SEED}"
if [[ "${CHESS_PASS_K_TRAINING}" =~ ^(1|true|True|TRUE|yes|YES|y|Y)$ ]]; then
  echo "[INFO] CHESS_PASS_K=${CHESS_PASS_K}"
fi
if [[ "${CHESS_RL_HF_UPLOAD_ENABLE}" =~ ^(1|true|True|TRUE|yes|YES|y|Y)$ ]]; then
  echo "[INFO] CHESS_RL_HF_CKPT_REPO_ID=${CHESS_RL_HF_CKPT_REPO_ID:-<unset>}"
  echo "[INFO] CHESS_RL_HF_TOKEN_FILE=${CHESS_RL_HF_TOKEN_FILE}"
  if [ -z "${CHESS_RL_HF_CKPT_REPO_ID}" ]; then
    echo "[WARN] CHESS_RL_HF_UPLOAD_ENABLE=True but CHESS_RL_HF_CKPT_REPO_ID is unset; sbatch defaults to 'Gabr1e11/a_lot_of_models' (likely not writable)." >&2
  fi
  if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -z "${HUGGINGFACE_TOKEN:-}" ] && [ ! -f "${CHESS_RL_HF_TOKEN_FILE}" ]; then
    echo "[WARN] HF upload is enabled but no token env var and no token file at '${CHESS_RL_HF_TOKEN_FILE}'. The job will fail early unless you provide one." >&2
  fi
else
  echo "[INFO] HF upload disabled; no checkpoints will be uploaded to Hugging Face."
  echo "[INFO] To enable: export CHESS_RL_HF_UPLOAD_ENABLE=True and CHESS_RL_HF_CKPT_REPO_ID=<user/repo>."
fi

for entry in "${MODEL_SWEEP[@]:1:2}"; do
  IFS='|' read -r model_name max_response_length tensor_parallel_level ppo_micro_batch_size ref_micro_batch_size <<< "${entry}"

    # GRPO n=8
	  CHESS_RL_VAL_FILES='[data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet,data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet]' \
	    sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=144 \
	    --partition=workq --account=brics.a5l --job-name=cr1-baseline-expectedscore-wdl-fulleval-n8 \
    --export=ALL,${HF_EXPORT_KVS},${PASSK_EXPORT_KVS},${SEED_EXPORT_KVS},CHESS_RL_VERL_BASE_DIR=/projects/a5l/ziyan/chess_rl_outputs/cr1_expectedscore_wdl_fulleval_$(date +%Y%m%d_%H%M%S),CHESS_DATA_DIR=data/chess_puzzles_chessr1_aligned_sharded_baseline,CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH=recipe/chess/prompt_templates/original_chessr1_prompt.jinja,USE_HARD_DATASET=False,CHESS_REWARD_FN=expected_score_wdl_vs_best,CHESS_RL_TRAIN_BATCH_SIZE=128,TOTAL_EPOCHS=1,CHESS_ALLOWED_MOVE_ELIM_ENABLE=False,FILTER_GROUPS_ENABLE=False,CHESS_RL_TENSOR_MODEL_PARALLEL_SIZE=${tensor_parallel_level},PPO_MICRO_BATCH_SIZE=${ppo_micro_batch_size},REF_LOGPROB_MICRO_BATCH_SIZE=${ref_micro_batch_size},TRAINER_TEST_FREQ=40,MAX_PROMPT_LENGTH=1536,MAX_RESPONSE_LENGTH=${max_response_length},ROLLOUT_N=8,MODEL_PATH=${model_name},CHESS_RL_WANDB_MODE=online,CHESS_RL_TRAINER_SAVE_FREQ=80,CHESS_RL_FULL_EVAL_FREQ=80,CHESS_RL_USE_KL_LOSS=True \
	    ./sbatch_train_chess_gh200.slurm

  sleep 10

  # GRPO n=32
	  CHESS_RL_VAL_FILES='[data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet,data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet]' \
	    sbatch --nodes=4 --ntasks=4 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=144 \
	    --partition=workq --account=brics.a5l --job-name=cr1-baseline-expectedscore-wdl-fulleval-n32 \
    --export=ALL,${HF_EXPORT_KVS},${PASSK_EXPORT_KVS},${SEED_EXPORT_KVS},CHESS_RL_VERL_BASE_DIR=/projects/a5l/ziyan/chess_rl_outputs/cr1_expectedscore_wdl_fulleval_$(date +%Y%m%d_%H%M%S),CHESS_DATA_DIR=data/chess_puzzles_chessr1_aligned_sharded_baseline,CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH=recipe/chess/prompt_templates/original_chessr1_prompt.jinja,USE_HARD_DATASET=False,CHESS_REWARD_FN=expected_score_wdl_vs_best,CHESS_RL_TRAIN_BATCH_SIZE=128,TOTAL_EPOCHS=1,CHESS_ALLOWED_MOVE_ELIM_ENABLE=False,FILTER_GROUPS_ENABLE=False,CHESS_RL_TENSOR_MODEL_PARALLEL_SIZE=${tensor_parallel_level},PPO_MICRO_BATCH_SIZE=${ppo_micro_batch_size},REF_LOGPROB_MICRO_BATCH_SIZE=${ref_micro_batch_size},TRAINER_TEST_FREQ=40,MAX_PROMPT_LENGTH=1536,MAX_RESPONSE_LENGTH=${max_response_length},ROLLOUT_N=32,MODEL_PATH=${model_name},CHESS_RL_WANDB_MODE=online,CHESS_RL_TRAINER_SAVE_FREQ=80,CHESS_RL_FULL_EVAL_FREQ=80,CHESS_RL_USE_KL_LOSS=True \
	    ./sbatch_train_chess_gh200.slurm

  sleep 10
done

for entry in "${MODEL_SWEEP[@]:1:2}"; do
  IFS='|' read -r model_name max_response_length tensor_parallel_level ppo_micro_batch_size ref_micro_batch_size <<< "${entry}"

    # GRPO n=8
	  CHESS_RL_VAL_FILES='[data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet,data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet]' \
	    sbatch --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=144 \
	    --partition=workq --account=brics.a5l --job-name=cr1-baseline-expectedscore-wdl-fulleval-n8 \
    --export=ALL,${HF_EXPORT_KVS},${PASSK_EXPORT_KVS},${SEED_EXPORT_KVS},CHESS_RL_VERL_BASE_DIR=/projects/a5l/ziyan/chess_rl_outputs/cr1_expectedscore_wdl_fulleval_$(date +%Y%m%d_%H%M%S),CHESS_DATA_DIR=data/chess_puzzles_chessr1_aligned_sharded_baseline,CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH=recipe/chess/prompt_templates/original_chessr1_prompt.jinja,USE_HARD_DATASET=False,CHESS_REWARD_FN=expected_score_wdl_vs_best,CHESS_RL_TRAIN_BATCH_SIZE=128,TOTAL_EPOCHS=1,CHESS_ALLOWED_MOVE_ELIM_ENABLE=False,FILTER_GROUPS_ENABLE=False,CHESS_RL_TENSOR_MODEL_PARALLEL_SIZE=${tensor_parallel_level},PPO_MICRO_BATCH_SIZE=${ppo_micro_batch_size},REF_LOGPROB_MICRO_BATCH_SIZE=${ref_micro_batch_size},TRAINER_TEST_FREQ=40,MAX_PROMPT_LENGTH=1536,MAX_RESPONSE_LENGTH=${max_response_length},ROLLOUT_N=8,MODEL_PATH=${model_name},CHESS_RL_WANDB_MODE=online,CHESS_RL_TRAINER_SAVE_FREQ=80,CHESS_RL_FULL_EVAL_FREQ=80,CHESS_RL_USE_KL_LOSS=True \
	    ./sbatch_train_chess_gh200.slurm

  sleep 10

  # GRPO n=32
	  CHESS_RL_VAL_FILES='[data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet,data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet]' \
	    sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=144 \
	    --partition=workq --account=brics.a5l --job-name=cr1-baseline-expectedscore-wdl-fulleval-n32 \
    --export=ALL,${HF_EXPORT_KVS},${PASSK_EXPORT_KVS},${SEED_EXPORT_KVS},CHESS_RL_VERL_BASE_DIR=/projects/a5l/ziyan/chess_rl_outputs/cr1_expectedscore_wdl_fulleval_$(date +%Y%m%d_%H%M%S),CHESS_DATA_DIR=data/chess_puzzles_chessr1_aligned_sharded_baseline,CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH=recipe/chess/prompt_templates/original_chessr1_prompt.jinja,USE_HARD_DATASET=False,CHESS_REWARD_FN=expected_score_wdl_vs_best,CHESS_RL_TRAIN_BATCH_SIZE=128,TOTAL_EPOCHS=1,CHESS_ALLOWED_MOVE_ELIM_ENABLE=False,FILTER_GROUPS_ENABLE=False,CHESS_RL_TENSOR_MODEL_PARALLEL_SIZE=${tensor_parallel_level},PPO_MICRO_BATCH_SIZE=${ppo_micro_batch_size},REF_LOGPROB_MICRO_BATCH_SIZE=${ref_micro_batch_size},TRAINER_TEST_FREQ=40,MAX_PROMPT_LENGTH=1536,MAX_RESPONSE_LENGTH=${max_response_length},ROLLOUT_N=32,MODEL_PATH=${model_name},CHESS_RL_WANDB_MODE=online,CHESS_RL_TRAINER_SAVE_FREQ=80,CHESS_RL_FULL_EVAL_FREQ=80,CHESS_RL_USE_KL_LOSS=True \
	    ./sbatch_train_chess_gh200.slurm

  sleep 10
done
