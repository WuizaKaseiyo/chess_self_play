#!/bin/bash
#SBATCH --job-name=chess-teacher-7b
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# Stage A — Teacher training.
# Qwen2.5-7B-Instruct + Pass@k GRPO (n=16, k=4) on Stockfish-graded chess puzzles.
# Produces the 7B "teacher" used as the on-policy distillation reference.
#
# Required env (pass via --export=ALL,VAR=...):
#   UPSTREAM_DIR  — local clone of upstream RL framework with chess recipe.
#                   Must contain: train_chess.sh, recipe/chess/, recipe/chess_distill/,
#                                 data/chess_puzzles_chessr1_aligned_sharded_baseline/*
#   WANDB_API_KEY — your wandb token (do NOT commit; use --export=ALL,WANDB_API_KEY=$WANDB_API_KEY)
#
# Optional:
#   MODEL_PATH    — base model (default: $HOME/models/Qwen2.5-7B-Instruct)
#   OUT_BASE_DIR  — checkpoint root (default: $HOME/chess_rl_outputs)
#   SMOKE=1       — 1-step smoke test
#
# Usage:
#   sbatch --export=ALL,UPSTREAM_DIR=$HOME/chess_upstream,WANDB_API_KEY=$WANDB_API_KEY \
#          scripts/sbatch_train_teacher_7b.slurm

set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# ---- WandB ---------------------------------------------------------------
: "${WANDB_API_KEY:?WANDB_API_KEY must be set via --export=ALL,WANDB_API_KEY=...}"
{ set +x; } 2>/dev/null
export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-chess_self_play}"
set -x

# ---- Conda env -----------------------------------------------------------
# Edit this to point at your conda installation.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-chess}"

nvidia-smi -L || true

# ---- Upstream framework path ---------------------------------------------
UPSTREAM_DIR="${UPSTREAM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/upstream/vam-chess}"
cd "${UPSTREAM_DIR}"
export PYTHONPATH="${UPSTREAM_DIR}:${PYTHONPATH:-}"

# Sanity: Pass@k advantage estimator present?
python -c "
import sys; sys.path.insert(0,'${UPSTREAM_DIR}')
import verl
from verl.trainer.ppo import core_algos
print('[sanity] verl path:', verl.__file__)
assert hasattr(core_algos, 'passk_advantages_max_subsets'), 'Pass@k not patched'
print('[sanity] Pass@k OK')
"

# ---- Output dir ----------------------------------------------------------
OUT_BASE_DIR="${OUT_BASE_DIR:-$HOME/chess_rl_outputs}"
RUN_TAG="teacher_7b_passk_${SMOKE:+smoke_}$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE_DIR="${OUT_BASE_DIR%/}/${RUN_TAG}"
mkdir -p "${OUTPUT_BASE_DIR}" "$(dirname "${OUTPUT_BASE_DIR}")/logs"
export CHESS_RL_VERL_BASE_DIR="${OUTPUT_BASE_DIR}"

# ---- SMOKE vs FULL toggle ------------------------------------------------
SMOKE="${SMOKE:-0}"
if [ "${SMOKE}" = "1" ]; then
    export CHESS_RL_TOTAL_TRAINING_STEPS=1
    export VAL_BEFORE_TRAIN=False
else
    export CHESS_RL_TOTAL_TRAINING_STEPS=null
    export VAL_BEFORE_TRAIN=True
fi

# ---- Model + tokenizer --------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen2.5-7B-Instruct}"

# ---- Algorithm: Pass@k GRPO (n=16, k=4) ---------------------------------
export CHESS_PASS_K_TRAINING=True
export CHESS_PASS_K=4
export CHESS_ALLOWED_MOVE_ELIM_ENABLE=False
export FILTER_GROUPS_ENABLE=False
export CHESS_DIVERSITY_ENABLE=False
export CHESS_RL_USE_KL_LOSS=True
export FORCED_PREFIX_ENABLE=False

# ---- Reward + data -------------------------------------------------------
export CHESS_REWARD_FN=expected_score_wdl_vs_best
export CHESS_DATA_DIR=data/chess_puzzles_chessr1_aligned_sharded_baseline
export USE_HARD_DATASET=False
export CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH=recipe/chess/prompt_templates/original_chessr1_prompt.jinja
export CHESS_RL_VAL_FILES='[data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet,data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet]'

# ---- Generation envelope -------------------------------------------------
export MAX_PROMPT_LENGTH=1536
export MAX_RESPONSE_LENGTH=2000
export ROLLOUT_N=16

# ---- Batch + parallelism -------------------------------------------------
export CHESS_RL_TRAIN_BATCH_SIZE=128
export TOTAL_EPOCHS=1
export CHESS_RL_TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export CHESS_RL_PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"
export CHESS_RL_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-8}"
export CHESS_RL_REF_LOGPROB_MICRO_BATCH_SIZE="${REF_LOGPROB_MICRO_BATCH_SIZE:-8}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"

# ---- Eval / save cadence -------------------------------------------------
export TRAINER_TEST_FREQ=40
export CHESS_RL_TRAINER_SAVE_FREQ=80
export CHESS_RL_FULL_EVAL_FREQ=-1   # in-training Stockfish eval off (no SF binary on train nodes)
export CHESS_RL_MODEL_ONLY_CKPT=True

# ---- WandB / HF ----------------------------------------------------------
export CHESS_RL_WANDB_MODE=online
export CHESS_RL_HF_UPLOAD_ENABLE=False

# ---- Single-node --------------------------------------------------------
export TRAINER_NNODES=1
export TRAINER_N_GPUS_PER_NODE=4

# ---- Env var translations (CHESS_* / CHESS_RL_* -> internal names) ------
# train_chess.sh expects non-prefixed names like PASS_K_TRAINING; replicate them.
export PASS_K_TRAINING="${CHESS_PASS_K_TRAINING}"
export PASS_K="${CHESS_PASS_K}"
export DIVERSITY_ENABLE="${CHESS_DIVERSITY_ENABLE}"
export DIVERSITY_METHOD="${DIVERSITY_METHOD:-none}"
export DIVERSITY_LAMBDA="${DIVERSITY_LAMBDA:-0.0}"
export DIVERSITY_DISTINCT_K="${DIVERSITY_DISTINCT_K:-4}"
export DIVERSITY_INCLUDE_BASE_ADVANTAGE="${DIVERSITY_INCLUDE_BASE_ADVANTAGE:-True}"
export ALLOWED_MOVE_ELIM_ENABLE="${CHESS_ALLOWED_MOVE_ELIM_ENABLE}"
export FILTER_GROUPS_ENABLE
export TRAIN_BATCH_SIZE="${CHESS_RL_TRAIN_BATCH_SIZE}"
export TOTAL_TRAINING_STEPS="${CHESS_RL_TOTAL_TRAINING_STEPS}"
export TRAINER_SAVE_FREQ="${CHESS_RL_TRAINER_SAVE_FREQ}"
export TRAINER_RESUME_MODE=disable
export FULL_EVAL_FREQ="${CHESS_RL_FULL_EVAL_FREQ}"
export PPO_MICRO_BATCH_SIZE="${CHESS_RL_PPO_MICRO_BATCH_SIZE}"
export ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="${CHESS_RL_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE}"
export REF_LOGPROB_MICRO_BATCH_SIZE="${CHESS_RL_REF_LOGPROB_MICRO_BATCH_SIZE}"
export USE_KL_LOSS="${CHESS_RL_USE_KL_LOSS}"
export WANDB_MODE="${CHESS_RL_WANDB_MODE}"
if [ -n "${CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH:-}" ]; then
    export FULL_EVAL_PROMPT_TEMPLATE_PATH="${CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH}"
fi
export CHESS_RL_JOB_NAME="${SLURM_JOB_NAME:-chess-teacher-7b}"

# ---- Launch -------------------------------------------------------------
bash train_chess.sh
