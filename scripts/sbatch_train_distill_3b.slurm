#!/bin/bash
#SBATCH --job-name=chess-distill-3b
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# Stage B — On-policy distillation.
# Qwen2.5-3B-Instruct student distilled from the 7B teacher (Stage A output).
# Implements the Thinking Machines blog recipe: per-token reverse-KL as the
# advantage signal, discount=0, no group normalization.
#
# Required env (pass via --export=ALL,VAR=...):
#   UPSTREAM_DIR  — local clone of upstream RL framework with chess + distill recipes.
#   TEACHER_CKPT  — HF-format teacher model dir (merged from Stage A best ckpt
#                   via `python -m verl.model_merger`).
#   WANDB_API_KEY — your wandb token.
#
# Optional:
#   STUDENT_MODEL — default $HOME/models/Qwen2.5-3B-Instruct
#   ROLLOUT_N     — default 8
#   TOTAL_TRAINING_STEPS — default 300
#   SMOKE=1       — 1-step smoke test
#
# Usage:
#   sbatch --export=ALL,UPSTREAM_DIR=$HOME/chess_upstream,TEACHER_CKPT=$HOME/chess_rl_outputs/teacher_hf,WANDB_API_KEY=$WANDB_API_KEY \
#          scripts/sbatch_train_distill_3b.slurm

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
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-chess}"

nvidia-smi -L || true

# ---- Inputs --------------------------------------------------------------
UPSTREAM_DIR="${UPSTREAM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/upstream/vam-chess}"
: "${TEACHER_CKPT:?TEACHER_CKPT must point to an HF-format teacher model dir}"
export TEACHER_CKPT
export MODEL_PATH="${STUDENT_MODEL:-$HOME/models/Qwen2.5-3B-Instruct}"

cd "${UPSTREAM_DIR}"
export PYTHONPATH="${UPSTREAM_DIR}:${PYTHONPATH:-}"

# Sanity: distill estimator registered?
python -c "
import sys; sys.path.insert(0,'${UPSTREAM_DIR}')
from verl.trainer.ppo.core_algos import AdvantageEstimator, get_adv_estimator_fn
assert AdvantageEstimator.DISTILL.value == 'distill'
fn = get_adv_estimator_fn('distill')
print('[sanity] distill estimator OK:', fn)
"

# ---- Output dir ---------------------------------------------------------
OUT_BASE_DIR="${OUT_BASE_DIR:-$HOME/chess_rl_outputs}"
RUN_TAG="distill_3b_${SMOKE:+smoke_}$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE_DIR="${OUT_BASE_DIR%/}/${RUN_TAG}"
mkdir -p "${OUTPUT_BASE_DIR}" "$(dirname "${OUTPUT_BASE_DIR}")/logs"
export CHESS_RL_VERL_BASE_DIR="${OUTPUT_BASE_DIR}"

# ---- SMOKE vs FULL toggle -----------------------------------------------
SMOKE="${SMOKE:-0}"
if [ "${SMOKE}" = "1" ]; then
    export TOTAL_TRAINING_STEPS=1
    export VAL_BEFORE_TRAIN=False
else
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
    export VAL_BEFORE_TRAIN=True
fi

# ---- Data + prompt (same as teacher) ------------------------------------
export CHESS_DATA_DIR="${CHESS_DATA_DIR:-data/chess_puzzles_chessr1_aligned_sharded_baseline}"
export USE_HARD_DATASET=False
export CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH=recipe/chess/prompt_templates/original_chessr1_prompt.jinja
export CHESS_RL_VAL_FILES='[data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet,data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet]'

# ---- Reward (computed for logging but ignored by distill advantage) ----
export CHESS_REWARD_FN=expected_score_wdl_vs_best

# ---- Generation envelope ------------------------------------------------
export MAX_PROMPT_LENGTH=1536
export MAX_RESPONSE_LENGTH=2000
export ROLLOUT_N="${ROLLOUT_N:-8}"

# ---- Batch + parallelism ------------------------------------------------
# Student is 3B -> TP=1 fits; teacher (ref) is 7B -> FSDP across 4 GPUs with offload.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export TOTAL_EPOCHS=1
export CHESS_RL_TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export CHESS_RL_PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
export CHESS_RL_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-16}"
export CHESS_RL_REF_LOGPROB_MICRO_BATCH_SIZE="${REF_LOGPROB_MICRO_BATCH_SIZE:-8}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"

# ---- Eval / save cadence (distill converges fast) ----------------------
export TRAINER_TEST_FREQ=25
export CHESS_RL_TRAINER_SAVE_FREQ=50
export CHESS_RL_FULL_EVAL_FREQ=-1
export CHESS_RL_MODEL_ONLY_CKPT=True

# ---- WandB / HF --------------------------------------------------------
export CHESS_RL_WANDB_MODE=online
export CHESS_RL_HF_UPLOAD_ENABLE=False

# ---- Single-node -------------------------------------------------------
export TRAINER_NNODES=1
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"

# ---- Env var translations (CHESS_* -> internal names train_chess_distill.sh reads) --
export TRAIN_BATCH_SIZE
export TOTAL_TRAINING_STEPS
export TRAINER_SAVE_FREQ="${CHESS_RL_TRAINER_SAVE_FREQ}"
export TRAINER_RESUME_MODE=disable
export FULL_EVAL_FREQ="${CHESS_RL_FULL_EVAL_FREQ}"
export PPO_MICRO_BATCH_SIZE="${CHESS_RL_PPO_MICRO_BATCH_SIZE}"
export ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="${CHESS_RL_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE}"
export REF_LOGPROB_MICRO_BATCH_SIZE="${CHESS_RL_REF_LOGPROB_MICRO_BATCH_SIZE}"
export WANDB_MODE="${CHESS_RL_WANDB_MODE}"
if [ -n "${CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH:-}" ]; then
    export FULL_EVAL_PROMPT_TEMPLATE_PATH="${CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH}"
fi
export CHESS_RL_JOB_NAME="${SLURM_JOB_NAME:-chess-distill-3b}"

# ---- Launch via distill wrapper ----------------------------------------
exec bash "${UPSTREAM_DIR}/recipe/chess_distill/train_chess_distill.sh"
