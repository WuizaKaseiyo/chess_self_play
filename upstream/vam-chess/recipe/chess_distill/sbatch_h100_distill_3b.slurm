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
#SBATCH --output=$HOME/chess_rl_outputs/logs/slurm-%x-%j.out
#SBATCH --error=$HOME/chess_rl_outputs/logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# 4-GPU H100 launcher: on-policy distillation of Qwen2.5-3B from a 7B
# chess teacher. Implements Thinking Machines blog: per-token reverse-KL
# as advantage signal, discount=0.
#
# Required env (pass via --export or set inline):
#   TEACHER_CKPT  — HF-formatted teacher dir (Phase A best ckpt, merged via verl.model_merger)
#
# Optional:
#   STUDENT_MODEL — default $HOME/models/Qwen2.5-3B-Instruct
#   SMOKE=1       — 1-step smoke test (also bypasses val_before_train)
#   ROLLOUT_N     — default 8
#   TOTAL_TRAINING_STEPS — default 300
#
# Usage:
#   sbatch --partition=agent-xlong-15 \
#     --export=ALL,TEACHER_CKPT=/path/to/teacher_hf \
#     ./recipe/chess_distill/sbatch_h100_distill_3b.slurm

set -xeuo pipefail

# ---- Cluster idiosyncrasy ------------------------------------------------
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# ---- WandB (key hidden; same key as teacher run) -------------------------
{ set +x; } 2>/dev/null
export WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY must be set externally}
export WANDB_PROJECT=chess_rl
set -x

# ---- Conda env ----------------------------------------------------------
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate chess

echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

# ---- Repo + Python path -------------------------------------------------
PROJECT_DIR=$HOME/chess/vam-chess
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Sanity: distill estimator is registered.
python -c "
import sys; sys.path.insert(0,'${PROJECT_DIR}')
from verl.trainer.ppo.core_algos import AdvantageEstimator, get_adv_estimator_fn
assert AdvantageEstimator.DISTILL.value == 'distill'
fn = get_adv_estimator_fn('distill')
print('[sanity] distill estimator OK:', fn)
"

# ---- Inputs ------------------------------------------------------------
: "${TEACHER_CKPT:?TEACHER_CKPT must point to an HF-format teacher model dir}"
export TEACHER_CKPT
export MODEL_PATH="${STUDENT_MODEL:-$HOME/models/Qwen2.5-3B-Instruct}"

# ---- Output dir --------------------------------------------------------
RUN_TAG="chess_distill_3b_${SMOKE:+smoke_}$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE_DIR=$HOME/chess_rl_outputs/${RUN_TAG}
mkdir -p "${OUTPUT_BASE_DIR}" "${OUTPUT_BASE_DIR%/*}/logs"
export CHESS_RL_VERL_BASE_DIR="${OUTPUT_BASE_DIR}"

# ---- SMOKE vs FULL toggle ---------------------------------------------
SMOKE="${SMOKE:-0}"
if [ "${SMOKE}" = "1" ]; then
    export TOTAL_TRAINING_STEPS=1
    export VAL_BEFORE_TRAIN=False
    RUN_KIND=smoke
else
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
    export VAL_BEFORE_TRAIN=True
    RUN_KIND=full
fi
echo "[INFO] RUN_TAG=${RUN_TAG}  RUN_KIND=${RUN_KIND}"

# ---- Data + prompt (same as teacher) ------------------------------------
export CHESS_DATA_DIR="${CHESS_DATA_DIR:-data/chess_puzzles_chessr1_aligned_sharded_baseline}"
export USE_HARD_DATASET=False
export CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH=recipe/chess/prompt_templates/original_chessr1_prompt.jinja
export CHESS_RL_VAL_FILES='[data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet,data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet]'

# ---- Reward (still computed but ignored by distill advantage) -----------
export CHESS_REWARD_FN=expected_score_wdl_vs_best

# ---- Generation envelope ----------------------------------------------
export MAX_PROMPT_LENGTH=1536
export MAX_RESPONSE_LENGTH=2000
export ROLLOUT_N="${ROLLOUT_N:-8}"

# ---- Batch + parallelism ----------------------------------------------
# Student is 3B → TP=1 fits; teacher (ref) is 7B → FSDP across 4 GPUs with offload.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export TOTAL_EPOCHS=1
export CHESS_RL_TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export CHESS_RL_PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
export CHESS_RL_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-16}"
export CHESS_RL_REF_LOGPROB_MICRO_BATCH_SIZE="${REF_LOGPROB_MICRO_BATCH_SIZE:-8}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"

# ---- Eval / save cadence (smaller test_freq since distill converges fast) -
export TRAINER_TEST_FREQ=25
export CHESS_RL_TRAINER_SAVE_FREQ=50
export CHESS_RL_FULL_EVAL_FREQ=-1  # no in-training Stockfish (no binary at training nodes)
export CHESS_RL_MODEL_ONLY_CKPT=True

# ---- WandB / HF --------------------------------------------------------
export CHESS_RL_WANDB_MODE=online
export CHESS_RL_HF_UPLOAD_ENABLE=False

# ---- Single-node --------------------------------------------------------
export TRAINER_NNODES=1
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"

# ---- Env var translations (CHESS_* → internal names train_chess.sh reads) --
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

# Print config snapshot
{ set +x; } 2>/dev/null
echo "================ Resolved Distill Config ================"
echo "STUDENT (MODEL_PATH)=${MODEL_PATH}"
echo "TEACHER (REF_MODEL_PATH via train_chess_distill.sh)=${TEACHER_CKPT}"
echo "ROLLOUT_N=${ROLLOUT_N}  TRAIN_BATCH=${TRAIN_BATCH_SIZE}  STEPS=${TOTAL_TRAINING_STEPS}"
echo "DATA_DIR=${CHESS_DATA_DIR}"
echo "RUN_KIND=${RUN_KIND}"
echo "OUTPUT=${CHESS_RL_VERL_BASE_DIR}"
echo "=========================================================="
set -x

# ---- Launch via distill wrapper ----------------------------------------
exec bash "${PROJECT_DIR}/recipe/chess_distill/train_chess_distill.sh"
