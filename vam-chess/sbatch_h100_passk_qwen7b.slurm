#!/bin/bash
#SBATCH --job-name=cr1-passk-qwen7b
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=5-00:00:00
#SBATCH --output=$HOME/chess_rl_outputs/logs/slurm-%x-%j.out
#SBATCH --error=$HOME/chess_rl_outputs/logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# H100 launcher: Qwen2.5-7B-Instruct + Pass@k (n=16, k=4) baseline.
# Extrapolates the f5guq4ti 3B config (vam-chess paper) to 7B as
# teacher model for downstream on-policy distillation experiments.
#
# Usage:
#   sbatch ./sbatch_h100_passk_qwen7b.slurm                  # full run
#   SMOKE=1 sbatch --export=ALL,SMOKE=1 ./sbatch_h100_passk_qwen7b.slurm   # 1-step smoke
#
# Overrides:
#   --export=ALL,MODEL_PATH=/path/to/other/model
#   --export=ALL,TENSOR_MODEL_PARALLEL_SIZE=2  (if OOM with TP=1)

set -xeuo pipefail

# ---- Cluster-idiosyncrasy fixes (mirrors submit_chess_dupo.sh) -------------
# SLURM ships both AMD and NVIDIA device vars; verl's worker.py refuses to start
# when both are set. Strip the AMD ones; this is an NVIDIA-only cluster.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

# vLLM 0.10 reads envs.VLLM_USE_V1 as False by default which crashes the HTTP
# server. Force V1 explicitly.
export VLLM_USE_V1=1

# Full Hydra traceback on any error.
export HYDRA_FULL_ERROR=1

# Deterministic device ordering.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# ---- WandB ---------------------------------------------------------------
# Key kept here (in the launcher under vam-chess/, NOT under any git repo)
# to keep it out of git history. set +x so it does not get echoed.
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
# Put vam-chess first on sys.path so its bundled verl (with Pass@k)
# is preferred over anything else.
PROJECT_DIR=$HOME/chess/vam-chess
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Sanity print: which verl + has Pass@k?
python -c "
import sys; sys.path.insert(0,'${PROJECT_DIR}')
import verl
from verl.trainer.ppo import core_algos
print('[sanity] verl path:', verl.__file__)
print('[sanity] has Pass@k:', hasattr(core_algos, 'passk_advantages_max_subsets'))
"

# ---- Output dir ----------------------------------------------------------
RUN_TAG="cr1_passk_qwen7b_${SMOKE:+smoke_}$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE_DIR=$HOME/chess_rl_outputs/${RUN_TAG}
mkdir -p "${OUTPUT_BASE_DIR}" "${OUTPUT_BASE_DIR%/*}/logs"
export CHESS_RL_VERL_BASE_DIR="${OUTPUT_BASE_DIR}"

# ---- SMOKE vs FULL toggle ---------------------------------------------
SMOKE="${SMOKE:-0}"
if [ "${SMOKE}" = "1" ]; then
    export CHESS_RL_TOTAL_TRAINING_STEPS=1
    export VAL_BEFORE_TRAIN=False
    RUN_KIND=smoke
else
    export CHESS_RL_TOTAL_TRAINING_STEPS=null
    export VAL_BEFORE_TRAIN=True
    RUN_KIND=full
fi
echo "[INFO] RUN_TAG=${RUN_TAG}  RUN_KIND=${RUN_KIND}"

# ---- Model + tokenizer ------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen2.5-7B-Instruct}"

# ---- 7B Pass@k config (translated from launch_baseline_passk_qwen3b.sh) -
# Algorithm
export CHESS_PASS_K_TRAINING=True
export CHESS_PASS_K=4
export CHESS_ALLOWED_MOVE_ELIM_ENABLE=False
export FILTER_GROUPS_ENABLE=False
export CHESS_DIVERSITY_ENABLE=False
export CHESS_RL_USE_KL_LOSS=True
export FORCED_PREFIX_ENABLE=False

# Reward + data
export CHESS_REWARD_FN=expected_score_wdl_vs_best
export CHESS_DATA_DIR=data/chess_puzzles_chessr1_aligned_sharded_baseline
export USE_HARD_DATASET=False
export CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH=recipe/chess/prompt_templates/original_chessr1_prompt.jinja
export CHESS_RL_VAL_FILES='[data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet,data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet]'

# Generation envelope
export MAX_PROMPT_LENGTH=1536
export MAX_RESPONSE_LENGTH=2000
export ROLLOUT_N=16

# Batch sizes (3B used micro=8; for 7B start with micro=4 and TP=1 on H100 80G).
# Bump TP to 2 if smoke OOMs.
export CHESS_RL_TRAIN_BATCH_SIZE=128
export TOTAL_EPOCHS=1
export CHESS_RL_TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export CHESS_RL_PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"
export CHESS_RL_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-8}"
export CHESS_RL_REF_LOGPROB_MICRO_BATCH_SIZE="${REF_LOGPROB_MICRO_BATCH_SIZE:-8}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"

# Eval / save cadence (matches f5guq4ti)
# NOTE: FULL_EVAL_FREQ=-1 disables the in-training Stockfish full-game eval
# (we don't have stockfish binary installed; teacher quality is measured via
# pass@k eval after training, which uses a different code path).
export TRAINER_TEST_FREQ=40
export CHESS_RL_TRAINER_SAVE_FREQ=80
export CHESS_RL_FULL_EVAL_FREQ=-1
export CHESS_RL_MODEL_ONLY_CKPT=True

# WandB
export CHESS_RL_WANDB_MODE=online

# HF upload disabled (no HF token configured)
export CHESS_RL_HF_UPLOAD_ENABLE=False

# Multi-node not used (single H100 node)
export TRAINER_NNODES=1
export TRAINER_N_GPUS_PER_NODE=4

# ---- Env var translations (CHESS_* / CHESS_RL_* -> internal names) -----
# train_chess.sh reads non-prefixed names like PASS_K_TRAINING; the original
# GH200 slurm wrapper does these exports. We replicate them here since we
# bypass that wrapper. This is the CRITICAL block — without it Pass@k stays off.
export PASS_K_TRAINING="${CHESS_PASS_K_TRAINING}"
export PASS_K="${CHESS_PASS_K}"
export DIVERSITY_ENABLE="${CHESS_DIVERSITY_ENABLE}"
export DIVERSITY_METHOD="${DIVERSITY_METHOD:-none}"
export DIVERSITY_LAMBDA="${DIVERSITY_LAMBDA:-0.0}"
export DIVERSITY_DISTINCT_K="${DIVERSITY_DISTINCT_K:-4}"
export DIVERSITY_INCLUDE_BASE_ADVANTAGE="${DIVERSITY_INCLUDE_BASE_ADVANTAGE:-True}"
export ALLOWED_MOVE_ELIM_ENABLE="${CHESS_ALLOWED_MOVE_ELIM_ENABLE}"
export FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE}"
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
export CHESS_RL_JOB_NAME="${SLURM_JOB_NAME:-cr1-passk-qwen7b}"

# ---- Print the resolved config (for debugging) ------------------------
{ set +x; } 2>/dev/null
echo "================ Resolved Config ================"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CHESS_PASS_K_TRAINING=${CHESS_PASS_K_TRAINING}  CHESS_PASS_K=${CHESS_PASS_K}"
echo "ROLLOUT_N=${ROLLOUT_N}  TRAIN_BATCH=${CHESS_RL_TRAIN_BATCH_SIZE}"
echo "TP=${CHESS_RL_TENSOR_MODEL_PARALLEL_SIZE}  PPO_MICRO=${CHESS_RL_PPO_MICRO_BATCH_SIZE}"
echo "DATA_DIR=${CHESS_DATA_DIR}"
echo "FULL_EVAL_TEMPLATE=${CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH}"
echo "REWARD_FN=${CHESS_REWARD_FN}"
echo "RUN_KIND=${RUN_KIND}  TOTAL_STEPS=${CHESS_RL_TOTAL_TRAINING_STEPS}"
echo "OUTPUT=${CHESS_RL_VERL_BASE_DIR}"
echo "=================================================="
set -x

# ---- Launch -----------------------------------------------------------
bash train_chess.sh
