#!/bin/bash
#SBATCH --job-name=student-base-passk
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=$HOME/chess_rl_outputs/logs/slurm-%x-%j.out
#SBATCH --error=$HOME/chess_rl_outputs/logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# Phase B: Qwen2.5-3B-Instruct base pass@8 eval on chess puzzles test split.
# Establishes the student-baseline control group for the Phase E 3-way comparison
# (base 3B / GRPO 3B from paper / distill 3B).
#
# Single GPU, TP=1 (3B fits trivially on one H100 80GB). Lands on agent-xlong
# (any of 507-11..14) since 507-15 is occupied by teacher training job 8723.

set -xeuo pipefail

# ---- Cluster-idiosyncrasy fixes ------------------------------------------
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# ---- WandB (no streaming for eval; key kept hidden from echo) -----------
{ set +x; } 2>/dev/null
export WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY must be set externally}
export WANDB_PROJECT=chess_rl
export WANDB_MODE="${WANDB_MODE:-disabled}"
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

# ---- Eval config --------------------------------------------------------
MODEL="${MODEL:-$HOME/models/Qwen2.5-3B-Instruct}"
PARQUET="${PARQUET:-data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet}"
OUT_JSON="${OUT_JSON:-eval_results/student_baseline_passk.json}"
K_MAX="${K_MAX:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3584}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-0}"

mkdir -p "$(dirname "${OUT_JSON}")"

echo "[eval] MODEL=${MODEL}"
echo "[eval] PARQUET=${PARQUET}"
echo "[eval] OUT_JSON=${OUT_JSON}  K_MAX=${K_MAX}"

python -m scripts.eval_chess_passk \
    --model "${MODEL}" \
    --parquet "${PARQUET}" \
    --k_max "${K_MAX}" \
    --do_sample \
    --seed "${SEED}" \
    --seed_mode engine \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --batch_size "${BATCH_SIZE}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --max_response_length "${MAX_RESPONSE_LENGTH}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --acc_key acc \
    --out_json "${OUT_JSON}"

echo "[eval] Done: ${OUT_JSON}"
