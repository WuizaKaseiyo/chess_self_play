#!/bin/bash
#SBATCH --job-name=chess-passk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm/slurm-%x-%j.out
#SBATCH --error=slurm/slurm-%x-%j.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

# Evaluate chess pass@k using vLLM on GH200 (Apptainer).
#
# Example (base model, pass@32, test split):
#   sbatch --wait --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,K_MAX=32,PARQUET=data/chess_puzzles/test.parquet,OUT_JSON=/projects/a5l/ziyan/chess_rl_outputs/passk_base_qwen2p5_7b_test_k32.json ./sbatch_eval_chess_passk_gh200.slurm

set -euo pipefail

# Preferred shared scratch (persistent, non-tmpfs on Isambard):
CHESS_RL_PROJECTS_ROOT="${CHESS_RL_PROJECTS_ROOT:-/projects/a5l/ziyan/chess_rl}"
PROJECTS_SCRATCH_OK=0
if [ -d "/projects" ]; then
  if mkdir -p "${CHESS_RL_PROJECTS_ROOT}" 2>/dev/null && [ -w "${CHESS_RL_PROJECTS_ROOT}" ]; then
    PROJECTS_SCRATCH_OK=1
  fi
fi
if [ "${PROJECTS_SCRATCH_OK}" -eq 1 ]; then
  export CHESS_RL_SINGULARITY_TMPDIR="${CHESS_RL_SINGULARITY_TMPDIR:-${CHESS_RL_PROJECTS_ROOT%/}/singularity}"
  export CHESS_RL_TMPDIR="${CHESS_RL_TMPDIR:-${CHESS_RL_PROJECTS_ROOT%/}/tmp}"
  export CHESS_RL_APPTAINER_CACHEDIR="${CHESS_RL_APPTAINER_CACHEDIR:-${CHESS_RL_PROJECTS_ROOT%/}/apptainer_cache}"
  export CHESS_RL_WANDB_DIR="${CHESS_RL_WANDB_DIR:-${CHESS_RL_PROJECTS_ROOT%/}/wandb}"
  mkdir -p "${CHESS_RL_SINGULARITY_TMPDIR}" "${CHESS_RL_TMPDIR}" "${CHESS_RL_APPTAINER_CACHEDIR}" "${CHESS_RL_WANDB_DIR}" || true
fi
if [ -n "${CHESS_RL_TMPDIR:-}" ]; then
  export TMPDIR="${CHESS_RL_TMPDIR}"
fi
if [ -n "${CHESS_RL_SINGULARITY_TMPDIR:-}" ]; then
  export SINGULARITY_TMPDIR="${CHESS_RL_SINGULARITY_TMPDIR}"
  export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SINGULARITY_TMPDIR}}"
fi
if [ -n "${CHESS_RL_APPTAINER_CACHEDIR:-}" ]; then
  export APPTAINER_CACHEDIR="${CHESS_RL_APPTAINER_CACHEDIR%/}/apptainer_cache_${SLURM_JOB_ID:-$$}"
  export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${APPTAINER_CACHEDIR}}"
fi
if [ -n "${CHESS_RL_WANDB_DIR:-}" ]; then
  export WANDB_DIR="${CHESS_RL_WANDB_DIR}"
fi

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

echo "[SLURM] Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "[SLURM] Job name:    ${SLURM_JOB_NAME:-chess-passk}"
echo "[SLURM] Node list:   ${SLURM_JOB_NODELIST:-N/A}"
echo "[SLURM] GPUs:        ${SLURM_GPUS_ON_NODE:-1} (requested via --gres=gpu:1)"
echo "[SLURM] Submit dir:  ${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  THIS_DIR="${SLURM_SUBMIT_DIR}"
else
  THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${THIS_DIR}"
echo "[SLURM] Repo dir:    ${THIS_DIR}"

HOST_ARCH="$(uname -m 2>/dev/null || echo unknown)"
echo "[SLURM] Host arch:   ${HOST_ARCH}"
if [ "${HOST_ARCH}" != "aarch64" ] && [ "${HOST_ARCH}" != "arm64" ]; then
  die "This script targets GH200 (arm64/aarch64) nodes, but uname reports '${HOST_ARCH}'."
fi

SIF_IMAGE="${CHESS_RL_SIF_IMAGE:-${HOME}/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif}"
CONTAINER_REPO_DIR="${CHESS_RL_CONTAINER_REPO_DIR:-/workspace/chess_rl}"
if [ ! -f "${SIF_IMAGE}" ]; then
  die "Missing SIF image: ${SIF_IMAGE}"
fi
echo "[SLURM] SIF_IMAGE=${SIF_IMAGE}"
echo "[SLURM] Repo mount: ${CONTAINER_REPO_DIR}"

# Cache root (avoid repeated model downloads).
PROJECT_ROOT="${CHESS_RL_PROJECT_ROOT:-/projects/a5l/ziyan}"
HF_CACHE_ROOT="${CHESS_RL_HF_CACHE_ROOT:-${PROJECT_ROOT%/}/hf_cache}"
mkdir -p "${HF_CACHE_ROOT}"
export HF_HOME="${HF_CACHE_ROOT%/}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"
echo "[SLURM] HF_HOME=${HF_HOME}"

# Eval config (defaults match the SFT ablation script).
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PARQUET="${PARQUET:-data/chess_puzzles/test.parquet}"
K_MAX="${K_MAX:-32}"
SEED="${SEED:-0}"
SEED_MODE="${SEED_MODE:-engine}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PRINT_PROMPT_EXAMPLES="${PRINT_PROMPT_EXAMPLES:-1}"
LIMIT="${LIMIT:-}"

OUT_JSON="${OUT_JSON:-${PROJECT_ROOT%/}/chess_rl_outputs/passk_eval_${SLURM_JOB_ID:-$$}.json}"
mkdir -p "$(dirname "${OUT_JSON}")"

echo "[SLURM] MODEL=${MODEL}"
echo "[SLURM] PARQUET=${PARQUET}"
echo "[SLURM] K_MAX=${K_MAX} SEED=${SEED} SEED_MODE=${SEED_MODE} TEMP=${TEMPERATURE} TOP_P=${TOP_P}"
echo "[SLURM] BATCH_SIZE=${BATCH_SIZE} MAX_NUM_SEQS=${MAX_NUM_SEQS} TP=${TENSOR_PARALLEL_SIZE}"
echo "[SLURM] MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-<unset>}"
echo "[SLURM] OUT_JSON=${OUT_JSON}"

# Use an absolute python path to avoid inheriting a host/conda python via PATH.
LIMIT_ARG=""
if [ -n "${LIMIT}" ]; then
  LIMIT_ARG=" --limit \"${LIMIT}\""
fi
MAX_NUM_BATCHED_TOKENS_ARG=""
if [ -n "${MAX_NUM_BATCHED_TOKENS}" ]; then
  MAX_NUM_BATCHED_TOKENS_ARG=" --max_num_batched_tokens \"${MAX_NUM_BATCHED_TOKENS}\""
fi
CMD="unset PYTHONHOME PYTHONPATH; cd \"${CONTAINER_REPO_DIR}\" && /usr/bin/python3 -c \"import numpy as np; import torch; import sys; print('sanity', np.__version__, torch.__version__, sys.executable)\" && /usr/bin/python3 -m scripts.eval_chess_passk --model \"${MODEL}\" --parquet \"${PARQUET}\" --k_max \"${K_MAX}\" --do_sample --seed \"${SEED}\" --seed_mode \"${SEED_MODE}\" --temperature \"${TEMPERATURE}\" --top_p \"${TOP_P}\" --batch_size \"${BATCH_SIZE}\" --max_num_seqs \"${MAX_NUM_SEQS}\"${MAX_NUM_BATCHED_TOKENS_ARG} --max_prompt_length \"${MAX_PROMPT_LENGTH}\" --max_response_length \"${MAX_RESPONSE_LENGTH}\" --max_model_len \"${MAX_MODEL_LEN}\" --gpu_memory_utilization \"${GPU_MEMORY_UTILIZATION}\" --tensor_parallel_size \"${TENSOR_PARALLEL_SIZE}\" --print_prompt_examples \"${PRINT_PROMPT_EXAMPLES}\"${LIMIT_ARG} --out_json \"${OUT_JSON}\""
echo "[SLURM] Container CMD: ${CMD}"

if command -v srun >/dev/null 2>&1; then
  srun --export=ALL \
    apptainer exec --nv \
      --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
      --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
      --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
      "${SIF_IMAGE}" \
      bash -c "${CMD}"
else
  apptainer exec --nv \
    --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
    "${SIF_IMAGE}" \
    bash -c "${CMD}"
fi

echo "[SLURM] Done: ${OUT_JSON}"
