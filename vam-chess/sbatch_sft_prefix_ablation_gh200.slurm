#!/bin/bash
#SBATCH --job-name=chess-sft-prefix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/slurm-%x-%A_%a.out
#SBATCH --error=slurm/slurm-%x-%A_%a.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL
#SBATCH --array=0-4

# SFT forced-prefix ablation on GH200 / Isambard (Apptainer).
#
# This runs the 5 SFT settings (A–E) in parallel via a Slurm job array.
# Each task:
#   - builds an SFT parquet variant from `data/chess_puzzles/train_hard.parquet`
#   - trains 2 epochs with `verl/trainer/fsdp_sft_trainer.py`
#   - evaluates pass@k (default: k=32) on `data/chess_puzzles/test.parquet` using vLLM
#   - writes `${OUT_DIR}/results.json` for aggregation
#
# Submit:
#   sbatch ./sbatch_sft_prefix_ablation_gh200.slurm
#
# Useful overrides:
#   # Smoke (tiny model + tiny k + limits):
#   sbatch --array=0 --export=ALL,MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct,EVAL_K=2,EVAL_LIMIT=50,TRAIN_LIMIT=64 ./sbatch_sft_prefix_ablation_gh200.slurm
#
#   # Full run (default model Qwen2.5-7B):
#   sbatch ./sbatch_sft_prefix_ablation_gh200.slurm

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
echo "[SLURM] Array ID:    ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "[SLURM] Task ID:     ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "[SLURM] Node list:   ${SLURM_JOB_NODELIST:-N/A}"
echo "[SLURM] GPUs:        ${SLURM_GPUS_ON_NODE:-1} (requested via --gres=gpu:1)"
echo "[SLURM] Submit dir:  ${SLURM_SUBMIT_DIR:-$(pwd)}"

# Run relative to this script's directory (repo root) so relative paths work.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  THIS_DIR="${SLURM_SUBMIT_DIR}"
else
  THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${THIS_DIR}"
echo "[SLURM] Repo dir:    ${THIS_DIR}"

# This launcher targets Arm64 / aarch64 GH200 nodes.
HOST_ARCH="$(uname -m 2>/dev/null || echo unknown)"
echo "[SLURM] Host arch:   ${HOST_ARCH}"
if [ "${HOST_ARCH}" != "aarch64" ] && [ "${HOST_ARCH}" != "arm64" ]; then
  die "Expected arm64/aarch64 compute node, got '${HOST_ARCH}'."
fi

# Variant selection (A–E) via job array.
VARIANTS=(A B C D E)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${#VARIANTS[@]}" ]; then
  die "Invalid SLURM_ARRAY_TASK_ID='${TASK_ID}' (expected 0..$(( ${#VARIANTS[@]} - 1 )))."
fi
VARIANT="${CHESS_SFT_VARIANT:-${VARIANTS[${TASK_ID}]}}"
echo "[SLURM] VARIANT=${VARIANT}"

# Avoid torchrun rendezvous port collisions when multiple array tasks land on the same node.
# - torchrun defaults to port 29500; with 1 GPU/task it's common for Slurm to pack tasks.
# - Use an array+task derived port in a safe ephemeral range.
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + (ARRAY_ID % 4000) * 10 + TASK_ID))}"
echo "[SLURM] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

# Container image (prebuilt SIF is preferred; building on compute nodes can be flaky).
SIF_IMAGE="${CHESS_RL_SIF_IMAGE:-${HOME}/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif}"
CONTAINER_REPO_DIR="${CHESS_RL_CONTAINER_REPO_DIR:-/workspace/chess_rl}"
if [ ! -f "${SIF_IMAGE}" ]; then
  cat >&2 <<EOF
[ERROR] Missing SIF image: ${SIF_IMAGE}

Build it once on a login node (recommended), then re-run:
  apptainer build "${SIF_IMAGE}" docker://gabr1e1/chess_rl:v1-arm-stockfish-flashinfer
EOF
  exit 1
fi
echo "[SLURM] SIF_IMAGE=${SIF_IMAGE}"
echo "[SLURM] Repo mount: ${CONTAINER_REPO_DIR}"

# Persistent output root (avoid $HOME quota).
PROJECT_ROOT="${CHESS_RL_PROJECT_ROOT:-/projects/a5l/ziyan}"
RUN_ID="${CHESS_SFT_RUN_ID:-sft_prefix_ablation_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$$}}}"
OUT_ROOT="${CHESS_SFT_OUT_ROOT:-${PROJECT_ROOT%/}/chess_rl_outputs/${RUN_ID}}"
OUT_DIR="${OUT_ROOT%/}/${VARIANT}"
mkdir -p "${OUT_DIR}"
echo "[SLURM] OUT_DIR=${OUT_DIR}"

# Share HF caches across jobs to avoid downloading the base model 5x.
HF_CACHE_ROOT="${CHESS_RL_HF_CACHE_ROOT:-${PROJECT_ROOT%/}/hf_cache}"
mkdir -p "${HF_CACHE_ROOT}"
export HF_HOME="${HF_CACHE_ROOT%/}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"
echo "[SLURM] HF_HOME=${HF_HOME}"

# Runtime knobs (override at submit time via --export=ALL,...)
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export TRAIN_IN_PARQUET="${TRAIN_IN_PARQUET:-data/chess_puzzles/train_hard.parquet}"
export TEST_IN_PARQUET="${TEST_IN_PARQUET:-data/chess_puzzles/test.parquet}"

export OUT_ROOT
export OUT_DIR

# Use single-GPU SFT + eval by default on GH200.
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export EPOCHS="${EPOCHS:-2}"

export EVAL_K="${EVAL_K:-32}"
export EVAL_SEED="${EVAL_SEED:-0}"
export EVAL_SEED_MODE="${EVAL_SEED_MODE:-engine}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-1024}"
export EVAL_MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-2000}"
export EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-4096}"
export EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.8}"
export EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-1}"
export EVAL_EVERY_FRAC="${EVAL_EVERY_FRAC:-0.25}"

echo "[SLURM] MODEL_PATH=${MODEL_PATH}"
echo "[SLURM] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU} MAX_LENGTH=${MAX_LENGTH} EPOCHS=${EPOCHS}"
echo "[SLURM] EVAL_K=${EVAL_K} EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE} EVAL_TENSOR_PARALLEL_SIZE=${EVAL_TENSOR_PARALLEL_SIZE}"
echo "[SLURM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

mkdir -p slurm

CMD="cd \"${CONTAINER_REPO_DIR}\" && bash scripts/run_chess_sft_prefix_ablation_one.sh \"${VARIANT}\""
echo "[SLURM] Container CMD: ${CMD}"

if command -v srun >/dev/null 2>&1; then
  srun --export=ALL \
    apptainer exec --nv \
      --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
      --bind "${OUT_ROOT}:${OUT_ROOT}" \
      --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
      "${SIF_IMAGE}" \
      bash -lc "${CMD}"
else
  apptainer exec --nv \
    --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
    --bind "${OUT_ROOT}:${OUT_ROOT}" \
    --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
    "${SIF_IMAGE}" \
    bash -lc "${CMD}"
fi

echo "[SLURM] Done: variant=${VARIANT} out_dir=${OUT_DIR}"
