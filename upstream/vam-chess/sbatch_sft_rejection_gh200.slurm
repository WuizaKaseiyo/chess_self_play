#!/bin/bash
#SBATCH --job-name=chess-sft-reject-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/slurm-%x-%A.out
#SBATCH --error=slurm/slurm-%x-%A.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

# Train 1 epoch of SFT on the rejection-sampled dataset, then run:
#   - pass@1 exact-match eval on baseline test parquets
#   - full-game eval vs Stockfish
#
# Required env:
#   - CHESS_SFT_RUN_ID: output run id under /projects/a5l/ziyan/chess_rl_outputs/
#
# Optional env:
#   - CHESS_SFT_MODE: {smoke|full} (default smoke)
#   - MODEL_PATH: base model HF id/path (default depends on mode)
#   - TRAIN_SFT_PARQUET: path to the rejection-sampled parquet
#
# Submit example (blocks):
#   sbatch --wait --export=ALL,CHESS_SFT_RUN_ID=rs_smoke_...,CHESS_SFT_MODE=smoke,MODEL_PATH=Qwen/Qwen2.5-3B-Instruct sbatch_sft_rejection_gh200.slurm

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
echo "[SLURM] Node list:   ${SLURM_JOB_NODELIST:-N/A}"
echo "[SLURM] GPUs:        ${SLURM_GPUS_ON_NODE:-4} (requested via --gres=gpu:4)"
echo "[SLURM] Submit dir:  ${SLURM_SUBMIT_DIR:-$(pwd)}"

# Run relative to submit dir (repo root).
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
  die "Expected arm64/aarch64 compute node, got '${HOST_ARCH}'."
fi

RUN_ID="${CHESS_SFT_RUN_ID:-}"
if [ -z "${RUN_ID}" ]; then
  die "CHESS_SFT_RUN_ID is required"
fi

MODE="${CHESS_SFT_MODE:-smoke}"
case "${MODE}" in
  smoke|full) ;;
  *) die "CHESS_SFT_MODE must be one of {smoke,full}, got '${MODE}'" ;;
esac

if [ "${MODE}" == "smoke" ]; then
  export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
  export EPOCHS="${EPOCHS:-1}"
  export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
  export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
  export MAX_LENGTH="${MAX_LENGTH:-3584}"
  export EVAL_LIMIT="${EVAL_LIMIT:-}"
  export FULLGAME_SMOKE="${FULLGAME_SMOKE:-1}"
  export FULLGAME_MAX_RESPONSE_TOKENS="${FULLGAME_MAX_RESPONSE_TOKENS:-2000}"
else
  export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
  export EPOCHS="${EPOCHS:-2}"
  export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
  export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
  export MAX_LENGTH="${MAX_LENGTH:-3584}"
  export EVAL_LIMIT="${EVAL_LIMIT:-}"
  export FULLGAME_SMOKE="${FULLGAME_SMOKE:-0}"
  export FULLGAME_GAMES_PER_DEPTH="${FULLGAME_GAMES_PER_DEPTH:-50}"
  export FULLGAME_MAX_RESPONSE_TOKENS="${FULLGAME_MAX_RESPONSE_TOKENS:-2000}"
  export FULLGAME_ACPL_CP_CAP="${FULLGAME_ACPL_CP_CAP:-5000}"
  export FULLGAME_MATE_SCORE_CP="${FULLGAME_MATE_SCORE_CP:-5000}"
fi

# Throughput defaults for puzzle eval and full-game eval.
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-1024}"
export EVAL_MAX_NUM_BATCHED_TOKENS="${EVAL_MAX_NUM_BATCHED_TOKENS:-65536}"
export FULLGAME_MAX_NUM_SEQS="${FULLGAME_MAX_NUM_SEQS:-1024}"
export FULLGAME_MAX_NUM_BATCHED_TOKENS="${FULLGAME_MAX_NUM_BATCHED_TOKENS:-65536}"
export FULLGAME_MAX_RETRIES_PER_TURN="${FULLGAME_MAX_RETRIES_PER_TURN:-1}"
export FULLGAME_ACPL_WORKERS="${FULLGAME_ACPL_WORKERS:-${SLURM_CPUS_PER_TASK:-48}}"
export FULLGAME_ACPL_THREADS="${FULLGAME_ACPL_THREADS:-2}"

# Upload defaults: automatically publish final checkpoint artifacts to HF.
export HF_UPLOAD_ENABLE="${HF_UPLOAD_ENABLE:-1}"
export HF_UPLOAD_REQUIRED="${HF_UPLOAD_REQUIRED:-1}"
export HF_UPLOAD_REPO_ID="${HF_UPLOAD_REPO_ID:-Gabr1e11/a_lot_of_models}"
export HF_UPLOAD_TOKEN_PATH="${HF_UPLOAD_TOKEN_PATH:-${HOME}/.huggingface_token_zzc}"
if [ "${HF_UPLOAD_ENABLE}" = "1" ] && [ ! -f "${HF_UPLOAD_TOKEN_PATH}" ]; then
  die "HF upload enabled but token file missing: ${HF_UPLOAD_TOKEN_PATH}"
fi

sanitize_model_tag() {
  local m="$1"
  local base
  base="$(basename "${m}")"
  echo "${base}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g'
}

MODEL_TAG="$(sanitize_model_tag "${MODEL_PATH}")"
DATASET_STEM="train_sft__${MODEL_TAG}"
echo "[CFG] MODEL_PATH=${MODEL_PATH}"
echo "[CFG] MODEL_TAG=${MODEL_TAG}"
echo "[CFG] EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE} EVAL_MAX_NUM_SEQS=${EVAL_MAX_NUM_SEQS} EVAL_MAX_NUM_BATCHED_TOKENS=${EVAL_MAX_NUM_BATCHED_TOKENS}"
echo "[CFG] FULLGAME_MAX_NUM_SEQS=${FULLGAME_MAX_NUM_SEQS} FULLGAME_MAX_NUM_BATCHED_TOKENS=${FULLGAME_MAX_NUM_BATCHED_TOKENS} FULLGAME_MAX_RETRIES_PER_TURN=${FULLGAME_MAX_RETRIES_PER_TURN}"
echo "[CFG] FULLGAME_ACPL_WORKERS=${FULLGAME_ACPL_WORKERS} FULLGAME_ACPL_THREADS=${FULLGAME_ACPL_THREADS}"
echo "[CFG] HF_UPLOAD_ENABLE=${HF_UPLOAD_ENABLE} HF_UPLOAD_REQUIRED=${HF_UPLOAD_REQUIRED} HF_UPLOAD_REPO_ID=${HF_UPLOAD_REPO_ID}"
echo "[CFG] HF_UPLOAD_TOKEN_PATH=${HF_UPLOAD_TOKEN_PATH}"

PROJECT_ROOT="${CHESS_RL_PROJECT_ROOT:-/projects/a5l/ziyan}"
OUT_ROOT="${CHESS_SFT_OUT_ROOT:-${PROJECT_ROOT%/}/chess_rl_outputs/${RUN_ID}}"
mkdir -p "${OUT_ROOT}"
echo "[SLURM] OUT_ROOT=${OUT_ROOT}"

# Default dataset path (output of sbatch_build_chess_sft_rejection_dataset_gh200.slurm).
export TRAIN_SFT_PARQUET="${TRAIN_SFT_PARQUET:-${OUT_ROOT%/}/rejection_sft_dataset/${DATASET_STEM}.parquet}"

RUN_NAME="${RUN_NAME:-rejection_sft}"
export OUT_DIR="${OUT_DIR:-${OUT_ROOT%/}/sft_train/${RUN_NAME}}"
mkdir -p "${OUT_DIR}"

# Container image.
SIF_IMAGE="${CHESS_RL_SIF_IMAGE:-${HOME}/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif}"
CONTAINER_REPO_DIR="${CHESS_RL_CONTAINER_REPO_DIR:-/workspace/chess_rl}"
if [ ! -f "${SIF_IMAGE}" ]; then
  cat >&2 <<EOF
[ERROR] Missing SIF image: ${SIF_IMAGE}

Build it once on a login node, then re-run:
  apptainer build "${SIF_IMAGE}" docker://gabr1e1/chess_rl:v1-arm-stockfish-flashinfer
EOF
  exit 1
fi
echo "[SLURM] SIF_IMAGE=${SIF_IMAGE}"

# Share HF caches across jobs (avoid repeated downloads; avoid $HOME quota).
HF_CACHE_ROOT="${CHESS_RL_HF_CACHE_ROOT:-${PROJECT_ROOT%/}/hf_cache}"
mkdir -p "${HF_CACHE_ROOT}"
export HF_HOME="${HF_CACHE_ROOT%/}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

# Use all 4 GPUs for SFT DP and for vLLM eval.
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-4}"
export FULLGAME_TP="${FULLGAME_TP:-4}"

# Avoid torchrun rendezvous collisions across jobs.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID % 4000) * 10))}"

export TOKENIZERS_PARALLELISM=false

apptainer exec --nv \
  --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
  --bind "${OUT_ROOT}:${OUT_ROOT}" \
  --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
  "${SIF_IMAGE}" \
  bash -lc '
    set -euo pipefail
    cd "'"${CONTAINER_REPO_DIR}"'"

    export MODEL_PATH="'"${MODEL_PATH}"'"
    export TRAIN_SFT_PARQUET="'"${TRAIN_SFT_PARQUET}"'"
    export TEST_PARQUET="data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet"
    export TEST_SHUFFLED_PARQUET="data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet"

    export OUT_DIR="'"${OUT_DIR}"'"
    export RUN_NAME="'"${RUN_NAME}"'"

    bash scripts/run_chess_sft_rejection_one.sh
  '

echo "[DONE] SFT+eval outputs under:"
echo "  ${OUT_DIR}"
