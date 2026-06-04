#!/bin/bash
#SBATCH --job-name=chess-sft-sampled-one
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

# Run a *single* sampled-move SFT variant (offline or online) on GH200 / Isambard.
#
# Required env:
#   - CHESS_SFT_RUN_ID: output run id under /projects/a5l/ziyan/chess_rl_outputs/
#   - CHESS_SFT_MODE: {smoke|full}
#   - CHESS_SFT_LETTER: {A|B|C|E}
#   - CHESS_SFT_KIND: {offline|online}
#   - CHESS_SFT_WEIGHTING: {awr|uniform|best_only}
#   - (offline only) CHESS_SFT_ORDERING: {shuffle|no_shuffle}
#
# This launcher runs inside the repo's Apptainer image and calls:
#   - offline: scripts/run_chess_sft_sampled_move_offline_one.sh
#   - online:  scripts/run_chess_sft_sampled_move_online_one.sh
#
# Notes:
# - Uses the same output contract as the grouped launcher, but each Slurm job handles only one variant.
# - This is designed for "fan-out" parallelism across many nodes.

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

MODE="${CHESS_SFT_MODE:-full}"
case "${MODE}" in
  smoke|full) ;;
  *) die "CHESS_SFT_MODE must be one of {smoke,full}, got '${MODE}'" ;;
esac

LETTER="${CHESS_SFT_LETTER:-A}"
case "${LETTER}" in
  A|B|C|E) ;;
  *) die "CHESS_SFT_LETTER must be one of {A,B,C,E}, got '${LETTER}'" ;;
esac

KIND="${CHESS_SFT_KIND:-offline}"
case "${KIND}" in
  offline|online) ;;
  *) die "CHESS_SFT_KIND must be one of {offline,online}, got '${KIND}'" ;;
esac

WEIGHTING="${CHESS_SFT_WEIGHTING:-}"
case "${WEIGHTING}" in
  awr|uniform|best_only) ;;
  *) die "CHESS_SFT_WEIGHTING must be one of {awr,uniform,best_only}, got '${WEIGHTING}'" ;;
esac

ORDERING="${CHESS_SFT_ORDERING:-}"
if [ "${KIND}" == "offline" ]; then
  case "${ORDERING}" in
    shuffle|no_shuffle) ;;
    *) die "CHESS_SFT_ORDERING must be one of {shuffle,no_shuffle} for offline, got '${ORDERING}'" ;;
  esac
fi

weight_suffix=""
case "${WEIGHTING}" in
  awr) weight_suffix="wAWR" ;;
  uniform) weight_suffix="wUniform" ;;
  best_only) weight_suffix="wBestOnly" ;;
esac

RUN_NAME=""
if [ "${KIND}" == "offline" ]; then
  RUN_NAME="${LETTER}_sample_${ORDERING}_${weight_suffix}"
else
  RUN_NAME="${LETTER}_online_${weight_suffix}"
fi
echo "[SLURM] RUN_NAME=${RUN_NAME}"

PROJECT_ROOT="${CHESS_RL_PROJECT_ROOT:-/projects/a5l/ziyan}"
OUT_ROOT="${CHESS_SFT_OUT_ROOT:-${PROJECT_ROOT%/}/chess_rl_outputs/${RUN_ID}}"
mkdir -p "${OUT_ROOT}"
echo "[SLURM] OUT_ROOT=${OUT_ROOT}"

# Fast skip if the run already completed.
if [ -f "${OUT_ROOT}/${RUN_NAME}/results.json" ]; then
  echo "[SLURM] Found existing results.json; skipping: ${OUT_ROOT}/${RUN_NAME}/results.json"
  exit 0
fi

# Container image.
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

# Share HF caches across jobs (avoid repeated downloads; avoid $HOME quota).
HF_CACHE_ROOT="${CHESS_RL_HF_CACHE_ROOT:-${PROJECT_ROOT%/}/hf_cache}"
mkdir -p "${HF_CACHE_ROOT}"
export HF_HOME="${HF_CACHE_ROOT%/}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

export TRAIN_IN_PARQUET="${TRAIN_IN_PARQUET:-data/chess_puzzles/train_hard.parquet}"
export TEST_IN_PARQUET="${TEST_IN_PARQUET:-data/chess_puzzles/test.parquet}"

# Use 4 GPUs for each run (training + vLLM eval).
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-4}"

# Avoid torchrun rendezvous collisions across jobs.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID % 4000) * 10))}"

if [ "${MODE}" == "smoke" ]; then
  export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
  # Qwen2.5-0.5B has 14 heads, so TP=2 works, TP=4 fails.
  export EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-2}"
  export EPOCHS="${EPOCHS:-1}"
  export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
  export TRAIN_LIMIT="${TRAIN_LIMIT:-32}"
  export EVAL_K="${EVAL_K:-2}"
  export EVAL_LIMIT="${EVAL_LIMIT:-20}"
  export MAX_LENGTH="${MAX_LENGTH:-512}"
  export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
  export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
  export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
  export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-256}"
  export EVAL_MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-256}"
  export EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-1024}"
  export MOVE_TEMPERATURE="${MOVE_TEMPERATURE:-2.0}"
  export MOVE_SEED="${MOVE_SEED:-0}"
  export SFT_AWR_BETA="${SFT_AWR_BETA:-2.0}"
  export ONLINE_TOTAL_STEPS="${ONLINE_TOTAL_STEPS:-2}"
  export ONLINE_EPOCHS="${ONLINE_EPOCHS:-1}"
  export ONLINE_PROMPTS_PER_STEP="${ONLINE_PROMPTS_PER_STEP:-128}"
  export ONLINE_MOVES_PER_PROMPT="${ONLINE_MOVES_PER_PROMPT:-8}"
  export ONLINE_GEN_MAX_NEW_TOKENS="${ONLINE_GEN_MAX_NEW_TOKENS:-128}"
  export ONLINE_GEN_DO_SAMPLE="${ONLINE_GEN_DO_SAMPLE:-0}"
  export ONLINE_TRAIN_MAX_LENGTH="${ONLINE_TRAIN_MAX_LENGTH:-512}"
  export ONLINE_MICRO_BATCH_SIZE="${ONLINE_MICRO_BATCH_SIZE:-8}"
else
  export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
  export EPOCHS="${EPOCHS:-2}"
  export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
  export TRAIN_LIMIT="${TRAIN_LIMIT:-}"
  export EVAL_K="${EVAL_K:-32}"
  export EVAL_LIMIT="${EVAL_LIMIT:-}"
  export MAX_LENGTH="${MAX_LENGTH:-1024}"
  export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
  export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
  export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
  export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-1024}"
  export EVAL_MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-2000}"
  export EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-4096}"
  export MOVE_TEMPERATURE="${MOVE_TEMPERATURE:-2.0}"
  export MOVE_SEED="${MOVE_SEED:-0}"
  export SFT_AWR_BETA="${SFT_AWR_BETA:-2.0}"
  export ONLINE_TOTAL_STEPS="${ONLINE_TOTAL_STEPS:-}"
  export ONLINE_EPOCHS="${ONLINE_EPOCHS:-2}"
  export ONLINE_PROMPTS_PER_STEP="${ONLINE_PROMPTS_PER_STEP:-128}"
  export ONLINE_MOVES_PER_PROMPT="${ONLINE_MOVES_PER_PROMPT:-8}"
  export ONLINE_GEN_MAX_NEW_TOKENS="${ONLINE_GEN_MAX_NEW_TOKENS:-256}"
  export ONLINE_GEN_DO_SAMPLE="${ONLINE_GEN_DO_SAMPLE:-0}"
  # Online loop uses HF `generate()` before SFT; keep the rollout batch conservative to avoid OOM.
  export ONLINE_GEN_BATCH_SIZE="${ONLINE_GEN_BATCH_SIZE:-8}"
  export ONLINE_TRAIN_MAX_LENGTH="${ONLINE_TRAIN_MAX_LENGTH:-1024}"
  export ONLINE_MICRO_BATCH_SIZE="${ONLINE_MICRO_BATCH_SIZE:-8}"
fi

mkdir -p slurm

CMD=""
if [ "${KIND}" == "offline" ]; then
  CMD="cd \"${CONTAINER_REPO_DIR}\" && OUT_ROOT=\"${OUT_ROOT}\" bash scripts/run_chess_sft_sampled_move_offline_one.sh \"${LETTER}\" \"${ORDERING}\" \"${WEIGHTING}\""
else
  CMD="cd \"${CONTAINER_REPO_DIR}\" && OUT_ROOT=\"${OUT_ROOT}\" bash scripts/run_chess_sft_sampled_move_online_one.sh \"${LETTER}\" \"${WEIGHTING}\""
fi

echo "[SLURM] Running: kind=${KIND} letter=${LETTER} ordering=${ORDERING:-NA} weighting=${WEIGHTING} mode=${MODE}"
echo "[SLURM] Container CMD: ${CMD}"

apptainer exec --nv \
  --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
  --bind "${OUT_ROOT}:${OUT_ROOT}" \
  --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
  "${SIF_IMAGE}" \
  bash -lc "${CMD}"

echo "[SLURM] Done: ${RUN_NAME} out_root=${OUT_ROOT}"
