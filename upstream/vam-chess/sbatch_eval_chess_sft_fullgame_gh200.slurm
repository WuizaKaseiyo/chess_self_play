#!/bin/bash
#SBATCH --job-name=chess-sft-fullgame-eval
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

# Run full-game eval (vLLM + Stockfish ACPL) from an existing SFT checkpoint.
#
# Required env:
#   - CHESS_SFT_RUN_ID: run directory key under /projects/.../chess_rl_outputs/
#     e.g., sft_reject_req_full_20260206_202325_qwen2_5_7b_instruct
#
# Optional env:
#   - RUN_NAME: SFT run subdir under sft_train/ (default: rejection_sft)
#   - CKPT_STEP: global step checkpoint to evaluate (default: latest checkpoint)
#   - CKPT_HF_DIR: explicit HF checkpoint dir; overrides CHESS_SFT_RUN_ID/RUN_NAME/CKPT_STEP
#   - FULLGAME_OUT_DIR: output directory (default: <run>/fullgame_eval)
#   - OVERWRITE_FULLGAME_DIR=1 to archive existing FULLGAME_OUT_DIR before rerun
#
# Submit example (blocking):
#   sbatch --wait --export=ALL,CHESS_SFT_RUN_ID=...,RUN_NAME=rejection_sft_qwen2_5_7b_instruct \
#     ./sbatch_eval_chess_sft_fullgame_gh200.slurm
#
# Submit example (non-blocking):
#   sbatch --export=ALL,CHESS_SFT_RUN_ID=...,RUN_NAME=rejection_sft_qwen2_5_7b_instruct \
#     ./sbatch_eval_chess_sft_fullgame_gh200.slurm

set -euo pipefail

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

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

mkdir -p slurm
echo "[SLURM] Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "[SLURM] Node list:   ${SLURM_JOB_NODELIST:-N/A}"
echo "[SLURM] GPUs:        ${SLURM_GPUS_ON_NODE:-4} (requested via --gres=gpu:4)"
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
  die "Expected arm64/aarch64 compute node, got '${HOST_ARCH}'."
fi

PROJECT_ROOT="${CHESS_RL_PROJECT_ROOT:-/projects/a5l/ziyan}"
RUN_ID="${CHESS_SFT_RUN_ID:-}"
RUN_NAME="${RUN_NAME:-rejection_sft}"
CKPT_STEP="${CKPT_STEP:-}"
CKPT_HF_DIR="${CKPT_HF_DIR:-}"

if [ -z "${CKPT_HF_DIR}" ]; then
  [ -n "${RUN_ID}" ] || die "Set CHESS_SFT_RUN_ID or CKPT_HF_DIR."
  RUN_ROOT="${PROJECT_ROOT%/}/chess_rl_outputs/${RUN_ID}/sft_train/${RUN_NAME}"
  CKPT_DIR="${RUN_ROOT}/checkpoints"
  [ -d "${CKPT_DIR}" ] || die "Missing checkpoint directory: ${CKPT_DIR}"

  if [ -n "${CKPT_STEP}" ]; then
    CKPT_HF_DIR="${CKPT_DIR}/global_step_${CKPT_STEP}/huggingface"
  else
    CKPT_HF_DIR="$(ls -1d "${CKPT_DIR}"/global_step_*/huggingface 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
fi

[ -d "${CKPT_HF_DIR}" ] || die "Missing HF checkpoint directory: ${CKPT_HF_DIR}"

FULLGAME_OUT_DIR="${FULLGAME_OUT_DIR:-$(dirname "${CKPT_HF_DIR}")/../../fullgame_eval}"

OVERWRITE_FULLGAME_DIR="${OVERWRITE_FULLGAME_DIR:-0}"
if [ -d "${FULLGAME_OUT_DIR}" ] && [ "${OVERWRITE_FULLGAME_DIR}" = "1" ]; then
  mv "${FULLGAME_OUT_DIR}" "${FULLGAME_OUT_DIR}.failed_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${FULLGAME_OUT_DIR}"

# Full-game eval defaults (throughput-first; aligned with current SFT runs).
FULLGAME_PROMPT_TEMPLATE_PATH="${FULLGAME_PROMPT_TEMPLATE_PATH:-recipe/chess/prompt_templates/original_chessr1_prompt.jinja}"
FULLGAME_TP="${FULLGAME_TP:-4}"
FULLGAME_GPU_MEMORY_UTILIZATION="${FULLGAME_GPU_MEMORY_UTILIZATION:-0.9}"
FULLGAME_MAX_MODEL_LEN="${FULLGAME_MAX_MODEL_LEN:-4096}"
FULLGAME_MAX_RESPONSE_TOKENS="${FULLGAME_MAX_RESPONSE_TOKENS:-2000}"
FULLGAME_MAX_NUM_SEQS="${FULLGAME_MAX_NUM_SEQS:-1024}"
FULLGAME_MAX_NUM_BATCHED_TOKENS="${FULLGAME_MAX_NUM_BATCHED_TOKENS:-65536}"
FULLGAME_MAX_RETRIES_PER_TURN="${FULLGAME_MAX_RETRIES_PER_TURN:-1}"
FULLGAME_GAMES_PER_DEPTH="${FULLGAME_GAMES_PER_DEPTH:-50}"
FULLGAME_TEMPERATURE="${FULLGAME_TEMPERATURE:-0.6}"
FULLGAME_TOP_P="${FULLGAME_TOP_P:-0.9}"
FULLGAME_SEED="${FULLGAME_SEED:-0}"
FULLGAME_ACPL_CP_CAP="${FULLGAME_ACPL_CP_CAP:-5000}"
FULLGAME_MATE_SCORE_CP="${FULLGAME_MATE_SCORE_CP:-5000}"
FULLGAME_ACPL_WORKERS="${FULLGAME_ACPL_WORKERS:-${SLURM_CPUS_PER_TASK:-48}}"
FULLGAME_ACPL_THREADS="${FULLGAME_ACPL_THREADS:-2}"
FULLGAME_STOCKFISH_PATH="${FULLGAME_STOCKFISH_PATH:-/usr/local/bin/stockfish}"

echo "[CFG] CKPT_HF_DIR=${CKPT_HF_DIR}"
echo "[CFG] FULLGAME_OUT_DIR=${FULLGAME_OUT_DIR}"
echo "[CFG] FULLGAME_TP=${FULLGAME_TP} FULLGAME_GPU_MEMORY_UTILIZATION=${FULLGAME_GPU_MEMORY_UTILIZATION}"
echo "[CFG] FULLGAME_MAX_NUM_SEQS=${FULLGAME_MAX_NUM_SEQS} FULLGAME_MAX_NUM_BATCHED_TOKENS=${FULLGAME_MAX_NUM_BATCHED_TOKENS}"
echo "[CFG] FULLGAME_ACPL_WORKERS=${FULLGAME_ACPL_WORKERS} FULLGAME_ACPL_THREADS=${FULLGAME_ACPL_THREADS}"
echo "[CFG] FULLGAME_STOCKFISH_PATH=${FULLGAME_STOCKFISH_PATH}"

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

# Share HF caches across jobs.
HF_CACHE_ROOT="${CHESS_RL_HF_CACHE_ROOT:-${PROJECT_ROOT%/}/hf_cache}"
mkdir -p "${HF_CACHE_ROOT}"
export HF_HOME="${HF_CACHE_ROOT%/}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"
export TOKENIZERS_PARALLELISM=false

apptainer exec --nv \
  --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
  "${SIF_IMAGE}" \
  bash -lc '
    set -euo pipefail
    cd "'"${CONTAINER_REPO_DIR}"'"
    python3 scripts/eval_chess_fullgame.py \
      --model "'"${CKPT_HF_DIR}"'" \
      --out-dir "'"${FULLGAME_OUT_DIR}"'" \
      --prompt-template-path "'"${FULLGAME_PROMPT_TEMPLATE_PATH}"'" \
      --tensor-parallel-size "'"${FULLGAME_TP}"'" \
      --gpu-memory-utilization "'"${FULLGAME_GPU_MEMORY_UTILIZATION}"'" \
      --max-model-len "'"${FULLGAME_MAX_MODEL_LEN}"'" \
      --max-response-tokens "'"${FULLGAME_MAX_RESPONSE_TOKENS}"'" \
      --max-num-seqs "'"${FULLGAME_MAX_NUM_SEQS}"'" \
      --max-num-batched-tokens "'"${FULLGAME_MAX_NUM_BATCHED_TOKENS}"'" \
      --max-retries-per-turn "'"${FULLGAME_MAX_RETRIES_PER_TURN}"'" \
      --acpl-workers "'"${FULLGAME_ACPL_WORKERS}"'" \
      --acpl-threads "'"${FULLGAME_ACPL_THREADS}"'" \
      --acpl-cp-cap "'"${FULLGAME_ACPL_CP_CAP}"'" \
      --mate-score-cp "'"${FULLGAME_MATE_SCORE_CP}"'" \
      --games-per-depth "'"${FULLGAME_GAMES_PER_DEPTH}"'" \
      --stockfish-path "'"${FULLGAME_STOCKFISH_PATH}"'" \
      --temperature "'"${FULLGAME_TEMPERATURE}"'" --top-p "'"${FULLGAME_TOP_P}"'" \
      --seed "'"${FULLGAME_SEED}"'" \
      --trust-remote-code
  '

echo "[DONE] Full-game eval outputs under:"
echo "  ${FULLGAME_OUT_DIR}"
