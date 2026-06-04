#!/bin/bash
#SBATCH --job-name=chess-rr-acpl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=144
#SBATCH --time=0-10:00:00
#SBATCH --output=slurm/slurm-%x-%j.out
#SBATCH --error=slurm/slurm-%x-%j.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

# Round-robin pilot (ACPL stage):
# - Reads PGNs from a completed infer-stage directory.
# - Computes starter-kit-aligned ACPL/accuracy with Stockfish (parallel workers).
#
# NOTE: Isambard may not allow CPU-only jobs; we request 1 GPU but do not use it.
#
# Example:
#   sbatch --wait --export=ALL,INFER_OUT_DIR=/projects/.../outputs/full_game_eval_rr_pilot/job_<id> ./sbatch_eval_chess_fullgame_round_robin_pilot_acpl_gh200.slurm

set -euo pipefail

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

mkdir -p slurm

echo "[SLURM] Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "[SLURM] Job name:    ${SLURM_JOB_NAME:-chess-rr-acpl}"
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

SIF_IMAGE="${CHESS_RL_SIF_IMAGE:-/projects/a5l/ziyan/sif-images/chess_rl_v4-arm-stockfish-pytorch-25.05.sif}"
CONTAINER_REPO_DIR="${CHESS_RL_CONTAINER_REPO_DIR:-/workspace/chess_rl}"
if [ ! -f "${SIF_IMAGE}" ]; then
  die "Missing SIF image: ${SIF_IMAGE}"
fi
echo "[SLURM] SIF_IMAGE=${SIF_IMAGE}"
echo "[SLURM] Repo mount: ${CONTAINER_REPO_DIR}"

PROJECT_ROOT="${CHESS_RL_PROJECT_ROOT:-/projects/a5l/ziyan}"
HF_CACHE_ROOT="${CHESS_RL_HF_CACHE_ROOT:-${PROJECT_ROOT%/}/hf_cache}"
mkdir -p "${HF_CACHE_ROOT}"
export HF_HOME="${HF_CACHE_ROOT%/}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"
echo "[SLURM] HF_HOME=${HF_HOME}"

INFER_OUT_DIR="${INFER_OUT_DIR:-}"
if [ -z "${INFER_OUT_DIR}" ]; then
  die "Set INFER_OUT_DIR to the infer-stage OUT_DIR path."
fi
echo "[SLURM] INFER_OUT_DIR=${INFER_OUT_DIR}"

STOCKFISH_PATH="${STOCKFISH_PATH:-/usr/local/bin/stockfish}"
STOCKFISH_HASH_MB="${STOCKFISH_HASH_MB:-128}"
ACPL_DEPTH="${ACPL_DEPTH:-20}"
ACPL_MOVETIME_MS="${ACPL_MOVETIME_MS:-1000}"
ACPL_CP_CAP="${ACPL_CP_CAP:-1000}"
MATE_SCORE_CP="${MATE_SCORE_CP:-1000}"
RESIGNATION_CPL="${RESIGNATION_CPL:-1000}"
ACPL_WORKERS="${ACPL_WORKERS:-72}"
ACPL_THREADS="${ACPL_THREADS:-2}"

echo "[SLURM] stockfish_path=${STOCKFISH_PATH} depth=${ACPL_DEPTH} movetime_ms=${ACPL_MOVETIME_MS} workers=${ACPL_WORKERS} threads=${ACPL_THREADS}"

CMD="
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
cd ${CONTAINER_REPO_DIR}
python3 scripts/eval_chess_fullgame_round_robin_acpl.py \\
  --infer-out-dir \"${INFER_OUT_DIR}\" \\
  --stockfish-path \"${STOCKFISH_PATH}\" \\
  --stockfish-hash-mb \"${STOCKFISH_HASH_MB}\" \\
  --acpl-depth \"${ACPL_DEPTH}\" \\
  --acpl-movetime-ms \"${ACPL_MOVETIME_MS}\" \\
  --acpl-cp-cap \"${ACPL_CP_CAP}\" \\
  --mate-score-cp \"${MATE_SCORE_CP}\" \\
  --resignation-cpl \"${RESIGNATION_CPL}\" \\
  --acpl-workers \"${ACPL_WORKERS}\" \\
  --acpl-threads \"${ACPL_THREADS}\"
echo \"[RUN] Done. Outputs under: ${INFER_OUT_DIR}\"
ls -la \"${INFER_OUT_DIR}\" || true
"

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

echo "[SLURM] Done."

