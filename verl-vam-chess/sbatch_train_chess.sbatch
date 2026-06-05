#!/bin/bash
#SBATCH --job-name=chess-rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


# Minimal Slurm launcher for running `train_chess.sh` inside the prebuilt
# `gabr1e1/chess_rl:v1` container via Singularity/Apptainer.
#
# If the image does not exist yet, this script will build it on-demand from
# Docker Hub (per cluster admin docs).
#
# Submit:
#   sbatch sbatch_train_chess.sbatch
#
# Optional overrides at submit time:
#   sbatch --export=ALL,FORCED_PREFIX_PROB_END=0.2,FORCED_PREFIX_ANNEAL_FRAC=0.5,FORCED_PREFIX_TEMP=2.0 sbatch_train_chess.sbatch


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
set -x

SIF_IMAGE="${HOME}/sif-images/chess_rl_v1.sif"
SIF_URI="docker://gabr1e1/chess_rl:v1"

SINGULARITY_BIN="${SINGULARITY_BIN:-}"
if [ -z "${SINGULARITY_BIN}" ]; then
  if command -v singularity >/dev/null 2>&1; then
    SINGULARITY_BIN="singularity"
  elif command -v apptainer >/dev/null 2>&1; then
    SINGULARITY_BIN="apptainer"
  else
    echo "[ERROR] Neither 'singularity' nor 'apptainer' is available on PATH." >&2
    exit 1
  fi
fi

mkdir -p "$(dirname "${SIF_IMAGE}")"
if [ ! -f "${SIF_IMAGE}" ]; then
  echo "[SLURM] Building image: ${SIF_IMAGE}"
  echo "[SLURM] From: ${SIF_URI}"
  "${SINGULARITY_BIN}" build "${SIF_IMAGE}" "${SIF_URI}"
fi

# Run relative to this script's directory (repo root) on the host.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

# Make Python logs unbuffered for timely flush to Slurm output.
export PYTHONUNBUFFERED=1

# Bind the repo into the container at a stable path (mirrors the Docker workflow).
CONTAINER_WORKDIR="/workspace"

echo "[SLURM] Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "[SLURM] Node list:   ${SLURM_JOB_NODELIST:-N/A}"
echo "[SLURM] GPUs:        ${SLURM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"
echo "[SLURM] Host repo:   ${REPO_DIR}"
echo "[SLURM] Image:       ${SIF_IMAGE}"
echo "[SLURM] Container wd:${CONTAINER_WORKDIR}"

# This job always requests 4 GPUs; tell the training launcher to use them all.
export TRAINER_N_GPUS_PER_NODE=4

# Some clusters only populate CUDA_VISIBLE_DEVICES for job steps launched via srun.
# If it's missing in the batch step, fall back to Slurm's GPU lists.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${SLURM_STEP_GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS//gpu:/}"
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS//gpu:/}"
fi
echo "[SLURM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

"${SINGULARITY_BIN}" exec --nv \
  --bind "${REPO_DIR}:${CONTAINER_WORKDIR}" \
  "${SIF_IMAGE}" \
  bash -lc "cd \"${CONTAINER_WORKDIR}\" && ./train_chess.sh"
