#!/bin/bash
#SBATCH --job-name=chess-small-legal-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --time=1-00:00:00
#SBATCH --output=/projects/a5l/ziyan/chess_rl/logs/slurm-%x-%j.out
#SBATCH --error=/projects/a5l/ziyan/chess_rl/logs/slurm-%x-%j.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL
#SBATCH --exclude=nid010586,nid010331,nid010230,nid010223,nid010226,nid010025

# GH200 / Isambard launcher:
# - Runs `train_chess.sh` inside the ARM64 container image `gabr1e1/chess_rl:v1-arm-stockfish-flashinfer`.
# - Uses Apptainer/Singularity (Docker is typically unavailable on HPC).
#
# Example submissions (partition/qos vary by site):
#   # Default: small-legal reference profile (matches run t4elop1z / Slurm 2449991).
#   sbatch ./sbatch_chess_small_legal_gh200.slurm
#
#   # Disable forced-prefix exploration (sets pstart=pend=0.0 inside the job):
#   sbatch --export=ALL,FORCED_PREFIX_ENABLE=False ./sbatch_chess_small_legal_gh200.slurm
#
#   # Optional: train on a hard subset if your dataset dir provides `train_hard.parquet`:
#   sbatch --export=ALL,USE_HARD_DATASET=True ./sbatch_chess_small_legal_gh200.slurm
#
#   # Enable Hugging Face checkpoint upload + auto-resume (opt-in):
#   # - Create a token file on the cluster filesystem and keep permissions tight:
#   #     echo "hf_***" > $HOME/.hf_token && chmod 600 $HOME/.hf_token
#   # - Then submit with upload enabled (and consider a lower save cadence):
#   #     sbatch --export=ALL,CHESS_RL_HF_UPLOAD_ENABLE=True,CHESS_RL_TRAINER_SAVE_FREQ=200 ./sbatch_chess_small_legal_gh200.slurm
#
#   # Change chess reward shaping (see recipe/chess/reward_fn.py for options):
#   sbatch --export=ALL,CHESS_REWARD_FN=delta_winrate ./sbatch_chess_small_legal_gh200.slurm
#
# Dataset selection:
# - Defaults to the small-legal Chess-R1-aligned rewrite used in current runs.
# - By default, validation uses both normal and shuffled-legal eval splits.
# - To use an alternative dataset directory, pass `CHESS_DATA_DIR=...` via `--export`.
#
# Optional overrides (avoid leaking interactive env vars via `--export=ALL`):
#   sbatch --export=ALL,CHESS_RL_WANDB_MODE=offline,CHESS_RL_TOTAL_TRAINING_STEPS=500 ./sbatch_chess_small_legal_gh200.slurm
#
# Checkpoint cadence + full-game eval cadence:
# - `CHESS_RL_TRAINER_SAVE_FREQ` controls the checkpoint save interval (trainer.save_freq).
#   - Default is `80` (matches the reference full run).
# - `CHESS_RL_FULL_EVAL_FREQ` controls full-game evaluation cadence (trainer.full_eval_freq).
#   - Default is `80` (matches the reference full run).
# - Example: save every 200 steps and full-game eval every 200 steps:
#     sbatch --export=ALL,CHESS_RL_TRAINER_SAVE_FREQ=200,CHESS_RL_FULL_EVAL_FREQ=200 ./sbatch_chess_small_legal_gh200.slurm

set -euo pipefail

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

warn() {
  echo "[WARN] $*" >&2
}

# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export VLLM_ALLREDUCE_USE_SYMM_MEM=0

echo "[SLURM] Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "[SLURM] Job name:    ${SLURM_JOB_NAME:-chess-small-legal-gh200}"
echo "[SLURM] Node list:   ${SLURM_JOB_NODELIST:-N/A}"
JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
echo "[SLURM] Nodes:       ${JOB_NUM_NODES}"
echo "[SLURM] GPUs:        ${SLURM_GPUS_ON_NODE:-4} (requested via --gres=gpu:4)"
echo "[SLURM] Submit dir:  ${SLURM_SUBMIT_DIR:-$(pwd)}"

# Run relative to this script's directory (repo root) so relative paths work.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  THIS_DIR="${SLURM_SUBMIT_DIR}"
else
  THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${THIS_DIR}"
echo "[SLURM] Repo dir:    ${THIS_DIR}"

# This launcher is intended for Arm64 / aarch64 compute nodes (GH200).
HOST_ARCH="$(uname -m 2>/dev/null || echo unknown)"
echo "[SLURM] Host arch:   ${HOST_ARCH}"
if [ "${HOST_ARCH}" != "aarch64" ] && [ "${HOST_ARCH}" != "arm64" ]; then
  die "This script targets GH200 (arm64/aarch64) nodes, but uname reports '${HOST_ARCH}'. If you're on x86_64, build/run an amd64 image (e.g., 'docker build --platform linux/amd64 ...')."
fi

# Preferred shared scratch (persistent, non-tmpfs on Isambard):
# - Default to /projects/a5l/ziyan/chess_rl if writable, to avoid tmpfs pressure.
# - Users can override any path explicitly via CHESS_RL_* env vars.
CHESS_RL_PROJECTS_ROOT="${CHESS_RL_PROJECTS_ROOT:-/projects/a5l/ziyan/chess_rl}"
PROJECTS_SCRATCH_OK=0
if [ -d "/projects" ]; then
  if mkdir -p "${CHESS_RL_PROJECTS_ROOT}" 2>/dev/null && [ -w "${CHESS_RL_PROJECTS_ROOT}" ]; then
    PROJECTS_SCRATCH_OK=1
  fi
fi
if [ "${PROJECTS_SCRATCH_OK}" -eq 1 ]; then
  CHESS_RL_SINGULARITY_TMPDIR="${CHESS_RL_SINGULARITY_TMPDIR:-${CHESS_RL_PROJECTS_ROOT%/}/singularity}"
  CHESS_RL_TMPDIR="${CHESS_RL_TMPDIR:-${CHESS_RL_PROJECTS_ROOT%/}/tmp}"
  CHESS_RL_APPTAINER_CACHEDIR="${CHESS_RL_APPTAINER_CACHEDIR:-${CHESS_RL_PROJECTS_ROOT%/}/apptainer_cache}"
  CHESS_RL_WANDB_DIR="${CHESS_RL_WANDB_DIR:-${CHESS_RL_PROJECTS_ROOT%/}/wandb}"
fi

# Apptainer/Singularity tmp directory:
# - Apptainer does a rootless OCI extraction during `build docker://...`.
# - Many HPC home filesystems do **not** support extended attributes (xattrs), which can break
#   rootless extraction and yield a subtly broken image (e.g., failed Python imports).
# - Prefer node-local scratch when available (SLURM_TMPDIR or /tmp).
#
# You can override explicitly at submit time, e.g.:
#   sbatch --export=ALL,CHESS_RL_SINGULARITY_TMPDIR=$SLURM_TMPDIR ./sbatch_chess_small_legal_gh200.slurm
CHESS_RL_SINGULARITY_TMPDIR="${CHESS_RL_SINGULARITY_TMPDIR:-}"
LOCAL_SCRATCH_BASE="/local/user/1483802860"
LOCAL_SCRATCH_OK=0
LOCAL_SCRATCH_TEST_DIR="${LOCAL_SCRATCH_BASE%/}/${USER}/.chess_rl_scratch_${SLURM_JOB_ID:-$$}"
if [ -d "${LOCAL_SCRATCH_BASE}" ]; then
  if command -v srun >/dev/null 2>&1; then
    if srun --nodes="${JOB_NUM_NODES}" --ntasks="${JOB_NUM_NODES}" --export=ALL,TMPDIR=/tmp \
      bash -lc "mkdir -p \"${LOCAL_SCRATCH_TEST_DIR}\" && touch \"${LOCAL_SCRATCH_TEST_DIR}/.probe\" && rm -rf \"${LOCAL_SCRATCH_TEST_DIR}\"" >/dev/null 2>&1; then
      LOCAL_SCRATCH_OK=1
    else
      warn "Local scratch '${LOCAL_SCRATCH_BASE}' not writable on all nodes; falling back."
    fi
  else
    if mkdir -p "${LOCAL_SCRATCH_TEST_DIR}" 2>/dev/null && touch "${LOCAL_SCRATCH_TEST_DIR}/.probe" 2>/dev/null; then
      rm -rf "${LOCAL_SCRATCH_TEST_DIR}" 2>/dev/null || true
      LOCAL_SCRATCH_OK=1
    else
      warn "Local scratch '${LOCAL_SCRATCH_BASE}' not writable; falling back."
    fi
  fi
fi

if [ -n "${CHESS_RL_SINGULARITY_TMPDIR}" ]; then
  DEFAULT_SINGULARITY_TMP="${CHESS_RL_SINGULARITY_TMPDIR}"
elif [ "${LOCAL_SCRATCH_OK}" -eq 1 ]; then
  # Isambard manual scratch path (job-local, node-local). Prefer this when available.
  DEFAULT_SINGULARITY_TMP="${LOCAL_SCRATCH_BASE%/}/${USER}/singularity"
elif [ -n "${SLURM_TMPDIR:-}" ]; then
  DEFAULT_SINGULARITY_TMP="${SLURM_TMPDIR}/singularity"
else
  DEFAULT_SINGULARITY_TMP="/tmp/${USER}/singularity"
fi
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${DEFAULT_SINGULARITY_TMP}}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SINGULARITY_TMPDIR}}"
if ! mkdir -p "${SINGULARITY_TMPDIR}" 2>/dev/null; then
  if [ -n "${CHESS_RL_SINGULARITY_TMPDIR}" ]; then
    die "Cannot create SINGULARITY_TMPDIR='${SINGULARITY_TMPDIR}'. Set CHESS_RL_SINGULARITY_TMPDIR to a writable scratch path."
  fi
  warn "Cannot create SINGULARITY_TMPDIR='${SINGULARITY_TMPDIR}'; falling back to /tmp/${USER}/singularity."
  SINGULARITY_TMPDIR="/tmp/${USER}/singularity"
  export SINGULARITY_TMPDIR
  export APPTAINER_TMPDIR="${SINGULARITY_TMPDIR}"
  mkdir -p "${SINGULARITY_TMPDIR}" 2>/dev/null || die "Failed to create fallback SINGULARITY_TMPDIR='${SINGULARITY_TMPDIR}'."
fi
# Best-effort xattr probe:
# - Rootless `apptainer build docker://...` relies on xattrs during OCI extraction.
# - On some HPC filesystems xattrs are unsupported, which can produce subtly broken SIFs.
# - We probe once here and decide whether to hard-fail later *only if* we actually need to build a SIF.
SINGULARITY_TMPDIR_HAS_XATTRS="unknown"
if command -v python3 >/dev/null 2>&1; then
  if python3 - <<'PY' "${SINGULARITY_TMPDIR}" >/dev/null 2>&1; then
import os
import sys
import tempfile

tmpdir = sys.argv[1]
os.makedirs(tmpdir, exist_ok=True)
fd, path = tempfile.mkstemp(dir=tmpdir)
os.close(fd)
try:
    os.setxattr(path, "user.chess_rl_xattr_test", b"1")
    os.removexattr(path, "user.chess_rl_xattr_test")
finally:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
PY
    SINGULARITY_TMPDIR_HAS_XATTRS="1"
  else
    SINGULARITY_TMPDIR_HAS_XATTRS="0"
  fi
fi

# If the default temp dir doesn't support xattrs, try common alternatives automatically
# (unless the user explicitly pinned CHESS_RL_SINGULARITY_TMPDIR).
if [ "${SINGULARITY_TMPDIR_HAS_XATTRS}" = "0" ] && [ -z "${CHESS_RL_SINGULARITY_TMPDIR:-}" ] && command -v python3 >/dev/null 2>&1; then
  warn "Default SINGULARITY_TMPDIR='${SINGULARITY_TMPDIR}' lacks xattrs; trying fallback temp directories..."
  for _tmp_base in "/local/user/1483802860" "${SLURM_TMPDIR:-}" "${TMPDIR:-}" "/var/tmp" "/tmp" "${THIS_DIR}" "${HOME}"; do
    if [ -z "${_tmp_base}" ]; then
      continue
    fi
    # Prefer a user-scoped directory to avoid collisions.
    _tmp_try="${_tmp_base%/}/${USER}/singularity"
    if ! mkdir -p "${_tmp_try}" 2>/dev/null || [ ! -w "${_tmp_try}" ]; then
      continue
    fi
    if python3 - <<'PY' "${_tmp_try}" >/dev/null 2>&1; then
import os
import sys
import tempfile

tmpdir = sys.argv[1]
os.makedirs(tmpdir, exist_ok=True)
fd, path = tempfile.mkstemp(dir=tmpdir)
os.close(fd)
try:
    os.setxattr(path, "user.chess_rl_xattr_test", b"1")
    os.removexattr(path, "user.chess_rl_xattr_test")
finally:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
PY
      SINGULARITY_TMPDIR="${_tmp_try}"
      export SINGULARITY_TMPDIR
      # Keep Apptainer's tmpdir consistent with Singularity's; this controls OCI extraction temp.
      export APPTAINER_TMPDIR="${SINGULARITY_TMPDIR}"
      SINGULARITY_TMPDIR_HAS_XATTRS="1"
      echo "[SLURM] Selected xattr-capable SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR}"
      break
    fi
  done
fi

# Runtime temp (Ray, etc.):
# - Ignore inherited TMPDIR (e.g., /local) because --export=ALL can leak read-only paths.
# - Prefer an explicit CHESS_RL_TMPDIR override, else reuse SINGULARITY_TMPDIR, else /tmp.
TMPDIR_CANDIDATE="${CHESS_RL_TMPDIR:-${SINGULARITY_TMPDIR}}"
if ! mkdir -p "${TMPDIR_CANDIDATE}" 2>/dev/null || [ ! -w "${TMPDIR_CANDIDATE}" ]; then
  warn "TMPDIR='${TMPDIR_CANDIDATE}' not writable; falling back to /tmp"
  TMPDIR_CANDIDATE="/tmp"
  mkdir -p "${TMPDIR_CANDIDATE}"
fi
export TMPDIR="${TMPDIR_CANDIDATE}"

# Fail fast if /tmp is full on any node (prevents Ray/IPC failures).
if command -v srun >/dev/null 2>&1; then
  echo "[SLURM] Checking /tmp capacity on all nodes..."
  TMP_CHECK_OUT="$(srun --nodes="${JOB_NUM_NODES}" --ntasks="${JOB_NUM_NODES}" --export=ALL,TMPDIR=/tmp \
    bash -lc 'h=$(hostname); line=$(df -P /tmp 2>/dev/null | tail -n 1); use=$(echo "$line" | awk "{print \$(NF-1)}"); avail=$(echo "$line" | awk "{print \$(NF-2)}"); use=${use%%%}; if [ -z "$use" ]; then echo "$h TMP_CHECK_FAILED $line"; exit 2; fi; if [ "$use" -ge 100 ] || [ "${avail:-0}" -eq 0 ]; then echo "$h TMP_FULL use=${use} avail=${avail}"; exit 3; else echo "$h TMP_OK use=${use} avail=${avail}"; fi' 2>&1 || true)"
  echo "${TMP_CHECK_OUT}"
  if echo "${TMP_CHECK_OUT}" | grep -q "TMP_FULL\|TMP_CHECK_FAILED"; then
    die "One or more nodes have full /tmp. Aborting to avoid Ray/IPC failures."
  fi
fi

# Apptainer/Singularity cache directory:
# - On some clusters the site sets APPTAINER_CACHEDIR to a shared or transient /local path.
#   We've seen this cause flaky OCI layer extraction with errors like:
#     "error writing layer: rename ... no such file or directory"
# - Use a job-scoped cache by default to avoid collisions and reduce reliance on site defaults.
# - Users can override explicitly via CHESS_RL_APPTAINER_CACHEDIR.
APPTAINER_CACHEDIR_CANDIDATE="${CHESS_RL_APPTAINER_CACHEDIR:-${SLURM_TMPDIR:-${TMPDIR}}}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR_CANDIDATE%/}/apptainer_cache_${SLURM_JOB_ID:-$$}"
if ! mkdir -p "${APPTAINER_CACHEDIR}" 2>/dev/null || [ ! -w "${APPTAINER_CACHEDIR}" ]; then
  warn "APPTAINER_CACHEDIR='${APPTAINER_CACHEDIR}' not writable; falling back to TMPDIR-based cache."
  APPTAINER_CACHEDIR="${TMPDIR%/}/apptainer_cache_${SLURM_JOB_ID:-$$}"
  mkdir -p "${APPTAINER_CACHEDIR}"
fi
export APPTAINER_CACHEDIR
# SingularityCE uses SINGULARITY_CACHEDIR; set it too for compatibility.
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"

# Ensure tmp/cache directories exist on every node before any `srun`/apptainer steps.
if command -v srun >/dev/null 2>&1; then
  echo "[SLURM] Ensuring tmp/cache dirs exist on all nodes..."
  # Force a safe TMPDIR for this preflight so Slurm can launch tasks even if the desired TMPDIR
  # does not exist yet on worker nodes.
  srun --nodes="${JOB_NUM_NODES}" --ntasks="${JOB_NUM_NODES}" --export=ALL,TMPDIR=/tmp \
    bash -lc "mkdir -p \"${SINGULARITY_TMPDIR}\" \"${APPTAINER_TMPDIR}\" \"${TMPDIR}\" \"${APPTAINER_CACHEDIR}\" \"${SINGULARITY_CACHEDIR}\"" || true
fi

echo "[SLURM] SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR}"
echo "[SLURM] APPTAINER_TMPDIR=${APPTAINER_TMPDIR}"
echo "[SLURM] TMPDIR=${TMPDIR}"
echo "[SLURM] APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR}"
echo "[SLURM] SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR}"
echo "[SLURM] SINGULARITY_TMPDIR_HAS_XATTRS=${SINGULARITY_TMPDIR_HAS_XATTRS}"

# Avoid picking up user-site Python packages from the login node (can mismatch container Torch/Functorch).
export PYTHONNOUSERSITE=1
echo "[SLURM] PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"

# W&B:
# - Default to online so cluster runs are tracked by default.
# - Avoid inheriting interactive-shell values like `WANDB_MODE=offline` from `--export=ALL`.
# - Override explicitly via `CHESS_RL_WANDB_MODE` at submit time if desired.
CHESS_RL_WANDB_MODE="${CHESS_RL_WANDB_MODE:-online}"
export WANDB_MODE="${CHESS_RL_WANDB_MODE}"
export PYTHONUNBUFFERED=1
echo "[SLURM] WANDB_MODE=${WANDB_MODE}"

# W&B API key propagation (multi-node Ray):
# - Ray head/workers are started **before** `train_chess.sh` runs, so exporting `WANDB_API_KEY`
#   inside `train_chess.sh` is too late for Ray worker processes.
# - Load `WANDB_API_KEY` early (before starting Ray) from a user-scoped key file if available.
# - IMPORTANT: do not `echo` the key (it may appear in Slurm logs).
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${HOME}/.wandb_api_key_zzc}"
if [ "${WANDB_MODE}" != "offline" ] && [ "${WANDB_MODE}" != "disabled" ]; then
  if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${WANDB_API_KEY_FILE}" ]; then
    export WANDB_API_KEY="$(tr -d ' \n\r\t' <"${WANDB_API_KEY_FILE}")"
  fi
  if [ -z "${WANDB_API_KEY:-}" ]; then
    warn "WANDB_MODE=${WANDB_MODE} but WANDB_API_KEY is unset; wandb.init may prompt/fail on Ray workers. Set WANDB_API_KEY or create ${WANDB_API_KEY_FILE}."
  fi
fi

# Dataset directory (consumed by train_chess.sh):
# - Default to the small-legal chess-r1 aligned rewrite used by this experiment.
# - Use a repo-relative path so it works both on the host (for checks) and inside the container.
export CHESS_DATA_DIR="${CHESS_DATA_DIR:-data/chess_puzzles_chessr1_aligned_sharded_ours_small_legal}"
echo "[SLURM] CHESS_DATA_DIR=${CHESS_DATA_DIR}"
VAL_FILES_DEFAULT="[${CHESS_DATA_DIR}/test.parquet,${CHESS_DATA_DIR}/test_shuffled_legal_moves.parquet]"
CHESS_RL_VAL_FILES="${CHESS_RL_VAL_FILES:-${VAL_FILES_DEFAULT}}"
export CHESS_RL_VAL_FILES
echo "[SLURM] CHESS_RL_VAL_FILES=${CHESS_RL_VAL_FILES}"

# Training batch size (consumed by `train_chess.sh` as TRAIN_BATCH_SIZE).
# - Default is 128 (matches run t4elop1z / Slurm 2449991).
CHESS_RL_TRAIN_BATCH_SIZE="${CHESS_RL_TRAIN_BATCH_SIZE:-128}"
export TRAIN_BATCH_SIZE="${CHESS_RL_TRAIN_BATCH_SIZE}"
echo "[SLURM] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"

# Core model and sequence-shape defaults (match the reference full run).
CHESS_RL_MODEL_PATH="${CHESS_RL_MODEL_PATH:-/projects/a5l/ziyan/models/Qwen/Qwen2.5-3B-Instruct}"
export MODEL_PATH="${CHESS_RL_MODEL_PATH}"
CHESS_RL_MAX_PROMPT_LENGTH="${CHESS_RL_MAX_PROMPT_LENGTH:-1536}"
export MAX_PROMPT_LENGTH="${CHESS_RL_MAX_PROMPT_LENGTH}"
CHESS_RL_MAX_RESPONSE_LENGTH="${CHESS_RL_MAX_RESPONSE_LENGTH:-2000}"
export MAX_RESPONSE_LENGTH="${CHESS_RL_MAX_RESPONSE_LENGTH}"
CHESS_RL_TOTAL_EPOCHS="${CHESS_RL_TOTAL_EPOCHS:-1}"
export TOTAL_EPOCHS="${CHESS_RL_TOTAL_EPOCHS}"
CHESS_RL_VAL_BEFORE_TRAIN="${CHESS_RL_VAL_BEFORE_TRAIN:-True}"
export VAL_BEFORE_TRAIN="${CHESS_RL_VAL_BEFORE_TRAIN}"
CHESS_RL_TRAINER_TEST_FREQ="${CHESS_RL_TRAINER_TEST_FREQ:-40}"
export TRAINER_TEST_FREQ="${CHESS_RL_TRAINER_TEST_FREQ}"
echo "[SLURM] MODEL_PATH=${MODEL_PATH}"
echo "[SLURM] MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}"
echo "[SLURM] MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
echo "[SLURM] TOTAL_EPOCHS=${TOTAL_EPOCHS}"
echo "[SLURM] VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN}"
echo "[SLURM] TRAINER_TEST_FREQ=${TRAINER_TEST_FREQ}"

# Optional dataset toggles (used by train_chess.sh)
export USE_CRITIC_DATASET="${USE_CRITIC_DATASET:-False}"
echo "[SLURM] USE_CRITIC_DATASET=${USE_CRITIC_DATASET}"
export USE_HARD_DATASET="${USE_HARD_DATASET:-False}"
echo "[SLURM] USE_HARD_DATASET=${USE_HARD_DATASET}"
if [ "${USE_CRITIC_DATASET}" = "True" ] && [ "${USE_HARD_DATASET}" = "True" ]; then
  die "USE_CRITIC_DATASET=True and USE_HARD_DATASET=True is unsupported (no *_hard_critic.parquet)."
fi

# Optional: filter_groups rejected-group logging (debug).
# - These are consumed by `train_chess.sh` and forwarded to the trainer config.
# - Default to enabled; override to empty/0 to disable.
# - Leave the log dir empty here so `train_chess.sh` can default under VERL_BASE_DIR.
CHESS_RL_REJECTED_ROLLOUT_DATA_DIR="${CHESS_RL_REJECTED_ROLLOUT_DATA_DIR:-}"
export REJECTED_ROLLOUT_DATA_DIR="${CHESS_RL_REJECTED_ROLLOUT_DATA_DIR}"
CHESS_RL_REJECTED_GROUP_SUMMARY_MAX_GROUPS_PER_STEP="${CHESS_RL_REJECTED_GROUP_SUMMARY_MAX_GROUPS_PER_STEP:--1}"
export REJECTED_GROUP_SUMMARY_MAX_GROUPS_PER_STEP="${CHESS_RL_REJECTED_GROUP_SUMMARY_MAX_GROUPS_PER_STEP}"
CHESS_RL_REJECTED_ROLLOUT_MAX_GROUPS_PER_STEP="${CHESS_RL_REJECTED_ROLLOUT_MAX_GROUPS_PER_STEP:--1}"
export REJECTED_ROLLOUT_MAX_GROUPS_PER_STEP="${CHESS_RL_REJECTED_ROLLOUT_MAX_GROUPS_PER_STEP}"
echo "[SLURM] REJECTED_ROLLOUT_DATA_DIR=${REJECTED_ROLLOUT_DATA_DIR}"
echo "[SLURM] REJECTED_GROUP_SUMMARY_MAX_GROUPS_PER_STEP=${REJECTED_GROUP_SUMMARY_MAX_GROUPS_PER_STEP}"
echo "[SLURM] REJECTED_ROLLOUT_MAX_GROUPS_PER_STEP=${REJECTED_ROLLOUT_MAX_GROUPS_PER_STEP}"

# On-the-fly allowed-move elimination (selection sampling) toggles:
# - When enabled, train_chess.sh will disable filter_groups automatically.
CHESS_ALLOWED_MOVE_ELIM_ENABLE="${CHESS_ALLOWED_MOVE_ELIM_ENABLE:-True}"
export ALLOWED_MOVE_ELIM_ENABLE="${CHESS_ALLOWED_MOVE_ELIM_ENABLE}"
CHESS_ALLOWED_MOVE_ELIM_TEMPLATE_PATH="${CHESS_ALLOWED_MOVE_ELIM_TEMPLATE_PATH:-recipe/chess/prompt_templates/select_prompt_small_legal.jinja}"
export ALLOWED_MOVE_ELIM_TEMPLATE_PATH="${CHESS_ALLOWED_MOVE_ELIM_TEMPLATE_PATH}"
CHESS_ALLOWED_MOVE_ELIM_UID_MODE="${CHESS_ALLOWED_MOVE_ELIM_UID_MODE:-per_prompt}"
export ALLOWED_MOVE_ELIM_UID_MODE="${CHESS_ALLOWED_MOVE_ELIM_UID_MODE}"
CHESS_ALLOWED_MOVE_ELIM_NO_SUCCESS_POLICY="${CHESS_ALLOWED_MOVE_ELIM_NO_SUCCESS_POLICY:-accept_last}"
export ALLOWED_MOVE_ELIM_NO_SUCCESS_POLICY="${CHESS_ALLOWED_MOVE_ELIM_NO_SUCCESS_POLICY}"
CHESS_ALLOWED_MOVE_ELIM_R_MAX_START="${CHESS_ALLOWED_MOVE_ELIM_R_MAX_START:-4}"
export ALLOWED_MOVE_ELIM_R_MAX_START="${CHESS_ALLOWED_MOVE_ELIM_R_MAX_START}"
CHESS_ALLOWED_MOVE_ELIM_R_MAX_END="${CHESS_ALLOWED_MOVE_ELIM_R_MAX_END:-4}"
export ALLOWED_MOVE_ELIM_R_MAX_END="${CHESS_ALLOWED_MOVE_ELIM_R_MAX_END}"
CHESS_ALLOWED_MOVE_ELIM_ANNEAL_FRAC="${CHESS_ALLOWED_MOVE_ELIM_ANNEAL_FRAC:-1.0}"
export ALLOWED_MOVE_ELIM_ANNEAL_FRAC="${CHESS_ALLOWED_MOVE_ELIM_ANNEAL_FRAC}"
CHESS_ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB="${CHESS_ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB:-True}"
export ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB="${CHESS_ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB}"
CHESS_ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI="${CHESS_ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI:-True}"
export ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI="${CHESS_ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI}"
CHESS_ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN="${CHESS_ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN:-0.0}"
export ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN="${CHESS_ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN}"
CHESS_ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_DUMP_MAX_GROUPS="${CHESS_ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_DUMP_MAX_GROUPS:--1}"
export ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_DUMP_MAX_GROUPS="${CHESS_ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_DUMP_MAX_GROUPS}"
CHESS_ALLOWED_MOVE_ELIM_GAIN_THRESHOLD="${CHESS_ALLOWED_MOVE_ELIM_GAIN_THRESHOLD:-2.302585092994046}"
export ALLOWED_MOVE_ELIM_GAIN_THRESHOLD="${CHESS_ALLOWED_MOVE_ELIM_GAIN_THRESHOLD}"
CHESS_ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL="${CHESS_ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL:-False}"
export ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL="${CHESS_ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL}"
CHESS_ALLOWED_MOVE_ELIM_PASS_K_K="${CHESS_ALLOWED_MOVE_ELIM_PASS_K_K:-4}"
export ALLOWED_MOVE_ELIM_PASS_K_K="${CHESS_ALLOWED_MOVE_ELIM_PASS_K_K}"
CHESS_ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL="${CHESS_ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL:-False}"
export ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL="${CHESS_ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL}"
echo "[SLURM] ALLOWED_MOVE_ELIM_ENABLE=${ALLOWED_MOVE_ELIM_ENABLE}"
if [ "${ALLOWED_MOVE_ELIM_ENABLE}" = "True" ]; then
  echo "[SLURM] ALLOWED_MOVE_ELIM_TEMPLATE_PATH=${ALLOWED_MOVE_ELIM_TEMPLATE_PATH}"
  echo "[SLURM] ALLOWED_MOVE_ELIM_UID_MODE=${ALLOWED_MOVE_ELIM_UID_MODE}"
  echo "[SLURM] ALLOWED_MOVE_ELIM_NO_SUCCESS_POLICY=${ALLOWED_MOVE_ELIM_NO_SUCCESS_POLICY}"
  echo "[SLURM] ALLOWED_MOVE_ELIM_R_MAX_START=${ALLOWED_MOVE_ELIM_R_MAX_START}"
  echo "[SLURM] ALLOWED_MOVE_ELIM_R_MAX_END=${ALLOWED_MOVE_ELIM_R_MAX_END}"
  echo "[SLURM] ALLOWED_MOVE_ELIM_ANNEAL_FRAC=${ALLOWED_MOVE_ELIM_ANNEAL_FRAC}"
  echo "[SLURM] ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB=${ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB}"
  echo "[SLURM] ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI=${ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI}"
  echo "[SLURM] ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN=${ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN}"
  echo "[SLURM] ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_DUMP_MAX_GROUPS=${ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_DUMP_MAX_GROUPS}"
  echo "[SLURM] ALLOWED_MOVE_ELIM_GAIN_THRESHOLD=${ALLOWED_MOVE_ELIM_GAIN_THRESHOLD}"
  echo "[SLURM] ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL=${ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL}"
  echo "[SLURM] ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL=${ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL}"
  if [ "${ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL}" = "True" ]; then
    echo "[SLURM] ALLOWED_MOVE_ELIM_PASS_K_K=${ALLOWED_MOVE_ELIM_PASS_K_K}"
  fi
fi

# Optional: Pass@k analytic advantage variant for baseline GRPO (default-off).
CHESS_PASS_K_TRAINING="${CHESS_PASS_K_TRAINING:-False}"
export PASS_K_TRAINING="${CHESS_PASS_K_TRAINING}"
CHESS_PASS_K="${CHESS_PASS_K:-1}"
export PASS_K="${CHESS_PASS_K}"
echo "[SLURM] PASS_K_TRAINING=${PASS_K_TRAINING}"
if [ "${PASS_K_TRAINING}" = "True" ]; then
  echo "[SLURM] PASS_K=${PASS_K}"
fi

# Optional: diversity auxiliary advantages for GRPO/Pass@k (default-off).
CHESS_DIVERSITY_ENABLE="${CHESS_DIVERSITY_ENABLE:-False}"
export DIVERSITY_ENABLE="${CHESS_DIVERSITY_ENABLE}"
CHESS_DIVERSITY_METHOD="${CHESS_DIVERSITY_METHOD:-none}"
export DIVERSITY_METHOD="${CHESS_DIVERSITY_METHOD}"
CHESS_DIVERSITY_LAMBDA="${CHESS_DIVERSITY_LAMBDA:-0.0}"
export DIVERSITY_LAMBDA="${CHESS_DIVERSITY_LAMBDA}"
CHESS_DIVERSITY_DISTINCT_K="${CHESS_DIVERSITY_DISTINCT_K:-4}"
export DIVERSITY_DISTINCT_K="${CHESS_DIVERSITY_DISTINCT_K}"
CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE="${CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE:-True}"
export DIVERSITY_INCLUDE_BASE_ADVANTAGE="${CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE}"
echo "[SLURM] DIVERSITY_ENABLE=${DIVERSITY_ENABLE}"
if [ "${DIVERSITY_ENABLE}" = "True" ]; then
  echo "[SLURM] DIVERSITY_METHOD=${DIVERSITY_METHOD}"
  echo "[SLURM] DIVERSITY_LAMBDA=${DIVERSITY_LAMBDA}"
  echo "[SLURM] DIVERSITY_DISTINCT_K=${DIVERSITY_DISTINCT_K}"
  echo "[SLURM] DIVERSITY_INCLUDE_BASE_ADVANTAGE=${DIVERSITY_INCLUDE_BASE_ADVANTAGE}"
fi

# Fail fast if required dataset files are missing.
# - Support both single-file parquets (`train.parquet`) and numeric shards (`train_[0-9]*.parquet`).
CHESS_DATA_DIR_HOST="${CHESS_DATA_DIR}"
if [[ "${CHESS_DATA_DIR_HOST}" != /* ]]; then
  CHESS_DATA_DIR_HOST="${THIS_DIR%/}/${CHESS_DATA_DIR_HOST}"
fi
echo "[SLURM] CHESS_DATA_DIR_HOST=${CHESS_DATA_DIR_HOST}"

if [ ! -f "${CHESS_DATA_DIR_HOST}/train.parquet" ] && ! compgen -G "${CHESS_DATA_DIR_HOST}/train_[0-9]*.parquet" >/dev/null; then
  die "Missing training data under '${CHESS_DATA_DIR_HOST}' (expected train.parquet or train_[0-9]*.parquet)."
fi
if [ ! -f "${CHESS_DATA_DIR_HOST}/test.parquet" ] && ! compgen -G "${CHESS_DATA_DIR_HOST}/test_[0-9]*.parquet" >/dev/null; then
  die "Missing eval data under '${CHESS_DATA_DIR_HOST}' (expected test.parquet or test_[0-9]*.parquet)."
fi
if [ "${USE_HARD_DATASET}" = "True" ] && [ ! -f "${CHESS_DATA_DIR_HOST}/train_hard.parquet" ] && ! compgen -G "${CHESS_DATA_DIR_HOST}/train_hard_[0-9]*.parquet" >/dev/null; then
  die "USE_HARD_DATASET=True but missing train_hard.parquet or train_hard_[0-9]*.parquet under '${CHESS_DATA_DIR_HOST}'."
fi

# Chess reward shaping (passed through to train_chess.sh):
# - `train_chess.sh` consumes CHESS_REWARD_FN and forwards it as:
#     custom_reward_function.reward_kwargs.chess_reward_fn=${CHESS_REWARD_FN}
if [ -z "${CHESS_REWARD_FN:-}" ]; then
  CHESS_REWARD_FN="expected_score_wdl_vs_best"
fi
export CHESS_REWARD_FN
echo "[SLURM] CHESS_REWARD_FN=${CHESS_REWARD_FN}"

# Training length:
# - Default to 800 total steps (matches the reference full run).
# - Override explicitly via `CHESS_RL_TOTAL_TRAINING_STEPS` at submit time if you want a different step budget.
CHESS_RL_TOTAL_TRAINING_STEPS="${CHESS_RL_TOTAL_TRAINING_STEPS:-800}"
export TOTAL_TRAINING_STEPS="${CHESS_RL_TOTAL_TRAINING_STEPS}"
echo "[SLURM] TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"

# Optional data sampler seed (forwarded to `data.seed` in train_chess.sh).
CHESS_RL_DATA_SEED="${CHESS_RL_DATA_SEED:-null}"
export DATA_SEED="${CHESS_RL_DATA_SEED}"
echo "[SLURM] DATA_SEED=${DATA_SEED}"

# Checkpointing / resume:
# - `train_chess.sh` forwards these env vars to Hydra:
#     TRAINER_SAVE_FREQ   -> trainer.save_freq
#     TRAINER_RESUME_MODE -> trainer.resume_mode
#
# Isambard note (2026-01):
# - Full-precision FSDP checkpoints that include optimizer state are **very large** (tens of GB).
# - We observed Ray CPU-memory pressure during full-legal runs while a HF checkpoint upload
#   was in progress (large uploads + checkpoint materialization can be expensive on shared nodes).
#
# Small-legal experiment defaults:
# - Default save/full-eval cadence to 80 (the cadence used by current full runs).
# - Keep resume disabled unless explicitly requested.
# - Default to **model-only** actor checkpoints (no optimizer) when saving is enabled.
# - Default to **no HF upload** (opt-in) to avoid expensive / failure-prone network I/O on shared nodes.
CHESS_RL_TRAINER_SAVE_FREQ="${CHESS_RL_TRAINER_SAVE_FREQ:-80}"
export TRAINER_SAVE_FREQ="${CHESS_RL_TRAINER_SAVE_FREQ}"
export TRAINER_RESUME_MODE="${CHESS_RL_TRAINER_RESUME_MODE:-disable}"
echo "[SLURM] TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ}"
echo "[SLURM] TRAINER_RESUME_MODE=${TRAINER_RESUME_MODE}"

# Hugging Face checkpoint upload / auto-resume (optional):
# - The trainer uses HF token env vars in `verl/trainer/ppo/ray_trainer.py::_get_hf_token`.
# - Reference run enabled HF upload; disable by setting CHESS_RL_HF_UPLOAD_ENABLE=False.
export HF_CKPT_REPO_ID="${CHESS_RL_HF_CKPT_REPO_ID:-Gabr1e11/a_lot_of_models}"
CHESS_RL_HF_UPLOAD_ENABLE="${CHESS_RL_HF_UPLOAD_ENABLE:-True}"
HF_TOKEN_FILE="${CHESS_RL_HF_TOKEN_FILE:-${HOME}/.hf_token}"
echo "[SLURM] HF_CKPT_REPO_ID=${HF_CKPT_REPO_ID}"
echo "[SLURM] CHESS_RL_HF_UPLOAD_ENABLE=${CHESS_RL_HF_UPLOAD_ENABLE}"
echo "[SLURM] HF_TOKEN_FILE=${HF_TOKEN_FILE}"

HF_UPLOAD_ENABLED=0
case "${CHESS_RL_HF_UPLOAD_ENABLE}" in
  1|true|True|TRUE|yes|YES|y|Y)
    HF_UPLOAD_ENABLED=1
    # Default to HF-only retention (use local storage only as a staging area).
    # - The trainer will still write a local checkpoint folder first, then upload it.
    # - After a successful upload, it deletes the local checkpoint folder to avoid filling persistent storage.
    export VERL_HF_UPLOAD_DELETE_LOCAL="${VERL_HF_UPLOAD_DELETE_LOCAL:-1}"
    echo "[SLURM] VERL_HF_UPLOAD_DELETE_LOCAL=${VERL_HF_UPLOAD_DELETE_LOCAL}"

    # Prefer an explicit HF_TOKEN (passed via --export or set in the environment).
    # Otherwise, load it from a private token file.
    if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
      if [ -f "${HF_TOKEN_FILE}" ]; then
        export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
      else
        die "CHESS_RL_HF_UPLOAD_ENABLE=True but no token found (set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or create '${HF_TOKEN_FILE}' with chmod 600)."
      fi
    fi
    # Reference run used HF transfer disabled.
    CHESS_RL_HF_TRANSFER_ENABLE="${CHESS_RL_HF_TRANSFER_ENABLE:-0}"
    if [ "${CHESS_RL_HF_TRANSFER_ENABLE}" = "1" ]; then
      export HF_HUB_ENABLE_HF_TRANSFER=1
    else
      export HF_HUB_ENABLE_HF_TRANSFER=0
    fi
    echo "[SLURM] CHESS_RL_HF_TRANSFER_ENABLE=${CHESS_RL_HF_TRANSFER_ENABLE}"
    echo "[SLURM] HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"
    ;;
  *)
    # If HF upload isn't enabled, ensure we don't accidentally delete local checkpoints.
    export VERL_HF_UPLOAD_DELETE_LOCAL="${VERL_HF_UPLOAD_DELETE_LOCAL:-0}"
    echo "[SLURM] VERL_HF_UPLOAD_DELETE_LOCAL=${VERL_HF_UPLOAD_DELETE_LOCAL}"
    # Make accidental uploads impossible even if the submit environment had HF tokens and --export=ALL was used.
    unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGINGFACE_TOKEN || true
    ;;
esac

if [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ] || [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
  echo "[SLURM] HF_TOKEN_PRESENT=1"
else
  echo "[SLURM] HF_TOKEN_PRESENT=0"
fi

# Actor checkpoint contents:
# - Some cluster filesystems + offload configs have been flaky when saving optimizer state
#   (very large, high-write-rate checkpoints). Allow a safer mode that only checkpoints the
#   model weights + extra state.
# - This is wired via `train_chess.sh` which forwards:
#     ACTOR_CKPT_SAVE_CONTENTS -> actor_rollout_ref.actor.checkpoint.save_contents
#     ACTOR_CKPT_LOAD_CONTENTS -> actor_rollout_ref.actor.checkpoint.load_contents
# - IMPORTANT: Do NOT pass list-like values through `sbatch --export=...` because Slurm uses
#   commas as separators. Use the boolean switch below, or set ACTOR_CKPT_* in the shell env.
CHESS_RL_MODEL_ONLY_CKPT="${CHESS_RL_MODEL_ONLY_CKPT:-True}"
echo "[SLURM] CHESS_RL_MODEL_ONLY_CKPT=${CHESS_RL_MODEL_ONLY_CKPT}"
if [ "${CHESS_RL_MODEL_ONLY_CKPT}" = "True" ]; then
  export ACTOR_CKPT_SAVE_CONTENTS="[model,extra]"
  export ACTOR_CKPT_LOAD_CONTENTS="[model,extra]"
  echo "[SLURM] ACTOR_CKPT_SAVE_CONTENTS=${ACTOR_CKPT_SAVE_CONTENTS}"
  echo "[SLURM] ACTOR_CKPT_LOAD_CONTENTS=${ACTOR_CKPT_LOAD_CONTENTS}"
fi

# Full-game evaluation (vLLM + Stockfish):
# - Default cadence matches the small-legal full-run profile (every 80 steps).
# - Allow explicit override when a lighter smoke profile is desired.
# - Prompt template is the **canonical training prompt template** (the submission copies it).
# - Stockfish path must be ARM64 inside the container (`/usr/local/bin/stockfish` in v1-arm-stockfish-flashinfer).
CHESS_RL_FULL_EVAL_FREQ="${CHESS_RL_FULL_EVAL_FREQ:-80}"
export FULL_EVAL_FREQ="${CHESS_RL_FULL_EVAL_FREQ}"
CHESS_RL_FULL_EVAL_OPPONENT_DEPTHS="${CHESS_RL_FULL_EVAL_OPPONENT_DEPTHS:-1,5}"
export FULL_EVAL_OPPONENT_DEPTHS="${CHESS_RL_FULL_EVAL_OPPONENT_DEPTHS}"
export FULL_EVAL_ROUNDS="${CHESS_RL_FULL_EVAL_ROUNDS:-1}"
export FULL_EVAL_GAMES_PER_ROUND="${CHESS_RL_FULL_EVAL_GAMES_PER_ROUND:-50}"
CHESS_RL_FULL_EVAL_GAMES_PER_DEPTH="${CHESS_RL_FULL_EVAL_GAMES_PER_DEPTH:-$((FULL_EVAL_ROUNDS * FULL_EVAL_GAMES_PER_ROUND))}"
export FULL_EVAL_GAMES_PER_DEPTH="${CHESS_RL_FULL_EVAL_GAMES_PER_DEPTH}"
# Keep full-game-eval sampling aligned with validation-style generation by default.
# - `train_chess.sh` uses CHESS_RL_VAL_{TEMPERATURE,TOP_P} (defaults 0.6/0.95) for validation sampling.
# - When MAX_RESPONSE_LENGTH is set (common in experiment launchers), default full-game max tokens to it.
export FULL_EVAL_TEMPERATURE="${CHESS_RL_FULL_EVAL_TEMPERATURE:-${CHESS_RL_VAL_TEMPERATURE:-0.6}}"
export FULL_EVAL_TOP_P="${CHESS_RL_FULL_EVAL_TOP_P:-${CHESS_RL_VAL_TOP_P:-0.95}}"
export FULL_EVAL_MAX_RESPONSE_TOKENS="${CHESS_RL_FULL_EVAL_MAX_RESPONSE_TOKENS:-${MAX_RESPONSE_LENGTH:-}}"
export FULL_EVAL_MAX_RETRIES_PER_TURN="${CHESS_RL_FULL_EVAL_MAX_RETRIES_PER_TURN:-1}"
export FULL_EVAL_STOCKFISH_PATH="${CHESS_RL_FULL_EVAL_STOCKFISH_PATH:-/usr/local/bin/stockfish}"
export FULL_EVAL_STOCKFISH_THREADS="${CHESS_RL_FULL_EVAL_STOCKFISH_THREADS:-1}"
export FULL_EVAL_STOCKFISH_HASH_MB="${CHESS_RL_FULL_EVAL_STOCKFISH_HASH_MB:-128}"
export FULL_EVAL_ACPL_WORKERS="${CHESS_RL_FULL_EVAL_ACPL_WORKERS:-72}"
export FULL_EVAL_ACPL_THREADS="${CHESS_RL_FULL_EVAL_ACPL_THREADS:-2}"
echo "[SLURM] FULL_EVAL_FREQ=${FULL_EVAL_FREQ}"
echo "[SLURM] FULL_EVAL_OPPONENT_DEPTHS=${FULL_EVAL_OPPONENT_DEPTHS}"
echo "[SLURM] FULL_EVAL_ROUNDS=${FULL_EVAL_ROUNDS}"
echo "[SLURM] FULL_EVAL_GAMES_PER_ROUND=${FULL_EVAL_GAMES_PER_ROUND}"
echo "[SLURM] FULL_EVAL_GAMES_PER_DEPTH=${FULL_EVAL_GAMES_PER_DEPTH}"
echo "[SLURM] FULL_EVAL_TEMPERATURE=${FULL_EVAL_TEMPERATURE}"
echo "[SLURM] FULL_EVAL_TOP_P=${FULL_EVAL_TOP_P}"
echo "[SLURM] FULL_EVAL_MAX_RESPONSE_TOKENS=${FULL_EVAL_MAX_RESPONSE_TOKENS}"
echo "[SLURM] FULL_EVAL_MAX_RETRIES_PER_TURN=${FULL_EVAL_MAX_RETRIES_PER_TURN}"
echo "[SLURM] FULL_EVAL_STOCKFISH_PATH=${FULL_EVAL_STOCKFISH_PATH}"
echo "[SLURM] FULL_EVAL_ACPL_WORKERS=${FULL_EVAL_ACPL_WORKERS}"
echo "[SLURM] FULL_EVAL_ACPL_THREADS=${FULL_EVAL_ACPL_THREADS}"

# vLLM + FSDP performance knobs (passed through to train_chess.sh):
# - Disable eager mode and cache-engine freeing by default for GH200 runs.
# - Enable CPU/NVMe offload by default (can help fit larger configs; adjust as needed).
# - Override any inherited values from `--export=ALL` to keep behavior predictable.
export ENFORCE_EAGER=False
export FREE_CACHE_ENGINE=True
# Default to 8 rollouts (group_size) to match the reference n8/r4 run.
export ROLLOUT_N="${ROLLOUT_N:-8}"
export PARAM_OFFLOAD=False
# FSDP param offload (actor vs ref):
# - A small GH200 10-step matrix (2026-01-04) found the fastest configuration to be:
#     actor.fsdp_config.param_offload = False
#     ref.fsdp_config.param_offload   = True
# - Use CHESS_RL_* to avoid inheriting interactive env values from `--export=ALL`.
CHESS_RL_ACTOR_PARAM_OFFLOAD="${CHESS_RL_ACTOR_PARAM_OFFLOAD:-False}"
export ACTOR_PARAM_OFFLOAD="${CHESS_RL_ACTOR_PARAM_OFFLOAD}"
CHESS_RL_REF_PARAM_OFFLOAD="${CHESS_RL_REF_PARAM_OFFLOAD:-False}"
export REF_PARAM_OFFLOAD="${CHESS_RL_REF_PARAM_OFFLOAD}"
export OPTIMIZER_OFFLOAD=False
export PPO_MICRO_BATCH_SIZE="${CHESS_RL_PPO_MICRO_BATCH_SIZE:-8}"
export ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="${CHESS_RL_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-16}"
export REF_LOGPROB_MICRO_BATCH_SIZE="${CHESS_RL_REF_LOGPROB_MICRO_BATCH_SIZE:-16}"
echo "[SLURM] ENFORCE_EAGER=${ENFORCE_EAGER}"
echo "[SLURM] FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE}"
echo "[SLURM] PARAM_OFFLOAD=${PARAM_OFFLOAD}"
echo "[SLURM] ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD}"
echo "[SLURM] REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD}"
echo "[SLURM] OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD}"
echo "[SLURM] PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE}"
echo "[SLURM] ROLLOUT_LOGPROB_MICRO_BATCH_SIZE=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE}"
echo "[SLURM] REF_LOGPROB_MICRO_BATCH_SIZE=${REF_LOGPROB_MICRO_BATCH_SIZE}"

SIF_URI="${CHESS_RL_IMAGE_URI:-}"
SIF_IMAGE="${CHESS_RL_SIF_IMAGE:-/projects/a5l/ziyan/sif-images/chess_rl_v4-arm-stockfish-pytorch-25.05.sif}"
CONTAINER_REPO_DIR="${CHESS_RL_CONTAINER_REPO_DIR:-/workspace/chess_rl}"
echo "[SLURM] Image URI:   ${SIF_URI:-<unset>}"
echo "[SLURM] Image SIF:   ${SIF_IMAGE}"
echo "[SLURM] Repo mount:  ${CONTAINER_REPO_DIR}"

# Output directory (VERL_BASE_DIR):
# - Checkpoints can be **tens of GB** per save.
# - On Isambard compute nodes, `/tmp` is typically `tmpfs` (RAM-backed). Writing large checkpoints to TMPDIR
#   can consume CPU memory and trigger Ray's memory monitor.
# - Safer default:
#   - if checkpoint saving is enabled (TRAINER_SAVE_FREQ>0) and HF upload is disabled: write outputs to `/projects` (persistent FS)
#   - otherwise: write to job TMPDIR (fast, ephemeral; good for smoke runs)
# - Override at submit time if you want a custom persistent output location:
#     sbatch --export=ALL,CHESS_RL_VERL_BASE_DIR=/path/on/persistent/fs/my_run ./sbatch_chess_small_legal_gh200.slurm
if [ -n "${CHESS_RL_VERL_BASE_DIR:-}" ]; then
  export VERL_BASE_DIR="${CHESS_RL_VERL_BASE_DIR}"
  echo "[SLURM] VERL_BASE_DIR=${VERL_BASE_DIR} (override)"
else
  if [[ "${TRAINER_SAVE_FREQ}" =~ ^-?[0-9]+$ ]] && [ "${TRAINER_SAVE_FREQ}" -gt 0 ]; then
    if [ "${HF_UPLOAD_ENABLED}" = "1" ]; then
      # HF upload enabled:
      # - Single node: prefer node-local scratch for fast staging.
      # - Multi-node: use a shared filesystem so all ranks can see checkpoint shards.
      if [ "${JOB_NUM_NODES}" -gt 1 ] && [ -d "/projects" ]; then
        CHESS_RL_PROJECTS_OUTPUT_ROOT="${CHESS_RL_PROJECTS_OUTPUT_ROOT:-/projects/a5l/ziyan/chess_rl_outputs}"
        export VERL_BASE_DIR="${CHESS_RL_PROJECTS_OUTPUT_ROOT%/}/train_${SLURM_JOB_ID:-$$}"
        mkdir -p "${VERL_BASE_DIR}"
        echo "[SLURM] VERL_BASE_DIR=${VERL_BASE_DIR} (HF upload enabled; shared /projects for multi-node)"
      else
        if [ -n "${SLURM_TMPDIR:-}" ]; then
          export VERL_BASE_DIR="${SLURM_TMPDIR%/}/chess_rl_outputs_${SLURM_JOB_ID:-$$}"
        elif [ -d "/var/tmp" ]; then
          export VERL_BASE_DIR="/var/tmp/${USER}/chess_rl_outputs_${SLURM_JOB_ID:-$$}"
        elif [ -d "/projects" ]; then
          # Last resort: persistent storage (still better than tmpfs).
          CHESS_RL_PROJECTS_OUTPUT_ROOT="${CHESS_RL_PROJECTS_OUTPUT_ROOT:-/projects/a5l/ziyan/chess_rl_outputs}"
          export VERL_BASE_DIR="${CHESS_RL_PROJECTS_OUTPUT_ROOT%/}/train_${SLURM_JOB_ID:-$$}"
        else
          export VERL_BASE_DIR="${TMPDIR%/}/chess_rl_outputs_${SLURM_JOB_ID:-$$}"
        fi
        mkdir -p "${VERL_BASE_DIR}"
        echo "[SLURM] VERL_BASE_DIR=${VERL_BASE_DIR} (HF upload enabled; node-local staging)"
      fi
    else
      # HF upload disabled: checkpoint files are the source-of-truth, so write them to a persistent filesystem.
      if [ -d "/projects" ]; then
        CHESS_RL_PROJECTS_OUTPUT_ROOT="${CHESS_RL_PROJECTS_OUTPUT_ROOT:-/projects/a5l/ziyan/chess_rl_outputs}"
        export VERL_BASE_DIR="${CHESS_RL_PROJECTS_OUTPUT_ROOT%/}/train_${SLURM_JOB_ID:-$$}"
        mkdir -p "${VERL_BASE_DIR}"
        echo "[SLURM] VERL_BASE_DIR=${VERL_BASE_DIR} (/projects default; TRAINER_SAVE_FREQ>0)"
      else
        export VERL_BASE_DIR="${TMPDIR%/}/chess_rl_outputs_${SLURM_JOB_ID:-$$}"
        mkdir -p "${VERL_BASE_DIR}"
        echo "[SLURM] VERL_BASE_DIR=${VERL_BASE_DIR} (TMPDIR default; TRAINER_SAVE_FREQ>0 but /projects missing)"
      fi
    fi
  else
    export VERL_BASE_DIR="${TMPDIR%/}/chess_rl_outputs_${SLURM_JOB_ID:-$$}"
    mkdir -p "${VERL_BASE_DIR}"
    echo "[SLURM] VERL_BASE_DIR=${VERL_BASE_DIR} (TMPDIR default)"
  fi
fi

# W&B local run directory:
# - Keep `wandb/` off $HOME by default (same quota motivation as VERL_BASE_DIR).
# - W&B still syncs runs/artifacts to the server in online mode.
if [ -n "${CHESS_RL_WANDB_DIR:-}" ]; then
  export WANDB_DIR="${CHESS_RL_WANDB_DIR}"
else
  export WANDB_DIR="${VERL_BASE_DIR%/}/wandb"
fi
mkdir -p "${WANDB_DIR}"
echo "[SLURM] WANDB_DIR=${WANDB_DIR}"

# Hugging Face / Transformers caches:
# - Avoid writing large model/ckpt caches into $HOME (quota risk on shared filesystems).
# - Use the job-scoped VERL_BASE_DIR (TMPDIR by default).
export HF_HOME="${VERL_BASE_DIR%/}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"
echo "[SLURM] HF_HOME=${HF_HOME}"

# Full-game evaluation prompt template (forwarded to train_chess.sh).
# - Default to the small-legal prompt to match training and the reference run.
CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH="${CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH:-recipe/chess/prompt_templates/select_prompt_small_legal.jinja}"
export FULL_EVAL_PROMPT_TEMPLATE_PATH="${CHESS_RL_FULL_EVAL_PROMPT_TEMPLATE_PATH}"
echo "[SLURM] FULL_EVAL_PROMPT_TEMPLATE_PATH=${FULL_EVAL_PROMPT_TEMPLATE_PATH}"

CHESS_RL_CONTAINER_RUNTIME="${CHESS_RL_CONTAINER_RUNTIME:-apptainer}"
CONTAINER_RUNTIME="${CHESS_RL_CONTAINER_RUNTIME}"
if [ "${CONTAINER_RUNTIME}" = "podman-hpc" ]; then
  if ! command -v podman-hpc >/dev/null 2>&1; then
    die "CHESS_RL_CONTAINER_RUNTIME=podman-hpc but podman-hpc is not on PATH."
  fi
  CONTAINER_IMAGE="${SIF_URI#docker://}"
  PODMAN_OPENMPI_FLAG="--openmpi-pmi2"
  PODMAN_ENV_ARGS=(--env-host)
  # Avoid Triton JIT linking against host libc via /host/usr/lib64 on Isambard.
  # Keep the search path container-local so ld doesn't pick RHEL-style lib64 paths.
  if [ -z "${TRITON_LIBRARY_PATH:-}" ]; then
    export TRITON_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/lib"
  fi
  export TRITON_CUDA_PATH="${TRITON_CUDA_PATH:-/usr/local/cuda}"
  if [ -z "${TRITON_LIBCUDA_PATH:-}" ] && [ -n "${CHESS_RL_TRITON_LIBCUDA_PATH:-}" ]; then
    export TRITON_LIBCUDA_PATH="${CHESS_RL_TRITON_LIBCUDA_PATH}"
  fi
  if [ -n "${CONTAINERS_CONF:-}" ] && [ ! -f "${CONTAINERS_CONF}" ]; then
    echo "[SLURM] CONTAINERS_CONF missing at ${CONTAINERS_CONF}; creating."
    mkdir -p "$(dirname "${CONTAINERS_CONF}")"
    cat > "${CONTAINERS_CONF}" <<'EOF'
[engine]
lock_type = "file"
EOF
  fi
  echo "[SLURM] Container runtime: podman-hpc ($(podman-hpc --version 2>/dev/null || echo 'version-unknown'))"
  echo "[SLURM] Container image:  ${CONTAINER_IMAGE}"
else
  SINGULARITY_BIN="${SINGULARITY_BIN:-}"
  if [ -z "${SINGULARITY_BIN}" ]; then
    if command -v apptainer >/dev/null 2>&1; then
      SINGULARITY_BIN="apptainer"
    elif command -v singularity >/dev/null 2>&1; then
      SINGULARITY_BIN="singularity"
    fi
  fi
  if [ -z "${SINGULARITY_BIN}" ]; then
    cat >&2 <<'EOF'
[ERROR] Could not find a container runtime on PATH (expected 'apptainer' or 'singularity').
Common fixes:
  - `module load apptainer` (or your site-specific module name)
  - ensure your cluster supports --nv GPU passthrough
EOF
    exit 1
  fi
  echo "[SLURM] Container runtime: ${SINGULARITY_BIN} ($(${SINGULARITY_BIN} --version 2>/dev/null || echo 'version-unknown'))"
fi

CONTAINER_BASH_PREFIX=""
if [ "${CONTAINER_RUNTIME}" = "podman-hpc" ]; then
  # podman-hpc --openmpi-pmi2 injects /host/adapt.sh automatically.
  NCCL_DEBUG_ORIG="${NCCL_DEBUG:-}"
  NCCL_DEBUG_SUBSYS_ORIG="${NCCL_DEBUG_SUBSYS:-}"
  CONTAINER_BASH_PREFIX="LD_LIBRARY_PATH=\"\$(echo \"\$LD_LIBRARY_PATH\" | tr ':' '\\n' | grep -E -v '^/host/opt/nvidia/hpc_sdk/.*/cuda/.*/lib64$' | paste -sd ':' -)\"; export LD_LIBRARY_PATH;"
  if [ -n "${NCCL_DEBUG_ORIG}" ]; then
    CONTAINER_BASH_PREFIX="${CONTAINER_BASH_PREFIX} export NCCL_DEBUG='${NCCL_DEBUG_ORIG}';"
  fi
  if [ -n "${NCCL_DEBUG_SUBSYS_ORIG}" ]; then
    CONTAINER_BASH_PREFIX="${CONTAINER_BASH_PREFIX} export NCCL_DEBUG_SUBSYS='${NCCL_DEBUG_SUBSYS_ORIG}';"
  fi
  CONTAINER_BASH_PREFIX="${CONTAINER_BASH_PREFIX} "
else
  # Optional: Isambard multi-node Apptainer adaptation (MPI/NCCL bind paths).
  # - Default is auto (enable when multi-node; preserves single-node behavior).
  # - True: load the module and source /host/adapt.sh inside the container.
  # - auto: enable only when JOB_NUM_NODES > 1.
  CHESS_RL_APPTAINER_MULTI_NODE_ENABLE="${CHESS_RL_APPTAINER_MULTI_NODE_ENABLE:-auto}"
  APPTAINER_MULTI_NODE_ENABLED=0
  case "${CHESS_RL_APPTAINER_MULTI_NODE_ENABLE}" in
    auto|Auto|AUTO)
      if [ "${JOB_NUM_NODES}" -gt 1 ]; then
        APPTAINER_MULTI_NODE_ENABLED=1
      fi
      ;;
    1|true|True|TRUE|yes|YES|y|Y)
      APPTAINER_MULTI_NODE_ENABLED=1
      ;;
    *)
      APPTAINER_MULTI_NODE_ENABLED=0
      ;;
  esac
  echo "[SLURM] CHESS_RL_APPTAINER_MULTI_NODE_ENABLE=${CHESS_RL_APPTAINER_MULTI_NODE_ENABLE} (enabled=${APPTAINER_MULTI_NODE_ENABLED})"
  if [ "${APPTAINER_MULTI_NODE_ENABLED}" -eq 1 ]; then
    if ! type module >/dev/null 2>&1; then
      if [ -f /etc/profile.d/modules.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
      fi
    fi
    if ! type module >/dev/null 2>&1; then
      die "CHESS_RL_APPTAINER_MULTI_NODE_ENABLE=True but the 'module' command is unavailable. Load the site module system or disable multi-node adaptation."
    fi
    echo "[SLURM] Loading module: brics/apptainer-multi-node"
    if ! module load brics/apptainer-multi-node; then
      die "Failed to load brics/apptainer-multi-node. Disable CHESS_RL_APPTAINER_MULTI_NODE_ENABLE or fix module availability."
    fi
    echo "[SLURM] SINGULARITY_BINDPATH=${SINGULARITY_BINDPATH:-UNSET}"
    echo "[SLURM] APPTAINER_BINDPATH=${APPTAINER_BINDPATH:-UNSET}"
  fi

  if [ "${APPTAINER_MULTI_NODE_ENABLED}" -eq 1 ]; then
    NCCL_DEBUG_ORIG="${NCCL_DEBUG:-}"
    NCCL_DEBUG_SUBSYS_ORIG="${NCCL_DEBUG_SUBSYS:-}"
    CONTAINER_BASH_PREFIX="source /host/adapt.sh && if [ ! -e /lib64 ]; then ln -s /lib/aarch64-linux-gnu /lib64; fi; if [ ! -e /usr/lib64 ]; then ln -s /usr/lib/aarch64-linux-gnu /usr/lib64; fi; LD_LIBRARY_PATH=\"\$(echo \"\$LD_LIBRARY_PATH\" | tr ':' '\\n' | grep -E -v '^/host/opt/nvidia/hpc_sdk/.*/cuda/.*/lib64$' | paste -sd ':' -)\"; export LD_LIBRARY_PATH;"
    if [ -n "${NCCL_DEBUG_ORIG}" ]; then
      CONTAINER_BASH_PREFIX="${CONTAINER_BASH_PREFIX} export NCCL_DEBUG='${NCCL_DEBUG_ORIG}';"
    fi
    if [ -n "${NCCL_DEBUG_SUBSYS_ORIG}" ]; then
      CONTAINER_BASH_PREFIX="${CONTAINER_BASH_PREFIX} export NCCL_DEBUG_SUBSYS='${NCCL_DEBUG_SUBSYS_ORIG}';"
    fi
    CONTAINER_BASH_PREFIX="${CONTAINER_BASH_PREFIX} "
  fi
fi

APPTAINER_TMPFS_FLAG=()
if [ "${CONTAINER_RUNTIME}" != "podman-hpc" ] && [ "${APPTAINER_MULTI_NODE_ENABLED:-0}" -eq 1 ]; then
  APPTAINER_TMPFS_FLAG=(--writable-tmpfs)
fi

if [ "${CONTAINER_RUNTIME}" = "podman-hpc" ]; then
  # For podman-hpc we rely on the local image store (or auto-pull on first run).
  SANITY_IMPORT_LOG="$(podman-hpc run --rm --gpu ${PODMAN_OPENMPI_FLAG} "${PODMAN_ENV_ARGS[@]}" "${CONTAINER_IMAGE}" \
    bash -lc "${CONTAINER_BASH_PREFIX}python3 -c \"import torch; import chess; import flash_attn; import flashinfer; print('ok:', torch.__version__, flashinfer.__version__)\"" 2>&1 || true)"
  if ! echo "${SANITY_IMPORT_LOG}" | grep -q "^ok:"; then
    echo "[ERROR] Container sanity-check failed (imports: torch/python-chess/flash-attn/flashinfer)." >&2
    echo "[ERROR] Import log (stdout/stderr) follows:" >&2
    echo "${SANITY_IMPORT_LOG}" >&2
    exit 1
  fi
  echo "[SLURM] Container sanity: ${SANITY_IMPORT_LOG}"
else
  mkdir -p "$(dirname "${SIF_IMAGE}")"
  if [ ! -f "${SIF_IMAGE}" ]; then
    die "SIF image not found at '${SIF_IMAGE}'. Set CHESS_RL_SIF_IMAGE to an existing .sif (no auto-build/pull)."
  fi
  # Fail fast if the SIF is corrupted or unreadable.
  SIF_CHECK_LOG="$("${SINGULARITY_BIN}" exec "${SIF_IMAGE}" true 2>&1 || true)"
  if echo "${SIF_CHECK_LOG}" | grep -qi "image format not recognized"; then
    die "SIF image is corrupted (image format not recognized): ${SIF_IMAGE}"
  elif echo "${SIF_CHECK_LOG}" | grep -qi "could not open image"; then
    die "SIF image is unreadable (could not open image): ${SIF_IMAGE}"
  fi

  SIF_SANITY_IMAGE="${SIF_IMAGE}"

  # If multi-node adaptation is enabled, /host/adapt.sh must be visible inside the container.
  if [ "${APPTAINER_MULTI_NODE_ENABLED}" -eq 1 ]; then
    if ! "${SINGULARITY_BIN}" exec --nv "${SIF_SANITY_IMAGE}" bash -lc "test -f /host/adapt.sh"; then
      die "CHESS_RL_APPTAINER_MULTI_NODE_ENABLE=True but /host/adapt.sh is not visible inside the container. Ensure the brics/apptainer-multi-node module is loaded and exported."
    fi
  fi

  # Quick image sanity checks (fail fast with actionable errors).
  SANITY_IMPORT_LOG="$("${SINGULARITY_BIN}" exec --nv "${SIF_SANITY_IMAGE}" bash -lc "${CONTAINER_BASH_PREFIX}python3 -c \"import torch; import chess; import flash_attn; import flashinfer; print('ok:', torch.__version__, flashinfer.__version__)\"" 2>&1 || true)"
  if ! echo "${SANITY_IMPORT_LOG}" | grep -q "^ok:"; then
    echo "[ERROR] Container sanity-check failed (imports: torch/python-chess/flash-attn/flashinfer)." >&2
    echo "[ERROR] Import log (stdout/stderr) follows:" >&2
    echo "${SANITY_IMPORT_LOG}" >&2
    cat >&2 <<EOF
This usually indicates a broken dependency stack inside the image (e.g., flash-attn/torch mismatch),
or a broken Apptainer/SIF build (often caused by building on a filesystem without xattrs).

Try inspecting inside the container:
  ${SINGULARITY_BIN} exec --nv "${SIF_IMAGE}" bash -lc "python3 -c 'import torch; print(torch.__version__)'"
  ${SINGULARITY_BIN} exec --nv "${SIF_IMAGE}" bash -lc "python3 -c 'import flash_attn; print(flash_attn.__version__)'"
  ${SINGULARITY_BIN} exec --nv "${SIF_IMAGE}" bash -lc "python3 -c 'import flashinfer; print(flashinfer.__version__)'"
EOF
    exit 1
  fi
  echo "[SLURM] Container sanity: ${SANITY_IMPORT_LOG}"
  if [ "${TRAINER_SAVE_FREQ}" -gt 0 ] || [ "${TRAINER_RESUME_MODE}" = "auto" ]; then
    if ! "${SINGULARITY_BIN}" exec --nv "${SIF_IMAGE}" python3 -c "import huggingface_hub; print('hf_ok')" >/dev/null 2>&1; then
      die "Hugging Face checkpointing/resume is enabled but 'huggingface_hub' is not importable inside the container image."
    fi
  fi
fi

# Sanity check the training script exists (host-side; it will be bind-mounted).
if [ ! -f "${THIS_DIR}/train_chess.sh" ]; then
  die "Missing training launcher: '${THIS_DIR}/train_chess.sh'. Submit the job from inside the repo checkout."
fi
# Sanity check prompt templates exist (host-side; they will be bind-mounted into the container).
if [ ! -f "${THIS_DIR}/recipe/chess/prompt_templates/original_chessr1_prompt.jinja" ]; then
  die "Missing prompt template: '${THIS_DIR}/recipe/chess/prompt_templates/original_chessr1_prompt.jinja'."
fi
if [ ! -f "${THIS_DIR}/recipe/chess/prompt_templates/select_prompt.jinja" ]; then
  die "Missing prompt template: '${THIS_DIR}/recipe/chess/prompt_templates/select_prompt.jinja'."
fi

# Forced-prefix knobs (passed through to train_chess.sh)
# - Selection framing disables this path by default; keep zeros unless explicitly enabled.
export FORCED_PREFIX_ENABLE="${FORCED_PREFIX_ENABLE:-False}"
export FORCED_PREFIX_PROB_START="${FORCED_PREFIX_PROB_START:-0.0}"
export FORCED_PREFIX_PROB_END="${FORCED_PREFIX_PROB_END:-0.0}"
export FORCED_PREFIX_ANNEAL_FRAC="${FORCED_PREFIX_ANNEAL_FRAC:-0.5}"
export FORCED_PREFIX_TEMP="${FORCED_PREFIX_TEMP:-2.0}"

# Optional hard switch for forced-prefix exploration:
# - When disabled, force pstart=pend=0 so no prompt groups are explored.
case "${FORCED_PREFIX_ENABLE}" in
  1|true|True|TRUE|yes|YES|y|Y) ;;
  *)
    export FORCED_PREFIX_PROB_START="0.0"
    export FORCED_PREFIX_PROB_END="0.0"
    ;;
esac

echo "[SLURM] FORCED_PREFIX_ENABLE=${FORCED_PREFIX_ENABLE}"
echo "[SLURM] FORCED_PREFIX_PROB_START=${FORCED_PREFIX_PROB_START}"
echo "[SLURM] FORCED_PREFIX_PROB_END=${FORCED_PREFIX_PROB_END}"
echo "[SLURM] FORCED_PREFIX_ANNEAL_FRAC=${FORCED_PREFIX_ANNEAL_FRAC}"
echo "[SLURM] FORCED_PREFIX_TEMP=${FORCED_PREFIX_TEMP}"
# KL-to-reference loss toggle (passed through to train_chess.sh):
# - When False (and `algorithm.use_kl_in_reward=False`), the trainer will not create a reference-policy worker.
CHESS_RL_USE_KL_LOSS="${CHESS_RL_USE_KL_LOSS:-True}"
export USE_KL_LOSS="${CHESS_RL_USE_KL_LOSS}"
echo "[SLURM] USE_KL_LOSS=${USE_KL_LOSS}"

# FSDP strategy + remove-padding toggles (passed through to train_chess.sh):
# - Default to `fsdp` for the simplest distributed path during debugging.
# - Remove-padding can be disabled to avoid flash-attn varlen packing complexity.
CHESS_RL_FSDP_STRATEGY="${CHESS_RL_FSDP_STRATEGY:-fsdp2}"
export FSDP_STRATEGY="${CHESS_RL_FSDP_STRATEGY}"
echo "[SLURM] FSDP_STRATEGY=${FSDP_STRATEGY}"
CHESS_RL_USE_REMOVE_PADDING="${CHESS_RL_USE_REMOVE_PADDING:-True}"
export USE_REMOVE_PADDING="${CHESS_RL_USE_REMOVE_PADDING}"
echo "[SLURM] USE_REMOVE_PADDING=${USE_REMOVE_PADDING}"
echo "[SLURM] ROLLOUT_N=${ROLLOUT_N}"

# This job requests 4 GPUs per node; default the trainer to using all of them.
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
if ! [[ "${TRAINER_N_GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || [ "${TRAINER_N_GPUS_PER_NODE}" -le 0 ]; then
  warn "Invalid TRAINER_N_GPUS_PER_NODE='${TRAINER_N_GPUS_PER_NODE}'; defaulting to 4."
  TRAINER_N_GPUS_PER_NODE=4
  export TRAINER_N_GPUS_PER_NODE
fi
echo "[SLURM] TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE}"
if [ "${TRAINER_N_GPUS_PER_NODE}" -ne 4 ]; then
  warn "TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE} but this Slurm script requests 4 GPUs. Ensure these match for best performance."
fi
export TRAINER_NNODES="${TRAINER_NNODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}}"
if ! [[ "${TRAINER_NNODES}" =~ ^[0-9]+$ ]] || [ "${TRAINER_NNODES}" -le 0 ]; then
  warn "Invalid TRAINER_NNODES='${TRAINER_NNODES}'; defaulting to 1."
  TRAINER_NNODES=1
  export TRAINER_NNODES
fi
echo "[SLURM] TRAINER_NNODES=${TRAINER_NNODES}"
if [ "${TRAINER_NNODES}" -ne "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}" ]; then
  warn "TRAINER_NNODES=${TRAINER_NNODES} but SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}. Ensure these match for best performance."
fi
echo "[SLURM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "[SLURM] Launching training inside container..."
# Repo staging (robustness against flaky/shared filesystems):
# - The v2 run `8i75c3v4` died mid-training with:
#     `./train_chess.sh: error reading input file: Stale file handle`
#   while executing `./train_chess.sh` from the bind-mounted submit directory.
# - Mitigation: copy the repo (code only) to node-local scratch and execute from there.
CHESS_RL_STAGE_REPO="${CHESS_RL_STAGE_REPO:-True}"
RUN_REPO_DIR="${THIS_DIR}"
case "${CHESS_RL_STAGE_REPO}" in
  1|true|True|TRUE|yes|YES|y|Y)
    STAGE_PARENT="${SLURM_TMPDIR:-${TMPDIR}}"
    STAGED_REPO_DIR="${CHESS_RL_STAGED_REPO_DIR:-${STAGE_PARENT%/}/chess_rl_repo_${SLURM_JOB_ID:-$$}}"
    RUN_REPO_DIR="${STAGED_REPO_DIR}"

    STAGE_SRC="${THIS_DIR}"
    export STAGE_SRC STAGED_REPO_DIR
    STAGE_SCRIPT='set -euo pipefail
mkdir -p "${STAGED_REPO_DIR}"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete \
    --exclude "/.git" \
    --exclude "/data" \
    --exclude "/outputs" \
    --exclude "/wandb" \
    --exclude "/slurm" \
    "${STAGE_SRC%/}/" "${STAGED_REPO_DIR%/}/"
else
  # Fallback: tar streaming copy with excludes (keeps staging lightweight).
  tar -C "${STAGE_SRC}" -cf - \
    --exclude="./.git" \
    --exclude="./data" \
    --exclude="./outputs" \
    --exclude="./wandb" \
    --exclude="./slurm" \
    . | tar -C "${STAGED_REPO_DIR}" -xf -
fi
# Ensure mountpoints exist inside the staged tree.
mkdir -p "${STAGED_REPO_DIR}/data"
'

    if command -v scontrol >/dev/null 2>&1 && [ -n "${SLURM_JOB_NODELIST:-}" ] && command -v srun >/dev/null 2>&1; then
      nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
      nodes_array=($nodes)
      for node in "${nodes_array[@]}"; do
        echo "[SLURM] Staging repo on ${node}: ${STAGE_SRC} -> ${STAGED_REPO_DIR}"
        srun --nodes=1 --ntasks=1 -w "${node}" --export=ALL bash -lc "${STAGE_SCRIPT}"
      done
    else
      echo "[SLURM] Staging repo locally: ${STAGE_SRC} -> ${STAGED_REPO_DIR}"
      bash -lc "${STAGE_SCRIPT}"
    fi
    ;;
  *)
    echo "[SLURM] CHESS_RL_STAGE_REPO=False (running from submit dir)"
    ;;
esac
echo "[SLURM] RUN_REPO_DIR=${RUN_REPO_DIR}"

# Apptainer does **not** automatically bind `/projects` on some Isambard nodes.
# If we want persistent outputs under `/projects/...` (recommended), we must bind it explicitly.
# Prefer binding the user subdir if it exists (avoids permission issues under podman-hpc).
BIND_ARGS=(--bind "${RUN_REPO_DIR}:${CONTAINER_REPO_DIR}")
# Dataset files are typically stored under the submit-dir repo tree (`${THIS_DIR}/data/...`).
# Bind them separately so staging can exclude large Parquet files.
if [ -d "${THIS_DIR}/data" ]; then
  BIND_ARGS+=(--bind "${THIS_DIR}/data:${CONTAINER_REPO_DIR}/data")
fi
PROJECTS_BIND=""
if [ -d "/projects/a5l/ziyan" ]; then
  PROJECTS_BIND="/projects/a5l/ziyan"
elif [ -d "/projects" ]; then
  PROJECTS_BIND="/projects"
fi
if [ -n "${PROJECTS_BIND}" ]; then
  BIND_ARGS+=(--bind "${PROJECTS_BIND}:${PROJECTS_BIND}")
fi
if [ "${LOCAL_SCRATCH_OK}" -eq 1 ]; then
  LOCAL_SCRATCH_BIND="${LOCAL_SCRATCH_BASE%/}/${USER}"
  if [ -d "${LOCAL_SCRATCH_BIND}" ]; then
    BIND_ARGS+=(--bind "${LOCAL_SCRATCH_BIND}:${LOCAL_SCRATCH_BIND}")
  fi
fi

PODMAN_BIND_ARGS=(-v "${RUN_REPO_DIR}:${CONTAINER_REPO_DIR}")
if [ -d "${THIS_DIR}/data" ]; then
  PODMAN_BIND_ARGS+=(-v "${THIS_DIR}/data:${CONTAINER_REPO_DIR}/data")
fi
if [ -n "${PROJECTS_BIND}" ]; then
  PODMAN_BIND_ARGS+=(-v "${PROJECTS_BIND}:${PROJECTS_BIND}")
fi
if [ "${LOCAL_SCRATCH_OK}" -eq 1 ]; then
  LOCAL_SCRATCH_BIND="${LOCAL_SCRATCH_BASE%/}/${USER}"
  if [ -d "${LOCAL_SCRATCH_BIND}" ]; then
    PODMAN_BIND_ARGS+=(-v "${LOCAL_SCRATCH_BIND}:${LOCAL_SCRATCH_BIND}")
  fi
fi
if [ -n "${UID:-}" ] && [ -d "/local/user/${UID}" ]; then
  PODMAN_BIND_ARGS+=(-v "/local/user/${UID}:/local/user/${UID}")
fi
if [ "${CONTAINER_RUNTIME}" = "podman-hpc" ]; then
  CONTAINER_LAUNCH=(podman-hpc run --rm --gpu ${PODMAN_OPENMPI_FLAG} "${PODMAN_ENV_ARGS[@]}" "${PODMAN_BIND_ARGS[@]}" -w "${CONTAINER_REPO_DIR}" "${CONTAINER_IMAGE}")
else
  CONTAINER_LAUNCH=("${SINGULARITY_BIN}" exec --nv "${APPTAINER_TMPFS_FLAG[@]}" "${BIND_ARGS[@]}" "${SIF_IMAGE}")
fi

# Multi-node Ray cluster (head + workers) for Ray-based training.
RAY_CLUSTER_STARTED=0
RAY_HEAD_PID=""
RAY_WORKER_PIDS=()
RAY_NODES_ARRAY=()
RAY_HEAD_NODE=""
RAY_CLUSTER_ENABLE="${CHESS_RL_RAY_CLUSTER_ENABLE:-auto}"
SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
if [ "${RAY_CLUSTER_ENABLE}" = "auto" ]; then
  if [ "${SLURM_JOB_NUM_NODES}" -gt 1 ]; then
    RAY_CLUSTER_ENABLE="True"
  else
    RAY_CLUSTER_ENABLE="False"
  fi
fi

case "${RAY_CLUSTER_ENABLE}" in
  1|true|True|TRUE|yes|YES|y|Y)
    if [ "${SLURM_JOB_NUM_NODES}" -le 1 ]; then
      warn "CHESS_RL_RAY_CLUSTER_ENABLE=True but SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}; skipping Ray multi-node startup."
      RAY_CLUSTER_ENABLE="False"
    elif ! command -v scontrol >/dev/null 2>&1 || ! command -v srun >/dev/null 2>&1; then
      die "Ray multi-node startup requires both scontrol and srun."
    else
      nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
      nodes_array=($nodes)
      RAY_NODES_ARRAY=("${nodes_array[@]}")
      RAY_HEAD_NODE="${nodes_array[0]}"
      HOSTNAME_BIN="$(command -v hostname || true)"
      if [ -z "${HOSTNAME_BIN}" ]; then
        for _candidate in /bin/hostname /usr/bin/hostname; do
          if [ -x "${_candidate}" ]; then
            HOSTNAME_BIN="${_candidate}"
            break
          fi
        done
      fi
      if [ -z "${HOSTNAME_BIN}" ]; then
        die "Could not locate a usable hostname binary (tried PATH, /bin/hostname, /usr/bin/hostname)."
      fi
      head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$RAY_HEAD_NODE" "${HOSTNAME_BIN}" --ip-address)
      if [[ "$head_node_ip" == *" "* ]]; then
        IFS=' ' read -ra ADDR <<<"$head_node_ip"
        if [[ ${#ADDR[0]} -gt 16 ]]; then
          head_node_ip=${ADDR[1]}
        else
          head_node_ip=${ADDR[0]}
        fi
        echo "[SLURM] IPV6 address detected. Using IPV4 address as ${head_node_ip}"
      fi

      RAY_PORT="${CHESS_RL_RAY_PORT:-6379}"
      export RAY_ADDRESS="${head_node_ip}:${RAY_PORT}"
      RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-}}"
      if [ -z "${RAY_NUM_CPUS}" ] && command -v nproc >/dev/null 2>&1; then
        RAY_NUM_CPUS="$(nproc)"
      fi
      export RAY_NUM_CPUS="${RAY_NUM_CPUS:-}"
      # Keep Ray temp paths short to avoid UNIX socket length limits (<=107 chars).
      # Prefer /tmp (node-local, short). Fall back only if /tmp is not writable.
      if [ -z "${RAY_TMPDIR:-}" ]; then
        RAY_TMPDIR_DEFAULT="/tmp/ray_${SLURM_JOB_ID:-$$}"
        if ! mkdir -p "${RAY_TMPDIR_DEFAULT}" 2>/dev/null; then
          if [ "${LOCAL_SCRATCH_OK}" -eq 1 ]; then
            RAY_TMPDIR_DEFAULT="${LOCAL_SCRATCH_BASE%/}/${USER}/ray_${SLURM_JOB_ID:-$$}"
          elif [ -n "${TMPDIR:-}" ]; then
            RAY_TMPDIR_DEFAULT="${TMPDIR%/}/ray_${SLURM_JOB_ID:-$$}"
          elif [ -n "${SLURM_TMPDIR:-}" ]; then
            RAY_TMPDIR_DEFAULT="${SLURM_TMPDIR%/}/ray_${SLURM_JOB_ID:-$$}"
          fi
        fi
      else
        RAY_TMPDIR_DEFAULT="${RAY_TMPDIR}"
      fi
      export RAY_TMPDIR="${RAY_TMPDIR:-${RAY_TMPDIR_DEFAULT}}"
      RAY_TMPDIR_BASE="${RAY_TMPDIR}"
      export RAY_TMPDIR_BASE
      RAY_TMPDIR_USE_NODE_SUBDIR=1
      case "${RAY_TMPDIR_BASE}" in
        /tmp/*|/var/tmp/*|/dev/shm/*|/local/*)
          RAY_TMPDIR_USE_NODE_SUBDIR=0
          ;;
      esac
      export RAY_TMPDIR_USE_NODE_SUBDIR

      echo "[SLURM] RAY_ADDRESS=${RAY_ADDRESS}"
      echo "[SLURM] RAY_NUM_CPUS=${RAY_NUM_CPUS:-<unset>}"
      echo "[SLURM] RAY_TMPDIR=${RAY_TMPDIR}"
      echo "[SLURM] RAY_TMPDIR_BASE=${RAY_TMPDIR_BASE}"
      echo "[SLURM] RAY_TMPDIR_USE_NODE_SUBDIR=${RAY_TMPDIR_USE_NODE_SUBDIR}"

      if [ -z "${RAY_NUM_CPUS}" ]; then
        die "RAY_NUM_CPUS is empty; set SLURM_CPUS_PER_TASK or RAY_NUM_CPUS."
      fi

      if [ "${RAY_TMPDIR_USE_NODE_SUBDIR}" -eq 1 ]; then
        RAY_TMPDIR_HEAD="${RAY_TMPDIR_BASE%/}/${RAY_HEAD_NODE}"
      else
        RAY_TMPDIR_HEAD="${RAY_TMPDIR_BASE%/}"
      fi
      export RAY_TMPDIR_HEAD
      echo "[SLURM] RAY_TMPDIR_HEAD=${RAY_TMPDIR_HEAD}"
      echo "[SLURM] Starting Ray head on ${RAY_HEAD_NODE}"
      srun --nodes=1 --ntasks=1 -w "$RAY_HEAD_NODE" --export=ALL \
        "${CONTAINER_LAUNCH[@]}" \
          bash -lc "RAY_TMPDIR=\"${RAY_TMPDIR_HEAD}\"; mkdir -p \"${RAY_TMPDIR_HEAD}\" && ${CONTAINER_BASH_PREFIX}ray start --head --node-ip-address=\"${head_node_ip}\" --port=${RAY_PORT} --num-cpus \"${RAY_NUM_CPUS}\" --num-gpus \"${TRAINER_N_GPUS_PER_NODE}\" --block" &
      RAY_HEAD_PID=$!

      # Allow the head to initialize before workers connect.
      sleep 10

      worker_num=$((SLURM_JOB_NUM_NODES - 1))
      for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "[SLURM] Starting Ray worker ${i} on ${node_i}"
        if [ "${RAY_TMPDIR_USE_NODE_SUBDIR}" -eq 1 ]; then
          WORKER_TMPDIR="${RAY_TMPDIR_BASE%/}/${node_i}"
        else
          WORKER_TMPDIR="${RAY_TMPDIR_BASE%/}"
        fi
        srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
          "${CONTAINER_LAUNCH[@]}" \
            bash -lc "RAY_TMPDIR=\"${WORKER_TMPDIR}\"; mkdir -p \"${WORKER_TMPDIR}\" && ${CONTAINER_BASH_PREFIX}ray start --address \"${RAY_ADDRESS}\" --num-cpus \"${RAY_NUM_CPUS}\" --num-gpus \"${TRAINER_N_GPUS_PER_NODE}\" --block" &
        RAY_WORKER_PIDS+=($!)
        sleep 5
      done

      # Wait for Ray to register all expected nodes/GPUs before starting training.
      RAY_EXPECTED_NODES="${SLURM_JOB_NUM_NODES}"
      RAY_EXPECTED_GPUS=$((TRAINER_N_GPUS_PER_NODE * RAY_EXPECTED_NODES))
      export RAY_EXPECTED_NODES RAY_EXPECTED_GPUS
      RAY_STARTUP_WAIT_SEC="${CHESS_RL_RAY_STARTUP_WAIT_SEC:-120}"
      RAY_STARTUP_POLL_SEC="${CHESS_RL_RAY_STARTUP_POLL_SEC:-5}"
      echo "[SLURM] Waiting for Ray cluster readiness (nodes=${RAY_EXPECTED_NODES}, gpus=${RAY_EXPECTED_GPUS})..."
      RAY_READY=0
      elapsed=0
      while [ "${elapsed}" -lt "${RAY_STARTUP_WAIT_SEC}" ]; do
        RAY_CHECK_LOG="$(mktemp)"
        if srun --overlap --nodes=1 --ntasks=1 -w "${RAY_HEAD_NODE}" --export=ALL \
          "${CONTAINER_LAUNCH[@]}" \
            bash -c "export RAY_TMPDIR=\"${RAY_TMPDIR_HEAD}\"; ${CONTAINER_BASH_PREFIX}python3 -c 'import os,sys,ray; addr=os.environ.get(\"RAY_ADDRESS\"); ray.init(address=addr, ignore_reinit_error=True); nodes=[n for n in ray.nodes() if n.get(\"Alive\")]; gpus=float(ray.cluster_resources().get(\"GPU\",0)); expected_nodes=int(os.environ.get(\"RAY_EXPECTED_NODES\",\"1\")); expected_gpus=float(os.environ.get(\"RAY_EXPECTED_GPUS\",\"0\")); ok=(len(nodes)>=expected_nodes and gpus>=expected_gpus); print((\"Ray ready: nodes=%s gpus=%s\" % (len(nodes), gpus)) if ok else (\"Ray not ready: nodes=%s gpus=%s expected_nodes=%s expected_gpus=%s\" % (len(nodes), gpus, expected_nodes, expected_gpus)), flush=True); sys.exit(0 if ok else 1)' || python -c 'import os,sys,ray; addr=os.environ.get(\"RAY_ADDRESS\"); ray.init(address=addr, ignore_reinit_error=True); nodes=[n for n in ray.nodes() if n.get(\"Alive\")]; gpus=float(ray.cluster_resources().get(\"GPU\",0)); expected_nodes=int(os.environ.get(\"RAY_EXPECTED_NODES\",\"1\")); expected_gpus=float(os.environ.get(\"RAY_EXPECTED_GPUS\",\"0\")); ok=(len(nodes)>=expected_nodes and gpus>=expected_gpus); print((\"Ray ready: nodes=%s gpus=%s\" % (len(nodes), gpus)) if ok else (\"Ray not ready: nodes=%s gpus=%s expected_nodes=%s expected_gpus=%s\" % (len(nodes), gpus, expected_nodes, expected_gpus)), flush=True); sys.exit(0 if ok else 1)'" >"${RAY_CHECK_LOG}" 2>&1; then
          cat "${RAY_CHECK_LOG}"
          rm -f "${RAY_CHECK_LOG}"
          RAY_READY=1
          break
        fi
        if grep -q "No module named 'ray'" "${RAY_CHECK_LOG}"; then
          warn "Ray readiness probe couldn't import ray (likely host Python). Skipping readiness wait."
          rm -f "${RAY_CHECK_LOG}"
          break
        fi
        cat "${RAY_CHECK_LOG}"
        rm -f "${RAY_CHECK_LOG}"
        sleep "${RAY_STARTUP_POLL_SEC}"
        elapsed=$((elapsed + RAY_STARTUP_POLL_SEC))
      done
      if [ "${RAY_READY}" -ne 1 ]; then
        warn "Ray cluster did not reach expected resources within ${RAY_STARTUP_WAIT_SEC}s; training will proceed anyway."
      fi

      RAY_CLUSTER_STARTED=1
    fi
    ;;
  *)
    echo "[SLURM] Ray multi-node startup disabled."
    ;;
esac

if command -v srun >/dev/null 2>&1; then
  if [ "${RAY_CLUSTER_STARTED}" -eq 1 ] && [ -n "${RAY_HEAD_NODE}" ]; then
    srun --overlap --nodes=1 --ntasks=1 -w "${RAY_HEAD_NODE}" --export=ALL \
      "${CONTAINER_LAUNCH[@]}" \
        bash -c "export RAY_TMPDIR=\"${RAY_TMPDIR_HEAD:-${RAY_TMPDIR}}\"; ${CONTAINER_BASH_PREFIX}cd \"${CONTAINER_REPO_DIR}\" && ./train_chess.sh"
  else
    srun --export=ALL \
      "${CONTAINER_LAUNCH[@]}" \
        bash -c "${CONTAINER_BASH_PREFIX}cd \"${CONTAINER_REPO_DIR}\" && ./train_chess.sh"
  fi
else
  "${CONTAINER_LAUNCH[@]}" \
    bash -c "${CONTAINER_BASH_PREFIX}cd \"${CONTAINER_REPO_DIR}\" && ./train_chess.sh"
fi

if [ "${RAY_CLUSTER_STARTED}" -eq 1 ]; then
  echo "[SLURM] Stopping Ray cluster..."
  for node in "${RAY_NODES_ARRAY[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "${node}" --export=ALL \
      "${CONTAINER_LAUNCH[@]}" \
        bash -lc "${CONTAINER_BASH_PREFIX}ray stop" || true
  done
  if [ -n "${RAY_HEAD_PID}" ] || [ "${#RAY_WORKER_PIDS[@]}" -gt 0 ]; then
    wait ${RAY_HEAD_PID} "${RAY_WORKER_PIDS[@]}" || true
  fi
fi

echo "[SLURM] Training finished."
