#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SBATCH_SCRIPT="${CHESS_RL_SBATCH_SCRIPT:-${SCRIPT_DIR}/sbatch_train_chess_gh200.slurm}"
if [ ! -f "${SBATCH_SCRIPT}" ]; then
  echo "[ERROR] Missing sbatch script: ${SBATCH_SCRIPT}" >&2
  exit 1
fi

CHESS_RL_SUBMIT_COUNT="${CHESS_RL_SUBMIT_COUNT:-1}"
if ! [[ "${CHESS_RL_SUBMIT_COUNT}" =~ ^[0-9]+$ ]] || [ "${CHESS_RL_SUBMIT_COUNT}" -le 0 ]; then
  echo "[ERROR] Invalid CHESS_RL_SUBMIT_COUNT='${CHESS_RL_SUBMIT_COUNT}' (expected positive int)." >&2
  exit 1
fi

# Max number of active Slurm jobs this script will allow before throttling.
# - Set to 0 to disable throttling (unlimited / "infinite").
CHESS_RL_MAX_ACTIVE_JOBS="${CHESS_RL_MAX_ACTIVE_JOBS:-0}"
if ! [[ "${CHESS_RL_MAX_ACTIVE_JOBS}" =~ ^[0-9]+$ ]] || [ "${CHESS_RL_MAX_ACTIVE_JOBS}" -lt 0 ]; then
  echo "[ERROR] Invalid CHESS_RL_MAX_ACTIVE_JOBS='${CHESS_RL_MAX_ACTIVE_JOBS}' (expected non-negative int; 0=unlimited)." >&2
  exit 1
fi
if [ "${CHESS_RL_MAX_ACTIVE_JOBS}" -eq 0 ]; then
  echo "[CONFIG] CHESS_RL_MAX_ACTIVE_JOBS=0 (unlimited; no throttling)"
else
  echo "[CONFIG] CHESS_RL_MAX_ACTIVE_JOBS=${CHESS_RL_MAX_ACTIVE_JOBS}"
fi

# Only throttle jobs whose names start with this prefix. Set to empty string to
# throttle *all* your jobs (not recommended unless you really want it).
CHESS_RL_JOB_NAME_PREFIX="${CHESS_RL_JOB_NAME_PREFIX:-chess-rl-gh200}"
echo "[CONFIG] CHESS_RL_JOB_NAME_PREFIX='${CHESS_RL_JOB_NAME_PREFIX}'"

# How often to poll Slurm while waiting for available "active job" slots.
CHESS_RL_SUBMIT_POLL_SECONDS="${CHESS_RL_SUBMIT_POLL_SECONDS:-300}"
if ! [[ "${CHESS_RL_SUBMIT_POLL_SECONDS}" =~ ^[0-9]+$ ]] || [ "${CHESS_RL_SUBMIT_POLL_SECONDS}" -le 0 ]; then
  echo "[ERROR] Invalid CHESS_RL_SUBMIT_POLL_SECONDS='${CHESS_RL_SUBMIT_POLL_SECONDS}' (expected positive int)." >&2
  exit 1
fi

# Behavior when the max active job limit is reached:
# - "wait" (default): block and poll until a slot is available, then submit.
# - "exit": stop submitting immediately (useful if you don't want a long-running shell).
CHESS_RL_SUBMIT_WAIT_MODE="${CHESS_RL_SUBMIT_WAIT_MODE:-wait}"
if [[ "${CHESS_RL_SUBMIT_WAIT_MODE}" != "wait" && "${CHESS_RL_SUBMIT_WAIT_MODE}" != "exit" ]]; then
  echo "[ERROR] Invalid CHESS_RL_SUBMIT_WAIT_MODE='${CHESS_RL_SUBMIT_WAIT_MODE}' (expected 'wait' or 'exit')." >&2
  exit 1
fi

CHESS_RL_SQUEUE_USER="${CHESS_RL_SQUEUE_USER:-${USER:-}}"
if [[ -z "${CHESS_RL_SQUEUE_USER}" ]]; then
  CHESS_RL_SQUEUE_USER="$(id -un)"
fi

count_active_jobs() {
  if ! command -v squeue >/dev/null 2>&1; then
    # Likely running locally; can't enforce Slurm queue limits.
    echo 0
    return 0
  fi

  local states="PD,R"
  if [[ -z "${CHESS_RL_JOB_NAME_PREFIX}" ]]; then
    squeue -h -u "${CHESS_RL_SQUEUE_USER}" -t "${states}" | wc -l | tr -d ' '
    return 0
  fi

  squeue -h -u "${CHESS_RL_SQUEUE_USER}" -t "${states}" -o "%j" \
    | awk -v p="${CHESS_RL_JOB_NAME_PREFIX}" '{sub(/^[ \t]+/, "", $0); if (index($0, p) == 1) c++} END {print c+0}'
}

print_active_jobs() {
  if ! command -v squeue >/dev/null 2>&1; then
    return 0
  fi

  local states="PD,R"
  if [[ -z "${CHESS_RL_JOB_NAME_PREFIX}" ]]; then
    squeue -h -u "${CHESS_RL_SQUEUE_USER}" -t "${states}" -o "%A %T %j" || true
    return 0
  fi

  squeue -h -u "${CHESS_RL_SQUEUE_USER}" -t "${states}" -o "%A %T %j" \
    | awk -v p="${CHESS_RL_JOB_NAME_PREFIX}" 'index($3, p) == 1' \
    || true
}

wait_for_slot() {
  local next_job_name="$1"

  # Unlimited mode: submit immediately (no throttling).
  if [ "${CHESS_RL_MAX_ACTIVE_JOBS}" -eq 0 ]; then
    return 0
  fi

  while true; do
    local active_count
    active_count="$(count_active_jobs)"

    if [ "${active_count}" -lt "${CHESS_RL_MAX_ACTIVE_JOBS}" ]; then
      return 0
    fi

    echo "[THROTTLE] Active jobs (${CHESS_RL_JOB_NAME_PREFIX:-ALL}): ${active_count} >= ${CHESS_RL_MAX_ACTIVE_JOBS}" >&2
    if [[ "${CHESS_RL_SUBMIT_WAIT_MODE}" == "exit" ]]; then
      echo "[THROTTLE] CHESS_RL_SUBMIT_WAIT_MODE=exit: not submitting '${next_job_name}' (rerun later)." >&2
      exit 0
    fi

    echo "[THROTTLE] Waiting ${CHESS_RL_SUBMIT_POLL_SECONDS}s to submit '${next_job_name}'..." >&2
    print_active_jobs >&2
    sleep "${CHESS_RL_SUBMIT_POLL_SECONDS}"
  done
}

submit() {
  local job_name="$1"
  local export_str="$2"
  local i
  for ((i=1; i<=CHESS_RL_SUBMIT_COUNT; i++)); do
    wait_for_slot "${job_name}"
    echo "[SUBMIT] ${job_name} (${i}/${CHESS_RL_SUBMIT_COUNT})"
    sbatch --job-name="${job_name}" --dependency=singleton --export="${export_str}" "${SBATCH_SCRIPT}"
  done
}

echo "[INFO] No jobs submitted by default."
echo "[INFO] Edit this file to add your experiment submits, or run sbatch directly."

# Example submits:
#   # Smoke run (few steps, no HF checkpoints):
#   submit \
#     "chess-rl-gh200-smoke" \
#     "ALL,CHESS_DATA_DIR=data/chess_puzzles_chessr1_aligned_sharded,USE_HARD_DATASET=False,USE_CRITIC_DATASET=False,FORCED_PREFIX_ENABLE=True,CHESS_RL_TRAIN_BATCH_SIZE=64,CHESS_RL_TOTAL_TRAINING_STEPS=10,CHESS_RL_TRAINER_SAVE_FREQ=-1"
#
#   # Disable forced-prefix exploration:
#   submit \
#     "chess-rl-gh200-nofp" \
#     "ALL,CHESS_DATA_DIR=data/chess_puzzles_chessr1_aligned_sharded,FORCED_PREFIX_ENABLE=False,CHESS_RL_TOTAL_TRAINING_STEPS=200,CHESS_RL_TRAINER_SAVE_FREQ=40"
