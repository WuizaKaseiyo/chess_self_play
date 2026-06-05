#!/usr/bin/env bash
set -euo pipefail

# Fan-out submit for setting A sampled-move variants.
#
# This submits the remaining 8 variants for letter A (excluding the already-finished
# A_sample_shuffle_wAWR) as *separate* Slurm jobs, in parallel, and blocks until all
# jobs finish using `sbatch --wait` + a final `wait`.
#
# Output root:
#   /projects/a5l/ziyan/chess_rl_outputs/${RUN_ID}/
#
# Usage (full, default):
#   bash submit_sft_sampled_move_A_parallel_gh200.bash
#
# Online estimate sizing:
#   ONLINE_TOTAL_STEPS=10 bash submit_sft_sampled_move_A_parallel_gh200.bash
#
# Optionally copy the finished A_sample_shuffle_wAWR results.json into the new root:
#   CHESS_SFT_DONE_SRC=/projects/.../sft_sampled_move_full_.../A_sample_shuffle_wAWR \
#   bash submit_sft_sampled_move_A_parallel_gh200.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SBATCH_SCRIPT="${CHESS_SFT_SBATCH_SCRIPT:-${SCRIPT_DIR}/sbatch_sft_sampled_move_one_variant_gh200.slurm}"
if [ ! -f "${SBATCH_SCRIPT}" ]; then
  echo "[ERROR] Missing sbatch script: ${SBATCH_SCRIPT}" >&2
  exit 1
fi

MODE="${CHESS_SFT_MODE:-full}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "[ERROR] CHESS_SFT_MODE must be smoke|full, got '${MODE}'" >&2
  exit 1
fi

RUN_ID="${CHESS_SFT_RUN_ID:-sft_sampled_move_A_parallel_${MODE}_$(date +%Y%m%d_%H%M%S)}"
echo "[SUBMIT] RUN_ID=${RUN_ID}"
echo "[SUBMIT] MODE=${MODE}"

PROJECT_ROOT="${CHESS_RL_PROJECT_ROOT:-/projects/a5l/ziyan}"
OUT_ROOT="${CHESS_SFT_OUT_ROOT:-${PROJECT_ROOT%/}/chess_rl_outputs/${RUN_ID}}"
mkdir -p "${OUT_ROOT}"
echo "[SUBMIT] OUT_ROOT=${OUT_ROOT}"

DONE_SRC="${CHESS_SFT_DONE_SRC:-}"
if [ -n "${DONE_SRC}" ]; then
  if [ ! -f "${DONE_SRC%/}/results.json" ]; then
    echo "[ERROR] CHESS_SFT_DONE_SRC does not contain results.json: ${DONE_SRC}" >&2
    exit 1
  fi
  mkdir -p "${OUT_ROOT}/A_sample_shuffle_wAWR"
  cp -f "${DONE_SRC%/}/results.json" "${OUT_ROOT}/A_sample_shuffle_wAWR/results.json"
  # Copy any small eval JSONs if present (optional; aggregation only needs results.json).
  for p in "${DONE_SRC%/}"/passk_*.json; do
    if [ -f "${p}" ]; then
      cp -f "${p}" "${OUT_ROOT}/A_sample_shuffle_wAWR/$(basename "${p}")"
    fi
  done
  echo "[SUBMIT] Copied finished A_sample_shuffle_wAWR into ${OUT_ROOT}/A_sample_shuffle_wAWR"
fi

CHESS_SFT_SBATCH_WAIT="${CHESS_SFT_SBATCH_WAIT:-1}"
SBATCH_WAIT_ARGS=()
if [[ "${CHESS_SFT_SBATCH_WAIT}" == "1" || "${CHESS_SFT_SBATCH_WAIT}" == "true" || "${CHESS_SFT_SBATCH_WAIT}" == "True" ]]; then
  SBATCH_WAIT_ARGS+=(--wait)
  echo "[SUBMIT] Blocking until completion (sbatch --wait + wait)"
else
  echo "[SUBMIT] Non-blocking submit (override CHESS_SFT_SBATCH_WAIT=1 to block)" >&2
fi

# Online estimate override (defaults to 10, per request).
ONLINE_TOTAL_STEPS="${ONLINE_TOTAL_STEPS:-}"
ONLINE_EPOCHS="${ONLINE_EPOCHS:-1}"

# Offline: 5 remaining (shuffle uniform/best_only, plus no_shuffle {awr,uniform,best_only}).
offline_jobs=(
  "offline shuffle uniform"
  "offline shuffle best_only"
  "offline no_shuffle awr"
  "offline no_shuffle uniform"
  "offline no_shuffle best_only"
)

# Online: 3 weightings (10-step estimate run unless overridden).
online_jobs=(
  "online awr"
  "online uniform"
  "online best_only"
)

submit_one() {
  local kind="$1"
  local ordering="$2"
  local weighting="$3"
  local job_name="$4"
  local extra_export="$5"

  sbatch \
    "${SBATCH_WAIT_ARGS[@]}" \
    --job-name="${job_name}" \
    --export="ALL,CHESS_SFT_RUN_ID=${RUN_ID},CHESS_SFT_MODE=${MODE},CHESS_SFT_LETTER=A,CHESS_SFT_KIND=${kind},CHESS_SFT_ORDERING=${ordering},CHESS_SFT_WEIGHTING=${weighting}${extra_export}" \
    "${SBATCH_SCRIPT}"
}

for spec in "${offline_jobs[@]}"; do
  read -r kind ordering weighting <<<"${spec}"
  (
    submit_one "${kind}" "${ordering}" "${weighting}" "chess-sft-A-${kind}-${ordering}-${weighting}" ""
  ) &
done

for spec in "${online_jobs[@]}"; do
  read -r kind weighting <<<"${spec}"
  (
    # Online runs:
    # - If ONLINE_TOTAL_STEPS is set, we treat this as an estimate run and also cap eval sizing.
    # - Otherwise, run the default full config (derived total_steps, full eval).
    extra=""
    if [ -n "${ONLINE_TOTAL_STEPS}" ]; then
      extra=",ONLINE_TOTAL_STEPS=${ONLINE_TOTAL_STEPS},ONLINE_EPOCHS=${ONLINE_EPOCHS},EVAL_K=2,EVAL_LIMIT=20"
    fi
    submit_one "${kind}" "" "${weighting}" "chess-sft-A-${kind}-${weighting}" "${extra}"
  ) &
done

wait
echo "[SUBMIT] Done: RUN_ID=${RUN_ID} MODE=${MODE} OUT_ROOT=${OUT_ROOT}"
