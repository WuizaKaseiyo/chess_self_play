#!/usr/bin/env bash
set -euo pipefail

# Submit grouped sampled-move SFT experiments (one job per letter A/B/C/E) and block until completion.
#
# Modes:
#   - smoke: 2-step quick runs (tiny model)
#   - full:  full runs (default model)
#
# Examples:
#   # Smoke (blocks):
#   CHESS_SFT_MODE=smoke CHESS_SFT_SBATCH_WAIT=1 bash submit_sft_sampled_move_grouped_gh200.bash
#
#   # Full (blocks):
#   CHESS_SFT_MODE=full CHESS_SFT_SBATCH_WAIT=1 bash submit_sft_sampled_move_grouped_gh200.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SBATCH_SCRIPT="${CHESS_SFT_SBATCH_SCRIPT:-${SCRIPT_DIR}/sbatch_sft_sampled_move_grouped_gh200.slurm}"
if [ ! -f "${SBATCH_SCRIPT}" ]; then
  echo "[ERROR] Missing sbatch script: ${SBATCH_SCRIPT}" >&2
  exit 1
fi

MODE="${CHESS_SFT_MODE:-smoke}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "[ERROR] CHESS_SFT_MODE must be smoke|full, got '${MODE}'" >&2
  exit 1
fi

RUN_ID="${CHESS_SFT_RUN_ID:-sft_sampled_move_${MODE}_$(date +%Y%m%d_%H%M%S)}"
echo "[SUBMIT] RUN_ID=${RUN_ID}"
echo "[SUBMIT] MODE=${MODE}"

CHESS_SFT_SBATCH_WAIT="${CHESS_SFT_SBATCH_WAIT:-1}"
SBATCH_WAIT_ARGS=()
if [[ "${CHESS_SFT_SBATCH_WAIT}" == "1" || "${CHESS_SFT_SBATCH_WAIT}" == "true" || "${CHESS_SFT_SBATCH_WAIT}" == "True" ]]; then
  SBATCH_WAIT_ARGS+=(--wait)
  echo "[SUBMIT] Blocking until completion (sbatch --wait)"
else
  echo "[SUBMIT] Non-blocking submit (override CHESS_SFT_SBATCH_WAIT=1 to block)"
fi

letters=(A B C E)
for letter in "${letters[@]}"; do
  (
    sbatch \
      "${SBATCH_WAIT_ARGS[@]}" \
      --job-name="chess-sft-sampled-${MODE}-${letter}" \
      --export="ALL,CHESS_SFT_RUN_ID=${RUN_ID},CHESS_SFT_MODE=${MODE},CHESS_SFT_LETTER=${letter}" \
      "${SBATCH_SCRIPT}"
  ) &
done

wait
echo "[SUBMIT] Done: RUN_ID=${RUN_ID} MODE=${MODE}"

