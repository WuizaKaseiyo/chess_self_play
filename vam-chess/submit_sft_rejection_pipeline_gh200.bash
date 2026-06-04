#!/usr/bin/env bash
set -euo pipefail

# Submit end-to-end rejection-sampling SFT pipelines on Isambard GH200.
#
# For each model run:
#   1) Build rejection-sampled SFT dataset (2 nodes)
#   2) Train SFT + per-epoch puzzle eval (1 node)
#   3) Final full-game eval (1 node; controlled by RUN_FULLGAME_EVAL)
#
# This script is intended to run on a Slurm login node.
#
# Usage (smoke, 1/10 prompts, Qwen2.5-3B, 1 epoch):
#   CHESS_SFT_MODE=smoke \
#   CHESS_SFT_RUN_ID=sft_reject_smoke_$(date +%Y%m%d_%H%M%S) \
#   bash submit_sft_rejection_pipeline_gh200.bash
#
# Usage (full, 3-model matrix, 2 epochs each):
#   CHESS_SFT_MODE=full \
#   CHESS_SFT_RUN_ID=sft_reject_full_$(date +%Y%m%d_%H%M%S) \
#   bash submit_sft_rejection_pipeline_gh200.bash

if ! command -v sbatch >/dev/null 2>&1; then
  echo "[ERROR] sbatch not found on PATH. Run this on a Slurm login node." >&2
  exit 1
fi

MODE="${CHESS_SFT_MODE:-smoke}"
case "${MODE}" in
  smoke|full) ;;
  *) echo "[ERROR] CHESS_SFT_MODE must be one of {smoke,full}, got '${MODE}'" >&2; exit 2 ;;
esac

RUN_ID_BASE="${CHESS_SFT_RUN_ID:-}"
if [ -z "${RUN_ID_BASE}" ]; then
  RUN_ID_BASE="sft_reject_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi

EPOCHS="${EPOCHS:-}"
if [ -z "${EPOCHS}" ]; then
  if [ "${MODE}" = "smoke" ]; then
    EPOCHS=1
  else
    EPOCHS=2
  fi
fi

RUN_SHUFFLED_EVAL="${RUN_SHUFFLED_EVAL:-0}"
RUN_FULLGAME_EVAL="${RUN_FULLGAME_EVAL:-1}"

# Throughput defaults for this rejection-SFT workflow.
BUILD_BATCH_SIZE="${BUILD_BATCH_SIZE:-$((128 * 8))}"
BUILD_MAX_NUM_SEQS="${BUILD_MAX_NUM_SEQS:-1024}"
BUILD_MAX_NUM_BATCHED_TOKENS="${BUILD_MAX_NUM_BATCHED_TOKENS:-65536}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-1024}"
EVAL_MAX_NUM_BATCHED_TOKENS="${EVAL_MAX_NUM_BATCHED_TOKENS:-65536}"
FULLGAME_MAX_NUM_SEQS="${FULLGAME_MAX_NUM_SEQS:-1024}"
FULLGAME_MAX_NUM_BATCHED_TOKENS="${FULLGAME_MAX_NUM_BATCHED_TOKENS:-65536}"
FULLGAME_MAX_RETRIES_PER_TURN="${FULLGAME_MAX_RETRIES_PER_TURN:-1}"

MODEL_PATH="${MODEL_PATH:-}"
CHESS_SFT_MODELS="${CHESS_SFT_MODELS:-}"

models=()
if [ -n "${CHESS_SFT_MODELS}" ]; then
  # Space-separated model list.
  read -r -a models <<< "${CHESS_SFT_MODELS}"
elif [ -n "${MODEL_PATH}" ]; then
  models=("${MODEL_PATH}")
elif [ "${MODE}" = "smoke" ]; then
  models=("Qwen/Qwen2.5-3B-Instruct")
else
  models=(
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
  )
fi

if [ "${#models[@]}" -eq 0 ]; then
  echo "[ERROR] No models resolved for execution." >&2
  exit 3
fi

sanitize_model_tag() {
  local m="$1"
  local base
  base="$(basename "${m}")"
  echo "${base}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g'
}

mkdir -p slurm

echo "[RUN] CHESS_SFT_MODE=${MODE}"
echo "[RUN] CHESS_SFT_RUN_ID(base)=${RUN_ID_BASE}"
echo "[RUN] EPOCHS=${EPOCHS}"
echo "[RUN] RUN_SHUFFLED_EVAL=${RUN_SHUFFLED_EVAL}"
echo "[RUN] RUN_FULLGAME_EVAL=${RUN_FULLGAME_EVAL}"
echo "[RUN] BUILD_BATCH_SIZE=${BUILD_BATCH_SIZE} BUILD_MAX_NUM_SEQS=${BUILD_MAX_NUM_SEQS} BUILD_MAX_NUM_BATCHED_TOKENS=${BUILD_MAX_NUM_BATCHED_TOKENS}"
echo "[RUN] EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE} EVAL_MAX_NUM_SEQS=${EVAL_MAX_NUM_SEQS} EVAL_MAX_NUM_BATCHED_TOKENS=${EVAL_MAX_NUM_BATCHED_TOKENS}"
echo "[RUN] FULLGAME_MAX_NUM_SEQS=${FULLGAME_MAX_NUM_SEQS} FULLGAME_MAX_NUM_BATCHED_TOKENS=${FULLGAME_MAX_NUM_BATCHED_TOKENS} FULLGAME_MAX_RETRIES_PER_TURN=${FULLGAME_MAX_RETRIES_PER_TURN}"
printf '[RUN] MODELS (%d):\n' "${#models[@]}"
for m in "${models[@]}"; do
  echo "  - ${m}"
done

run_roots=()
for idx in "${!models[@]}"; do
  model="${models[$idx]}"
  model_tag="$(sanitize_model_tag "${model}")"

  if [ "${#models[@]}" -gt 1 ]; then
    run_id="${RUN_ID_BASE}_${model_tag}"
  else
    run_id="${RUN_ID_BASE}"
  fi

  run_name="rejection_sft_${model_tag}"

  echo "[MODEL $((idx + 1))/${#models[@]}] ${model}"
  echo "[MODEL] run_id=${run_id} run_name=${run_name}"

  echo "[STEP 1/2] Build rejection-sampled dataset..."
  build_nodes_args=()
  if [ "${MODE}" = "smoke" ]; then
    # Smoke policy: use exactly one node.
    build_nodes_args=(--nodes=1 --ntasks=1 --ntasks-per-node=1)
  fi
  sbatch --wait \
    "${build_nodes_args[@]}" \
    --export=ALL,CHESS_SFT_RUN_ID="${run_id}",CHESS_SFT_MODE="${MODE}",MODEL_PATH="${model}",REBUILD_DATASET=1,BATCH_SIZE="${BUILD_BATCH_SIZE}",MAX_NUM_SEQS="${BUILD_MAX_NUM_SEQS}",MAX_NUM_BATCHED_TOKENS="${BUILD_MAX_NUM_BATCHED_TOKENS}" \
    sbatch_build_chess_sft_rejection_dataset_gh200.slurm

  echo "[STEP 2/2] Train SFT + per-epoch puzzle eval (+ optional full-game eval)..."
  sbatch --wait \
    --export=ALL,CHESS_SFT_RUN_ID="${run_id}",CHESS_SFT_MODE="${MODE}",MODEL_PATH="${model}",EPOCHS="${EPOCHS}",RUN_NAME="${run_name}",RUN_SHUFFLED_EVAL="${RUN_SHUFFLED_EVAL}",RUN_FULLGAME_EVAL="${RUN_FULLGAME_EVAL}",EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}",EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS}",EVAL_MAX_NUM_BATCHED_TOKENS="${EVAL_MAX_NUM_BATCHED_TOKENS}",FULLGAME_MAX_NUM_SEQS="${FULLGAME_MAX_NUM_SEQS}",FULLGAME_MAX_NUM_BATCHED_TOKENS="${FULLGAME_MAX_NUM_BATCHED_TOKENS}",FULLGAME_MAX_RETRIES_PER_TURN="${FULLGAME_MAX_RETRIES_PER_TURN}" \
    sbatch_sft_rejection_gh200.slurm

  run_root="/projects/a5l/ziyan/chess_rl_outputs/${run_id}/"
  run_roots+=("${run_root}")
  echo "[MODEL DONE] ${model} -> ${run_root}"
done

echo "[DONE] Completed runs:"
for r in "${run_roots[@]}"; do
  echo "  ${r}"
done
