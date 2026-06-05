#!/usr/bin/env bash
set -euo pipefail
set -x

# Run one *offline* sampled-move SFT variant for a forced-prefix letter (A/B/C/E):
#   - expands each prompt into NUM_MOVE_SAMPLES examples by sampling `{move}` from the RL distribution
#   - trains with veRL FSDP SFTTrainer
#   - evaluates with prefix-free vLLM pass@k
#
# Usage:
#   bash scripts/run_chess_sft_sampled_move_offline_one.sh <LETTER {A|B|C|E}> <ORDERING {shuffle|no_shuffle}> <WEIGHTING {awr|uniform|best_only}>
#
# Example:
#   OUT_ROOT=outputs/tmp_smoke_sampled_move \
#   MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct \
#   TOTAL_TRAINING_STEPS=2 TRAIN_LIMIT=64 EVAL_LIMIT=20 \
#   bash scripts/run_chess_sft_sampled_move_offline_one.sh B shuffle awr

if [ $# -lt 3 ]; then
  echo "Usage: $0 <LETTER {A|B|C|E}> <ORDERING {shuffle|no_shuffle}> <WEIGHTING {awr|uniform|best_only}>" >&2
  exit 2
fi

LETTER="$1"
ORDERING="$2"
WEIGHTING="$3"

case "${LETTER}" in
  A|B|C|E) ;;
  *) echo "[ERROR] LETTER must be one of {A,B,C,E}, got '${LETTER}'" >&2; exit 2 ;;
esac
case "${ORDERING}" in
  shuffle|no_shuffle) ;;
  *) echo "[ERROR] ORDERING must be one of {shuffle,no_shuffle}, got '${ORDERING}'" >&2; exit 2 ;;
esac
case "${WEIGHTING}" in
  awr|uniform|best_only) ;;
  *) echo "[ERROR] WEIGHTING must be one of {awr,uniform,best_only}, got '${WEIGHTING}'" >&2; exit 2 ;;
esac

weight_suffix=""
case "${WEIGHTING}" in
  awr) weight_suffix="wAWR" ;;
  uniform) weight_suffix="wUniform" ;;
  best_only) weight_suffix="wBestOnly" ;;
esac

export RUN_NAME="${LETTER}_sample_${ORDERING}_${weight_suffix}"
export NUM_MOVE_SAMPLES="${NUM_MOVE_SAMPLES:-8}"
export SAMPLE_ORDERING="${ORDERING}"
export SFT_WEIGHTING="${WEIGHTING}"
export SFT_AWR_BETA="${SFT_AWR_BETA:-2.0}"

bash scripts/run_chess_sft_prefix_ablation_one.sh "${LETTER}"

