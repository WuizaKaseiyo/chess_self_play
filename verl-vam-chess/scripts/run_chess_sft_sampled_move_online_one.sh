#!/usr/bin/env bash
set -euo pipefail
set -x

# Run one *online* sampled-move SFT variant for a forced-prefix letter (A/B/C/E).
#
# This is a standalone training loop (not GRPO):
#   - batches 128 prompts per step
#   - samples 8 `{move}` candidates per prompt from the RL distribution
#   - generates with HF `generate()` using forced-prefix injection
#   - runs exactly one SFT optimizer step per batch
#   - saves a final HF model checkpoint for prefix-free vLLM pass@k evaluation
#
# Usage:
#   bash scripts/run_chess_sft_sampled_move_online_one.sh <LETTER {A|B|C|E}> <WEIGHTING {awr|uniform|best_only}>
#
# Example smoke (2 steps):
#   MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct OUT_ROOT=outputs/tmp_smoke_online \
#   ONLINE_TOTAL_STEPS=2 ONLINE_EPOCHS=1 EVAL_K=2 EVAL_LIMIT=20 \
#   bash scripts/run_chess_sft_sampled_move_online_one.sh B awr

if [ $# -lt 2 ]; then
  echo "Usage: $0 <LETTER {A|B|C|E}> <WEIGHTING {awr|uniform|best_only}>" >&2
  exit 2
fi

LETTER="$1"
WEIGHTING="$2"

case "${LETTER}" in
  A|B|C|E) ;;
  *) echo "[ERROR] LETTER must be one of {A,B,C,E}, got '${LETTER}'" >&2; exit 2 ;;
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

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
TRAIN_IN_PARQUET="${TRAIN_IN_PARQUET:-data/chess_puzzles/train_hard.parquet}"
TEST_IN_PARQUET="${TEST_IN_PARQUET:-data/chess_puzzles/test.parquet}"

OUT_ROOT="${OUT_ROOT:-outputs/sft_sampled_move}"
RUN_NAME="${RUN_NAME:-${LETTER}_online_${weight_suffix}}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${RUN_NAME}}"
mkdir -p "${OUT_DIR}"

# DDP size for the online loop (4 GPUs for the cluster runs).
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Online loop sizing.
ONLINE_PROMPTS_PER_STEP="${ONLINE_PROMPTS_PER_STEP:-128}"
ONLINE_MOVES_PER_PROMPT="${ONLINE_MOVES_PER_PROMPT:-8}"
ONLINE_EPOCHS="${ONLINE_EPOCHS:-2}"
ONLINE_TOTAL_STEPS="${ONLINE_TOTAL_STEPS:-}"  # optional override

# Forced-move sampling (match RL).
MOVE_TEMPERATURE="${MOVE_TEMPERATURE:-2.0}"
MOVE_SEED="${MOVE_SEED:-0}"
SFT_AWR_BETA="${SFT_AWR_BETA:-2.0}"

# Generation (for online data).
ONLINE_GEN_MAX_NEW_TOKENS="${ONLINE_GEN_MAX_NEW_TOKENS:-256}"
ONLINE_GEN_DO_SAMPLE="${ONLINE_GEN_DO_SAMPLE:-0}"
ONLINE_GEN_TEMPERATURE="${ONLINE_GEN_TEMPERATURE:-0.6}"
ONLINE_GEN_TOP_P="${ONLINE_GEN_TOP_P:-0.95}"
ONLINE_GEN_BATCH_SIZE="${ONLINE_GEN_BATCH_SIZE:-32}"

# Train.
ONLINE_TRAIN_MAX_LENGTH="${ONLINE_TRAIN_MAX_LENGTH:-1024}"
ONLINE_MICRO_BATCH_SIZE="${ONLINE_MICRO_BATCH_SIZE:-8}"
ONLINE_LR="${ONLINE_LR:-1e-6}"

GEN_DO_SAMPLE_FLAG=()
if [[ "${ONLINE_GEN_DO_SAMPLE}" == "1" || "${ONLINE_GEN_DO_SAMPLE}" == "true" || "${ONLINE_GEN_DO_SAMPLE}" == "True" ]]; then
  GEN_DO_SAMPLE_FLAG+=(--gen_do_sample)
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  -m scripts.run_chess_sft_online_sampled_move \
  --variant "${LETTER}" \
  --model_path "${MODEL_PATH}" \
  --train_parquet "${TRAIN_IN_PARQUET}" \
  --out_dir "${OUT_DIR}" \
  --seed "${MOVE_SEED}" \
  --prompts_per_step "${ONLINE_PROMPTS_PER_STEP}" \
  --moves_per_prompt "${ONLINE_MOVES_PER_PROMPT}" \
  --epochs "${ONLINE_EPOCHS}" \
  ${ONLINE_TOTAL_STEPS:+--total_steps "${ONLINE_TOTAL_STEPS}"} \
  --move_temperature "${MOVE_TEMPERATURE}" \
  --sft_weighting "${WEIGHTING}" \
  --awr_beta "${SFT_AWR_BETA}" \
  --gen_max_new_tokens "${ONLINE_GEN_MAX_NEW_TOKENS}" \
  "${GEN_DO_SAMPLE_FLAG[@]}" \
  --gen_temperature "${ONLINE_GEN_TEMPERATURE}" \
  --gen_top_p "${ONLINE_GEN_TOP_P}" \
  --gen_batch_size "${ONLINE_GEN_BATCH_SIZE}" \
  --train_max_length "${ONLINE_TRAIN_MAX_LENGTH}" \
  --micro_batch_size "${ONLINE_MICRO_BATCH_SIZE}" \
  --lr "${ONLINE_LR}"

FINAL_STEP="$(cat "${OUT_DIR}/checkpoints/latest_checkpointed_iteration.txt")"
MODEL_DIR="${OUT_DIR}/checkpoints/global_step_${FINAL_STEP}/huggingface"

# Prefix-free pass@k evaluation (vLLM).
EVAL_K="${EVAL_K:-32}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_SEED_MODE="${EVAL_SEED_MODE:-engine}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-1024}"
EVAL_MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-2000}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-4096}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.6}"
EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-${NPROC_PER_NODE}}"
EVAL_LIMIT="${EVAL_LIMIT:-}"

PASSK_JSON="${OUT_DIR}/passk_final_k${EVAL_K}.json"
python3 -m scripts.eval_chess_passk \
  --model "${MODEL_DIR}" \
  --tokenizer "${MODEL_PATH}" \
  --parquet "${TEST_IN_PARQUET}" \
  --k_max "${EVAL_K}" --do_sample \
  --seed "${EVAL_SEED}" --seed_mode "${EVAL_SEED_MODE}" \
  --batch_size "${EVAL_BATCH_SIZE}" --max_num_seqs "${EVAL_MAX_NUM_SEQS}" \
  --max_response_length "${EVAL_MAX_RESPONSE_LENGTH}" --max_model_len "${EVAL_MAX_MODEL_LEN}" \
  --gpu_memory_utilization "${EVAL_GPU_MEMORY_UTILIZATION}" \
  --tensor_parallel_size "${EVAL_TENSOR_PARALLEL_SIZE}" \
  ${EVAL_LIMIT:+--limit "${EVAL_LIMIT}"} \
  --out_json "${PASSK_JSON}"

python3 - <<PY
import json
from pathlib import Path

out_dir = Path("${OUT_DIR}")
passk = json.loads((out_dir / "passk_final_k${EVAL_K}.json").read_text())

result = {
  "variant": "${LETTER}",
  "run_name": "${RUN_NAME}",
  "mode": "online",
  "model_base": "${MODEL_PATH}",
  "train_in_parquet": "${TRAIN_IN_PARQUET}",
  "test_in_parquet": "${TEST_IN_PARQUET}",
  "online": {
    "prompts_per_step": int("${ONLINE_PROMPTS_PER_STEP}"),
    "moves_per_prompt": int("${ONLINE_MOVES_PER_PROMPT}"),
    "epochs": int("${ONLINE_EPOCHS}"),
    "total_steps": int("${FINAL_STEP}"),
    "move_temperature": float("${MOVE_TEMPERATURE}"),
    "seed": int("${MOVE_SEED}"),
    "sft_weighting": "${WEIGHTING}",
    "awr_beta": float("${SFT_AWR_BETA}"),
    "gen": {
      "max_new_tokens": int("${ONLINE_GEN_MAX_NEW_TOKENS}"),
      "do_sample": bool(int("${ONLINE_GEN_DO_SAMPLE}" or "0")),
      "temperature": float("${ONLINE_GEN_TEMPERATURE}"),
      "top_p": float("${ONLINE_GEN_TOP_P}"),
      "batch_size": int("${ONLINE_GEN_BATCH_SIZE}"),
    },
    "train": {
      "max_length": int("${ONLINE_TRAIN_MAX_LENGTH}"),
      "micro_batch_size": int("${ONLINE_MICRO_BATCH_SIZE}"),
      "lr": float("${ONLINE_LR}"),
      "nproc_per_node": int("${NPROC_PER_NODE}"),
    },
  },
  "eval": passk.get("config", {}),
  "metrics": passk.get("summary", {}),
  "checkpoint_dir": str((out_dir / "checkpoints").resolve()),
}

(out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"Wrote {out_dir / 'results.json'}")
PY
