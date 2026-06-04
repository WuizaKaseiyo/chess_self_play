#!/usr/bin/env bash
set -euo pipefail
set -x

# Run SFT on a rejection-sampled SFT parquet, evaluate puzzle accuracy after
# every epoch checkpoint, and optionally run final full-game eval vs Stockfish.
#
# Intended to run inside the repo's training container on GH200 / Isambard.
#
# Required env:
#   - MODEL_PATH: base model HF id (e.g. Qwen/Qwen2.5-7B-Instruct)
#   - TRAIN_SFT_PARQUET: output of scripts/build_chess_sft_rejection_dataset.py
#
# Optional env (common):
#   - OUT_ROOT, OUT_DIR, RUN_NAME
#   - EPOCHS (default 1)
#   - TRAIN_BATCH_SIZE (global, divisible by DP size), MICRO_BATCH_SIZE_PER_GPU, MAX_LENGTH
#   - NPROC_PER_NODE (DP size; default 4)
#   - EVAL_* (see below)
#   - FULLGAME_* (see below)

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required (e.g. Qwen/Qwen2.5-7B-Instruct)}"
TRAIN_SFT_PARQUET="${TRAIN_SFT_PARQUET:?TRAIN_SFT_PARQUET is required}"

TEST_PARQUET="${TEST_PARQUET:-data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet}"
TEST_SHUFFLED_PARQUET="${TEST_SHUFFLED_PARQUET:-data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet}"

OUT_ROOT="${OUT_ROOT:-outputs/sft_rejection_sampling}"
RUN_NAME="${RUN_NAME:-rejection_sft}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT%/}/${RUN_NAME}}"
mkdir -p "${OUT_DIR}"

# SFT hyperparams.
EPOCHS="${EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"                # global
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}" # per rank
MAX_LENGTH="${MAX_LENGTH:-3584}"

CKPT_DIR="${CKPT_DIR:-${OUT_DIR}/checkpoints}"
mkdir -p "${CKPT_DIR}"

# veRL checkpoint contents: for eval we only need `hf_model`.
SAVE_CONTENTS="${SAVE_CONTENTS:-[\"hf_model\"]}"
LOAD_CONTENTS="${LOAD_CONTENTS:-[]}"
export SAVE_CONTENTS LOAD_CONTENTS

# torchrun / distributed.
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
DP_SIZE="${DP_SIZE:-${NPROC_PER_NODE}}"
if [ "$((TRAIN_BATCH_SIZE % DP_SIZE))" -ne 0 ]; then
  echo "[ERROR] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} must be divisible by DP_SIZE=${DP_SIZE} (NPROC_PER_NODE=${NPROC_PER_NODE})." >&2
  exit 1
fi
TRAIN_BATCH_SIZE_PER_RANK="$((TRAIN_BATCH_SIZE / DP_SIZE))"

TRAIN_ROWS="$(python3 - <<PY
import pyarrow.parquet as pq
pf = pq.ParquetFile("${TRAIN_SFT_PARQUET}")
print(int(pf.metadata.num_rows))
PY
)"
if [ "${TRAIN_ROWS}" -le 0 ]; then
  echo "[ERROR] TRAIN_SFT_PARQUET has 0 rows: ${TRAIN_SFT_PARQUET}" >&2
  echo "[ERROR] Check the rejection-sampling acceptance rate and generation settings." >&2
  exit 1
fi

# Match `verl/trainer/fsdp_sft_trainer.py` drop_last behavior (see `sft.md`).
SAMPLES_PER_RANK="$((TRAIN_ROWS / DP_SIZE))"
STEPS_PER_EPOCH="$((SAMPLES_PER_RANK / TRAIN_BATCH_SIZE_PER_RANK))"
if [ "${STEPS_PER_EPOCH}" -le 0 ]; then
  echo "[ERROR] Computed STEPS_PER_EPOCH=${STEPS_PER_EPOCH} (TRAIN_ROWS=${TRAIN_ROWS}, DP_SIZE=${DP_SIZE}, TRAIN_BATCH_SIZE_PER_RANK=${TRAIN_BATCH_SIZE_PER_RANK})." >&2
  exit 1
fi
TOTAL_STEPS="$((STEPS_PER_EPOCH * EPOCHS))"

SAVE_STEPS_JSON="$(python3 - <<PY
import json
steps_per_epoch = int(${STEPS_PER_EPOCH})
epochs = int(${EPOCHS})
if epochs <= 0:
    raise SystemExit(f"EPOCHS must be > 0, got {epochs}")
steps = [steps_per_epoch * (i + 1) for i in range(epochs)]
print(json.dumps(steps))
PY
)"
SAVE_STEPS_SPACE="$(python3 - <<PY
import json
steps = json.loads('''${SAVE_STEPS_JSON}''')
print(' '.join(str(int(x)) for x in steps))
PY
)"
FINAL_STEP="$(python3 - <<PY
import json
steps = json.loads('''${SAVE_STEPS_JSON}''')
print(int(steps[-1]))
PY
)"

# Train.
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" -m verl.trainer.fsdp_sft_trainer \
  data.train_files="${TRAIN_SFT_PARQUET}" \
  data.val_files="${TRAIN_SFT_PARQUET}" \
  data.multiturn.enable=True \
  data.multiturn.messages_key=messages \
  +data.multiturn.weight_key=sft_weight \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  data.max_length="${MAX_LENGTH}" \
  data.truncation=right \
  model.partial_pretrain="${MODEL_PATH}" \
  model.trust_remote_code=True \
  model.fsdp_config.model_dtype=bf16 \
  trainer.project_name=chess_sft_rejection \
  trainer.experiment_name="sft_rejection_${RUN_NAME}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.total_epochs="${EPOCHS}" \
  trainer.save_freq=-1 \
  +trainer.save_steps="${SAVE_STEPS_JSON}" \
  trainer.test_freq=-1 \
  trainer.logger='["console"]' \
  trainer.resume_mode=disable \
  trainer.checkpoint.save_contents="${SAVE_CONTENTS}" \
  trainer.checkpoint.load_contents="${LOAD_CONTENTS}"

FINAL_HF_DIR="${CKPT_DIR}/global_step_${FINAL_STEP}/huggingface"
if [ ! -d "${FINAL_HF_DIR}" ]; then
  # Best-effort fallback: use the latest checkpoint dir.
  FINAL_HF_DIR="$(ls -1d "${CKPT_DIR}"/global_step_*/huggingface 2>/dev/null | sort | tail -n 1 || true)"
fi
if [ -z "${FINAL_HF_DIR}" ] || [ ! -d "${FINAL_HF_DIR}" ]; then
  echo "[ERROR] Could not locate a huggingface checkpoint under ${CKPT_DIR}" >&2
  exit 1
fi

ln -sfn "${FINAL_HF_DIR}" "${OUT_DIR}/final_hf_model"

# -----------------------------------------------------------------------------
# Eval: pass@1 exact-match on baseline test split after every epoch checkpoint.

EVAL_K="${EVAL_K:-1}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_SEED_MODE="${EVAL_SEED_MODE:-per_prompt}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-1024}"
EVAL_MAX_NUM_BATCHED_TOKENS="${EVAL_MAX_NUM_BATCHED_TOKENS:-65536}"
EVAL_MAX_PROMPT_LENGTH="${EVAL_MAX_PROMPT_LENGTH:-1536}"
EVAL_MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-2000}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-4096}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-4}"
EVAL_K_CHUNK="${EVAL_K_CHUNK:-}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
EVAL_ACC_KEY="${EVAL_ACC_KEY:-exact_match}"
EVAL_ROLLOUTS_DIR="${EVAL_ROLLOUTS_DIR:-${OUT_DIR}/puzzle_eval_rollouts}"
RUN_SHUFFLED_EVAL="${RUN_SHUFFLED_EVAL:-0}"
mkdir -p "${EVAL_ROLLOUTS_DIR}"

EPOCH_EVAL_JSONS=()
EPOCH_EVAL_ROLLOUTS=()
EPOCH_EVAL_SHUFFLED_JSONS=()
EPOCH_EVAL_SHUFFLED_ROLLOUTS=()

for step in ${SAVE_STEPS_SPACE}; do
  step_int="${step}"
  if [ "$((step_int % STEPS_PER_EPOCH))" -ne 0 ]; then
    echo "[ERROR] save step ${step_int} is not an epoch boundary (STEPS_PER_EPOCH=${STEPS_PER_EPOCH})." >&2
    exit 1
  fi
  epoch_idx="$((step_int / STEPS_PER_EPOCH))"
  ckpt_hf="${CKPT_DIR}/global_step_${step_int}/huggingface"
  if [ ! -d "${ckpt_hf}" ]; then
    echo "[ERROR] Missing checkpoint for epoch ${epoch_idx}: ${ckpt_hf}" >&2
    exit 1
  fi

  out_json="${OUT_DIR}/acc_test_epoch${epoch_idx}_k${EVAL_K}.json"
  out_rollouts="${EVAL_ROLLOUTS_DIR}/test.epoch${epoch_idx}.k${EVAL_K}.seed${EVAL_SEED}.jsonl"

  python3 -m scripts.eval_chess_passk \
    --model "${ckpt_hf}" \
    --tokenizer "${MODEL_PATH}" \
    --parquet "${TEST_PARQUET}" \
    --acc_key "${EVAL_ACC_KEY}" \
    --k_max "${EVAL_K}" --do_sample \
    ${EVAL_K_CHUNK:+--k_chunk "${EVAL_K_CHUNK}"} \
    --seed "${EVAL_SEED}" --seed_mode "${EVAL_SEED_MODE}" \
    --temperature "${EVAL_TEMPERATURE}" --top_p "${EVAL_TOP_P}" \
    --batch_size "${EVAL_BATCH_SIZE}" --max_num_seqs "${EVAL_MAX_NUM_SEQS}" \
    --max_num_batched_tokens "${EVAL_MAX_NUM_BATCHED_TOKENS}" \
    --max_prompt_length "${EVAL_MAX_PROMPT_LENGTH}" \
    --max_response_length "${EVAL_MAX_RESPONSE_LENGTH}" --max_model_len "${EVAL_MAX_MODEL_LEN}" \
    --gpu_memory_utilization "${EVAL_GPU_MEMORY_UTILIZATION}" \
    --tensor_parallel_size "${EVAL_TENSOR_PARALLEL_SIZE}" \
    ${EVAL_LIMIT:+--limit "${EVAL_LIMIT}"} \
    --out_rollouts_jsonl "${out_rollouts}" \
    --out_json "${out_json}"

  EPOCH_EVAL_JSONS+=("${out_json}")
  EPOCH_EVAL_ROLLOUTS+=("${out_rollouts}")

  if [ "${RUN_SHUFFLED_EVAL}" = "1" ]; then
    out_shuf_json="${OUT_DIR}/acc_test_shuffled_legal_moves_epoch${epoch_idx}_k${EVAL_K}.json"
    out_shuf_rollouts="${EVAL_ROLLOUTS_DIR}/test_shuffled_legal_moves.epoch${epoch_idx}.k${EVAL_K}.seed${EVAL_SEED}.jsonl"

    python3 -m scripts.eval_chess_passk \
      --model "${ckpt_hf}" \
      --tokenizer "${MODEL_PATH}" \
      --parquet "${TEST_SHUFFLED_PARQUET}" \
      --acc_key "${EVAL_ACC_KEY}" \
      --k_max "${EVAL_K}" --do_sample \
      ${EVAL_K_CHUNK:+--k_chunk "${EVAL_K_CHUNK}"} \
      --seed "${EVAL_SEED}" --seed_mode "${EVAL_SEED_MODE}" \
      --temperature "${EVAL_TEMPERATURE}" --top_p "${EVAL_TOP_P}" \
      --batch_size "${EVAL_BATCH_SIZE}" --max_num_seqs "${EVAL_MAX_NUM_SEQS}" \
      --max_num_batched_tokens "${EVAL_MAX_NUM_BATCHED_TOKENS}" \
      --max_prompt_length "${EVAL_MAX_PROMPT_LENGTH}" \
      --max_response_length "${EVAL_MAX_RESPONSE_LENGTH}" --max_model_len "${EVAL_MAX_MODEL_LEN}" \
      --gpu_memory_utilization "${EVAL_GPU_MEMORY_UTILIZATION}" \
      --tensor_parallel_size "${EVAL_TENSOR_PARALLEL_SIZE}" \
      ${EVAL_LIMIT:+--limit "${EVAL_LIMIT}"} \
      --out_rollouts_jsonl "${out_shuf_rollouts}" \
      --out_json "${out_shuf_json}"

    EPOCH_EVAL_SHUFFLED_JSONS+=("${out_shuf_json}")
    EPOCH_EVAL_SHUFFLED_ROLLOUTS+=("${out_shuf_rollouts}")
  fi
done

if [ "${#EPOCH_EVAL_JSONS[@]}" -le 0 ]; then
  echo "[ERROR] No puzzle evaluation artifacts were produced." >&2
  exit 1
fi

LAST_EVAL_IDX="$(( ${#EPOCH_EVAL_JSONS[@]} - 1 ))"
ACC_JSON_TEST="${OUT_DIR}/acc_test_k${EVAL_K}.json"
ACC_ROLLOUTS_TEST_JSONL="${EVAL_ROLLOUTS_DIR}/test.k${EVAL_K}.seed${EVAL_SEED}.jsonl"
cp -f "${EPOCH_EVAL_JSONS[${LAST_EVAL_IDX}]}" "${ACC_JSON_TEST}"
cp -f "${EPOCH_EVAL_ROLLOUTS[${LAST_EVAL_IDX}]}" "${ACC_ROLLOUTS_TEST_JSONL}"

ACC_JSON_SHUFFLED="${OUT_DIR}/acc_test_shuffled_legal_moves_k${EVAL_K}.json"
ACC_ROLLOUTS_SHUFFLED_JSONL="${EVAL_ROLLOUTS_DIR}/test_shuffled_legal_moves.k${EVAL_K}.seed${EVAL_SEED}.jsonl"
if [ "${RUN_SHUFFLED_EVAL}" = "1" ] && [ "${#EPOCH_EVAL_SHUFFLED_JSONS[@]}" -gt 0 ]; then
  LAST_SHUF_IDX="$(( ${#EPOCH_EVAL_SHUFFLED_JSONS[@]} - 1 ))"
  cp -f "${EPOCH_EVAL_SHUFFLED_JSONS[${LAST_SHUF_IDX}]}" "${ACC_JSON_SHUFFLED}"
  cp -f "${EPOCH_EVAL_SHUFFLED_ROLLOUTS[${LAST_SHUF_IDX}]}" "${ACC_ROLLOUTS_SHUFFLED_JSONL}"
fi

# -----------------------------------------------------------------------------
# Eval: fullgame (Stockfish) on final checkpoint only.

RUN_FULLGAME_EVAL="${RUN_FULLGAME_EVAL:-1}"
FULLGAME_OUT_DIR="${FULLGAME_OUT_DIR:-${OUT_DIR}/fullgame_eval}"
FULLGAME_PROMPT_TEMPLATE_PATH="${FULLGAME_PROMPT_TEMPLATE_PATH:-recipe/chess/prompt_templates/original_chessr1_prompt.jinja}"
FULLGAME_TP="${FULLGAME_TP:-4}"
FULLGAME_MAX_MODEL_LEN="${FULLGAME_MAX_MODEL_LEN:-4096}"
FULLGAME_MAX_RESPONSE_TOKENS="${FULLGAME_MAX_RESPONSE_TOKENS:-2000}"
FULLGAME_MAX_NUM_SEQS="${FULLGAME_MAX_NUM_SEQS:-1024}"
FULLGAME_MAX_NUM_BATCHED_TOKENS="${FULLGAME_MAX_NUM_BATCHED_TOKENS:-65536}"
FULLGAME_GPU_MEMORY_UTILIZATION="${FULLGAME_GPU_MEMORY_UTILIZATION:-0.9}"
FULLGAME_MAX_RETRIES_PER_TURN="${FULLGAME_MAX_RETRIES_PER_TURN:-1}"
FULLGAME_ACPL_CP_CAP="${FULLGAME_ACPL_CP_CAP:-1000}"
FULLGAME_MATE_SCORE_CP="${FULLGAME_MATE_SCORE_CP:-1000}"
FULLGAME_ACPL_WORKERS="${FULLGAME_ACPL_WORKERS:-1}"
FULLGAME_ACPL_THREADS="${FULLGAME_ACPL_THREADS:-}"
# Stockfish path:
# - In this repo's default local workflow, Stockfish is built under `.third_party_cache/stockfish/src/stockfish`.
# - In the GH200 container image (`gabr1e1/chess_rl:v1-arm-stockfish-flashinfer`), Stockfish is available at
#   `/usr/local/bin/stockfish`.
FULLGAME_STOCKFISH_PATH="${FULLGAME_STOCKFISH_PATH:-.third_party_cache/stockfish/src/stockfish}"
if [ ! -x "${FULLGAME_STOCKFISH_PATH}" ] && [ -x "/usr/local/bin/stockfish" ]; then
  FULLGAME_STOCKFISH_PATH="/usr/local/bin/stockfish"
fi
FULLGAME_TEMPERATURE="${FULLGAME_TEMPERATURE:-0.6}"
FULLGAME_TOP_P="${FULLGAME_TOP_P:-0.9}"
FULLGAME_SEED="${FULLGAME_SEED:-0}"

FULLGAME_GAMES_PER_DEPTH="${FULLGAME_GAMES_PER_DEPTH:-}"
FULLGAME_SMOKE="${FULLGAME_SMOKE:-0}"

if [ "${RUN_FULLGAME_EVAL}" = "1" ]; then
  FULLGAME_EXTRA_ARGS=()
  if [ "${FULLGAME_SMOKE}" = "1" ]; then
    FULLGAME_EXTRA_ARGS+=(--smoke-test)
  else
    if [ -n "${FULLGAME_GAMES_PER_DEPTH}" ]; then
      FULLGAME_EXTRA_ARGS+=(--games-per-depth "${FULLGAME_GAMES_PER_DEPTH}")
    fi
  fi
  if [ -n "${FULLGAME_ACPL_THREADS}" ]; then
    FULLGAME_EXTRA_ARGS+=(--acpl-threads "${FULLGAME_ACPL_THREADS}")
  fi

  python3 scripts/eval_chess_fullgame.py \
    --model "${FINAL_HF_DIR}" \
    --out-dir "${FULLGAME_OUT_DIR}" \
    --prompt-template-path "${FULLGAME_PROMPT_TEMPLATE_PATH}" \
    --tensor-parallel-size "${FULLGAME_TP}" \
    --max-model-len "${FULLGAME_MAX_MODEL_LEN}" \
    --max-response-tokens "${FULLGAME_MAX_RESPONSE_TOKENS}" \
    --max-num-seqs "${FULLGAME_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${FULLGAME_MAX_NUM_BATCHED_TOKENS}" \
    --max-retries-per-turn "${FULLGAME_MAX_RETRIES_PER_TURN}" \
    --gpu-memory-utilization "${FULLGAME_GPU_MEMORY_UTILIZATION}" \
    --acpl-workers "${FULLGAME_ACPL_WORKERS}" \
    --acpl-cp-cap "${FULLGAME_ACPL_CP_CAP}" \
    --mate-score-cp "${FULLGAME_MATE_SCORE_CP}" \
    --stockfish-path "${FULLGAME_STOCKFISH_PATH}" \
    --temperature "${FULLGAME_TEMPERATURE}" --top-p "${FULLGAME_TOP_P}" \
    --seed "${FULLGAME_SEED}" \
    --trust-remote-code \
    "${FULLGAME_EXTRA_ARGS[@]}"
else
  echo "[INFO] RUN_FULLGAME_EVAL=${RUN_FULLGAME_EVAL}; skipping full-game evaluation."
fi

# -----------------------------------------------------------------------------
# Compact results artifact.

python3 - <<PY
import json
from pathlib import Path

out_dir = Path("${OUT_DIR}")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_pass1(path: Path):
    payload = load_json(path)
    summary = payload.get("summary") or {}
    return {
        "path": str(path.resolve()),
        "pass_at_1": float(summary.get("pass_at_1_mean", float("nan"))),
        "k1_acc": float(summary.get("k1_acc_mean", float("nan"))),
        "k1_reward_mean": float(summary.get("k1_reward_mean", float("nan"))),
        "response_len_mean": float(summary.get("response_len_mean", float("nan"))),
        "valid_count_mean": float(summary.get("k32_valid_count_mean", float("nan"))),
    }

save_steps = [int(x) for x in json.loads('''${SAVE_STEPS_JSON}''')]
per_epoch = []
for epoch_idx in range(1, int("${EPOCHS}") + 1):
    p = out_dir / f"acc_test_epoch{epoch_idx}_k${EVAL_K}.json"
    if not p.exists():
        raise SystemExit(f"Missing expected puzzle eval JSON: {p}")
    m = parse_pass1(p)
    m["epoch"] = int(epoch_idx)
    per_epoch.append(m)

acc_test_path = Path("${ACC_JSON_TEST}")
if not acc_test_path.exists():
    raise SystemExit(f"Missing final puzzle eval JSON: {acc_test_path}")
acc_test = load_json(acc_test_path)

per_epoch_shuffled = []
if str("${RUN_SHUFFLED_EVAL}") == "1":
    for epoch_idx in range(1, int("${EPOCHS}") + 1):
        p = out_dir / f"acc_test_shuffled_legal_moves_epoch{epoch_idx}_k${EVAL_K}.json"
        if not p.exists():
            raise SystemExit(f"Missing expected shuffled puzzle eval JSON: {p}")
        m = parse_pass1(p)
        m["epoch"] = int(epoch_idx)
        per_epoch_shuffled.append(m)

fullgame_summary = None
summary_path = Path("${FULLGAME_OUT_DIR}") / "summary.json"
if summary_path.exists():
    fullgame_summary = load_json(summary_path)

result = {
    "run_name": "${RUN_NAME}",
    "model_base": "${MODEL_PATH}",
    "train_sft_parquet": "${TRAIN_SFT_PARQUET}",
    "sft": {
        "epochs": int("${EPOCHS}"),
        "train_batch_size": int("${TRAIN_BATCH_SIZE}"),
        "micro_batch_size_per_gpu": int("${MICRO_BATCH_SIZE_PER_GPU}"),
        "max_length": int("${MAX_LENGTH}"),
        "dp_size": int("${DP_SIZE}"),
        "steps_per_epoch": int("${STEPS_PER_EPOCH}"),
        "total_steps": int("${TOTAL_STEPS}"),
        "save_steps": save_steps,
        "checkpoint_dir": str(Path("${CKPT_DIR}").resolve()),
        "final_step": int("${FINAL_STEP}"),
        "final_hf_dir": str(Path("${FINAL_HF_DIR}").resolve()),
    },
    "eval": {
        "passk": {
            "k": int("${EVAL_K}"),
            "acc_key": "${EVAL_ACC_KEY}",
            "seed": int("${EVAL_SEED}"),
            "seed_mode": "${EVAL_SEED_MODE}",
            "temperature": float("${EVAL_TEMPERATURE}"),
            "top_p": float("${EVAL_TOP_P}"),
            "batch_size": int("${EVAL_BATCH_SIZE}"),
            "max_num_seqs": int("${EVAL_MAX_NUM_SEQS}"),
            "max_num_batched_tokens": int("${EVAL_MAX_NUM_BATCHED_TOKENS}"),
            "max_response_length": int("${EVAL_MAX_RESPONSE_LENGTH}"),
            "max_model_len": int("${EVAL_MAX_MODEL_LEN}"),
            "tensor_parallel_size": int("${EVAL_TENSOR_PARALLEL_SIZE}"),
            "gpu_memory_utilization": float("${EVAL_GPU_MEMORY_UTILIZATION}"),
            "k_chunk": (int("${EVAL_K_CHUNK}") if "${EVAL_K_CHUNK}" else None),
            "limit": (int("${EVAL_LIMIT}") if "${EVAL_LIMIT}" else None),
            "rollouts_dir": str(Path("${EVAL_ROLLOUTS_DIR}").resolve()),
            "run_shuffled_eval": bool(int("${RUN_SHUFFLED_EVAL}")),
        },
        "fullgame": {
            "enabled": bool(int("${RUN_FULLGAME_EVAL}")),
            "out_dir": str(Path("${FULLGAME_OUT_DIR}").resolve()),
            "prompt_template_path": "${FULLGAME_PROMPT_TEMPLATE_PATH}",
            "smoke": bool(int("${FULLGAME_SMOKE}")),
            "max_num_seqs": int("${FULLGAME_MAX_NUM_SEQS}"),
            "max_num_batched_tokens": int("${FULLGAME_MAX_NUM_BATCHED_TOKENS}"),
            "max_retries_per_turn": int("${FULLGAME_MAX_RETRIES_PER_TURN}"),
            "acpl_workers": int("${FULLGAME_ACPL_WORKERS}"),
            "acpl_threads": (int("${FULLGAME_ACPL_THREADS}") if "${FULLGAME_ACPL_THREADS}" else None),
            "games_per_depth": (int("${FULLGAME_GAMES_PER_DEPTH}") if "${FULLGAME_GAMES_PER_DEPTH}" else None),
        },
    },
    "metrics": {
        "test": {
            "pass_at_1": float(acc_test["summary"]["pass_at_1_mean"]),
            "by_epoch": per_epoch,
        },
        "test_shuffled_legal_moves": {
            "by_epoch": per_epoch_shuffled,
        },
        "fullgame": {
            "acpl_mean": (
                float(fullgame_summary["summary_overall"].get("acpl_mean", float("nan")))
                if fullgame_summary
                else float("nan")
            ),
            "acpl_mean_per_move": (
                float(fullgame_summary["summary_overall"].get("acpl_mean_per_move", float("nan")))
                if fullgame_summary
                else float("nan")
            ),
        },
    },
}

path = out_dir / "results.json"
tmp = out_dir / "results.json.tmp"
tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
tmp.replace(path)
print(f"Wrote {path}")
PY

# -----------------------------------------------------------------------------
# Optional: upload final HF checkpoint (+ results) to Hugging Face Hub.
#
# Defaults target the shared team repo and are intended to run on-cluster.
# Smoke policy can require this check (HF_UPLOAD_REQUIRED=1).

HF_UPLOAD_ENABLE="${HF_UPLOAD_ENABLE:-1}"
HF_UPLOAD_REQUIRED="${HF_UPLOAD_REQUIRED:-1}"
HF_UPLOAD_REPO_ID="${HF_UPLOAD_REPO_ID:-Gabr1e11/a_lot_of_models}"
HF_UPLOAD_TOKEN_PATH="${HF_UPLOAD_TOKEN_PATH:-${HOME}/.huggingface_token_zzc}"
HF_UPLOAD_PATH_PREFIX="${HF_UPLOAD_PATH_PREFIX:-rejection_sft}"
HF_UPLOAD_SUMMARY_JSON="${OUT_DIR}/hf_upload_summary.json"
HF_UPLOAD_RESULTS_PATH_IN_REPO="${HF_UPLOAD_RESULTS_PATH_IN_REPO:-results.json}"

if [ "${HF_UPLOAD_ENABLE}" = "1" ]; then
  if ! python3 - <<PY
import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

summary_path = Path("${HF_UPLOAD_SUMMARY_JSON}")
out_dir = Path("${OUT_DIR}")
final_hf_dir = Path("${FINAL_HF_DIR}")
results_json = out_dir / "results.json"

repo_id = os.environ.get("HF_UPLOAD_REPO_ID", "").strip()
token_path = os.environ.get("HF_UPLOAD_TOKEN_PATH", "").strip()
prefix = os.environ.get("HF_UPLOAD_PATH_PREFIX", "rejection_sft").strip().strip("/")
run_id = os.environ.get("CHESS_SFT_RUN_ID", "").strip() or "manual_run"
run_name = os.environ.get("RUN_NAME", "").strip() or "rejection_sft"
results_path_suffix = os.environ.get("HF_UPLOAD_RESULTS_PATH_IN_REPO", "results.json").strip() or "results.json"

def write_summary(payload: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(summary_path)

def sanitize(s: str) -> str:
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-")
    return x or "x"

def read_token(path: str) -> str:
    if not path:
        raise RuntimeError("HF_UPLOAD_TOKEN_PATH is empty.")
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"HF token file not found: {p}")
    tok = p.read_text(encoding="utf-8").strip()
    if not tok:
        raise RuntimeError(f"HF token file is empty: {p}")
    return tok

try:
    if not repo_id:
        raise RuntimeError("HF_UPLOAD_REPO_ID is empty.")
    if not final_hf_dir.is_dir():
        raise RuntimeError(f"Final HF dir missing: {final_hf_dir}")
    if not results_json.is_file():
        raise RuntimeError(f"Missing results.json: {results_json}")

    try:
        from huggingface_hub import HfApi
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"huggingface_hub import failed: {e}") from e

    token = read_token(token_path)
    model_tag = sanitize(str(os.environ.get("MODEL_PATH", "")) or "model")
    path_in_repo = f"{prefix}/{sanitize(run_id)}/{sanitize(run_name)}/{model_tag}"
    model_path_in_repo = f"{path_in_repo}/final_hf_model"
    results_path_in_repo = f"{path_in_repo}/{results_path_suffix}"

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    commit_message = f"Upload rejection-SFT run {run_id} ({run_name})"

    # 1) Upload model folder (weights + tokenizer/config files).
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(final_hf_dir),
        path_in_repo=model_path_in_repo,
        commit_message=commit_message,
    )

    # 2) Upload results summary artifact.
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(results_json),
        path_in_repo=results_path_in_repo,
        commit_message=commit_message,
    )

    # 3) Verify via HF REST API that key uploaded files exist.
    verify_url = f"https://huggingface.co/api/models/{repo_id}"
    req = Request(
        verify_url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    with urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    siblings = payload.get("siblings") or []
    names = {str(x.get("rfilename", "")) for x in siblings}
    required = [
        f"{model_path_in_repo}/config.json",
        f"{model_path_in_repo}/model.safetensors.index.json",
        results_path_in_repo,
    ]
    missing = [x for x in required if x not in names]
    if missing:
        raise RuntimeError(f"HF upload verification failed, missing files: {missing}")

    summary = {
        "enabled": True,
        "success": True,
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "model_path_in_repo": model_path_in_repo,
        "results_path_in_repo": results_path_in_repo,
        "verified_required_files": required,
    }
    write_summary(summary)

    # Best-effort: append upload summary into local results.json for provenance.
    try:
        results_payload = json.loads(results_json.read_text(encoding="utf-8"))
        results_payload["artifact_upload"] = summary
        tmp = results_json.with_suffix(results_json.suffix + ".tmp")
        tmp.write_text(json.dumps(results_payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(results_json)
    except Exception:
        pass

    print(f"[HF-UPLOAD] success repo={repo_id} path={path_in_repo}", flush=True)
except Exception as e:
    summary = {
        "enabled": True,
        "success": False,
        "repo_id": repo_id,
        "error": str(e),
    }
    write_summary(summary)
    print(f"[HF-UPLOAD] failed: {e}", file=sys.stderr, flush=True)
    raise
PY
  then
    if [ "${HF_UPLOAD_REQUIRED}" = "1" ]; then
      echo "[ERROR] HF upload failed and HF_UPLOAD_REQUIRED=${HF_UPLOAD_REQUIRED}." >&2
      exit 1
    fi
    echo "[WARN] HF upload failed but HF_UPLOAD_REQUIRED=${HF_UPLOAD_REQUIRED}; continuing." >&2
  fi
else
  python3 - <<PY
import json
from pathlib import Path

summary_path = Path("${HF_UPLOAD_SUMMARY_JSON}")
summary_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "enabled": False,
    "success": False,
    "reason": "HF_UPLOAD_ENABLE=0",
}
tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
tmp.replace(summary_path)
print(f"[HF-UPLOAD] skipped; wrote {summary_path}")
PY
fi
