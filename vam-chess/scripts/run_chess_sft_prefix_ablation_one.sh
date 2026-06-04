#!/usr/bin/env bash
set -euo pipefail
set -x

# Run one SFT variant (A/B/C/D/E):
#   1) build SFT train/val parquets from the hard dataset
#   2) train with veRL FSDP SFTTrainer for 2 epochs (saving hf_model checkpoints)
#   3) evaluate pass@k (vLLM) after each epoch with *no forced prefix*
#
# Intended to be run inside the repo's training container (python3.10+).

if [ $# -lt 1 ]; then
  echo "Usage: $0 <VARIANT {A|B|C|D|E}>" >&2
  exit 2
fi

VARIANT="$1"
# Optional: decouple the *forced-prefix letter* (A/B/C/D/E) from the *run name* used for
# output directories and bookkeeping (e.g., A_sample_shuffle_wAWR).
RUN_NAME="${RUN_NAME:-${VARIANT}}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
TRAIN_IN_PARQUET="${TRAIN_IN_PARQUET:-data/chess_puzzles/train_hard.parquet}"
TEST_IN_PARQUET="${TEST_IN_PARQUET:-data/chess_puzzles/test.parquet}"

OUT_ROOT="${OUT_ROOT:-outputs/sft_prefix_ablation}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${RUN_NAME}}"
mkdir -p "${OUT_DIR}"

TRAIN_SFT_PARQUET="${TRAIN_SFT_PARQUET:-${OUT_DIR}/train_sft.parquet}"
VAL_SFT_PARQUET="${VAL_SFT_PARQUET:-${OUT_DIR}/val_sft.parquet}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
VAL_LIMIT="${VAL_LIMIT:-}"

# Forced-move sampling (mirrors `forced_prefix.move_temperature` in RL).
MOVE_TEMPERATURE="${MOVE_TEMPERATURE:-2.0}"
MOVE_SEED="${MOVE_SEED:-0}"

# Sampled-move offline variants: expand N `{move}` samples per prompt.
NUM_MOVE_SAMPLES="${NUM_MOVE_SAMPLES:-1}"
SAMPLE_ORDERING="${SAMPLE_ORDERING:-no_shuffle}"     # {shuffle,no_shuffle}
SFT_WEIGHTING="${SFT_WEIGHTING:-uniform}"           # {awr,uniform,best_only}
SFT_AWR_BETA="${SFT_AWR_BETA:-2.0}"                 # mirrors forced_prefix.beta in RL

# SFT hyperparams (keep constant across variants).
EPOCHS="${EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"                # global batch size (will be divided by dp)
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}" # per-rank micro batch
MAX_LENGTH="${MAX_LENGTH:-1024}"                          # prompt+response length cap in tokens
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"         # optional: early-exit after N optimizer steps

# vLLM pass@k eval hyperparams (prefix-free evaluation).
EVAL_K="${EVAL_K:-32}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_SEED_MODE="${EVAL_SEED_MODE:-engine}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-1024}"
EVAL_MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-2000}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-4096}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.6}"
EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-2}"
EVAL_K_CHUNK="${EVAL_K_CHUNK:-}"
EVAL_PRINT_PROMPT_EXAMPLES="${EVAL_PRINT_PROMPT_EXAMPLES:-0}"
EVAL_EVERY_FRAC="${EVAL_EVERY_FRAC:-0.25}"

# FSDP SFTTrainer checkpointing.
CKPT_DIR="${CKPT_DIR:-${OUT_DIR}/checkpoints}"
# Saving full sharded model + hf_model at many checkpoints is extremely expensive.
# For this ablation, default to saving only hf_model (sufficient for vLLM eval).
SAVE_CONTENTS="${SAVE_CONTENTS:-[\"hf_model\"]}"
LOAD_CONTENTS="${LOAD_CONTENTS:-[]}"
export SAVE_CONTENTS LOAD_CONTENTS

python3 -m scripts.build_chess_sft_prefix_dataset \
  --in_parquet "${TRAIN_IN_PARQUET}" \
  --out_parquet "${TRAIN_SFT_PARQUET}" \
  --variant "${VARIANT}" \
  --num_move_samples "${NUM_MOVE_SAMPLES}" \
  --sample_ordering "${SAMPLE_ORDERING}" \
  --sft_weighting "${SFT_WEIGHTING}" \
  --awr_beta "${SFT_AWR_BETA}" \
  --move_temperature "${MOVE_TEMPERATURE}" \
  --seed "${MOVE_SEED}" \
  ${TRAIN_LIMIT:+--limit "${TRAIN_LIMIT}"} \
  --print_examples 2

python3 -m scripts.build_chess_sft_prefix_dataset \
  --in_parquet "${TEST_IN_PARQUET}" \
  --out_parquet "${VAL_SFT_PARQUET}" \
  --variant "${VARIANT}" \
  --num_move_samples "${NUM_MOVE_SAMPLES}" \
  --sample_ordering "${SAMPLE_ORDERING}" \
  --sft_weighting "${SFT_WEIGHTING}" \
  --awr_beta "${SFT_AWR_BETA}" \
  --move_temperature "${MOVE_TEMPERATURE}" \
  --seed "${MOVE_SEED}" \
  ${VAL_LIMIT:+--limit "${VAL_LIMIT}"} \
  --print_examples 0

TRAIN_ROWS="$(python3 - <<PY
import pyarrow.parquet as pq
pf = pq.ParquetFile("${TRAIN_SFT_PARQUET}")
print(int(pf.metadata.num_rows))
PY
)"

# NOTE: FSDP SFTTrainer expects torchrun to set LOCAL_RANK/RANK/WORLD_SIZE.
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

# `FSDPSFTTrainer` uses DistributedSampler(drop_last=True) and DataLoader(drop_last=True).
# It normalizes the *global* batch size by DP size (world size).
DP_SIZE="${DP_SIZE:-${NPROC_PER_NODE}}"
if [ "$((TRAIN_BATCH_SIZE % DP_SIZE))" -ne 0 ]; then
  echo "[ERROR] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} must be divisible by DP_SIZE=${DP_SIZE} (NPROC_PER_NODE=${NPROC_PER_NODE})." >&2
  exit 1
fi
TRAIN_BATCH_SIZE_PER_RANK="$((TRAIN_BATCH_SIZE / DP_SIZE))"
# DistributedSampler(drop_last=True) drops the remainder so each rank gets exactly:
#   samples_per_rank = floor(TRAIN_ROWS / DP_SIZE)
# Then DataLoader(drop_last=True) drops the remainder again at the batch level:
#   steps_per_epoch = floor(samples_per_rank / TRAIN_BATCH_SIZE_PER_RANK)
SAMPLES_PER_RANK="$((TRAIN_ROWS / DP_SIZE))"
STEPS_PER_EPOCH="$((SAMPLES_PER_RANK / TRAIN_BATCH_SIZE_PER_RANK))"
if [ "${STEPS_PER_EPOCH}" -le 0 ]; then
  echo "[ERROR] Computed STEPS_PER_EPOCH=${STEPS_PER_EPOCH} (TRAIN_ROWS=${TRAIN_ROWS}, DP_SIZE=${DP_SIZE}, SAMPLES_PER_RANK=${SAMPLES_PER_RANK}, TRAIN_BATCH_SIZE_PER_RANK=${TRAIN_BATCH_SIZE_PER_RANK})." >&2
  exit 1
fi

EPOCH1_STEP="${STEPS_PER_EPOCH}"
TOTAL_STEPS="$((STEPS_PER_EPOCH * EPOCHS))"
EPOCH2_STEP="${TOTAL_STEPS}"
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
  TOTAL_STEPS="$(python3 - <<PY
import sys
v = "${TOTAL_TRAINING_STEPS}".strip()
if not v:
    print("")
    raise SystemExit(0)
try:
    n = int(v)
except Exception:
    raise SystemExit(f"TOTAL_TRAINING_STEPS must be int, got {v!r}")
if n <= 0:
    raise SystemExit(f"TOTAL_TRAINING_STEPS must be > 0, got {n}")
print(n)
PY
)"
  # In fixed-step mode, treat the "epoch endpoints" as the final step for compatibility.
  EPOCH1_STEP="${TOTAL_STEPS}"
  EPOCH2_STEP="${TOTAL_STEPS}"
fi

# Build quarter-epoch eval/save schedule.
SAVE_STEPS_JSON="$(python3 - <<PY
import json
import math

steps_per_epoch = int(${STEPS_PER_EPOCH})
epochs = int(${EPOCHS})
every = float(${EVAL_EVERY_FRAC})
if every <= 0 or every > 1.0:
    raise SystemExit(f"EVAL_EVERY_FRAC must be in (0,1], got {every}")
total_steps = steps_per_epoch * epochs
override_total = "${TOTAL_TRAINING_STEPS}".strip()
if override_total:
    total_steps = int(override_total)

n = int(round(epochs / every))
steps = []
for i in range(1, n + 1):
    frac = every * i
    step = int(round(steps_per_epoch * frac))
    step = max(1, min(total_steps, step))
    steps.append(step)

steps = sorted(set(steps))
if override_total and int(override_total) not in steps:
    # Always include the final step in fixed-step mode.
    steps.append(int(override_total))
    steps = sorted(set(steps))
print(json.dumps(steps))
PY
)"
SAVE_STEPS_SPACE="$(python3 - <<PY
import json
steps = json.loads('''${SAVE_STEPS_JSON}''')
print(' '.join(str(int(x)) for x in steps))
PY
)"

# Train.
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" -m verl.trainer.fsdp_sft_trainer \
  data.train_files="${TRAIN_SFT_PARQUET}" \
  data.val_files="${VAL_SFT_PARQUET}" \
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
  trainer.project_name=chess_sft_prefix \
  trainer.experiment_name="sft_${RUN_NAME}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.total_epochs="${EPOCHS}" \
  ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"} \
  trainer.save_freq=-1 \
  +trainer.save_steps="${SAVE_STEPS_JSON}" \
  trainer.test_freq=-1 \
  trainer.logger='["console"]' \
  trainer.resume_mode=disable \
  trainer.checkpoint.save_contents="${SAVE_CONTENTS}" \
  trainer.checkpoint.load_contents="${LOAD_CONTENTS}"

EVAL_LIMIT="${EVAL_LIMIT:-}"

# Pass@k evaluation at each checkpointed quarter-epoch step.
for step in ${SAVE_STEPS_SPACE}; do
  step_str="$(printf "%06d" "${step}")"
  out_json="${OUT_DIR}/passk_step${step_str}_k${EVAL_K}.json"
  python3 -m scripts.eval_chess_passk \
    --model "${CKPT_DIR}/global_step_${step}/huggingface" \
    --tokenizer "${MODEL_PATH}" \
    --parquet "${TEST_IN_PARQUET}" \
    --k_max "${EVAL_K}" --do_sample \
    ${EVAL_K_CHUNK:+--k_chunk "${EVAL_K_CHUNK}"} \
    --seed "${EVAL_SEED}" --seed_mode "${EVAL_SEED_MODE}" \
    --batch_size "${EVAL_BATCH_SIZE}" --max_num_seqs "${EVAL_MAX_NUM_SEQS}" \
    --max_response_length "${EVAL_MAX_RESPONSE_LENGTH}" --max_model_len "${EVAL_MAX_MODEL_LEN}" \
    --gpu_memory_utilization "${EVAL_GPU_MEMORY_UTILIZATION}" \
    --tensor_parallel_size "${EVAL_TENSOR_PARALLEL_SIZE}" \
    --print_prompt_examples "${EVAL_PRINT_PROMPT_EXAMPLES}" \
    ${EVAL_LIMIT:+--limit "${EVAL_LIMIT}"} \
    --out_json "${out_json}"
done

# Compatibility symlinks/copies for epoch endpoints.
EVAL_JSON_E1="${OUT_DIR}/passk_epoch1_k${EVAL_K}.json"
EVAL_JSON_E2="${OUT_DIR}/passk_epoch2_k${EVAL_K}.json"
cp -f "${OUT_DIR}/passk_step$(printf "%06d" "${EPOCH1_STEP}")_k${EVAL_K}.json" "${EVAL_JSON_E1}"
cp -f "${OUT_DIR}/passk_step$(printf "%06d" "${EPOCH2_STEP}")_k${EVAL_K}.json" "${EVAL_JSON_E2}"

# Write a compact artifact for downstream aggregation/plotting.
python3 - <<PY
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

variant = "${VARIANT}"
run_name = "${RUN_NAME}"
out_dir = Path("${OUT_DIR}")

def load(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

e1 = load(out_dir / f"passk_epoch1_k${EVAL_K}.json")
e2 = load(out_dir / f"passk_epoch2_k${EVAL_K}.json")

save_steps = json.loads('''${SAVE_STEPS_JSON}''')
eval_points = []
for s in save_steps:
    s = int(s)
    payload = load(out_dir / f"passk_step{str(s).zfill(6)}_k${EVAL_K}.json")
    eval_points.append(
        {
            "global_step": s,
            "epoch_frac": float(s) / float(${STEPS_PER_EPOCH}),
            "pass@k_acc": float(payload["summary"]["k32_acc_mean"]),
            "k1_acc": float(payload["summary"]["k1_acc_mean"]),
            "k32_reward_mean": float(payload["summary"]["k32_reward_mean"]),
            "valid_count_mean": float(payload["summary"].get("k32_valid_count_mean", float("nan"))),
            "unique_valid_moves_mean": float(payload["summary"].get("k32_unique_valid_moves_mean", float("nan"))),
            "expected_score_sum_mean": float(payload["summary"].get("k32_expected_score_sum_mean", float("nan"))),
        }
    )

# Fetch the variant definitions from the emitted SFT parquet (so the artifact is self-contained).
meta = pq.read_table("${TRAIN_SFT_PARQUET}", columns=["forced_prefix_template", "strip_phrase_template", "variant"]).slice(0, 1).to_pylist()
if meta:
    forced_prefix_template = meta[0].get("forced_prefix_template")
    strip_phrase_template = meta[0].get("strip_phrase_template")
else:
    forced_prefix_template = None
    strip_phrase_template = None

result = {
    "variant": variant,
    "run_name": run_name,
    "forced_prefix_template": forced_prefix_template,
    "strip_phrase_template": strip_phrase_template,
    "train_in_parquet": "${TRAIN_IN_PARQUET}",
    "test_in_parquet": "${TEST_IN_PARQUET}",
    "model_base": "${MODEL_PATH}",
    "move_temperature": float("${MOVE_TEMPERATURE}"),
    "move_seed": int("${MOVE_SEED}"),
    "offline_sampling": {
        "num_move_samples": int("${NUM_MOVE_SAMPLES}"),
        "sample_ordering": "${SAMPLE_ORDERING}",
        "sft_weighting": "${SFT_WEIGHTING}",
        "awr_beta": float("${SFT_AWR_BETA}"),
    },
    "sft": {
        "epochs": int("${EPOCHS}"),
        "total_training_steps": (int("${TOTAL_TRAINING_STEPS}") if "${TOTAL_TRAINING_STEPS}" else None),
        "train_batch_size": int("${TRAIN_BATCH_SIZE}"),
        "micro_batch_size_per_gpu": int("${MICRO_BATCH_SIZE_PER_GPU}"),
        "max_length": int("${MAX_LENGTH}"),
        "steps_per_epoch": int("${STEPS_PER_EPOCH}"),
        "epoch_steps": {"epoch1": int("${EPOCH1_STEP}"), "epoch2": int("${EPOCH2_STEP}")},
        "save_steps": [int(x) for x in save_steps],
        "checkpoint_dir": str(Path("${CKPT_DIR}").resolve()),
        "save_contents": json.loads(os.environ.get("SAVE_CONTENTS", "[]")),
        "load_contents": json.loads(os.environ.get("LOAD_CONTENTS", "[]")),
    },
    "eval": {
        "k": int("${EVAL_K}"),
        "k_chunk": (int("${EVAL_K_CHUNK}") if "${EVAL_K_CHUNK}" else None),
        "seed": int("${EVAL_SEED}"),
        "seed_mode": "${EVAL_SEED_MODE}",
        "batch_size": int("${EVAL_BATCH_SIZE}"),
        "max_num_seqs": int("${EVAL_MAX_NUM_SEQS}"),
        "max_response_length": int("${EVAL_MAX_RESPONSE_LENGTH}"),
        "max_model_len": int("${EVAL_MAX_MODEL_LEN}"),
        "gpu_memory_utilization": float("${EVAL_GPU_MEMORY_UTILIZATION}"),
        "tensor_parallel_size": int("${EVAL_TENSOR_PARALLEL_SIZE}"),
        "every_frac": float("${EVAL_EVERY_FRAC}"),
    },
    "metrics": {
        "eval_points": eval_points,
        # Convenience fields:
        "epoch1": {
            "pass@k_acc": float(e1["summary"]["k32_acc_mean"]),
            "valid_count_mean": float(e1["summary"].get("k32_valid_count_mean", float("nan"))),
            "unique_valid_moves_mean": float(e1["summary"].get("k32_unique_valid_moves_mean", float("nan"))),
            "expected_score_sum_mean": float(e1["summary"].get("k32_expected_score_sum_mean", float("nan"))),
        },
        "epoch2": {
            "pass@k_acc": float(e2["summary"]["k32_acc_mean"]),
            "valid_count_mean": float(e2["summary"].get("k32_valid_count_mean", float("nan"))),
            "unique_valid_moves_mean": float(e2["summary"].get("k32_unique_valid_moves_mean", float("nan"))),
            "expected_score_sum_mean": float(e2["summary"].get("k32_expected_score_sum_mean", float("nan"))),
        },
    },
}

out_path = out_dir / "results.json"
tmp_path = out_dir / "results.json.tmp"
with tmp_path.open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
tmp_path.replace(out_path)
print(f"Wrote {out_path}")
PY
