#!/bin/bash
#SBATCH --job-name=chess-sft-reject-build
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/slurm-%x-%A.out
#SBATCH --error=slurm/slurm-%x-%A.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

# Build a baseline-prompt SFT dataset via *rejection sampling* on GH200 / Isambard.
#
# This script runs `scripts/build_chess_sft_rejection_dataset.py` on 2 nodes by sharding prompts:
#   shard_idx = SLURM_PROCID, num_shards = SLURM_NTASKS
#
# Output contract (under OUT_ROOT):
#   rejection_sft_dataset/
#     - shards/train_sft__<model_tag>__shard_<i>.parquet (+ stats)
#     - train_sft__<model_tag>.parquet              (merged)
#     - train_sft__<model_tag>.parquet.stats.json   (merged stats)
#
# Required env:
#   - CHESS_SFT_RUN_ID: output run id under /projects/a5l/ziyan/chess_rl_outputs/
#
# Optional env:
#   - CHESS_SFT_MODE: {smoke|full} (default: smoke)
#   - MODEL_PATH: HF id for generation (default depends on mode)
#   - LIMIT_ROWS: cap input prompts before sharding (default smoke=1000, full=unset)
#
# Submit example (smoke, blocks):
#   sbatch --wait --export=ALL,CHESS_SFT_RUN_ID=rs_smoke_$(date +%Y%m%d_%H%M%S),CHESS_SFT_MODE=smoke,MODEL_PATH=Qwen/Qwen2.5-3B-Instruct,LIMIT_ROWS=1000 sbatch_build_chess_sft_rejection_dataset_gh200.slurm

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
  mkdir -p "${CHESS_RL_SINGULARITY_TMPDIR}" "${CHESS_RL_TMPDIR}" "${CHESS_RL_APPTAINER_CACHEDIR}" || true
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

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

echo "[SLURM] Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "[SLURM] Node list:   ${SLURM_JOB_NODELIST:-N/A}"
echo "[SLURM] Ntasks:      ${SLURM_NTASKS:-N/A} (ntasks-per-node=${SLURM_NTASKS_PER_NODE:-N/A})"
echo "[SLURM] GPUs/node:   ${SLURM_GPUS_ON_NODE:-4} (requested via --gres=gpu:4)"
echo "[SLURM] Submit dir:  ${SLURM_SUBMIT_DIR:-$(pwd)}"

# Run relative to submit dir (repo root).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  THIS_DIR="${SLURM_SUBMIT_DIR}"
else
  THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${THIS_DIR}"
echo "[SLURM] Repo dir:    ${THIS_DIR}"

HOST_ARCH="$(uname -m 2>/dev/null || echo unknown)"
echo "[SLURM] Host arch:   ${HOST_ARCH}"
if [ "${HOST_ARCH}" != "aarch64" ] && [ "${HOST_ARCH}" != "arm64" ]; then
  die "Expected arm64/aarch64 compute node, got '${HOST_ARCH}'."
fi

RUN_ID="${CHESS_SFT_RUN_ID:-}"
if [ -z "${RUN_ID}" ]; then
  die "CHESS_SFT_RUN_ID is required"
fi

MODE="${CHESS_SFT_MODE:-smoke}"
case "${MODE}" in
  smoke|full) ;;
  *) die "CHESS_SFT_MODE must be one of {smoke,full}, got '${MODE}'" ;;
esac

if [ "${MODE}" == "smoke" ]; then
  export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
  export LIMIT_ROWS="${LIMIT_ROWS:-}"
  export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2000}"
  export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
  export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
  export BATCH_SIZE="${BATCH_SIZE:-$((128 * 8))}"
  export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
  export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
  export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
else
  export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
  export LIMIT_ROWS="${LIMIT_ROWS:-}"
  export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2000}"
  export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
  export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
  export BATCH_SIZE="${BATCH_SIZE:-$((128 * 8))}"
  export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
  export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
  export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
fi

sanitize_model_tag() {
  local m="$1"
  local base
  base="$(basename "${m}")"
  echo "${base}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g'
}

MODEL_TAG="$(sanitize_model_tag "${MODEL_PATH}")"
DATASET_STEM="train_sft__${MODEL_TAG}"

# Container image.
SIF_IMAGE="${CHESS_RL_SIF_IMAGE:-${HOME}/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif}"
CONTAINER_REPO_DIR="${CHESS_RL_CONTAINER_REPO_DIR:-/workspace/chess_rl}"
if [ ! -f "${SIF_IMAGE}" ]; then
  cat >&2 <<EOF
[ERROR] Missing SIF image: ${SIF_IMAGE}

Build it once on a login node, then re-run:
  apptainer build "${SIF_IMAGE}" docker://gabr1e1/chess_rl:v1-arm-stockfish-flashinfer
EOF
  exit 1
fi
echo "[SLURM] SIF_IMAGE=${SIF_IMAGE}"

# Persistent output root (avoid $HOME quota).
PROJECT_ROOT="${CHESS_RL_PROJECT_ROOT:-/projects/a5l/ziyan}"
OUT_ROOT="${CHESS_SFT_OUT_ROOT:-${PROJECT_ROOT%/}/chess_rl_outputs/${RUN_ID}}"
mkdir -p "${OUT_ROOT}"
echo "[SLURM] OUT_ROOT=${OUT_ROOT}"

# Share HF caches across jobs (avoid repeated downloads; avoid $HOME quota).
HF_CACHE_ROOT="${CHESS_RL_HF_CACHE_ROOT:-${PROJECT_ROOT%/}/hf_cache}"
mkdir -p "${HF_CACHE_ROOT}"
export HF_HOME="${HF_CACHE_ROOT%/}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

DATASET_DIR="${OUT_ROOT%/}/rejection_sft_dataset"
SHARDS_DIR="${DATASET_DIR%/}/shards"
mkdir -p "${SHARDS_DIR}"

REBUILD_DATASET="${REBUILD_DATASET:-1}"

IN0="data/chess_puzzles_chessr1_aligned_sharded_baseline/train_0.parquet"
IN1="data/chess_puzzles_chessr1_aligned_sharded_baseline/train_1.parquet"

if [ "${MODE}" = "smoke" ] && [ -z "${LIMIT_ROWS:-}" ]; then
  echo "[CFG] LIMIT_ROWS is unset in smoke mode; computing exact 1/10 prompt count from ${IN0} + ${IN1}..."
  TOTAL_PROMPTS="$(
    apptainer exec --nv \
      --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
      "${SIF_IMAGE}" \
      bash -lc '
        set -euo pipefail
        cd "'"${CONTAINER_REPO_DIR}"'"
        python3 - <<PY
import pyarrow.parquet as pq
paths = [
    "data/chess_puzzles_chessr1_aligned_sharded_baseline/train_0.parquet",
    "data/chess_puzzles_chessr1_aligned_sharded_baseline/train_1.parquet",
]
total = sum(int(pq.ParquetFile(p).metadata.num_rows) for p in paths)
print(total)
PY
      '
  )"
  if [ -z "${TOTAL_PROMPTS}" ]; then
    die "Failed to compute TOTAL_PROMPTS for smoke LIMIT_ROWS."
  fi
  LIMIT_ROWS="$((TOTAL_PROMPTS / 10))"
  if [ "${LIMIT_ROWS}" -le 0 ]; then
    die "Computed LIMIT_ROWS=${LIMIT_ROWS} from TOTAL_PROMPTS=${TOTAL_PROMPTS}; expected > 0."
  fi
  export LIMIT_ROWS
  echo "[CFG] TOTAL_PROMPTS=${TOTAL_PROMPTS} -> LIMIT_ROWS=${LIMIT_ROWS} (exact 1/10)."
fi

echo "[CFG] MODE=${MODE}"
echo "[CFG] MODEL_PATH=${MODEL_PATH}"
echo "[CFG] MODEL_TAG=${MODEL_TAG}"
echo "[CFG] LIMIT_ROWS=${LIMIT_ROWS:-<unset>}"
echo "[CFG] REBUILD_DATASET=${REBUILD_DATASET}"
echo "[CFG] BATCH_SIZE=${BATCH_SIZE}"
echo "[CFG] GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "[CFG] MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "[CFG] MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"

OUT_MERGED="${DATASET_DIR%/}/${DATASET_STEM}.parquet"
OUT_STATS="${DATASET_DIR%/}/${DATASET_STEM}.parquet.stats.json"

mkdir -p slurm

echo "[PATH] IN0=${IN0}"
echo "[PATH] IN1=${IN1}"
echo "[PATH] SHARDS_DIR=${SHARDS_DIR}"
echo "[PATH] OUT_MERGED=${OUT_MERGED}"

if [ "${REBUILD_DATASET}" = "1" ]; then
  rm -f "${SHARDS_DIR%/}/${DATASET_STEM}__shard_"*.parquet || true
  rm -f "${SHARDS_DIR%/}/${DATASET_STEM}__shard_"*.parquet.stats.json || true
  rm -f "${SHARDS_DIR%/}/shard_"*.parquet || true
  rm -f "${SHARDS_DIR%/}/shard_"*.parquet.stats.json || true
  rm -f "${OUT_MERGED}" "${OUT_STATS}" || true
  echo "[CFG] Cleared existing dataset artifacts for ${DATASET_STEM}."
else
  if [ -e "${OUT_MERGED}" ] || [ -e "${OUT_STATS}" ]; then
    die "Output already exists and REBUILD_DATASET=${REBUILD_DATASET}: ${OUT_MERGED} / ${OUT_STATS}"
  fi
fi

# Run one task per node; each task uses all 4 GPUs via vLLM tensor parallel.
export TOKENIZERS_PARALLELISM=false

srun --ntasks="${SLURM_NTASKS}" --ntasks-per-node=1 \
  apptainer exec --nv \
    --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
    --bind "${OUT_ROOT}:${OUT_ROOT}" \
    --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
    "${SIF_IMAGE}" \
    bash -lc '
      set -euo pipefail
      cd "'"${CONTAINER_REPO_DIR}"'"

      shard_idx="${SLURM_PROCID:-0}"
      num_shards="${SLURM_NTASKS:-1}"
      out_parquet="'"${SHARDS_DIR}"'/'"${DATASET_STEM}"'__shard_${shard_idx}.parquet"
      out_stats="'"${SHARDS_DIR}"'/'"${DATASET_STEM}"'__shard_${shard_idx}.parquet.stats.json"

      echo "[TASK] shard_idx=${shard_idx} num_shards=${num_shards} out=${out_parquet}"

      python3 -m scripts.build_chess_sft_rejection_dataset \
        --model "'"${MODEL_PATH}"'" \
        --in-parquet "'"${IN0}"'" \
        --in-parquet "'"${IN1}"'" \
        --out-parquet "${out_parquet}" \
        --out-stats-json "${out_stats}" \
        ${LIMIT_ROWS:+--limit-rows ${LIMIT_ROWS}} \
        --shard-idx "${shard_idx}" --num-shards "${num_shards}" \
        --samples-per-prompt 8 \
        --batch-size "'"${BATCH_SIZE}"'" \
        --max-prompt-length 1536 \
        --max-response-length "'"${MAX_RESPONSE_LENGTH}"'" \
        --max-model-len "'"${MAX_MODEL_LEN}"'" \
        --tensor-parallel-size "'"${TENSOR_PARALLEL_SIZE}"'" \
        --gpu-memory-utilization "'"${GPU_MEMORY_UTILIZATION}"'" \
        --max-num-seqs "'"${MAX_NUM_SEQS}"'" \
        --max-num-batched-tokens "'"${MAX_NUM_BATCHED_TOKENS}"'" \
        --temperature 0.6 --top-p 0.9 \
        --assistant-source canonical
    '

echo "[MERGE] Merging shards -> ${OUT_MERGED}"
apptainer exec --nv \
  --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
  --bind "${OUT_ROOT}:${OUT_ROOT}" \
  "${SIF_IMAGE}" \
  bash -lc '
    set -euo pipefail
    cd "'"${CONTAINER_REPO_DIR}"'"
    python3 - <<PY
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

shards_dir = Path("'"${SHARDS_DIR}"'")
out_parquet = Path("'"${OUT_MERGED}"'")
out_stats = Path("'"${OUT_STATS}"'")

shard_paths = sorted(shards_dir.glob("'"${DATASET_STEM}"'__shard_*.parquet"))
if not shard_paths:
    raise SystemExit(f"No shard parquet files found under {shards_dir}")

schema = pq.ParquetFile(str(shard_paths[0])).schema_arrow
writer = pq.ParquetWriter(str(out_parquet), schema, compression="zstd")
rows = 0
for p in shard_paths:
    dataset = ds.dataset(str(p), format="parquet")
    for batch in dataset.scanner().to_batches():
        rows += int(batch.num_rows)
        writer.write_table(pa.Table.from_batches([batch], schema=schema))
writer.close()

stats_paths = sorted(shards_dir.glob("'"${DATASET_STEM}"'__shard_*.parquet.stats.json"))
stats = []
for p in stats_paths:
    try:
        stats.append(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        continue

def get(d, *keys, default=0.0):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur

total = {
    "shards_dir": str(shards_dir),
    "out_parquet": str(out_parquet),
    "out_rows": int(rows),
    "shards": [str(p) for p in shard_paths],
    "stats_paths": [str(p) for p in stats_paths],
    "aggregated": {},
}

if stats:
    total_prompts_used = int(sum(get(s, "counts", "total_prompts_used", default=0) for s in stats))
    total_samples = int(sum(get(s, "counts", "total_samples_generated", default=0) for s in stats))
    accepted_samples = int(sum(get(s, "counts", "accepted_samples", default=0) for s in stats))
    accepted_prompts = int(sum(get(s, "counts", "accepted_prompts", default=0) for s in stats))
    prompt_tokens_total = int(sum(get(s, "tokens", "prompt_tokens_total", default=0) for s in stats))
    response_tokens_total = int(sum(get(s, "tokens", "response_tokens_total", default=0) for s in stats))

    # Parallel wall time: approximate by max(shard elapsed).
    elapsed_max = float(max(get(s, "throughput", "elapsed_s", default=0.0) for s in stats))
    elapsed_sum = float(sum(get(s, "throughput", "elapsed_s", default=0.0) for s in stats))

    total["aggregated"] = {
        "total_prompts_used": total_prompts_used,
        "total_samples_generated": total_samples,
        "accepted_samples": accepted_samples,
        "accepted_prompts": accepted_prompts,
        "accept_rate_per_sample": (accepted_samples / total_samples) if total_samples else 0.0,
        "accept_rate_per_prompt": (accepted_prompts / total_prompts_used) if total_prompts_used else 0.0,
        "prompt_tokens_total": prompt_tokens_total,
        "response_tokens_total": response_tokens_total,
        "elapsed_s_max_shard": elapsed_max,
        "elapsed_s_sum_shards": elapsed_sum,
        "approx_prompts_per_s_parallel": (total_prompts_used / elapsed_max) if elapsed_max else 0.0,
        "approx_samples_per_s_parallel": (total_samples / elapsed_max) if elapsed_max else 0.0,
        "approx_response_tokens_per_s_parallel": (response_tokens_total / elapsed_max) if elapsed_max else 0.0,
    }

tmp = out_stats.with_suffix(out_stats.suffix + ".tmp")
tmp.write_text(json.dumps(total, indent=2) + "\\n", encoding="utf-8")
tmp.replace(out_stats)
print(f"[MERGE] wrote {out_parquet} rows={rows}")
print(f"[MERGE] wrote {out_stats}")
PY
  '

echo "[DONE] Built rejection-sampled dataset:"
echo "  ${OUT_MERGED}"
echo "  ${OUT_STATS}"
