#!/bin/bash
#SBATCH --job-name=chess-passk-dp4-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm/slurm-%x-%j.out
#SBATCH --error=slurm/slurm-%x-%j.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

set -euo pipefail

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
echo "[SLURM] Job name:    ${SLURM_JOB_NAME:-chess-passk-dp4-smoke}"
echo "[SLURM] Node list:   ${SLURM_JOB_NODELIST:-N/A}"
echo "[SLURM] GPUs:        ${SLURM_GPUS_ON_NODE:-4} (requested via --gres=gpu:4)"
echo "[SLURM] Submit dir:  ${SLURM_SUBMIT_DIR:-$(pwd)}"

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
  die "This script targets GH200 (arm64/aarch64) nodes, but uname reports '${HOST_ARCH}'."
fi

SIF_IMAGE="${CHESS_RL_SIF_IMAGE:-${HOME}/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif}"
CONTAINER_REPO_DIR="${CHESS_RL_CONTAINER_REPO_DIR:-/workspace/chess_rl}"
if [ ! -f "${SIF_IMAGE}" ]; then
  die "Missing SIF image: ${SIF_IMAGE}"
fi
echo "[SLURM] SIF_IMAGE=${SIF_IMAGE}"
echo "[SLURM] Repo mount: ${CONTAINER_REPO_DIR}"

PROJECT_ROOT="${CHESS_RL_PROJECT_ROOT:-/projects/a5l/ziyan}"
HF_CACHE_ROOT="${CHESS_RL_HF_CACHE_ROOT:-${PROJECT_ROOT%/}/hf_cache}"
mkdir -p "${HF_CACHE_ROOT}"
export HF_HOME="${HF_CACHE_ROOT%/}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"
echo "[SLURM] HF_HOME=${HF_HOME}"

# Smoke config
MODEL="${MODEL:-/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/82fpo6l0/global_step_781_merged_hf}"
PARQUET="${PARQUET:-data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet}"
LIMIT="${LIMIT:-100}"
K_MAX="${K_MAX:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DP_SHARDS="${DP_SHARDS:-4}"

OUT_JSON="${OUT_JSON:-${PROJECT_ROOT%/}/chess_rl_outputs/passk_eval_runs/dp4_smoke_passk_k8_rows100_${SLURM_JOB_ID:-$$}.json}"
WORK_DIR="${PROJECT_ROOT%/}/chess_rl_outputs/passk_eval_runs/tmp_dp4_smoke_${SLURM_JOB_ID:-$$}"
SHARD_DIR="${WORK_DIR}/shards"
mkdir -p "${WORK_DIR}" "${SHARD_DIR}" "$(dirname "${OUT_JSON}")"

export MODEL PARQUET LIMIT K_MAX TEMPERATURE TOP_P MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_MODEL_LEN
export BATCH_SIZE MAX_NUM_SEQS GPU_MEMORY_UTILIZATION DP_SHARDS OUT_JSON WORK_DIR SHARD_DIR CONTAINER_REPO_DIR

echo "[SLURM] MODEL=${MODEL}"
echo "[SLURM] PARQUET=${PARQUET}"
echo "[SLURM] LIMIT=${LIMIT} K_MAX=${K_MAX} DP_SHARDS=${DP_SHARDS}"
echo "[SLURM] BATCH_SIZE=${BATCH_SIZE} MAX_NUM_SEQS=${MAX_NUM_SEQS} GPU_MEM_UTIL=${GPU_MEMORY_UTILIZATION}"
echo "[SLURM] OUT_JSON=${OUT_JSON}"

CONTAINER_SCRIPT='
set -euo pipefail
cd "${CONTAINER_REPO_DIR}"

python3 - <<'"'"'PY'"'"'
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

parquet = os.environ["PARQUET"]
limit = int(os.environ["LIMIT"])
num_shards = int(os.environ["DP_SHARDS"])
shard_dir = Path(os.environ["SHARD_DIR"])
shard_dir.mkdir(parents=True, exist_ok=True)

dataset = ds.dataset(parquet, format="parquet")
table = dataset.to_table(columns=["prompt", "reward_model", "extra_info"])
rows = table.to_pylist()
if limit > 0:
    rows = rows[:limit]
if not rows:
    raise RuntimeError("No rows loaded for smoke test")

shards = [[] for _ in range(num_shards)]
for i, row in enumerate(rows):
    shards[i % num_shards].append(row)

for i, shard_rows in enumerate(shards):
    out_path = shard_dir / f"shard_{i}.parquet"
    pq.write_table(pa.Table.from_pylist(shard_rows), out_path)
    print(f"[dp4-smoke] shard={i} rows={len(shard_rows)} out={out_path}", flush=True)
PY

for rank in 0 1 2 3; do
  shard_json="${WORK_DIR}/result_shard_${rank}.json"
  CUDA_VISIBLE_DEVICES="${rank}" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    /usr/bin/python3 -m scripts.eval_chess_passk \
      --model "${MODEL}" \
      --parquet "${SHARD_DIR}/shard_${rank}.parquet" \
      --k_max "${K_MAX}" \
      --do_sample \
      --seed 0 \
      --seed_mode engine \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --batch_size "${BATCH_SIZE}" \
      --max_num_seqs "${MAX_NUM_SEQS}" \
      --max_prompt_length "${MAX_PROMPT_LENGTH}" \
      --max_response_length "${MAX_RESPONSE_LENGTH}" \
      --max_model_len "${MAX_MODEL_LEN}" \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --tensor_parallel_size 1 \
      --print_prompt_examples 0 \
      --out_json "${shard_json}" &
done

wait

python3 - <<'"'"'PY'"'"'
import json
import os
from pathlib import Path

work_dir = Path(os.environ["WORK_DIR"])
out_json = Path(os.environ["OUT_JSON"])
shard_paths = [work_dir / f"result_shard_{i}.json" for i in range(int(os.environ["DP_SHARDS"]))]
payloads = []
for p in shard_paths:
    with p.open() as f:
        payloads.append(json.load(f))

num_prompts = [int(p["num_prompts"]) for p in payloads]
total = sum(num_prompts)
if total <= 0:
    raise RuntimeError("No prompts in shard outputs")
weights = [n / total for n in num_prompts]

k = payloads[0]["k"]
for p in payloads[1:]:
    if p["k"] != k:
        raise RuntimeError("Shard k lists differ")

def weighted_mean(vals):
    return float(sum(v * w for v, w in zip(vals, weights)))

curves = {}
for key in payloads[0]["curves"].keys():
    first = payloads[0]["curves"][key]
    if isinstance(first, list):
        curves[key] = [weighted_mean([float(p["curves"][key][i]) for p in payloads]) for i in range(len(first))]
    else:
        curves[key] = weighted_mean([float(p["curves"][key]) for p in payloads])

summary = {}
for key, val in payloads[0]["summary"].items():
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        summary[key] = weighted_mean([float(p["summary"][key]) for p in payloads])
    else:
        summary[key] = val

cfg = dict(payloads[0].get("config") or {})
cfg["data_parallel_size"] = int(os.environ["DP_SHARDS"])
cfg["data_parallel_mode"] = "manual_sharded"
cfg["limit"] = int(os.environ["LIMIT"])

result = {
    "config": cfg,
    "num_prompts": total,
    "k": k,
    "curves": curves,
    "summary": summary,
    "shard_jsons": [str(p) for p in shard_paths],
}

out_json.parent.mkdir(parents=True, exist_ok=True)
with out_json.open("w") as f:
    json.dump(result, f, indent=2)
print(f"[dp4-smoke] wrote merged output: {out_json}", flush=True)
pass_at_k_mean = summary.get("pass_at_k_mean")
print(f"[dp4-smoke] pass_at_k_mean={pass_at_k_mean} num_prompts={total}", flush=True)
PY
'

if command -v srun >/dev/null 2>&1; then
  srun --export=ALL \
    apptainer exec --nv \
      --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
      --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
      --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
      "${SIF_IMAGE}" \
      bash -lc "${CONTAINER_SCRIPT}"
else
  apptainer exec --nv \
    --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
    "${SIF_IMAGE}" \
    bash -lc "${CONTAINER_SCRIPT}"
fi

echo "[SLURM] DP4 smoke completed: ${OUT_JSON}"
