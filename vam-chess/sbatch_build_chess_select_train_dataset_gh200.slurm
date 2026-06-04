#!/bin/bash
#SBATCH --job-name=chess-select-build-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm/slurm-%x-%j.out
#SBATCH --error=slurm/slurm-%x-%j.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

# GH200 / Isambard launcher: build a restricted-moves ("selection") *training* parquet
# using rollout-derived considered-move sets (see `scripts/build_chess_select_train_dataset.py`).
#
# This script runs 4 shard workers concurrently (1 per GPU) and then merges shards into a single parquet.
#
# Example (cluster smoke; 20 rows):
#   sbatch --wait --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,LIMIT_ROWS=20,SEED=0 ./sbatch_build_chess_select_train_dataset_gh200.slurm
#
# Example (full train set):
#   sbatch --wait --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,SEED=0 ./sbatch_build_chess_select_train_dataset_gh200.slurm

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
  export CHESS_RL_WANDB_DIR="${CHESS_RL_WANDB_DIR:-${CHESS_RL_PROJECTS_ROOT%/}/wandb}"
  mkdir -p "${CHESS_RL_SINGULARITY_TMPDIR}" "${CHESS_RL_TMPDIR}" "${CHESS_RL_APPTAINER_CACHEDIR}" "${CHESS_RL_WANDB_DIR}" || true
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
if [ -n "${CHESS_RL_WANDB_DIR:-}" ]; then
  export WANDB_DIR="${CHESS_RL_WANDB_DIR}"
fi

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

mkdir -p slurm

echo "[SLURM] Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "[SLURM] Job name:    ${SLURM_JOB_NAME:-chess-select-build-train}"
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

# Cache root (avoid repeated model downloads).
PROJECT_ROOT="${CHESS_RL_PROJECT_ROOT:-/projects/a5l/ziyan}"
HF_CACHE_ROOT="${CHESS_RL_HF_CACHE_ROOT:-${PROJECT_ROOT%/}/hf_cache}"
mkdir -p "${HF_CACHE_ROOT}"
export HF_HOME="${HF_CACHE_ROOT%/}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"
echo "[SLURM] HF_HOME=${HF_HOME}"

# Build config (override via sbatch --export=ALL,...).
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
INPUT_PARQUET="${INPUT_PARQUET:-data/chess_puzzles/train.parquet}"
TEMPLATE_PATH="${TEMPLATE_PATH:-recipe/chess/prompt_templates/select_prompt.jinja}"

NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
SEED="${SEED:-0}"
LIMIT_ROWS="${LIMIT_ROWS:-}"  # optional; empty => full dataset

MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-2048}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-128}"

NUM_SHARDS="${NUM_SHARDS:-4}"
SAMPLE_N="${SAMPLE_N:-50}"

echo "[SLURM] MODEL=${MODEL}"
echo "[SLURM] INPUT_PARQUET=${INPUT_PARQUET}"
echo "[SLURM] TEMPLATE_PATH=${TEMPLATE_PATH}"
echo "[SLURM] SEED=${SEED} NUM_ROLLOUTS=${NUM_ROLLOUTS} LIMIT_ROWS=${LIMIT_ROWS:-<full>}"
echo "[SLURM] NUM_SHARDS=${NUM_SHARDS} BATCH_SIZE=${BATCH_SIZE} MAX_NUM_SEQS=${MAX_NUM_SEQS}"

if [ "${NUM_SHARDS}" != "4" ]; then
  die "NUM_SHARDS must be 4 (matches --gres=gpu:4). Got NUM_SHARDS='${NUM_SHARDS}'."
fi

CMD="set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd \"${CONTAINER_REPO_DIR}\"

TEMPLATE_HASH=\$(sha256sum \"${TEMPLATE_PATH}\" | awk '{print substr(\$1,1,12)}')
MODEL_SAFE=\$(python3 -c \"import re,sys; s=sys.argv[1]; s=s.replace('/','__').replace(':','_'); s=re.sub(r'[^A-Za-z0-9._-]+','_',s); print(s)\" \"${MODEL}\")
INPUT_TAG=\$(python3 -c \"import re,sys,pathlib; p=str(sys.argv[1]); stem=pathlib.Path(p).name; stem=stem[:-8] if stem.endswith('.parquet') else stem; stem=re.sub(r'[^A-Za-z0-9._-]+','_',stem); print(stem)\" \"${INPUT_PARQUET}\")
LIMIT_TAG=\"${LIMIT_ROWS:-full}\"
OUT_DIR=\"${PROJECT_ROOT%/}/chess_rl_outputs/select_dataset_build/train_select_rollouts/\${INPUT_TAG}/\${MODEL_SAFE}__\${TEMPLATE_HASH}_seed${SEED}_rows\${LIMIT_TAG}_n${NUM_ROLLOUTS}\"
mkdir -p \"\${OUT_DIR}\"
echo \"[RUN] OUT_DIR=\${OUT_DIR}\"

COMMON_ARGS=\"--input_parquet ${INPUT_PARQUET} --template_path ${TEMPLATE_PATH} --model ${MODEL} --num_rollouts ${NUM_ROLLOUTS} --seed ${SEED} --max_prompt_tokens ${MAX_PROMPT_TOKENS} --max_response_tokens ${MAX_RESPONSE_TOKENS} --max_model_len ${MAX_MODEL_LEN} --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} --max_num_seqs ${MAX_NUM_SEQS} --batch_size ${BATCH_SIZE} --num_shards ${NUM_SHARDS}\"
if [ -n \"${LIMIT_ROWS}\" ]; then
  COMMON_ARGS=\"\${COMMON_ARGS} --limit_rows ${LIMIT_ROWS}\"
fi

echo \"[RUN] Launching shard workers...\"
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/build_chess_select_train_dataset.py \\
  \${COMMON_ARGS} --shard_idx 0 \\
  --output_parquet \"\${OUT_DIR}/train_shard0of${NUM_SHARDS}.parquet\" --overwrite \\
  --sample_jsonl \"\${OUT_DIR}/train_samples_shard0of${NUM_SHARDS}.jsonl\" --sample_n \"${SAMPLE_N}\" &
P0=\$!
CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 scripts/build_chess_select_train_dataset.py \\
  \${COMMON_ARGS} --shard_idx 1 \\
  --output_parquet \"\${OUT_DIR}/train_shard1of${NUM_SHARDS}.parquet\" --overwrite \\
  --sample_jsonl \"\${OUT_DIR}/train_samples_shard1of${NUM_SHARDS}.jsonl\" --sample_n \"${SAMPLE_N}\" &
P1=\$!
CUDA_VISIBLE_DEVICES=2 /usr/bin/python3 scripts/build_chess_select_train_dataset.py \\
  \${COMMON_ARGS} --shard_idx 2 \\
  --output_parquet \"\${OUT_DIR}/train_shard2of${NUM_SHARDS}.parquet\" --overwrite \\
  --sample_jsonl \"\${OUT_DIR}/train_samples_shard2of${NUM_SHARDS}.jsonl\" --sample_n \"${SAMPLE_N}\" &
P2=\$!
CUDA_VISIBLE_DEVICES=3 /usr/bin/python3 scripts/build_chess_select_train_dataset.py \\
  \${COMMON_ARGS} --shard_idx 3 \\
  --output_parquet \"\${OUT_DIR}/train_shard3of${NUM_SHARDS}.parquet\" --overwrite \\
  --sample_jsonl \"\${OUT_DIR}/train_samples_shard3of${NUM_SHARDS}.jsonl\" --sample_n \"${SAMPLE_N}\" &
P3=\$!

wait \"\${P0}\"
wait \"\${P1}\"
wait \"\${P2}\"
wait \"\${P3}\"

echo \"[RUN] Merging shards -> train.parquet (sorted by extra_info.index)...\"
/usr/bin/python3 - <<'PY' \"\${OUT_DIR}\" \"${NUM_SHARDS}\"
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

out_dir = Path(sys.argv[1])
num_shards = int(sys.argv[2])
paths = [out_dir / f\"train_shard{i}of{num_shards}.parquet\" for i in range(num_shards)]
for p in paths:
    if not p.exists():
        raise SystemExit(f\"Missing shard parquet: {p}\")

rows = []
for p in paths:
    rows.extend(pq.read_table(p).to_pylist())

def _row_id(r):
    ei = r.get(\"extra_info\") or {}
    try:
        return int(ei.get(\"index\"))
    except Exception:
        return 10**18

rows.sort(key=_row_id)
merged = pa.Table.from_pylist(rows)
pq.write_table(merged, out_dir / \"train.parquet\")
print(f\"[MERGE] wrote={(out_dir / 'train.parquet')} rows={merged.num_rows}\")
PY

cat \"\${OUT_DIR}\"/train_samples_shard*of${NUM_SHARDS}.jsonl > \"\${OUT_DIR}/train_samples.jsonl\"

echo \"[RUN] Verifying outputs...\"
ls -lh \"\${OUT_DIR}\" | sed -n '1,120p'
/usr/bin/python3 -c \"import pyarrow.parquet as pq; import sys; t=pq.read_table(sys.argv[1], columns=['extra_info']); print('[VERIFY] rows', t.num_rows)\" \"\${OUT_DIR}/train.parquet\"
echo \"[VERIFY] sample prompts: \${OUT_DIR}/train_samples.jsonl\"
sed -n '1,2p' \"\${OUT_DIR}/train_samples.jsonl\" || true

echo \"[RUN] Done. Artifacts live under: \${OUT_DIR}\"
"

echo "[SLURM] Container CMD: ${CMD}"

if command -v srun >/dev/null 2>&1; then
  srun --export=ALL \
    apptainer exec --nv \
      --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
      --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
      --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
      "${SIF_IMAGE}" \
      bash -c "${CMD}"
else
  apptainer exec --nv \
    --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
    "${SIF_IMAGE}" \
    bash -c "${CMD}"
fi

echo "[SLURM] Done."
