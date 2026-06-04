#!/bin/bash
#SBATCH --job-name=chess-select-build-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm/slurm-%x-%j.out
#SBATCH --error=slurm/slurm-%x-%j.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

# GH200 / Isambard launcher: build restricted-moves ("selection") datasets (v2 mixture).
#
# NOTE: v2 is a legacy mixture builder retained for reproduction/ablation.
# Current workflows typically start from full-legal selection datasets (v4) and apply any iterative
# action-masking online during training (see `iterative.md` and `scripts/build_chess_select_train_dataset_v4.py`).
#
# v2 builder: `scripts/build_chess_select_train_dataset_v2.py`
#
# This job builds all three splits under OUTPUT_DIR:
#   - train.parquet      (from data/chess_puzzles/train.parquet)
#   - train_hard.parquet (from data/chess_puzzles/train_hard.parquet)
#   - test.parquet       (from data/chess_puzzles/test.parquet)
#
# Example (cluster; writes into the repo under data/chess_puzzles_select_v2/):
#   sbatch --wait --export=ALL,SEED=0 ./sbatch_build_chess_select_v2_datasets_gh200.slurm
#
# Optional: enable vLLM-mined negatives for train/train_hard (requires GPU; expensive):
#   sbatch --wait --export=ALL,SEED=0,K_MINE=8,MINE_MODEL=Qwen/Qwen2.5-7B-Instruct,MINE_TOTAL_ROLLOUTS=64 \
#     ./sbatch_build_chess_select_v2_datasets_gh200.slurm

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
echo "[SLURM] Job name:    ${SLURM_JOB_NAME:-chess-select-build-v2}"
echo "[SLURM] Node list:   ${SLURM_JOB_NODELIST:-N/A}"
echo "[SLURM] GPUs:        ${SLURM_GPUS_ON_NODE:-1} (requested via --gres=gpu:1)"
echo "[SLURM] Submit dir:  ${SLURM_SUBMIT_DIR:-$(pwd)}"

# Run relative to the repo root so relative paths work.
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
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-data/chess_puzzles_select_v2}"
TEMPLATE_PATH="${TEMPLATE_PATH:-recipe/chess/prompt_templates/select_prompt.jinja}"

# v2 mixture knobs (train/train_hard)
INCLUDE_FULL_LEGAL="${INCLUDE_FULL_LEGAL:-True}"
INCLUDE_HARD_NEG="${INCLUDE_HARD_NEG:-True}"
INCLUDE_COVERAGE="${INCLUDE_COVERAGE:-True}"
L_MU="${L_MU:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_BLOCKS="${NUM_BLOCKS:-2}"
ENSURE_GROUND_TRUTH_INCLUDED="${ENSURE_GROUND_TRUTH_INCLUDED:-True}"

# Optional mining (expensive; leave disabled by default).
K_MINE="${K_MINE:-0}"
MINE_MODEL="${MINE_MODEL:-}"
MINE_TOTAL_ROLLOUTS="${MINE_TOTAL_ROLLOUTS:-0}"
MINE_TEMPERATURE="${MINE_TEMPERATURE:-0.6}"
MINE_TOP_P="${MINE_TOP_P:-0.95}"
MINE_MAX_PROMPT_TOKENS="${MINE_MAX_PROMPT_TOKENS:-2048}"
MINE_MAX_RESPONSE_TOKENS="${MINE_MAX_RESPONSE_TOKENS:-512}"
MINE_MAX_MODEL_LEN="${MINE_MAX_MODEL_LEN:-4096}"
MINE_GPU_MEMORY_UTILIZATION="${MINE_GPU_MEMORY_UTILIZATION:-0.8}"
MINE_MAX_NUM_SEQS="${MINE_MAX_NUM_SEQS:-1024}"
MINE_BATCH_SIZE="${MINE_BATCH_SIZE:-128}"

echo "[SLURM] SEED=${SEED}"
echo "[SLURM] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[SLURM] TEMPLATE_PATH=${TEMPLATE_PATH}"
echo "[SLURM] INCLUDE_FULL_LEGAL=${INCLUDE_FULL_LEGAL} INCLUDE_HARD_NEG=${INCLUDE_HARD_NEG} INCLUDE_COVERAGE=${INCLUDE_COVERAGE}"
echo "[SLURM] L_MU=${L_MU} BLOCK_SIZE=${BLOCK_SIZE} NUM_BLOCKS=${NUM_BLOCKS} ENSURE_GROUND_TRUTH_INCLUDED=${ENSURE_GROUND_TRUTH_INCLUDED}"
echo "[SLURM] K_MINE=${K_MINE} MINE_MODEL=${MINE_MODEL:-<unset>} MINE_TOTAL_ROLLOUTS=${MINE_TOTAL_ROLLOUTS}"

CMD="set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd \"${CONTAINER_REPO_DIR}\"

mkdir -p \"${OUTPUT_DIR}\"

TRAIN_ARGS=\"--template_path ${TEMPLATE_PATH} --seed ${SEED} --overwrite\"
if [ \"${INCLUDE_FULL_LEGAL}\" = \"True\" ]; then
  TRAIN_ARGS=\"\${TRAIN_ARGS} --include_full_legal\"
else
  TRAIN_ARGS=\"\${TRAIN_ARGS} --no-include_full_legal\"
fi
if [ \"${INCLUDE_HARD_NEG}\" = \"True\" ]; then
  TRAIN_ARGS=\"\${TRAIN_ARGS} --include_hard_neg\"
else
  TRAIN_ARGS=\"\${TRAIN_ARGS} --no-include_hard_neg\"
fi
if [ \"${INCLUDE_COVERAGE}\" = \"True\" ]; then
  TRAIN_ARGS=\"\${TRAIN_ARGS} --include_coverage\"
else
  TRAIN_ARGS=\"\${TRAIN_ARGS} --no-include_coverage\"
fi
TRAIN_ARGS=\"\${TRAIN_ARGS} --L_mu ${L_MU} --block_size ${BLOCK_SIZE} --num_blocks ${NUM_BLOCKS}\"
if [ \"${ENSURE_GROUND_TRUTH_INCLUDED}\" = \"True\" ]; then
  TRAIN_ARGS=\"\${TRAIN_ARGS} --ensure_ground_truth_included\"
else
  TRAIN_ARGS=\"\${TRAIN_ARGS} --no-ensure_ground_truth_included\"
fi

# Optional mining (only if K_MINE>0).
if [ \"${K_MINE}\" -gt 0 ]; then
  if [ -z \"${MINE_MODEL}\" ] || [ \"${MINE_TOTAL_ROLLOUTS}\" -le 0 ]; then
    echo \"[ERROR] K_MINE>0 requires MINE_MODEL and MINE_TOTAL_ROLLOUTS>0\" >&2
    exit 1
  fi
  TRAIN_ARGS=\"\${TRAIN_ARGS} --K_mine ${K_MINE} --mine_model ${MINE_MODEL} --mine_total_rollouts ${MINE_TOTAL_ROLLOUTS}\"
  TRAIN_ARGS=\"\${TRAIN_ARGS} --mine_temperature ${MINE_TEMPERATURE} --mine_top_p ${MINE_TOP_P}\"
  TRAIN_ARGS=\"\${TRAIN_ARGS} --mine_max_prompt_tokens ${MINE_MAX_PROMPT_TOKENS} --mine_max_response_tokens ${MINE_MAX_RESPONSE_TOKENS} --mine_max_model_len ${MINE_MAX_MODEL_LEN}\"
  TRAIN_ARGS=\"\${TRAIN_ARGS} --mine_gpu_memory_utilization ${MINE_GPU_MEMORY_UTILIZATION} --mine_max_num_seqs ${MINE_MAX_NUM_SEQS} --mine_batch_size ${MINE_BATCH_SIZE}\"
fi

echo \"[RUN] Building v2 train...\" 
python3 scripts/build_chess_select_train_dataset_v2.py \\
  --input_parquet data/chess_puzzles/train.parquet \\
  --output_parquet ${OUTPUT_DIR}/train.parquet \\
  \${TRAIN_ARGS} \\
  --sample_jsonl outputs/select_dataset_build_samples/train_select_v2_samples.jsonl \\
  --sample_n 50

echo \"[RUN] Building v2 train_hard...\"
python3 scripts/build_chess_select_train_dataset_v2.py \\
  --input_parquet data/chess_puzzles/train_hard.parquet \\
  --output_parquet ${OUTPUT_DIR}/train_hard.parquet \\
  \${TRAIN_ARGS} \\
  --sample_jsonl outputs/select_dataset_build_samples/train_hard_select_v2_samples.jsonl \\
  --sample_n 50

echo \"[RUN] Building v2 test...\"
python3 scripts/build_chess_select_test_dataset.py \\
  --input_parquet data/chess_puzzles/test.parquet \\
  --output_parquet ${OUTPUT_DIR}/test.parquet \\
  --template_path ${TEMPLATE_PATH} \\
  --seed ${SEED} --overwrite \\
  --sample_jsonl outputs/select_dataset_build_samples/test_select_v2_samples.jsonl \\
  --sample_n 50

echo \"[RUN] Verifying outputs under ${OUTPUT_DIR}...\"
ls -lh \"${OUTPUT_DIR}\" | sed -n '1,120p'
python3 - \"${OUTPUT_DIR}\" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

base = Path(sys.argv[1])
for name in [\"train.parquet\", \"train_hard.parquet\", \"test.parquet\"]:
    path = base / name
    pf = pq.ParquetFile(path)
    print(name, \"rows=\", pf.metadata.num_rows, \"row_groups=\", pf.num_row_groups)
PY

echo \"[RUN] Done.\"
"

echo "[SLURM] Container CMD (truncated): $(echo "${CMD}" | head -c 3000)"

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
