#!/bin/bash
#SBATCH --job-name=chess-select-build-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm/slurm-%x-%j.out
#SBATCH --error=slurm/slurm-%x-%j.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

# GH200 / Isambard launcher: build curriculum selection datasets (v3).
#
# NOTE: v3 is a legacy offline-curriculum builder retained for reproduction/ablation.
# Current workflows typically start from full-legal selection datasets (v4) and apply any iterative
# action-masking online during training (see `iterative.md` and `scripts/build_chess_select_train_dataset_v4.py`).
#
# v3 builder: `scripts/build_chess_select_train_dataset_v3.py`
#
# This job builds two *training* dataset directories (each with train/train_hard/test):
#   A) Full-legal-only curriculum: only full-legal rows (allowed_moves == legal_moves).
#   B) Mixture curriculum: v2-style mixture (full_legal + hard_neg + coverage_block).
#
# Example:
#   sbatch --wait --export=ALL,SEED=0,K_SCHEDULE="6,12,18" ./sbatch_build_chess_select_v3_datasets_gh200.slurm

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
echo "[SLURM] Job name:    ${SLURM_JOB_NAME:-chess-select-build-v3}"
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
TEMPLATE_PATH="${TEMPLATE_PATH:-recipe/chess/prompt_templates/select_prompt.jinja}"
K_SCHEDULE="${K_SCHEDULE:-6,12,18}"
ANNEAL_FRAC="${ANNEAL_FRAC:-1/2}"

# Dataset sizes in "passes" through the base parquet.
# - train_hard is what we usually train on (USE_HARD_DATASET=True).
# - train.parquet is still built for completeness (and to satisfy launcher checks).
TOTAL_PASSES_TRAIN="${TOTAL_PASSES_TRAIN:-5}"
TOTAL_PASSES_TRAIN_HARD="${TOTAL_PASSES_TRAIN_HARD:-5}"

OUTPUT_DIR_FULL="${OUTPUT_DIR_FULL:-data/chess_puzzles_select_v3_fulllegal_curriculum}"
OUTPUT_DIR_MIX="${OUTPUT_DIR_MIX:-data/chess_puzzles_select_v3_mixture_curriculum}"

# Mixture knobs (v2-style).
INCLUDE_HARD_NEG="${INCLUDE_HARD_NEG:-True}"
INCLUDE_COVERAGE="${INCLUDE_COVERAGE:-True}"
L_MU="${L_MU:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_BLOCKS="${NUM_BLOCKS:-2}"
MAX_NEGATIVES_HARD="${MAX_NEGATIVES_HARD:-12}"
ENSURE_GROUND_TRUTH_INCLUDED="${ENSURE_GROUND_TRUTH_INCLUDED:-True}"

echo "[SLURM] SEED=${SEED}"
echo "[SLURM] TEMPLATE_PATH=${TEMPLATE_PATH}"
echo "[SLURM] K_SCHEDULE=${K_SCHEDULE} ANNEAL_FRAC=${ANNEAL_FRAC}"
echo "[SLURM] TOTAL_PASSES_TRAIN=${TOTAL_PASSES_TRAIN} TOTAL_PASSES_TRAIN_HARD=${TOTAL_PASSES_TRAIN_HARD}"
echo "[SLURM] OUTPUT_DIR_FULL=${OUTPUT_DIR_FULL}"
echo "[SLURM] OUTPUT_DIR_MIX=${OUTPUT_DIR_MIX}"
echo "[SLURM] MIXTURE: INCLUDE_HARD_NEG=${INCLUDE_HARD_NEG} INCLUDE_COVERAGE=${INCLUDE_COVERAGE} L_MU=${L_MU} BLOCK_SIZE=${BLOCK_SIZE} NUM_BLOCKS=${NUM_BLOCKS} MAX_NEGATIVES_HARD=${MAX_NEGATIVES_HARD} ENSURE_GROUND_TRUTH_INCLUDED=${ENSURE_GROUND_TRUTH_INCLUDED}"

CMD="set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd \"${CONTAINER_REPO_DIR}\"

mkdir -p \"${OUTPUT_DIR_FULL}\" \"${OUTPUT_DIR_MIX}\" outputs/select_dataset_build_samples

echo \"[RUN] Building v3 (full-legal-only) train...\"
python3 scripts/build_chess_select_train_dataset_v3.py \\
  --input_parquet data/chess_puzzles/train.parquet \\
  --output_parquet ${OUTPUT_DIR_FULL}/train.parquet \\
  --template_path ${TEMPLATE_PATH} \\
  --seed ${SEED} --overwrite \\
  --dataset_flavor full_legal_only \\
  --k_schedule \"${K_SCHEDULE}\" --anneal_frac \"${ANNEAL_FRAC}\" \\
  --total_passes ${TOTAL_PASSES_TRAIN}

echo \"[RUN] Building v3 (full-legal-only) train_hard...\"
python3 scripts/build_chess_select_train_dataset_v3.py \\
  --input_parquet data/chess_puzzles/train_hard.parquet \\
  --output_parquet ${OUTPUT_DIR_FULL}/train_hard.parquet \\
  --template_path ${TEMPLATE_PATH} \\
  --seed ${SEED} --overwrite \\
  --dataset_flavor full_legal_only \\
  --k_schedule \"${K_SCHEDULE}\" --anneal_frac \"${ANNEAL_FRAC}\" \\
  --total_passes ${TOTAL_PASSES_TRAIN_HARD}

echo \"[RUN] Building v3 (full-legal-only) test...\"
python3 scripts/build_chess_select_test_dataset.py \\
  --input_parquet data/chess_puzzles/test.parquet \\
  --output_parquet ${OUTPUT_DIR_FULL}/test.parquet \\
  --template_path ${TEMPLATE_PATH} \\
  --seed ${SEED} --overwrite

echo \"[RUN] Building v3 (mixture) train...\"
# IMPORTANT: do NOT embed quotes into the value of --k_schedule/--anneal_frac.
# We want the shell to handle quoting, not pass literal quote characters through to argparse.
MIX_ARGS=\"--dataset_flavor mixture --k_schedule ${K_SCHEDULE} --anneal_frac ${ANNEAL_FRAC} --total_passes ${TOTAL_PASSES_TRAIN}\"
MIX_ARGS=\"\${MIX_ARGS} --L_mu ${L_MU} --block_size ${BLOCK_SIZE} --num_blocks ${NUM_BLOCKS} --max_negatives_hard ${MAX_NEGATIVES_HARD}\"
if [ \"${INCLUDE_HARD_NEG}\" = \"True\" ]; then
  MIX_ARGS=\"\${MIX_ARGS} --include_hard_neg\"
else
  MIX_ARGS=\"\${MIX_ARGS} --no-include_hard_neg\"
fi
if [ \"${INCLUDE_COVERAGE}\" = \"True\" ]; then
  MIX_ARGS=\"\${MIX_ARGS} --include_coverage\"
else
  MIX_ARGS=\"\${MIX_ARGS} --no-include_coverage\"
fi
if [ \"${ENSURE_GROUND_TRUTH_INCLUDED}\" = \"True\" ]; then
  MIX_ARGS=\"\${MIX_ARGS} --ensure_ground_truth_included\"
else
  MIX_ARGS=\"\${MIX_ARGS} --no-ensure_ground_truth_included\"
fi

python3 scripts/build_chess_select_train_dataset_v3.py \\
  --input_parquet data/chess_puzzles/train.parquet \\
  --output_parquet ${OUTPUT_DIR_MIX}/train.parquet \\
  --template_path ${TEMPLATE_PATH} \\
  --seed ${SEED} --overwrite \\
  \${MIX_ARGS}

echo \"[RUN] Building v3 (mixture) train_hard...\"
MIX_ARGS_HARD=\"\${MIX_ARGS/--total_passes ${TOTAL_PASSES_TRAIN}/--total_passes ${TOTAL_PASSES_TRAIN_HARD}}\"
python3 scripts/build_chess_select_train_dataset_v3.py \\
  --input_parquet data/chess_puzzles/train_hard.parquet \\
  --output_parquet ${OUTPUT_DIR_MIX}/train_hard.parquet \\
  --template_path ${TEMPLATE_PATH} \\
  --seed ${SEED} --overwrite \\
  \${MIX_ARGS_HARD}

echo \"[RUN] Building v3 (mixture) test...\"
python3 scripts/build_chess_select_test_dataset.py \\
  --input_parquet data/chess_puzzles/test.parquet \\
  --output_parquet ${OUTPUT_DIR_MIX}/test.parquet \\
  --template_path ${TEMPLATE_PATH} \\
  --seed ${SEED} --overwrite

echo \"[RUN] Verifying outputs...\"
python3 - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

for base in [Path(\"${OUTPUT_DIR_FULL}\"), Path(\"${OUTPUT_DIR_MIX}\")]:
    print(\"==\", base, \"==\")
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
