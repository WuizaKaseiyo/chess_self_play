#!/bin/bash
#SBATCH --job-name=math-explored-paths
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=0-12:00:00
#SBATCH --output=/projects/a5l/ziyan/chess_rl/logs/slurm-%x-%j.out
#SBATCH --error=/projects/a5l/ziyan/chess_rl/logs/slurm-%x-%j.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

set -euo pipefail

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

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

echo "[SLURM] Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "[SLURM] Job name:    ${SLURM_JOB_NAME:-math-explored-paths}"
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
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"
echo "[SLURM] HF_HOME=${HF_HOME}"

HF_TOKEN_FILE="${HF_TOKEN_FILE:-${CHESS_RL_HF_TOKEN_FILE:-${HOME}/.hf_token_zzc}}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_TOKEN_FILE}" ]; then
  export HF_TOKEN="$(tr -d ' \n\r\t' <"${HF_TOKEN_FILE}")"
fi
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
fi

MODEL_PATH="${MODEL_PATH:-/projects/a5l/ziyan/models/Qwen/Qwen3-1.7B}"
DATASETS="${DATASETS:-aime24 aime25 amc23 math500 minerva olympiadbench}"
SOLVE_MAX_TOKENS="${SOLVE_MAX_TOKENS:-8192 32768}"
ATTEMPTS="${ATTEMPTS:-8}"
METHODS="${METHODS:-baseline iterative}"
LIMIT="${LIMIT:-${LIMIT_PER_DATASET:-}}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.7}"
SUMMARY_MAX_TOKENS="${SUMMARY_MAX_TOKENS:-512}"
SUMMARY_TEMPERATURE="${SUMMARY_TEMPERATURE:-0.0}"
SUMMARY_TOP_P="${SUMMARY_TOP_P:-1.0}"
BASE_SEED="${BASE_SEED:-3407}"
CONCURRENCY="${CONCURRENCY:-16}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_SHARDS="${NUM_SHARDS:-${GPUS_PER_NODE}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
SERVER_PROMPT_LENGTH="${SERVER_PROMPT_LENGTH:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT%/}/chess_rl_outputs/math_explored_paths}"
RUN_TAG="${RUN_TAG:-qwen3_1p7b_${SLURM_JOB_ID:-$$}}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT%/}/${RUN_TAG}}"
mkdir -p "${OUT_DIR}"
export CONTAINER_REPO_DIR OUT_DIR LIMIT DATASETS SOLVE_MAX_TOKENS METHODS
export MODEL_PATH ATTEMPTS TEMPERATURE TOP_P SUMMARY_MAX_TOKENS SUMMARY_TEMPERATURE SUMMARY_TOP_P
export BASE_SEED CONCURRENCY GPUS_PER_NODE NUM_SHARDS TENSOR_PARALLEL_SIZE
export GPU_MEMORY_UTILIZATION MAX_MODEL_LEN SERVER_PROMPT_LENGTH MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS

if [ "${TENSOR_PARALLEL_SIZE}" != "1" ]; then
  die "This launcher only supports one-GPU direct-vLLM workers; set TENSOR_PARALLEL_SIZE=1."
fi

echo "[SLURM] MODEL_PATH=${MODEL_PATH}"
echo "[SLURM] DATASETS=${DATASETS}"
echo "[SLURM] SOLVE_MAX_TOKENS=${SOLVE_MAX_TOKENS}"
echo "[SLURM] ATTEMPTS=${ATTEMPTS} METHODS=${METHODS} LIMIT=${LIMIT:-<unset>}"
echo "[SLURM] GPUS_PER_NODE=${GPUS_PER_NODE} NUM_SHARDS=${NUM_SHARDS} TP=${TENSOR_PARALLEL_SIZE} CONCURRENCY=${CONCURRENCY}"
echo "[SLURM] OUT_DIR=${OUT_DIR}"

CONTAINER_SCRIPT='
set -euo pipefail
cd "${CONTAINER_REPO_DIR}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH

DATASET_ARGS=()
for item in ${DATASETS}; do
  DATASET_ARGS+=("${item}")
done

SOLVE_ARGS=()
for item in ${SOLVE_MAX_TOKENS}; do
  SOLVE_ARGS+=("${item}")
done

METHOD_ARGS=()
for item in ${METHODS}; do
  METHOD_ARGS+=("${item}")
done

LOCAL_GPU_COUNT="${SLURM_GPUS_ON_NODE:-${GPUS_PER_NODE}}"
if [ "${NUM_SHARDS}" -lt 1 ]; then
  echo "[ERROR] NUM_SHARDS must be >= 1, got ${NUM_SHARDS}" >&2
  exit 1
fi
if [ "${NUM_SHARDS}" -gt "${LOCAL_GPU_COUNT}" ]; then
  echo "[ERROR] NUM_SHARDS=${NUM_SHARDS} exceeds visible GPUs on node (${LOCAL_GPU_COUNT})." >&2
  exit 1
fi

COMMON_CMD=(
  --model "${MODEL_PATH}"
  --datasets "${DATASET_ARGS[@]}"
  --attempts "${ATTEMPTS}"
  --solve-max-tokens "${SOLVE_ARGS[@]}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --summary-max-tokens "${SUMMARY_MAX_TOKENS}"
  --summary-temperature "${SUMMARY_TEMPERATURE}"
  --summary-top-p "${SUMMARY_TOP_P}"
  --base-seed "${BASE_SEED}"
  --concurrency "${CONCURRENCY}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --server-prompt-length "${SERVER_PROMPT_LENGTH}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --methods "${METHOD_ARGS[@]}"
)
if [ -n "${LIMIT}" ]; then
  COMMON_CMD+=(--limit "${LIMIT}")
fi

echo "[RUN] Launching ${NUM_SHARDS} shard worker(s) under ${OUT_DIR}/shards"
PIDS=()
for gpu_idx in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_idx="${gpu_idx}"
  printf -v shard_name "shard%02dof%02d" "${shard_idx}" "${NUM_SHARDS}"
  shard_out_dir="${OUT_DIR}/shards/${shard_name}"
  mkdir -p "${shard_out_dir}"
  CMD=(
    /usr/bin/python3 -m scripts.eval_math_explored_paths_vllm
    "${COMMON_CMD[@]}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpus-per-node 1
    --num-shards "${NUM_SHARDS}"
    --shard-idx "${shard_idx}"
    --out-dir "${shard_out_dir}"
  )
  echo "[RUN] Shard ${shard_idx}/${NUM_SHARDS} -> ${shard_out_dir} (CUDA_VISIBLE_DEVICES=${gpu_idx})"
  printf "  %q" "${CMD[@]}"
  printf "\n"
  CUDA_VISIBLE_DEVICES="${gpu_idx}" "${CMD[@]}" &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

MERGE_CMD=(
  /usr/bin/python3 -m scripts.eval_math_explored_paths_vllm
  "${COMMON_CMD[@]}"
  --num-shards "${NUM_SHARDS}"
  --merge-only
  --out-dir "${OUT_DIR}"
)
echo "[RUN] Merging shard outputs into ${OUT_DIR}"
printf "  %q" "${MERGE_CMD[@]}"
printf "\n"
CUDA_VISIBLE_DEVICES="" "${MERGE_CMD[@]}"
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

echo "[SLURM] Done: ${OUT_DIR}"
