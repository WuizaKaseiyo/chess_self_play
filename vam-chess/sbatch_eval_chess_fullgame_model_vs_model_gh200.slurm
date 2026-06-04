#!/bin/bash
#SBATCH --job-name=chess-fullgame-mvm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --time=0-08:00:00
#SBATCH --output=slurm/slurm-%x-%j.out
#SBATCH --error=slurm/slurm-%x-%j.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

# GH200 / Isambard launcher for a single model-vs-model full-game match:
# - 1 node, 4 GPUs total
# - 2 GPUs per model (two vLLM OpenAI-compatible servers, TP=2 each)
# - Plays a color-balanced match and computes starter-kit-style ACPL (Stockfish depth=20).
#
# Smoke-test example (10 games, A vs B):
#   sbatch --wait --export=ALL,RUN_ID_A=...,RUN_ID_B=...,GAMES_TOTAL=10 ./sbatch_eval_chess_fullgame_model_vs_model_gh200.slurm

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
echo "[SLURM] Job name:    ${SLURM_JOB_NAME:-chess-fullgame-mvm}"
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

SIF_IMAGE="${CHESS_RL_SIF_IMAGE:-/projects/a5l/ziyan/sif-images/chess_rl_v4-arm-stockfish-pytorch-25.05.sif}"
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

# Required match args.
RUN_ID_A="${RUN_ID_A:-}"
RUN_ID_B="${RUN_ID_B:-}"
if [ -z "${RUN_ID_A}" ] || [ -z "${RUN_ID_B}" ]; then
  die "Set RUN_ID_A and RUN_ID_B (keys in checkpoints_by_run.json)."
fi

# Match sizing (smoke default).
GAMES_TOTAL="${GAMES_TOTAL:-10}"
SEED="${SEED:-0}"

# Generation settings (explicit, aligned).
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-2000}"
MAX_RETRIES_PER_TURN="${MAX_RETRIES_PER_TURN:-1}"
MAX_PLIES="${MAX_PLIES:-200}"

# Prompt template (match training contract for selection framing by default).
PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH:-recipe/chess/prompt_templates/select_prompt.jinja}"

# Stockfish / ACPL settings (mirror sbatch_train defaults).
STOCKFISH_PATH="${STOCKFISH_PATH:-/usr/local/bin/stockfish}"
STOCKFISH_HASH_MB="${STOCKFISH_HASH_MB:-128}"
ACPL_DEPTH="${ACPL_DEPTH:-20}"
ACPL_MOVETIME_MS="${ACPL_MOVETIME_MS:-1000}"
ACPL_CP_CAP="${ACPL_CP_CAP:-1000}"
MATE_SCORE_CP="${MATE_SCORE_CP:-1000}"
RESIGNATION_CPL="${RESIGNATION_CPL:-1000}"
ACPL_WORKERS="${ACPL_WORKERS:-72}"
ACPL_THREADS="${ACPL_THREADS:-2}"

# vLLM server settings (2 GPUs/model).
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
SERVER_PORT_A="${SERVER_PORT_A:-8000}"
SERVER_PORT_B="${SERVER_PORT_B:-8100}"
SERVER_GPUS_A="${SERVER_GPUS_A:-0,1}"
SERVER_GPUS_B="${SERVER_GPUS_B:-2,3}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-900}"
BACKEND_MAX_WORKERS="${BACKEND_MAX_WORKERS:-0}"

OUT_ROOT="${OUT_ROOT:-${CHESS_RL_PROJECTS_ROOT%/}/outputs/full_game_eval_mvm}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT%/}/${RUN_ID_A}_vs_${RUN_ID_B}/job_${SLURM_JOB_ID}}"

echo "[SLURM] RUN_ID_A=${RUN_ID_A}"
echo "[SLURM] RUN_ID_B=${RUN_ID_B}"
echo "[SLURM] OUT_DIR=${OUT_DIR}"
echo "[SLURM] GAMES_TOTAL=${GAMES_TOTAL}"
echo "[SLURM] temperature=${TEMPERATURE} top_p=${TOP_P} max_response_tokens=${MAX_RESPONSE_TOKENS} max_retries_per_turn=${MAX_RETRIES_PER_TURN}"
GPUS_PER_MODEL="$(( TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE ))"
echo "[SLURM] tp=${TENSOR_PARALLEL_SIZE} dp=${DATA_PARALLEL_SIZE} (gpus/model=${GPUS_PER_MODEL}) server_gpus_a=${SERVER_GPUS_A} server_gpus_b=${SERVER_GPUS_B}"
echo "[SLURM] stockfish_path=${STOCKFISH_PATH} acpl_workers=${ACPL_WORKERS} acpl_threads=${ACPL_THREADS}"

CMD="
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
cd ${CONTAINER_REPO_DIR}
mkdir -p \"${OUT_DIR}\"
python3 scripts/eval_chess_fullgame_model_vs_model.py \\
  --run-id-a \"${RUN_ID_A}\" --run-id-b \"${RUN_ID_B}\" \\
  --out-dir \"${OUT_DIR}\" \\
  --prompt-template-path \"${PROMPT_TEMPLATE_PATH}\" \\
  --games-total \"${GAMES_TOTAL}\" \\
  --seed \"${SEED}\" \\
  --temperature \"${TEMPERATURE}\" --top-p \"${TOP_P}\" \\
  --max-response-tokens \"${MAX_RESPONSE_TOKENS}\" \\
  --max-retries-per-turn \"${MAX_RETRIES_PER_TURN}\" \\
  --max-plies \"${MAX_PLIES}\" \\
  --stockfish-path \"${STOCKFISH_PATH}\" \\
  --stockfish-hash-mb \"${STOCKFISH_HASH_MB}\" \\
  --acpl-depth \"${ACPL_DEPTH}\" \\
  --acpl-movetime-ms \"${ACPL_MOVETIME_MS}\" \\
  --acpl-cp-cap \"${ACPL_CP_CAP}\" \\
  --mate-score-cp \"${MATE_SCORE_CP}\" \\
  --resignation-cpl \"${RESIGNATION_CPL}\" \\
  --acpl-workers \"${ACPL_WORKERS}\" \\
  --acpl-threads \"${ACPL_THREADS}\" \\
  --tensor-parallel-size \"${TENSOR_PARALLEL_SIZE}\" \\
  --data-parallel-size \"${DATA_PARALLEL_SIZE}\" \\
  --gpu-memory-utilization \"${GPU_MEMORY_UTILIZATION}\" \\
  --max-model-len \"${MAX_MODEL_LEN}\" \\
  --server-port-a \"${SERVER_PORT_A}\" --server-port-b \"${SERVER_PORT_B}\" \\
  --server-gpus-a \"${SERVER_GPUS_A}\" --server-gpus-b \"${SERVER_GPUS_B}\" \\
  --server-ready-timeout-s \"${SERVER_READY_TIMEOUT_S}\" \\
  --backend-max-workers \"${BACKEND_MAX_WORKERS}\" \\
  --trust-remote-code
echo \"[RUN] Done. Outputs under: ${OUT_DIR}\"
ls -la \"${OUT_DIR}\" || true
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
