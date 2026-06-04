#!/bin/bash
#SBATCH --job-name=chess-rr-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm/slurm-%x-%j.out
#SBATCH --error=slurm/slurm-%x-%j.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

# Round-robin pilot (inference stage):
# - Serve 4 models (1 GPU each) on a single GH200 node.
# - Play all 6 unordered pairs concurrently, `GAMES_PER_PAIR` per pair.
# - Writes PGN + per-move JSONL traces; does NOT compute ACPL (stage 2 does).
#
# Example:
#   sbatch --wait --export=ALL,RUN_ID_0=5cw6hm16,RUN_ID_1=82fpo6l0,RUN_ID_2=azs0jkjg,RUN_ID_3=dg41tlmo,GAMES_PER_PAIR=100 \
#     ./sbatch_eval_chess_fullgame_round_robin_pilot_infer_gh200.slurm

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
echo "[SLURM] Job name:    ${SLURM_JOB_NAME:-chess-rr-infer}"
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

# Required pilot args (commas are not allowed in sbatch --export values; use 4 vars).
RUN_ID_0="${RUN_ID_0:-}"
RUN_ID_1="${RUN_ID_1:-}"
RUN_ID_2="${RUN_ID_2:-}"
RUN_ID_3="${RUN_ID_3:-}"
if [ -z "${RUN_ID_0}" ] || [ -z "${RUN_ID_1}" ] || [ -z "${RUN_ID_2}" ] || [ -z "${RUN_ID_3}" ]; then
  die "Set RUN_ID_0..RUN_ID_3 (4 run ids; keys in checkpoints_by_run.json)."
fi
RUN_IDS="${RUN_ID_0},${RUN_ID_1},${RUN_ID_2},${RUN_ID_3}"

GAMES_PER_PAIR="${GAMES_PER_PAIR:-100}"
SEED="${SEED:-0}"

# Generation settings (explicit, aligned).
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-2000}"
MAX_RETRIES_PER_TURN="${MAX_RETRIES_PER_TURN:-1}"
MAX_PLIES="${MAX_PLIES:-200}"

PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH:-recipe/chess/prompt_templates/select_prompt.jinja}"

# vLLM server settings.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-8000}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-900}"
BACKEND_MAX_WORKERS="${BACKEND_MAX_WORKERS:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-0}"

OUT_ROOT="${OUT_ROOT:-${CHESS_RL_PROJECTS_ROOT%/}/outputs/full_game_eval_rr_pilot}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT%/}/job_${SLURM_JOB_ID}}"

echo "[SLURM] RUN_IDS=${RUN_IDS}"
echo "[SLURM] OUT_DIR=${OUT_DIR}"
echo "[SLURM] GAMES_PER_PAIR=${GAMES_PER_PAIR}"
echo "[SLURM] temperature=${TEMPERATURE} top_p=${TOP_P} max_response_tokens=${MAX_RESPONSE_TOKENS} max_retries_per_turn=${MAX_RETRIES_PER_TURN}"
echo "[SLURM] max_plies=${MAX_PLIES} prompt_template=${PROMPT_TEMPLATE_PATH}"
echo "[SLURM] enforce_eager=${ENFORCE_EAGER}"
echo "[SLURM] VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE}"

ENFORCE_EAGER_FLAG=""
if [ "${ENFORCE_EAGER}" = "1" ]; then
  ENFORCE_EAGER_FLAG="--enforce-eager"
fi

CMD="
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export VLLM_DISABLE_COMPILE_CACHE=\"${VLLM_DISABLE_COMPILE_CACHE}\"
cd ${CONTAINER_REPO_DIR}
mkdir -p \"${OUT_DIR}\"
python3 scripts/eval_chess_fullgame_round_robin_infer.py \\
  --checkpoints-json checkpoints_by_run.json \\
  --run-ids \"${RUN_IDS}\" \\
  --out-dir \"${OUT_DIR}\" \\
  --games-per-pair \"${GAMES_PER_PAIR}\" \\
  --seed \"${SEED}\" \\
  --temperature \"${TEMPERATURE}\" --top-p \"${TOP_P}\" \\
  --max-response-tokens \"${MAX_RESPONSE_TOKENS}\" \\
  --max-retries-per-turn \"${MAX_RETRIES_PER_TURN}\" \\
  --max-plies \"${MAX_PLIES}\" \\
  --prompt-template-path \"${PROMPT_TEMPLATE_PATH}\" \\
  --gpu-memory-utilization \"${GPU_MEMORY_UTILIZATION}\" \\
  --max-model-len \"${MAX_MODEL_LEN}\" \\
  --server-port-base \"${SERVER_PORT_BASE}\" \\
  --server-ready-timeout-s \"${SERVER_READY_TIMEOUT_S}\" \\
  --backend-max-workers \"${BACKEND_MAX_WORKERS}\" \\
  ${ENFORCE_EAGER_FLAG} \\
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
