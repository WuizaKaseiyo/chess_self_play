#!/bin/bash
#SBATCH --job-name=chess-sglang-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

warn() {
  echo "[WARN] $*" >&2
}

detect_cluster() {
  local requested host cluster_name
  requested="$(printf '%s' "${CHESS_RL_CLUSTER:-auto}" | tr '[:upper:]' '[:lower:]')"
  if [ "${requested}" != "auto" ] && [ -n "${requested}" ]; then
    printf '%s\n' "${requested}"
    return
  fi

  cluster_name="$(printf '%s' "${SLURM_CLUSTER_NAME:-}" | tr '[:upper:]' '[:lower:]')"
  case "${cluster_name}" in
    *isambard*|*aip*) printf '%s\n' "isambard"; return ;;
    *delta*) printf '%s\n' "deltaai"; return ;;
  esac

  host="$(hostname -f 2>/dev/null || hostname 2>/dev/null || true)"
  case "${host}" in
    *isambard*|*aip2*|login[0-9]*|nid[0-9]*) printf '%s\n' "isambard"; return ;;
    *delta*|*ghx*) printf '%s\n' "deltaai"; return ;;
  esac

  if [ -d /projects/a5l/ziyan ]; then
    printf '%s\n' "isambard"
  else
    # Preserve the launcher as DeltaAI-oriented when auto-detection has no signal.
    printf '%s\n' "deltaai"
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_JOB_ID="${SLURM_JOB_ID:-manual-$$}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  JOB_ID="${SLURM_ARRAY_JOB_ID:-${BASE_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
else
  JOB_ID="${BASE_JOB_ID}"
fi
CLUSTER="$(detect_cluster)"

case "${CLUSTER}" in
  deltaai)
    DEFAULT_REPO_DIR="/projects/bgba/zzhang69/code/chess-rl-wallclock-debug"
    DEFAULT_PROJECTS_ROOT="/projects/bgba/zzhang69/chess_rl_wallclock_diag"
    DEFAULT_SIF_IMAGE="/projects/bgba/zzhang69/sif-images/gabr1e1_verl_sglang059.sif"
    DEFAULT_PY_SITE="/projects/bgba/zzhang69/chess_rl_wallclock_diag/python_site_sglang"
    DEFAULT_MODEL_PATH="/projects/bgba/zzhang69/models/Qwen/Qwen3-4B-Base"
    DEFAULT_TMPDIR="/tmp/chess-sglang-${JOB_ID}"
    ;;
  isambard)
    DEFAULT_REPO_DIR="/projects/a5l/ziyan/chess-rl"
    DEFAULT_PROJECTS_ROOT="/projects/a5l/ziyan/chess_rl"
    DEFAULT_SIF_IMAGE="/projects/a5l/ziyan/sif-images/verl_sglang059_isambard.sif"
    DEFAULT_PY_SITE="/projects/a5l/ziyan/chess_rl/python_site_sglang"
    DEFAULT_MODEL_PATH="/projects/a5l/ziyan/models/Qwen/Qwen3-4B-Base"
    DEFAULT_TMPDIR=""
    ;;
  *)
    die "Unknown CHESS_RL_CLUSTER='${CLUSTER}'. Use 'deltaai', 'isambard', or leave it unset for auto."
    ;;
esac

REPO_DIR="${CHESS_RL_REPO_DIR:-${REPO_DIR:-${DEFAULT_REPO_DIR}}}"
if [ -d "${REPO_DIR}" ]; then
  REPO_DIR="$(cd "${REPO_DIR}" && pwd)"
fi
case "${REPO_DIR}" in
  /lus/lfs1aip2/projects/*) REPO_DIR="/projects/${REPO_DIR#/lus/lfs1aip2/projects/}" ;;
esac
CHESS_RL_PROJECTS_ROOT="${CHESS_RL_PROJECTS_ROOT:-${DEFAULT_PROJECTS_ROOT}}"
if [ -z "${DEFAULT_TMPDIR}" ]; then
  if [ "${CLUSTER}" = "isambard" ]; then
    DEFAULT_TMPDIR="${CHESS_RL_SHORT_TMP_ROOT:-$(dirname "${CHESS_RL_PROJECTS_ROOT%/}")/crl}/${JOB_ID}"
  else
    DEFAULT_TMPDIR="${CHESS_RL_PROJECTS_ROOT%/}/tmp/chess-sglang-${JOB_ID}"
  fi
fi
SIF_IMAGE="${CHESS_RL_SIF_IMAGE:-${SIF_IMAGE:-${DEFAULT_SIF_IMAGE}}}"
RUN_ROOT="${CHESS_RL_RUN_ROOT:-${RUN_ROOT:-${CHESS_RL_PROJECTS_ROOT%/}/runs/${JOB_ID}}}"
PY_SITE="${CHESS_RL_PY_SITE:-${PY_SITE:-${DEFAULT_PY_SITE}}}"

mkdir -p "${RUN_ROOT}" "${CHESS_RL_PROJECTS_ROOT%/}/logs"

if [ -n "${CHESS_RL_TMPDIR:-}" ]; then
  TMPDIR_CANDIDATE="${CHESS_RL_TMPDIR}"
elif [ "${CLUSTER}" = "isambard" ]; then
  # Isambard may provide a host-local TMPDIR under /local, but this path is not
  # guaranteed to be mounted inside the Apptainer container. Keep Ray tmp under
  # /projects by default so the container can create it.
  TMPDIR_CANDIDATE="${DEFAULT_TMPDIR}"
else
  TMPDIR_CANDIDATE="${TMPDIR:-${DEFAULT_TMPDIR}}"
fi
if ! mkdir -p "${TMPDIR_CANDIDATE}" 2>/dev/null || [ ! -w "${TMPDIR_CANDIDATE}" ]; then
  warn "TMPDIR='${TMPDIR_CANDIDATE}' is not writable; falling back to /tmp/chess-sglang-${JOB_ID}."
  TMPDIR_CANDIDATE="/tmp/chess-sglang-${JOB_ID}"
  mkdir -p "${TMPDIR_CANDIDATE}"
fi
export TMPDIR="${TMPDIR_CANDIDATE}"
export RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR}/ray}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${CHESS_RL_SINGULARITY_TMPDIR:-${TMPDIR}/apptainer_tmp}}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${CHESS_RL_APPTAINER_CACHEDIR:-${TMPDIR}/apptainer_cache}}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${APPTAINER_TMPDIR}}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${APPTAINER_CACHEDIR}}"
mkdir -p "${RAY_TMPDIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}" "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"

if [ -z "${APPTAINER_BINDPATH:-}" ]; then
  APPTAINER_BINDPATH="${CHESS_RL_APPTAINER_BINDPATH:-/projects:/projects}"
elif [ -n "${CHESS_RL_APPTAINER_BINDPATH:-}" ]; then
  APPTAINER_BINDPATH="${CHESS_RL_APPTAINER_BINDPATH}"
fi
export APPTAINER_BINDPATH

if [ -n "${PY_SITE}" ]; then
  export PYTHONPATH="${PY_SITE}:${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# SGLang/TVM JIT compilation happens inside the Apptainer image. DeltaAI
# login shells may export Cray-style wrappers such as CXX=CC, which are not
# present in this image, so use explicit compiler paths from the container.
export CC="${CHESS_RL_CONTAINER_CC:-/usr/bin/gcc}"
export CXX="${CHESS_RL_CONTAINER_CXX:-/usr/bin/g++}"

export WANDB_MODE="${CHESS_RL_WANDB_MODE:-${WANDB_MODE:-offline}}"
export WANDB_DIR="${CHESS_RL_WANDB_DIR:-${WANDB_DIR:-${RUN_ROOT}/wandb}}"
export VERL_BASE_DIR="${CHESS_RL_VERL_BASE_DIR:-${VERL_BASE_DIR:-${RUN_ROOT}/verl}}"
mkdir -p "${WANDB_DIR}" "${VERL_BASE_DIR}"

export PROJECT_DIR="${REPO_DIR}"
export MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
export CHESS_DATA_DIR="${CHESS_DATA_DIR:-${REPO_DIR}/data/chess_puzzles_chessr1_aligned_sharded_ours}"
export CHESS_RL_VAL_FILES="${CHESS_RL_VAL_FILES:-${REPO_DIR}/data/chess_puzzles_chessr1_aligned_sharded_ours/test.parquet}"

export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
export TRAINER_NNODES="${TRAINER_NNODES:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
export ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-8}"
export REF_LOGPROB_MICRO_BATCH_SIZE="${REF_LOGPROB_MICRO_BATCH_SIZE:-8}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-5632}"
export TOTAL_TRAINING_STEPS="${CHESS_RL_TOTAL_TRAINING_STEPS:-${TOTAL_TRAINING_STEPS:-1}}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:--1}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:--1}"
export TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
export FULL_EVAL_FREQ="${FULL_EVAL_FREQ:--1}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-0}"
export USE_CHAT_TEMPLATE="${CHESS_RL_USE_CHAT_TEMPLATE:-${USE_CHAT_TEMPLATE:-False}}"
export RESPONSE_PREFIX_ENABLE="${CHESS_RL_RESPONSE_PREFIX_ENABLE:-${RESPONSE_PREFIX_ENABLE:-False}}"
export RESPONSE_PREFIX="${CHESS_RL_RESPONSE_PREFIX:-${RESPONSE_PREFIX:-}}"
export CHESS_RL_RESPONSE_PREFIX_ENABLE="${RESPONSE_PREFIX_ENABLE}"
export CHESS_RL_RESPONSE_PREFIX="${RESPONSE_PREFIX}"

export ALLOWED_MOVE_ELIM_ENABLE="${ALLOWED_MOVE_ELIM_ENABLE:-True}"
export ALLOWED_MOVE_ELIM_UID_MODE="${ALLOWED_MOVE_ELIM_UID_MODE:-per_prompt}"
export ALLOWED_MOVE_ELIM_NO_SUCCESS_POLICY="${ALLOWED_MOVE_ELIM_NO_SUCCESS_POLICY:-accept_last}"
export ALLOWED_MOVE_ELIM_R_MAX_START="${ALLOWED_MOVE_ELIM_R_MAX_START:-4}"
export ALLOWED_MOVE_ELIM_R_MAX_END="${ALLOWED_MOVE_ELIM_R_MAX_END:-4}"
export ALLOWED_MOVE_ELIM_ANNEAL_FRAC="${ALLOWED_MOVE_ELIM_ANNEAL_FRAC:-0.0}"
export ALLOWED_MOVE_ELIM_GAIN_THRESHOLD="${ALLOWED_MOVE_ELIM_GAIN_THRESHOLD:-inf}"
export FORCED_PREFIX_ENABLE="${FORCED_PREFIX_ENABLE:-False}"

export CHESS_RL_ROLLOUT_NAME="${CHESS_RL_ROLLOUT_NAME:-sglang}"
export ROLLOUT_NAME="${ROLLOUT_NAME:-${CHESS_RL_ROLLOUT_NAME}}"
export FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}"
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"

SMOKE_VARIANT="${CHESS_RL_SMOKE_VARIANT:-${SMOKE_VARIANT:-}}"
if [ "${SMOKE_VARIANT}" = "array" ]; then
  case "${SLURM_ARRAY_TASK_ID:-}" in
    0) SMOKE_VARIANT="actoff" ;;
    1) SMOKE_VARIANT="nooff" ;;
    *) die "CHESS_RL_SMOKE_VARIANT=array requires SLURM_ARRAY_TASK_ID 0 or 1; got '${SLURM_ARRAY_TASK_ID:-unset}'." ;;
  esac
fi

case "${SMOKE_VARIANT}" in
  ""|"manual")
    ;;
  "actoff")
    export CHESS_RL_MODEL_ENABLE_ACTIVATION_OFFLOAD=False
    export MODEL_ENABLE_ACTIVATION_OFFLOAD=False
    export CHESS_RL_ACTOR_PARAM_OFFLOAD=False
    export ACTOR_PARAM_OFFLOAD=False
    export CHESS_RL_ACTOR_OPTIMIZER_OFFLOAD=False
    export ACTOR_OPTIMIZER_OFFLOAD=False
    export CHESS_RL_ACTOR_OFFLOAD_POLICY=False
    export ACTOR_OFFLOAD_POLICY=False
    export CHESS_RL_REF_PARAM_OFFLOAD=True
    export REF_PARAM_OFFLOAD=True
    export CHESS_RL_REF_OPTIMIZER_OFFLOAD=False
    export REF_OPTIMIZER_OFFLOAD=False
    export CHESS_RL_REF_OFFLOAD_POLICY=True
    export REF_OFFLOAD_POLICY=True
    ;;
  "nooff")
    export CHESS_RL_MODEL_ENABLE_ACTIVATION_OFFLOAD=False
    export MODEL_ENABLE_ACTIVATION_OFFLOAD=False
    export CHESS_RL_ACTOR_PARAM_OFFLOAD=False
    export ACTOR_PARAM_OFFLOAD=False
    export CHESS_RL_ACTOR_OPTIMIZER_OFFLOAD=False
    export ACTOR_OPTIMIZER_OFFLOAD=False
    export CHESS_RL_ACTOR_OFFLOAD_POLICY=False
    export ACTOR_OFFLOAD_POLICY=False
    export CHESS_RL_REF_PARAM_OFFLOAD=False
    export REF_PARAM_OFFLOAD=False
    export CHESS_RL_REF_OPTIMIZER_OFFLOAD=False
    export REF_OPTIMIZER_OFFLOAD=False
    export CHESS_RL_REF_OFFLOAD_POLICY=False
    export REF_OFFLOAD_POLICY=False
    export PARAM_OFFLOAD=False
    export OPTIMIZER_OFFLOAD=False
    ;;
  *)
    die "Unknown CHESS_RL_SMOKE_VARIANT='${SMOKE_VARIANT}'. Use 'actoff', 'nooff', 'array', or leave unset."
    ;;
esac

export CHESS_RL_MODEL_ENABLE_ACTIVATION_OFFLOAD="${CHESS_RL_MODEL_ENABLE_ACTIVATION_OFFLOAD:-${MODEL_ENABLE_ACTIVATION_OFFLOAD:-True}}"
export MODEL_ENABLE_ACTIVATION_OFFLOAD="${MODEL_ENABLE_ACTIVATION_OFFLOAD:-${CHESS_RL_MODEL_ENABLE_ACTIVATION_OFFLOAD}}"

export ACTOR_PARAM_OFFLOAD="${CHESS_RL_ACTOR_PARAM_OFFLOAD:-${ACTOR_PARAM_OFFLOAD:-True}}"
export ACTOR_OPTIMIZER_OFFLOAD="${CHESS_RL_ACTOR_OPTIMIZER_OFFLOAD:-${ACTOR_OPTIMIZER_OFFLOAD:-False}}"
export ACTOR_OFFLOAD_POLICY="${CHESS_RL_ACTOR_OFFLOAD_POLICY:-${ACTOR_OFFLOAD_POLICY:-False}}"
export REF_PARAM_OFFLOAD="${CHESS_RL_REF_PARAM_OFFLOAD:-${REF_PARAM_OFFLOAD:-True}}"
export REF_OPTIMIZER_OFFLOAD="${CHESS_RL_REF_OPTIMIZER_OFFLOAD:-${REF_OPTIMIZER_OFFLOAD:-False}}"
export REF_OFFLOAD_POLICY="${CHESS_RL_REF_OFFLOAD_POLICY:-${REF_OFFLOAD_POLICY:-True}}"

export VLLM_GPU_MEMORY_UTILIZATION="${CHESS_RL_VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION:-0.7}}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2048}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
export FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"

APPTAINER_BIN="${APPTAINER_BIN:-}"
if [ -z "${APPTAINER_BIN}" ]; then
  if command -v apptainer >/dev/null 2>&1; then
    APPTAINER_BIN="apptainer"
  elif command -v singularity >/dev/null 2>&1; then
    APPTAINER_BIN="singularity"
  fi
fi
[ -n "${APPTAINER_BIN}" ] || die "Could not find apptainer or singularity on PATH."
[ -f "${SIF_IMAGE}" ] || die "SIF image not found: ${SIF_IMAGE}. Override with CHESS_RL_SIF_IMAGE or SIF_IMAGE."
[ -f "${REPO_DIR}/train_chess.sh" ] || die "Missing train_chess.sh under REPO_DIR='${REPO_DIR}'. Override with CHESS_RL_REPO_DIR or REPO_DIR."

echo "[SLURM] job_id=${SLURM_JOB_ID:-N/A}"
echo "[SLURM] node=${SLURM_NODELIST:-unknown}"
echo "[SLURM] cluster=${CLUSTER}"
echo "[SLURM] smoke_variant=${SMOKE_VARIANT:-manual}"
echo "[SLURM] repo=${REPO_DIR}"
echo "[SLURM] run_root=${RUN_ROOT}"
echo "[SLURM] tmpdir=${TMPDIR}"
echo "[SLURM] image=${SIF_IMAGE}"
echo "[SLURM] model=${MODEL_PATH}"
echo "[SLURM] apptainer=${APPTAINER_BIN}"
echo "[SLURM] apptainer_bindpath=${APPTAINER_BINDPATH}"
echo "[SLURM] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[SLURM] rollout=${ROLLOUT_NAME}"
echo "[SLURM] fsdp_strategy=${FSDP_STRATEGY}"
echo "[SLURM] enforce_eager=${ENFORCE_EAGER}"
echo "[SLURM] gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "[SLURM] model_activation_offload=${MODEL_ENABLE_ACTIVATION_OFFLOAD}"
echo "[SLURM] actor_param_offload=${ACTOR_PARAM_OFFLOAD}"
echo "[SLURM] actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
echo "[SLURM] actor_offload_policy=${ACTOR_OFFLOAD_POLICY}"
echo "[SLURM] ref_param_offload=${REF_PARAM_OFFLOAD}"
echo "[SLURM] ref_optimizer_offload=${REF_OPTIMIZER_OFFLOAD}"
echo "[SLURM] ref_offload_policy=${REF_OFFLOAD_POLICY}"

"${APPTAINER_BIN}" exec --nv --bind "${APPTAINER_BINDPATH}" "${SIF_IMAGE}" \
  bash -lc 'python3 -c "import torch, sglang, chess; print(\"preflight ok\", torch.__version__, sglang.__version__, chess.__version__)"'

"${APPTAINER_BIN}" exec --nv --bind "${APPTAINER_BINDPATH}" "${SIF_IMAGE}" \
  bash -lc 'cd "${PROJECT_DIR}" && ./train_chess.sh'
