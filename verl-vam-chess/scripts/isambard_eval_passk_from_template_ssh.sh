#!/usr/bin/env bash
set -euo pipefail

# Evaluate pass@1/pass@32 on Isambard (GH200) with vLLM, re-rendering prompts from a *local* Jinja template.
#
# Usage (single local arg):
#   scripts/isambard_eval_passk_from_template_ssh.sh /path/to/prompt_template.jinja
#
# This script:
#   - streams the local template to the cluster over SSH stdin
#   - creates a remote temp dir via mktemp -d (cleaned up via trap)
#   - submits a Slurm job with `sbatch --wait`
#   - runs inside the same Apptainer image as training (`gabr1e1/chess_rl:v1-arm-stockfish-flashinfer`)
#   - prints final metrics JSON to stdout
#
# Optional env overrides (set locally; this script forwards them over SSH):
#   CHESS_RL_ISAMBARD_HOST        (default: a5l.aip2.isambard)
#   CHESS_RL_ISAMBARD_REPO_DIR    (default: ~/code/chess-rl)
#   CHESS_RL_SIF_IMAGE            (default: ~/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif)
#   CHESS_RL_PROJECT_ROOT         (default: /projects/a5l/ziyan)
#   CHESS_RL_SLURM_PARTITION      (default: workq)
#   CHESS_RL_SLURM_ACCOUNT        (default: brics.a5l)
#   CHESS_RL_SLURM_TIME           (default: 0-04:00:00)
#
# Optional eval overrides (set locally; forwarded to the Slurm job environment):
#   CHESS_EVAL_LIMIT              (default: full parquet)
#   CHESS_EVAL_BATCH_SIZE         (default: 32)
#   CHESS_EVAL_MAX_NUM_SEQS       (default: 1024)
#
# Async mode:
#   CHESS_RL_SLURM_WAIT=0         Submit via sbatch (no --wait) and return immediately,
#                                printing job/log locations for manual monitoring.

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/prompt_template.jinja" >&2
  exit 2
fi

TEMPLATE_LOCAL_PATH="$1"
if [ ! -f "${TEMPLATE_LOCAL_PATH}" ]; then
  echo "[ERROR] Template not found: ${TEMPLATE_LOCAL_PATH}" >&2
  exit 1
fi

ISAMBARD_HOST="${CHESS_RL_ISAMBARD_HOST:-a5l.aip2.isambard}"
TEMPLATE_BASENAME="$(basename "${TEMPLATE_LOCAL_PATH}")"
if [ -z "${TEMPLATE_BASENAME}" ] || [ "${TEMPLATE_BASENAME}" = "/" ] || echo "${TEMPLATE_BASENAME}" | grep -q '/'; then
  echo "[ERROR] Could not derive a safe template basename from: ${TEMPLATE_LOCAL_PATH}" >&2
  exit 1
fi

# Forward selected env vars over SSH without relying on sshd AcceptEnv config.
REMOTE_ENV_VARS=(
  CHESS_RL_ISAMBARD_REPO_DIR
  CHESS_RL_SIF_IMAGE
  CHESS_RL_PROJECT_ROOT
  CHESS_RL_SLURM_PARTITION
  CHESS_RL_SLURM_ACCOUNT
  CHESS_RL_SLURM_TIME
  CHESS_RL_SLURM_WAIT
  CHESS_EVAL_LIMIT
  CHESS_EVAL_BATCH_SIZE
  CHESS_EVAL_MAX_NUM_SEQS
)
REMOTE_ENV_ASSIGNMENTS=""
for _var in "${REMOTE_ENV_VARS[@]}"; do
  _val="${!_var-}"
  if [ -n "${_val}" ]; then
    REMOTE_ENV_ASSIGNMENTS+="${_var}=$(printf '%q' "${_val}") "
  fi
done

# Preserve the original local template filename on the remote side so output log
# filenames can be derived from it deterministically.
REMOTE_ENV_ASSIGNMENTS+="CHESS_EVAL_TEMPLATE_BASENAME=$(printf '%q' "${TEMPLATE_BASENAME}") "

REMOTE_CMD="$(cat <<'REMOTE'
set -euo pipefail

REPO_DIR="${CHESS_RL_ISAMBARD_REPO_DIR:-$HOME/code/chess-rl}"
SIF_IMAGE="${CHESS_RL_SIF_IMAGE:-$HOME/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif}"
PROJECT_ROOT="${CHESS_RL_PROJECT_ROOT:-/projects/a5l/ziyan}"
SLURM_PARTITION="${CHESS_RL_SLURM_PARTITION:-workq}"
SLURM_ACCOUNT="${CHESS_RL_SLURM_ACCOUNT:-brics.a5l}"
SLURM_TIME="${CHESS_RL_SLURM_TIME:-0-04:00:00}"
SLURM_WAIT="${CHESS_RL_SLURM_WAIT:-1}"

if [ ! -d "${REPO_DIR}" ]; then
  echo "[ERROR] Remote repo dir not found: ${REPO_DIR}" >&2
  echo "[HINT] Set CHESS_RL_ISAMBARD_REPO_DIR or clone the repo to ~/code/chess-rl on the cluster." >&2
  exit 1
fi

if command -v git >/dev/null 2>&1; then
  # If multiple SSH sessions run this script concurrently, serialize git operations
  # so the repo doesn't end up with a corrupted/ambiguous FETCH_HEAD.
  echo "[REMOTE] Syncing repo (git pull --ff-only origin main)..." >&2
  if command -v flock >/dev/null 2>&1; then
    GIT_LOCK_FILE="${REPO_DIR}/.git/chess_rl_eval_git.lock"
    mkdir -p "$(dirname "${GIT_LOCK_FILE}")"
    exec 9>"${GIT_LOCK_FILE}"
    flock 9
    (cd "${REPO_DIR}" && git pull --ff-only origin main) || {
      echo "[ERROR] git pull failed (repo may have local changes). Fix the remote checkout and retry." >&2
      exit 1
    }
    flock -u 9
  else
    (cd "${REPO_DIR}" && git pull --ff-only origin main) || {
      echo "[ERROR] git pull failed (repo may have local changes). Fix the remote checkout and retry." >&2
      exit 1
    }
  fi
fi

TMPDIR="$(mktemp -d -p "${HOME}" chess_rl_eval_template.XXXXXXXX)"
cleanup() { rm -rf "${TMPDIR}"; }
trap cleanup EXIT

TEMPLATE_PATH="${TMPDIR}/prompt_template.jinja"
TEMPLATE_BASENAME="${CHESS_EVAL_TEMPLATE_BASENAME:-prompt_template.jinja}"
TEMPLATE_BASENAME="$(basename "${TEMPLATE_BASENAME}")"
if [ -z "${TEMPLATE_BASENAME}" ] || [ "${TEMPLATE_BASENAME}" = "/" ] || echo "${TEMPLATE_BASENAME}" | grep -q '/'; then
  TEMPLATE_BASENAME="prompt_template.jinja"
fi
TEMPLATE_PATH="${TMPDIR}/${TEMPLATE_BASENAME}"
cat > "${TEMPLATE_PATH}"
echo "[REMOTE] Wrote template: ${TEMPLATE_PATH}"

# Derive stable output names from the *original* template basename.
TEMPLATE_NAME_RAW="$(basename "${TEMPLATE_PATH}")"
TEMPLATE_STEM="${TEMPLATE_NAME_RAW%.*}"
TEMPLATE_SAFE="$(echo "${TEMPLATE_STEM}" | tr -c 'A-Za-z0-9._-' '_' | sed 's/^_*//; s/_*$//')"
if [ -z "${TEMPLATE_SAFE}" ]; then
  TEMPLATE_SAFE="template"
fi
OUT_DIR="${PROJECT_ROOT%/}/chess_rl_outputs"
mkdir -p "${OUT_DIR}"

# Ensure the container image exists (build on login node if missing).
if [ ! -f "${SIF_IMAGE}" ]; then
  echo "[REMOTE] Missing SIF image: ${SIF_IMAGE}" >&2
  echo "[REMOTE] Building it on the login node (one-time)..." >&2
  if ! command -v apptainer >/dev/null 2>&1; then
    echo "[ERROR] apptainer not found on PATH; load the site module, then retry." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${SIF_IMAGE}")"
  apptainer build "${SIF_IMAGE}" docker://gabr1e1/chess_rl:v1-arm-stockfish-flashinfer
fi

SBATCH_FILE="${TMPDIR}/job.sbatch"
cat > "${SBATCH_FILE}" <<EOF
#!/bin/bash
#SBATCH --job-name=chess-passk-template
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --time=${SLURM_TIME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --export=ALL
#SBATCH --output=${OUT_DIR%/}/slurm-passk-template-${TEMPLATE_SAFE}-%j.out
#SBATCH --error=${OUT_DIR%/}/slurm-passk-template-${TEMPLATE_SAFE}-%j.err

set -euo pipefail

die() { echo "[ERROR] \$*" >&2; exit 1; }

echo "[SLURM] Job ID: \${SLURM_JOB_ID:-N/A}"
echo "[SLURM] Node:   \${SLURM_JOB_NODELIST:-N/A}"
echo "[SLURM] GPUs:   \${SLURM_GPUS_ON_NODE:-4}"

HOST_ARCH="\$(uname -m 2>/dev/null || echo unknown)"
echo "[SLURM] Host arch: \${HOST_ARCH}"
if [ "\${HOST_ARCH}" != "aarch64" ] && [ "\${HOST_ARCH}" != "arm64" ]; then
  die "Expected GH200 (arm64/aarch64) node, got '\${HOST_ARCH}'."
fi

THIS_DIR="\${SLURM_SUBMIT_DIR:-\$(pwd)}"
cd "\${THIS_DIR}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID

HOST_TMPDIR="${TMPDIR}"
cleanup_host_tmp() { rm -rf "\${HOST_TMPDIR}"; }
trap cleanup_host_tmp EXIT

SIF_IMAGE="${SIF_IMAGE}"
CONTAINER_REPO_DIR="\${CHESS_RL_CONTAINER_REPO_DIR:-/workspace/chess_rl}"
if [ ! -f "\${SIF_IMAGE}" ]; then
  die "Missing SIF image: \${SIF_IMAGE}"
fi

PROJECT_ROOT="${PROJECT_ROOT}"
HF_CACHE_ROOT="\${CHESS_RL_HF_CACHE_ROOT:-\${PROJECT_ROOT%/}/hf_cache}"
mkdir -p "\${HF_CACHE_ROOT}"
export HF_HOME="\${HF_CACHE_ROOT%/}/hf_home"
export HUGGINGFACE_HUB_CACHE="\${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="\${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="\${HF_HOME%/}/datasets"
mkdir -p "\${HUGGINGFACE_HUB_CACHE}" "\${TRANSFORMERS_CACHE}" "\${HF_DATASETS_CACHE}"
echo "[SLURM] HF_HOME=\${HF_HOME}"

TEMPLATE_HOST_PATH="${TEMPLATE_PATH}"
if [ ! -f "\${TEMPLATE_HOST_PATH}" ]; then
  die "Template missing on host: \${TEMPLATE_HOST_PATH}"
fi
TEMPLATE_HOST_DIR="\$(dirname "\${TEMPLATE_HOST_PATH}")"

TEMPLATE_SAFE="${TEMPLATE_SAFE}"
OUT_DIR="${OUT_DIR}"
OUT_JSONL="\${OUT_DIR%/}/passk_template_\${TEMPLATE_SAFE}_job\${SLURM_JOB_ID}.jsonl"
export CHESS_EVAL_OUT_JSONL="\${OUT_JSONL}"

# Use 4 GPUs via tensor parallelism (vLLM).
export CHESS_EVAL_TENSOR_PARALLEL_SIZE=4
# Default to a larger batch size on-cluster (caller may override via env var).
export CHESS_EVAL_BATCH_SIZE="\${CHESS_EVAL_BATCH_SIZE:-128}"

CMD="unset PYTHONHOME PYTHONPATH; cd \"\${CONTAINER_REPO_DIR}\" && /usr/bin/python3 -m scripts.eval_chess_passk_from_template \"\${TEMPLATE_HOST_PATH}\""
echo "[SLURM] Container CMD: \${CMD}"
echo "[SLURM] OUT_JSONL=\${OUT_JSONL}"

JOB_START_S="\$(date +%s)"

apptainer exec --nv \
  --bind "\${THIS_DIR}:\${CONTAINER_REPO_DIR}" \
  --bind "\${PROJECT_ROOT}:\${PROJECT_ROOT}" \
  --bind "\${HF_CACHE_ROOT}:\${HF_CACHE_ROOT}" \
  --bind "\${TEMPLATE_HOST_DIR}:\${TEMPLATE_HOST_DIR}" \
  "\${SIF_IMAGE}" \
  bash -c "\${CMD}"

JOB_END_S="\$(date +%s)"
echo "[SLURM] Wall time (apptainer exec): \$((JOB_END_S - JOB_START_S))s"
EOF

chmod 700 "${SBATCH_FILE}"

cd "${REPO_DIR}"

if [ "${SLURM_WAIT}" = "0" ]; then
  echo "[REMOTE] Submitting (async): sbatch ${SBATCH_FILE}"
  SUBMIT_OUT="$(sbatch "${SBATCH_FILE}")"
  echo "${SUBMIT_OUT}"
  JOB_ID="$(echo "${SUBMIT_OUT}" | awk '{print $4}')"
  if ! echo "${JOB_ID}" | grep -Eq '^[0-9]+$'; then
    echo "[ERROR] Could not parse job id from: ${SUBMIT_OUT}" >&2
    exit 1
  fi
  OUT_JSONL="${OUT_DIR%/}/passk_template_${TEMPLATE_SAFE}_job${JOB_ID}.jsonl"
  SLURM_OUT="${OUT_DIR%/}/slurm-passk-template-${TEMPLATE_SAFE}-${JOB_ID}.out"
  SLURM_ERR="${OUT_DIR%/}/slurm-passk-template-${TEMPLATE_SAFE}-${JOB_ID}.err"

  echo "[REMOTE] Job ID:    ${JOB_ID}"
  echo "[REMOTE] Stdout:    ${SLURM_OUT}"
  echo "[REMOTE] Stderr:    ${SLURM_ERR}" >&2
  echo "[REMOTE] OUT_JSONL: ${OUT_JSONL}"

  # Important: keep TMPDIR until the job finishes (job will delete it via trap).
  trap - EXIT
  exit 0
fi

SLURM_OUT="${TMPDIR}/slurm.out"
SLURM_ERR="${TMPDIR}/slurm.err"

echo "[REMOTE] Submitting: sbatch --wait ${SBATCH_FILE}"
set +e
sbatch --wait --output="${SLURM_OUT}" --error="${SLURM_ERR}" "${SBATCH_FILE}"
JOB_RC="$?"
set -e

echo "[REMOTE] ===== Slurm stdout ====="
cat "${SLURM_OUT}" || true
echo "[REMOTE] ===== Slurm stderr =====" >&2
cat "${SLURM_ERR}" >&2 || true

exit "${JOB_RC}"
REMOTE
)"

# Stream the template file to the remote script via SSH stdin.
cat "${TEMPLATE_LOCAL_PATH}" | ssh "${ISAMBARD_HOST}" "${REMOTE_ENV_ASSIGNMENTS}bash -lc $(printf '%q' "${REMOTE_CMD}")"
