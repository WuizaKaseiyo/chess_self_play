#!/bin/bash
#SBATCH --job-name=chess-select-pos-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=0-05:00:00
#SBATCH --output=slurm/slurm-%x-%j.out
#SBATCH --error=slurm/slurm-%x-%j.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

# Position-sweep evaluation on GH200 (Apptainer + vLLM).
# Example smoke:
# sbatch --wait --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,LIMIT_ROWS=2,SAMPLES_PER_PROMPT=8,MAX_K=4,SEED=0 ./sbatch_eval_chess_select_position_sweep_gh200.slurm

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

mkdir -p slurm

echo "[SLURM] Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "[SLURM] Job name:    ${SLURM_JOB_NAME:-chess-select-pos-sweep}"
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
if [ "${HOST_ARCH}" != "aarch64" ] && [ "${HOST_ARCH}" != "arm64" ]; then
  echo "[ERROR] This script targets GH200 (arm64/aarch64), but uname reports '${HOST_ARCH}'." >&2
  exit 1
fi

SIF_IMAGE="${CHESS_RL_SIF_IMAGE:-${HOME}/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif}"
CONTAINER_REPO_DIR="${CHESS_RL_CONTAINER_REPO_DIR:-/workspace/chess_rl}"
if [ ! -f "${SIF_IMAGE}" ]; then
  echo "[ERROR] Missing SIF image: ${SIF_IMAGE}" >&2
  exit 1
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

# Eval config (override via sbatch --export=ALL,...).
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PARQUET="${PARQUET:-data/chess_puzzles/test.parquet}"
TEMPLATE_PATH="${TEMPLATE_PATH:-recipe/chess/prompt_templates/select_prompt.jinja}"

LIMIT_ROWS="${LIMIT_ROWS:-1000}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-32}"
SEED="${SEED:-0}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"

MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

NUM_SHARDS="${NUM_SHARDS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_K="${MAX_K:- -1}"
MODE="${MODE:-both}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-0}"

echo "[SLURM] MODEL=${MODEL}"
echo "[SLURM] PARQUET=${PARQUET}"
echo "[SLURM] TEMPLATE_PATH=${TEMPLATE_PATH}"
echo "[SLURM] LIMIT_ROWS=${LIMIT_ROWS} SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT} MAX_K=${MAX_K}"
JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
LOCAL_GPU_COUNT="${SLURM_GPUS_ON_NODE:-4}"
if [ "${JOB_NUM_NODES}" -gt 1 ]; then
  MIN_SHARDS=$((JOB_NUM_NODES * LOCAL_GPU_COUNT))
  if [ "${NUM_SHARDS}" -lt "${MIN_SHARDS}" ]; then
    NUM_SHARDS="${MIN_SHARDS}"
  fi
fi
echo "[SLURM] NUM_SHARDS=${NUM_SHARDS} BATCH_SIZE=${BATCH_SIZE} MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "[SLURM] MODE=${MODE} USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE}"

CMD="set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd \"${CONTAINER_REPO_DIR}\"

TEMPLATE_HASH=\$(sha256sum \"${TEMPLATE_PATH}\" | awk '{print substr(\$1,1,12)}')
MODEL_SAFE=\$(python3 -c \"import re,sys; s=sys.argv[1]; s=s.replace('/','__').replace(':','_'); s=re.sub(r'[^A-Za-z0-9._-]+','_',s); print(s)\" \"${MODEL}\")
OUT_DIR=\"${PROJECT_ROOT%/}/chess_rl_outputs/select_pos_sweep/\${MODEL_SAFE}__\${TEMPLATE_HASH}_seed${SEED}_rows${LIMIT_ROWS}_samples${SAMPLES_PER_PROMPT}\"
mkdir -p \"\${OUT_DIR}\"
echo \"[RUN] OUT_DIR=\${OUT_DIR}\"

NODE_ID=\"\${SLURM_NODEID:-0}\"
TOTAL_NODES=\"\${SLURM_JOB_NUM_NODES:-\${SLURM_NNODES:-1}}\"
LOCAL_GPU_COUNT=\"\${SLURM_GPUS_ON_NODE:-4}\"
SHARD_BASE=\$((NODE_ID * LOCAL_GPU_COUNT))
echo \"[RUN] Launching shard workers... (node_id=\${NODE_ID} shard_base=\${SHARD_BASE} num_shards=${NUM_SHARDS})\"
EXTRA_ARGS=\"\"
if [ \"${USE_CHAT_TEMPLATE}\" = \"1\" ] || [ \"${USE_CHAT_TEMPLATE}\" = \"true\" ]; then
  EXTRA_ARGS=\"--use_chat_template\"
fi

CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -m scripts.eval_chess_select_position_sweep \\
  --model \"${MODEL}\" --parquet \"${PARQUET}\" --template_path \"${TEMPLATE_PATH}\" --out_dir \"\${OUT_DIR}\" \\
  --limit_rows \"${LIMIT_ROWS}\" --samples_per_prompt \"${SAMPLES_PER_PROMPT}\" --max_k \"${MAX_K}\" \\
  --seed \"${SEED}\" --temperature \"${TEMPERATURE}\" --top_p \"${TOP_P}\" \\
  --max_prompt_tokens \"${MAX_PROMPT_TOKENS}\" --max_response_tokens \"${MAX_RESPONSE_TOKENS}\" --max_model_len \"${MAX_MODEL_LEN}\" \\
  --tensor_parallel_size 1 --batch_size \"${BATCH_SIZE}\" --max_num_seqs \"${MAX_NUM_SEQS}\" --gpu_memory_utilization \"${GPU_MEMORY_UTILIZATION}\" \\
  --num_shards \"${NUM_SHARDS}\" --shard_idx \$((SHARD_BASE + 0)) --mode \"${MODE}\" \${EXTRA_ARGS} &
P0=\$!
CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 -m scripts.eval_chess_select_position_sweep \\
  --model \"${MODEL}\" --parquet \"${PARQUET}\" --template_path \"${TEMPLATE_PATH}\" --out_dir \"\${OUT_DIR}\" \\
  --limit_rows \"${LIMIT_ROWS}\" --samples_per_prompt \"${SAMPLES_PER_PROMPT}\" --max_k \"${MAX_K}\" \\
  --seed \"${SEED}\" --temperature \"${TEMPERATURE}\" --top_p \"${TOP_P}\" \\
  --max_prompt_tokens \"${MAX_PROMPT_TOKENS}\" --max_response_tokens \"${MAX_RESPONSE_TOKENS}\" --max_model_len \"${MAX_MODEL_LEN}\" \\
  --tensor_parallel_size 1 --batch_size \"${BATCH_SIZE}\" --max_num_seqs \"${MAX_NUM_SEQS}\" --gpu_memory_utilization \"${GPU_MEMORY_UTILIZATION}\" \\
  --num_shards \"${NUM_SHARDS}\" --shard_idx \$((SHARD_BASE + 1)) --mode \"${MODE}\" \${EXTRA_ARGS} &
P1=\$!
CUDA_VISIBLE_DEVICES=2 /usr/bin/python3 -m scripts.eval_chess_select_position_sweep \\
  --model \"${MODEL}\" --parquet \"${PARQUET}\" --template_path \"${TEMPLATE_PATH}\" --out_dir \"\${OUT_DIR}\" \\
  --limit_rows \"${LIMIT_ROWS}\" --samples_per_prompt \"${SAMPLES_PER_PROMPT}\" --max_k \"${MAX_K}\" \\
  --seed \"${SEED}\" --temperature \"${TEMPERATURE}\" --top_p \"${TOP_P}\" \\
  --max_prompt_tokens \"${MAX_PROMPT_TOKENS}\" --max_response_tokens \"${MAX_RESPONSE_TOKENS}\" --max_model_len \"${MAX_MODEL_LEN}\" \\
  --tensor_parallel_size 1 --batch_size \"${BATCH_SIZE}\" --max_num_seqs \"${MAX_NUM_SEQS}\" --gpu_memory_utilization \"${GPU_MEMORY_UTILIZATION}\" \\
  --num_shards \"${NUM_SHARDS}\" --shard_idx \$((SHARD_BASE + 2)) --mode \"${MODE}\" \${EXTRA_ARGS} &
P2=\$!
CUDA_VISIBLE_DEVICES=3 /usr/bin/python3 -m scripts.eval_chess_select_position_sweep \\
  --model \"${MODEL}\" --parquet \"${PARQUET}\" --template_path \"${TEMPLATE_PATH}\" --out_dir \"\${OUT_DIR}\" \\
  --limit_rows \"${LIMIT_ROWS}\" --samples_per_prompt \"${SAMPLES_PER_PROMPT}\" --max_k \"${MAX_K}\" \\
  --seed \"${SEED}\" --temperature \"${TEMPERATURE}\" --top_p \"${TOP_P}\" \\
  --max_prompt_tokens \"${MAX_PROMPT_TOKENS}\" --max_response_tokens \"${MAX_RESPONSE_TOKENS}\" --max_model_len \"${MAX_MODEL_LEN}\" \\
  --tensor_parallel_size 1 --batch_size \"${BATCH_SIZE}\" --max_num_seqs \"${MAX_NUM_SEQS}\" --gpu_memory_utilization \"${GPU_MEMORY_UTILIZATION}\" \\
  --num_shards \"${NUM_SHARDS}\" --shard_idx \$((SHARD_BASE + 3)) --mode \"${MODE}\" \${EXTRA_ARGS} &
P3=\$!

wait \"\${P0}\"
wait \"\${P1}\"
wait \"\${P2}\"
wait \"\${P3}\"

/usr/bin/python3 -m scripts.analyze_chess_select_position_sweep --results_dir \"\${OUT_DIR}\" --out_dir \"\${OUT_DIR}/analysis\" --smooth_window 5 --no_plots
"

echo "[SLURM] Launching container..."

if command -v srun >/dev/null 2>&1; then
  srun --export=ALL \
    apptainer exec --nv \
      --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
      --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
      --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
      --env "HF_HOME=${HF_HOME}" \
      --env "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}" \
      --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
      --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" \
      "${SIF_IMAGE}" \
      bash -lc "${CMD}"
else
  apptainer exec --nv \
    --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
    --env "HF_HOME=${HF_HOME}" \
    --env "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}" \
    --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
    --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" \
    "${SIF_IMAGE}" \
    bash -lc "${CMD}"
fi
