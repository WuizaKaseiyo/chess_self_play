#!/usr/bin/env bash
set -euo pipefail
set -x

# Default to the repo root (script dir) so relative paths keep working even if
# this script is invoked from elsewhere (e.g., inside a Docker container).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
cd "${PROJECT_DIR}"

# Prefer deterministic device ordering; vLLM warns when mixing GPU SKUs.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# How many GPUs this job should use on the node.
# - Local default is 2 (matches the known-good local setup).
# - Cluster runs should export TRAINER_N_GPUS_PER_NODE (the sbatch script does).
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-2}"
if ! [[ "${TRAINER_N_GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || [ "${TRAINER_N_GPUS_PER_NODE}" -le 0 ]; then
    echo "Invalid TRAINER_N_GPUS_PER_NODE='${TRAINER_N_GPUS_PER_NODE}'; defaulting to 2." >&2
    TRAINER_N_GPUS_PER_NODE=2
fi
export TRAINER_N_GPUS_PER_NODE

# How many nodes to expect in the Ray cluster.
# - Default to 1 for local runs.
# - Multi-node Slurm launchers should export TRAINER_NNODES.
TRAINER_NNODES="${TRAINER_NNODES:-1}"
if ! [[ "${TRAINER_NNODES}" =~ ^[0-9]+$ ]] || [ "${TRAINER_NNODES}" -le 0 ]; then
    echo "Invalid TRAINER_NNODES='${TRAINER_NNODES}'; defaulting to 1." >&2
    TRAINER_NNODES=1
fi
export TRAINER_NNODES

# Some components (e.g., Megatron-Core unified memory) JIT-compile CUDA extensions
# at import time. Ray may start workers with `CUDA_VISIBLE_DEVICES=` (no GPU), which
# can make PyTorch fail to infer architectures and crash. Export a stable arch list
# from `nvidia-smi` so extension builds don't depend on GPU visibility.
if command -v nvidia-smi >/dev/null 2>&1; then
    NVIDIA_SMI_GPU_ARGS=()
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NVIDIA_SMI_GPU_ARGS=(-i "${CUDA_VISIBLE_DEVICES}")
    fi
    DETECTED_ARCHES="$(nvidia-smi "${NVIDIA_SMI_GPU_ARGS[@]}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u | paste -sd ';' -)"
    if [ -n "${DETECTED_ARCHES}" ]; then
        export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${DETECTED_ARCHES}}"
    fi
fi

set +x
# W&B (research-mode):
# - Default to the embedded API key for convenience.
# - Allow callers (e.g., cluster launchers) to override via env var.
export WANDB_API_KEY="${WANDB_API_KEY:-1bc80152d957aadd5c7d81b5422e76d5352fa324}"

set -x
USE_CRITIC_DATASET="${USE_CRITIC_DATASET:-False}"
USE_HARD_DATASET="${USE_HARD_DATASET:-False}"
CHESS_SELF_PLAY_ENABLE="${CHESS_SELF_PLAY_ENABLE:-False}"

# Dataset directory override.
# - Default matches the in-repo restricted-moves ("selection") datasets under
#   `data/chess_puzzles_select/` (generated via `scripts/build_chess_select_*_dataset.py`).
# - Set this to point at an alternative dataset dir containing either:
#     - `train.parquet` (single file), or `train_[0-9]*.parquet` (numeric shards)
#     - `test.parquet`  (single file), or `test_[0-9]*.parquet`  (numeric shards)
CHESS_DATA_DIR="${CHESS_DATA_DIR:-${PROJECT_DIR}/data/chess_puzzles_select}"
# Optional override for validation files (single path or Hydra list).
# - When set, this is passed verbatim to `data.val_files` (can include multiple parquets).
CHESS_RL_VAL_FILES="${CHESS_RL_VAL_FILES:-${VAL_FILES:-}}"

# Baseline naming guardrail: if job name contains "baseline", require the baseline Chess-R1 dataset dir.
JOB_NAME="${CHESS_RL_JOB_NAME:-${SLURM_JOB_NAME:-${JOB_NAME:-}}}"
JOB_NAME_LC="$(printf '%s' "${JOB_NAME}" | tr '[:upper:]' '[:lower:]')"
if [[ "${JOB_NAME_LC}" == *baseline* ]]; then
    if [[ "${CHESS_DATA_DIR}" != *chess_puzzles_chessr1_aligned_sharded_baseline* ]]; then
        echo "ERROR: job name contains 'baseline' but CHESS_DATA_DIR='${CHESS_DATA_DIR}'" >&2
        echo "Expected baseline dataset: data/chess_puzzles_chessr1_aligned_sharded_baseline" >&2
        exit 1
    fi
fi

# Capture whether the selection template was explicitly set before defaults are applied.
ALLOWED_MOVE_ELIM_TEMPLATE_PATH_WAS_SET=False
if [ -n "${ALLOWED_MOVE_ELIM_TEMPLATE_PATH+x}" ]; then
    ALLOWED_MOVE_ELIM_TEMPLATE_PATH_WAS_SET=True
fi

# Fixed (non-micro) batch sizes. Memory is governed by micro-batching.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_PATH="${CHESS_RL_TOKENIZER_PATH:-${TOKENIZER_PATH:-}}"
MODEL_PATH_LC="$(printf '%s' "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')"
MODEL_IS_QWEN3_BASE=False
if [[ "${MODEL_PATH_LC}" == *qwen3-* && "${MODEL_PATH_LC}" == *-base* ]]; then
    MODEL_IS_QWEN3_BASE=True
fi
USE_CHAT_TEMPLATE="${CHESS_RL_USE_CHAT_TEMPLATE:-${USE_CHAT_TEMPLATE:-}}"
if [ -z "${USE_CHAT_TEMPLATE}" ]; then
    if [ "${MODEL_IS_QWEN3_BASE}" = "True" ]; then
        USE_CHAT_TEMPLATE=False
    else
        USE_CHAT_TEMPLATE=True
    fi
fi
case "${USE_CHAT_TEMPLATE}" in
    True|False) ;;
    true) USE_CHAT_TEMPLATE=True ;;
    false) USE_CHAT_TEMPLATE=False ;;
    1|yes|on) USE_CHAT_TEMPLATE=True ;;
    0|no|off) USE_CHAT_TEMPLATE=False ;;
    *)
        echo "Invalid USE_CHAT_TEMPLATE='${USE_CHAT_TEMPLATE}'; expected True or False." >&2
        exit 1
        ;;
esac
if [ "${MODEL_IS_QWEN3_BASE}" = "True" ] && [ "${USE_CHAT_TEMPLATE}" != "False" ]; then
    echo "ERROR: Qwen3 base runs must use USE_CHAT_TEMPLATE=False." >&2
    exit 1
fi
RESPONSE_PREFIX_ENABLE="${CHESS_RL_RESPONSE_PREFIX_ENABLE:-${RESPONSE_PREFIX_ENABLE:-}}"
if [ -z "${RESPONSE_PREFIX_ENABLE}" ]; then
    RESPONSE_PREFIX_ENABLE=False
fi
case "${RESPONSE_PREFIX_ENABLE}" in
    True|False) ;;
    true) RESPONSE_PREFIX_ENABLE=True ;;
    false) RESPONSE_PREFIX_ENABLE=False ;;
    1|yes|on) RESPONSE_PREFIX_ENABLE=True ;;
    0|no|off) RESPONSE_PREFIX_ENABLE=False ;;
    *)
        echo "Invalid RESPONSE_PREFIX_ENABLE='${RESPONSE_PREFIX_ENABLE}'; expected True or False." >&2
        exit 1
        ;;
esac
RESPONSE_PREFIX="${CHESS_RL_RESPONSE_PREFIX:-${RESPONSE_PREFIX:-}}"
if [ -z "${RESPONSE_PREFIX}" ]; then
    if [ "${MODEL_IS_QWEN3_BASE}" = "True" ]; then
        RESPONSE_PREFIX=""
    else
        RESPONSE_PREFIX="<think>"
    fi
fi
if [ "${MODEL_IS_QWEN3_BASE}" = "True" ] && [ "${ALLOWED_MOVE_ELIM_TEMPLATE_PATH_WAS_SET}" != "True" ]; then
    ALLOWED_MOVE_ELIM_TEMPLATE_PATH="${PROJECT_DIR}/recipe/chess/prompt_templates/select_prompt_base.jinja"
fi
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
# NOTE: Some models can become verbose in reasoning tags. Keep this high enough to avoid truncation,
# but do not rely on stop-sequence hacks like `</uci_move>` for correctness.
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
DATA_SHUFFLE="${DATA_SHUFFLE:-True}"
# Optional dataset sampler seed.
# - Set to an integer for reproducible dataloader order.
# - Keep `null` to preserve existing behavior.
DATA_SEED="${DATA_SEED:-null}"

# Generation settings used for *validation-style* sampling.
# - These are used for one-step validation (val kwargs) and are also the default source-of-truth
#   for full-game eval sampling unless FULL_EVAL_* overrides are provided.
VAL_TEMPERATURE="${CHESS_RL_VAL_TEMPERATURE:-0.6}"
VAL_TOP_P="${CHESS_RL_VAL_TOP_P:-0.95}"

# GRPO group size / total samples per prompt.
# - Forced-prefix injection is applied *per rollout* with an annealed Bernoulli schedule p(t).
ROLLOUT_N="${ROLLOUT_N:-8}"

# DAPO-style dynamic sampling (group filtering):
# - When enabled, the trainer keeps generating prompt batches of size `GEN_BATCH_SIZE` until it has
#   enough *qualified* prompt groups (variance in the chosen metric) to fill `TRAIN_BATCH_SIZE`,
#   or hits the cap `MAX_NUM_GEN_BATCHES`.
# - Baseline runs are expected to match Chess-R1 baseline behavior (`filter_groups=False`) unless
#   explicitly overridden via env var.
FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-}"
if [ -z "${FILTER_GROUPS_ENABLE}" ]; then
    if [[ "${JOB_NAME_LC}" == *baseline* ]]; then
        FILTER_GROUPS_ENABLE="False"
    else
        FILTER_GROUPS_ENABLE="True"
    fi
fi
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-score}"  # acc / score / seq_reward / seq_final_reward / ...
MAX_NUM_GEN_BATCHES="${MAX_NUM_GEN_BATCHES:-10}"

# On-the-fly allowed-move elimination (selection sampling):
# - When enabled, this replaces filter_groups with a batched multi-round sampler that
#   updates the allowed-move list per round.
ALLOWED_MOVE_ELIM_ENABLE="${ALLOWED_MOVE_ELIM_ENABLE:-False}"
ALLOWED_MOVE_ELIM_TEMPLATE_PATH="${ALLOWED_MOVE_ELIM_TEMPLATE_PATH:-${PROJECT_DIR}/recipe/chess/prompt_templates/select_prompt.jinja}"
ALLOWED_MOVE_ELIM_UID_MODE="${ALLOWED_MOVE_ELIM_UID_MODE:-per_round}"
ALLOWED_MOVE_ELIM_NO_SUCCESS_POLICY="${ALLOWED_MOVE_ELIM_NO_SUCCESS_POLICY:-accept_last}"
ALLOWED_MOVE_ELIM_R_MAX_START="${ALLOWED_MOVE_ELIM_R_MAX_START:-4}"
ALLOWED_MOVE_ELIM_R_MAX_END="${ALLOWED_MOVE_ELIM_R_MAX_END:-1}"
ALLOWED_MOVE_ELIM_ANNEAL_FRAC="${ALLOWED_MOVE_ELIM_ANNEAL_FRAC:-0.5}"
ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB="${ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB:-False}"
ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI="${ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI:-False}"
# Optional: reject GRPO groups with low within-group reward range (disabled by default).
ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN="${ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN:-0.0}"
ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_DUMP_MAX_GROUPS="${ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_DUMP_MAX_GROUPS:-16}"
# Gain filter threshold for iterative rounds: Gain = logprob(r|p_i) - logprob(r|p_0).
# - Set to "inf" to disable gain filtering. This is the default for current
#   allowed_move_elim runs because the filter can reject an entire online step.
ALLOWED_MOVE_ELIM_GAIN_THRESHOLD="${ALLOWED_MOVE_ELIM_GAIN_THRESHOLD:-inf}"

# Optional: conditional Pass@k advantage for allowed_move_elim groups that contain no optimal rollout.
# - This is separate from the baseline `PASS_K_TRAINING` switch, which applies to all GRPO groups.
ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL="${ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL:-False}"
ALLOWED_MOVE_ELIM_PASS_K_K="${ALLOWED_MOVE_ELIM_PASS_K_K:-4}"
# Optional: apply diversity advantages only to no-optimal allowed_move_elim GRPO groups.
ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL="${ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL:-False}"

# Optional: Pass@k analytic advantage variant for baseline GRPO.
# - Default-off for backward compatibility.
PASS_K_TRAINING="${PASS_K_TRAINING:-False}"
PASS_K="${PASS_K:-1}"

# Optional: diversity auxiliary advantages for GRPO/Pass@k.
# - Default-off and lambda=0 preserve baseline behavior.
# - Methods: none | obe_batch | gapo | distinct_k
DIVERSITY_ENABLE="${DIVERSITY_ENABLE:-False}"
DIVERSITY_METHOD="${DIVERSITY_METHOD:-none}"
DIVERSITY_LAMBDA="${DIVERSITY_LAMBDA:-0.0}"
DIVERSITY_DISTINCT_K="${DIVERSITY_DISTINCT_K:-4}"
# Optional diversity-only mode (disables baseline advantage term).
DIVERSITY_INCLUDE_BASE_ADVANTAGE="${DIVERSITY_INCLUDE_BASE_ADVANTAGE:-True}"

if [ "${ALLOWED_MOVE_ELIM_ENABLE}" = "True" ]; then
    FILTER_GROUPS_ENABLE="False"
fi

if [ "${CHESS_SELF_PLAY_ENABLE}" = "True" ] && [ "${ALLOWED_MOVE_ELIM_ENABLE}" != "True" ]; then
    echo "ERROR: CHESS_SELF_PLAY_ENABLE=True requires ALLOWED_MOVE_ELIM_ENABLE=True (online play vs engine opponent feeds the iterative sampler)." >&2
    exit 1
fi

# Note: online play vs engine opponent (misnamed self-play) maintains a persistent pool of
# `TRAIN_BATCH_SIZE` parallel games (model vs Stockfish depth=1). Each trainer step:
# - records exactly one position per game (both model-to-move and engine-to-move), then
# - advances each game by exactly one ply, and
# - restarts ended games immediately (toggling who moves first for that slot).

# Forced-prefix exploration is disabled for selection framing (no `<guess>` scheme).
FORCED_PREFIX_ENABLE="${FORCED_PREFIX_ENABLE:-False}"
FORCED_PREFIX_PROB_START="0.0"
FORCED_PREFIX_PROB_END="0.0"
FORCED_PREFIX_ANNEAL_FRAC="${FORCED_PREFIX_ANNEAL_FRAC:-0.5}"
FORCED_PREFIX_TEMP="${FORCED_PREFIX_TEMP:-2.0}"

# Whether to include a KL-to-reference loss term in the actor objective.
# - When False (and `algorithm.use_kl_in_reward=False`), the trainer will not create a
#   reference-policy worker, which saves compute and isolates the GRPO objective.
USE_KL_LOSS="${USE_KL_LOSS:-True}"

# Distributed strategy used for actor/ref/critic/reward-model training.
# - Use `fsdp` (FSDP1) for the simplest, most battle-tested path.
# - Use `fsdp2` (PyTorch fully_shard APIs) for newer optimized sharding.
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}"
case "${FSDP_STRATEGY}" in
    fsdp|fsdp2) ;;
    *)
        echo "Invalid FSDP_STRATEGY='${FSDP_STRATEGY}'; defaulting to 'fsdp'." >&2
        FSDP_STRATEGY="fsdp"
        ;;
esac

# Remove-padding optimization (flash-attn varlen) for actor/ref forward passes.
# - Useful for speed/memory, but can complicate debugging.
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"

echo "[CONFIG] USE_KL_LOSS=${USE_KL_LOSS}"
echo "[CONFIG] FSDP_STRATEGY=${FSDP_STRATEGY}"
echo "[CONFIG] USE_REMOVE_PADDING=${USE_REMOVE_PADDING}"
echo "[CONFIG] MODEL_PATH=${MODEL_PATH}"
echo "[CONFIG] TOKENIZER_PATH=${TOKENIZER_PATH:-<model>}"
echo "[CONFIG] USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE}"
echo "[CONFIG] RESPONSE_PREFIX_ENABLE=${RESPONSE_PREFIX_ENABLE}"
echo "[CONFIG] RESPONSE_PREFIX=${RESPONSE_PREFIX}"
echo "[CONFIG] TRAINER_NNODES=${TRAINER_NNODES}"
echo "[CONFIG] FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE}"
echo "[CONFIG] ALLOWED_MOVE_ELIM_ENABLE=${ALLOWED_MOVE_ELIM_ENABLE}"
echo "[CONFIG] DATA_SEED=${DATA_SEED}"
echo "[CONFIG] PASS_K_TRAINING=${PASS_K_TRAINING}"
if [ "${PASS_K_TRAINING}" = "True" ]; then
    echo "[CONFIG] PASS_K=${PASS_K}"
fi
echo "[CONFIG] DIVERSITY_ENABLE=${DIVERSITY_ENABLE}"
if [ "${DIVERSITY_ENABLE}" = "True" ]; then
    echo "[CONFIG] DIVERSITY_METHOD=${DIVERSITY_METHOD}"
    echo "[CONFIG] DIVERSITY_LAMBDA=${DIVERSITY_LAMBDA}"
    echo "[CONFIG] DIVERSITY_DISTINCT_K=${DIVERSITY_DISTINCT_K}"
    echo "[CONFIG] DIVERSITY_INCLUDE_BASE_ADVANTAGE=${DIVERSITY_INCLUDE_BASE_ADVANTAGE}"
fi
if [ "${ALLOWED_MOVE_ELIM_ENABLE}" = "True" ]; then
    echo "[CONFIG] ALLOWED_MOVE_ELIM_TEMPLATE_PATH=${ALLOWED_MOVE_ELIM_TEMPLATE_PATH}"
    echo "[CONFIG] ALLOWED_MOVE_ELIM_UID_MODE=${ALLOWED_MOVE_ELIM_UID_MODE}"
    echo "[CONFIG] ALLOWED_MOVE_ELIM_NO_SUCCESS_POLICY=${ALLOWED_MOVE_ELIM_NO_SUCCESS_POLICY}"
    echo "[CONFIG] ALLOWED_MOVE_ELIM_R_MAX_START=${ALLOWED_MOVE_ELIM_R_MAX_START}"
    echo "[CONFIG] ALLOWED_MOVE_ELIM_R_MAX_END=${ALLOWED_MOVE_ELIM_R_MAX_END}"
    echo "[CONFIG] ALLOWED_MOVE_ELIM_ANNEAL_FRAC=${ALLOWED_MOVE_ELIM_ANNEAL_FRAC}"
    echo "[CONFIG] ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB=${ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB}"
    echo "[CONFIG] ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI=${ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI}"
    echo "[CONFIG] ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN=${ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN}"
    echo "[CONFIG] ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_DUMP_MAX_GROUPS=${ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_DUMP_MAX_GROUPS}"
    echo "[CONFIG] ALLOWED_MOVE_ELIM_GAIN_THRESHOLD=${ALLOWED_MOVE_ELIM_GAIN_THRESHOLD}"
    echo "[CONFIG] ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL=${ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL}"
    echo "[CONFIG] ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL=${ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL}"
    if [ "${ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL}" = "True" ]; then
        echo "[CONFIG] ALLOWED_MOVE_ELIM_PASS_K_K=${ALLOWED_MOVE_ELIM_PASS_K_K}"
    fi
fi

# Optional Ray connection overrides (useful for multi-node Slurm launches).
# - When unset, Ray will default to a local cluster.
RAY_ADDRESS="${RAY_ADDRESS:-}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-}"
RAY_TMPDIR="${RAY_TMPDIR:-}"
RAY_INIT_ARGS=()
if [ -n "${RAY_ADDRESS}" ]; then
    RAY_INIT_ARGS+=(+ray_kwargs.ray_init.address="${RAY_ADDRESS}")
else
    if [ -n "${RAY_NUM_CPUS}" ]; then
        RAY_INIT_ARGS+=(ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}")
    fi
    if [ -n "${RAY_TMPDIR}" ]; then
        RAY_INIT_ARGS+=(+ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}")
    fi
fi
if [ -n "${RAY_ADDRESS}" ] || [ -n "${RAY_NUM_CPUS}" ] || [ -n "${RAY_TMPDIR}" ]; then
    echo "[CONFIG] RAY_ADDRESS=${RAY_ADDRESS:-<unset>}"
    echo "[CONFIG] RAY_NUM_CPUS=${RAY_NUM_CPUS:-<unset>}"
    echo "[CONFIG] RAY_TMPDIR=${RAY_TMPDIR:-<unset>}"
fi
# Ensure Ray workers see W&B mode when training is distributed.
# NOTE: Do NOT pass `WANDB_API_KEY` via Ray `runtime_env.env_vars`:
# - it leaks into the Hydra config (synced `config.yaml`) and can show up in logs.
# - rely on normal environment propagation (Slurm/container already export WANDB_API_KEY).
if [ -n "${WANDB_MODE:-}" ]; then
    RAY_INIT_ARGS+=(+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE="${WANDB_MODE}")
fi

# Chess reward shaping (selection framing).
# - Selection correctness is subset-relative (μ-best among the current `considered_moves_uci`).
# - Keep the reward mode identical across baseline and allowed_move_elim runs unless explicitly overridden.
if [ -z "${CHESS_REWARD_FN:-}" ]; then
    CHESS_REWARD_FN="expected_score_wdl_vs_best"
fi

# Validation control:
# - With a large eval parquet (e.g., Searchless 10k), `val_before_train=True` can dominate runtime.
# - For quick smoke/verification runs, set `VAL_BEFORE_TRAIN=False`.
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-20}"
# Actor-update warmup (steps): useful for lightweight “reward-only” smoke runs.
TRAINER_CRITIC_WARMUP="${TRAINER_CRITIC_WARMUP:-0}"

# Checkpointing / resume
TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:--1}"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
CHESS_RL_HF_UPLOAD_MAX_WORKERS="${CHESS_RL_HF_UPLOAD_MAX_WORKERS:-}"
CHESS_RL_MAX_ACTOR_CKPT_TO_KEEP="${CHESS_RL_MAX_ACTOR_CKPT_TO_KEEP:-}"

# Optional: filter_groups rejected-group logging (debug).
# - These knobs are consumed by `verl/trainer/ppo/ray_trainer.py`.
# - Default to enabled; override to empty/0 to disable.
# - If VERL_BASE_DIR is set (cluster runs), default logs under its rollout/ subtree so W&B picks them up.
REJECTED_ROLLOUT_DATA_DIR="${REJECTED_ROLLOUT_DATA_DIR:-}"
if [ -z "${REJECTED_ROLLOUT_DATA_DIR}" ] && [ -n "${VERL_BASE_DIR:-}" ]; then
    REJECTED_ROLLOUT_DATA_DIR="${VERL_BASE_DIR%/}/rollout/rejected_rollout_logs"
fi
REJECTED_GROUP_SUMMARY_MAX_GROUPS_PER_STEP="${REJECTED_GROUP_SUMMARY_MAX_GROUPS_PER_STEP:--1}"
REJECTED_ROLLOUT_MAX_GROUPS_PER_STEP="${REJECTED_ROLLOUT_MAX_GROUPS_PER_STEP:--1}"

# Optional: override actor checkpoint contents (useful for cluster-specific I/O/memory constraints).
# Format: must be a Hydra-parsable list literal, e.g. "['model','extra']" or "['model','optimizer','extra']".
# - Default (empty): use the repo config defaults (save model+optimizer+extra).
# - When set, we also default load_contents to the same list unless explicitly overridden.
ACTOR_CKPT_SAVE_CONTENTS="${ACTOR_CKPT_SAVE_CONTENTS:-}"
ACTOR_CKPT_LOAD_CONTENTS="${ACTOR_CKPT_LOAD_CONTENTS:-}"

# Full-game evaluation control (optional):
# - Uses vLLM + python-chess + Stockfish to play full games starting from the initial position.
# - Enabled via `FULL_EVAL_FREQ` (distinct from `trainer.test_freq`).
# - When checkpoint saving is enabled (`TRAINER_SAVE_FREQ > 0`), default to running full-game eval
#   at the same cadence as checkpoint saves (so evals line up with saved checkpoints).
# - If you want full-game eval without saving (e.g., local smoke runs with `TRAINER_SAVE_FREQ=-1`),
#   set `FULL_EVAL_FREQ` explicitly.
FULL_EVAL_FREQ="${FULL_EVAL_FREQ:-}"
if [ -z "${FULL_EVAL_FREQ}" ]; then
    if [[ "${TRAINER_SAVE_FREQ}" =~ ^-?[0-9]+$ ]] && [ "${TRAINER_SAVE_FREQ}" -gt 0 ]; then
        FULL_EVAL_FREQ="${TRAINER_SAVE_FREQ}"
    else
        FULL_EVAL_FREQ="-1"
    fi
fi
if [[ "${TRAINER_SAVE_FREQ}" =~ ^-?[0-9]+$ ]] && [ "${TRAINER_SAVE_FREQ}" -gt 0 ]; then
    if [[ "${FULL_EVAL_FREQ}" =~ ^-?[0-9]+$ ]] && [ "${FULL_EVAL_FREQ}" -gt 0 ] && [ "${FULL_EVAL_FREQ}" -ne "${TRAINER_SAVE_FREQ}" ]; then
        echo "[WARN] FULL_EVAL_FREQ (${FULL_EVAL_FREQ}) != TRAINER_SAVE_FREQ (${TRAINER_SAVE_FREQ}); forcing FULL_EVAL_FREQ=${TRAINER_SAVE_FREQ}." >&2
        FULL_EVAL_FREQ="${TRAINER_SAVE_FREQ}"
    fi
fi
FULL_EVAL_OPPONENT_DEPTHS="${FULL_EVAL_OPPONENT_DEPTHS:-1,5}"   # comma-separated ints, e.g. "1,5"
# Competition-style defaults:
# - 1 round * 50 games = 50 games per depth.
FULL_EVAL_ROUNDS="${FULL_EVAL_ROUNDS:-1}"
FULL_EVAL_GAMES_PER_ROUND="${FULL_EVAL_GAMES_PER_ROUND:-50}"
FULL_EVAL_GAMES_PER_DEPTH="${FULL_EVAL_GAMES_PER_DEPTH:-}"
if [ -z "${FULL_EVAL_GAMES_PER_DEPTH}" ]; then
    FULL_EVAL_GAMES_PER_DEPTH="$((FULL_EVAL_ROUNDS * FULL_EVAL_GAMES_PER_ROUND))"
fi
# Align full-game generation sampling with validation defaults unless explicitly overridden.
FULL_EVAL_TEMPERATURE="${FULL_EVAL_TEMPERATURE:-${VAL_TEMPERATURE}}"
FULL_EVAL_TOP_P="${FULL_EVAL_TOP_P:-${VAL_TOP_P}}"
FULL_EVAL_MAX_RESPONSE_TOKENS="${FULL_EVAL_MAX_RESPONSE_TOKENS:-${MAX_RESPONSE_LENGTH}}"
FULL_EVAL_MAX_RETRIES_PER_TURN="${FULL_EVAL_MAX_RETRIES_PER_TURN:-1}"
# Full-game eval prompt template:
# - Default to matching the training prompt contract.
#   - When `ALLOWED_MOVE_ELIM_ENABLE=True`, full-game eval should use the selection template
#     (same as training) rather than the legacy "<guess>" scheme.
#   - Otherwise, use the baseline Chess-R1 prompt (no "<guess>").
# - Override explicitly (e.g. to use the legacy "<guess>" scheme) by setting
#   `FULL_EVAL_PROMPT_TEMPLATE_PATH=...` in the environment.
FULL_EVAL_PROMPT_TEMPLATE_PATH="${FULL_EVAL_PROMPT_TEMPLATE_PATH:-}"
if [ -z "${FULL_EVAL_PROMPT_TEMPLATE_PATH}" ]; then
    if [ "${ALLOWED_MOVE_ELIM_ENABLE}" = "True" ]; then
        if [[ "${ALLOWED_MOVE_ELIM_TEMPLATE_PATH}" == /* ]]; then
            FULL_EVAL_PROMPT_TEMPLATE_PATH="${ALLOWED_MOVE_ELIM_TEMPLATE_PATH}"
        else
            FULL_EVAL_PROMPT_TEMPLATE_PATH="${PROJECT_DIR%/}/${ALLOWED_MOVE_ELIM_TEMPLATE_PATH}"
        fi
    elif [ "${MODEL_IS_QWEN3_BASE}" = "True" ]; then
        FULL_EVAL_PROMPT_TEMPLATE_PATH="${PROJECT_DIR}/recipe/chess/prompt_templates/original_chessr1_prompt_base.jinja"
    else
        FULL_EVAL_PROMPT_TEMPLATE_PATH="${PROJECT_DIR}/recipe/chess/prompt_templates/original_chessr1_prompt.jinja"
    fi
fi
echo "[CONFIG] FULL_EVAL_FREQ=${FULL_EVAL_FREQ}"
echo "[CONFIG] FULL_EVAL_PROMPT_TEMPLATE_PATH=${FULL_EVAL_PROMPT_TEMPLATE_PATH}"
FULL_EVAL_OPPONENT_DEPTHS_LIST="[${FULL_EVAL_OPPONENT_DEPTHS}]"
FULL_EVAL_STOCKFISH_PATH="${FULL_EVAL_STOCKFISH_PATH:-.third_party_cache/stockfish/src/stockfish}"
FULL_EVAL_STOCKFISH_THREADS="${FULL_EVAL_STOCKFISH_THREADS:-2}"
FULL_EVAL_STOCKFISH_HASH_MB="${FULL_EVAL_STOCKFISH_HASH_MB:-128}"
# ACPL parallelism (optional; defaults keep local runs lightweight).
FULL_EVAL_ACPL_WORKERS="${FULL_EVAL_ACPL_WORKERS:-72}"
FULL_EVAL_ACPL_THREADS="${FULL_EVAL_ACPL_THREADS:-${FULL_EVAL_STOCKFISH_THREADS}}"

# Hardcoded knobs (research-mode).
PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-False}"
# Allow toggling param offload separately for actor vs reference model.
# - Default to the cluster-tested split:
#     actor.fsdp_config.param_offload = False
#     ref.fsdp_config.param_offload   = True
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-False}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-${OPTIMIZER_OFFLOAD}}"
REF_OPTIMIZER_OFFLOAD="${REF_OPTIMIZER_OFFLOAD:-${OPTIMIZER_OFFLOAD}}"
ACTOR_OFFLOAD_POLICY="${ACTOR_OFFLOAD_POLICY:-False}"
REF_OFFLOAD_POLICY="${REF_OFFLOAD_POLICY:-False}"
MODEL_ENABLE_ACTIVATION_OFFLOAD="${CHESS_RL_MODEL_ENABLE_ACTIVATION_OFFLOAD:-${MODEL_ENABLE_ACTIVATION_OFFLOAD:-False}}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-8}"
REF_LOGPROB_MICRO_BATCH_SIZE="${REF_LOGPROB_MICRO_BATCH_SIZE:-8}"
ACTOR_USE_DYNAMIC_BSZ="${CHESS_RL_ACTOR_USE_DYNAMIC_BSZ:-${ACTOR_USE_DYNAMIC_BSZ:-False}}"
PPO_MAX_TOKEN_LEN_PER_GPU="${CHESS_RL_PPO_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU:-}}"
ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ="${CHESS_RL_ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ:-${ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ:-}}"
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${CHESS_RL_ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-}}"
REF_LOG_PROB_USE_DYNAMIC_BSZ="${CHESS_RL_REF_LOG_PROB_USE_DYNAMIC_BSZ:-${REF_LOG_PROB_USE_DYNAMIC_BSZ:-${ACTOR_USE_DYNAMIC_BSZ}}}"
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${CHESS_RL_REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}}"
ACTOR_LR="${CHESS_RL_LR:-${LR:-1e-6}}"
LR_WARMUP_STEPS="${CHESS_RL_LR_WARMUP_STEPS:-${LR_WARMUP_STEPS:-10}}"
LR_WARMUP_STEPS_RATIO="${CHESS_RL_LR_WARMUP_STEPS_RATIO:-${LR_WARMUP_STEPS_RATIO:--1}}"

# vLLM reserves KV cache based on total GPU memory (not free memory). If GPUs are shared
# (common on workstations), lowering this avoids vLLM refusing to start.
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.7}"
# vLLM maximum concurrent sequences.
# - Cluster runs historically used 2048.
# - Local single-GPU runs often need a much smaller value (e.g., 64–256) to avoid OOM during warmup.
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2048}"

# Rollout tensor-parallelism. Hardcode TP=1 for cluster simplicity.
ROLLOUT_NAME="${CHESS_RL_ROLLOUT_NAME:-${ROLLOUT_NAME:-vllm}}"
TENSOR_MODEL_PARALLEL_SIZE="${CHESS_RL_TENSOR_MODEL_PARALLEL_SIZE:-${TENSOR_MODEL_PARALLEL_SIZE:-1}}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-0}"

resolve_parquet_inputs() {
    # Resolve a dataset split into a Hydra-friendly value for `data.{train,val}_files`:
    # - If `<prefix>.parquet` exists, return that single path.
    # - Else, if `<prefix>_[0-9]*.parquet` exists, return a Hydra list like:
    #     [/path/prefix_0.parquet,/path/prefix_1.parquet]
    # This allows sharding large parquets to stay under GitHub's 100MB per-file limit.
    local data_dir="$1"
    local prefix="$2"
    local single="${data_dir}/${prefix}.parquet"
    if [ -f "${single}" ]; then
        echo "${single}"
        return 0
    fi

    shopt -s nullglob
    # Only treat numeric-suffixed shards as part of the split.
    # This avoids accidental mixing across similarly-prefixed files, e.g.:
    #   - prefix=train      should NOT match train_hard_0.parquet
    #   - prefix=test       should NOT match test_shuffled_legal_moves.parquet
    local shards=( "${data_dir}/${prefix}_"[0-9]*.parquet )
    shopt -u nullglob
    if [ "${#shards[@]}" -le 0 ]; then
        echo "Missing dataset files for prefix='${prefix}' in CHESS_DATA_DIR='${data_dir}'" >&2
        echo "Expected either '${single}' or '${data_dir}/${prefix}_[0-9]*.parquet'" >&2
        return 1
    fi

    # Sort for determinism.
    local sorted
    IFS=$'\n' sorted=($(printf '%s\n' "${shards[@]}" | sort))
    local out="[${sorted[0]}"
    for f in "${sorted[@]:1}"; do
        out+=",${f}"
    done
    out+="]"
    echo "${out}"
    return 0
}

if [ "$USE_CRITIC_DATASET" = "True" ]; then
    if [ "$USE_HARD_DATASET" = "True" ]; then
        echo "Invalid config: USE_CRITIC_DATASET=True and USE_HARD_DATASET=True (no *_hard_critic.parquet)." >&2
        exit 1
    fi
    train_files="${CHESS_DATA_DIR}/train_critic.parquet"
else
    if [ "${CHESS_SELF_PLAY_ENABLE}" = "True" ]; then
        # Training positions are generated online; keep train_files empty so the
        # prompt guardrail check (below) no-ops for the train split.
        train_files="[]"
    else
        if [ "$USE_HARD_DATASET" = "True" ]; then
            train_files="$(resolve_parquet_inputs "${CHESS_DATA_DIR}" "train_hard")"
        else
            train_files="$(resolve_parquet_inputs "${CHESS_DATA_DIR}" "train")"
        fi
    fi
fi
if [ -z "${CHESS_RL_VAL_FILES}" ]; then
    echo "ERROR: CHESS_RL_VAL_FILES must be set explicitly (single parquet path or Hydra list)." >&2
    echo "Example:" >&2
    echo "  CHESS_RL_VAL_FILES='[data/.../test.parquet,data/.../test_shuffled_legal_moves.parquet]'" >&2
    exit 1
fi
test_files="${CHESS_RL_VAL_FILES}"

# Guardrail: fail fast if the dataset prompt is not in the expected selection scheme.
# This avoids silent runs where *every* rollout gets reward=-1 due to `format_error`
# or where prompts don't include a considered-moves candidate list.
python3 - <<'PY' "$train_files" "$test_files" "${JOB_NAME_LC:-}"
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

ANSWER_RE = re.compile(r"<\s*answer\s*>", re.IGNORECASE)
UCI_MOVE_RE = re.compile(r"<\s*uci_move\s*>", re.IGNORECASE)
GUESS_RE = re.compile(r"<\s*guess\s*>", re.IGNORECASE)
# This prompt check is intentionally loose:
# - We historically used "considered_moves" language in prompts.
# - We now prefer the clearer "allowed_moves" language, while the underlying Jinja variable remains
#   `considered_moves_uci_list` for backward compatibility with eval code.
CONSIDERED_RE = re.compile(
    r"(considered_moves|allowed_moves|legal\s+moves|Considered\s+moves\s*\(UCI\)|Allowed\s+moves\s*\(UCI\)|Legal\s+moves\s*\(UCI\))",
    re.IGNORECASE,
)


def _parse_files(spec: str) -> list[str]:
    spec = (spec or "").strip()
    if spec.startswith("[") and spec.endswith("]"):
        inner = spec[1:-1].strip()
        if not inner:
            return []
        return [p.strip() for p in inner.split(",") if p.strip()]
    return [spec] if spec else []


def _check_one(path_str: str, *, label: str) -> None:
    p = Path(path_str)
    if not p.exists():
        return
    pf = pq.ParquetFile(str(p))
    if pf.metadata is None or pf.metadata.num_rows <= 0:
        return
    # Read a tiny slice (first row-group, first 4 rows) to keep this guardrail cheap.
    table = pf.read_row_group(0, columns=["prompt"])
    prompts = table.column("prompt").to_pylist()[:4]
    for prompt in prompts:
        try:
            msgs = list(prompt) if not isinstance(prompt, dict) else [prompt]
        except TypeError:
            msgs = []
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            content = str(msg.get("content") or "")
            has_answer = bool(ANSWER_RE.search(content))
            has_uci_move = bool(UCI_MOVE_RE.search(content))
            has_guess = bool(GUESS_RE.search(content))

            if has_answer:
                raise SystemExit(
                    f"ERROR: {label} dataset prompt contains legacy <answer> tags: {p}\n\n"
                    "This repo no longer supports the <answer> output contract.\n"
                    "Fix options:\n"
                    "  1) Use a Chess-R1-aligned dataset dir (default): data/chess_puzzles_chessr1_aligned_sharded\n"
                    "  2) Rewrite prompts to the strict <uci_move> scheme:\n"
                    f"     python scripts/rewrite_chess_prompts_to_uci_move.py --input_parquet {p} --output_parquet <out> --overwrite\n"
                )

            if not has_uci_move:
                raise SystemExit(
                    f"ERROR: {label} dataset prompt is missing <uci_move> tags: {p}\n\n"
                    "This repo expects prompts to request <uci_move> output (not SAN / not <answer>).\n"
                )

            if has_guess:
                raise SystemExit(
                    f"ERROR: {label} dataset prompt still contains legacy <guess> tags: {p}\n\n"
                    "Selection framing uses the strict reasoning-tag + `<uci_move>...</uci_move>` contract only.\n"
                    "Fix options:\n"
                    "  1) Rebuild the selection dataset prompts from the selection template:\n"
                    f"     python scripts/build_chess_select_test_dataset.py --input_parquet {p} --output_parquet <out> --overwrite\n"
                )

            if not is_baseline_job:
                has_candidate_list = bool(CONSIDERED_RE.search(content))
                if not has_candidate_list:
                    raise SystemExit(
                        f"ERROR: {label} dataset prompt is missing candidate-move-list instructions: {p}\n\n"
                        "Selection training/eval requires prompts to include a candidate move list "
                        "(allowed_moves / considered_moves / legal moves).\n"
                        "Fix options:\n"
                        "  1) Rebuild the selection dataset with `scripts/build_chess_select_*_dataset.py`.\n"
                    )


train_spec, val_spec = sys.argv[1], sys.argv[2]
job_name = (sys.argv[3] if len(sys.argv) > 3 else "").lower()
is_baseline_job = "baseline" in job_name if job_name else False
for f in _parse_files(train_spec)[:1]:
    _check_one(f, label="train")
for f in _parse_files(val_spec)[:1]:
    _check_one(f, label="val")
PY

# Long, explicit experiment name (used for W&B run naming).
# Local outputs (checkpoints/rollout logs) are stored under `outputs/{config_hash}` by default
# (see `verl/trainer/main_ppo.py`), so the experiment name no longer needs to be filesystem-friendly.
MODEL_TAG="$(basename "${MODEL_PATH}")"
MODEL_TAG="$(printf '%s' "${MODEL_TAG}" | tr -c '[:alnum:]_.-' '_')"
EXPERIMENT_NAME="chess_grpo"\
"__model_${MODEL_TAG}"\
"__chat_template_${USE_CHAT_TEMPLATE}"\
"__response_prefix_${RESPONSE_PREFIX_ENABLE}"\
"__reward_${CHESS_REWARD_FN}"\
"__ame_${ALLOWED_MOVE_ELIM_ENABLE}"\
"__selfplay_${CHESS_SELF_PLAY_ENABLE}"\
"__use_hard_${USE_HARD_DATASET}"\
"__use_critic_${USE_CRITIC_DATASET}"\
"__anneal${FORCED_PREFIX_PROB_START}_${FORCED_PREFIX_PROB_END}_${FORCED_PREFIX_ANNEAL_FRAC}"\
"__forced_temp_${FORCED_PREFIX_TEMP}"\
"__rollout_n_${ROLLOUT_N}"\
"__kl_${USE_KL_LOSS}"\
"__actor_po_${ACTOR_PARAM_OFFLOAD}"\
"__ref_po_${REF_PARAM_OFFLOAD}"\

SELF_PLAY_ARGS=()
if [ "${CHESS_SELF_PLAY_ENABLE}" = "True" ]; then
    # Default dump dir: keep under VERL_BASE_DIR on cluster so W&B can sync it; otherwise use repo outputs/.
    SELF_PLAY_DUMP_DIR="${CHESS_SELF_PLAY_DUMP_DIR:-}"
    if [ -z "${SELF_PLAY_DUMP_DIR}" ]; then
        if [ -n "${VERL_BASE_DIR:-}" ]; then
            SELF_PLAY_DUMP_DIR="${VERL_BASE_DIR%/}/rollout/self_play_batches"
        else
            SELF_PLAY_DUMP_DIR="${PROJECT_DIR}/outputs/self_play_batches"
        fi
    fi
    if [[ "${SELF_PLAY_DUMP_DIR}" != /* ]]; then
        SELF_PLAY_DUMP_DIR="${PROJECT_DIR%/}/${SELF_PLAY_DUMP_DIR#./}"
    fi
    SELF_PLAY_DUMP_STEPS="${CHESS_SELF_PLAY_DUMP_STEPS:-[1]}"

    SELF_PLAY_ARGS+=(
        data.custom_cls.path="${PROJECT_DIR}/recipe/chess/self_play_stub_dataset.py"
        data.custom_cls.name=SelfPlayStubDataset
        data.self_play.enable=True
        data.self_play.num_parallel_games="${TRAIN_BATCH_SIZE}"
        data.self_play.opponent_depth=1
        data.self_play.analysis_depth=1
        data.self_play.max_retries_per_turn=3
        data.self_play.stockfish_path="${FULL_EVAL_STOCKFISH_PATH}"
        data.self_play.stockfish_threads="${FULL_EVAL_STOCKFISH_THREADS}"
        data.self_play.stockfish_hash_mb="${FULL_EVAL_STOCKFISH_HASH_MB}"
        data.self_play.template_path="${ALLOWED_MOVE_ELIM_TEMPLATE_PATH}"
        data.self_play.dump_dir="${SELF_PLAY_DUMP_DIR}"
        data.self_play.dump_steps="${SELF_PLAY_DUMP_STEPS}"
        data.self_play.sampling_kwargs.temperature="${VAL_TEMPERATURE}"
        data.self_play.sampling_kwargs.top_p="${VAL_TOP_P}"
        data.self_play.sampling_kwargs.top_k=-1
        data.self_play.sampling_kwargs.detokenize=True
    )
fi

VLLM_DISABLE_COMPILE_CACHE=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${ADV_ESTIMATOR:-grpo} \
    ${REF_MODEL_PATH:++actor_rollout_ref.ref.model.path=${REF_MODEL_PATH}} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    +data.gen_batch_size=${GEN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.use_chat_template=${USE_CHAT_TEMPLATE} \
    +algorithm.filter_groups._target_=verl.trainer.config.FilterGroupsConfig \
    +algorithm.filter_groups.enable=${FILTER_GROUPS_ENABLE} \
    +algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC} \
    +algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES} \
    +algorithm.allowed_move_elim.enable=${ALLOWED_MOVE_ELIM_ENABLE} \
    +algorithm.allowed_move_elim.template_path=${ALLOWED_MOVE_ELIM_TEMPLATE_PATH} \
    +algorithm.allowed_move_elim.uid_mode=${ALLOWED_MOVE_ELIM_UID_MODE} \
    +algorithm.allowed_move_elim.no_success_policy=${ALLOWED_MOVE_ELIM_NO_SUCCESS_POLICY} \
    +algorithm.allowed_move_elim.r_max_start=${ALLOWED_MOVE_ELIM_R_MAX_START} \
    +algorithm.allowed_move_elim.r_max_end=${ALLOWED_MOVE_ELIM_R_MAX_END} \
	    +algorithm.allowed_move_elim.anneal_frac=${ALLOWED_MOVE_ELIM_ANNEAL_FRAC} \
	    +algorithm.allowed_move_elim.stitch_round0_prompt_for_logprob=${ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB} \
	    +algorithm.allowed_move_elim.force_use_considered_moves_uci=${ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI} \
	    +algorithm.allowed_move_elim.group_reward_range_min=${ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN} \
	    +algorithm.allowed_move_elim.group_reward_range_dump_max_groups=${ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_DUMP_MAX_GROUPS} \
	    +algorithm.allowed_move_elim.gain_threshold=${ALLOWED_MOVE_ELIM_GAIN_THRESHOLD} \
	    +algorithm.allowed_move_elim.pass_k_when_no_optimal=${ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL} \
	    +algorithm.allowed_move_elim.pass_k_k=${ALLOWED_MOVE_ELIM_PASS_K_K} \
	    +algorithm.allowed_move_elim.diversity_when_no_optimal=${ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL} \
	    algorithm.pass_k_training=${PASS_K_TRAINING} \
	    algorithm.pass_k_k=${PASS_K} \
	    algorithm.diversity.enable=${DIVERSITY_ENABLE} \
	    algorithm.diversity.method=${DIVERSITY_METHOD} \
	    algorithm.diversity.lambda_coeff=${DIVERSITY_LAMBDA} \
	    algorithm.diversity.distinct_k=${DIVERSITY_DISTINCT_K} \
	    algorithm.diversity.include_base_advantage=${DIVERSITY_INCLUDE_BASE_ADVANTAGE} \
	    custom_reward_function.path=${PROJECT_DIR}/recipe/chess/reward_fn.py \
    custom_reward_function.name=compute_score_batch \
    custom_reward_function.reward_kwargs.chess_reward_fn=${CHESS_REWARD_FN} \
    ${CHESS_GT_EXPECTED_SCORE_DIFF_THRESHOLD:++custom_reward_function.reward_kwargs.gt_expected_score_diff_threshold=${CHESS_GT_EXPECTED_SCORE_DIFF_THRESHOLD}} \
    forced_prefix.enable=False \
    data.reward_fn_key=reward_model \
    reward_model.reward_manager=batch \
    actor_rollout_ref.actor.strategy=${FSDP_STRATEGY} \
    actor_rollout_ref.actor.fsdp_config.strategy=${FSDP_STRATEGY} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    ${TOKENIZER_PATH:+actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH}} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO} \
    actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=${ACTOR_USE_DYNAMIC_BSZ} \
    ${PPO_MAX_TOKEN_LEN_PER_GPU:+actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}} \
    ${ACTOR_CKPT_SAVE_CONTENTS:+actor_rollout_ref.actor.checkpoint.save_contents=${ACTOR_CKPT_SAVE_CONTENTS}} \
    ${ACTOR_CKPT_SAVE_CONTENTS:+actor_rollout_ref.actor.checkpoint.load_contents=${ACTOR_CKPT_LOAD_CONTENTS:-${ACTOR_CKPT_SAVE_CONTENTS}}} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=${MODEL_ENABLE_ACTIVATION_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.offload_policy=${ACTOR_OFFLOAD_POLICY} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE} \
    ${ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ:+actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ}} \
    ${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:+actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_seqs=${VLLM_MAX_NUM_SEQS} \
    ${ROLLOUT_MAX_MODEL_LEN:+actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
    actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE} \
    actor_rollout_ref.rollout.response_prefix_enable=${RESPONSE_PREFIX_ENABLE} \
    "actor_rollout_ref.rollout.response_prefix='${RESPONSE_PREFIX}'" \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${REF_LOG_PROB_USE_DYNAMIC_BSZ} \
    ${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:+actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}} \
    actor_rollout_ref.ref.strategy=${FSDP_STRATEGY} \
    actor_rollout_ref.ref.fsdp_config.strategy=${FSDP_STRATEGY} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD} \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=${REF_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.ref.fsdp_config.offload_policy=${REF_OFFLOAD_POLICY} \
    critic.strategy=${FSDP_STRATEGY} \
    critic.model.fsdp_config.strategy=${FSDP_STRATEGY} \
    critic.model.enable_activation_offload=True \
    reward_model.strategy=${FSDP_STRATEGY} \
    reward_model.model.fsdp_config.strategy=${FSDP_STRATEGY} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=${TRAINER_CRITIC_WARMUP} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='chess_rl' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE} \
    trainer.nnodes=${TRAINER_NNODES} \
    trainer.save_freq=${TRAINER_SAVE_FREQ} \
    trainer.test_freq=${TRAINER_TEST_FREQ} \
    trainer.full_eval_freq=${FULL_EVAL_FREQ} \
    trainer.full_eval.opponent_depths="${FULL_EVAL_OPPONENT_DEPTHS_LIST}" \
    trainer.full_eval.games_per_depth=${FULL_EVAL_GAMES_PER_DEPTH} \
    trainer.full_eval.rounds=${FULL_EVAL_ROUNDS} \
    trainer.full_eval.games_per_round=${FULL_EVAL_GAMES_PER_ROUND} \
    trainer.full_eval.temperature=${FULL_EVAL_TEMPERATURE} \
    trainer.full_eval.top_p=${FULL_EVAL_TOP_P} \
    trainer.full_eval.max_response_tokens=${FULL_EVAL_MAX_RESPONSE_TOKENS} \
    trainer.full_eval.max_retries_per_turn=${FULL_EVAL_MAX_RETRIES_PER_TURN} \
    trainer.full_eval.prompt_template_path=${FULL_EVAL_PROMPT_TEMPLATE_PATH} \
    trainer.full_eval.stockfish_path=${FULL_EVAL_STOCKFISH_PATH} \
    trainer.full_eval.stockfish_threads=${FULL_EVAL_STOCKFISH_THREADS} \
    trainer.full_eval.stockfish_hash_mb=${FULL_EVAL_STOCKFISH_HASH_MB} \
    trainer.full_eval.acpl_workers=${FULL_EVAL_ACPL_WORKERS} \
    trainer.full_eval.acpl_threads=${FULL_EVAL_ACPL_THREADS} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    data.shuffle=${DATA_SHUFFLE} \
    data.seed=${DATA_SEED} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.log_val_generations=${LOG_VAL_GENERATIONS} \
    ${REJECTED_ROLLOUT_DATA_DIR:+trainer.rejected_rollout_data_dir=${REJECTED_ROLLOUT_DATA_DIR}} \
    trainer.rejected_group_summary_max_groups_per_step=${REJECTED_GROUP_SUMMARY_MAX_GROUPS_PER_STEP} \
    trainer.rejected_rollout_max_groups_per_step=${REJECTED_ROLLOUT_MAX_GROUPS_PER_STEP} \
    trainer.resume_mode=${TRAINER_RESUME_MODE} \
    ${CHESS_RL_HF_UPLOAD_MAX_WORKERS:+ +trainer.hf_upload_max_workers=${CHESS_RL_HF_UPLOAD_MAX_WORKERS}} \
    ${CHESS_RL_MAX_ACTOR_CKPT_TO_KEEP:+ trainer.max_actor_ckpt_to_keep=${CHESS_RL_MAX_ACTOR_CKPT_TO_KEEP}} \
    "${RAY_INIT_ARGS[@]}" \
    "${SELF_PLAY_ARGS[@]}"
