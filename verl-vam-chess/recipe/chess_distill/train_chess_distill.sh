#!/usr/bin/env bash
#
# On-policy distillation launcher for chess: 3B student learns from 7B teacher
# via per-token reverse-KL advantage (Thinking Machines blog).
#
# Required env (set by caller / sbatch wrapper):
#   MODEL_PATH       — student HF model dir (e.g. /home/.../models/Qwen2.5-3B-Instruct)
#   TEACHER_CKPT     — teacher HF model dir (Phase A best ckpt, merged FSDP→HF)
#   CHESS_DATA_DIR   — training data dir (same baseline parquet as Phase A)
#
# Optional (have sane defaults):
#   ROLLOUT_N        — samples per prompt (default 8; distill needs less than GRPO)
#   TRAIN_BATCH_SIZE — default 128
#   TOTAL_TRAINING_STEPS — default 300 (blog claims 5-10× faster than RL)
#   TRAINER_SAVE_FREQ — default 50
#   TRAINER_TEST_FREQ — default 25
#   TRAINER_N_GPUS_PER_NODE — default 4
#
# This script wraps train_chess.sh by exporting distill-specific env vars
# and letting the same hydra cmd-builder produce the launch line.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ---- Required inputs ----
: "${MODEL_PATH:?MODEL_PATH (student HF dir) must be set}"
: "${TEACHER_CKPT:?TEACHER_CKPT (teacher HF dir) must be set}"

# ---- Distill algorithm setup ----
export ADV_ESTIMATOR=distill
export REF_MODEL_PATH="${TEACHER_CKPT}"

# Disable GRPO/Pass@k group structure: distill is per-sample, per-token.
export PASS_K_TRAINING=False
export PASS_K=1
export FILTER_GROUPS_ENABLE=False
export ALLOWED_MOVE_ELIM_ENABLE=False
export DIVERSITY_ENABLE=False
export FORCED_PREFIX_ENABLE=False

# Disable in-reward KL (the distill advantage IS the reverse-KL signal).
# (train_chess.sh hard-codes `algorithm.use_kl_in_reward=False` already.)

# Disable the separate kl_loss term in the actor loss; the distill advantage
# already pushes student towards teacher. Keeping a second KL channel would
# double-count.
export USE_KL_LOSS=False

# Distill uses smaller rollout n than Pass@k GRPO.
export ROLLOUT_N="${ROLLOUT_N:-8}"

# Batch + cadence defaults tuned for ~300-step distill run.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-50}"
export TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-25}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

# 7B teacher as ref → offload its params to CPU when not computing logprobs.
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
export REF_OPTIMIZER_OFFLOAD="${REF_OPTIMIZER_OFFLOAD:-True}"

# Stockfish reward still computed (cheap, parquet lookup) but advantage ignores it.
# Optionally swap CHESS_REWARD_FN to a no-op constant if you want to save the
# few ms of dict lookups — not worth it for now.

# Hand off to the canonical chess launcher.
echo "[distill] STUDENT=${MODEL_PATH}"
echo "[distill] TEACHER=${TEACHER_CKPT}"
echo "[distill] ADV_ESTIMATOR=${ADV_ESTIMATOR}"
echo "[distill] ROLLOUT_N=${ROLLOUT_N}  BATCH=${TRAIN_BATCH_SIZE}  STEPS=${TOTAL_TRAINING_STEPS}"

cd "${PROJECT_DIR}"
exec bash "${PROJECT_DIR}/train_chess.sh"
