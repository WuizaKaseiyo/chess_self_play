#!/bin/bash
set -euo pipefail

# Minimal example submission for this repo:
# - Model: Qwen/Qwen2.5-7B-Instruct (<8B params)
# - Prompt: our in-repo Chess-R1-aligned prompt scheme (guess-first + strict tags)
#
# Usage (from repo root):
#   bash competition/submission/aicrowd_submit.sh
#
# You can override defaults via environment variables:
#   HF_REPO=Gabr1e11/my-aicrowd-model HF_REPO_TAG=main bash competition/submission/aicrowd_submit.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHALLENGE="${CHALLENGE:-global-chess-challenge-2025}"

# Default to our own gated repo (contains mirrored Qwen2.5-7B-Instruct weights).
# This matters because the competition commonly expects submissions to reference your own HF repo
# (and the starter-kit gating doc requires repo names to include "aicrowd").
HF_REPO="${HF_REPO:-Gabr1e11/chess-rl-aicrowd-qwen2.5-7b-instruct}"
HF_REPO_TAG="${HF_REPO_TAG:-main}"

# vLLM/Neuron runtime knobs (tuned to match this repo's training defaults).
# NOTE: aicrowd-cli passes arbitrary key-value pairs; the backend validates an allowlist.
NUM_GAMES="${NUM_GAMES:-1}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-5120}"       # 1024 prompt + 4096 response
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"            # community workaround: avoid batched decoding weirdness
VLLM_INF_MAX_TOKENS="${VLLM_INF_MAX_TOKENS:-4096}"     # match MAX_RESPONSE_LENGTH in train_chess.sh
VLLM_INF_TEMPERATURE="${VLLM_INF_TEMPERATURE:-0.6}"    # match FULL_EVAL_TEMPERATURE in train_chess.sh
VLLM_INF_TOP_P="${VLLM_INF_TOP_P:-0.95}"               # match FULL_EVAL_TOP_P in train_chess.sh

# Neuron backend needs the correct model family string.
# - For Qwen2.5 weights, the accepted Neuron model type is `qwen2` (NOT `qwen2.5`).
# - For Qwen3 weights, use `qwen3`.
NEURON_MODEL_TYPE="${NEURON_MODEL_TYPE:-qwen2}"

# Use an absolute path so the script works no matter what directory it's run from.
# NOTE: This uses a legacy full-legal prompt template (the starter kit provides only `FEN` and the full
# `legal_moves_uci_list`). The repo's current selection/action-masking prompts use `allowed_moves`
# (`considered_moves_uci_list`): see `recipe/chess/prompt_templates/select_prompt.jinja` and `restricted_moves.md`.
PROMPT_TEMPLATE_DEFAULT="${SCRIPT_DIR}/player_agents/chess_rl_chessr1_prompt.jinja"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-${PROMPT_TEMPLATE_DEFAULT}}"

echo "[submit] challenge=${CHALLENGE}"
echo "[submit] hf_repo=${HF_REPO}"
echo "[submit] hf_repo_tag=${HF_REPO_TAG}"
echo "[submit] prompt_template=${PROMPT_TEMPLATE}"
echo "[submit] neuron.model-type=${NEURON_MODEL_TYPE}"
echo "[submit] num_games=${NUM_GAMES}"
echo "[submit] vllm.dtype=${VLLM_DTYPE}"
echo "[submit] vllm.max-model-len=${VLLM_MAX_MODEL_LEN}"
echo "[submit] vllm.max-num-seqs=${VLLM_MAX_NUM_SEQS}"
echo "[submit] vllm-inference.max-tokens=${VLLM_INF_MAX_TOKENS}"
echo "[submit] vllm-inference.temperature=${VLLM_INF_TEMPERATURE}"
echo "[submit] vllm-inference.top-p=${VLLM_INF_TOP_P}"

# NOTE:
# - The evaluation platform runs on AWS Trainium (Neuron). The submission backend
#   validates `--neuron.model-type` against an allowlist. For Qwen2.5 weights,
#   the correct/accepted Neuron model type is `qwen2` (NOT `qwen2.5`).
# - Keep max tokens small-ish to reduce latency and avoid long `<think>` rambles.

aicrowd submit-model \
  --challenge "${CHALLENGE}" \
  --hf-repo "${HF_REPO}" \
  --hf-repo-tag "${HF_REPO_TAG}" \
  --prompt_template_path "${PROMPT_TEMPLATE}" \
  --neuron.model-type "${NEURON_MODEL_TYPE}" \
  --num-games "${NUM_GAMES}" \
  --vllm.dtype "${VLLM_DTYPE}" \
  --vllm.max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --vllm.max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --vllm-inference.max-tokens "${VLLM_INF_MAX_TOKENS}" \
  --vllm-inference.temperature "${VLLM_INF_TEMPERATURE}" \
  --vllm-inference.top-p "${VLLM_INF_TOP_P}"
