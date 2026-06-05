#!/bin/bash
# HTCondor entrypoint: activate `chess` venv and run chesslesson PPO training.
# Submitted via recipe/hgpo/chesslesson_train.sub (2x H100).
set -euo pipefail
set -x

source $HOME/envs/chess/bin/activate
cd $HOME/verl-agent
export PYTHONPATH=$HOME/verl-agent:${PYTHONPATH}

export TMPDIR="${TMPDIR:-$HOME/fast/tmp}"
mkdir -p "${TMPDIR}"
export HF_HOME="${HF_HOME:-$HOME/fast/hf}"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$HOME/fast/checkpoints/verl-agent}"
export WANDB_DIR="${WANDB_DIR:-$HOME/fast/wandb}"
mkdir -p "${HF_HOME}" "${CHECKPOINTS_DIR}" "${WANDB_DIR}"

echo "host: $(hostname)"
echo "python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

exec bash $HOME/verl-agent/recipe/hgpo/run_chesslesson_train.sh "$@"
