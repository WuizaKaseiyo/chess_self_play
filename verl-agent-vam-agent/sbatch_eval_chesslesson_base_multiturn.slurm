#!/bin/bash
#SBATCH --job-name=chesslesson-base-multiturn
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# Multi-turn zero-shot baseline eval for chesslesson — paradigm-aligned with
# verl-agent HGPO/PPO training. Same LessonStepper + same prompts as the env.

set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_USE_V1=1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-chess}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p logs eval_results

MODEL="${MODEL:-$HOME/models/Qwen2.5-7B-Instruct}"
OUT_PATH="${OUT_PATH:-eval_results/base_qwen7b_chesslesson_multiturn_$(date +%Y%m%d_%H%M%S).json}"

python eval_chesslesson_base_multiturn.py \
    --model "${MODEL}" \
    --output "${OUT_PATH}" \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-tokens 512 \
    --max-turns-cap 12 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85
