#!/bin/bash
#SBATCH --job-name=chesslesson-base-eval
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/chess_rl_outputs/logs/slurm-%x-%j.out
#SBATCH --error=$HOME/chess_rl_outputs/logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# Zero-shot eval of base Qwen2.5-7B-Instruct on chesslesson (175 tasks).
# 1 GPU, ~10-15 min.

set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate chess

cd $HOME/chess/chess-agent
export PYTHONPATH=$HOME/chess/chess-agent:${PYTHONPATH:-}

MODEL="${MODEL:-$HOME/models/Qwen2.5-7B-Instruct}"
OUT_PATH="${OUT_PATH:-eval_results/base_qwen7b_chesslesson_$(date +%Y%m%d_%H%M%S).json}"
N_SAMPLES="${N_SAMPLES:-1}"

mkdir -p "$(dirname "${OUT_PATH}")"

python eval_chesslesson_base.py \
    --model "${MODEL}" \
    --output "${OUT_PATH}" \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-tokens 1024 \
    --n-samples ${N_SAMPLES} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85
