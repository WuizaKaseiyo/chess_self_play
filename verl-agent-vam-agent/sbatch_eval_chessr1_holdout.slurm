#!/bin/bash
#SBATCH --job-name=chessr1-holdout
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# Chess-R1 hold-out eval: phase X model on Chess-R1 puzzles (unseen, same skill).
# Required env: MODEL, PARQUET, OUT_PATH
# Optional:    N_SAMPLES (default 200)

set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-chess}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p logs eval_results

: "${MODEL:?set MODEL=/path/to/hf/model}"
: "${PARQUET:?set PARQUET=/path/to/chess-r1/parquet}"
: "${OUT_PATH:?set OUT_PATH=eval_results/...}"
N_SAMPLES="${N_SAMPLES:-200}"

python eval_chessr1_holdout.py \
    --model "${MODEL}" \
    --parquet "${PARQUET}" \
    --n-samples "${N_SAMPLES}" \
    --output "${OUT_PATH}" \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-tokens 512 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85
