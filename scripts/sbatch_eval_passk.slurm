#!/bin/bash
#SBATCH --job-name=chess-eval-passk
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# Stage C-1 — Puzzle pass@k eval.
# Score a checkpoint on the held-out chess puzzle test set with k=1,2,4,8.
#
# Required env:
#   UPSTREAM_DIR  — local clone of upstream framework
#   MODEL         — HF-format model dir (teacher / distill / baseline)
#   OUT_PATH      — output JSON path (e.g. results/teacher_step640_passk.json)

set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_USE_V1=1

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-chess}"

UPSTREAM_DIR="${UPSTREAM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/verl-vam-chess}"
: "${MODEL:?MODEL (HF dir) required}"
: "${OUT_PATH:?OUT_PATH (json) required}"

cd "${UPSTREAM_DIR}"
export PYTHONPATH="${UPSTREAM_DIR}:${PYTHONPATH:-}"

mkdir -p "$(dirname "${OUT_PATH}")"

python -m scripts.eval_chess_passk \
    --model "${MODEL}" \
    --parquet data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet \
    --out-json "${OUT_PATH}" \
    --k-max 8 \
    --batch-size 32 \
    --max-prompt-length 1024 \
    --max-response-length 2000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --temperature 0.6 \
    --top-p 0.95 \
    --trust-remote-code \
    --seed 0
