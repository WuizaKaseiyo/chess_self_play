#!/bin/bash
#SBATCH --job-name=distill-fullgame-eval
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=$HOME/chess_rl_outputs/logs/slurm-%x-%j.out
#SBATCH --error=$HOME/chess_rl_outputs/logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# Full-game eval: distill 3B step_300 vs Stockfish depth 5, 50 games.
# Required env: MODEL (HF dir), OUT_DIR

set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate chess

PROJECT_DIR=$HOME/chess/vam-chess
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

MODEL="${MODEL:?MODEL (HF dir) must be set}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/eval_results/distill_step300_fullgame}"
OPPONENT_DEPTHS="${OPPONENT_DEPTHS:-5}"
GAMES_PER_DEPTH="${GAMES_PER_DEPTH:-50}"

mkdir -p "${OUT_DIR}"

echo "[fullgame] MODEL=${MODEL}"
echo "[fullgame] OUT_DIR=${OUT_DIR}"
echo "[fullgame] vs Stockfish depth=${OPPONENT_DEPTHS}, games=${GAMES_PER_DEPTH}"

python -m scripts.eval_chess_fullgame \
    --model "${MODEL}" \
    --out-dir "${OUT_DIR}" \
    --opponent-depths ${OPPONENT_DEPTHS} \
    --games-per-depth ${GAMES_PER_DEPTH} \
    --stockfish-path .third_party_cache/stockfish/src/stockfish \
    --stockfish-threads 4 \
    --stockfish-skill-level 0 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-response-tokens 2000 \
    --trust-remote-code \
    --seed 0
