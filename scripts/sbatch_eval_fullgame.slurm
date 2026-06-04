#!/bin/bash
#SBATCH --job-name=chess-eval-fullgame
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# Stage C-2 — Full-game eval vs Stockfish.
# Model plays 50 complete games against Stockfish at a configurable depth.
# Reports W/D/L, ACPL per move, forfeit rate.
#
# Required env:
#   UPSTREAM_DIR     — upstream clone
#   MODEL            — HF-format model dir
#   OUT_DIR          — output dir (e.g. results/distill_step300_fullgame)
#   STOCKFISH_BIN    — path to stockfish binary (default: $UPSTREAM_DIR/.third_party_cache/stockfish/src/stockfish)
# Optional:
#   OPPONENT_DEPTHS  — comma list (default 5)
#   GAMES_PER_DEPTH  — default 50

set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_USE_V1=1

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-chess}"

: "${UPSTREAM_DIR:?UPSTREAM_DIR required}"
: "${MODEL:?MODEL (HF dir) required}"
: "${OUT_DIR:?OUT_DIR required}"
STOCKFISH_BIN="${STOCKFISH_BIN:-${UPSTREAM_DIR}/.third_party_cache/stockfish/src/stockfish}"
OPPONENT_DEPTHS="${OPPONENT_DEPTHS:-5}"
GAMES_PER_DEPTH="${GAMES_PER_DEPTH:-50}"

cd "${UPSTREAM_DIR}"
export PYTHONPATH="${UPSTREAM_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}"

python -m scripts.eval_chess_fullgame \
    --model "${MODEL}" \
    --out-dir "${OUT_DIR}" \
    --opponent-depths ${OPPONENT_DEPTHS} \
    --games-per-depth ${GAMES_PER_DEPTH} \
    --stockfish-path "${STOCKFISH_BIN}" \
    --stockfish-threads 4 \
    --stockfish-skill-level 0 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-response-tokens 2000 \
    --trust-remote-code \
    --seed 0
