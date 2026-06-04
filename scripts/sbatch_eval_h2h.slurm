#!/bin/bash
#SBATCH --job-name=chess-eval-h2h
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# Stage C-3 — Model-vs-model head-to-head (e.g. distill vs teacher).
# Both models loaded on separate GPUs (1 each, TP=1) and play 100 full games.
# Reports W/D/L from MODEL_A's perspective + per-color split + forfeit reasons.
#
# Required env:
#   UPSTREAM_DIR — upstream clone
#   MODEL_A      — HF dir for model A (the model under test)
#   MODEL_B      — HF dir for model B (the opponent)
#   OUT_DIR      — output dir

set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_USE_V1=1

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-chess}"

UPSTREAM_DIR="${UPSTREAM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/vam-chess}"
: "${MODEL_A:?MODEL_A required}"
: "${MODEL_B:?MODEL_B required}"
: "${OUT_DIR:?OUT_DIR required}"
GAMES_TOTAL="${GAMES_TOTAL:-100}"
STOCKFISH_BIN="${STOCKFISH_BIN:-${UPSTREAM_DIR}/.third_party_cache/stockfish/src/stockfish}"

cd "${UPSTREAM_DIR}"
export PYTHONPATH="${UPSTREAM_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}"

python -m scripts.eval_chess_fullgame_model_vs_model \
    --model-a "${MODEL_A}" \
    --model-b "${MODEL_B}" \
    --out-dir "${OUT_DIR}" \
    --games-total ${GAMES_TOTAL} \
    --tensor-parallel-size 1 \
    --server-gpus-a 0 \
    --server-gpus-b 1 \
    --stockfish-path "${STOCKFISH_BIN}" \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-response-tokens 2000 \
    --max-retries-per-turn 1 \
    --max-plies 200 \
    --acpl-workers 8 \
    --acpl-threads 1 \
    --seed 0
