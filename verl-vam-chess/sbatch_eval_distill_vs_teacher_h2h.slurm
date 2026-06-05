#!/bin/bash
#SBATCH --job-name=distill-vs-teacher-h2h
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=$HOME/chess_rl_outputs/logs/slurm-%x-%j.out
#SBATCH --error=$HOME/chess_rl_outputs/logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# Model-vs-model head-to-head: distill 3B step_300 vs teacher 7B step_640.
# 2 GPUs: each model on 1 GPU, TP=1.

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

MODEL_A="${MODEL_A:?MODEL_A (distill 3B HF dir) must be set}"
MODEL_B="${MODEL_B:?MODEL_B (teacher 7B HF dir) must be set}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/eval_results/distill_vs_teacher_h2h}"
GAMES_TOTAL="${GAMES_TOTAL:-100}"

mkdir -p "${OUT_DIR}"

echo "[h2h] MODEL_A (distill 3B)=${MODEL_A}"
echo "[h2h] MODEL_B (teacher 7B)=${MODEL_B}"
echo "[h2h] OUT_DIR=${OUT_DIR}"
echo "[h2h] GAMES_TOTAL=${GAMES_TOTAL}"

python -m scripts.eval_chess_fullgame_model_vs_model \
    --model-a "${MODEL_A}" \
    --model-b "${MODEL_B}" \
    --out-dir "${OUT_DIR}" \
    --games-total ${GAMES_TOTAL} \
    --tensor-parallel-size 1 \
    --server-gpus-a 0 \
    --server-gpus-b 1 \
    --stockfish-path .third_party_cache/stockfish/src/stockfish \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-response-tokens 2000 \
    --max-retries-per-turn 1 \
    --max-plies 200 \
    --acpl-workers 8 \
    --acpl-threads 1 \
    --seed 0
