#!/bin/bash
#SBATCH --job-name=chess-baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=/fsx-project/zhichengzhang/slurm/slurm-%j.out
#SBATCH --error=/fsx-project/zhichengzhang/slurm/slurm-%j.err
#SBATCH --export=ALL

set -euo pipefail

# --- Activate conda env (hardcoded: chess-rl) ---
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] Could not find conda initialization script. Make sure conda is installed and available." >&2
  exit 1
fi

conda activate chess-rl

./train_chess_baseline.sh
