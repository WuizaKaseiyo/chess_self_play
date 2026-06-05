#!/bin/bash
#SBATCH --job-name=vam-chess-smoke
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# Smoke training run for chess (WhiteVsRandom) with VAM enabled (random subset).
# Goal: validate that env.chess.vam.* flows into the env_kwargs, ChessWorker
#       initializes with VAM state, and HGPO trainer can roll out 1-2 train steps.
#
# Required env:
#   WANDB_API_KEY  (or wandb logging gets disabled / fails)
#
# Optional override knobs:
#   --export=ALL,VAM_K=12,VAM_ITERATIVE=True,VAM_SOURCE=random
#   --export=ALL,VAM_SOURCE=stockfish,STOCKFISH_BIN=/path/to/stockfish

set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-chess}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p logs

# ---- WandB ---------------------------------------------------------------
: "${WANDB_API_KEY:?set WANDB_API_KEY via --export=ALL,WANDB_API_KEY=...}"
export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-chess_self_play_vam_smoke}"

# ---- HF cache ------------------------------------------------------------
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ---- VAM knobs (overridable via --export) --------------------------------
VAM_K="${VAM_K:-8}"
VAM_ITERATIVE="${VAM_ITERATIVE:-False}"
VAM_SOURCE="${VAM_SOURCE:-random}"
STOCKFISH_BIN="${STOCKFISH_BIN:-}"
VAM_STOCKFISH_DEPTH="${VAM_STOCKFISH_DEPTH:-1}"
VAM_PENALTY="${VAM_PENALTY:--1.0}"

# ---- Smoke training knobs ------------------------------------------------
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
GROUP_SIZE="${GROUP_SIZE:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"

# ---- 1) Generate placeholder parquet (size hint only) --------------------
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size "${TRAIN_BATCH_SIZE}" \
    --val_data_size "${VAL_BATCH_SIZE}"

# ---- 2) Launch HGPO trainer on chess/WhiteVsRandom with VAM --------------
python3 -m verl.trainer.main_chess_agent \
    env.env_name=chess/WhiteVsRandom \
    env.seed=0 \
    env.max_steps=24 \
    env.history_length=1 \
    env.chess.max_agent_plies=24 \
    env.chess.vam.enable=True \
    env.chess.vam.k="${VAM_K}" \
    env.chess.vam.iterative="${VAM_ITERATIVE}" \
    env.chess.vam.subset_source="${VAM_SOURCE}" \
    env.chess.vam.penalty="${VAM_PENALTY}" \
    env.chess.vam.stockfish_path="${STOCKFISH_BIN}" \
    env.chess.vam.stockfish_depth="${VAM_STOCKFISH_DEPTH}" \
    env.rollout.n="${GROUP_SIZE}" \
    data.train_files="$HOME/data/verl-agent/text/train.parquet" \
    data.val_files="$HOME/data/verl-agent/text/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH:-$HOME/models/Qwen2.5-7B-Instruct}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=1.0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="vam_k${VAM_K}_${VAM_SOURCE}$([ "${VAM_ITERATIVE}" = "True" ] && echo "_iter")" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.val_only=False \
    trainer.val_before_train=False \
    "$@"
