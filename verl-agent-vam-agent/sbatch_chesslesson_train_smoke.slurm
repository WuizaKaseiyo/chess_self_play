#!/bin/bash
#SBATCH --job-name=chesslesson-smoke
#SBATCH --partition=agent-xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
# Smoke training run for chesslesson/LichessLearn (multi-turn online puzzles).
# Wraps recipe/hgpo/run_chesslesson_train.sh under slurm.
#
# Required env:
#   WANDB_API_KEY
#
# Optional overrides (env or sbatch --export):
#   TRAIN_DATA_SIZE  (default 16)
#   GROUP_SIZE       (default 8)
#   TOTAL_EPOCHS     (default 2)
#   MODEL_PATH       (default $HOME/models/Qwen2.5-7B-Instruct)

set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-chess}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p logs

: "${WANDB_API_KEY:?set WANDB_API_KEY via --export=ALL,WANDB_API_KEY=...}"
export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-chess_self_play_chesslesson_smoke}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"

# ---- Smoke knobs (small for fast feedback) ----
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-16}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-16}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
HISTORY_LENGTH="${HISTORY_LENGTH:-0}"

# 1) Placeholder parquet for the DataLoader (data is in env, not parquet)
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size "${TRAIN_DATA_SIZE}" \
    --val_data_size "${VAL_DATA_SIZE}"

# 2) HGPO trainer on chesslesson/LichessLearn (multi-turn online puzzles)
python3 -m verl.trainer.main_chesslesson_agent \
    env.env_name=chesslesson/LichessLearn \
    env.seed=0 \
    env.max_steps=8 \
    env.history_length="${HISTORY_LENGTH}" \
    env.chesslesson.include_chess=True \
    env.chesslesson.include_coord=True \
    env.rollout.n="${GROUP_SIZE}" \
    data.train_files="$HOME/data/verl-agent/text/train.parquet" \
    data.val_files="$HOME/data/verl-agent/text/test.parquet" \
    data.train_batch_size="${TRAIN_DATA_SIZE}" \
    data.val_batch_size="${VAL_DATA_SIZE}" \
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
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=1.0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="cl_g${GROUP_SIZE}_h${HISTORY_LENGTH}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.val_only=False \
    trainer.val_before_train=False \
    trainer.default_local_dir="${CHECKPOINTS_DIR}/${WANDB_PROJECT}/cl_smoke" \
    "$@"
