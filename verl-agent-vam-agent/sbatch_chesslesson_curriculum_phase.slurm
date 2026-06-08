#!/bin/bash
#SBATCH --job-name=chesslesson-curr
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
# Budget-based curriculum training for chesslesson.
# Submit with PHASE, MAX_BUDGET, MODEL_PATH (and optional TOTAL_EPOCHS).
#
#   sbatch --export=ALL,WANDB_API_KEY=...,PHASE=1,MAX_BUDGET=1,\
#          MODEL_PATH=$HOME/models/Qwen2.5-7B-Instruct sbatch_chesslesson_curriculum_phase.slurm
#
# After training, auto-merges the final FSDP ckpt to HF format at
# $MERGED_OUT_DIR/phase${PHASE}.

set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export HYDRA_FULL_ERROR=1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-chess}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p logs

: "${PHASE:?set PHASE=1|2|3|4}"
: "${MAX_BUDGET:?set MAX_BUDGET=1|2|4|null}"
: "${MODEL_PATH:?set MODEL_PATH (base for phase1, or prev phase merged dir)}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"

export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-chess_self_play_chesslesson_curriculum}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
MERGED_OUT_DIR="${MERGED_OUT_DIR:-$HOME/models/chesslesson_curriculum}"
mkdir -p "${MERGED_OUT_DIR}"

TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-32}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-32}"
GROUP_SIZE="${GROUP_SIZE:-16}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
HISTORY_LENGTH="${HISTORY_LENGTH:-0}"

EXP="curr_p${PHASE}_b${MAX_BUDGET}_g${GROUP_SIZE}"
CKPT_DIR="${CHECKPOINTS_DIR}/${WANDB_PROJECT}/${EXP}"
MERGED_DIR="${MERGED_OUT_DIR}/phase${PHASE}"

# 1) Placeholder parquet
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size "${TRAIN_DATA_SIZE}" \
    --val_data_size "${VAL_DATA_SIZE}"

# 2) HGPO trainer with curriculum-filtered task pool
EXTRA_OVERRIDES=()
if [ -n "${MAX_BUDGET:-}" ] && [ "${MAX_BUDGET}" != "null" ]; then
    EXTRA_OVERRIDES+=("env.chesslesson.max_budget=${MAX_BUDGET}")
fi
if [ -n "${ALLOWED_STAGES:-}" ] && [ "${ALLOWED_STAGES}" != "null" ]; then
    EXTRA_OVERRIDES+=("+env.chesslesson.allowed_stages=${ALLOWED_STAGES}")
fi
if [ "${COORD_ONLY:-False}" = "True" ] || [ "${COORD_ONLY:-0}" = "1" ]; then
    EXTRA_OVERRIDES+=("+env.chesslesson.coord_only=True")
fi

python3 -m verl.trainer.main_chesslesson_agent \
    env.env_name=chesslesson/LichessLearn \
    env.seed=0 \
    env.max_steps=8 \
    env.history_length="${HISTORY_LENGTH}" \
    env.chesslesson.include_chess=True \
    env.chesslesson.include_coord=True \
    env.rollout.n="${GROUP_SIZE}" \
    "${EXTRA_OVERRIDES[@]}" \
    data.train_files="$HOME/data/verl-agent/text/train.parquet" \
    data.val_files="$HOME/data/verl-agent/text/test.parquet" \
    data.train_batch_size="${TRAIN_DATA_SIZE}" \
    data.val_batch_size="${VAL_DATA_SIZE}" \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
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
    algorithm.adv_estimator="${ADV_ESTIMATOR:-hgpo}" \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=1.0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXP}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.val_only=False \
    trainer.val_before_train=False \
    trainer.default_local_dir="${CKPT_DIR}"

# 3) Merge final ckpt FSDP → HF for next phase to use
LATEST=$(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt")
echo "[merge] latest=${LATEST}  src=${CKPT_DIR}/global_step_${LATEST}/actor  dst=${MERGED_DIR}"
python3 scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}/global_step_${LATEST}/actor" \
    --target_dir "${MERGED_DIR}"

echo "[done] Phase ${PHASE} complete. Merged HF model at ${MERGED_DIR}"
