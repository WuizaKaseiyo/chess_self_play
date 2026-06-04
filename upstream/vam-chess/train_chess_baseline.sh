set -x

# Baseline GRPO (no forced-prefix) training script.
# Mirrors train_chess.sh but disables forced-prefix exploration entirely.

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
USE_CRITIC_DATASET="${USE_CRITIC_DATASET:-False}"
PASS_K_TRAINING="${PASS_K_TRAINING:-False}"
PASS_K="${PASS_K:-1}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Base}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-False}"
RESPONSE_PREFIX_ENABLE="${RESPONSE_PREFIX_ENABLE:-False}"
RESPONSE_PREFIX="${RESPONSE_PREFIX:-}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"

if [ "$USE_CRITIC_DATASET" = "True" ]; then
    train_files="data/chess_puzzles/train_critic.parquet"
    test_files="data/chess_puzzles/test_critic.parquet"
    VERL_BASE_DIR="/fsx-project/zhichengzhang/chess_rl/baseline_critic"
    EXPERIMENT_NAME="baseline_critic"
else
    train_files="data/chess_puzzles/train.parquet"
    test_files="data/chess_puzzles/test.parquet"
    VERL_BASE_DIR="/fsx-project/zhichengzhang/chess_rl/baseline"
    EXPERIMENT_NAME="baseline"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.pass_k_training=${PASS_K_TRAINING} \
    algorithm.pass_k_k=${PASS_K} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.use_chat_template=${USE_CHAT_TEMPLATE} \
    custom_reward_function.path=${PROJECT_DIR}/recipe/chess/reward_fn.py \
    custom_reward_function.name=compute_score_batch \
    data.reward_fn_key=reward_model \
    reward_model.reward_manager=batch \
    forced_prefix.enable=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.response_prefix_enable=${RESPONSE_PREFIX_ENABLE} \
    "actor_rollout_ref.rollout.response_prefix='${RESPONSE_PREFIX}'" \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='chess_rl' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 \
    trainer.val_before_train=True \
    trainer.rollout_data_dir=${VERL_BASE_DIR}/rollout/rollout_logs \
    trainer.validation_data_dir=${VERL_BASE_DIR}/rollout/validation_logs \
    trainer.log_val_generations=100 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=${VERL_BASE_DIR}/checkpoints \
    hydra.run.dir=${VERL_BASE_DIR}/hydra_outputs
