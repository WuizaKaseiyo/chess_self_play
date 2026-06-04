# Training Small-Legal (Ours Method)
This document records the small-legal training setup, the verified pre-gating reference run profile used for alignment,
and the current runtime semantics that remain in force after later follow-up work such as logprob gain gating.

## 1) Motivation (Most Important)
Core insight: the new prompting scheme narrows the move space presented at each iterative-elimination step, so the model is scoring choices in a smaller legal subset than before; this is why round-conditioned logprob drops much more slowly than in the old prompt style.

Evidence from `round_vs_logprob.md` (same model/data/diagnostic loop):
- Original prompt style mean reference logprob (round 1 -> 16): `-209.4241 -> -360.1835`.
- Small-legal prompt style mean reference logprob (round 1 -> 16): `-151.0275 -> -178.5183`.
- The small-legal curve degrades far less across rounds, which is the main reason this prompt framing was adopted.

What did not change:
- Selection correctness remains strict (`<uci_move>` must match the candidate set exactly).
- Iterative elimination mechanics remain the same.
- Optimization still uses round-0/full-legal prompt-context stitching for logprob.

## 2) Data and Prompt Framing
Base dataset source:
- `data/chess_puzzles_chessr1_aligned_sharded_ours`

Small-legal dataset variant:
- `data/chess_puzzles_chessr1_aligned_sharded_ours_small_legal`

Prompt template:
- `recipe/chess/prompt_templates/select_prompt_small_legal.jinja`

Small-legal prompt semantics:
- The prompt does **not** contain a literal `allowed_moves` section.
- The model sees a single move list labeled as legal moves; at dataset time this list is initialized from
  `reward_model.considered_moves_uci == reward_model.legal_moves_uci`.
- During iterative elimination, the trainer shrinks `reward_model.considered_moves_uci` round by round, so the
  displayed legal-move list becomes the current candidate set.

Build command:
```bash
python scripts/rewrite_chess_prompts_from_template.py \
  --input_dir data/chess_puzzles_chessr1_aligned_sharded_ours \
  --output_dir data/chess_puzzles_chessr1_aligned_sharded_ours_small_legal \
  --template_path recipe/chess/prompt_templates/select_prompt_small_legal.jinja \
  --set_considered_moves_uci \
  --overwrite
```

Expected files in the rewritten dataset:
- `train_0.parquet`
- `train_1.parquet`
- `train_hard_0.parquet`
- `train_hard_1.parquet`
- `test.parquet`
- `test_shuffled_legal_moves.parquet`

## 3) Runtime Semantics Kept Intact
Core behavior:
- Vanilla GRPO behavior (`CHESS_PASS_K_TRAINING=False`, `CHESS_DIVERSITY_ENABLE=False`).
- Iterative elimination enabled (`CHESS_ALLOWED_MOVE_ELIM_ENABLE=True`).
- Per-prompt aggregation (`CHESS_ALLOWED_MOVE_ELIM_UID_MODE=per_prompt`).
- Round-0 prompt-context stitching enabled (`CHESS_ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB=True`).
- Reward gating forced to considered subset (`CHESS_ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI=True`).

Important nuance:
- The reward path still uses the conceptual candidate set via `reward_model.considered_moves_uci`.
- Because the prompt omits the literal `allowed_moves` token, the trainer must force
  `use_considered_moves_uci=True` so reward parsing and subset penalties continue to track the displayed move list.

Related code map:
- `verl/trainer/ppo/ray_trainer.py` (iterative loop + logprob stitching behavior)
- `train_chess.sh` (Hydra wiring + required `CHESS_RL_VAL_FILES`)
- `launch_ours.sh` (experiment launcher profiles)
- `sbatch_chess_small_legal_gh200.slurm` (small-legal dedicated GH200 launcher)
- `round_vs_logprob.md` (diagnostic rationale and measurements)

## 4) Verified Reference Run Profile
Reference identifiers:
- W&B run: `https://wandb.ai/gabr1e11/chess_rl/runs/t4elop1z`
- Slurm job: `2449991` (`ours-n8_r4-full`)
- Creation timestamp (UTC): `2026-02-23T22:33:57Z`

Scope note:
- This is the **pre-gating** small-legal reference run used for alignment.
- The paired companion run was `zf6smgzg` / Slurm `2449992` (`ours-n2_r16-full`), created at
  `2026-02-23T22:33:56Z`.
- Later follow-up runs that added logprob gain gating are tracked separately in `logprob_gating.md`.

Verified hyperparameters for that run:
- Cluster shape: `1` node, `gpu:4`, `cpus-per-task=144`.
- Data dir: `data/chess_puzzles_chessr1_aligned_sharded_ours_small_legal`.
- Validation file for the reference run: `data/chess_puzzles_chessr1_aligned_sharded_ours_small_legal/test.parquet`.
- Model: `/projects/a5l/ziyan/models/Qwen/Qwen2.5-3B-Instruct`.
- Batch/lengths: `train_batch_size=128`, `max_prompt_length=1536`, `max_response_length=2000`, `total_epochs=1`.
- Iterative rollout profile: `ROLLOUT_N=8`, `r_max_start=4`, `r_max_end=4`.
- Training schedule: `total_training_steps=800`, `trainer.test_freq=40`.
- Checkpoint/eval cadence: `save_freq=80`, `full_eval_freq=80`.
- HF checkpointing: upload enabled, repo `Gabr1e11/a_lot_of_models`, token file `/home/a5l/ziyan.a5l/.hf_token`.
- HF transfer acceleration: disabled (`CHESS_RL_HF_TRANSFER_ENABLE=0`, `HF_HUB_ENABLE_HF_TRANSFER=0`).

## 5) Launcher Defaults (Current)
`sbatch_chess_small_legal_gh200.slurm` still matches the same overall small-legal regime, but it no longer
exactly matches the pre-gating reference run above:
- `#SBATCH --nodes=1`
- `CHESS_RL_VAL_FILES=[${CHESS_DATA_DIR}/test.parquet,${CHESS_DATA_DIR}/test_shuffled_legal_moves.parquet]`
- `CHESS_RL_TRAIN_BATCH_SIZE=128`
- `CHESS_RL_TOTAL_TRAINING_STEPS=800`
- `CHESS_RL_TRAINER_SAVE_FREQ=80`
- `CHESS_RL_FULL_EVAL_FREQ=80`
- `ROLLOUT_N=8`
- `CHESS_RL_HF_UPLOAD_ENABLE=True`
- `CHESS_RL_HF_TRANSFER_ENABLE=0`
- `CHESS_ALLOWED_MOVE_ELIM_FORCE_USE_CONSIDERED_MOVES_UCI=True`
- `CHESS_ALLOWED_MOVE_ELIM_GAIN_THRESHOLD=2.302585092994046` (added later; this is post-gating launcher wiring)

If you need a different regime (for example n2/r16, multi-file validation, or no HF upload), override via `sbatch --export=ALL,...`.

## 6) Validation Status and Limitation
This method is validated for the current offline parquet workflow (`CHESS_SELF_PLAY_ENABLE=False`).

Not yet fully validated:
- Final combined setup with online engine-play data generation (`CHESS_SELF_PLAY_ENABLE=True`) plus this small-legal configuration.

Before adopting online self-play with this method, run dedicated smoke validation in that exact mode and re-check reward/prompt/round semantics on that path.
