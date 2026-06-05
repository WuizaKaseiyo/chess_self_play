# Iterative allowed-move elimination training (selection framing)

This document describes the current iterative training loop for restricted-moves ("selection")
chess. The implementation lives primarily in `verl/trainer/ppo/ray_trainer.py` with loss-weight
support in `verl/trainer/ppo/core_algos.py` and the actor workers.

This repo supports two sources of **training positions** for `allowed_move_elim`:
- **Offline full-legal selection parquets** (recommended): prebuilt rows where `allowed_moves == legal_moves`.
  - Searchless-chess v4: `data/chess_puzzles_select_v4/`
  - Chess‑R1 aligned (selection prompt variant): `data/chess_puzzles_chessr1_aligned_sharded_ours/`
- **Online play vs engine opponent (opt-in; legacy name: `self_play`)**: maintain a persistent pool of
  `B = data.train_batch_size` parallel games (model vs Stockfish depth=1), collect **one position per game per
  trainer step** (from *both* model-to-move and engine-to-move turns), and construct `reward_model` payloads
  on-the-fly.

## Dataset contract (offline full-legal selection)

Offline `allowed_move_elim` starts from **full-legal selection rows** where:
`reward_model.considered_moves_uci == reward_model.legal_moves_uci` (order-preserving).

Searchless-chess v4:
- Source puzzles: `data/chess_puzzles/{train,train_hard,test}.parquet`
- Selection dataset (full-legal only): `data/chess_puzzles_select_v4/{train,train_hard,test}.parquet`
- Builder: `scripts/build_chess_select_train_dataset_v4.py`

Chess‑R1 aligned (selection prompt variant; prompt-only rewrite, no rescoring):
- Source (sharded): `data/chess_puzzles_chessr1_aligned_sharded/`
- Selection prompt variant: `data/chess_puzzles_chessr1_aligned_sharded_ours/`
- Builder: `scripts/rewrite_chess_prompts_from_template.py` with `--template_path recipe/chess/prompt_templates/select_prompt.jinja`
  and `--set_considered_moves_uci`.

Shared per-row requirements:
- `reward_model.legal_moves_uci` is an ordered legal move list (UCI).
- `reward_model.considered_moves_uci` is set to the full legal list (order-preserving).
- `prompt` is rendered from `recipe/chess/prompt_templates/select_prompt.jinja` and includes `allowed_moves`.
- To train on the hard split, set `USE_HARD_DATASET=True` (uses `train_hard*`).

Small-legal variant note:
- The later `recipe/chess/prompt_templates/select_prompt_small_legal.jinja` prompt does **not** include a literal
  `allowed_moves` section. It shows only one move list labeled as legal moves, while the operative candidate set
  is still carried in `reward_model.considered_moves_uci`.
- In that variant, the base offline rows still start with `considered_moves_uci == legal_moves_uci`; the smaller
  candidate set emerges later when trainer-side elimination rounds prune `considered_moves_uci`.

## Online play vs engine opponent data source (opt-in; `data.self_play`)

When enabled, online play vs engine opponent replaces the offline **training** dataset with online generation:
- The trainer ignores `data.train_files` and builds an in-memory batch of rows each step.
- Validation remains unchanged and still uses parquet(s) via `data.val_files`.

### Enable / defaults

This online data source is opt-in and currently gated on the iterative sampler:
- Enable: `CHESS_SELF_PLAY_ENABLE=True`
- Required (when using `sbatch_train_chess_gh200.slurm`): `CHESS_ALLOWED_MOVE_ELIM_ENABLE=True`
  - When running `train_chess.sh` directly, set `ALLOWED_MOVE_ELIM_ENABLE=True`.

Fixed defaults for this research iteration:
- pool size is tied to `data.train_batch_size` (one game per training row per step; games are **not** truncated)
- opponent policy: Stockfish depth=1

Note on naming / back-compat:
- This feature is called `self_play` in configs/env vars for back-compat, but it is **not** true self-play:
  the model plays against an engine opponent (Stockfish).
- Older configs may still set `data.self_play.num_parallel_games` (or `expected_length`), but the
  trainer ties the pool size to `data.train_batch_size` and will warn if these disagree.

### Per-step generation semantics

Let `B = data.train_batch_size`.

Online play vs engine opponent runs a rolling pool of `B` games that persists across training steps.
The pool is initialized with ~50/50 “model moves first” vs “engine moves first” games (deterministic shuffle).

On every training step:
1. Record the current position for **every** game (one row per game), as a training position.
   - This includes both **model-to-move** and **engine-to-move** positions; we learn from all encountered turns.
2. Advance each game by **exactly one ply**:
   - If it is the model’s turn, query the model using the selection template with
     `allowed_moves = legal_moves` (full-legal, AIcrowd/python-chess order).
   - If it is the opponent’s turn, play Stockfish `engine.play` at depth=1.
3. Games are **never** truncated to a fixed length. If a game ends naturally
   (mate/stalemate/insufficient material/...) or is forfeited due to invalid model output after retries,
   it is immediately replaced by a fresh game.
   - On restart, the model color is **toggled** for that slot so long-run “who moves first” remains ~50/50
     even if game lengths differ.

Move parsing / retry semantics are aligned to the competition-style full-game evaluation loop:
- Parse the first `<uci_move>...</uci_move>` span.
- Retry by resending the **same prompt** up to `max_retries_per_turn` attempts (default 3).
- `<uci_move>resign</uci_move>` is treated as “no move” and triggers retry.
- After retries are exhausted, the game is forfeited and replaced by a new game.

### Per-position `reward_model` construction (online-play rows)

For every recorded position:
- `reward_model.fen`: the recorded FEN
- `reward_model.legal_moves_uci`: **ordered** legal move list in python-chess / AIcrowd order
- `reward_model.considered_moves_uci`: initialized to the full legal list (order-preserving)
- μ map over **all** legal moves:
  - Always writes `reward_model.move_values_json` (win-probabilities in `[0,1]`), derived from
    Stockfish centipawn eval via `centipawn_to_win_prob` (same mapping as `scripts/rescore_puzzles_cp.py`).
  - Always writes `reward_model.move_expected_scores_json` (expected score in `[0,1]`):
    - Prefer true WDL expected scores when Stockfish provides WDL for every legal move.
    - Otherwise fall back deterministically to the CP-derived win-probabilities (schema-compatible, avoids NaNs).
  - JSON formatting is stable (sorted keys, compact separators) for reproducibility.
- `reward_model.ground_truth`: the μ-best legal move with deterministic tie-break:
  higher μ wins; if μ ties, lexicographically smaller UCI wins.

Additional online-play metadata is stored on each row under `extra_info` to make it easy to verify
the “one row per game per step” and “learn from both sides” semantics:
- `extra_info.self_play_slot_id`: integer slot in `[0, B)`
- `extra_info.self_play_model_color`: `"white"` / `"black"` (which side the model is playing for this game)
- `extra_info.self_play_to_move_color`: `"white"` / `"black"` at the recorded position
- `extra_info.self_play_is_model_to_move`: boolean
- `extra_info.self_play_ply`: ply index at the recorded position (len(move_stack))
- `extra_info.self_play_game_id`, `extra_info.self_play_step`, `extra_info.self_play_turn_idx` (debug)

Implementation references:
- Trainer-side batch builder: `verl/trainer/ppo/ray_trainer.py` (`_build_self_play_train_batch`)
- Shared Stockfish scoring helper: `recipe/chess/stockfish_scoring.py`

### Debugging dumps / inspection

Online play vs engine opponent can optionally dump the generated batch at selected training steps as JSONL:
- Default dump dir (via `train_chess.sh`): `${VERL_BASE_DIR}/rollout/self_play_batches/`
- Default dump steps: `[1]` (trainer `global_steps` is 1-indexed)
- Override via env:
  - `CHESS_SELF_PLAY_DUMP_DIR=/path`
  - `CHESS_SELF_PLAY_DUMP_STEPS=[1,50,100]`

Invariant inspection script:
`scripts/inspect_self_play_batch.py` validates:
- `len(legal_moves_uci) > 0`
- `considered_moves_uci == legal_moves_uci` at initialization
- μ map covers all legal moves
- `ground_truth` equals μ-best with tie-break
- prompt includes the `allowed_moves` section

This prompt-shape check currently reflects the canonical `select_prompt.jinja` path, not the small-legal variant.

## Trainer algorithm (allowed_move_elim)

The trainer-side sampling loop replaces `filter_groups` and runs **batched rounds** of
allowed-move elimination. Key code paths:
`verl/trainer/ppo/ray_trainer.py` (`allowed_move_elim` section),
`recipe/chess/reward_fn.py` (selection reward + metadata),
`recipe/chess/prompt_templates/select_prompt.jinja` (prompt rendering).

For the small-legal prompt variant, the same elimination mechanics apply, but the prompt surface omits the literal
`allowed_moves` token and reward correctness relies on `force_use_considered_moves_uci=True` so subset penalties
continue to follow `reward_model.considered_moves_uci`.

Per training step:
1. Sample a base batch of size `data.train_batch_size`. For each prompt `i`, initialize
   `B_i = legal_moves_uci` (order-preserving).
2. Maintain `unresolved` prompt indices. For round `r = 1..R_max(t)`:
   - Build a round batch from unresolved prompts with `allowed_moves = B_i`.
     `_build_allowed_move_elim_batch` also updates `reward_model.considered_moves_uci = B_i`
     so reward parsing and the prompt are consistent.
   - Add bookkeeping fields:
     - `allowed_move_elim_prompt_idx` (original prompt index)
     - `allowed_move_elim_round` (round number)
   - Repeat each prompt `rollout.n` times (interleaved) to form GRPO groups, and generate
     **one batched** rollout for the round (no per-prompt loops).
     - GRPO grouping key is `uid`:
       - Default: `uid_mode=per_round` → one uid per *(original prompt, round)* (i.e., each round is its own GRPO group).
       - Optional: `uid_mode=per_prompt` → one uid per *original prompt* across all rounds (i.e., collapse rounds into one GRPO group).
   - **DP padding:** the generation batch is padded to the DP divisor via
     `pad_dataproto_to_divisor(...)`, then unpadded after generation to avoid
     "only support equal chunk" errors when unresolved counts are not divisible
     by DP size.
   - Compute rewards using `recipe/chess/reward_fn.py`. Use:
     - `pred_move`, `gt_uci`, `target_move`, `penalty_applied`, `in_subset` from reward metadata.
     - Success criterion: any rollout in the group has `pred_move == gt_uci` AND
       `penalty_applied == False` (and `in_subset == True` when present).
       - Note: `gt_uci` is the dataset label, while `target_move` is the μ-best move within the
         *current* candidate set. For v4 these match except for rare μ ties, but keep the distinction
         in mind when changing datasets / candidate lists.
   - Update `B_i` by removing valid in-subset predicted moves (order preserved).
   - If a prompt succeeds, stop further rounds for that prompt.
   - If still unresolved at `r_max` and policy is `accept_last`, mark the last round as
     forced accept for bookkeeping (training still keeps **all** rounds).
3. Concatenate **all** round batches into the training batch.
   - Default (`uid_mode=per_round`): each original prompt contributes `k` GRPO uid-groups where `k` is the
     number of rounds used for that prompt (1..`R_max`).
   - Optional (`uid_mode=per_prompt`): each original prompt contributes **exactly 1** GRPO uid-group, with
     group size `(k * rollout.n)` (aggregating samples across all elimination rounds for that prompt).
4. Normalize loss per original prompt:
   - Compute `groups_per_prompt[i]` (rounds used for prompt `i`).
   - Set `loss_weights = 1 / groups_per_prompt[prompt_idx]` and attach as
     `batch.batch["loss_weights"]` (shape `[N, 1]`, broadcastable).
   - Policy loss, entropy loss, and KL loss are multiplied by `loss_weights`.
   - Code: `verl/trainer/ppo/ray_trainer.py` (weights creation),
     `verl/trainer/ppo/core_algos.py` (loss weighting),
     `verl/workers/actor/dp_actor.py` and `verl/workers/actor/megatron_actor.py`
     (weight passthrough).

## R_max schedule

`R_max(t)` is computed by `_compute_allowed_move_elim_r_max` in
`verl/trainer/ppo/ray_trainer.py` and logs to `selection_sampler/r_max`.

Config knobs:
- `algorithm.allowed_move_elim.r_max_start`
- `algorithm.allowed_move_elim.r_max_end`
- `algorithm.allowed_move_elim.anneal_frac` (fraction of total steps used for linear anneal)

## Logging and W&B artifacts

When `trainer.rejected_rollout_data_dir` or `trainer.rollout_data_dir` is set, **every round**
is dumped to JSONL under:

`$VERL_BASE_DIR/rollout/rejected_rollout_logs/allowed_move_elim_rounds/<global_step>_round<r>.jsonl`

Each record includes:
- `allowed_move_elim_round`, `allowed_move_elim_prompt_idx`, `allowed_move_elim_b_size`
- `allowed_move_elim_success`, `allowed_move_elim_forced_accept`, `allowed_move_elim_accepted`
- reward metadata from `recipe/chess/reward_fn.py`

The GH200 launcher (`sbatch_train_chess_gh200.slurm`) syncs these files to W&B by default.

## Config knobs (train_chess.sh / sbatch_train_chess_gh200.slurm)

Enable the iterative loop:
- `CHESS_ALLOWED_MOVE_ELIM_ENABLE=True` (disables `filter_groups`)
- `CHESS_ALLOWED_MOVE_ELIM_UID_MODE={per_round,per_prompt}` (default `per_round`)
  - `per_round`: one GRPO uid-group per *(prompt, round)* (historical behavior; `R` rounds → `R` groups per prompt).
  - `per_prompt`: one GRPO uid-group per *prompt* (collapse rounds; aggregate rollouts across rounds into one group).
- `CHESS_ALLOWED_MOVE_ELIM_R_MAX_START`, `CHESS_ALLOWED_MOVE_ELIM_R_MAX_END`,
  `CHESS_ALLOWED_MOVE_ELIM_ANNEAL_FRAC`

Dataset selection:
- `CHESS_DATA_DIR=data/chess_puzzles_select_v4`
- `USE_HARD_DATASET=True` (optional)

Reward shaping:
- **Recommended for allowed_move_elim + GRPO:** `CHESS_REWARD_FN=winrate_vs_best`
  - Expected-score maps (`move_expected_scores_json`) are highly quantized in the v4 full-legal setting
    (often only a few unique values across all legal moves), which creates many tied rewards and thus
    many zero-variance (“dead”) GRPO groups when `filter_groups` is disabled.
  - Win-probability maps (`move_values_json`) are often much denser than expected-score buckets in full-legal,
    which can restore within-group reward variance while still staying strongly aligned with “best move”.
- Alternative (variance-focused): `CHESS_REWARD_FN=rank_among_moves`
  - Also avoids dead groups, but may be a weaker objective than `winrate_vs_best` in some settings.
- **Avoid for allowed_move_elim + GRPO:** `CHESS_REWARD_FN=expected_score_wdl_vs_best`
  - In the full-legal setting, many distinct moves can share identical expected score / WDL buckets,
    causing a large fraction of groups to have `score_std == 0`.
  - Symptom: low `grpo/effective_batch_frac` (often ~0.5) even though the model samples diverse moves.
  - **Online-play note:** online-play rows always include `reward_model.move_expected_scores_json`.
    `extra_info.self_play_expected_score_source` indicates whether true WDL (`wdl`) was available or a
    deterministic CP-derived fallback (`cp_winprob`) was used.

Optional: reject low-variance GRPO uid-groups (allowed_move_elim compatible):
- `CHESS_ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN=<float>` (default `0.0` disables)
  - Define per-uid `group_reward_range = max(score) - min(score)` within each GRPO group.
  - `score` is the same scalar logged by the trainer (after reward shaping and penalties; any hard penalty
    forces `score=-1.0`).
  - If `group_reward_range < threshold`, the entire uid group is dropped for that step
    (smaller effective batch; see `grpo/effective_batch_size` and `selection_sampler/reward_range_*`).
  - **Calibration note (online play / self-play):** in `winrate_vs_best` units, many self-play positions
    have very small win-prob deltas across reasonable moves (often <0.05). Setting
    `CHESS_ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN=0.05` can therefore reject the *vast majority* of
    groups and collapse the effective batch size. Prefer starting with `0.0` (disabled) or a much smaller
    threshold (e.g., `<=0.01`) and validate via `selection_sampler/reward_range_*` on a short run.
  - Interaction with `CHESS_ALLOWED_MOVE_ELIM_UID_MODE`:
    - `per_round`: the range check is per *(prompt, round)* group (size `rollout.n`).
    - `per_prompt`: the range check is per *prompt* group (size `k * rollout.n`, aggregating across rounds).
  - Implementation detail: the trainer computes both `group_reward_range_all` (over all samples) and
    `group_reward_range_valid` (ignoring penalty-applied / out-of-subset samples) for debugging, but the
    rejection decision uses the **all-samples** range.
- `CHESS_ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_DUMP_MAX_GROUPS=<int>` (default `16`)
  - Controls optional rejected-group sample dumps (under `$VERL_BASE_DIR/rollout/rejected_rollout_logs/` when enabled):
    - `0`: disable sample dumps
    - `-1`: dump all rejected groups
    - `N>0`: dump up to `N` rejected groups per step

Other common knobs:
- `CHESS_RL_TRAIN_BATCH_SIZE`, `ROLLOUT_N`
- `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH`
- `TOTAL_EPOCHS` or `CHESS_RL_TOTAL_TRAINING_STEPS`

## Constraints and gotchas

- `data.gen_batch_size` must equal `data.train_batch_size` when
  `allowed_move_elim` is enabled (enforced by the trainer).
- If `CHESS_ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN>0`, the trainer may drop some GRPO uid-groups.
  If the threshold is too high and it rejects **all** groups in a step, training will error; lower the
  threshold and/or increase `CHESS_RL_TRAIN_BATCH_SIZE`.
- `reward_model.considered_moves_uci` must be an order-preserving subsequence of
  `reward_model.legal_moves_uci`.
- If `MAX_PROMPT_LENGTH` is too small, prompt filtering can drop all rows and
  training will fail with `num_samples=0`. Use a large enough value (e.g., 1536
  for full-legal prompts).

Online play vs engine opponent (`self_play`) specific:
- Always set an explicit step budget on cluster runs (e.g., `CHESS_RL_TOTAL_TRAINING_STEPS=...`).
  Unlike offline parquet training, online play vs engine opponent does not have a natural “epoch length”.
- There is **no** `TRAIN_BATCH_SIZE % 10 == 0` requirement (games are not capped to a fixed length).
- Also ensure PPO micro-batching divisibility after normalization across GPUs:
  `train_batch_size * rollout_n / world_size` must be divisible by `ppo_micro_batch_size_per_gpu`.
  Practical safe defaults:
  - 1 node (4 GPUs), `ROLLOUT_N=8`, `PPO_MICRO_BATCH_SIZE=8` → use `TRAIN_BATCH_SIZE=4,8,12,...` (e.g., 128)
  - 4 nodes (16 GPUs), `ROLLOUT_N=8`, `PPO_MICRO_BATCH_SIZE=8` → use `TRAIN_BATCH_SIZE=16,32,48,...` (e.g., 128)
