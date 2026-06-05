# Logprob Gain Gating for Iterative Allowed-Move Elimination

This note documents the gain-based sample filter added to iterative GRPO training.
It covers both the conceptual intent and the exact runtime behavior in code.

## 1) Problem and Goal

In iterative allowed-move elimination, one original prompt can generate multiple round-specific prompt variants:

- Round prompt `p_i`: uses a reduced candidate set after elimination updates.
- Base prompt `p_0`: original incoming base-batch prompt context before elimination.
  In current offline setups this is usually full-legal, but the code does not re-render or enforce that here.

If optimization aggregates samples from later rounds, reduced prompt spaces can make some responses artificially easier under `p_i`.
That can create shortcuts (for example, single-candidate or near-single-candidate rounds) that are not aligned with the original decision difficulty.

Goal: remove samples whose round-conditioned context gives too much logprob advantage over the base context.

## 2) Core Rule

For each sampled response `r_i` from prompt variant `p_i`, define:

`Gain(r_i) = logprob(r_i | p_i) - logprob(r_i | p_0)`

Important:

- Here `logprob(r_i | p)` means the **masked sum of per-token response logprobs** under prompt context `p`.
- It is **not** a per-token average and is **not** normalized by response length.
- So gain gating is intentionally **length-sensitive**: longer responses can accumulate larger positive gain.

Filtering rule:

- Keep sample if `Gain(r_i) <= gain_threshold`.
- Drop sample if `Gain(r_i) > gain_threshold`.

Default threshold:

- `gain_threshold = log(10) = 2.302585092994046`.

Disable behavior:

- Any non-finite threshold (`+inf` or `-inf`) disables gain filtering.
- `NaN` is rejected with an explicit error.

## 3) High-Level Integration in Training Pipeline

The insertion point is in the `allowed_move_elim` branch after rounds are concatenated and before GRPO optimization flow.

Pipeline order:

1. Build round batches (`p_i`), rollout responses, compute rewards.
2. Concatenate all rounds into one training batch.
3. Apply existing optional reward-range group filter (if enabled).
4. Apply new gain filter (sample-level), using both `p_i` and `p_0` scoring.
5. Apply round-0 prompt stitching for optimization context only if `stitch_round0_prompt_for_logprob=True`.
   If gain filtering is disabled, stitching can happen per-round before concat (existing behavior).
   If gain filtering is enabled, stitching is deferred until after gain scoring.
6. Compute per-sample loss weights, pad, recompute `old_log_probs`, compute GRPO advantages, update actor.

Because filtering is done before steps 5-6 and before `compute_advantage`, filtered samples are excluded from:

- GRPO group mean/std and advantage computation.
- Actor optimization.

## 4) Detailed Implementation

### 4.1 New helpers

In `verl/trainer/ppo/ray_trainer.py`:

- `_clone_allowed_move_elim_logprob_batch(...)`
  - Clones only needed tensor keys for alternate logprob scoring context.
  - Preserves `allowed_move_elim_prompt_idx` for round-0 stitching mapping.
- `_sum_masked_token_log_probs(...)`
  - Converts token logprobs into sequence logprob using `response_mask`.

This keeps `r_i` fixed while changing prompt context between `p_i` and `p_0`.

### 4.2 Config parse and gating switches

Runtime parse in allowed-move-elim config section:

- `allowed_move_elim_gain_threshold = float(cfg.get("gain_threshold", np.log(10.0)))`
- `gain_filter_enabled = isfinite(gain_threshold)`
- NaN threshold is rejected with explicit error.

Interaction with prompt-stitching knob:

- Existing `stitch_round0_prompt_for_logprob=True` used to stitch each round immediately.
- With gain filter enabled, per-round stitching is deferred until after gain scoring.
- This is required so `logprob(r_i | p_i)` remains available.
- With gain filter disabled, stitching behavior remains per-round as before.

### 4.3 Exact scoring path

For concatenated allowed-move-elim batch:

1. Ensure `response_mask`.
2. Compute `old_log_probs` on current batch context (`p_i`) via `actor_rollout_wg.compute_log_prob(...)`.
3. Reduce to sequence logprob: masked sum over response tokens.
   - No length normalization is applied here.
4. Clone batch, stitch prompts to round-0 context (`p_0`) using `allowed_move_elim_prompt_idx`.
5. Compute second `old_log_probs` on stitched clone.
6. Reduce to sequence logprob on stitched context.
   - Again, this is a masked token-logprob sum, not a mean.
7. Compute `gain = logprob_pi - logprob_p0`.

### 4.4 Filtering behavior

Sample mask:

- `keep = gain <= threshold`.

If any are dropped:

- Batch is sliced to kept indices only.
- Counts are printed and logged.

If all are dropped:

- Training step errors with explicit message:
  - increase threshold, or
  - disable with `gain_threshold=inf`.

### 4.5 Bookkeeping fields

When gain filtering is enabled, trainer stores:

- `non_tensor_batch["allowed_move_elim_logprob_pi"]`
- `non_tensor_batch["allowed_move_elim_logprob_p0"]`
- `non_tensor_batch["allowed_move_elim_gain"]`

These fields are aligned with the post-filter batch rows.
When gain filtering is disabled, these fields are not added.

## 5) Metrics and Observability

Added `selection_sampler/*` metrics when gain filtering is enabled:

- threshold and on/off:
  - `selection_sampler/gain_threshold`
  - `selection_sampler/gain_filter_enabled`
- sample counts:
  - `selection_sampler/gain_total_samples`
  - `selection_sampler/gain_kept_samples`
  - `selection_sampler/gain_filtered_samples`
  - `selection_sampler/gain_kept_frac`
  - `selection_sampler/gain_filtered_frac`
- gain distribution:
  - `selection_sampler/gain_mean`
  - `selection_sampler/gain_std`
  - `selection_sampler/gain_min`
  - `selection_sampler/gain_p50`
  - `selection_sampler/gain_p90`
  - `selection_sampler/gain_p99`
  - `selection_sampler/gain_max`
- uid-group diagnostics under current group ids:
  - `selection_sampler/gain_uid_groups_total`
  - `selection_sampler/gain_uid_groups_kept`
  - `selection_sampler/gain_uid_groups_all_filtered`
  - `selection_sampler/gain_uid_groups_partially_filtered`

Timing metrics include:

- `timing_s/allowed_move_elim_gain_logprob_pi`
- `timing_s/allowed_move_elim_gain_logprob_p0`

This makes the compute overhead explicit.

When gain filtering is disabled, only `selection_sampler/gain_filter_enabled=0.0` is emitted from this block.

## 6) Config and Launcher Wiring

### 6.1 `train_chess.sh`

Added:

- env default:
  - `ALLOWED_MOVE_ELIM_GAIN_THRESHOLD=2.302585092994046`
- config echo under allowed-move-elim block
- hydra passthrough:
  - `+algorithm.allowed_move_elim.gain_threshold=${ALLOWED_MOVE_ELIM_GAIN_THRESHOLD}`

### 6.2 `sbatch_chess_small_legal_gh200.slurm`

Added:

- `CHESS_ALLOWED_MOVE_ELIM_GAIN_THRESHOLD` -> `ALLOWED_MOVE_ELIM_GAIN_THRESHOLD`
- print of effective gain threshold in launcher logs

Also added:

- `CHESS_RL_VAL_BEFORE_TRAIN` -> `VAL_BEFORE_TRAIN`

to support fast smoke runs without pre-train validation.

## 7) Semantics and Caveats

1. This is a sample-level filter, not a whole-uid-group filter.
- In `uid_mode=per_prompt`, groups can become partially filtered.
- GRPO then operates on remaining samples in those groups.

2. Gain uses current actor model logprobs before update.
- Both terms are scored with the same model parameters at that step.
- Only prompt context differs (`p_i` vs `p_0`).

3. Existing `group_reward_range_min` filter remains independent.
- Reward-range filtering runs before gain filtering.

4. Compute cost increases.
- Two additional logprob passes per step in allowed-move-elim when enabled.

5. `p_0` semantics are "incoming base prompt context."
- In common runs this is full-legal prompt context, but this depends on the dataset/self-play path feeding `base_batch`.

## 8) Smoke Validation Snapshot (GH200)

One-step smoke run (job `2477021`) with:

- `CHESS_RL_TOTAL_TRAINING_STEPS=1`
- `CHESS_RL_VAL_BEFORE_TRAIN=False`
- `CHESS_RL_TRAINER_TEST_FREQ=-1`
- `CHESS_RL_FULL_EVAL_FREQ=-1`
- `CHESS_RL_TRAINER_SAVE_FREQ=-1`

Observed in logs:

- gain filter enabled with threshold `2.302585092994046`
- filtered samples: `1378/2608` (kept `1230`)
- gain distribution and uid-group diagnostics logged
- run completed (`sacct`: `COMPLETED`, exit `0:0`)

## 9) W&B Threshold Sweep Runs

These runs are **post-paper follow-up experiments**, not the ICML-paper reference runs.

The first same-day small-legal run after gain-gating landed kept the default threshold. A later four-run sweep
varied `CHESS_ALLOWED_MOVE_ELIM_GAIN_THRESHOLD` / `algorithm.allowed_move_elim.gain_threshold`.

| W&B run id | Created at (UTC) | State | Slurm job | Gain threshold setting | Notes |
|---|---|---|---|---|---|
| `3rpry94f` | `2026-02-25T20:21:02Z` | `crashed` | `2484107` (`chess-small-legal-gh200`) | `2.302585092994046` (`log(10)`) | Default-threshold follow-up run |
| `ei5x4314` | `2026-02-25T22:04:59Z` | `crashed` | `2485462` (`chess-small-legal-gain-10000`) | `10000` | Gain filter enabled but effectively non-binding (`gain_filtered_samples=0`) |
| `ih0nj1oy` | `2026-02-25T22:05:03Z` | `crashed` | `2485461` (`chess-small-legal-gain-log64`) | `4.1588830833596715` (`log(64)`) | Threshold sweep |
| `l31qkw8w` | `2026-02-25T22:05:06Z` | `crashed` | `2485459` (`chess-small-legal-gain-log4`) | `1.3862943611198906` (`log(4)`) | Threshold sweep |
| `qv3bfr00` | `2026-02-25T22:05:08Z` | `crashed` | `2485460` (`chess-small-legal-gain-log16`) | `2.772588722239781` (`log(16)`) | Threshold sweep |

For this sweep, all other major knobs remained aligned with the launcher defaults, including:

- `ALLOWED_MOVE_ELIM_ENABLE=True`
- `ALLOWED_MOVE_ELIM_UID_MODE=per_prompt`
- `ALLOWED_MOVE_ELIM_R_MAX_START=4`, `ALLOWED_MOVE_ELIM_R_MAX_END=4`
- `ALLOWED_MOVE_ELIM_STITCH_ROUND0_PROMPT_FOR_LOGPROB=True`
- `TRAIN_BATCH_SIZE=128`, `ROLLOUT_N=8`
- `TOTAL_TRAINING_STEPS=800`
- `CHESS_REWARD_FN=expected_score_wdl_vs_best`

Pre-gating comparison point:
- `t4elop1z` (`2026-02-23T22:33:57Z`, Slurm `2449991`, `ours-n8_r4-full`) and
  `zf6smgzg` (`2026-02-23T22:33:56Z`, Slurm `2449992`, `ours-n2_r16-full`) are the pre-gating small-legal
  reference pair. Their configs have no `gain_threshold`, and their histories contain no
  `selection_sampler/gain_*` metrics.

Provenance note:
- W&B did not preserve a usable per-run git SHA for this run family, so the pre/post-gating classification is
  grounded by config/history evidence plus run timestamps relative to the code-introduction commit.

## 10) Code Reference Map

Primary implementation:

- `verl/trainer/ppo/ray_trainer.py`
  - helper clone/reduce methods
  - config parse for `gain_threshold`
  - gain scoring/filtering block
  - stitch timing adjustment
- `train_chess.sh`
  - env defaults and hydra override
- `sbatch_chess_small_legal_gh200.slurm`
  - launcher wiring for gain threshold and `VAL_BEFORE_TRAIN`
