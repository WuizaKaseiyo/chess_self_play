# Iterative (ours method) Pass@k: Investigation Notes and Results

Date: 2026-02-15

This document records the analysis of why `grpo/effective_batch_size` continued to drop in an
**iterative allowed-move elimination** run even when **conditional Pass@k advantages** were enabled.

It is intended to be **reproducible** (commands, file paths, and the specific W&B run IDs are included).

Related implementation doc:
- `pass_k_training.md` (Section 16: iterative conditional Pass@k implementation + knobs)

---

## Scope

Investigate:
- W&B run `xie1sbcg` (iterative `allowed_move_elim`, conditional Pass@k enabled, `rollout.n=16`, `k=4`)
- Compare against a previous “ours method” run `s0anl08n` (iterative `allowed_move_elim`, no conditional Pass@k)

User observation:
- `grpo/effective_batch_size` (and `grpo/effective_batch_frac`) still drops over time.
- Expectation was that Pass@k training would keep it roughly stable (as observed in some baseline Pass@k runs).

---

## Runs and Job Evidence

### `xie1sbcg` (iterative conditional Pass@k)

- W&B: `gabr1e11/chess_rl/xie1sbcg`
- Slurm job: `2318653`
- State: cancelled on request
  - Evidence command:
    ```bash
    ssh a5l.aip2.isambard 'sacct -j 2318653 -o JobID,JobName%40,State,Elapsed,ExitCode'
    ```

Key config (from W&B config export):
- `algorithm.adv_estimator=grpo`
- `algorithm.pass_k_training=False` (global baseline Pass@k is off)
- `algorithm.allowed_move_elim.enable=True`
- `algorithm.allowed_move_elim.pass_k_when_no_optimal=True`
- `algorithm.allowed_move_elim.pass_k_k=4`
- `actor_rollout_ref.rollout.n=16`
- Reward shaping: `custom_reward_function.reward_kwargs.chess_reward_fn=expected_score_wdl_vs_best`

### `s0anl08n` (previous ours method reference)

- W&B: `gabr1e11/chess_rl/s0anl08n`
- Iterative `allowed_move_elim` enabled.
- `rollout.n=8` in this run.

---

## How to Reproduce the W&B Evidence Locally

Download evidence (config + full metric history) for both runs:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. conda run -n verl \
  python scripts/download_wandb_run_evidence.py \
    --entity gabr1e11 --project chess_rl --run xie1sbcg \
    --outdir analysis/wandb_evidence/xie1sbcg

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. conda run -n verl \
  python scripts/download_wandb_run_evidence.py \
    --entity gabr1e11 --project chess_rl --run s0anl08n \
    --outdir analysis/wandb_evidence/s0anl08n
```

The key files used in this investigation:
- `analysis/wandb_evidence/xie1sbcg/history.parquet`
- `analysis/wandb_evidence/xie1sbcg/config_api.json`
- `analysis/wandb_evidence/s0anl08n/history.parquet`
- `analysis/wandb_evidence/s0anl08n/config_api.json`

---

## The Critical Point: What `grpo/effective_batch_size` Measures

In this repo, `grpo/effective_batch_size` is **not** “effective batch size after Pass@k”.

It is computed as:
- Group rollouts by `uid` (GRPO groups).
- Sum token-level rewards to get one scalar reward per rollout.
- Count how many uid-groups have **non-zero reward standard deviation**.

Code:
- `verl/trainer/ppo/ray_trainer.py`, `RayPPOTrainer._compute_grpo_effective_batch`
  - Uses `token_level_rewards` (i.e., post reward-shaping; if `use_kl_in_reward=True`, it includes KL penalty).

Consequence:
- **Changing advantages** (vanilla vs Pass@k) does not directly change this metric.
- If a group’s reward vector is tied (all rollouts have the same scalar reward), then:
  - its std is zero,
  - it contributes nothing to `effective_batch_size`,
  - and (importantly) there is no learning signal for GRPO-style within-group methods anyway.

---

## High-Level Metric Comparison (`xie1sbcg` vs `s0anl08n`)

### `xie1sbcg` (rollout.n=16, conditional Pass@k enabled)

From `analysis/wandb_evidence/xie1sbcg/history.parquet`:
- Step 1: `grpo/effective_batch_size=178`, `grpo/group_count=257`, `grpo/effective_batch_frac≈0.693`
- Step 100: `grpo/effective_batch_size=144`, `grpo/group_count=293`, `grpo/effective_batch_frac≈0.491`
- Step 130: `grpo/effective_batch_size=121`, `grpo/group_count=319`, `grpo/effective_batch_frac≈0.379`

### `s0anl08n` (rollout.n=8, no conditional Pass@k)

From `analysis/wandb_evidence/s0anl08n/history.parquet`:
- Step 1: `grpo/effective_batch_size=170`, `grpo/group_count=294`, `grpo/effective_batch_frac≈0.578`
- Step 100: `grpo/effective_batch_size=142`, `grpo/group_count=326`, `grpo/effective_batch_frac≈0.436`
- Step 200: `grpo/effective_batch_size=83`, `grpo/group_count=306`, `grpo/effective_batch_frac≈0.271`
- Step 600: `grpo/effective_batch_size=35`, `grpo/group_count=326`, `grpo/effective_batch_frac≈0.107`

Takeaway:
- The “effective batch drops over time” phenomenon is **not new** to conditional Pass@k; it already appeared in
  the earlier iterative run.

---

## Verifying Conditional Pass@k Actually Executed in `xie1sbcg`

The conditional path prints a one-time banner:

```
[ALLOWED_MOVE_ELIM_PASSK] enabled=True k=4 groups_total=257 groups_passk_no_optimal=142 groups_vanilla_has_optimal=115
```

Evidence source:
- Slurm log:
  `/projects/a5l/ziyan/chess_rl/logs/slurm-ours-dataset-expectedscorewdl-selectprompt-fulleval-2318653.out`

This proves:
- conditional Pass@k was enabled,
- both branches were exercised (not all groups were “has optimal” or “no optimal”),
- and the logic was active inside the GRPO advantage computation.

---

## Deeper Dive: “Dead groups” dominate the **no-optimal** branch

The core reason `effective_batch_size` keeps dropping is that many iterative uid-groups become reward-tied.
This is especially true for groups where **no rollout hits the target move**.

For `xie1sbcg`, step 135 has allowed-move-elim round dumps under:

`/projects/a5l/ziyan/chess_rl_outputs/ours_dataset_expectedscorewdlvsbest_selectprompt_fulleval_20260215_082857/rollout/rejected_rollout_logs/allowed_move_elim_rounds/`

### Step 135 group statistics (computed from round JSONLs)

Grouping key (because `uid_mode=per_round`):
- One group per `(allowed_move_elim_prompt_idx, allowed_move_elim_round)`

At step 135:
- total groups: `309`
- groups with zero reward variance (only 1 unique score across 16 rollouts): `178` (~57.6%)
- groups with ≥1 optimal rollout: `96`
  - zero-variance among these: `13` (~13.5%)
- groups with no optimal rollout: `213`
  - zero-variance among these: `165` (~77.5%)

Interpretation:
- The “no optimal move” case is exactly where we hoped Pass@k would help.
- But most of those groups are reward-tied (often all `-1.0`), so there is no reward ranking signal for Pass@k
  to exploit.

### Not (primarily) penalties: most `-1.0` rewards are “valid-but-bad”

At step 135:
- `penalty_applied` is low: ~0.75% (37 / 4944 rollouts)
- Yet `score == -1.0` is very common: ~65.9% (3260 / 4944 rollouts)
  - and almost all of those `-1.0` scores were **not** penalties (3223 / 3260).

This points to reward-shaping quantization rather than parsing failures.

---

## Why reward quantization matters here (and why this is expected)

This experiment used:
- `CHESS_REWARD_FN=expected_score_wdl_vs_best` (i.e., `expected_score(pred) - expected_score(best_legal)`, so it is
  `<= 0` for all valid moves).

In the full-legal selection setting, `expected_score_wdl_vs_best` is often highly quantized (WDL buckets),
so many distinct “wrong” moves share the exact same expected score. When the model isn’t reliably selecting
the best move, many rollouts collapse to the same lowest bucket, producing:
- group reward std ≈ 0
- “dead” GRPO groups
- falling `grpo/effective_batch_frac`

This is explicitly called out in `iterative.md`:
- “Avoid for allowed_move_elim + GRPO: `CHESS_REWARD_FN=expected_score_wdl_vs_best`”
- Symptom: low `grpo/effective_batch_frac` (~0.5) even with diverse moves.

---

## Conclusion: likely not a conditional Pass@k bug

What we expected:
- Conditional Pass@k would improve learning signal for “no-optimal” groups.

What we observed:
- Conditional Pass@k is enabled and executed.
- However, many “no-optimal” groups have **tied rewards** (often all `-1.0` under the quantized reward),
  so both vanilla GRPO and Pass@k have little/no signal.
- Therefore `grpo/effective_batch_size` can still drop, because it is driven by **reward variance**, not advantage
  construction.

This means the drop in `effective_batch_size` is best explained by:
- metric definition (`std(reward) != 0` per uid-group), plus
- reward shaping choice and its quantization properties in iterative GRPO.

---

## Actionable Next Steps (if the goal is stable effective batch)

If the objective is specifically to keep `grpo/effective_batch_frac` high in iterative training:

1. Prefer a denser reward shaping for `allowed_move_elim + GRPO`
   - `CHESS_REWARD_FN=winrate_vs_best` (recommended in `iterative.md`)
   - or `CHESS_REWARD_FN=rank_among_moves` (variance-focused)

2. Optionally consider `CHESS_ALLOWED_MOVE_ELIM_UID_MODE=per_prompt`
   - This aggregates across rounds, which can increase within-group reward diversity.
   - It changes the meaning of “group size” for Pass@k (`n` becomes `rounds_used * rollout.n`).

3. If using reward-range rejection (`CHESS_ALLOWED_MOVE_ELIM_GROUP_REWARD_RANGE_MIN>0`)
   - Be careful: it drops low-variance groups entirely and can shrink the batch.
   - Start with `0.0` (disabled) and calibrate based on short runs.

If the goal is instead “Pass@k should help learning even when rewards are tied”:
- That’s not something Pass@k (or any within-group ranking-based method) can do without changing the reward signal,
  because tied rewards provide no ordering information.
