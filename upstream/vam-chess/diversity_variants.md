# Diversity Variants (GRPO Auxiliary Advantages)

This note documents the three diversity-focused auxiliary advantage variants integrated into the chess GRPO/Pass@k training path, plus the run launches performed on Isambard.

Scope:
- Method definitions and exact math for the three diversity variants.
- Repo-specific implementation mapping (answer canonicalization, validity, grouping, and combination).
- Integration approach (where the diversity logic lives, and how the iterative conditional gate reuses the Pass@k branch structure).
- Launch records only (no result analysis in this document).

## Method Definition

### Final combined advantage
- Baseline score term: `A_base` (existing GRPO / optional Pass@k path).
- Diversity auxiliary term: `A_div` (one of OBE-Batch, GAPO, Distinct@k analytic).
- Combination used in code: `A_final = A_base + lambda * A_div` (or diversity-only if base term is disabled).

### Exact Math (Verbatim)

```text
[
  A_{\text{final}}(i) = A_{\text{base}}(i) + \lambda , A_{\text{div}}(i).
  ]

Variant A — OBE “Batch exploration”
[
  b_i = -\frac{c(a_i)-1}{n}.
  ]

Variant B — GAPO frequency-aware reward
[
  f(a) = \frac{#{i: v_i=1 \text{ and } a_i=a}}{#{i: v_i=1}}.
  ]
[
  r^{\text{gapo}}_i =
  \begin{cases}
  1 - \left(f(a_i) - \frac{1}{L}\right), & v_i=1,\
  -1, & v_i=0.
  \end{cases}
  ]

Variant C — Distinct@k analytic advantage
[
  D(S) = \left|{a_i : i\in S}\right|
  \quad \text{for } |S|=k,
  ]

1. Group-level mean
[
  \mu = \mathbb{E}[D(S)]
  = \sum_{t=1}^T \left(1-\frac{\binom{n-c_t}{k}}{\binom{n}{k}}\right).
  ]

2. Group-level variance
* For each type (t): (p_t = 1-\frac{\binom{n-c_t}{k}}{\binom{n}{k}}).
* For each pair (t<u):
  [
  p_{tu} = 1-\frac{\binom{n-c_t}{k}}{\binom{n}{k}}-\frac{\binom{n-c_u}{k}}{\binom{n}{k}}+\frac{\binom{n-c_t-c_u}{k}}{\binom{n}{k}}.
  ]
  Then:
  [
  \mathbb{E}[D(S)^2] = \sum_t p_t + 2\sum_{t<u} p_{tu},\quad
  \sigma^2=\mathbb{E}[D(S)^2]-\mu^2.
  ]
  Set (\sigma=\sqrt{\max(\sigma^2,0)}). If (\sigma) is ~0, return all-zero advantages.

3. Conditional mean for each rollout
[
  q_t = \frac{\binom{(n-1)-c_t}{k-1}}{\binom{n-1}{k-1}}.
  ]
Let (Q=\sum_t q_t). For a rollout (i) whose answer type is (s):
[
  \mu_i = T - (Q - q_s).
  ]

4. Per-rollout analytic advantage
[
  A_{\text{div}}(i)=\frac{\mu_i-\mu}{\sigma}.
  ]
```

## Repo Mapping (Chess)

- Canonical answer ID `a_i`:
  - `pred_move` (lowercased) when compliant.
  - Deterministic sentinel `__invalid__` otherwise.
- Validity `v_i`:
  - `True` iff `penalty_applied=False`, `in_subset=True`, and non-empty predicted move.
- Candidate size `L`:
  - Per sample from `n_considered_moves`, with fallback to `reward_model.considered_moves_uci` / `reward_model.considered_moves_uci_list` / `legal_moves_uci`, then `1`.
- Grouping:
  - Diversity statistics and `A_div` are computed per GRPO `uid` group.
- Z-scoring:
  - OBE-Batch and GAPO raw values are z-scored per group.
  - If group std is near zero, diversity outputs are zeros for that group.
- Distinct@k:
  - Computed analytically (no sampled subsets) with robust combinatorial ratios.
- Baseline-preserving defaults:
  - `algorithm.diversity.enable=False` and `algorithm.diversity.lambda_coeff=0.0` preserve baseline behavior.

## Config / Launcher Surface

Hydra keys:
- `algorithm.diversity.enable`
- `algorithm.diversity.method` (`none|obe_batch|gapo|distinct_k`)
- `algorithm.diversity.lambda_coeff`
- `algorithm.diversity.distinct_k`
- `algorithm.diversity.include_base_advantage`
- (iterative only) `algorithm.allowed_move_elim.diversity_when_no_optimal`

Environment keys (train/Slurm wiring):
- `DIVERSITY_ENABLE` / `CHESS_DIVERSITY_ENABLE`
- `DIVERSITY_METHOD` / `CHESS_DIVERSITY_METHOD`
- `DIVERSITY_LAMBDA` / `CHESS_DIVERSITY_LAMBDA`
- `DIVERSITY_DISTINCT_K` / `CHESS_DIVERSITY_DISTINCT_K`
- `DIVERSITY_INCLUDE_BASE_ADVANTAGE` / `CHESS_DIVERSITY_INCLUDE_BASE_ADVANTAGE`
- (iterative only) `ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL` / `CHESS_ALLOWED_MOVE_ELIM_DIVERSITY_WHEN_NO_OPTIMAL`
- `DATA_SEED` / `CHESS_RL_DATA_SEED`

## Iterative allowed_move_elim: conditional diversity gate

This repo’s iterative (“ours method”) training uses `allowed_move_elim` and produces multiple GRPO `uid` groups.

When `algorithm.allowed_move_elim.diversity_when_no_optimal=True`, diversity advantages are applied **only** to
uid-groups that contain **no** optimal rollout (same success criterion as the conditional Pass@k gate):
- **has-optimal** uid-groups: vanilla GRPO advantages (no diversity auxiliary term)
- **no-optimal** uid-groups: selected diversity variant is applied (OBE / GAPO / Distinct@k)

Runtime proof banner (printed once in `compute_grpo_outcome_advantage` when active):
`[ALLOWED_MOVE_ELIM_COND_DIVERSITY] enabled=True ... groups_diversity_no_optimal=... groups_vanilla_has_optimal=...`

## Integration Approach (How It’s Wired)

This section describes how the three diversity variants are integrated into GRPO training in this repo, with special
attention to the iterative `allowed_move_elim` (“ours method”) contract:

- **has-optimal** uid-groups: vanilla GRPO advantages (**no diversity auxiliary term**)
- **no-optimal** uid-groups: apply the selected diversity variant (OBE / GAPO / Distinct@k)

The implementation intentionally reuses the existing **conditional Pass@k** branch structure to avoid duplicating
trainer logic.

### Single advantage entrypoint (GRPO)

All GRPO outcome advantages (vanilla GRPO, global Pass@k, conditional Pass@k, and diversity) flow through a single
function:

- `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage` (registered for `AdvantageEstimator.GRPO`)

This function computes:
1) baseline within-group advantages `A_base` (GRPO or Pass@k depending on config),
2) group-level diversity diagnostics (always),
3) optional diversity auxiliary term `A_div` (method-dependent),
4) and the final combined score `A_final`.

### Trainer-side preprocessing (what the advantage function needs)

In the GRPO advantage branch of `verl/trainer/ppo/ray_trainer.py`, immediately before calling
`compute_grpo_outcome_advantage`, the trainer derives the per-sample inputs used by the diversity methods:

- `answer_ids` (canonical answer ID `a_i`): lowercased `pred_move` when compliant, else `__invalid__`
- `sample_valid` (validity `v_i`): `True` iff `(penalty_applied=False) AND (in_subset=True) AND (pred_move != '')`
- `candidate_sizes` (candidate size `L`): derived from `n_considered_moves`, with fallback to the `reward_model` struct

For iterative conditional modes, the trainer also computes:

- `sample_is_optimal`: `True` iff the model’s predicted move matches `gt_uci` exactly (and is not penalized / is in-subset)

This `sample_is_optimal` array is **only computed when needed**, i.e. when:
- `algorithm.allowed_move_elim.enable=True`, and
- either `algorithm.allowed_move_elim.pass_k_when_no_optimal=True` (conditional Pass@k) **or**
  `algorithm.allowed_move_elim.diversity_when_no_optimal=True` (conditional diversity)

### Branching logic (Pass@k and diversity share the same “has-optimal” split)

Inside `compute_grpo_outcome_advantage` the high-level flow is:

1) **Group** rollouts by GRPO group id (`index` array provided by the trainer; corresponds to `uid` groups).
2) **Compute `A_base`**:
   - If `algorithm.pass_k_training=True`: apply Pass@k advantages to *all* groups.
   - Else if `algorithm.allowed_move_elim.pass_k_when_no_optimal=True`: apply Pass@k advantages only to *no-optimal* groups; has-optimal groups keep vanilla GRPO normalization.
   - Else: vanilla GRPO normalization for all groups.
3) **Compute diversity diagnostics** per group (unique-answer count, top frequency, collisions, etc.) for logging
   regardless of whether diversity training is enabled.
4) If `algorithm.diversity.enable=True` and `algorithm.diversity.method!=none`, **compute `A_div`** per group for the
   selected method.
5) **Combine**:
   - Global diversity (baseline behavior): `A_final = A_base + lambda * A_div` for all groups.
   - Conditional diversity (iterative-only): if `algorithm.allowed_move_elim.diversity_when_no_optimal=True`,
     apply the `lambda * A_div` term only for *no-optimal* groups; has-optimal groups use vanilla GRPO.

Important nuance: the conditional diversity gate is only “active” when diversity is actually enabled, i.e.
`allowed_move_elim.diversity_when_no_optimal=True` **and** `algorithm.diversity.enable=True` **and**
`algorithm.diversity.method!=none`. If diversity is disabled, the conditional flag is a no-op.

#### Summary table (iterative `allowed_move_elim`)

| Settings | has-optimal groups | no-optimal groups |
|---|---|---|
| `pass_k_when_no_optimal=False`, `diversity_when_no_optimal=True`, diversity enabled | vanilla GRPO | GRPO + diversity |
| `pass_k_when_no_optimal=True`, `diversity_when_no_optimal=True`, diversity enabled | vanilla GRPO | Pass@k (base) + diversity |
| `diversity_when_no_optimal=False`, diversity enabled | GRPO + diversity | GRPO + diversity |
| diversity disabled | vanilla GRPO / (optional Pass@k modes) | vanilla GRPO / (optional Pass@k modes) |

### Diversity-only mode and the iterative contract

`algorithm.diversity.include_base_advantage=False` requests “diversity-only” advantages (i.e., dropping `A_base`).

In conditional diversity mode, the implementation still includes `A_base` for **has-optimal** groups so the contract
“has-optimal → vanilla GRPO” is preserved even under diversity-only settings.

### Logging and verification hooks

When debugging whether the intended branch executed:

- Conditional Pass@k prints once per job: `[ALLOWED_MOVE_ELIM_PASSK] ...`
- Conditional diversity prints once per job: `[ALLOWED_MOVE_ELIM_COND_DIVERSITY] ... groups_diversity_no_optimal=...`
- Diversity scalar metrics are emitted under `diversity/*` via `RayPPOTrainer._compute_diversity_metrics` and flow to:
  - console logs, and
  - W&B (when `trainer.logger` includes `"wandb"`; `train_chess.sh` sets `["console","wandb"]`)

The logged key `diversity/enabled` reflects whether the diversity auxiliary term was active (`1.0`) vs disabled (`0.0`).

## Launch Scripts

Primary launchers used:
- `launch_diversity_smoke_qwen3b.sh`
- `launch_diversity_core_comparison_qwen3b.sh`
- (iterative conditional diversity) `launch_iterative_diversity_smoke_qwen3b.sh`

Typical commands used:

```bash
# Smoke suite (1 step, no val/eval)
ssh a5l.aip2.isambard 'cd ~/code/chess-rl && VARIANT=all ./launch_diversity_smoke_qwen3b.sh'

# Iterative conditional-diversity smoke suite (1 step, sbatch --wait)
ssh a5l.aip2.isambard 'cd ~/code/chess-rl && VARIANT=all SMOKE=1 WAIT=1 ./launch_iterative_diversity_smoke_qwen3b.sh'

# Full core comparison (no wait)
ssh a5l.aip2.isambard 'cd ~/code/chess-rl && WAIT=0 SMOKE=0 VARIANT=all ./launch_diversity_core_comparison_qwen3b.sh'

# Vanilla GRPO baseline (no pass@k, no diversity)
ssh a5l.aip2.isambard 'cd ~/code/chess-rl && WAIT=0 SMOKE=0 VARIANT=baseline CHESS_PASS_K_TRAINING=False CHESS_PASS_K=1 CHESS_DIVERSITY_ENABLE=False CHESS_DIVERSITY_METHOD=none CHESS_DIVERSITY_LAMBDA=0.0 CHESS_RL_DATA_SEED=3407 ./launch_diversity_core_comparison_qwen3b.sh'

# Pass@k baseline (no diversity)
ssh a5l.aip2.isambard 'cd ~/code/chess-rl && WAIT=0 SMOKE=0 VARIANT=baseline CHESS_PASS_K_TRAINING=True CHESS_PASS_K=4 CHESS_DIVERSITY_ENABLE=False CHESS_DIVERSITY_METHOD=none CHESS_DIVERSITY_LAMBDA=0.0 CHESS_RL_DATA_SEED=3407 ./launch_diversity_core_comparison_qwen3b.sh'
```

## Launched Runs (No Analysis)

Source of truth for states below: `sacct` on Isambard as queried on 2026-02-19.

### Smoke Runs

| Job ID | Job Name | Variant | Run Tag | State | ExitCode | Elapsed |
|---|---|---|---|---|---|---|
| 2342818 | `cr1-div-baseline-smoke` | baseline control (initial attempt) | `cr1_diversity_baseline_smoke_20260217_081646` | FAILED | `1:0` | `00:00:53` |
| 2342848 | `cr1-div-control-smoke` | baseline control | `cr1_diversity_baseline_smoke_20260217_081918` | COMPLETED | `0:0` | `00:07:45` |
| 2342913 | `cr1-div-obe-smoke` | OBE | `cr1_diversity_obe_smoke_20260217_082751` | COMPLETED | `0:0` | `00:07:25` |
| 2342974 | `cr1-div-gapo-smoke` | GAPO | `cr1_diversity_gapo_smoke_20260217_083539` | COMPLETED | `0:0` | `00:07:23` |
| 2343017 | `cr1-div-distinct-smoke` | Distinct@k | `cr1_diversity_distinct_smoke_20260217_084354` | COMPLETED | `0:0` | `00:08:31` |
| 2343185 | `cr1-div-obe-smoke` | OBE (`ALLOWED_MOVE_ELIM_ENABLE=True` compatibility check) | `cr1_diversity_obe_smoke_20260217_085603` | COMPLETED | `0:0` | `00:08:34` |

### Iterative Conditional-Diversity Smoke Runs

Source of truth for states below: `sacct` on Isambard as queried on 2026-02-20.

| Job ID | Job Name | Variant | Run Tag | State | ExitCode | Elapsed |
|---|---|---|---|---|---|---|
| 2382324 | `cr1-iterdiv-obe-smoke` | OBE (conditional, iterative) | `cr1_iterdiv_obe_smoke_20260220_035604` | COMPLETED | `0:0` | `00:09:14` |
| 2382338 | `cr1-iterdiv-gapo-smoke` | GAPO (conditional, iterative) | `cr1_iterdiv_gapo_smoke_20260220_040550` | COMPLETED | `0:0` | `00:09:16` |
| 2382354 | `cr1-iterdiv-distinct-smoke` | Distinct@k (conditional, iterative) | `cr1_iterdiv_distinct_smoke_20260220_041507` | COMPLETED | `0:0` | `00:09:05` |

### Full Core-Comparison: First Attempt (preflight failure)

| Job ID | Job Name | Variant | Run Tag | State | ExitCode | Elapsed |
|---|---|---|---|---|---|---|
| 2343493 | `cr1-div-core-obe-full` | OBE | `cr1_diversity_core_obe_full_20260217_092526` | FAILED | `1:0` | `00:00:56` |
| 2343494 | `cr1-div-core-gapo-full` | GAPO | `cr1_diversity_core_gapo_full_20260217_092526` | FAILED | `1:0` | `00:00:58` |
| 2343495 | `cr1-div-core-distinct-full` | Distinct@k | `cr1_diversity_core_distinct_full_20260217_092526` | FAILED | `1:0` | `00:00:58` |

### Full Core-Comparison: Relaunch

| Job ID | Job Name | Variant | Run Tag | State | ExitCode | Elapsed |
|---|---|---|---|---|---|---|
| 2343523 | `cr1-baseline-div-core-obe-full` | OBE | `cr1_diversity_core_obe_full_20260217_093543` | TIMEOUT | `0:0` | `1-00:00:17` |
| 2343524 | `cr1-baseline-div-core-gapo-full` | GAPO | `cr1_diversity_core_gapo_full_20260217_093543` | TIMEOUT | `0:0` | `1-00:00:17` |
| 2343525 | `cr1-baseline-div-core-distinct-full` | Distinct@k | `cr1_diversity_core_distinct_full_20260217_093543` | TIMEOUT | `0:0` | `1-00:00:17` |

### Reference Baseline Launches (same envelope)

| Job ID | Job Name | Mode | Run Tag | State | ExitCode | Elapsed |
|---|---|---|---|---|---|---|
| 2343492 | `cr1-div-core-baseline-full` | baseline control (later cancelled) | `cr1_diversity_core_baseline_full_20260217_092526` | CANCELLED by user | `0:0` | `00:08:49` |
| 2355211 | `cr1-baseline-div-core-baseline-full` | vanilla GRPO baseline (no pass@k, no diversity) | `cr1_diversity_core_baseline_full_20260217_183745` | CANCELLED by user | `0:0` | `00:00:00` |
| 2355242 | `cr1-baseline-div-core-baseline-full` | vanilla GRPO baseline (relaunch) | `cr1_diversity_core_baseline_full_20260217_184507` | CANCELLED by user | `0:0` | `00:00:00` |
| 2368513 | `cr1-baseline-div-core-baseline-full` | vanilla GRPO baseline (relaunch) | `cr1_diversity_core_baseline_full_20260218_203340` | CANCELLED by user | `0:0` | `10:40:06` |
| 2370624 | `cr1-baseline-div-core-baseline-full` | Pass@k baseline (`k=4`, diversity off) | `cr1_diversity_core_baseline_full_20260219_071317` | CANCELLED by user | `0:0` | `13:29:56` |

## Related Files

- `docs/diversity_advantages_math.md`
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/ray_trainer.py`
- `launch_diversity_smoke_qwen3b.sh`
- `launch_diversity_core_comparison_qwen3b.sh`
