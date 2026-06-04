# Pass@k Training: Exact Implementation and Reproduction Process

This document records the exact process used in this repo to add and run the Pass@k baseline variant for GRPO, with emphasis on reproducibility and portability to other codebases.

It is written as a practical playbook, not a high-level summary.

Companion results document:
- `pass_8_results.md` (run outcomes and result-focused notes)
- `pass_k_ours_results.md` (iterative/ours-method pass@k investigation: effective batch + reward variance)
- `reports/passk_effective_batch/root_cause_report.md` (evidence-backed root cause of baseline-vs-iterative effective-batch behavior)

---

## 1) Objective and Constraints

Goal:
- Add an opt-in baseline switch `pass_k_training` that changes only per-prompt advantage construction.
- Keep baseline behavior unchanged when disabled (default-off).
- Match baseline run `82fpo6l0` hyperparameters exactly, except:
  - rollout count `n=16` (instead of `n=8`)
  - Pass@k subset size `k=4`
- Keep `adv_estimator=grpo`; do not switch to a different estimator.

Non-goals:
- No production hardening.
- No trainer architecture rewrite.
- No changes to reward function semantics for this variant.

---

## 2) Where Pass@k Lives in This Repo

Core algorithm and integration:
- `verl/trainer/ppo/core_algos.py`
  - `passk_advantages_max_subsets(...)`
  - `compute_grpo_outcome_advantage(...)`

Launcher plumbing:
- `train_chess.sh`
  - env knobs: `PASS_K_TRAINING`, `PASS_K`
  - hydra wiring: `algorithm.pass_k_training`, `algorithm.pass_k_k`
- `sbatch_train_chess_gh200.slurm`
  - cluster env knobs: `CHESS_PASS_K_TRAINING`, `CHESS_PASS_K`
  - export into `train_chess.sh` env

Manual correctness checks:
- `scripts/check_pass_k_advantage.py`

Baseline-aligned launch entrypoint used for final runs:
- `launch_baseline_passk_qwen3b.sh`

---

## 3) Exact Pass@k Algorithm Used

The implementation computes analytic Pass@k advantages over all size-`k` subsets (without replacement), per prompt group.

Given per-prompt rollout rewards:
- `rewards = [r_1, ..., r_N]`
- `1 <= k <= N`

Compute:
- Group-max distribution mean/std: `(mu, sigma)`
- Per-response conditional expected max: `E_i = E[R | i in S]`
- Advantage: `A_i = (E_i - mu) / sigma`

### 3.1 Important edge cases

1. `k == 1`:
- Falls back to GRPO-style within-group normalization.
- In this repo, parity uses sample std (`torch.std` default behavior), plus epsilon in denominator.

2. `sigma <= eps`:
- Returns all-zero advantages for the prompt group.
- Avoids exploding noise when group reward is effectively constant.

### 3.2 Numerical recipe

Implementation in `passk_advantages_max_subsets` uses stable recurrences (no huge binomial integers):

1. Sort rewards descending.
2. Unconditional max weights `w_j`:
- `w_1 = k/N`
- `w_{j+1} = w_j * (N-j-k+1)/(N-j)`
  - only ranks `j <= N-k+1` are non-zero (implemented via `last_nonzero = N-k`)
3. Compute:
- `mu = sum_j w_j r_(j)`
- `m2 = sum_j w_j r_(j)^2`
- `sigma = sqrt(max(m2 - mu^2, 0))`
4. Conditional coefficients:
- `beta_1 = (k-1)/(N-1)`
- `beta_{j+1} = beta_j * (N-j-k+1)/(N-j-1)`
- `alpha_1 = 1`
- `alpha_{p+1} = alpha_p * (N-p-k+1)/(N-p)`
  - `alpha_p`/`beta_j` beyond `N-k+1` are left at zero
5. Prefix:
- `prefix[p] = sum_{j < p} r_(j) beta_j`
6. Conditional expected max:
- `E_p = r_(p) alpha_p + prefix[p]`
7. Advantage:
- `A_(p) = (E_p - mu) / sigma`
8. Map sorted advantages back to original rollout order.

Implementation notes for exact parity:
- For `k==1`, the function uses sample std (`n-1` denominator) and returns
  `(x - mu) / (sig + eps)`.
- For `k>1`, normalization is `(E_p - mu) / sig` with no extra `+eps` in the denominator;
  zero-signal handling is done by the explicit `sig <= eps` early return.
- In trainer integration, `eps` is passed from GRPO's `epsilon` argument
  (`compute_grpo_outcome_advantage`, default `1e-6`).

---

## 4) How It Is Wired Into GRPO (and Only GRPO Advantage)

In `compute_grpo_outcome_advantage(...)`:
- Default:
  - `pass_k_training = False`
  - `pass_k_k = 1`
- If config has `pass_k_training=True`, then for each prompt group:
  - read scalar outcome rewards per sample
  - call `passk_advantages_max_subsets(rewards, pass_k_k, eps)`
  - write returned advantages back to that group's samples
- Otherwise use standard GRPO mean/std normalization path.
- The Pass@k branch writes already-normalized group advantages directly and does not use
  `norm_adv_by_std_in_grpo`; that flag only affects the non-Pass@k GRPO branch.
- This integration is inside `adv_estimator=grpo`.
  It does not use the separate `AdvantageEstimator.GRPO_PASSK` implementation.

No other PPO/GRPO logic is changed:
- policy loss/clipping path unchanged
- KL handling unchanged
- sampling pipeline unchanged

---

## 5) Runtime Knobs and Their Mapping

### 5.1 User-facing env knobs

Local/launcher env:
- `PASS_K_TRAINING=True|False`
- `PASS_K=<int>`

Cluster submission env:
- `CHESS_PASS_K_TRAINING=True|False`
- `CHESS_PASS_K=<int>`

### 5.2 Hydra config fields

Forwarded by `train_chess.sh`:
- `algorithm.pass_k_training=${PASS_K_TRAINING}`
- `algorithm.pass_k_k=${PASS_K}`

---

## 6) Baseline Alignment Process (Critical)

### 6.1 Source of truth

Use W&B baseline run `82fpo6l0` as the canonical envelope.

Extract config evidence:
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 conda run -n verl \
  python scripts/download_wandb_run_evidence.py \
    --entity gabr1e11 --project chess_rl --run 82fpo6l0 \
    --outdir analysis/wandb_evidence/82fpo6l0
```

Lock these key fields to match baseline:
- model path: `Qwen2.5-3B-Instruct` project-local path
- reward: `expected_score_wdl_vs_best`
- train/gen batch size: `128`
- prompt/response length: `1536`/`2000`
- `adv_estimator=grpo`
- `filter_groups.enable=False`
- `trainer.nnodes=1`, `trainer.n_gpus_per_node=4`
- `trainer.total_epochs=1`
- `trainer.total_training_steps=None` (full run)
- `trainer.val_before_train=True` (full run)
- `trainer.test_freq=40`, `save_freq=80`, `full_eval_freq=80`
- full eval prompt template:
  `recipe/chess/prompt_templates/original_chessr1_prompt.jinja`

Intended differences only:
- `actor_rollout_ref.rollout.n=16`
- `algorithm.pass_k_training=True`
- `algorithm.pass_k_k=4`

### 6.2 Filter-groups pitfall and fix

Observed issue:
- A full run had `filter_groups` effectively active, causing non-baseline behavior.

Fixes applied:
1. Force baseline default in `train_chess.sh`:
- baseline job names default `FILTER_GROUPS_ENABLE=False` unless explicitly overridden.
2. Add `[CONFIG] FILTER_GROUPS_ENABLE=...` log line for unambiguous verification.
3. Keep explicit `FILTER_GROUPS_ENABLE=False` in baseline launch script anyway.

---

## 7) Baseline-Aligned Launcher Added

File:
- `launch_baseline_passk_qwen3b.sh`

Purpose:
- Fork baseline launch behavior from `launch_baselines.sh`.
- Pin exact 3B baseline envelope.
- Expose only needed variation knobs.

Modes:
- Smoke mode (`SMOKE=1`, default):
  - `total_training_steps=1`
  - `val_before_train=False`
- Full mode (`SMOKE=0`):
  - `total_training_steps=null`
  - `val_before_train=True`

Wait behavior:
- `WAIT=1` => submit with `sbatch --wait`
- `WAIT=0` => submit without wait

---

## 8) Manual Correctness Checks for Pass@k Math

Run:
```bash
python scripts/check_pass_k_advantage.py
```

Checks included:
1. Default-off parity (`config=None` vs explicit `pass_k_training=False`)
2. `k=1` parity vs existing GRPO normalization path
3. `sigma <= eps` returns all-zero advantages
4. Small-`N` brute-force subset agreement (`mu`, `sigma`, and `adv`)
5. Stable mapping back to original rollout order (permutation invariance test)

This script is intentionally print-based for research debugging.

---

## 9) Alignment Smoke Workflow (What We Actually Used)

### 9.1 Submit smoke (foreground wait)

```bash
ssh a5l.aip2.isambard '
  cd /home/a5l/ziyan.a5l/code/chess-rl &&
  SMOKE=1 WAIT=1 CHESS_RL_HF_UPLOAD_ENABLE=False ./launch_baseline_passk_qwen3b.sh
'
```

Concrete completed smoke example:
- Slurm job: `2265298`
- Job name: `cr1-baseline-passk-qwen3b-smoke`
- State: `COMPLETED`
- W&B run: `sxu6aw69`

### 9.2 Verify effective config from Slurm log

Log example:
- `/projects/a5l/ziyan/chess_rl/logs/slurm-cr1-baseline-passk-qwen3b-smoke-2265298.out`

Check at minimum:
- `[CONFIG] FILTER_GROUPS_ENABLE=False`
- `algorithm.filter_groups.enable: False`
- `pass_k_training: True`
- `pass_k_k: 4`
- `rollout_n: 16`
- plus baseline envelope fields listed in Section 6.

### 9.3 Diff smoke config vs baseline config

Download smoke run evidence:
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 conda run -n verl \
  python scripts/download_wandb_run_evidence.py \
    --entity gabr1e11 --project chess_rl --run sxu6aw69 \
    --outdir analysis/wandb_evidence/sxu6aw69
```

Expected diff pattern vs `82fpo6l0`:
- Intended:
  - `rollout.n: 8 -> 16`
  - `algorithm.pass_k_training: missing -> True`
  - `algorithm.pass_k_k: missing -> 4`
- Smoke-only:
  - `trainer.total_training_steps: None -> 1`
  - `trainer.val_before_train: True -> False`
- Path/hash derived differences (output dirs, config hash, experiment name suffix).
- No unexpected algorithm/hparam drift (especially `filter_groups.enable`).

---

## 10) Full-Run Submission Workflow

Submit full run (no wait):
```bash
ssh a5l.aip2.isambard '
  cd /home/a5l/ziyan.a5l/code/chess-rl &&
  SMOKE=0 WAIT=0 \
  CHESS_PASS_K_TRAINING=True CHESS_PASS_K=4 \
  CHESS_RL_HF_UPLOAD_ENABLE=True \
  CHESS_RL_HF_CKPT_REPO_ID=Gabr1e11/a_lot_of_models \
  CHESS_RL_HF_TOKEN_FILE=/home/a5l/ziyan.a5l/.hf_token \
  ./launch_baseline_passk_qwen3b.sh
'
```

Concrete submitted full example:
- Slurm job: `2265658`
- Job name: `cr1-baseline-passk-qwen3b-full`
- Output root:
  `/projects/a5l/ziyan/chess_rl_outputs/cr1_passk_qwen3b_full_20260210_202348`

For result-focused tracking and outcome summaries, see:
- `pass_8_results.md`

---

## 11) W&B and HF Upload Requirements

W&B:
- Use online mode for real runs:
  - `CHESS_RL_WANDB_MODE=online`
- Confirm run appears in `gabr1e11/chess_rl`.

HF checkpoint upload:
- Must enable:
  - `CHESS_RL_HF_UPLOAD_ENABLE=True`
- Must set target repo:
  - `CHESS_RL_HF_CKPT_REPO_ID=Gabr1e11/a_lot_of_models`
- Must provide valid token file:
  - `CHESS_RL_HF_TOKEN_FILE=/home/a5l/ziyan.a5l/.hf_token`
  - permissions should be `600`

Notes:
- Cluster launcher intentionally unsets HF token env vars when HF upload is disabled.
- Keep HF upload explicitly enabled in full run submissions.

---

## 12) Porting This Approach to Another Codebase

Minimum port checklist:
1. Keep existing GRPO estimator, replace only within-group advantage computation.
2. Implement analytic Pass@k max-subset math with stable recurrences and edge-case handling.
3. Add opt-in config:
  - `pass_k_training` (default `False`)
  - `pass_k_k` (default `1`)
4. Thread knobs from cluster launcher -> train wrapper -> trainer config.
5. Add print-based/manual correctness checks:
  - default-off parity
  - `k=1` parity
  - `sigma<=eps`
  - brute-force small-`N`
  - original-order mapping stability
6. Build a baseline-aligned launcher script to avoid ad-hoc env drift.
7. Validate with a one-step smoke run and explicit config diff against a known baseline run.

---

## 13) Known Failure Modes and Fast Fixes

1. `filter_groups` silently on:
- Symptom: rejected-group dump files and `filter_groups/*` metrics active.
- Fix: force `FILTER_GROUPS_ENABLE=False` in launcher and verify in log.

2. Baseline mismatch due hidden defaults:
- Symptom: config drift in unrelated fields.
- Fix: baseline-aligned launcher script + config diff against reference W&B run.

3. Pass@k no-signal groups (`sigma` near zero):
- Symptom: unstable/noisy gradients if not handled.
- Fix: return zero advantages when `sigma<=eps`.

4. Slurm `--export` list corruption with comma-containing values:
- Symptom: malformed env or missing fields.
- Fix: pass list-like values (for example `CHESS_RL_VAL_FILES`) via prefixed env assignment before `sbatch`, not inside comma-separated `--export`.

---

## 14) Files Touched for This Process

- `verl/trainer/ppo/core_algos.py`
- `train_chess.sh`
- `sbatch_train_chess_gh200.slurm`
- `scripts/check_pass_k_advantage.py`
- `launch_baseline_passk_qwen3b.sh`

---

## 15) Quick Re-run Commands

Smoke alignment:
```bash
ssh a5l.aip2.isambard '
  cd /home/a5l/ziyan.a5l/code/chess-rl &&
  SMOKE=1 WAIT=1 CHESS_RL_HF_UPLOAD_ENABLE=False ./launch_baseline_passk_qwen3b.sh
'
```

Full run (async):
```bash
ssh a5l.aip2.isambard '
  cd /home/a5l/ziyan.a5l/code/chess-rl &&
  SMOKE=0 WAIT=0 CHESS_PASS_K_TRAINING=True CHESS_PASS_K=4 \
  CHESS_RL_HF_UPLOAD_ENABLE=True \
  CHESS_RL_HF_CKPT_REPO_ID=Gabr1e11/a_lot_of_models \
  CHESS_RL_HF_TOKEN_FILE=/home/a5l/ziyan.a5l/.hf_token \
  ./launch_baseline_passk_qwen3b.sh
'
```

---

## 16) Pass@k for Iterative allowed_move_elim (“ours method”)

This repo’s “ours method” training uses the **iterative allowed-move elimination** loop (`allowed_move_elim`)
described in `iterative.md`. In this loop, a single trainer step produces **multiple GRPO uid-groups per prompt**
(typically one group per round when `uid_mode=per_round`).

The key requirement for Pass@k in the iterative method is **two-branch behavior per uid-group**:

1. If the uid-group contains **any optimal rollout**, keep **vanilla GRPO** advantage behavior.
2. If the uid-group contains **no optimal rollouts**, compute advantages via **analytic Pass@k** with:
   - `n = group_size` (the actual number of samples in that uid-group)
   - `k = 4` (fixed for this experiment)

This is intentionally **not** the same as the baseline/global `pass_k_training` switch:
- Baseline Pass@k (`algorithm.pass_k_training=True`) applies Pass@k to *all* GRPO groups.
- Iterative conditional Pass@k applies Pass@k only to **no-optimal** groups under `allowed_move_elim`.

### 16.1 Code path and precedence rules

Implementation is inside the GRPO advantage path:

1. `verl/trainer/ppo/ray_trainer.py` (`compute_advantage`, GRPO branch)
   - Computes a boolean `sample_is_optimal` array **only when**:
     - `algorithm.allowed_move_elim.enable=True`, and
     - `algorithm.allowed_move_elim.pass_k_when_no_optimal=True`.
   - Passes `sample_is_optimal` into:
     - `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage(...)`

2. `verl/trainer/ppo/core_algos.py` (`compute_grpo_outcome_advantage`)
   - Branch order (important):
     1) If `algorithm.pass_k_training=True`, run the **baseline** Pass@k logic (unchanged).
     2) Else if `algorithm.allowed_move_elim.pass_k_when_no_optimal=True` and `sample_is_optimal` is provided:
        - for uid-groups with **no** optimal samples: use `passk_advantages_max_subsets(rewards, k=4)`
        - for uid-groups with **any** optimal sample: use vanilla GRPO normalization within the group
     3) Else: vanilla GRPO behavior for all groups.

Practical consequence:
- For the iterative conditional Pass@k behavior, keep `PASS_K_TRAINING=False` (do **not** enable the global knob),
  and enable only the `allowed_move_elim`-scoped knob.

### 16.2 What counts as “optimal” (exact definition)

The iterative loop already has a success criterion (“did we find the right move within the candidate set?”).
The conditional Pass@k implementation reuses that same definition.

In `verl/trainer/ppo/ray_trainer.py`, a sample is marked optimal iff:
- `penalty_applied == False`, and
- `pred_move` and `gt_uci` are present and match (case-insensitive, whitespace-stripped), and
- if `in_subset` is present, it must be `True`.

Then a uid-group is “has optimal” iff **any** sample in that group is optimal.

### 16.3 Hyperparameters: enforcing `n=group_size` and `k=4`

- `k` comes from `algorithm.allowed_move_elim.pass_k_k` and is intended to be `4` for this experiment.
- `n` is not a separate knob: it is the **actual per-uid group size** (length of the reward list for that group).

To match the intended `n=16` setup:
- Use `actor_rollout_ref.rollout.n=16` (`ROLLOUT_N=16` in launchers).
- Use `algorithm.allowed_move_elim.uid_mode=per_round` so each uid-group corresponds to one (prompt, round)
  and has size exactly `rollout.n` (16).

If `uid_mode=per_prompt` is used, a prompt’s uid-group aggregates across multiple rounds, so `n` can become
`rounds_used * rollout.n` (larger than 16). The conditional Pass@k math still applies (with `n` equal to that
group size), but it is no longer the “n=16” experiment.

### 16.4 Runtime knobs and launcher plumbing

Local env (via `train_chess.sh`):
- `ALLOWED_MOVE_ELIM_ENABLE=True`
- `ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL=True`
- `ALLOWED_MOVE_ELIM_PASS_K_K=4`

Cluster env (via `sbatch_train_chess_gh200*.slurm`):
- `CHESS_ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL=True`
- `CHESS_ALLOWED_MOVE_ELIM_PASS_K_K=4`

Hydra fields wired by `train_chess.sh`:
- `algorithm.allowed_move_elim.pass_k_when_no_optimal=${ALLOWED_MOVE_ELIM_PASS_K_WHEN_NO_OPTIMAL}`
- `algorithm.allowed_move_elim.pass_k_k=${ALLOWED_MOVE_ELIM_PASS_K_K}`

### 16.5 How to verify the conditional path executed

`verl/trainer/ppo/core_algos.py` prints a one-time banner when the conditional path is active:

```
[ALLOWED_MOVE_ELIM_PASSK] enabled=True k=4 groups_total=<...> groups_passk_no_optimal=<...> groups_vanilla_has_optimal=<...>
```

This banner is the simplest “proof” that:
- the run is in iterative `allowed_move_elim` mode,
- the conditional Pass@k branch is on,
- and both branches are being exercised (counts are non-trivial).

For a deeper, results-focused analysis of what this changes (and what it does *not* change),
see:
- `pass_k_ours_results.md`
- `reports/passk_effective_batch/root_cause_report.md`

---

## 17) Why baseline Pass@k improves effective batch, but iterative conditional Pass@k doesn’t

This section documents the **mechanistic reason** we observe a strong improvement in
`grpo/effective_batch_size` / `grpo/effective_batch_frac` for baseline/global Pass@k runs, but a continued
effective-batch drop for iterative/ours runs even when conditional Pass@k is enabled.

The detailed evidence, reproducible commands, run IDs, and custom rollout-log-derived metrics live in:
- `reports/passk_effective_batch/root_cause_report.md`

Primary runs used as evidence (as referenced by `pass_8_results.md` / `pass_k_ours_results.md`):
- Baseline/global Pass@k: `f5guq4ti`
- Baseline references: `82fpo6l0` (no Pass@k), `u2cuw56a` (larger `rollout.n`)
- Iterative conditional Pass@k: `xie1sbcg`
- Iterative reference (no conditional Pass@k): `s0anl08n`

### 17.1 The key intuition (keep this mental model)

The core intuition that motivates the investigation (and is useful for guiding fixes) is:

> 1) Baseline + pass@k works because we apply pass@k across all groups. That creates a consistent incentive:
> for every group, the model is rewarded for producing at least one good move among the k samples.
>
> 2) Our method + pass@k doesn’t work because we only apply pass@k to the groups that don’t contain an optimal move.
> But pass@k is most useful precisely in groups that do contain an optimal move—because then the model can “succeed”
> by including at least one good move in its k candidates and gets a learning signal that pushes it in the right direction.
> Since we exclude the groups that have an optimal move, the remaining groups have essentially no variation / no positive
> signal (effectively std ≈ 0), so pass@k has nothing meaningful to amplify and the model doesn’t learn.

Two clarifications that matter for implementation correctness:
- In the iterative method, we do **not** drop “has-optimal” groups from training; we only skip applying Pass@k to them.
  Those groups still contribute gradients via vanilla GRPO within-group normalization.
- “optimal” in the conditional gate is currently the strict success criterion `pred_move == gt_uci` (plus metadata gates),
  not “μ-best” or “highest-reward sample” in general (see Section 16.2).

### 17.2 What `grpo/effective_batch_size` actually measures (and what it does not)

Before interpreting any curves, keep this definition in mind:

- `grpo/effective_batch_size` is computed by grouping samples by `uid`, then counting how many uid-groups have
  **non-zero within-group reward std**.
- This is implemented in `verl/trainer/ppo/ray_trainer.py` (`_compute_grpo_effective_batch`).
- It uses `token_level_rewards.sum(dim=-1)` as the per-sample scalar reward; for chess this is outcome-only (one token)
  via `verl/workers/reward_manager/batch.py`.

**Implication:** Pass@k advantage construction does **not** change `grpo/effective_batch_size` directly (the metric
depends on reward variance, not on advantages). Pass@k can only affect it **indirectly**, by changing the learned
policy so that future rollout groups have more diverse rewards.

### 17.3 Why splitting into rounds is correct (and why it changes effective-batch dynamics)

Splitting iterative sampling into rounds (one GRPO uid-group per *(prompt, round)* when `uid_mode=per_round`) is the
intended and semantically correct behavior:
- Each round uses a *different* `allowed_moves` candidate set (the state of elimination), so it is a different decision
  problem with different reward parsing context.
- `iterative.md` explicitly defines `uid_mode=per_round` as the default: each round is its own GRPO group.

However, this design has a mechanical side effect on effective batch monitoring:
- Each original prompt can contribute multiple uid-groups per trainer step (one per round used).
- Later rounds are disproportionately “hard” / “no-optimal” (by construction), so the distribution of reward variance
  across uid-groups is not uniform across rounds.
- Therefore, a per-uid effective-batch metric will drift as the fraction of later-round groups grows.

### 17.4 The proved root cause (what actually kills effective batch in iterative conditional Pass@k)

From the artifact-backed analysis in `reports/passk_effective_batch/root_cause_report.md`:

1) In iterative `allowed_move_elim` with `uid_mode=per_round`, a large and growing share of per-round uid-groups become
   **reward-tied** (`std(score) == 0`), especially the **no-optimal** groups (the conditional Pass@k branch).
2) Reward ties are driven by the interaction of:
   - reward quantization / bucket collisions in `expected_score_wdl_vs_best` (many moves map to the same scalar), and
   - shrinking within-group diversity in later rounds (the model repeats moves and stays in the same reward bucket).
3) When a group is reward-tied, it is simultaneously:
   - excluded from `grpo/effective_batch_size` (by metric definition), and
   - unable to benefit from Pass@k (the Pass@k implementation returns all-zero advantages when `sigma <= eps`).

So the effective-batch drop in iterative runs is not “because pass@k wasn’t enabled”; it is because the groups that
conditional Pass@k targets are often exactly the groups with **no reward ranking signal** to amplify.

### 17.5 How this relates to the intuition above (what’s primary vs downstream)

The intuition in Section 17.1 is directionally correct and explains why conditional Pass@k is fragile:
- Applying Pass@k only to “no-optimal” groups focuses the algorithm on the hardest / least-informative groups.
- The groups where Pass@k would be most naturally useful (“at least one good sample exists”) are exactly the groups
  that are routed to vanilla GRPO by the conditional gate.

But the evidence-backed root cause chain has an additional crucial piece:
- The no-optimal groups are not merely “hard”; they become **degenerate** under per-round grouping + quantized reward,
  producing `std(score)==0`, which defeats both the effective-batch metric *and* Pass@k’s advantage construction.

This is why the “sparse/quantized reward” observation in `pass_k_ours_results.md` is a **downstream symptom**:
baseline runs use the same reward function, but do not create as many reward-tied per-round groups.

### 17.6 Practical debugging + mitigation (without changing round-splitting semantics)

If you see `grpo/effective_batch_frac` dropping in an iterative conditional Pass@k run, prioritize these checks:

1) Verify conditional Pass@k is actually active:
   - look for `[ALLOWED_MOVE_ELIM_PASSK] ...` banner in the Slurm log (see Section 16.5).
2) Inspect per-round rollout logs (`allowed_move_elim_rounds/*.jsonl`) and compute:
   - fraction of uid-groups with `std(score)==0` (dead groups),
   - score histograms (many dead groups are constant `score == -1.0`, not penalties),
   - within-group move diversity (unique predicted moves / group).
   The repo script `scripts/analyze_passk_effective_batch_root_cause.py` reproduces these metrics.

Mitigation options that preserve the “one group per round” semantics (evaluate via minimal ablations; no training
runs should be launched without explicit approval):
- Make per-round rewards less tie-prone within the candidate subset (reduce quantization / add finer-grained shaping).
- Improve logging/monitoring: track per-round dead-group rates and diversity metrics alongside effective batch.
- Revisit the conditional gate definition: if “success” is defined by `gt_uci` but reward is defined by expected score,
  consider whether the gate should align to “best-in-subset reward” rather than exact label match.
- Consider applying Pass@k more broadly than “no-optimal only” (for example, also to has-optimal groups), if the goal is
  to preserve the “at least one good sample” learning signal that makes baseline/global Pass@k effective.
