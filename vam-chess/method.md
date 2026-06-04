# Forced-Guess GRPO (Chess Move Selection)

This document describes the current RL method used in this repo to train a language model to select a strong
chess move from a single position (FEN + legal moves).

The method is “GRPO + forced-prefix exploration”, but the forced prefix is now a **tag-based `<guess>` block**
instead of natural-language scaffolding.

For concrete entry points (launcher scripts, reward implementation, rollout injection code), see `AGENTS.md`.

---

## Overview

Training iterates:

1. For each prompt, generate `N = rollout.n` rollouts.
2. For each rollout independently, sample `is_forced ~ Bernoulli(p(t))` (annealed over steps).
   - If forced: inject a forced prefix on the **response side**:
     - `<guess> {move} </guess>`
   - If free: inject no forced prefix.
   - In both cases, the *prompt* still requests the guess-first format, so a correct response is always:
     - `<guess>...</guess><think>...</think><uci_move>...</uci_move>`
3. Score each rollout with the rule-based chess reward (`recipe/chess/reward_fn.py`).
4. Compute GRPO advantages within prompt groups keyed by `uid` (no forced/free stratification).
5. Update the policy on **model-generated continuation tokens** only (forced prefix tokens are excluded from the RL actor loss).

Important breaking changes relative to older versions of this repo:

- Forced-prefix injection is **per rollout**, not “all-forced vs all-free per uid group”.
- Advantage estimation groups by `uid` only (no forced/free suffix).
- The forced-rollout auxiliary teacher-forcing loss is removed.

---

## 1) Task setup (what the model is trained to do)

Each training example is a one-step decision problem:

- Input: a chess position in **FEN** plus a list of legal moves in **UCI**.
- Output: the model is prompted to emit:
  1. A leading guess line (format-required, advisory for scoring):

     ```text
     <guess> GUESS_UCI </guess>
     ```

  2. Then the strict answer contract:

     ```text
     <think> ... </think><uci_move> ... </uci_move>
     ```

Only the `<uci_move>` payload is used for chess scoring; the `<guess>` payload is ignored.

---

## 2) Data carried per example (reward payload)

Each row contains a `reward_model` dict (passed to the reward function) including:

- `fen`: the FEN string.
- `legal_moves_uci`: list of legal moves in UCI.
- `ground_truth`: target move in UCI (lowercase).
- `move_values_json`: JSON map `{uci_move: float}` in `[0, 1]` (winrate-like shaping).
- Optional richer maps/baselines for alternative reward shapings (e.g., `move_expected_scores_json`, `move_cps_json`).

This payload supports:

1. Reward computation.
2. Forced-move sampling for forced-prefix exploration (prefer expected score map when available).

---

## 3) Reward function (strict gate + shaped per-move score)

The reward function is rule-based and applied per rollout.

### 3.1 Format gate

The decoded response must strictly match:

```text
<guess> ... </guess><think> ... </think><uci_move> ... </uci_move>
```

with only whitespace allowed around/between the tags.

Notes:

- The `<guess>` block is **required** for the strict format gate.
- Tag casing is ignored (e.g., `<GUESS>` is accepted), but tags must not contain extra whitespace
  (e.g., `<guess >` is rejected).
- The `<guess>` payload is not validated by the reward function (it may be non-UCI); it is logged for analysis only.

### 3.2 Move extraction

The `<uci_move>` payload must be exactly one strict UCI move:

- `from_square + to_square` with optional promotion `q/r/b/n`
- SAN is disallowed

Failures (format errors, bad UCI, illegal/unscorable move) are penalized (default `-1`).

### 3.3 Scalar reward and exact-match metric

When the move is valid and present in the eval map:

- Default shaping (`chess_reward_fn=winrate`): `move_values_json[pred_move]`

The reward function also emits:

- `acc` / `exact_match`: `1.0` iff `pred_move == ground_truth`, else `0.0`.

---

## 4) Forced-prefix exploration (the `<guess>` scheme)

### 4.1 What “forced prefix” means

For some rollouts, we do not let the model freely choose the first response tokens. Instead, we **prepend** a
forced prefix to the prompt token IDs before decoding, so generation begins after that prefix.

Default forced-prefix template (response-side):

```text
<guess> {move} </guess>
```

### 4.2 Sampling the forced move

When a rollout is forced, the `{move}` is sampled from the per-example move map:

- Prefer `move_expected_scores_json` (bounded expected score in `[0, 1]`).
- Fall back to `move_values_json` (winrate-like in `[0, 1]`).

Sampling:

1. Drop moves with value `<= 0`.
2. If the filtered set is empty, fall back to `ground_truth`.
3. Apply a power-law temperature:
   - `weight = value ** (1 / T)` where `T = forced_prefix.move_temperature`.
4. Normalize weights and sample a move.

### 4.3 When forcing is applied (per-rollout Bernoulli)

Let `p(t)` be an annealed schedule over training steps.

For each prompt group of size `N`, each rollout independently samples:

```text
is_forced_i ~ Bernoulli(p(t))
```

So a single `uid` group can contain a mix of forced and free rollouts.

### 4.4 Token-level injection and masks

For each rollout we attach:

- `forced_prefix_token_ids`: token IDs of the forced prefix (empty for free rollouts).

At rollout time we construct:

- `response_token_ids = forced_prefix_token_ids + generated_token_ids`
- `forced_token_mask`: boolean mask that is `True` on the forced prefix span.

---

## 5) GRPO grouping and advantages

Each prompt is assigned a `uid` and repeated `N` times, forming a rollout group.

- Group key: `uid` only (no forced/free stratification).

Let each rollout have scalar reward `R_i`. GRPO computes a per-group baseline (mean and optionally std) and
produces scalar advantages `A_i` used for token-level policy gradient on the response tokens.

---

## 6) Actor loss masking (no auxiliary loss)

Forced-prefix tokens are externally injected; we treat them as conditioning context.

- The actor RL loss mask excludes forced tokens:
  - `loss_mask = response_mask AND (NOT forced_token_mask)`

There is no forced-rollout auxiliary teacher-forcing loss in the current codebase.

---

## 7) Key implementation entry points

- Reward parsing/scoring: `recipe/chess/reward_fn.py`
- Forced-prefix sampling + injection metadata: `verl/trainer/ppo/ray_trainer.py`
- vLLM prompt+prefix concatenation and forced-token masking: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- Actor loss masking: `verl/workers/actor/dp_actor.py`
- Training launcher: `train_chess.sh`
