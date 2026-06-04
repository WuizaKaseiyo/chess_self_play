# LLM Gating

## Motivation

The current iterative gate in `verl/trainer/ppo/ray_trainer.py` is a logprob-gain
filter:

`gain = logprob(r_i | p_i) - logprob(r_i | p_0)`

where:

- `p_i` is the round-specific prompt after allowed-move elimination
- `p_0` is the original incoming prompt context for that training example

The intended failure mode is real: later elimination rounds can make a response
look artificially easy because the reduced candidate set removes uncertainty that
will not be present at test time.

For this repo's current selection setup, the logprob proxy is brittle for the
wrong reasons:

- it is sensitive to `<think>...</think>` length and formatting details
- it depends on prompt-surface details, including whether prompt text literally
  says `allowed_moves`
- it assumes cross-prompt logprob comparisons are well calibrated
- it can react to response-style differences instead of the actual issue we care
  about: whether the trace still looks plausible under the student's visible
  test-time context alone

The replacement explored here is a judge question:

> If the student only saw the normal visible context `x`, would the trace `y`
> still look like a plausible standalone reasoning trace, or does it appear to
> rely on hidden information `h` such as a reduced candidate set?

False positives on clean traces are much worse than false negatives, so the
judge is deliberately conservative: if uncertain, ACCEPT.

## Scope

This note is strictly about a local-only sanity and prompt-optimization
prototype. It does not wire the judge into trainer code, launchers, or Slurm
flows.

Primary local harness:

- `scripts/diag_llm_gating_vllm.py`

Optional posthoc re-summarizer:

- `scripts/posthoc_llm_gating_summary.py`

Implementation choice:

- I added a new dedicated local harness instead of extending
  `scripts/diag_iterative_round_logprob_vllm.py`
- the iterative logprob script was the right reference for loading rows,
  rendering prompts, and mirroring the `considered_moves_uci_list` flow, but a
  separate harness was cleaner because judge-model sanity needs paired clean vs
  singleton buckets, explicit `x` / `h` / `y` artifact capture, and a second
  vLLM pass for judging
- the new harness still reuses the same core repo semantics: prompt rendering
  from the current template shape and reward parsing through
  `recipe/chess/reward_fn.py`

This prototype keeps the existing selection/output contract intact:

- the generation prompt still uses the current selection template
- reward parsing still goes through `recipe/chess/reward_fn.py`
- `reward_model.considered_moves_uci` remains the operative candidate set

## Why This Dataset / Prompt Path

There are multiple full-legal selection datasets in the repo. For the main local
sanity and prompt-optimization work I used:

- parquet: `data/chess_puzzles_chessr1_aligned_sharded_ours/test.parquet`
- prompt template: `recipe/chess/prompt_templates/select_prompt.jinja`

Reasoning:

- `iterative.md` and `restricted_moves.md` both treat
  `data/chess_puzzles_chessr1_aligned_sharded_ours/` as a valid full-legal
  selection dataset where `reward_model.considered_moves_uci == legal_moves_uci`
- `scripts/diag_iterative_round_logprob_vllm.py`, the closest local diagnostic
  harness, also defaults to the Chess-R1 aligned `..._ours` path
- this path avoids ambiguity from the later small-legal prompt surface, where
  prompt text omits the literal token `allowed_moves` even though runtime subset
  semantics still live in `reward_model.considered_moves_uci`

The harness explicitly rejects source rows whose stored
`considered_moves_uci != legal_moves_uci`, so the clean bucket is genuinely
full-legal.

## `x`, `h`, and `y`

For each paired example:

- `x` is the visible full-action-space context:
  - `x.prompt_text`
  - `x.fen`
  - `x.legal_moves_uci`
  - `x.visible_candidate_count`
  - `x.visible_candidate_moves_uci`
- `h` is:
  - `null` for the clean bucket
  - the actual hidden singleton candidate list used during generation for the
    suspicious bucket
- `y` is the candidate response trace:
  - extracted reasoning text when `<think>...</think>` exists
  - otherwise the raw response text

Bucket construction:

- `clean`: generate from the normal full-legal selection prompt
- `suspicious`: generate from the same source row after replacing
  `reward_model.considered_moves_uci` with a singleton list

Singleton construction in the main runs:

- strategy: `mu_best`
- meaning: the singleton move is the subset-relative best legal move under the
  row's stored move-score map

## Judge Prompt Status

The judge prompt currently lives in `scripts/diag_llm_gating_vllm.py`.

Three prompt variants from the fixed-set optimization loop are archived in:

- `outputs/llm_gating/prompt_optimization/prompt_iteration_comparison.md`

Those three are the most decision-relevant prompt variants so far:

- `iter01_rejudge`: first major improvement, and the **least restrictive**
  prompt of the three
- `iter02_rejudge`: best observed fixed-set agreement for the effective gate
  when the prompt changed and the deterministic contradiction backstop stayed
  fixed
- `final_best_prompt_rejudge`: confirmation rerun of the selected prompt text

Important status note:

- the current harness prompt in `scripts/diag_llm_gating_vllm.py` reflects the
  prompt-optimization loop, not the later user preference for the least
  restrictive prompt
- if the priority is benchmark fit on the fixed 200-case set, `iter02` is the
  best observed effective-gate result so far
- if the priority is being least restrictive and more likely to generalize
  beyond the singleton/small-list setting, `iter01` is the leading candidate

The current prompt text stored in the harness is:

```text
You are filtering reasoning traces for training.

At student test time, the student sees only the visible context `x`, not the hidden hint `h`
(for example a reduced candidate set, legality filter, or other privileged restriction).

Judge whether `y` still looks like a plausible, self-contained response from a strong model that saw only `x`.
Do not judge chess quality or move correctness.

Important: false positives on clean traces are much worse than false negatives.
If you are unsure, ambiguous, or `y` could plausibly have been written from `x` alone, ACCEPT.

Use `x` as the authoritative visible context. Use `h` only to detect hidden-hint dependence.
Reject only for clear evidence that `y` depends on `h`.

Crucial distinction:
- It is usually ACCEPT if `y` simply discusses one chosen candidate from the visible list, even with wording like
  "the allowed move e1g3 is strong", "from the allowed list, e1g3 is best", or "the best move from the allowed
  moves is e1g3". Mentioning one move from the visible list is not enough for REJECT.
- It is also usually ACCEPT if `y` compares that move against other candidates on normal chess grounds, for
  example "other moves like g8g7 or g8g6 are more direct but less strategic". Strategic comparison is fine.
- It is usually REJECT if `y` describes the visible allowed-move list itself as smaller than it is in `x`, or if
  it treats the hidden shortlist in `h` as though it were the visible list.

Usually REJECT if `x` lists many candidates but `y` says or clearly implies things like:
- there was only one allowed/candidate/legal move
- the allowed list was limited, short, narrow, or focused on just a few moves
- the allowed moves were exactly a small set such as "e1e2 and f1f2"
- some other move was not in the allowed list
- some move "is not an option"
- "the only allowed move is e2e4"
- "the allowed moves are e2e4 and g1f3"
- "the allowed moves are very limited"
- "other moves are not in the allowed list"

Do not reject merely because:
- `y` is short, confident, efficient, or omits alternatives
- `y` names one move from the visible list as the best allowed move
- `y` says "the allowed move X is strong" or "the move X from the allowed list is best" without making claims
  about what else was or was not on the list
- the final move also appears in `h`
- the final move is unusual, wrong, or invalid under `x`

The final move alone is almost never enough evidence for REJECT.
If `x` lists many moves and `y` simply argues for one move on normal chess grounds without asserting that the
visible list was small, restricted, or enumerated by `h`, that is usually ACCEPT.

Examples:
- ACCEPT: "The allowed move c1c8 is a promotion move and the best move from the allowed list."
- ACCEPT: "Other moves like g8g7 or g8g6 are more direct, but e1g3 has better long-term value."
- REJECT: if `x` lists many candidates but `y` says "the allowed moves are f4e4 and f4d4."
- REJECT: if `x` lists many candidates but `y` says "the allowed moves are very limited, focusing on a single
  pawn move."
- REJECT: if `x` lists many candidates but `y` says "the allowed move h7h8 is not an option."

Return exactly one JSON object with:
- `verdict`: `ACCEPT` or `REJECT`
- `confidence`: `high`, `medium`, or `low`
- `reason`: a short explanation focused on hint dependence vs plausibility under `x`
```

## Structured Output

The judge uses guided JSON decoding through vLLM with this schema:

```json
{
  "type": "object",
  "properties": {
    "verdict": {
      "type": "string",
      "enum": ["ACCEPT", "REJECT"]
    },
    "confidence": {
      "type": "string",
      "enum": ["high", "medium", "low"]
    },
    "reason": {
      "type": "string"
    }
  },
  "required": ["verdict", "confidence", "reason"],
  "additionalProperties": false
}
```

In addition to the LLM judge, the harness applies a deterministic contradiction
backstop for explicit singleton leaks such as:

- `the only allowed move`
- `the allowed move f1f8 is the only one in the list`
- `the only move in the allowed list`

This backstop is narrow by design. It only fires when `x` visibly has more than
one candidate move and `y` explicitly contradicts that.

## Sanity-Check Protocol

1. Load full-legal selection rows only.
2. Render the visible full-action-space prompt from `select_prompt.jinja`.
3. Build paired cases from the same row:
   - one clean full-legal case
   - one suspicious singleton case
4. Generate local traces through vLLM.
5. Parse generation outputs with `recipe/chess/reward_fn.py` so the existing
   `<think>...</think><uci_move>...</uci_move>` contract and
   `considered_moves_uci` semantics remain unchanged.
6. Judge every saved case under `x` vs `h`.
7. Summarize clean vs suspicious rejection rates.
8. Manually inspect:
   - clean rejects first
   - suspicious rejects
   - suspicious accepts that look like plausible false negatives
9. Optionally, freeze one cases file and run prompt-change rejudge with
   `--reuse_cases_jsonl`.
10. Optionally, label the fixed cases file manually and compare the judge
    against that ground truth.

Artifacts saved by the harness:

- `summary.json`
- `cases.jsonl`
- `representatives.json`
- `config.json`
- `judge_prompt.txt`

Each saved case preserves:

- bucket label
- source row metadata
- visible prompt text and visible candidate list (`x`)
- hidden hint payload (`h`)
- candidate response trace (`y`)
- reward/contract metadata from `reward_fn.py`, including `score` and `acc`
- judge raw output and parsed verdict/confidence/reason

## Exact Local Commands

Environment:

```bash
source ~/.bashrc
conda activate verl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,3
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=ERROR
```

Syntax check:

```bash
python -m py_compile scripts/diag_llm_gating_vllm.py scripts/posthoc_llm_gating_summary.py
```

100-row long-prompt baseline generation run (`100` source rows -> `200` paired
cases):

```bash
python scripts/diag_llm_gating_vllm.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --parquets data/chess_puzzles_chessr1_aligned_sharded_ours/test.parquet \
  --template_path recipe/chess/prompt_templates/select_prompt.jinja \
  --out_dir outputs/llm_gating/20260316_longprompt_100rows_baseline \
  --limit_rows 100 \
  --row_seed 0 \
  --singleton_strategy mu_best \
  --max_prompt_tokens 4096 \
  --max_response_tokens 384 \
  --judge_max_tokens 192 \
  --generation_batch_size 8 \
  --judge_batch_size 8 \
  --temperature 0.6 \
  --top_p 0.95 \
  --seed 0 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.78 \
  --max_model_len 4096 \
  --max_num_seqs 256 \
  --overwrite
```

Reproducibility note:

- the exact historical artifact directories in `outputs/llm_gating/...` were
  produced with the judge prompt snapshots saved inside those directories
- the harness now supports `--judge_prompt_path`, so exact historical reruns
  should pass the saved prompt artifact explicitly
- without `--judge_prompt_path`, the script uses the current in-file prompt and
  will reproduce the workflow, but not necessarily the exact historical result

Example exact-prompt rejudge command using an archived prompt snapshot:

```bash
python scripts/diag_llm_gating_vllm.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --reuse_cases_jsonl outputs/llm_gating/20260316_mainagent_verify_shortprompt_rejudge100_v2/cases.jsonl \
  --judge_prompt_path outputs/llm_gating/prompt_optimization/iter01_rejudge/judge_prompt.txt \
  --template_path recipe/chess/prompt_templates/select_prompt.jinja \
  --out_dir outputs/llm_gating/example_iter01_exact_rejudge \
  --max_prompt_tokens 4096 \
  --max_response_tokens 384 \
  --judge_max_tokens 192 \
  --generation_batch_size 8 \
  --judge_batch_size 8 \
  --temperature 0.6 \
  --top_p 0.95 \
  --seed 0 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.78 \
  --max_model_len 4096 \
  --max_num_seqs 256 \
  --overwrite
```

Same-trace rejudge on the exact same saved cases:

```bash
python scripts/diag_llm_gating_vllm.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --reuse_cases_jsonl outputs/llm_gating/20260316_longprompt_100rows_baseline/cases.jsonl \
  --template_path recipe/chess/prompt_templates/select_prompt.jinja \
  --out_dir outputs/llm_gating/20260316_mainagent_verify_shortprompt_rejudge100_v2 \
  --max_prompt_tokens 4096 \
  --max_response_tokens 384 \
  --judge_max_tokens 192 \
  --generation_batch_size 8 \
  --judge_batch_size 8 \
  --temperature 0.6 \
  --top_p 0.95 \
  --seed 0 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.78 \
  --max_model_len 4096 \
  --max_num_seqs 256 \
  --overwrite
```

Fresh 100-row generation rerun with the then-current short prompt:

```bash
python scripts/diag_llm_gating_vllm.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --parquets data/chess_puzzles_chessr1_aligned_sharded_ours/test.parquet \
  --template_path recipe/chess/prompt_templates/select_prompt.jinja \
  --out_dir outputs/llm_gating/20260316_main_shortprompt_confirm100 \
  --limit_rows 100 \
  --row_seed 0 \
  --singleton_strategy mu_best \
  --max_prompt_tokens 4096 \
  --max_response_tokens 384 \
  --judge_max_tokens 192 \
  --generation_batch_size 8 \
  --judge_batch_size 8 \
  --temperature 0.6 \
  --top_p 0.95 \
  --seed 0 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.78 \
  --max_model_len 4096 \
  --max_num_seqs 256 \
  --overwrite
```

## Model Used

Judge and generator model used throughout the main local sanity and
prompt-optimization work:

- `Qwen/Qwen2.5-7B-Instruct`

Why 7B:

- it was already cached locally in the Hugging Face cache
- it was the strongest low-friction instruct model available in the local
  environment
- it fit locally with `tensor_parallel_size=2` on the 100-row reruns

The repo-allowed fallback remains `Qwen/Qwen2.5-3B-Instruct`.

## What We Have Done So Far

### 1. Built a Local-Only Judge Harness

The harness already supported larger sweeps via `--limit_rows`. The main code
additions during this work were:

- the local judge harness: `scripts/diag_llm_gating_vllm.py`
- the posthoc summarizer: `scripts/posthoc_llm_gating_summary.py`
- a fixed-trace rejudge path: `--reuse_cases_jsonl`

The rejudge path matters because it lets prompt changes be evaluated on the
same saved traces instead of mixing prompt edits with generation noise.

### 2. Established 100-Row Judge Sanity Artifacts

Bucket-level judge results on the 100-row / 200-case slice:

- long-prompt baseline:
  `outputs/llm_gating/20260316_longprompt_100rows_baseline/`
  - clean bucket: `1 / 100` rejects
  - suspicious bucket: `83 / 100` rejects
- fixed-trace short-prompt rejudge:
  `outputs/llm_gating/20260316_mainagent_verify_shortprompt_rejudge100_v2/`
  - clean bucket: `0 / 100` rejects
  - suspicious bucket: `83 / 100` rejects
- fresh short-prompt generation rerun:
  `outputs/llm_gating/20260316_main_shortprompt_confirm100/`
  - clean bucket: `0 / 100` rejects
  - suspicious bucket: `82 / 100` rejects

These are raw judge outputs, not yet ground-truth corrected labels.

### 3. Froze a Fixed Dataset and Labeled It Manually

Fixed dataset for ground-truth comparison:

- `outputs/llm_gating/20260316_mainagent_verify_shortprompt_rejudge100_v2/cases.jsonl`

Ground-truth labels produced by a dedicated coding agent:

- `outputs/llm_gating/20260316_mainagent_verify_shortprompt_rejudge100_v2/manual_ground_truth_labels.jsonl`
- `outputs/llm_gating/20260316_mainagent_verify_shortprompt_rejudge100_v2/manual_ground_truth_summary.json`

Ground-truth totals on this fixed 200-case set:

- clean bucket: `100 / 100` `SHOULD_ACCEPT`
- suspicious bucket: `77 / 100` `SHOULD_REJECT`, `23 / 100` `SHOULD_ACCEPT`

Against the current judge on that fixed set:

- `judge_accept_manual_accept = 110`
- `judge_accept_manual_reject = 7`
- `judge_reject_manual_accept = 13`
- `judge_reject_manual_reject = 70`
- agreement: `180 / 200 = 90%`

Important interpretation:

- on this fixed dataset, the current judge's larger problem is **over-rejection**
  (`13` false rejects) rather than under-rejection (`7` false accepts)

The manual summary groups the suspicious `SHOULD_REJECT` cases into:

- `explicit_only_or_singleton = 60`
- `limited_or_short_list = 10`
- `enumerated_small_list = 7`

### 4. Optimized the Prompt Against the Fixed Ground Truth

Prompt-optimization artifacts live under:

- `outputs/llm_gating/prompt_optimization/`

Rollup:

- `outputs/llm_gating/prompt_optimization/optimization_summary.json`

Decision-oriented prompt comparison:

- `outputs/llm_gating/prompt_optimization/prompt_iteration_comparison.md`

Prompt-iteration results on the fixed 200-case benchmark, scored using
`judge_effective_verdict` with the deterministic singleton-contradiction
backstop held fixed:

- `iter01_rejudge`: `190 / 200 = 95.0%`
- `iter02_rejudge`: `191 / 200 = 95.5%` (best observed result in this loop)
- `iter03_rejudge`: `189 / 200 = 94.5%`
- `iter04_rejudge`: `188 / 200 = 94.0%`
- `final_best_prompt_rejudge`: `188 / 200 = 94.0%` on a confirmation rerun of
  the selected prompt text

Best observed confusion matrix in that loop (`iter02_rejudge`):

- `judge_accept_manual_accept = 115`
- `judge_accept_manual_reject = 1`
- `judge_reject_manual_accept = 8`
- `judge_reject_manual_reject = 76`

Interpretation:

- prompt iteration improved the fixed-set effective-gate result substantially
- it did **not** reach `100%`
- some judge-time variance remained even with fixed traces, since the later
  confirmation rerun of the chosen prompt landed lower than the best observed
  iteration

## Representative Cases

Representative clean accept:

- `test.parquet:7811:clean`
  - this was the one long-prompt clean false positive on the 100-row baseline
  - later prompt revisions accepted it correctly

Representative suspicious reject:

- `test.parquet:1194:suspicious:h5h1`
  - explicit singleton leak: `the only allowed move`

Representative suspicious false accept from manual ground truth:

- `test.parquet:9878:suspicious:e1e2`
  - judge accepted it
  - manual label says `SHOULD_REJECT`
  - reason: it narrates a small allowed-move list inconsistent with `x`

Representative suspicious false reject from manual ground truth:

- `test.parquet:7022:suspicious:e1g3`
  - judge rejected it
  - manual label says `SHOULD_ACCEPT`
  - reason: ordinary chosen-move reasoning remains plausible under full `x`
    alone

## Observed Failure Modes

### 1. Hidden-shortlist narration is the core reject case

The traces that most clearly should be rejected are the ones that treat the
hidden shortlist as if it were the visible `allowed_moves` set in `x`.

Typical bad patterns:

- explicit singleton language
- "allowed moves are ..." enumerations that match a tiny shortlist
- "very limited" / "short" list language
- "not in the allowed list" phrasing that contradicts `x`

### 2. Prompt iteration still trades false accepts against false rejects

The fixed ground-truth benchmark showed that the current judge can miss some
hint-dependent shortlist narration, but it can also over-reject traces that are
still plausible under `x` alone.

This is why the fixed 200-case manual label set matters: it provides a concrete
target rather than relying only on bucket rates.

### 3. Prompt changes alone appear insufficient for exact agreement

The best observed fixed-set effective-gate result in the prompt loop is
`95.5%`, not `100%`.

The evidence now points to:

- prompt changes as useful, but insufficient for exact agreement
- prompt-plus-backstop as the likely next move if exact agreement is the goal

## Current Status

Current state of the work:

- local-only prototype only; no trainer integration
- fixed 100-row / 200-case sanity set established
- full manual ground-truth labels established for one fixed dataset
- prompt-optimization loop run against that fixed set
- three decision-relevant prompt variants documented in
  `prompt_iteration_comparison.md`

Current recommendation framing:

- if the priority is **least restrictive / likely more generalizable**, prefer
  `iter01_rejudge`
- if the priority is **best fit on the fixed benchmark with the current fixed
  backstop**, prefer
  `iter02_rejudge`
- if the goal is **exact agreement on the fixed benchmark**, the next step is
  likely prompt-plus-backstop, not more prompt-only tuning alone

This note intentionally stops before trainer integration. The next step, if
desired, is to decide whether to:

- switch the live harness prompt to one of the archived prompt variants
- use the fixed manual-ground-truth set to tune prompt-plus-backstop behavior
