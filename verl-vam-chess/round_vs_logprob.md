# Round-vs-Logprob Diagnostic: What Was Done

## TL;DR

1. Implemented a reproducible iterative-pruning diagnostic in `scripts/diag_iterative_round_logprob_vllm.py` that mirrors trainer semantics and computes reference logprobs under round-0 prompt context with vLLM.
2. Final completed run used:
   - model/settings: `Qwen/Qwen2.5-3B-Instruct`, `temperature=0.6`, `top_p=0.95`, `max_output_tokens=2000`, `max_model_length=4096`
   - data: `data/chess_puzzles_chessr1_aligned_sharded_ours/train_0.parquet` + `train_1.parquet`
   - runtime: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=1,3`, `tensor_parallel_size=2`, `gpu_memory_utilization=0.6`
   - loop: `K=16`, one sample/round, `stop_criterion=gt_uci`, generation submit-all + scoring chunked (`score_batch_size=4`)
   - sampling scope: fixed-seed `sample_frac=0.25` (25,000 prompts)
3. Completed-run artifacts:
   - run dir: `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113`
   - key files: `config.json`, `round_records.jsonl`, `round_summary.csv`, `prompt_summary.csv`, `summary.json`, `run.log`
   - plots:
     - `round_vs_logprob.png` (baseline)
     - `round_vs_logprob_pretty.png` (improved)
4. Key results from `summary.json` / `round_summary.csv`:
   - `num_prompts=25000`, `num_round_records=394104`
   - `num_prompts_success_gt=771`, `num_prompts_forced_accept=24229`
   - `elapsed_sec=39580.84` (~10.99 hours)
   - mean reference logprob sum: round 1 `-209.4241` -> round 16 `-360.1835`
5. Added reusable plotting script `scripts/plot_iter_round_logprob_pretty.py` to generate a clearer figure (mean/median lines + IQR + 95% CI).
6. Added and tested a second prompt variant Jinja template:
   - `recipe/chess/prompt_templates/select_prompt_small_legal.jinja`
   - This template does not explicitly mention `allowed_moves`; it presents only a shortened legal-move list (`considered_moves_uci_list`).
7. Apples-to-apples (both `sample_frac=0.25`) comparison result:
   - Original template mean: `-209.4241` -> `-360.1835` (rapid degradation)
   - Small-legal template mean: `-151.0275` -> `-178.5183` (much more stable)
   - Mean/median gap across rounds:
     - Original average gap: `-127.27`
     - Small-legal average gap: `-1.12`
   - Comparison plot:
     - `artifacts/iter_round_logprob_diag/compare_round_vs_logprob_original_vs_smalllegal_25pct.png`

## Objective

Implement and run a reproducible diagnostic for iterative allowed-move elimination chess that:

1. Uses `Qwen/Qwen2.5-3B-Instruct` with:
   - `temperature=0.6`
   - `top_p=0.95`
   - `max_output_tokens=2000`
   - `max_model_length=4096`
2. Uses exactly:
   - `data/chess_puzzles_chessr1_aligned_sharded_ours/train_0.parquet`
   - `data/chess_puzzles_chessr1_aligned_sharded_ours/train_1.parquet`
3. Runs iterative pruning up to `K=16` with one generation per round.
4. Stops each prompt when the best response is sampled (trainer-faithful criterion configured as `stop_criterion=gt_uci`).
5. Computes per-round response logprob under the original round-0 prompt context.
6. Produces round-vs-logprob artifacts and plots.

---

## Ground-Truth Semantics Used (Code/Doc References)

### Iterative pruning and stopping semantics

These were grounded in both docs and implementation:

1. `iterative.md:133` defines the trainer algorithm and references the exact code paths.
2. `iterative.md:162` states success is checked against `pred_move == gt_uci` with no penalty (and subset-validity if present).
3. `iterative.md:167` states move-elimination removes valid in-subset predictions from the candidate set.
4. `iterative.md:169` states unresolved prompts are force-accepted at `r_max` under `accept_last`.
5. `verl/trainer/ppo/ray_trainer.py:3597` starts the `allowed_move_elim` loop.
6. `verl/trainer/ppo/ray_trainer.py:3651` builds per-round batch with current allowed set.
7. `verl/trainer/ppo/ray_trainer.py:3741` removes valid in-subset predicted moves from allowed candidates.
8. `verl/trainer/ppo/ray_trainer.py:3746` marks success when `pred_move == gt_uci`.
9. `verl/trainer/ppo/ray_trainer.py:3763` applies forced-accept on final round for unresolved prompts.
10. `verl/trainer/ppo/ray_trainer.py:1207` writes `reward_model.considered_moves_uci = allowed`.
11. `recipe/chess/reward_fn.py:379` defines subset target move (`target_move`) as argmax over current considered set.
12. `recipe/chess/reward_fn.py:649` returns both `gt_uci` and `target_move` in reward metadata.
13. `verl/workers/reward_manager/batch.py:82` injects `prompt_text` into reward `extra_info` for subset gating.

### "Best response" criterion used in this run

For the completed run here, stopping used `stop_criterion=gt_uci` (trainer-faithful), not `target_move`.

Reason:

1. Trainer success logic in `verl/trainer/ppo/ray_trainer.py:3746` checks equality to `gt_uci`.
2. `iterative.md:162` documents the same.
3. `target_move` can differ from `gt_uci` on a small fraction of rows due to tie behavior and map/source nuances.

Observed mismatch rates between dataset `ground_truth` and round-0 subset argmax target:

1. `train_0.parquet`: `386/50008` (`0.772%`)
2. `train_1.parquet`: `357/49992` (`0.714%`)

---

## Dataset Inspection Performed

The two required shards were inspected directly for schema and sample payloads.

From direct read:

1. `train_0.parquet`: `50008` rows, columns `['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']`
2. `train_1.parquet`: `49992` rows, same schema
3. Sample rows confirm:
   - `reward_model.fen` present
   - `reward_model.ground_truth` present
   - `reward_model.legal_moves_uci` present
   - `reward_model.considered_moves_uci` present and equal-length to legal list in samples
   - `extra_info.index` present

---

## Implemented Code

### 1) Main diagnostic script

Created/updated:

1. `scripts/diag_iterative_round_logprob_vllm.py`

Core behavior implemented:

1. Loads required parquet shards and selection prompt template.
2. Initializes each prompt with full legal moves as allowed set.
3. Runs iterative rounds up to `K`.
4. Generates exactly one response per active prompt per round with vLLM.
5. Parses and scores with `compute_score` from `recipe/chess/reward_fn.py`.
6. Stops prompt on selected success criterion (`gt_uci` in final run), otherwise prunes valid in-subset predicted move.
7. Applies forced-accept on unresolved prompts at round `K`.
8. Computes per-response reference logprob under round-0 context using vLLM `prompt_logprobs` with teacher-forced `prompt_token_ids = round0 + response`.
9. Writes reproducible artifacts:
   - `config.json`
   - `round_records.jsonl`
   - `round_summary.csv`
   - `prompt_summary.csv`
   - `summary.json`
   - `round_vs_logprob.png`

Key locations:

1. Script header and intent: `scripts/diag_iterative_round_logprob_vllm.py:2`
2. Default required parquet paths: `scripts/diag_iterative_round_logprob_vllm.py:49`
3. Sampling/model args and defaults: `scripts/diag_iterative_round_logprob_vllm.py:384`
4. Generation submit-all switch: `scripts/diag_iterative_round_logprob_vllm.py:569`
5. Scoring chunking logic: `scripts/diag_iterative_round_logprob_vllm.py:620`
6. Reward parse and prune/update: `scripts/diag_iterative_round_logprob_vllm.py:698`
7. Summary + artifact writing: `scripts/diag_iterative_round_logprob_vllm.py:875`

### 2) Runtime optimization changes requested later

Applied changes:

1. Added deterministic subsampling (`--sample_frac`, `--sample_seed`, `--sample_n_max`).
2. Added `--submit_all_per_round` for generation-side submit-all behavior.
3. Added progress bar (`tqdm`) with round-level postfix metrics.
4. Added fast-path for round-0-equivalent contexts reusing generation cumulative logprob.
5. Corrected batching semantics so submit-all applies only to generation; scoring is always chunked with `--score_batch_size` (default now `4`).

Relevant lines:

1. `scripts/diag_iterative_round_logprob_vllm.py:394` (`--score_batch_size` default `4`)
2. `scripts/diag_iterative_round_logprob_vllm.py:396` (`--submit_all_per_round` generation-only help)
3. `scripts/diag_iterative_round_logprob_vllm.py:516` init print confirms generation-only semantics
4. `scripts/diag_iterative_round_logprob_vllm.py:621` scoring always chunked

### 3) Small-legal prompt variant + subset-gating safety

Added/updated for the prompt-variant experiment:

1. New Jinja template:
   - `recipe/chess/prompt_templates/select_prompt_small_legal.jinja`
   - Behavior:
     - does not mention `allowed_moves` text explicitly
     - only prints `Legal moves (UCI): {{ considered_moves_uci_list | join(', ') }}`
2. Reward subset-gating override in `recipe/chess/reward_fn.py`:
   - Added explicit override support via `extra_info["use_considered_moves_uci"]`.
   - This ensures subset semantics remain tied to the current pruned set even when prompt text omits the `allowed_moves` keyword.
3. Diagnostic wiring:
   - `scripts/diag_iterative_round_logprob_vllm.py` now passes
     `extra_info={"prompt_text": ..., "use_considered_moves_uci": True}` to `compute_score`.

Why this was necessary:

1. `recipe/chess/reward_fn.py` historically inferred subset-gating from prompt text containing `allowed_moves`.
2. With the small-legal template, that keyword is absent by design.
3. Explicit override prevents accidental fallback to full-legal gating during this experiment.

### 4) Improved plotting utilities

Added:

1. `scripts/plot_iter_round_logprob_pretty.py`
2. `scripts/plot_iter_round_logprob_compare.py`

This script builds a clearer publication-style plot using:

1. Mean and median round logprob lines.
2. IQR band (25-75%).
3. 95% CI around mean.
4. Improved labels, grid, markers, typography, and DPI.

Reference:

1. `scripts/plot_iter_round_logprob_pretty.py:1`
2. `scripts/plot_iter_round_logprob_compare.py:1`

---

## vLLM Logprob Approach Used

For a prompt `X` and generated response `Y`, the reference score is:

1. Build round-0 prompt token ids for `X`.
2. Concatenate with response token ids for `Y`.
3. Call vLLM generation with:
   - `prompt_logprobs=0`
   - `max_tokens=1`
4. Sum logprobs of the response-token positions from `prompt_logprobs`.

Implementation references:

1. Score sampling params: `scripts/diag_iterative_round_logprob_vllm.py:540`
2. Combined token scoring inputs: `scripts/diag_iterative_round_logprob_vllm.py:645`
3. Per-token sum logic: `scripts/diag_iterative_round_logprob_vllm.py:669`

Notes:

1. In this environment (`vllm==0.8.5.post1`), `max_tokens=0` is not used; scoring uses `max_tokens=1` and ignores generated token.
2. A direct fast-path is used when round prompt equals round-0 prompt (`reference_logprob_status=direct_round_prompt_match`).

---

## Subagents Used

1. `coding_agent` for implementation support and vLLM/logprob method checks.
2. `researcher` was attempted for vLLM API validation; repeated timeouts occurred in this environment, so final validation was completed directly in code/tests.
3. `coding_agent` was used again for the improved plotting script and figure generation.

---

## Commands and Launch Recipes

## Foreground local run (reference form)

```bash
source ~/.bashrc
conda activate verl
cd /usr0/home/zhichen3/chess-rl

export PYTHONPATH=.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,3

python scripts/diag_iterative_round_logprob_vllm.py \
  --parquets data/chess_puzzles_chessr1_aligned_sharded_ours/train_0.parquet data/chess_puzzles_chessr1_aligned_sharded_ours/train_1.parquet \
  --out_dir artifacts/iter_round_logprob_diag/<run_name> \
  --k_max 16 \
  --sample_frac 0.25 \
  --sample_seed 0 \
  --submit_all_per_round \
  --score_batch_size 4 \
  --stop_criterion gt_uci \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_output_tokens 2000 \
  --max_model_length 4096 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.6 \
  --overwrite
```

## Detached launch (used later operationally)

```bash
source ~/.bashrc && conda activate verl && cd /usr0/home/zhichen3/chess-rl && export PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3 && RUN_ID=$(date +%Y%m%d_%H%M%S) && OUT_DIR=artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_${RUN_ID} && mkdir -p "$OUT_DIR" && nohup python scripts/diag_iterative_round_logprob_vllm.py --parquets data/chess_puzzles_chessr1_aligned_sharded_ours/train_0.parquet data/chess_puzzles_chessr1_aligned_sharded_ours/train_1.parquet --out_dir "$OUT_DIR" --k_max 16 --sample_frac 0.25 --sample_seed 0 --submit_all_per_round --score_batch_size 4 --stop_criterion gt_uci --temperature 0.6 --top_p 0.95 --max_output_tokens 2000 --max_model_length 4096 --tensor_parallel_size 2 --gpu_memory_utilization 0.6 --overwrite > "$OUT_DIR/run.log" 2>&1 & echo "PID=$! OUT_DIR=$OUT_DIR LOG=$OUT_DIR/run.log"
```

Monitor/cancel:

```bash
tail -f <run_dir>/run.log
kill <pid>
```

## Pretty plot generation

```bash
python scripts/plot_iter_round_logprob_pretty.py \
  --run-dir artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113 \
  --output artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113/round_vs_logprob_pretty.png
```

---

## Existing Results (Completed 25% Run)

Run directory:

1. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113`

From `summary.json`:

1. `num_prompts`: `25000`
2. `num_round_records`: `394104`
3. `num_prompts_success_gt`: `771`
4. `num_prompts_forced_accept`: `24229`
5. `elapsed_sec`: `39580.84` (about `10.99` hours)
6. `stop_criterion`: `gt_uci`

From `round_summary.csv`:

1. Round 1 mean reference logprob sum: `-209.4241`
2. Round 16 mean reference logprob sum: `-360.1835`
3. Round 1 median reference logprob sum: `-131.9980`
4. Round 16 median reference logprob sum: `-190.3282`
5. Accepted total across rounds: `25000`
6. Unresolved after final round: `0`

Primary artifacts:

1. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113/config.json`
2. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113/run.log`
3. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113/round_records.jsonl`
4. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113/round_summary.csv`
5. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113/prompt_summary.csv`
6. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113/summary.json`
7. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113/round_vs_logprob.png`
8. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113/round_vs_logprob_pretty.png`

---

## Small-Legal Prompt Variant

Template used:

1. `recipe/chess/prompt_templates/select_prompt_small_legal.jinja`

Key template behavior:

1. The prompt does not explicitly mention `allowed_moves`.
2. The candidate set is presented as a shortened legal list by rendering:
   - `Legal moves (UCI): {{ considered_moves_uci_list | join(', ') }}`
3. Output contract remains strict:
   - `<think>...</think><uci_move>...</uci_move>`

### Dataset Curation Procedure (`..._ours_small_legal`)

Source and target:

1. Source shards: `data/chess_puzzles_chessr1_aligned_sharded_ours/`
2. Curated shards: `data/chess_puzzles_chessr1_aligned_sharded_ours_small_legal/`
3. Transform type: prompt rewrite only, with explicit `considered_moves_uci` reset to legal moves (no engine rescoring).

What was changed per row during curation:

1. `prompt` rewritten using `recipe/chess/prompt_templates/select_prompt_small_legal.jinja`.
2. `reward_model.considered_moves_uci` explicitly set to `reward_model.legal_moves_uci` via `--set_considered_moves_uci`.

What was intentionally not changed:

1. `reward_model.fen`
2. `reward_model.ground_truth`
3. `reward_model.move_expected_scores_json` / `reward_model.move_values_json`
4. Top-level row count and split structure (`train_0`, `train_1`, etc.)

Command used to curate the dataset:

```bash
python scripts/rewrite_chess_prompts_from_template.py \
  --input_dir data/chess_puzzles_chessr1_aligned_sharded_ours \
  --output_dir data/chess_puzzles_chessr1_aligned_sharded_ours_small_legal \
  --template_path recipe/chess/prompt_templates/select_prompt_small_legal.jinja \
  --set_considered_moves_uci \
  --overwrite
```

Primary rewrite outputs observed:

1. `train_0.parquet`: `50008` rows rewritten
2. `train_1.parquet`: `49992` rows rewritten
3. `test.parquet`: `10000` rows rewritten
4. `train_hard_0.parquet`: `41480` rows rewritten
5. `train_hard_1.parquet`: `41519` rows rewritten

Validation performed after rewrite:

1. `data/chess_puzzles_chessr1_aligned_sharded_ours_small_legal/train_0.parquet`:
   - `contains_allowed_moves = 0/50008`
2. `data/chess_puzzles_chessr1_aligned_sharded_ours_small_legal/train_1.parquet`:
   - `contains_allowed_moves = 0/49992`
3. `considered_moves_uci == legal_moves_uci` held for all rows in both shards.

Reproducible validation commands:

```bash
python - <<'PY'
import pyarrow.parquet as pq
for p in [
    "data/chess_puzzles_chessr1_aligned_sharded_ours_small_legal/train_0.parquet",
    "data/chess_puzzles_chessr1_aligned_sharded_ours_small_legal/train_1.parquet",
]:
    t = pq.read_table(p, columns=["prompt", "reward_model"])
    prompts = t.column("prompt").to_pylist()
    rms = t.column("reward_model").to_pylist()
    contains_allowed = 0
    eq_count = 0
    for prm, rm in zip(prompts, rms):
        if isinstance(prm, list) and prm and isinstance(prm[0], dict):
            s = str(prm[0].get("content") or "")
        else:
            s = str(prm)
        if "allowed_moves" in s.lower():
            contains_allowed += 1
        rm = rm or {}
        if (rm.get("considered_moves_uci") or []) == (rm.get("legal_moves_uci") or []):
            eq_count += 1
    print(p, "rows", t.num_rows, "contains_allowed_moves", contains_allowed, "considered_eq_legal", eq_count)
PY
```

Reproducibility note:

1. This curated dataset is fully reproducible from source + template + rewrite command; it does not need to be versioned in Git.

---

## New Run Results (Small-Legal)

### 10% run (quick comparison run)

Run directory:

1. `artifacts/iter_round_logprob_diag/full10pct_submitall_genonly_u06_smalllegal_20260221_171746`

From `summary.json` / `run.log`:

1. `num_prompts=10000`
2. `elapsed_sec=14160.2` (~3.93 hours)
3. mean reference logprob sum:
   - round 1: `-150.5521`
   - round 16: `-179.2067`

### 25% run (apples-to-apples with original)

Run directory:

1. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_smalllegal_20260222_021600`

From `summary.json`:

1. `num_prompts`: `25000`
2. `num_round_records`: `263801`
3. `num_prompts_success_gt`: `14631`
4. `num_prompts_forced_accept`: `10369`
5. `elapsed_sec`: `35547.17` (~9.87 hours)

From `round_summary.csv`:

1. Round 1 mean: `-151.0275`
2. Round 16 mean: `-178.5183`
3. Round 1 median: `-149.0224`
4. Round 16 median: `-178.0879`
5. Accepted total across rounds: `25000`
6. Unresolved after final round: `0`

Artifacts:

1. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_smalllegal_20260222_021600/config.json`
2. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_smalllegal_20260222_021600/round_records.jsonl`
3. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_smalllegal_20260222_021600/round_summary.csv`
4. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_smalllegal_20260222_021600/prompt_summary.csv`
5. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_smalllegal_20260222_021600/summary.json`
6. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_smalllegal_20260222_021600/round_vs_logprob.png`

---

## Comparison Plots

1. Original vs small-legal (10% new run):
   - `artifacts/iter_round_logprob_diag/compare_round_vs_logprob_original_vs_smalllegal.png`
2. Original vs small-legal (25% apples-to-apples):
   - `artifacts/iter_round_logprob_diag/compare_round_vs_logprob_original_vs_smalllegal_25pct.png`

Command used for the apples-to-apples plot:

```bash
python scripts/plot_iter_round_logprob_compare.py \
  --original-run-dir artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113 \
  --new-run-dir artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_smalllegal_20260222_021600 \
  --output artifacts/iter_round_logprob_diag/compare_round_vs_logprob_original_vs_smalllegal_25pct.png
```

---

## Interpretation of Final Result

Apples-to-apples comparison (`sample_frac=0.25` for both runs) shows:

1. The original template’s **mean** degrades rapidly across rounds:
   - `-209.4241` -> `-360.1835` (delta `-150.7594`)
2. Under the new small-legal template, both **mean and median** are much more stable across rounds:
   - mean: `-151.0275` -> `-178.5183` (delta `-27.4908`)
   - median: `-149.0224` -> `-178.0879` (delta `-29.0655`)
3. Mean-vs-median separation differs sharply:
   - original average `(mean - median)` gap across rounds: `-127.27`
   - small-legal average `(mean - median)` gap across rounds: `-1.12`

Practical reading:

1. The original setup has a strong heavy-left-tail effect on round logprob sums, pulling the mean far below median.
2. The small-legal setup largely removes this divergence, producing tightly aligned mean/median trajectories.

---

## Smoke/Validation Runs Performed

Validation and debugging runs were done before full launch to de-risk OOM and semantics:

1. `artifacts/iter_round_logprob_diag/smoke_submitall_genonly`
2. `artifacts/iter_round_logprob_diag/smoke_submitall_genonly_k2`
3. `artifacts/iter_round_logprob_diag/smoke_submitall_genonly_k2_longer`
4. `artifacts/iter_round_logprob_diag/validate_submit_all`
5. `artifacts/iter_round_logprob_diag/sanity_fastpath`
6. `artifacts/iter_round_logprob_diag/subset16_u085_sb4_expandable`
7. `artifacts/iter_round_logprob_diag/subset64`
8. `artifacts/iter_round_logprob_diag/full10pct_submitall_genonly_u06_smalllegal_20260221_171746`
9. `artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_smalllegal_20260222_021600`

---

## Caveats and Notes

1. With large active sets, scoring all prompts in one call is memory-risky; generation and scoring were intentionally decoupled:
   - generation can submit-all per round,
   - scoring stays chunked (`--score_batch_size 4`).
2. `gpu_memory_utilization=0.6` was used for safety in the completed large run.
3. Stop criterion in this completed run is trainer-faithful (`gt_uci`), not subset argmax target (`target_move`).
4. The script supports both criteria for controlled comparison (`--stop_criterion {gt_uci,target_move}`).
5. Prompt text changes can affect both generation behavior and reference-context scoring; keep all non-prompt knobs fixed for fair A/B attribution.
