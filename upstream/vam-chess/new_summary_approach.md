# New Summary Approach

## Current status

If you only read one part of this document, read sections `8` and `9`.

The short version is:

- the summary-generation harness is now reliable enough to complete full `32k` runs
- the newest iterative prompt still underperforms baseline on `pass@8`
- the current blocker is route lock-in and summary-driven anchoring, not widespread harness failure

## Method

### Goal

The goal of this line of work is to test a softer inference-time alternative to explicit action-space restriction.

Historically, the repo’s restricted-moves work used an explicit shortlist such as:

```text
legal moves: {set}
allowed moves: {subset}
```

That older setup tells the model in advance which moves are allowed. The current math experiment is intentionally different. We are **not** enumerating an action space and then masking it. Instead, we are testing whether we can improve search coverage by giving the model a compact memory of reasoning paths that have already been tried and did not solve the problem, then asking it to solve the problem through a materially different route.

This is prompt-only steering, not a hard constraint. Later attempts can still repeat an old route if the model ignores or only weakly responds to the explored-path memory.

The central prompt shape is:

```text
[Problem]
...

[Explored paths]
<explore id="1">
{summary of a previous failed path}
</explore>

<explore id="2">
{summary of another previous failed path}
</explore>

[Instruction]
The <explore> blocks above are route memory from prior failed attempts, not verified facts or proofs.
- Do not repeat the same reasoning route.
- Solve the problem using a materially different reasoning path.
- Make the difference substantive: change the framing, representation, tactic, case split, invariant, algorithm, examples, or another major reasoning choice.
```

### What the method is trying to achieve

The intended behavior is:

1. Sample an initial solution attempt from the normal math prompt.
2. If the attempt fails, summarize the route the model took.
3. Feed that route summary back into the next prompt as negative memory.
4. Repeat for up to 8 attempts.

The hypothesis is that this route-level exclusion mechanism may increase effective coverage compared with naive repeated sampling, even though it never enumerates all legal next actions.

### Current summary format

The current summary prompt is no longer a loose one-line string. It is a comparative route-memory prompt with guided JSON decoding.

The summarizer sees three explicit blocks:

```text
PROBLEM:
{original problem}

PRIOR_SUMMARIES:
{summaries of earlier failed attempts}

CURRENT_ATTEMPT:
{full reasoning trace for the current attempt}
```

The active summary schema lives in [explored_paths_summary_prompt.py](/usr0/home/zhichen3/chess-rl/explored_paths_summary_prompt.py) and has these fields:

- `attempted_answer`
- `route`
- `route_signature`
- `relationship_to_prior.differences`
- `relationship_to_prior.overlap`

The important design idea is that `route` and `route_signature` should summarize only the current attempt, while `relationship_to_prior` records how it differs from or overlaps with earlier attempts.

### What is being compared

Each comparison keeps the total attempt budget fixed at 8 and uses the same model, sampling settings, and output-length cap.

The two methods are:

- `baseline`: 8 independent samples from the ordinary prompt
- `iterative`: up to 8 sequential samples, where each later prompt includes accumulated summaries of earlier failed routes

### What is being measured

The main outcome metrics are:

- `pass@1`
- `pass@8`
- diversity-style diagnostics such as unique final answers and distinct path summaries

Later in the session we added summary-quality analysis, because the iterative method depends on the path summaries being useful. If the summaries are poor, then the explored-path memory may fail to steer the model away from repeated reasoning routes.

### Current implementation path

The actual implementation lives in:

- [scripts/eval_aime24_explored_paths_vllm.py](/usr0/home/zhichen3/chess-rl/scripts/eval_aime24_explored_paths_vllm.py)

For new multi-dataset runs, prefer launching through the stable wrapper:

- [scripts/eval_math_explored_paths_vllm.py](/usr0/home/zhichen3/chess-rl/scripts/eval_math_explored_paths_vllm.py)

Important implementation details:

- The active summary prompt and schema live in [explored_paths_summary_prompt.py](/usr0/home/zhichen3/chess-rl/explored_paths_summary_prompt.py), not inline in the evaluator anymore.
- Summary generation uses guided JSON decoding plus schema-level length bounds.
- Summary generation retries up to `5` times. If a problem still fails after all retries, that problem is skipped and recorded in `*_skipped_problems.jsonl` rather than crashing the whole run.
- The current cluster path uses direct `vllm.LLM.generate`, not the repo’s older async OpenAI-style wrapper path.
- Cluster `dp=4` is implemented as 4 one-GPU workers sharded by problem, then merged afterward.
- The iterative method stores negative memory only at the problem level. It does **not** split a single problem’s attempt history across shards.

The cluster launcher is:

- [sbatch_eval_math_explored_paths_gh200.slurm](/usr0/home/zhichen3/chess-rl/sbatch_eval_math_explored_paths_gh200.slurm)

It launches one shard per GPU and then runs a merge-only pass over raw traces.

### Retained artifacts and what is canonical after cleanup

The explored-path work generated a lot of smoke output, partial reruns, and judge-execution residue. The artifacts below are the retained bundles worth documenting for later sessions. Some are canonical records, while others are kept as browse views, provenance, or regeneration context.

- Local proof-of-concept summary: [analysis/aime24_explored_paths/20260416_qwen3_8b_localmodel_summary.md](/usr0/home/zhichen3/chess-rl/analysis/aime24_explored_paths/20260416_qwen3_8b_localmodel_summary.md) and [analysis/aime24_explored_paths/20260416_qwen3_8b_localmodel_summary.json](/usr0/home/zhichen3/chess-rl/analysis/aime24_explored_paths/20260416_qwen3_8b_localmodel_summary.json). These are the quickest entrypoint for the local `Qwen3-8B` AIME24 proof-of-concept and point to the retained `*_localmodel` trace directories.
- Local proof-of-concept raw traces: [analysis/aime24_explored_paths/20260416_qwen3_8b_8k_localmodel](/usr0/home/zhichen3/chess-rl/analysis/aime24_explored_paths/20260416_qwen3_8b_8k_localmodel) and [analysis/aime24_explored_paths/20260416_qwen3_8b_32k_localmodel](/usr0/home/zhichen3/chess-rl/analysis/aime24_explored_paths/20260416_qwen3_8b_32k_localmodel). Keep these when you need the per-attempt traces, metrics, or curated examples behind the summary file.
- Completed GH200 bundle: [analysis/math_explored_paths/cluster_job_3877433](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433). This is the authoritative retained copy of the completed `AIME24/AIME25/AMC23` `dp=4` run and contains the merged results overview, run config, raw traces, and downstream analyses.
- Canonical iterative trace layout inside that bundle: `raw_iterative_traces/<dataset>/len_<max_tokens>/iterative_traces.jsonl`. Older flat duplicate copies were pruned during cleanup; later analyses should read the nested layout.
- Intermediate GPQA summary-fix browse bundle: [analysis/math_explored_paths/summary_fix_aime25_gpqa_diamond_32k_wait_20260419T053407Z](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/summary_fix_aime25_gpqa_diamond_32k_wait_20260419T053407Z). Keep this only as the retained local inspection bundle for the GPQA-only summary-fix pass. The canonical retained file inside it is `raw/gpqa_diamond_iterative_traces.jsonl`; [gpqa_diamond_iterative_problem_md/index.md](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/summary_fix_aime25_gpqa_diamond_32k_wait_20260419T053407Z/gpqa_diamond_iterative_problem_md/index.md) and the sibling `gpqa_diamond_iterative_problem_md/*.md` files are the concise human-readable view. `gpqa_diamond_iterative_problem_pages/*.md` is a more verbose alternate render and should be treated as derived browse output, not a separate source of truth.
- Failure-analysis companion bundle for the newest full run: [analysis/math_explored_paths/full_iterative_vs_baseline_aime25_gpqa_32k_ea08b62_20260420T195129Z](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/full_iterative_vs_baseline_aime25_gpqa_32k_ea08b62_20260420T195129Z). This local tree is the retained post-hoc analysis companion to the cluster run, not a replacement for the run root itself. Treat [iterative_failure_case_analysis.md](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/full_iterative_vs_baseline_aime25_gpqa_32k_ea08b62_20260420T195129Z/iterative_failure_case_analysis.md) as the canonical readout; keep [iterative_failure_case_analysis_manifest.json](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/full_iterative_vs_baseline_aime25_gpqa_32k_ea08b62_20260420T195129Z/iterative_failure_case_analysis_manifest.json), `source_traces/*.jsonl`, and `run_iterative_failure_case_analysis.py` as regeneration context. The per-case `iterative_failure_case_analysis_outputs/*.analysis.txt`, `iterative_failure_case_analysis_prompts/*.prompt.txt`, `iterative_failure_case_analysis_logs/*.stderr.log`, and `iterative_failure_case_overall_summary*` files are retained audit/provenance material rather than canonical headline results, and `__pycache__/` is cache-like residue.
- Summary-quality canonical dataset: [summary_quality_scored_examples.jsonl](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/summary_quality_scored_examples.jsonl). This is the authoritative joined table because it carries the judged summary, the underlying attempted solution text, and the final `codex exec` rating in one place. [summary_quality_results.jsonl](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/summary_quality_results.jsonl) is still worth keeping as provenance, but it is an append-only audit log rather than the canonical analysis table.
- Summary-quality regeneration context: [judge_prompt_template.txt](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/judge_prompt_template.txt) and [commands_used.sh](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/commands_used.sh). `commands_used.sh` is a preserved historical command log from the original run, not a normalized post-cleanup manifest of the retained nested trace layout, so use it as provenance/regeneration context and use `raw_iterative_traces/<dataset>/len_<max_tokens>/iterative_traces.jsonl` as the canonical trace layout.
- Outcome-join and contamination reports: keep the derived markdown/json/csv files under [summary_quality_eval](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval) because they encode the final conclusions from this session without requiring another expensive judge pass.

## Experiments

This section summarizes the experiments that were actually run in this session and the key scripts needed to reproduce them.

Important scope note:

- The full six-dataset expansion (`AIME24`, `AIME25`, `AMC23`, `MATH500`, `Minerva`, `OlympiadBench`) was smoke-validated in this session.
- The first completed multi-dataset `dp=4` bundle in this line of work was the smaller three-dataset run on `AIME24`, `AIME25`, and `AMC23`.
- A later full `32k` bundle with the newest comparative summary prompt completed on `AIME25` and `GPQA Diamond`.

### 1. Local AIME24 proof-of-concept with Qwen3-8B

This was the initial narrow test used to validate the explored-path idea before expanding to the cluster bundle.

#### What it tested

- Dataset: `AIME24`
- Model: local `Qwen3-8B`
- Backend: local `vllm`
- Output lengths: `8192` and `32768`
- Comparison: `baseline` vs `iterative`

#### Main script

- [scripts/eval_aime24_explored_paths_vllm.py](/usr0/home/zhichen3/chess-rl/scripts/eval_aime24_explored_paths_vllm.py)

#### Environment

```bash
source ~/.bashrc
conda activate verl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,3
export VLLM_CACHE_ROOT=/tmp/zhichen3/vllm_cache_qwen3_8b_eval
export TMPDIR=/tmp/zhichen3/chess_rl_eval_tmp
export RAY_TMPDIR=/tmp/zhichen3/ray_qwen3_8b_eval
mkdir -p "$VLLM_CACHE_ROOT" "$TMPDIR" "$RAY_TMPDIR"
```

#### Commands

`32k` baseline + partial iterative run:

```bash
python scripts/eval_aime24_explored_paths_vllm.py \
  --model /usr0/home/zhichen3/models/Qwen/Qwen3-8B \
  --datasets aime24 \
  --solve-max-tokens 32768 \
  --concurrency 4 \
  --tensor-parallel-size 2 \
  --gpus-per-node 2 \
  --gpu-memory-utilization 0.8 \
  --out-dir analysis/aime24_explored_paths/20260416_qwen3_8b_32k_localmodel
```

`32k` iterative-only resume:

```bash
python scripts/eval_aime24_explored_paths_vllm.py \
  --model /usr0/home/zhichen3/models/Qwen/Qwen3-8B \
  --datasets aime24 \
  --solve-max-tokens 32768 \
  --methods iterative \
  --concurrency 4 \
  --tensor-parallel-size 2 \
  --gpus-per-node 2 \
  --gpu-memory-utilization 0.8 \
  --out-dir analysis/aime24_explored_paths/20260416_qwen3_8b_32k_localmodel
```

`8k` full run:

```bash
python scripts/eval_aime24_explored_paths_vllm.py \
  --model /usr0/home/zhichen3/models/Qwen/Qwen3-8B \
  --datasets aime24 \
  --solve-max-tokens 8192 \
  --concurrency 4 \
  --tensor-parallel-size 2 \
  --gpus-per-node 2 \
  --gpu-memory-utilization 0.8 \
  --out-dir analysis/aime24_explored_paths/20260416_qwen3_8b_8k_localmodel
```

#### Results summary

See:

- [20260416_qwen3_8b_localmodel_summary.md](/usr0/home/zhichen3/chess-rl/analysis/aime24_explored_paths/20260416_qwen3_8b_localmodel_summary.md)

The short version is:

- `8k`: iterative improved `pass@8` from `0.5000` to `0.5667`
- `32k`: iterative improved `pass@8` from `0.8333` to `0.9000`

This was encouraging, but too small and narrow to be conclusive.

### 2. Cluster smoke path for the multi-dataset Qwen3-1.7B setup

After the local proof-of-concept, the experiment was expanded to a cluster path intended for:

- `AIME24`
- `AIME25`
- `AMC23`
- `MATH500`
- `Minerva`
- `OlympiadBench`

using `Qwen3-1.7B` on Isambard GH200.

#### What the smoke run was for

The smoke run was not meant to produce final scientific results. It was used to verify:

- model loading on the cluster
- Hugging Face dataset access
- local `vllm` generation
- per-dataset trace writing
- baseline/iterative loop correctness
- prompt/summary plumbing

#### Main scripts

- [scripts/eval_aime24_explored_paths_vllm.py](/usr0/home/zhichen3/chess-rl/scripts/eval_aime24_explored_paths_vllm.py)
- [scripts/eval_math_explored_paths_vllm.py](/usr0/home/zhichen3/chess-rl/scripts/eval_math_explored_paths_vllm.py)
- [sbatch_eval_math_explored_paths_gh200.slurm](/usr0/home/zhichen3/chess-rl/sbatch_eval_math_explored_paths_gh200.slurm)

#### Representative smoke command

From the cluster repo checkout:

```bash
RUN_TAG=math_exp_paths_smoke \
MODEL_PATH=/projects/a5l/ziyan/models/Qwen/Qwen3-1.7B \
HF_TOKEN_FILE=~/.hf_token_zzc \
DATASETS="aime24 aime25 amc23 math500 minerva olympiadbench" \
METHODS="baseline iterative" \
ATTEMPTS=8 \
LIMIT_PER_DATASET=1 \
NUM_SHARDS=1 \
GPUS_PER_NODE=1 \
TENSOR_PARALLEL_SIZE=1 \
CONCURRENCY=4 \
SOLVE_MAX_TOKENS="8192 32768" \
sbatch --wait --gres=gpu:1 --cpus-per-task=8 \
  ./sbatch_eval_math_explored_paths_gh200.slurm
```

#### Launcher defaults that matter

The cluster launcher reads or defaults the following values:

- Hugging Face token file: `~/.hf_token_zzc`
- model path: `/projects/a5l/ziyan/models/Qwen/Qwen3-1.7B`
- datasets: `aime24 aime25 amc23 math500 minerva olympiadbench`
- methods: `baseline iterative`
- attempts: `8`
- solve max tokens: `8192 32768`
- `TENSOR_PARALLEL_SIZE=1` is required in this shard-per-GPU path; the launcher hard-fails otherwise

All of those come from [sbatch_eval_math_explored_paths_gh200.slurm](/usr0/home/zhichen3/chess-rl/sbatch_eval_math_explored_paths_gh200.slurm), so they should be spelled out when reproducing the run instead of relying on memory or launcher defaults.

#### Important note

One smoke job timed out because the Slurm time limit was too short, not because the harness was fundamentally broken. That smoke was still useful because it validated the actual data flow and wrote real artifacts for the early datasets before timing out.

### 3. Completed cluster `dp=4` bundle on AIME24 / AIME25 / AMC23 with Qwen3-1.7B

This was the first completed cluster run using the final `dp=4` shard-and-merge path.

#### What it tested

- Model: `/projects/a5l/ziyan/models/Qwen/Qwen3-1.7B`
- Datasets: `AIME24`, `AIME25`, `AMC23`
- Output lengths: `8192`, `32768`
- Attempts per problem: `8`
- Methods: `baseline`, `iterative`
- Cluster parallelism: 4 one-GPU shards merged afterward

#### Main script and launcher

- [scripts/eval_aime24_explored_paths_vllm.py](/usr0/home/zhichen3/chess-rl/scripts/eval_aime24_explored_paths_vllm.py)
- [sbatch_eval_math_explored_paths_gh200.slurm](/usr0/home/zhichen3/chess-rl/sbatch_eval_math_explored_paths_gh200.slurm)

#### Command

From the cluster checkout:

```bash
RUN_TAG=math_exp_paths_small_dp4_20260417_v1 \
MODEL_PATH=/projects/a5l/ziyan/models/Qwen/Qwen3-1.7B \
HF_TOKEN_FILE=~/.hf_token_zzc \
DATASETS="aime24 aime25 amc23" \
METHODS="baseline iterative" \
ATTEMPTS=8 \
SOLVE_MAX_TOKENS="8192 32768" \
GPUS_PER_NODE=4 \
NUM_SHARDS=4 \
TENSOR_PARALLEL_SIZE=1 \
CONCURRENCY=16 \
sbatch --job-name=math-exp-paths-small-dp4 --time=06:00:00 \
  ./sbatch_eval_math_explored_paths_gh200.slurm
```

#### Main output artifacts

Cluster output root:

- `/projects/a5l/ziyan/chess_rl_outputs/math_explored_paths/math_exp_paths_small_dp4_20260417_v1`

Copied local summaries:

- [results_overview.json](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/results_overview.json)
- [results_overview.md](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/results_overview.md)
- [run_config.json](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/run_config.json)

One nuance for newcomers: the merged [run_config.json](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/run_config.json) reflects one shard worker plus merge metadata, so fields such as visible GPUs are shard-local. The real job-level parallelism for this run was still `NUM_SHARDS=4`.

#### Results summary

See:

- [results_overview.md](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/results_overview.md)

Selected results:

- `AIME24`, `8192`: baseline `pass@8 = 0.4667`, iterative `0.4667`
- `AIME24`, `32768`: baseline `0.7667`, iterative `0.7333`
- `AIME25`, `8192`: baseline `0.2667`, iterative `0.3333`
- `AIME25`, `32768`: baseline `0.6000`, iterative `0.5333`
- `AMC23`, `8192`: baseline `0.8000`, iterative `0.8500`
- `AMC23`, `32768`: baseline `0.9500`, iterative `0.9250`

This run showed mixed behavior rather than a clean overall win for the iterative method.

### 4. Full iterative trajectory export for the completed cluster bundle

Once the cluster run finished, the full iterative trajectories were materialized into a human-readable artifact.

This was an analysis artifact derived from the merged iterative traces of job `3877433`. There is not yet a dedicated checked-in exporter script for this exact markdown/json rendering. The authoritative source of truth is still the bundle’s raw iterative traces:

- [raw_iterative_traces](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/raw_iterative_traces)

That means a newcomer should treat the saved export below as the exact artifact from this session, and treat the nested raw trace layout under `raw_iterative_traces/<dataset>/len_<max_tokens>/iterative_traces.jsonl` as the reproducible input for later analyses.

#### Artifact

- [iterative_trajectories_aime24_aime25_amc23.md](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/iterative_trajectories_aime24_aime25_amc23.md)
- [iterative_trajectories_aime24_aime25_amc23.json](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/iterative_trajectories_aime24_aime25_amc23.json)

#### What it contains

For every problem and both output lengths:

- attempts `1..8`
- extracted answer
- correctness
- score
- summary
- explored paths shown before the attempt

This artifact made it possible to inspect repeated-route behavior and notice summary-pathology patterns directly.

### 5. Summary quality evaluation with `codex exec`

After seeing inconsistent iterative performance, the next question was whether low-quality summaries were part of the problem.

#### What it tested

Every stored iterative summary in the completed cluster bundle was scored with `codex exec` on a `1–5` scale.

Important nuance:

- For these datasets, there is generally **not** a full official worked solution in the benchmark rows.
- The summary is meant to summarize the model’s own attempted route.
- Therefore the evaluation judged the summary against the original problem and the **full raw attempted solution** from the trace, not against an official gold proof.

#### Main script

- [scripts/score_summary_quality_codex_exec.py](/usr0/home/zhichen3/chess-rl/scripts/score_summary_quality_codex_exec.py)

#### Execution requirements used

- `codex exec` launched from `/usr0/home/zhichen3/codex_exec`
- `16` workers
- `15` seconds sleep between batches
- outputs saved in the project tree under `analysis/`
- the `codex exec` judge was not version-pinned in this session, so the run is operationally reproducible but not perfectly frozen at the judge-model level

#### Command

From the repo root:

```bash
python scripts/score_summary_quality_codex_exec.py \
  --trace-dir analysis/math_explored_paths/cluster_job_3877433/raw_iterative_traces \
  --output-dir analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval \
  --codex-cwd /usr0/home/zhichen3/codex_exec \
  --codex-workspace /usr0/home/zhichen3/chess-rl \
  --workers 16 \
  --sleep-between-batches 15
```

#### Main output artifacts

- [judge_prompt_template.txt](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/judge_prompt_template.txt)
- [summary_quality_scored_examples.jsonl](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/summary_quality_scored_examples.jsonl)
- [summary_quality_results.jsonl](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/summary_quality_results.jsonl)
- [summary_quality_stats.md](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/summary_quality_stats.md)
- [commands_used.sh](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/commands_used.sh)

Treat [summary_quality_scored_examples.jsonl](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/summary_quality_scored_examples.jsonl) as the authoritative scored table. [summary_quality_results.jsonl](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/summary_quality_results.jsonl) is useful provenance because it records the append-only judge outputs, but it is not the canonical post-cleanup analysis table. Likewise, [commands_used.sh](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/commands_used.sh) is historical context from the original run rather than a normalized manifest of the retained nested trace layout; after cleanup, the canonical trace input layout is `raw_iterative_traces/<dataset>/len_<max_tokens>/iterative_traces.jsonl`.

The transient request dump and per-example judge-response cache used during scoring were removed during cleanup because they are derivable from the raw traces plus the retained scored-examples file.

#### Results summary

Overall summary quality:

- total rows: `1600`
- mean rating: `3.630625`
- median rating: `4`
- histogram: `1:99, 2:301, 3:242, 4:408, 5:550`

Important splits:

- `32768`: mean `4.0125`
- `8192`: mean `3.24875`
- non-fallback summaries: mean `3.719844`
- fallback summaries: mean `1.258621`

Important caveat:

- this early fallback/non-fallback split used the narrower exact-string detector that marked `58` rows as fallback
- the later dedicated fallback-contamination analysis broadened the placeholder definition and found `162` fallback cases, which should be treated as the more faithful count

This provided strong quantitative evidence that the summary layer was usable but noisy, and that the fallback summaries were especially poor.

### 6. Join summary quality back to iterative outcomes

After scoring summary quality, the next question was whether better summaries actually corresponded to better iterative recovery.

#### Main script

- [scripts/analyze_summary_quality_vs_outcome.py](/usr0/home/zhichen3/chess-rl/scripts/analyze_summary_quality_vs_outcome.py)

#### Command

From the repo root:

```bash
python scripts/analyze_summary_quality_vs_outcome.py
```

This command uses the default inputs:

- [summary_quality_scored_examples.jsonl](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/summary_quality_scored_examples.jsonl)
- `analysis/math_explored_paths/cluster_job_3877433/raw_baseline_traces`

#### Output artifacts

- [summary_quality_vs_outcome_summary.md](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/summary_quality_vs_outcome_summary.md)
- [summary_quality_vs_outcome_problem_level.csv](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/summary_quality_vs_outcome_problem_level.csv)

#### Results summary

The key result was that higher failed-summary quality moved iterative recovery in the expected direction, but not enough to fully explain the method’s behavior on its own.

In the `baseline failed` slice:

- high failed-summary quality (`>= 4`): iterative recovery `22.2%`
- mid quality (`3 to <4`): `12.0%`
- low quality (`<3`): `9.1%`

That supported the view that summary quality matters, but it did not yet isolate the fallback-placeholder problem sharply enough.

### 7. Dedicated fallback-contamination analysis

The strongest later analysis was a dedicated look at the fallback-summary failure mode.

#### Main script

- [scripts/analyze_fallback_contamination.py](/usr0/home/zhichen3/chess-rl/scripts/analyze_fallback_contamination.py)

#### Command

From the repo root:

```bash
python scripts/analyze_fallback_contamination.py \
  --bundle-dir analysis/math_explored_paths/cluster_job_3877433
```

#### Output artifacts

- [fallback_contamination_summary.md](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/fallback_contamination_summary.md)
- [fallback_contamination_summary.json](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/fallback_contamination_summary.json)
- [fallback_contamination_problem_level.jsonl](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/fallback_contamination_problem_level.jsonl)

#### Results summary

This analysis concluded that fallback contamination is a **major** issue for the completed cluster bundle.

Key findings:

- `162 / 1600` iterative attempts are fallback placeholders (`10.1%`)
- `87 / 200` problem-length records contain at least one fallback placeholder (`43.5%`)
- `206 / 1400` later attempts had fallback memory already in the prompt (`14.7%`)
- `172 / 749` failed later attempts ran under fallback-contaminated prompt memory (`23.0%`)

Most important outcome slice:

- baseline-fail problems with no pre-success prompt contamination: `8 / 41` iterative recoveries (`19.5%`)
- baseline-fail problems with pre-success prompt contamination: `0 / 26` iterative recoveries (`0.0%`)

This was strong enough to justify pausing any attempted fix in this session and explicitly carrying it forward as the main problem for the next session.

Important historical note:

- This conclusion applied to the older `cluster_job_3877433` bundle before the summary rewrite and retry-and-skip changes.
- It is still important background, but it is **not** the current main blocker for the newest prompt.
- The current blocker is described in section `9`, after the newer `AIME25 / GPQA Diamond` full run.

### 8. Full `32k` AIME25 / GPQA Diamond bundle with the newest comparative summary prompt

This run used the current comparative route-memory prompt, the `<explore>...</explore>` iterative prompt format, the structured summary schema, and the retry-and-skip summary handling.

#### What it tested

- Model: `/projects/a5l/ziyan/models/Qwen/Qwen3-1.7B`
- Datasets: `AIME25`, `GPQA Diamond`
- Output lengths: `32768`
- Attempts per problem: `8`
- Methods: `baseline`, `iterative`
- Cluster parallelism: 4 one-GPU shards merged afterward

#### Main script and launcher

- [scripts/eval_aime24_explored_paths_vllm.py](/usr0/home/zhichen3/chess-rl/scripts/eval_aime24_explored_paths_vllm.py)
- [scripts/eval_math_explored_paths_vllm.py](/usr0/home/zhichen3/chess-rl/scripts/eval_math_explored_paths_vllm.py)
- [sbatch_eval_math_explored_paths_gh200.slurm](/usr0/home/zhichen3/chess-rl/sbatch_eval_math_explored_paths_gh200.slurm)

#### Command

From a clean cluster worktree at commit `ea08b62`:

```bash
RUN_TAG=full_iterative_vs_baseline_aime25_gpqa_32k_ea08b62_20260420T195129Z \
HF_TOKEN_FILE=~/.hf_token_zzc \
MODEL_PATH=/projects/a5l/ziyan/models/Qwen/Qwen3-1.7B \
DATASETS="aime25 gpqa_diamond" \
METHODS="baseline iterative" \
ATTEMPTS=8 \
NUM_SHARDS=4 \
GPUS_PER_NODE=4 \
TENSOR_PARALLEL_SIZE=1 \
CONCURRENCY=16 \
SOLVE_MAX_TOKENS="32768" \
sbatch ./sbatch_eval_math_explored_paths_gh200.slurm
```

#### Main output artifacts

Cluster output root:

- `/projects/a5l/ziyan/chess_rl_outputs/math_explored_paths/full_iterative_vs_baseline_aime25_gpqa_32k_ea08b62_20260420T195129Z`

Useful merged files inside that root:

- `results_overview.json`
- `aime25/len_32768/metrics.json`
- `gpqa_diamond/len_32768/metrics.json`
- `aime25/len_32768/{baseline,iterative}_traces.jsonl`
- `gpqa_diamond/len_32768/{baseline,iterative}_traces.jsonl`

#### Results summary

Final merged metrics:

- `AIME25`
  - baseline: `pass@1 = 0.3103`, `pass@8 = 0.5862`, `29/30` scored, `1` skipped
  - iterative: `pass@1 = 0.4000`, `pass@8 = 0.5000`, `30/30` scored, `0` skipped
- `GPQA Diamond`
  - baseline: `pass@1 = 0.3706`, `pass@8 = 0.6548`, `197/198` scored, `1` skipped
  - iterative: `pass@1 = 0.3636`, `pass@8 = 0.4646`, `198/198` scored, `0` skipped

Important coverage note:

- The harness problem is no longer the main blocker.
- The retry-and-skip path dropped only `2` baseline problems total across both datasets.
- Iterative coverage was `100%` in this full run.

Important comparison note:

- Relative to the earlier completed reference results for these datasets under the older summary setup, baseline stayed close while iterative got worse on `pass@8`.
- `AIME25` iterative `pass@8` moved from `0.6000` to `0.5000`.
- `GPQA Diamond` iterative `pass@8` moved from `0.5303` to `0.4646`.

Important prompt-behavior note:

- Iterative path-summary diversity was much lower than baseline in the merged metrics.
- `AIME25`: iterative `avg_distinct_path_summaries = 4.5` vs baseline `8.0`
- `GPQA Diamond`: iterative `3.67` vs baseline `7.99`
- The fraction of problems with repeated path summaries was also much higher for iterative.

That means the current open problem is no longer “the summary harness crashes too often.” The current open problem is that the iterative method still underperforms baseline, especially on `pass@8`, even after the summary-generation path was made reliable enough to finish the run.

### 9. Failure analysis of the newest prompt

After the full `32k` run completed, the next question was why iterative was still losing to baseline.

#### Analysis method

We focused on the sharpest failure slice:

- cases where baseline solved the problem within 8 attempts
- but iterative never solved it

Using the merged top-level traces from the run root above:

- `AIME25`: `3` such cases
- `GPQA Diamond`: `45` such cases
- total: `48` cases

We then ran a case-by-case analysis using `codex exec -c features.fast_mode=false` from `~/codex_exec`.

The workflow was:

1. Build a compact per-case prompt from the actual merged traces.
2. Include the problem text, gold answer, baseline attempts, iterative attempts, and the explored-path summaries shown before later iterative attempts.
3. Ask `codex exec` to briefly explain why iterative failed.
4. Process cases in parallel in batches of `32` workers, following the same `15` second between-batch cadence recorded in the retained analysis workflow.
5. Run one final `codex exec` pass to summarize the cross-case findings.

The main local analysis artifacts are:

- [iterative_failure_case_analysis.md](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/full_iterative_vs_baseline_aime25_gpqa_32k_ea08b62_20260420T195129Z/iterative_failure_case_analysis.md)
- [iterative_failure_case_analysis_manifest.json](/usr0/home/zhichen3/chess-rl/analysis/math_explored_paths/full_iterative_vs_baseline_aime25_gpqa_32k_ea08b62_20260420T195129Z/iterative_failure_case_analysis_manifest.json)

Within that tree, the markdown report is the canonical human-readable conclusion. The per-case `*.analysis.txt`, `*.prompt.txt`, and `*.stderr.log` files are retained audit trails, `iterative_failure_case_overall_summary*` is provenance from the cross-case summary step, and `source_traces/` plus `run_iterative_failure_case_analysis.py` provide the regeneration context.

#### What the failure analysis found

The main conclusion is that the newest iterative prompt usually did **not** produce real route search after a bad first attempt.

Across the `48` baseline-win / iterative-fail cases:

- `47 / 48` stayed on a single iterative route signature
- `48 / 48` used two or fewer iterative route signatures total
- `25 / 48` had effectively one route text for all iterative attempts
- `22 / 48` showed fake diversification: more than one route text, but still only one route signature
- `35 / 48` reused the first answer at least `7` times
- `29 / 48` reused the first answer all `8` attempts
- `18 / 48` changed extracted answers on the surface while still staying on a single route signature

Baseline behavior in the same cases looked very different:

- baseline found a correct branch by attempt `2` in `24 / 48` cases
- baseline found a correct branch by attempt `4` in `36 / 48` cases
- median first-correct baseline attempt in this slice was `2.5`

#### What we currently believe

The current open problem is now prompt-behavioral rather than harness-level.

The strongest working interpretation is:

- the comparative summary prompt is now reliable enough to run at scale
- but the iterative prompting scheme still produces strong anchoring
- later attempts often paraphrase or lightly perturb the first failed route instead of switching branches
- the summaries often act as error reinforcement rather than useful negative memory

In short, the current iterative method is spending extra attempts on repeated or near-repeated failures, while baseline often solves the same problems by branching earlier.

#### What should happen next session

The next session should focus on reducing route lock-in rather than on further harness repair.

The main questions are now:

1. how to make the explored-path memory discourage repetition without anchoring the model to the first bad route
2. whether the summaries should be shorter, less prescriptive, or filtered before being fed back into the prompt
3. whether later attempts should see all prior summaries, only a subset, or a transformed version of them

The evidence in this document says that the current bottleneck is not “too many summary-generation crashes.” It is “too little genuine route diversification after the first failed attempt.”
