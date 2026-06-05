# Wallclock Debug Notes

Last updated: 2026-05-24

This note tracks long-running chess RL jobs that matter for debugging wallclock
time before relaunching the EMNLP Qwen3 ours experiments.

## New Session Handoff

Use **`4737303_1`** as the current reference SGLang+Isambard smoke run.

This is the run to compare against for near-term timing work:

- Slurm job: `4737303_1`
- Cluster: Isambard
- Submission style: foreground `sbatch --wait`, array task `1`, 45-minute limit
- Resources: 1 node, 4 GH200 GPUs
- Launcher: `deltaai_sglang_smoke.slurm`
- Log:
  `/projects/a5l/ziyan/chess_rl/logs/slurm-chess-sglang-smoke-4737303_1.out`
- Run dir: `/projects/a5l/ziyan/chess_rl/runs/4737303_1`
- Image: `/projects/a5l/ziyan/sif-images/verl_sglang059_isambard.sif`
- Model: `/projects/a5l/ziyan/models/Qwen/Qwen3-4B-Instruct-2507`
- Backend: SGLang, `fsdp2`, `enforce_eager=False`
- Gain filter: disabled via `ALLOWED_MOVE_ELIM_GAIN_THRESHOLD=inf`
- Lower-level verl SGLang rollout code: **not modified**

Reference offload configuration:

```bash
actor_rollout_ref.model.enable_activation_offload=False
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
actor_rollout_ref.actor.fsdp_config.offload_policy=False
actor_rollout_ref.ref.fsdp_config.param_offload=False
actor_rollout_ref.ref.fsdp_config.optimizer_offload=False
actor_rollout_ref.ref.fsdp_config.offload_policy=False
```

Reference timing:

| Metric | `4737303_1` |
| --- | ---: |
| Slurm elapsed | 25m36s |
| Full iteration, `timing_s/iteration` | 1275.1s |
| Rollout/pruning block, `timing_s/step` | 611.2s |
| Iterative generation, `timing_s/gen` | 602.6s |
| Old logprob | 119.7s |
| Reference logprob | 114.3s |
| Actor update | 425.7s |
| Old+ref+actor update | 659.7s |
| Avg rounds used | 3.445 |
| Success rate before forced accept | 32.0% |
| Forced accept fraction | 68.0% |
| Response length mean | 3197 tokens |
| Response clip ratio | 73.8% |
| Max GPU memory allocated | 56.54 GB |
| Max GPU memory reserved | 60.94 GB |
| Slurm batch MaxRSS | 176.4 GB |

Interpretation to carry into a new session:

- `4737303_1` is the clean reference for the requested all-offload-false
  SGLang+verl setup on Isambard.
- The run completed successfully with CUDA graph capture and
  `enforce_eager=False`, so do not patch verl's lower-level SGLang rollout code.
- The main bottleneck remains wall-clock spent in iterative generation and
  old/ref/actor update. Turning all offload knobs off helped, but it did not
  change the overall bottleneck structure.
- Ignore failed pre-training launcher attempts from the migration; they were Ray
  tmp/socket-path issues and do not provide useful training timing.

## Baseline Comparison Caveat

Do not compare `4737303_1` directly against the late-step `37-38s` core-step
speed from baseline job `4726530` without accounting for response length and
training stage.

The important observation from `4726530` is that the baseline became fast only
after it learned a very short output surface:

- Late `4726530` rows around global step 400:
  - `response_length/mean ~= 13` tokens
  - typical output: `<uci_move> f8f7 </uci_move>`
  - no `<think>...</think>` block
  - `timing_s/step ~= 4-5s`
  - old logprob + ref + actor update ~= `31-32s`
  - core train step ~= `36-38s`, excluding periodic validation/full-game eval

At the beginning of the same baseline run, before this short-output collapse,
the comparison is much closer to the expected `r_max` factor:

| Run/stage | Response length | Rollout/gen | Old+ref+actor | Core train time |
| --- | ---: | ---: | ---: | ---: |
| `4726530`, early global step 1 | ~4003 tokens | 185.1s | 208.2s | ~393s |
| `4737303_1`, ours smoke step 1 | ~3197 tokens | 602.6s | 659.7s | ~1271s |

The same-stage ratio is about `3.2x`, which is consistent with iterative
allowed-move elimination using `avg_rounds_used ~= 3.45` and `r_max=4`.

The apparent `>30x` gap comes from comparing late baseline behavior after the
model has learned to emit a 13-token answer against a first-step selection smoke
whose outputs are still thousands of tokens long. The selection smoke also
showed many long `<tool_call>...` style outputs, which are not useful training
responses.

This exposes a real contract issue:

- The prompts ask for `<think>...</think><uci_move>...</uci_move>`.
- The current reward parser gives `format_reward=1.0` for exactly one
  `<uci_move>...</uci_move>` span and intentionally ignores extra text outside
  it; it does not require a `<think>` block.
- Therefore the baseline can become very fast by emitting only `<uci_move>`.
  That may be acceptable if we intentionally want a terse answer-only policy,
  but it is not the strict selection contract described in `restricted_moves.md`
  and `AGENTS.md`.

Next debugging should decide explicitly whether speed/fairness should come from
relaxing the selection prompt to the same terse answer-only surface, or from
enforcing the strict two-tag contract for both baseline and ours. Mixing these
contracts makes wall-clock comparisons misleading.

## Current EMNLP Relaunch State

After smoke tests passed, three full EMNLP Qwen3 jobs were submitted on
Isambard. The two ours jobs were then canceled before they started, because the
4-node allocation was copied from the prior reference runs and may be more than
we want to spend while debugging wallclock.

Canceled jobs:

- `4726532`: `emnlp-qwen3-4b-ours-fixed`
- `4726534`: `emnlp-qwen3-4b-ours-online`

Kept running/pending:

- `4726530`: `emnlp-qwen3-4b-baseline-n8`

## Prior 4-Node Ours Runs That Nearly Filled 24h

Both prior ours reference runs below used 4 nodes with 4 GPUs per node
(16 GPUs total). W&B reports both as `crashed`, with `_runtime` very close to
24 hours. They are the main evidence that the 4-node ours configuration can
consume nearly a full 24h allocation.

### Fixed-Dataset Ours

- W&B run: [`h4rhtpg5`](https://wandb.ai/gabr1e11/chess_rl/runs/h4rhtpg5)
- W&B state: `crashed`
- W&B created at: `2026-01-28T11:55:14Z`
- W&B `_runtime`: `85993.17` seconds, about `23.89` hours
- W&B `_step`: `727`
- Nodes: `4`
- GPUs per node: `4`
- Total GPUs: `16`
- Approx GPU-hours: `382.2`
- Model: `/projects/a5l/ziyan/models/Qwen/Qwen2.5-3B-Instruct`
- Dataset: `data/chess_puzzles_chessr1_aligned_sharded_ours`
- Train files:
  - `data/chess_puzzles_chessr1_aligned_sharded_ours/train_0.parquet`
  - `data/chess_puzzles_chessr1_aligned_sharded_ours/train_1.parquet`
- Validation files:
  - `data/chess_puzzles_chessr1_aligned_sharded_ours/test.parquet`
  - `data/chess_puzzles_chessr1_aligned_sharded_ours/test_shuffled_legal_moves.parquet`
- `train_batch_size`: `128`
- `gen_batch_size`: `128`
- `total_epochs`: `1`
- `total_training_steps`: `None` (epoch-derived)
- `max_prompt_length`: `1536`
- `max_response_length`: `2000`
- `allowed_move_elim.enable`: `True`
- `allowed_move_elim.uid_mode`: `per_prompt`
- `allowed_move_elim.r_max_start`: `4`
- `allowed_move_elim.r_max_end`: `4`
- `allowed_move_elim.group_reward_range_dump_max_groups`: `16`
- `self_play.enable`: `False`
- `trainer.test_freq`: `40`
- `trainer.save_freq`: `80`
- `trainer.full_eval_freq`: `80`

### Online Engine-Play Ours

- W&B run: [`h6sqp0z4`](https://wandb.ai/gabr1e11/chess_rl/runs/h6sqp0z4)
- W&B state: `crashed`
- W&B created at: `2026-01-28T11:53:19Z`
- W&B `_runtime`: `86116.12` seconds, about `23.92` hours
- W&B `_step`: `720`
- Nodes: `4`
- GPUs per node: `4`
- Total GPUs: `16`
- Approx GPU-hours: `382.7`
- Model: `/projects/a5l/ziyan/models/Qwen/Qwen2.5-3B-Instruct`
- Training source: online self-play (`data.train_files=[]`)
- Validation files:
  - `data/chess_puzzles_chessr1_aligned_sharded_ours/test.parquet`
  - `data/chess_puzzles_chessr1_aligned_sharded_ours/test_shuffled_legal_moves.parquet`
- `train_batch_size`: `128`
- `gen_batch_size`: `128`
- `total_epochs`: `1`
- `total_training_steps`: `800`
- `max_prompt_length`: `1536`
- `max_response_length`: `2000`
- `allowed_move_elim.enable`: `True`
- `allowed_move_elim.uid_mode`: `per_prompt`
- `allowed_move_elim.r_max_start`: `4`
- `allowed_move_elim.r_max_end`: `4`
- `allowed_move_elim.group_reward_range_dump_max_groups`: `-1`
- `self_play.enable`: `True`
- `self_play.num_parallel_games`: `128`
- `trainer.test_freq`: `40`
- `trainer.save_freq`: `80`
- `trainer.full_eval_freq`: `80`

## Immediate Debugging Questions

- How much wallclock is spent in iterative allowed-move elimination rounds versus
  actor update, old log-prob, reference log-prob, validation, saving, and
  full-game evaluation?
- Does `r_max=4` usually run all four rounds on fixed-dataset prompts, or does it
  terminate early often enough to amortize the extra sampling?
- How much do `trainer.test_freq=40`, `trainer.save_freq=80`, and
  `trainer.full_eval_freq=80` contribute near the 24h limit?
- The prior 4-node runs used Qwen2.5-3B with `max_response_length=2000`.
  The new EMNLP runs use Qwen3-4B with `max_response_length=4096`, so wallclock
  may be worse unless step throughput improves elsewhere or evaluation/saving is
  reduced.

## Useful New Smoke Reference Points

These smokes used Qwen3-4B with `max_response_length=4096` and all micro-batch
settings set to `8`.

- Baseline smoke `4715073`: one trainer step completed with full
  `TRAIN_BATCH_SIZE=128`.
- Fixed-dataset smoke `4715234`: full `TRAIN_BATCH_SIZE=128`, 15-minute smoke
  timed out before completing one trainer step.
- Fixed-dataset smoke `4715484`: smoke-only `TRAIN_BATCH_SIZE=16`, one trainer
  step completed; logged `timing_s/step=157.28`.
- Online smoke `4715807`: smoke-only `TRAIN_BATCH_SIZE=16`, one trainer step
  completed; logged `timing_s/step=11.01`.

## Existing W&B Timing Evidence

W&B history is useful, but it has an important caveat: in the trainer version
used by these runs, `timing_s/step` / `perf/time_per_step` is not the full
trainer iteration wall time. It is the timed rollout-generation block. It does
not include old log-prob, reference log-prob, actor update, validation,
checkpointing, or full-game eval. The closest full-iteration estimate from
existing W&B runs is the `_timestamp` delta between logged training rows.

Downloaded local evidence:

- `analysis/wandb_evidence/h4rhtpg5/`
- `analysis/wandb_evidence/h6sqp0z4/`
- `analysis/wandb_evidence/82fpo6l0/`
- `analysis/wandb_evidence/s0anl08n/`

Mean per logged training row:

| Run | Regime | Wall delta mean | Wall p50 | Gen/step | Old logprob | Ref | Actor update | Periodic eval amortized | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `h4rhtpg5` | fixed ours, Qwen2.5-3B, 4 nodes | 117.4s | 106.9s | 58.6s | 7.7s | 6.9s | 26.6s | 11.9s | 727 rows, 23.7h by timestamp deltas |
| `h6sqp0z4` | online ours, Qwen2.5-3B, 4 nodes | 118.8s | 107.2s | 47.1s | 7.9s | 7.3s | 28.6s | 11.5s | 720 rows, 23.8h by timestamp deltas |
| `82fpo6l0` | baseline, Qwen2.5-3B, 1 node | 95.5s | 72.2s | 24.8s | 8.3s | 7.8s | 29.2s | 24.2s | 781 rows, validation was unusually expensive |
| `s0anl08n` | fixed ours, Qwen2.5-7B, 4 nodes | 141.1s | 130.0s | 60.8s | 11.0s | 10.4s | 41.8s | 12.3s | 603 rows, slower actor/logprob path |

Aggregate totals over the nearly-24h ours runs:

| Run | Generation total | Actor update total | Old+ref logprob total | Testing total | Full-game eval total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `h4rhtpg5` | 11.83h | 5.37h | 2.94h | 1.21h | 1.20h |
| `h6sqp0z4` | 9.42h | 5.72h | 3.05h | 0.90h | 1.41h |
| `82fpo6l0` | 5.39h | 6.33h | 3.49h | 4.30h | 0.96h |
| `s0anl08n` | 10.18h | 7.00h | 3.57h | 1.01h | 1.05h |

Takeaways from existing W&B:

- For the prior Qwen2.5-3B ours runs, generation plus actor update dominate.
  Periodic validation/full-game eval was meaningful but not the main cause:
  about 2.3-2.4h of a 24h run for the two main ours references.
- The baseline reference is not a clean speed baseline for training throughput:
  it used 1 node and spent much more time in validation/testing. Its training
  core is still faster because it does one rollout pass rather than iterative
  pruning.
- Existing W&B cannot answer the key question inside iterative pruning:
  per-round generation, per-round reward/postprocess/dump, and gain-filter
  logprob were not separately logged.

## Gain Filtering Provenance

The archived January 2026 Qwen2.5 runs did not use the newly added gain filter.
Evidence checked:

- W&B configs for `h4rhtpg5`, `h6sqp0z4`, and `s0anl08n` have
  `allowed_move_elim` enabled but no
  `algorithm.allowed_move_elim.gain_threshold`.
- Their histories contain normal `selection_sampler/*` metrics but no
  `selection_sampler/gain_*`, `allowed_move_elim_gain*`, or
  `timing_s/allowed_move_elim/*gain*` columns.
- The baseline `82fpo6l0` run has `allowed_move_elim.enable=False` and no gain
  metrics.
- Repo history shows the feature arrived later in commit `42d3ff9`
  (`2026-02-25`, "Add gain-based iterative sample filter for
  allowed_move_elim"). The W&B runs were created on `2026-01-28`.

This matters because the current code defaults missing
`allowed_move_elim.gain_threshold` to `ln(10)`, which is not faithful to those
Qwen2.5 runs. For faithful Qwen2.5-style comparisons under current code, set:

```bash
ALLOWED_MOVE_ELIM_GAIN_THRESHOLD=inf
```

One caveat from launcher inspection: `sbatch_train_chess_gh200.slurm` maps many
`CHESS_ALLOWED_MOVE_ELIM_*` variables into unprefixed variables, but currently
does not map `CHESS_ALLOWED_MOVE_ELIM_GAIN_THRESHOLD`. Use the unprefixed
variable directly unless the Slurm wrapper is updated.

## Added Timing Instrumentation

I added trainer-side timers in `verl/trainer/ppo/ray_trainer.py` so the next
diagnostic logs the missing major components:

- `timing_s/iteration`: full training-loop iteration wall time.
- `timing_s/balance_batch`.
- `timing_s/allowed_move_elim/round{r}_build_round_batch`.
- `timing_s/allowed_move_elim/round{r}_gen`.
- `timing_s/allowed_move_elim/round{r}_generate_sequences` from worker
  `meta_info`, accumulated without losing per-round values.
- `timing_s/allowed_move_elim/round{r}_reward`.
- `timing_s/allowed_move_elim/round{r}_postprocess`.
- `timing_s/allowed_move_elim/round{r}_dump_rollouts`.
- `timing_s/allowed_move_elim/concat_round_batches`.
- `timing_s/allowed_move_elim/gain_filter_total`.
- `timing_s/allowed_move_elim/loss_weight_and_padding`.
- Self-play batch construction timers for future online diagnostics, including
  model move generation, opponent moves, Stockfish scoring, and row building.
- Also fixed `allowed_move_elim.anneal_frac=0.0` parsing: the previous
  `or 0.5` fallback silently converted an explicit zero back to `0.5`. This
  did not affect diagnostic job `2331039` because `r_max_start == r_max_end`.

Local and container compile checks passed:

```bash
python -m py_compile verl/trainer/ppo/ray_trainer.py
```

## DeltaAI Diagnostic Run

The requested short diagnostic was launched on DeltaAI with `sbatch --wait`.

There was one setup-only failure first:

- Job `2331033`, failed after 29s before training because the remote rsync
  excluded directories named `data`, which accidentally omitted
  `verl/trainer/config/data/legacy_data.yaml`.
- Fixed by syncing that config file to the remote copy.

Successful diagnostic:

- Job: `2331039`
- Cluster/partition: DeltaAI `ghx4-interactive`
- Resources: 1 node, 4 GH200 GPUs, `--time=01:00:00`
- Result: completed successfully
- Slurm elapsed: about 52m
- Training progress time: `2/2` steps in 49m34s
- Log: `/projects/bgba/zzhang69/chess_rl_wallclock_diag/logs/slurm-chess-wc-diag-2331039.out`
- W&B offline run:
  `/projects/bgba/zzhang69/chess_rl_wallclock_diag/runs/2331039/wandb/wandb/offline-run-20260523_134744-ouwe0si7`
- Remote repo:
  `/projects/bgba/zzhang69/code/chess-rl-wallclock-debug`
- Container:
  `/projects/bgba/zzhang69/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif`

Diagnostic config:

- Model: `/projects/bgba/zzhang69/models/Qwen/Qwen3-4B-Instruct-2507`
- Dataset:
  `/projects/bgba/zzhang69/code/chess-rl-wallclock-debug/data/chess_puzzles_chessr1_aligned_sharded_ours`
- `TRAIN_BATCH_SIZE=128`
- `GEN_BATCH_SIZE=128`
- `PPO_MINI_BATCH_SIZE=128`
- `ROLLOUT_N=8`
- `MAX_PROMPT_LENGTH=1536`
- `MAX_RESPONSE_LENGTH=4096`
- `TOTAL_TRAINING_STEPS=2`
- `VAL_BEFORE_TRAIN=False`
- `TRAINER_TEST_FREQ=-1`
- `TRAINER_SAVE_FREQ=-1`
- `FULL_EVAL_FREQ=-1`
- `ALLOWED_MOVE_ELIM_ENABLE=True`
- `ALLOWED_MOVE_ELIM_UID_MODE=per_prompt`
- `ALLOWED_MOVE_ELIM_R_MAX_START=4`
- `ALLOWED_MOVE_ELIM_R_MAX_END=4`
- `ALLOWED_MOVE_ELIM_GAIN_THRESHOLD=2.302585092994046`

Mean timing over the two diagnostic steps:

| Component | Mean seconds/step | Share of full iteration | Notes |
| --- | ---: | ---: | --- |
| Full iteration, `timing_s/iteration` | 1485.0s | 100.0% | New full-loop timer |
| Rollout/pruning block, `timing_s/step` | 1087.0s | 73.2% | Includes iterative generation plus gain filter |
| Iterative generation wall, `timing_s/gen` | 806.1s | 54.3% | Four rollout rounds |
| Gain filter total | 272.8s | 18.4% | Two extra logprob passes over generated samples |
| Actor update | 257.6s | 17.3% | After rollout/gain filtering |
| Old logprob | 68.8s | 4.6% | Training batch logprob |
| Reference logprob | 68.9s | 4.6% | KL/reference path |
| Reward | 3.2s | 0.2% | Reward parsing/scoring is not the bottleneck |
| Round rollout JSONL dumps | 3.5s | 0.2% | Per-round dump overhead is small |
| Final rollout JSONL dump | 2.0s | 0.1% | Not the bottleneck |
| Build round batches | 1.3s | 0.1% | Not the bottleneck |
| Balance batch | 0.5s | 0.0% | Not the bottleneck |

Per-round generation timing:

| Round | Prompt count mean | Success frac mean | Wall gen mean | Worker generate mean |
| --- | ---: | ---: | ---: | ---: |
| 1 | 128.0 | 7.4% | 212.5s | 200.4s |
| 2 | 118.5 | 7.6% | 201.1s | 190.3s |
| 3 | 109.5 | 6.8% | 200.0s | 185.8s |
| 4 | 102.0 | 5.4% | 192.6s | 179.4s |

Sampler behavior:

| Metric | Step 1 | Step 2 | Mean |
| --- | ---: | ---: | ---: |
| Avg rounds used | 3.594 | 3.562 | 3.578 |
| Success rate before forced accept | 27.3% | 21.9% | 24.6% |
| Forced accept fraction | 72.7% | 78.1% | 75.4% |
| Gain samples | 3680 | 3648 | 3664 |
| Gain-filtered samples | 1509 | 1331 | 1420 |
| Gain-filtered fraction | 41.0% | 36.5% | 38.7% |
| Response length mean | 2903 | 3000 | 2951 |
| Response clip ratio at 4096 | 66.9% | 70.3% | 68.6% |

Key diagnostic takeaways:

- Iterative pruning is not amortizing much in this Qwen3 fixed-dataset setting:
  the sampler uses 3.58 of 4 rounds on average. About three quarters of prompts
  reach forced accept in round 4 rather than finding a successful earlier round.
- Generation is the largest component by far. The run is producing very long
  responses: mean response length is about 2950 tokens and about 69% of
  responses hit the 4096-token cap. This directly explains why full-batch Qwen3
  rollout is slow.
- The gain filter is also expensive: about 273s/step on 1 node, because it does
  two additional logprob passes over about 3.6k generated samples. It filtered
  about 39% of samples in this diagnostic, so disabling it would save time but
  would also materially change the training data.
- Reward computation, batch construction, JSONL dumping, and batch balancing are
  all small relative to generation/logprob/update. They are not first-order
  wallclock bottlenecks.
- Old logprob plus reference logprob plus actor update are still substantial:
  together they are about 396s/step on this 1-node diagnostic, roughly 27% of
  the full iteration.

## Current Bottleneck Ranking

For the current Qwen3 full-batch ours path, the likely bottleneck order is:

1. Long rollout generation across nearly all 4 iterative rounds.
2. Gain-filter logprob passes, if the gain filter remains enabled.
3. Actor update plus old/reference logprob.
4. Periodic validation/full-game eval in full runs.
5. Reward parsing, JSONL dumping, batch construction, and balancing.

This differs slightly from the prior Qwen2.5-3B W&B runs. There, generation and
actor update dominated, but per-step generation was much shorter because the
old runs used `max_response_length=2000` and the model/run mix was different.

## SGLang + verl Image Smoke

I tested the newly available SGLang+verl image on DeltaAI with the requested
offload settings and `fsdp2`.

The clean result to trust is job `2331978`: it used the stock
`verl/workers/rollout/sglang_rollout/sglang_rollout.py`, set
`actor_rollout_ref.rollout.enforce_eager=False`, captured CUDA graphs
successfully, decoded with `cuda graph: True`, and completed a full trainer
step. No lower-level rollout code modification was needed.

Clean stock-code SGLang smoke:

- Job: `2331978`
- Cluster/partition: DeltaAI `ghx4-interactive`
- Resources: 1 node, 4 GH200 GPUs, `--time=01:00:00`
- Result: completed successfully
- Slurm elapsed: `00:25:53`, exit code `0:0`
- Training progress time: `1/1` step in `22:44.29`
- Log:
  `/projects/bgba/zzhang69/chess_rl_wallclock_diag/logs/slurm-chess-sglang-stock-2331978.out`
- W&B offline run:
  `/projects/bgba/zzhang69/chess_rl_wallclock_diag/runs/2331978/wandb/wandb/offline-run-20260523_181554-xo99cwtm`
- Image: `/projects/bgba/zzhang69/sif-images/gabr1e1_verl_sglang059.sif`
- Preflight: `torch 2.9.1+cu129`, `sglang 0.5.9`, `python-chess 1.11.2`

Clean smoke config:

- Model: `/projects/bgba/zzhang69/models/Qwen/Qwen3-4B-Instruct-2507`
- Dataset:
  `/projects/bgba/zzhang69/code/chess-rl-wallclock-debug/data/chess_puzzles_chessr1_aligned_sharded_ours`
- `TRAIN_BATCH_SIZE=128`
- `GEN_BATCH_SIZE=128`
- `PPO_MINI_BATCH_SIZE=128`
- `ROLLOUT_N=8`
- `MAX_PROMPT_LENGTH=1536`
- `MAX_RESPONSE_LENGTH=4096`
- `TOTAL_TRAINING_STEPS=1`
- `VAL_BEFORE_TRAIN=False`
- `TRAINER_TEST_FREQ=-1`
- `TRAINER_SAVE_FREQ=-1`
- `FULL_EVAL_FREQ=-1`
- `ALLOWED_MOVE_ELIM_ENABLE=True`
- `ALLOWED_MOVE_ELIM_UID_MODE=per_prompt`
- `ALLOWED_MOVE_ELIM_R_MAX_START=4`
- `ALLOWED_MOVE_ELIM_R_MAX_END=4`
- `ALLOWED_MOVE_ELIM_GAIN_THRESHOLD=inf`
- `ROLLOUT_NAME=sglang`
- `ENFORCE_EAGER=False`
- `FSDP_STRATEGY=fsdp2`
- `USE_REMOVE_PADDING=True`
- `actor_rollout_ref.model.enable_activation_offload=True`
- Actor FSDP:
  `param_offload=True`, `optimizer_offload=False`, `offload_policy=False`
- Reference FSDP:
  `param_offload=True`, `optimizer_offload=False`, `offload_policy=True`

Timing for the clean stock-code SGLang smoke:

| Component | Seconds | Share of full iteration | Notes |
| --- | ---: | ---: | --- |
| Full iteration, `timing_s/iteration` | 1363.6s | 100.0% | One completed trainer iteration |
| Rollout/pruning block, `timing_s/step` | 640.3s | 47.0% | Iterative generation; gain filter disabled |
| Iterative generation wall, `timing_s/gen` | 631.6s | 46.3% | Four SGLang rollout rounds |
| Actor update | 471.3s | 34.6% | Same slow post-rollout path as the exploratory run |
| Old logprob | 126.7s | 9.3% | Training batch logprob |
| Reference logprob | 120.9s | 8.9% | KL/reference path |
| Reward | 3.6s | 0.3% | Not a bottleneck |
| Round rollout JSONL dumps | 3.7s | 0.3% | Per-round dump overhead is small |
| Final rollout JSONL dump | 3.5s | 0.3% | Not a bottleneck |
| Gain filter total | ~0.0s | 0.0% | Disabled via `inf` |

Per-round clean SGLang generation timing:

| Round | Prompt count | Success frac | Wall gen | Worker generate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 128 | 7.8% | 179.2s | 174.2s |
| 2 | 118 | 7.6% | 159.9s | 155.9s |
| 3 | 109 | 3.7% | 152.8s | 149.6s |
| 4 | 105 | 7.6% | 139.7s | 136.1s |

Sampler behavior in the clean SGLang smoke:

| Metric | Value |
| --- | ---: |
| Avg rounds used | 3.594 |
| Success rate before forced accept | 24.2% |
| Forced accept fraction | 75.8% |
| Response length mean | 3252 tokens |
| Response clip ratio at 4096 | 75.3% |
| Total tokens | 14.84M |
| Throughput | 5793 tokens/s |

Clean SGLang takeaways:

- The stock SGLang+verl path works with `enforce_eager=False`; the run captured
  CUDA graphs and decoded with CUDA graph enabled. This points away from a
  lower-level SGLang rollout-code problem.
- Compared with the earlier vLLM diagnostic, SGLang generation was materially
  faster in this smoke (`632s` vs the vLLM two-step mean of `806s` for
  `timing_s/gen`) even though this SGLang step had longer average responses.
  This is still not a perfectly controlled backend comparison because the vLLM
  diagnostic had gain filtering enabled and averaged two stochastic steps, while
  this SGLang run disabled gain filtering and ran one step.
- The post-rollout FSDP2/offload path is now the larger share of wall time:
  old logprob plus reference logprob plus actor update took about `719s`, which
  is more than the `640s` rollout/pruning block. The requested offload settings
  likely trade memory headroom for wallclock here.
- The same high-level bottleneck remains: iterative generation still uses almost
  all four rounds, and generated responses are very long. SGLang helps rollout
  time, but the 4096-token cap and round count still dominate rollout cost.

Actor parameter offload ablation:

I reran the same clean stock SGLang smoke with
`actor_rollout_ref.actor.fsdp_config.param_offload=False`, leaving rollout
`gpu_memory_utilization=0.7` unchanged because no OOM occurred.

- Job: `2332073`
- Cluster/partition: DeltaAI `ghx4-interactive`
- Resources: 1 node, 4 GH200 GPUs, `--time=01:00:00`
- Result: completed successfully
- Slurm elapsed: `00:25:32`, exit code `0:0`
- Training progress time: `1/1` step in `22:27.62`
- Log:
  `/projects/bgba/zzhang69/chess_rl_wallclock_diag/logs/slurm-chess-sglang-actor-no-po-2332073.out`
- W&B offline run:
  `/projects/bgba/zzhang69/chess_rl_wallclock_diag/runs/2332073/wandb/wandb/offline-run-20260523_185009-d8efbvn8`

Comparison against the previous clean SGLang run:

| Metric | Actor param offload True (`2331978`) | Actor param offload False (`2332073`) | Delta |
| --- | ---: | ---: | ---: |
| Slurm elapsed | 25m53s | 25m32s | -21s |
| Full iteration | 1363.6s | 1346.7s | -16.9s |
| Rollout/pruning block | 640.3s | 636.7s | -3.7s |
| Iterative generation | 631.6s | 628.4s | -3.3s |
| Old logprob | 126.7s | 124.6s | -2.1s |
| Reference logprob | 120.9s | 120.1s | -0.7s |
| Actor update | 471.3s | 460.6s | -10.7s |
| Old+ref+actor update | 718.8s | 705.3s | -13.5s |
| Max GPU memory allocated | 49.77 GB | 49.77 GB | ~0 |
| Max GPU memory reserved | 54.62 GB | 54.82 GB | +0.20 GB |
| Slurm batch MaxRSS | ~209.5 GB | 181.5 GB | -28.0 GB |
| Response length mean | 3252 tokens | 3291 tokens | +39 |
| Response clip ratio | 75.3% | 76.4% | +1.1 pp |
| Avg rounds used | 3.594 | 3.539 | -0.055 |

Interpretation:

- Turning actor parameter offload off worked without changing rollout memory
  utilization and without OOM.
- It slightly improved wall-clock in this one-step smoke, mostly in actor
  update. The improvement is modest, around 17s on a 1347s iteration, so this is
  not a first-order speed fix by itself.
- The GPU memory metrics were essentially unchanged, while Slurm batch RSS was
  lower in the no-actor-param-offload run. Treat that RSS comparison cautiously
  because Slurm CPU RSS accounting can vary with timing and process lifetime,
  but there is no evidence here that disabling actor param offload increased CPU
  memory pressure.
- The bottleneck conclusion is unchanged: rollout still spends almost all four
  rounds generating very long responses, and old/ref/actor post-processing
  remains about 705s even with actor param offload disabled.

Isambard reference SGLang smoke:

Use `4737303_1` as the reference SGLang+Isambard timing point. It is the
successful all-offload-false run summarized in the "New Session Handoff" section
at the top of this file.

The portable launcher changes that matter for future runs are:

- Use the SGLang+verl image
  `/projects/a5l/ziyan/sif-images/verl_sglang059_isambard.sif`.
- Keep `ROLLOUT_NAME=sglang`, `FSDP_STRATEGY=fsdp2`, and
  `ENFORCE_EAGER=False`.
- Use foreground smoke submission, e.g.
  `sbatch --wait --time=00:45:00 ... ./deltaai_sglang_smoke.slurm`.
- On Isambard, keep Ray/tmp under the short project path
  `/projects/a5l/ziyan/crl/...`; long Ray temp paths can exceed the Unix socket
  limit, and host-local `/local/...` is not reliable inside the container.
- Do not touch `verl/workers/rollout/sglang_rollout/sglang_rollout.py`.

Non-reference migration attempts were pruned from this note because they failed
before training or were canceled and contain no useful timing signal.

The old exploratory patched-adapter SGLang run was removed from this note. It
temporarily modified lower-level rollout code during debugging, so it should not
be used as timing evidence.

## Qwen3-4B-Base One-Step Baseline

DeltaAI run:

- Job: `2333748` on `ghx4-interactive`, node `gh104`
- State: `COMPLETED`, Slurm elapsed `00:05:12`
- Log:
  `/projects/bgba/zzhang69/chess_rl_wallclock_diag/logs/slurm-emnlp-qwen3-4b-base-baseline-1step-2333748.out`
- Run root:
  `/projects/bgba/zzhang69/chess_rl_wallclock_diag/runs/2333748`
- Model: `/projects/bgba/zzhang69/models/Qwen/Qwen3-4B-Base`
- Dataset:
  `data/chess_puzzles_chessr1_aligned_sharded_baseline_base/{train_0,train_1,test}.parquet`
- Prompt format: plain base prompt, `USE_CHAT_TEMPLATE=False`
- Rollout: SGLang, FSDP2, all offload false, `ENFORCE_EAGER=False`,
  `ROLLOUT_N=8`, `MAX_RESPONSE_LENGTH=4096`

The launcher had to force container-side compiler paths for SGLang/TVM JIT
compilation on DeltaAI. Host shells can export `CXX=CC`, but the Apptainer image
does not provide that wrapper; the smoke launcher now sets `CC=/usr/bin/gcc` and
`CXX=/usr/bin/g++` unless explicitly overridden via
`CHESS_RL_CONTAINER_{CC,CXX}`.

Training-step metrics:

| Metric | Value |
| --- | ---: |
| Full iteration | 132.96s |
| Step timing | 58.14s |
| Generate sequences | 55.11s |
| Old logprob | 16.49s |
| Reference logprob | 12.48s |
| Actor update | 44.68s |
| Total tokens | 1,321,931 |
| Throughput | 5,684 tokens/s |
| Max GPU memory allocated | 46.84 GB |
| Max GPU memory reserved | 50.97 GB |
| CPU memory used | 467.25 GB |
| Reward mean | -0.84375 |
| Effective batch size | 88 / 128 groups |

Length and format observations:

- Official rollout metrics: response length mean `911.81`, min `1`, max `4096`,
  clip ratio `10.45%`; prompt length mean `379.14`, max `527`, clip ratio `0`.
- Retokenized rollout dump quantiles for output text:
  p50 `273.5`, p75 `990.75`, p90 `4071.4`, p95 `4096`.
- Dumped rollout rows: `1024` samples, with score counts `{1.0: 11, 0.0: 138,
  -1.0: 875}`.
- Strict two-tag contract count was `0/1024`; `<uci_move>` tags appeared in
  `371/1024` samples, but `<think>...</think>` did not appear in the dump.

Interpretation:

- The base model setup now runs without applying a chat template, and SGLang
  confirms no HF chat template is active.
- Base-model outputs are shorter than the earlier instruct iterative diagnostic
  but still often run near the 4096 cap.
- The current reward path still gives useful scores to some non-strict outputs;
  this is a reward-parser/contract issue, not a chat-template issue.

### Revised Base Prompt + Response Prefix

DeltaAI run:

- Job: `2334134` on `ghx4-interactive`, node `gh018`
- State: `COMPLETED`, Slurm elapsed `00:04:12`
- Log:
  `/projects/bgba/zzhang69/chess_rl_wallclock_diag/logs/slurm-emnlp-qwen3-4b-base-baseline-prefix-1step-2334134.out`
- Run root:
  `/projects/bgba/zzhang69/chess_rl_wallclock_diag/runs/2334134`
- Prompt format: revised plain base prompt, `USE_CHAT_TEMPLATE=False`
- Response prefix: `actor_rollout_ref.rollout.response_prefix_enable=True`,
  `response_prefix='<think>'`

Training-step metrics:

| Metric | Previous base run | Revised prompt + prefix |
| --- | ---: | ---: |
| Slurm elapsed | 5m12s | 4m12s |
| Full iteration | 132.96s | 79.69s |
| Step timing | 58.14s | 34.14s |
| Generate sequences | 55.11s | 31.15s |
| Old logprob | 16.49s | 11.09s |
| Reference logprob | 12.48s | 7.35s |
| Actor update | 44.68s | 26.06s |
| Total tokens | 1,321,931 | 593,920 |
| Throughput | 5,684 tokens/s | 4,349 tokens/s |
| Max GPU memory allocated | 46.84 GB | 43.11 GB |
| Max GPU memory reserved | 50.97 GB | 47.22 GB |
| Response length mean | 911.81 | 302.86 |
| Response clip ratio | 10.45% | 2.34% |
| Prompt length mean | 379.14 | 277.14 |
| Reward mean | -0.84375 | -0.87988 |
| Effective batch size | 88 / 128 groups | 76 / 128 groups |

Rollout dump checks:

- Rows: `1024`
- Decoded responses starting with `<think>`: `1024/1024`
- Strict two-tag contract: `0/1024`
- Any `<uci_move>...</uci_move>` tag pair: `445/1024`
- Score counts: `{1.0: 9, 0.0: 105, -1.0: 910}`
- Format reward counts: `{1.0: 395, 0.0: 629}`
- Retokenized response quantiles:
  p50 `14`, p75 `107.25`, p90 `854.3`, p95 `2225.9`, p99 `4096`

Interpretation:

- The response-side prefix is active and visible to reward/logging.
- The revised prompt plus prefix reduced response length and wall-clock
  substantially for this one-step diagnostic.
- The model still does not close `</think>` under this setup. More explicit
  continuation control, stop handling, or training signal is still needed for
  strict contract compliance.

## Iterative Generation Speedup Findings

The current allowed-move-elimination loop is round-synchronous:

1. Build a round batch for all unresolved prompts.
2. Repeat it by `rollout.n`.
3. Call `generate_sequences`.
4. Wait for the full round batch to finish.
5. Parse/reward/prune, then build the next round.

The algorithm itself does not require a global round barrier. A prompt can move
from round `r` to round `r+1` as soon as all `rollout.n` samples for that prompt
and round have completed, because the allowed-move update is per original
prompt. The current `verl` trainer/rollout API does not expose that partial
completion surface: sync rollout returns one completed `DataProto`, and the
existing async manager still gathers all requests before returning to
`PPOTrainer`.

Ranked improvement options:

| Rank | Option | Expected impact | Complexity | Notes |
| ---: | --- | ---: | ---: | --- |
| 1 | Stop at `</uci_move>` with `include_stop_str_in_output=True` | Potentially very high | Low | Needs posthoc check that completions often contain the closing tag before the 4096 cap. |
| 2 | Reuse gain-filter logprobs as old logprobs | Medium when gain filter is enabled | Low-medium | The gain filter already computes actor logprobs; kept samples can reuse the matching tensor for PPO old logprobs if context/stitching is handled exactly. |
| 3 | Reduce or adapt `r_max` | High | Low-medium | Round 4 is expensive and low-success in the Qwen3 diagnostic, but this changes training data. |
| 4 | Keep rollout mode awake across all iterative rounds | Low-medium | Medium | Avoids repeated rollout/trainer mode transitions; diagnostic upper bound is roughly the wall-vs-worker generation gap, about 50s/step. |
| 5 | Per-prompt async scheduler across rounds | Medium-high | High | Directly answers "do not wait for the large batch"; requires a new scheduler/rollout surface using async server requests or a dedicated rollouter. |

The safest sequence is to test response termination first, then exact logprob
reuse if the gain filter stays enabled, then `r_max`/adaptive policy ablations.
The async per-prompt scheduler is feasible, but it is a real scheduler redesign,
not a small config flip.

## Improvement Room To Test Next

These are the highest-value speed experiments suggested by the timing data:

1. Quantify response-length control for Qwen3.
   Run a paired diagnostic with the same config but `MAX_RESPONSE_LENGTH=2000`
   and possibly a stricter prompt/format setup. The current 4096-token cap is
   being hit by about 69% of responses, so generation time should be highly
   sensitive to response length.

2. Measure gain-filter cost/benefit directly.
   Run the same 2-step diagnostic with `ALLOWED_MOVE_ELIM_GAIN_THRESHOLD=inf`.
   Expected direct saving on 1 node is about 273s/step, but quality/signal may
   change because the current filter drops about 39% of samples.

3. Reconsider `r_max=4` or the pruning policy.
   The current sampler still uses 3.58 rounds on average. If earlier rounds do
   not succeed often enough, iterative pruning behaves close to four full
   rollout passes plus extra logprob work.

4. Keep periodic eval disabled or less frequent while debugging throughput.
   From existing W&B, validation/full-game eval contributed about 2.3-2.4h of
   the prior 24h ours runs. That is worth reducing, but it is not the main
   bottleneck compared with current Qwen3 generation.

5. Check whether Qwen3 thinking behavior can be controlled without violating
   the selection contract.
   The long response and high clip ratio suggest the model often keeps thinking
   until the cap. Any change here needs correctness checks because the output
   must remain exactly `<think>...</think><uci_move>...</uci_move>`.
