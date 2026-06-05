# pass@8 Evaluation Report (Isambard, DP=4)

Date: 2026-02-12

## Scope

- Evaluate chess puzzle `pass@8` for all discovered checkpoints of W&B runs `f5guq4ti`, `82fpo6l0`, `u2cuw56a` on Isambard only.
- Use vLLM sampling `temperature=0.6`, `top_p=0.95` and 4-GPU data-parallel sharded execution.
- Produce per-checkpoint metrics and a curve plot.

## Scripts and Code Paths

- `scripts/eval_chess_passk.py`: pass@k evaluator (vLLM generation + reward parsing).
- `sbatch_eval_chess_passk_gh200.slurm`: single-process launcher (used in early attempts and diagnostics).
- `sbatch_eval_chess_passk_dp4_smoke_gh200.slurm`: 4-way shard launcher (manual DP=4; 1 worker per GPU).
- `sbatch_merge_passk_checkpoints_gh200.slurm`: checkpoint merge job to materialize HF-weighted model dirs.
- `reports/passk_eval/passk8_dp4_summary_18checkpoints_sorted.csv`: consolidated final metrics used for this report.
- `reports/passk_eval/passk8_dp4_curves_all_runs.png`: local curve figure.

### Relevant Commits

- `6e54560`: `eval/passk: use k_max-specific summary key names` (removes hardcoded k32 summary naming).
- `6fb1643`, `96f9c61`: add/fix merge launcher for checkpoint sweep.
- `bb4297b`: add DP4 sharded smoke/full launcher.
- `4abb01f`, `7c9a4de`: fix DP4 launcher quoting/aggregation bugs.

## Resolved Checkpoints

- Source of truth: `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/merge_manifest_2279162.tsv`
- Checkpoint counts from manifest + final outputs:
  - `82fpo6l0`: 10 checkpoints
  - `f5guq4ti`: 1 checkpoints
  - `u2cuw56a`: 7 checkpoints
- Total evaluated checkpoints: 18

## Evaluation Configuration (Final Runs)

- Dataset: `data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet`
- Metric extraction: `summary.pass_at_k_mean` with `K_MAX=8`
- Generation: `temperature=0.6`, `top_p=0.95`, `--do_sample`
- DP launcher: `DP_SHARDS=4`, each shard runs `--tensor_parallel_size 1` on one GPU
- Throughput knobs: `BATCH_SIZE=256`, `MAX_NUM_SEQS=2048`
- Memory knob: `GPU_MEMORY_UTILIZATION=0.9`
- Full-eval rows per checkpoint: `num_prompts=10000` (all outputs)

## Job Timeline and Debug History

| Phase | Jobs | State | Evidence / Root cause |
|---|---:|---|---|
| Initial 18-checkpoint submission with wrong model subdir | `2279128-2279145` | FAILED | Missing weights from `.../actor/huggingface`; see `slurm/slurm-chess-passk-2279128.out` (`Cannot find any model weights ...`). |
| Single merge job to build reusable merged HF dirs | `2279162` | COMPLETED | Manifest written: `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/merge_manifest_2279162.tsv`. |
| 18-checkpoint retry at TP=4 + aggressive batching | `2279166-2279183` | FAILED | OOM on GH200; see `slurm/slurm-chess-passk-2279166.out` (`torch.OutOfMemoryError: CUDA out of memory`). |
| DP4 smoke attempt #1 | `2279208` | FAILED | Aggregator bug: `NameError` in `slurm/slurm-chess-passk-dp4-smoke-2279208.err`. |
| DP4 smoke attempt #2 | `2279218` | FAILED | Aggregator quoting bug: `SyntaxError` in `slurm/slurm-chess-passk-dp4-smoke-2279218.err`. |
| DP4 smoke fixed | `2279219` | COMPLETED | 100-row smoke pass@8 = `0.20999999716877937`; line in `slurm/slurm-chess-passk-dp4-smoke-2279219.out`. |
| Accidental TP=4 full run (not DP) | `2279230` | CANCELLED | User-requested cancellation (`sacct` reports `CANCELLED by 1483802860`). |
| First full DP4 run (`f5guq4ti`) | `2279238` | COMPLETED | Baseline full run for one checkpoint validated end-to-end. |
| Remaining 17 full DP4 runs | `2279372-2279388` | COMPLETED | All success (`job_state=COMPLETED`, `job_exit_code=0:0`). |

## Final Results (All 18 Checkpoints)

| run_id | step | pass@8 | pass@1 | job_id | elapsed | checkpoint_path | output_json | log_out |
|---|---:|---:|---:|---:|---:|---|---|---|
| 82fpo6l0 | 80 | 0.3095 | 0.0737 | 2279372 | 00:22:25 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/82fpo6l0/global_step_80_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/82fpo6l0/global_step_80_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-82fpo6l0-global_step_80-dp4-2279372.out` |
| 82fpo6l0 | 160 | 0.2845 | 0.0751 | 2279373 | 00:25:37 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/82fpo6l0/global_step_160_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/82fpo6l0/global_step_160_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-82fpo6l0-global_step_160-dp4-2279373.out` |
| 82fpo6l0 | 240 | 0.2768 | 0.0866 | 2279374 | 00:28:07 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/82fpo6l0/global_step_240_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/82fpo6l0/global_step_240_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-82fpo6l0-global_step_240-dp4-2279374.out` |
| 82fpo6l0 | 320 | 0.2999 | 0.1141 | 2279375 | 00:21:36 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/82fpo6l0/global_step_320_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/82fpo6l0/global_step_320_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-82fpo6l0-global_step_320-dp4-2279375.out` |
| 82fpo6l0 | 400 | 0.2802 | 0.1187 | 2279376 | 00:19:19 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/82fpo6l0/global_step_400_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/82fpo6l0/global_step_400_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-82fpo6l0-global_step_400-dp4-2279376.out` |
| 82fpo6l0 | 480 | 0.2640 | 0.1214 | 2279377 | 00:22:17 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/82fpo6l0/global_step_480_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/82fpo6l0/global_step_480_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-82fpo6l0-global_step_480-dp4-2279377.out` |
| 82fpo6l0 | 560 | 0.2628 | 0.1330 | 2279378 | 00:23:07 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/82fpo6l0/global_step_560_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/82fpo6l0/global_step_560_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-82fpo6l0-global_step_560-dp4-2279378.out` |
| 82fpo6l0 | 640 | 0.2960 | 0.1629 | 2279379 | 00:20:54 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/82fpo6l0/global_step_640_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/82fpo6l0/global_step_640_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-82fpo6l0-global_step_640-dp4-2279379.out` |
| 82fpo6l0 | 720 | 0.3271 | 0.1887 | 2279380 | 00:18:30 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/82fpo6l0/global_step_720_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/82fpo6l0/global_step_720_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-82fpo6l0-global_step_720-dp4-2279380.out` |
| 82fpo6l0 | 781 | 0.3165 | 0.1930 | 2279381 | 00:18:35 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/82fpo6l0/global_step_781_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/82fpo6l0/global_step_781_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-82fpo6l0-global_step_781-dp4-2279381.out` |
| f5guq4ti | 560 | 0.4253 | 0.1447 | 2279238 | 00:19:31 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/f5guq4ti/global_step_560_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/f5guq4ti/global_step_560_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-f5-dp4-2279238.out` |
| u2cuw56a | 80 | 0.3250 | 0.0872 | 2279382 | 00:23:01 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/u2cuw56a/global_step_80_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/u2cuw56a/global_step_80_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-u2cuw56a-global_step_80-dp4-2279382.out` |
| u2cuw56a | 160 | 0.3282 | 0.1148 | 2279383 | 00:19:26 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/u2cuw56a/global_step_160_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/u2cuw56a/global_step_160_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-u2cuw56a-global_step_160-dp4-2279383.out` |
| u2cuw56a | 240 | 0.3514 | 0.1435 | 2279384 | 00:21:22 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/u2cuw56a/global_step_240_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/u2cuw56a/global_step_240_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-u2cuw56a-global_step_240-dp4-2279384.out` |
| u2cuw56a | 320 | 0.3536 | 0.1784 | 2279385 | 00:20:02 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/u2cuw56a/global_step_320_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/u2cuw56a/global_step_320_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-u2cuw56a-global_step_320-dp4-2279385.out` |
| u2cuw56a | 400 | 0.3456 | 0.2051 | 2279386 | 00:21:15 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/u2cuw56a/global_step_400_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/u2cuw56a/global_step_400_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-u2cuw56a-global_step_400-dp4-2279386.out` |
| u2cuw56a | 480 | 0.3575 | 0.2091 | 2279387 | 00:21:06 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/u2cuw56a/global_step_480_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/u2cuw56a/global_step_480_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-u2cuw56a-global_step_480-dp4-2279387.out` |
| u2cuw56a | 560 | 0.3456 | 0.2206 | 2279388 | 00:20:58 | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/checkpoint_sweep/u2cuw56a/global_step_560_merged_hf` | `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/u2cuw56a/global_step_560_passk8_dp4_bs256_mseq2048_gmu09.json` | `/home/a5l/ziyan.a5l/code/chess-rl/slurm/slurm-chess-passk-u2cuw56a-global_step_560-dp4-2279388.out` |

### Best pass@8 by Run

| run_id | best_step | best_pass@8 | job_id |
|---|---:|---:|---:|
| 82fpo6l0 | 720 | 0.3271 | 2279380 |
| f5guq4ti | 560 | 0.4253 | 2279238 |
| u2cuw56a | 480 | 0.3575 | 2279387 |

## Output Artifacts

- Cluster summary CSV: `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/passk8_dp4_summary_18checkpoints.csv`
- Cluster sorted CSV: `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/passk8_dp4_summary_18checkpoints_sorted.csv`
- Cluster curve figure: `/projects/a5l/ziyan/chess_rl_outputs/passk_eval_runs/full/passk8_dp4_curves_all_runs.png`
- Local summary CSV: `reports/passk_eval/passk8_dp4_summary_18checkpoints_sorted.csv`
- Local curve figure: `reports/passk_eval/passk8_dp4_curves_all_runs.png`

## Reproduction Notes

- Cluster sync workflow used throughout: local `git push` then cluster `git pull --ff-only` from `~/code/chess-rl`.
- DP4 smoke/full launcher is the same script (`sbatch_eval_chess_passk_dp4_smoke_gh200.slurm`); full runs were submitted with `LIMIT=0` (no row cap).
- Final successful submission pattern (single checkpoint, no wait):

```bash
ssh a5l.aip2.isambard "cd ~/code/chess-rl && sbatch --nodes=1 --gres=gpu:4 \
  --export=ALL,MODEL=<merged_ckpt_dir>,PARQUET=data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet,\
K_MAX=8,LIMIT=0,DP_SHARDS=4,TEMPERATURE=0.6,TOP_P=0.95,BATCH_SIZE=256,MAX_NUM_SEQS=2048,GPU_MEMORY_UTILIZATION=0.9,\
OUT_JSON=<output_json> ./sbatch_eval_chess_passk_dp4_smoke_gh200.slurm"
```

- Curve plotting source: `reports/passk_eval/passk8_dp4_summary_18checkpoints_sorted.csv` -> `reports/passk_eval/passk8_dp4_curves_all_runs.png`.
