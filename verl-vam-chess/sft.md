# Rejection-Sampling SFT Runbook (Chess-R1 Baseline)

Last updated: 2026-02-09 (UTC)

This is the current runbook for the baseline-prompt rejection-sampling SFT workflow.
It is written to be restart-safe and reproducible from scratch.

## 0) Scope

This document covers the pipeline that:
1. Builds rejection-sampled SFT datasets from the Chess-R1-aligned baseline train shards.
2. Trains SFT for 1 or 2 epochs.
3. Runs puzzle pass@1 eval after each epoch.
4. Runs full-game eval vs Stockfish on final checkpoint.
5. Uploads final model artifacts to Hugging Face.

This runbook is specific to the baseline prompt dataset and the current GH200 launch stack.

## 1) Source-of-Truth Code Paths

Pipeline orchestration:
- Blocking wrapper: `submit_sft_rejection_pipeline_gh200.bash`

Dataset build:
- Slurm launcher: `sbatch_build_chess_sft_rejection_dataset_gh200.slurm`
- Builder: `scripts/build_chess_sft_rejection_dataset.py`

Train + puzzle eval + full-game eval + upload:
- Slurm launcher: `sbatch_sft_rejection_gh200.slurm`
- Runner: `scripts/run_chess_sft_rejection_one.sh`
- Puzzle eval implementation: `scripts/eval_chess_passk.py`
- Full-game eval implementation: `scripts/eval_chess_fullgame.py`

Full-game-only rerun from checkpoint (new infra):
- `sbatch_eval_chess_sft_fullgame_gh200.slurm`

Full-game ACPL strictness logic:
- `recipe/chess/full_game_eval.py`

## 2) Fixed Contracts

### 2.1 Input datasets

Build inputs (required):
- `data/chess_puzzles_chessr1_aligned_sharded_baseline/train_0.parquet`
- `data/chess_puzzles_chessr1_aligned_sharded_baseline/train_1.parquet`

Puzzle eval set (fixed):
- `data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet`

Optional shuffled eval set:
- `data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet`

### 2.2 Rejection sampling settings (build)

Required and currently enforced:
- `samples_per_prompt=8`
- `temperature=0.6`
- `top_p=0.9`
- `max_response_length=2000` (except explicit model-specific overrides)
- No explicit stop token

Throughput defaults now used in this pipeline:
- `batch_size=1024` (`128 * 8`)
- `tensor_parallel_size=4`
- `max_num_seqs=1024`
- `max_num_batched_tokens=65536`
- `gpu_memory_utilization=0.9`

Implementation note:
- Builder sampling params are intentionally minimal (`temperature`, `top_p`, `max_tokens`, and `n`; optional seed only if explicitly configured). See `scripts/build_chess_sft_rejection_dataset.py`.

### 2.3 Output dataset naming

SFT dataset outputs are model-correlated:
- `rejection_sft_dataset/train_sft__<model_tag>.parquet`

Examples:
- `train_sft__qwen2_5_3b_instruct.parquet`
- `train_sft__qwen2_5_7b_instruct.parquet`
- `train_sft__qwen3_4b_instruct_2507.parquet`

### 2.4 Stage topology

- Full mode build launcher defaults to 2 nodes (`sbatch_build_chess_sft_rejection_dataset_gh200.slurm`).
- Smoke mode is forced to exactly 1 node by wrapper (`submit_sft_rejection_pipeline_gh200.bash`).
- Train/eval launcher is 1 node, 4 GPUs (`sbatch_sft_rejection_gh200.slurm`).

### 2.5 Full-game ACPL is now strict (no fallback)

Current behavior in `recipe/chess/full_game_eval.py`:
- ACPL workers are retried up to 3 times (`acpl_worker_retries=3`).
- If worker failures remain after retries, eval hard-fails.
- If any game ACPL output is missing, eval hard-fails.
- No serial fallback, no synthetic cp-cap fallback path.

## 3) Infrastructure Updates Added This Round

### 3.1 New launcher script

Added:
- `sbatch_eval_chess_sft_fullgame_gh200.slurm`

Purpose:
- Run full-game eval from an existing SFT checkpoint without re-running training.
- Useful when train succeeded through checkpoint + puzzle eval but full-game eval failed.

Defaults aligned with current throughput policy:
- `TP=4`, `gpu_memory_utilization=0.9`
- `max_num_seqs=1024`, `max_num_batched_tokens=65536`
- `acpl_workers=48`, `acpl_threads=2`
- `stockfish_path=/usr/local/bin/stockfish`

### 3.2 ACPL strictness hardening

Updated:
- `recipe/chess/full_game_eval.py`

Key behavior changes:
- Retry failed worker shards before failing.
- Hard fail if workers still fail after retries.
- Hard fail if ACPL outputs do not cover all scheduled games.
- Removed fallback behavior for worker/game-level ACPL failures.

## 4) Cluster Prerequisites

Run these on Isambard login node:

```bash
ssh a5l.aip2.isambard '
  set -euo pipefail
  cd ~/code/chess-rl
  git pull --ff-only
  test -f "$HOME/.huggingface_token_zzc"
  test -f "$HOME/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif"
'
```

Optional dataset row-count check:

```bash
ssh a5l.aip2.isambard '
  cd ~/code/chess-rl
  apptainer exec --nv "$HOME/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif" \
    python3 - <<PY
import pyarrow.parquet as pq
for p in [
  "data/chess_puzzles_chessr1_aligned_sharded_baseline/train_0.parquet",
  "data/chess_puzzles_chessr1_aligned_sharded_baseline/train_1.parquet",
  "data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet",
]:
  print(p, int(pq.ParquetFile(p).metadata.num_rows))
PY
'
```

## 5) Repro Commands

### 5.1 Smoke run (1k prompts, blocking)

```bash
ssh a5l.aip2.isambard '
  set -euo pipefail
  cd ~/code/chess-rl
  git pull --ff-only
  CHESS_SFT_MODE=smoke \
  CHESS_SFT_RUN_ID=sft_reject_smoke_$(date +%Y%m%d_%H%M%S) \
  LIMIT_ROWS=1000 \
  bash submit_sft_rejection_pipeline_gh200.bash
'
```

Notes:
- Smoke mode build is forced to `--nodes=1`.
- If `LIMIT_ROWS` is unset in smoke mode, wrapper computes exact 1/10 prompts.

### 5.2 Full 3-model matrix (blocking wrapper)

```bash
ssh a5l.aip2.isambard '
  set -euo pipefail
  cd ~/code/chess-rl
  git pull --ff-only
  CHESS_SFT_MODE=full \
  CHESS_SFT_RUN_ID=sft_reject_full_$(date +%Y%m%d_%H%M%S) \
  bash submit_sft_rejection_pipeline_gh200.bash
'
```

Default model list in full mode:
1. `Qwen/Qwen2.5-3B-Instruct`
2. `Qwen/Qwen2.5-7B-Instruct`
3. `Qwen/Qwen3-4B-Instruct-2507`

### 5.3 Full matrix (non-blocking submission)

Use this when you want jobs detached from your shell:

```bash
cat <<'EOF' | ssh a5l.aip2.isambard 'bash -s'
set -euo pipefail
cd ~/code/chess-rl
git pull --ff-only

TS="$(date +%Y%m%d_%H%M%S)"
declare -a tags=(
  "qwen2_5_3b_instruct"
  "qwen2_5_7b_instruct"
  "qwen3_4b_instruct_2507"
)
declare -A models
models["qwen2_5_3b_instruct"]="Qwen/Qwen2.5-3B-Instruct"
models["qwen2_5_7b_instruct"]="Qwen/Qwen2.5-7B-Instruct"
models["qwen3_4b_instruct_2507"]="Qwen/Qwen3-4B-Instruct-2507"

for tag in "${tags[@]}"; do
  model="${models[$tag]}"
  run_id="sft_reject_full_${TS}_${tag}"
  run_name="rejection_sft_${tag}"

  build_submit="$(sbatch \
    --export=ALL,CHESS_SFT_RUN_ID=${run_id},CHESS_SFT_MODE=full,MODEL_PATH=${model},REBUILD_DATASET=1,BATCH_SIZE=1024,MAX_NUM_SEQS=1024,MAX_NUM_BATCHED_TOKENS=65536,GPU_MEMORY_UTILIZATION=0.9,TENSOR_PARALLEL_SIZE=4 \
    sbatch_build_chess_sft_rejection_dataset_gh200.slurm)"
  build_job="$(echo "${build_submit}" | awk '{print $4}')"

  sbatch \
    --dependency=afterok:${build_job} \
    --export=ALL,CHESS_SFT_RUN_ID=${run_id},CHESS_SFT_MODE=full,MODEL_PATH=${model},EPOCHS=2,RUN_NAME=${run_name},RUN_SHUFFLED_EVAL=0,RUN_FULLGAME_EVAL=1,EVAL_BATCH_SIZE=1024,EVAL_MAX_NUM_SEQS=1024,EVAL_MAX_NUM_BATCHED_TOKENS=65536,EVAL_GPU_MEMORY_UTILIZATION=0.9,EVAL_TENSOR_PARALLEL_SIZE=4,FULLGAME_TP=4,FULLGAME_MAX_NUM_SEQS=1024,FULLGAME_MAX_NUM_BATCHED_TOKENS=65536,FULLGAME_MAX_RETRIES_PER_TURN=1,FULLGAME_GPU_MEMORY_UTILIZATION=0.9,HF_UPLOAD_ENABLE=1,HF_UPLOAD_REQUIRED=1,HF_UPLOAD_REPO_ID=Gabr1e11/a_lot_of_models,HF_UPLOAD_TOKEN_PATH=${HOME}/.huggingface_token_zzc \
    sbatch_sft_rejection_gh200.slurm

done
EOF
```

## 6) Full-Game-Only Rerun From Existing Checkpoint

New launcher:
- `sbatch_eval_chess_sft_fullgame_gh200.slurm`

Example (blocking):

```bash
ssh a5l.aip2.isambard '
  cd ~/code/chess-rl
  sbatch --wait \
    --export=ALL,CHESS_SFT_RUN_ID=sft_reject_req_full_20260206_202325_qwen2_5_7b_instruct,RUN_NAME=rejection_sft_qwen2_5_7b_instruct,CKPT_STEP=4448,OVERWRITE_FULLGAME_DIR=1 \
    ./sbatch_eval_chess_sft_fullgame_gh200.slurm
'
```

Example (non-blocking):

```bash
ssh a5l.aip2.isambard '
  cd ~/code/chess-rl
  sbatch \
    --export=ALL,CHESS_SFT_RUN_ID=sft_reject_req_full_20260206_202325_qwen2_5_7b_instruct,RUN_NAME=rejection_sft_qwen2_5_7b_instruct,CKPT_STEP=4448,OVERWRITE_FULLGAME_DIR=1 \
    ./sbatch_eval_chess_sft_fullgame_gh200.slurm
'
```

## 7) Verification Commands (Audit Trail)

### 7.1 Verify build settings and acceptance stats

```bash
cat <<'EOF' | ssh a5l.aip2.isambard 'python3 -'
import json
from pathlib import Path

run_id = "sft_reject_req_full_20260206_202325_qwen2_5_7b_instruct"
stats = Path(f"/projects/a5l/ziyan/chess_rl_outputs/{run_id}/rejection_sft_dataset/train_sft__qwen2_5_7b_instruct.parquet.stats.json")
obj = json.loads(stats.read_text())
print("out_rows", obj.get("out_rows"))
print("aggregated", obj.get("aggregated", {}))
EOF
```

### 7.2 Inspect accepted dataset rows manually

```bash
cat <<'EOF' | ssh a5l.aip2.isambard 'bash -s'
set -euo pipefail
cd ~/code/chess-rl
RUN_ID="sft_reject_req_full_20260206_202325_qwen2_5_3b_instruct"
PARQ="/projects/a5l/ziyan/chess_rl_outputs/${RUN_ID}/rejection_sft_dataset/train_sft__qwen2_5_3b_instruct.parquet"

apptainer exec --nv "$HOME/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif" \
  python3 - <<PY
import pyarrow.parquet as pq
p = "${PARQ}"
t = pq.read_table(p, columns=["source_parquet","row_id","fen","ground_truth","pred_uci","messages"], use_threads=True)
rows = t.slice(0, 3).to_pylist()
for i, r in enumerate(rows, 1):
    print("--- row", i, "---")
    print("source", r.get("source_parquet"))
    print("row_id", r.get("row_id"), "fen", r.get("fen"))
    print("gt", r.get("ground_truth"), "pred", r.get("pred_uci"))
    msg = r.get("messages") or []
    if msg:
        print("assistant_tail", (msg[-1].get("content") or "")[:220])
PY
EOF
```

### 7.3 Verify per-epoch puzzle eval artifacts

```bash
ssh a5l.aip2.isambard '
  python3 - <<PY
import json
from pathlib import Path
p = Path("/projects/a5l/ziyan/chess_rl_outputs/sft_reject_req_full_20260206_202325_qwen2_5_7b_instruct/sft_train/rejection_sft_qwen2_5_7b_instruct")
for ep in [1,2]:
  j = p / f"acc_test_epoch{ep}_k1.json"
  s = json.loads(j.read_text())["summary"]
  print(ep, s["pass_at_1_mean"], s["k1_acc_mean"], s["response_len_mean"])
PY
'
```

### 7.4 Verify strict ACPL completion in full-game logs

```bash
ssh a5l.aip2.isambard '
  cd ~/code/chess-rl
  grep -n "\[fullgame\]\[acpl\] merged games" slurm/slurm-chess-sft-fullgame-eval-2208078.out
  grep -n "\[fullgame\]\[acpl\] merged games" slurm/slurm-chess-sft-fullgame-eval-2208125.out
'
```

## 8) Latest Confirmed Results

All rows below are grounded in:
- run artifact files under `/projects/a5l/ziyan/chess_rl_outputs/...`
- `sacct` job states and elapsed times
- explicit full-game rerun logs for strict ACPL post-hardening

### 8.1 Job status summary

| Scope | Run ID | Build Job | Build State/Elapsed | Train Job | Train State/Elapsed | Full-game rerun job |
|---|---|---:|---|---:|---|---:|
| Smoke 3B (1k) | `sft_reject_req_smoke_20260206_gpu09_upload1k` | 2187206 | COMPLETED / 00:08:07 | 2187218 | COMPLETED / 00:20:23 | n/a |
| Full 3B | `sft_reject_req_full_20260206_202325_qwen2_5_3b_instruct` | 2187643 | COMPLETED / 06:27:23 | 2187644 | COMPLETED / 03:28:02 | 2208125 (COMPLETED / 00:04:11) |
| Full 7B | `sft_reject_req_full_20260206_202325_qwen2_5_7b_instruct` | 2187645 | COMPLETED / 05:28:30 | 2187646 | FAILED / 07:03:59 | 2208078 (COMPLETED / 00:07:59) |
| Full Qwen3-4B | `sft_reject_req_full_20260206_202325_qwen3_4b_instruct_2507` | 2187647 | COMPLETED / 14:30:17 | 2187648 | COMPLETED / 00:57:13 | n/a |

### 8.2 Dataset acceptance and puzzle pass@1

| Model/Run | Accepted rows | Accept/sample | Accept/prompt | Pass@1 epoch 1 | Pass@1 epoch 2 |
|---|---:|---:|---:|---:|---:|
| Smoke 3B (`sft_reject_req_smoke_20260206_gpu09_upload1k`) | 480 | 0.060000 | 0.267000 | 0.056900 | n/a |
| Full 3B (`...qwen2_5_3b_instruct`) | 45,806 | 0.0572575 | 0.24948 | 0.066500 | 0.070200 |
| Full 7B (`...qwen2_5_7b_instruct`) | 71,181 | 0.08897625 | 0.27816 | 0.090900 | 0.093400 |
| Full Qwen3-4B (`...qwen3_4b_instruct_2507`) | 1,453 | 0.00181625 | 0.01219 | 0.006400 | 0.004900 |

### 8.3 Full-game metrics

| Model/Run | Full-game source | Games | ACPL mean | ACPL mean per move | Strict ACPL path |
|---|---|---:|---:|---:|---|
| Smoke 3B (`sft_reject_req_smoke_20260206_gpu09_upload1k`) | Train job output | 2 | 310.8750 | 294.5455 | Legacy (pre-hardening) |
| Full 3B (`...qwen2_5_3b_instruct`) | Rerun job 2208125 | 100 | 559.0850 | 478.0811 | Yes |
| Full 7B (`...qwen2_5_7b_instruct`) | Rerun job 2208078 | 100 | 526.4406 | 497.3587 | Yes |
| Full Qwen3-4B (`...qwen3_4b_instruct_2507`) | Train job output | 100 | 891.4600 | 788.0923 | Legacy (pre-hardening) |

### 8.4 Upload status

| Run | `hf_upload_summary.json` |
|---|---|
| Smoke 3B | present, success=true |
| Full 3B | present, success=true |
| Full 7B | missing (train job failed before results/upload stage) |
| Full Qwen3-4B | present, success=true |

## 9) Known Issues and Current Guidance

### 9.1 Qwen3 acceptance collapse under short response cap

Observed in run `sft_reject_req_full_20260206_202325_qwen3_4b_instruct_2507`:
- `max_response_length=2000`
- `max_model_len=4096`
- shard stats show `response_tokens_mean_per_sample ~= 1988.35`
- acceptance dropped to `0.00181625` per sample

Guidance for future Qwen3 runs:
- Use at least `max_response_length=8192`.
- Increase `max_model_len` accordingly (prompt + response budget), for example `12288`.
- Apply this consistently to build, puzzle eval, and full-game eval.

### 9.2 7B full-game failure root cause and fix status

Original train job `2187646` failed in full-game ACPL with worker exit `-11`.
Fixes now in place:
- strict ACPL worker retries + hard fail behavior in `recipe/chess/full_game_eval.py`
- dedicated rerun launcher `sbatch_eval_chess_sft_fullgame_gh200.slurm`

Confirmed rerun success:
- job `2208078` completed with full coverage (`merged games=100/100`).

### 9.3 Full-game-only reruns do not rewrite `results.json`

When you rerun full-game separately with `sbatch_eval_chess_sft_fullgame_gh200.slurm`,
`fullgame_eval/summary.json` is refreshed, but any prior `results.json` from the original train job is not automatically rebuilt.

For reporting, use `fullgame_eval/summary.json` as source of truth after a rerun.

## 10) How To Run More Experiments Safely

### 10.1 One-model full run template

```bash
ssh a5l.aip2.isambard '
  set -euo pipefail
  cd ~/code/chess-rl
  git pull --ff-only
  CHESS_SFT_MODE=full \
  CHESS_SFT_MODELS="Qwen/Qwen2.5-7B-Instruct" \
  CHESS_SFT_RUN_ID=sft_reject_full_7b_$(date +%Y%m%d_%H%M%S) \
  bash submit_sft_rejection_pipeline_gh200.bash
'
```

### 10.2 Qwen3 high-length template (recommended)

```bash
ssh a5l.aip2.isambard '
  set -euo pipefail
  cd ~/code/chess-rl
  git pull --ff-only
  CHESS_SFT_MODE=full \
  CHESS_SFT_MODELS="Qwen/Qwen3-4B-Instruct-2507" \
  CHESS_SFT_RUN_ID=sft_reject_full_qwen3_len8192_$(date +%Y%m%d_%H%M%S) \
  MAX_RESPONSE_LENGTH=8192 \
  MAX_MODEL_LEN=12288 \
  EVAL_MAX_RESPONSE_LENGTH=8192 \
  EVAL_MAX_MODEL_LEN=12288 \
  FULLGAME_MAX_RESPONSE_TOKENS=8192 \
  FULLGAME_MAX_MODEL_LEN=12288 \
  bash submit_sft_rejection_pipeline_gh200.bash
'
```

If memory pressure appears, reduce `BATCH_SIZE` first while keeping `max_num_seqs` and `max_num_batched_tokens` as high as feasible.

## 11) Monitoring Commands

Queue:

```bash
ssh a5l.aip2.isambard 'squeue -u $USER -o "%i %t %M %R %j"'
```

Accounting:

```bash
ssh a5l.aip2.isambard 'sacct -j <jobid1>,<jobid2> --format=JobID,JobName%35,State,Elapsed,Start,End,ExitCode -P'
```

Tail logs:

```bash
ssh a5l.aip2.isambard 'cd ~/code/chess-rl && tail -f slurm/slurm-chess-sft-reject-build-<jobid>.out'
ssh a5l.aip2.isambard 'cd ~/code/chess-rl && tail -f slurm/slurm-chess-sft-reject-train-<jobid>.out'
ssh a5l.aip2.isambard 'cd ~/code/chess-rl && tail -f slurm/slurm-chess-sft-fullgame-eval-<jobid>.out'
```

Cancel:

```bash
ssh a5l.aip2.isambard 'scancel <jobid>'
```
