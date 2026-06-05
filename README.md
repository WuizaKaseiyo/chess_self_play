# chess_self_play

Two-phase research repo for chess-RL with LLM agents.

| Phase | Subdir | Framework | Tasks |
|---|---|---|---|
| **Phase 1** (complete) | [`verl-vam-chess/`](verl-vam-chess/) | vanilla verl | Single-step puzzle RL — 7B teacher + 3B distillation |
| **Phase 2** (in progress) | [`verl-agent-vam-agent/`](verl-agent-vam-agent/) | verl-agent | Multi-turn chess agents (full game / chesslesson / puzzle env) — with **Verbalized Action Masking (VAM)** |

**Phase 1 headline**: 3B student distilled from 7B teacher reaches puzzle pass@8 = **0.476** vs paper's 0.425, with **~16× fewer training samples**.

**Phase 2 headline (so far)**: chesslesson base zero-shot ≈ **1.5% multi-move acc** (Qwen2.5-7B-Instruct). RL training is the next step.

---

## Layout

```
chess_self_play/
├── README.md            ← this file
├── LICENSE / NOTICE     ← Apache 2.0
├── scripts/             ← Phase 1 sbatch launchers (verl-vam-chess style)
├── results/             ← Phase 1 distill eval artifacts (pass@k, full-game PGNs, h2h)
├── progress_report/     ← Phase 1 written report (md + html with charts)
├── verl-vam-chess/           ← Phase 1 vendored upstream (vanilla verl + chess recipe)
└── verl-agent-vam-agent/           ← Phase 2 vendored upstream (verl-agent + chess envs + VAM)
```

---

# Phase 1 — Pass@k GRPO teacher + distillation (verl-vam-chess)

## Environment setup

```bash
conda create -n chess python=3.10 -y
conda activate chess

cd verl-vam-chess
pip install -r requirements.txt
pip install -r requirements-cuda.txt
pip install vllm==0.10.0
pip install -e .

# Stockfish 16 (bmi2 build)
git clone https://github.com/official-stockfish/Stockfish.git /tmp/Stockfish
cd /tmp/Stockfish/src && make -j build ARCH=x86-64-bmi2
export STOCKFISH_BIN=/tmp/Stockfish/src/stockfish

# Base models
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir $HOME/models/Qwen2.5-3B-Instruct

export WANDB_API_KEY=...

# Puzzle data (Chess-R1 aligned + SF μ-grading, ~5 GB)
cd verl-vam-chess && python scripts/build_chessr1_aligned_dataset.py \
    --out-dir data/chess_puzzles_chessr1_aligned_sharded_baseline
```

## Train teacher (7B, Pass@k GRPO, ~640 steps on 4× H100)

```bash
sbatch --export=ALL,WANDB_API_KEY=$WANDB_API_KEY scripts/sbatch_train_teacher_7b.slurm

# Merge FSDP → HF
python -m verl.model_merger merge --backend fsdp \
    --local_dir $HOME/chess_rl_outputs/teacher_7b_passk_<ts>/actor/global_step_640 \
    --target_dir $HOME/models/chess_teacher_7b_step640
```

Result: pass@1=0.220, pass@8=0.514.

## Train distill student (3B from 7B, ~300 steps)

Per-token reverse-KL as advantage; teacher reuses the `ref_policy` worker slot.

```bash
sbatch --export=ALL,\
TEACHER_CKPT=$HOME/models/chess_teacher_7b_step640,\
WANDB_API_KEY=$WANDB_API_KEY \
    scripts/sbatch_train_distill_3b.slurm
```

Result: pass@1=0.213, pass@8=0.476 (92.6% of teacher with ~16× fewer samples).

## Eval

```bash
# Puzzle pass@k
sbatch --export=ALL,MODEL=<hf_dir>,OUT_PATH=results/x_passk.json scripts/sbatch_eval_passk.slurm

# Full game vs Stockfish
sbatch --export=ALL,MODEL=<hf_dir>,OUT_DIR=results/x_fullgame,OPPONENT_DEPTHS=5,STOCKFISH_BIN=... \
    scripts/sbatch_eval_fullgame.slurm

# Head-to-head model-vs-model
sbatch --export=ALL,MODEL_A=<a>,MODEL_B=<b>,OUT_DIR=results/x_h2h scripts/sbatch_eval_h2h.slurm
```

## Phase 1 results summary

| Model | Pass@1 | Pass@8 | Train samples |
|---|---|---|---|
| Base 3B | 0.019 | 0.102 | — |
| Paper RL 3B (800 step) | ~0.20 | 0.425 | 1.64M |
| **Distill 3B (300 step)** | **0.213** | **0.476** | **307k** |
| Teacher 7B (640 step) | 0.220 | 0.514 | 1.31M |

Full discussion + charts: [`progress_report/chess_distill_summary.md`](progress_report/chess_distill_summary.md).

Caveats: Pass@k on puzzles ≠ real chess strength. Both models lose 0/50 to Stockfish d=1. Distill vs teacher h2h: distill "wins" 56/80 but ~49 wins were teacher format failures — pure-chess strength ratio ≈ 37/63. This motivates Phase 2.

---

# Phase 2 — Multi-turn chess agents with VAM (verl-agent-vam-agent)

`verl-agent-vam-agent/` is a vendored fork of [verl-agent](https://github.com/langfengQ/verl-agent) (chess-agent variant) extended with **Verbalized Action Masking** (VAM, from chess-rl-C224 EMNLP paper).

## Three environments, all VAM-aware

| Env name | Type | Data | VAM subset source |
|---|---|---|---|
| `chess/WhiteVsRandom` | Multi-turn full game | runtime (start position + random Black) | `random` / `stockfish` (SF top-k via MultiPV) |
| `chesslesson/LichessLearn` | Multi-turn online puzzle | 170 Lichess "Learn chess" tasks (110 lessons + 60 coord) | *not wired yet* |
| `lichess_puzzle/Curriculum` | Single-turn puzzle | chess-rl-C224 schema parquet (SF μ-graded) | `mu_topk` (precomputed μ) / `random` |

Algorithm: **HGPO** (verl-agent's hierarchical-group GRPO variant for long-horizon tasks).

## Quick env smoke (CPU, no GPU)

```bash
cd verl-agent-vam-agent
python tests/test_vam_chess_env.py          # chess env w/ VAM, 6 tests
python tests/test_vam_lichess_puzzle_env.py # lichess_puzzle env w/ VAM, 7 tests
```

## VAM config (env.chess.vam.* / env.lichess_puzzle.vam.*)

```yaml
env:
  chess:
    vam:
      enable: True
      k: 8                       # subset size
      iterative: False           # remove prior picks across plies
      subset_source: random      # random | stockfish
      penalty: -1.0              # reward for out-of-subset pick
      stockfish_path: ""         # if subset_source=stockfish
      stockfish_depth: 1
  lichess_puzzle:
    vam:
      enable: True
      k: 8
      subset_source: mu_topk     # use precomputed μ for top-k (chess-rl-C224 style)
      iterative: False
      penalty: -1.0
```

## Train

```bash
cd verl-agent-vam-agent
export WANDB_API_KEY=...

# Multi-turn full game with VAM
sbatch --export=ALL,WANDB_API_KEY=$WANDB_API_KEY sbatch_vam_chess_train_smoke.slurm

# chesslesson 170 tasks (multi-turn)
sbatch --export=ALL,WANDB_API_KEY=$WANDB_API_KEY sbatch_chesslesson_train_smoke.slurm

# Single-turn puzzle data with VAM (μ top-k)
sbatch --export=ALL,WANDB_API_KEY=$WANDB_API_KEY sbatch_vam_lichess_puzzle_smoke.slurm
```

Override VAM knobs:
```bash
sbatch --export=ALL,WANDB_API_KEY=...,VAM_K=12,VAM_ITERATIVE=True,VAM_SOURCE=stockfish,STOCKFISH_BIN=/path/to/stockfish \
    sbatch_vam_chess_train_smoke.slurm
```

## Phase 2 baseline (no training)

Base Qwen2.5-7B-Instruct zero-shot on chesslesson 170 tasks:

```
overall acc          0.126  (22/175)
parse_ok_rate        0.971
single-move acc      0.083  (4/48 lessons)
multi-move acc       0.015  (1/67 lessons)   ← multi-step solving floor ≈ 0
coord drill acc      0.283  (~17/60)

By move budget:
  1 move:  8.3%
  2 move:  0.0%
  3 move: 20.0%   (1 of 5 — outlier)
  4-9:     0%
```

After RL training: target is to move multi-move acc from 1.5% → 30%+.

---

## Roadmap

- **Track A (verl-vam-chess)**: Lichess-pedagogy curriculum (mate → tactics → endgame), retrain teacher with theme-stratified data.
- **Track B (verl-agent-vam-agent)**: chesslesson + chess/WhiteVsRandom training with VAM, then asymmetric self-play (RAG opponent, dual trainable actors).

---

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Both `verl-vam-chess/` and `verl-agent-vam-agent/` are vendored from upstream Apache-2.0 projects; original copyrights are preserved in their respective `LICENSE` / `Notice.txt` files.
