# chess_self_play

Research repo for chess-RL with LLM agents — two parallel training methods.

| Method | Approach | Subdir | Status |
|---|---|---|---|
| **Method A** | vanilla **verl** + Chess-R1 puzzle data + **On-Policy Distillation (OPD)** | [`verl-vam-chess/`](verl-vam-chess/) | ✅ trained, full report |
| **Method B** | **verl-agent** + new chesslesson dataset (multi-turn, VAM-aware) | [`verl-agent-vam-agent/`](verl-agent-vam-agent/) | 🟡 baselines done, training in progress |

---

## Two methods

### 1. verl + Chess-R1 data + OPD distillation (Method A, `verl-vam-chess/`)

```
Lichess puzzle DB ─→ Chess-R1 paper preprocess ─→ Stockfish d=14 μ-grading
                                                          ↓
                                                ~100k single-step puzzles
                                                          ↓
                                            Pass@k GRPO (n=16, k=4) on Qwen2.5-7B
                                                          ↓
                                                   Teacher 7B
                                                          ↓
                                      On-Policy Distillation (per-token reverse-KL
                                      as advantage, teacher reused in ref_policy slot)
                                                          ↓
                                                   Student 3B
```

Single-step puzzle: model sees FEN + legal_moves, picks one UCI; reward = μ from Stockfish table.

**Result**: 3B student reaches puzzle pass@8 = **0.476** vs paper's 3B baseline 0.425, with **~16× fewer samples**.

### 2. verl-agent + chesslesson dataset (Method B, `verl-agent-vam-agent/`)

```
chesslesson (170 task)         chess/WhiteVsRandom           lichess_puzzle (Chess-R1)
   ↓                                ↓                                ↓
ChessLessonWorker               ChessWorker                  LichessPuzzleWorker
(multi-turn online,             (multi-turn full game         (single-turn puzzle,
 LessonStepper one              vs random Black)              μ-table reward)
 move per turn)                       ↓                                ↓
   ↓                            with optional VAM                  with optional
   ↓                            (random / stockfish               VAM mu_topk
   ↓                             top-k subset)                    (paper-style)
   ↓                                  ↓                                ↓
                          ┌────────────────────────────────┐
                          │  HGPO (verl-agent's            │
                          │  hierarchical-group GRPO)      │
                          │  multi-turn rollout +          │
                          │  step-level advantage          │
                          └────────────────────────────────┘
```

Three environments, all gym-style `reset/step`, all VAM-aware (chess-rl-C224 paper
verbalized action masking: optionally expose a subset of `k` allowed moves in the
prompt, penalise out-of-subset picks).

**Multi-turn data flow**: each `env.step(action)` advances the lesson by one ply
(LessonStepper applies model's UCI + any scripted opponent reply), then the
worker re-renders the board and asks for the next move. Reward is binary 0/1 on
the terminal turn. The whole chat history accumulates as the rollout context.

**Baseline (no training)**: Qwen2.5-7B-Instruct zero-shot ≈ **1.5% multi-move acc**
on chesslesson lessons. RL training is what moves this number up.

---

## Layout

```
chess_self_play/
├── README.md              ← this file
├── LICENSE / NOTICE       ← Apache 2.0
│
├── scripts/               ← Method A sbatch launchers
│   ├── sbatch_train_teacher_7b.slurm
│   ├── sbatch_train_distill_3b.slurm
│   ├── sbatch_eval_passk.slurm
│   ├── sbatch_eval_fullgame.slurm
│   └── sbatch_eval_h2h.slurm
│
├── results/               ← Method A eval artifacts (pass@k JSONs, full-game PGNs, h2h)
├── progress_report/       ← Method A full writeup (md + html with SVG charts)
│
├── verl-vam-chess/        ← Method A framework + recipe (vanilla verl + chess recipe)
│   ├── verl/              upstream verl
│   ├── recipe/chess/      chess RL reward fn + prompt templates
│   ├── recipe/chess_distill/  on-policy distillation recipe
│   ├── scripts/           data prep, eval scripts
│   └── train_chess.sh     main training entry
│
└── verl-agent-vam-agent/  ← Method B framework + envs + VAM
    ├── verl/              upstream verl-agent
    ├── agent_system/      EnvironmentManager + prompts
    ├── chess_game/
    │   ├── ray_envs.py                    chess/WhiteVsRandom (with VAM)
    │   ├── chesslesson_envs.py            chesslesson/LichessLearn (multi-turn)
    │   ├── lichess_puzzle_envs.py         lichess_puzzle/Curriculum (Chess-R1 + VAM)
    │   ├── prompts_shared.py              shared prompt builders
    │   └── chesslesson/                   115 lesson + 60 coord task data
    ├── recipe/hgpo/                       HGPO algorithm + condor launchers
    ├── tests/                             env-level smoke tests
    ├── eval_chesslesson_base_multiturn.py  paradigm-aligned base eval
    └── sbatch_*.slurm                     ← train + eval launchers (see below)
```

---

## Setup

```bash
# 1. Conda env (one env serves both methods)
conda create -n chess python=3.10 -y
conda activate chess

# 2. Method A deps
cd verl-vam-chess
pip install -r requirements.txt
pip install -r requirements-cuda.txt    # flash-attn
pip install vllm==0.10.0
pip install -e .                         # install verl in editable mode

# 3. Method B deps (adds verl-agent + env-specific deps)
cd ../verl-agent-vam-agent
pip install -r requirements.txt
pip install -e .                         # install verl-agent
pip install chess>=1.10.0                # for python-chess in env

# 4. Stockfish 16 (need bmi2 build, not the apt one)
git clone https://github.com/official-stockfish/Stockfish.git /tmp/Stockfish
cd /tmp/Stockfish/src && make -j build ARCH=x86-64-bmi2
export STOCKFISH_BIN=/tmp/Stockfish/src/stockfish

# 5. Base models
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir $HOME/models/Qwen2.5-3B-Instruct

# 6. WandB
export WANDB_API_KEY=...

# 7. Method A puzzle data (Chess-R1 aligned + SF μ-grading, ~5 GB, hours to build)
cd verl-vam-chess && python scripts/build_chessr1_aligned_dataset.py \
    --out-dir data/chess_puzzles_chessr1_aligned_sharded_baseline

# Method B data is bundled (chess_game/chesslesson/instructions.jsonl).
```

A Dockerfile is provided at `verl-vam-chess/Dockerfile`.

---

## Training scripts

### Method A — Teacher + Distillation (verl-vam-chess)

```bash
# A. Train 7B teacher (~640 steps on 4× H100, Pass@k GRPO)
sbatch --export=ALL,WANDB_API_KEY=$WANDB_API_KEY \
    scripts/sbatch_train_teacher_7b.slurm
# Smoke: append SMOKE=1 to --export.

# B. Merge FSDP → HF (required for distill)
python -m verl.model_merger merge --backend fsdp \
    --local_dir $HOME/chess_rl_outputs/teacher_7b_passk_<ts>/actor/global_step_640 \
    --target_dir $HOME/models/chess_teacher_7b_step640

# C. Train 3B student via on-policy distillation (~300 steps)
sbatch --export=ALL,\
TEACHER_CKPT=$HOME/models/chess_teacher_7b_step640,\
WANDB_API_KEY=$WANDB_API_KEY \
    scripts/sbatch_train_distill_3b.slurm

# D. Eval (pass@k / full-game vs Stockfish / model-vs-model h2h)
sbatch --export=ALL,MODEL=<hf_dir>,OUT_PATH=results/x_passk.json \
    scripts/sbatch_eval_passk.slurm
sbatch --export=ALL,MODEL=<hf_dir>,OUT_DIR=results/x_fullgame,\
OPPONENT_DEPTHS=5,STOCKFISH_BIN=$STOCKFISH_BIN \
    scripts/sbatch_eval_fullgame.slurm
sbatch --export=ALL,MODEL_A=<a>,MODEL_B=<b>,OUT_DIR=results/x_h2h \
    scripts/sbatch_eval_h2h.slurm
```

### Method B — verl-agent training (verl-agent-vam-agent)

All Method B sbatch must be submitted **from** `verl-agent-vam-agent/`
so `$SLURM_SUBMIT_DIR` resolves correctly.

```bash
cd verl-agent-vam-agent

# A. chesslesson HGPO training (multi-turn online puzzles)
#    Data: 110 lesson + 60 coord (chess_game/chesslesson/instructions.jsonl)
sbatch --export=ALL,WANDB_API_KEY=$WANDB_API_KEY \
    sbatch_chesslesson_train_smoke.slurm
# Knobs: TRAIN_DATA_SIZE, GROUP_SIZE, TOTAL_EPOCHS, HISTORY_LENGTH

# B. chess/WhiteVsRandom HGPO training (full game vs random Black, optional VAM)
sbatch --export=ALL,WANDB_API_KEY=$WANDB_API_KEY \
    sbatch_vam_chess_train_smoke.slurm
# Knobs: VAM_K, VAM_ITERATIVE, VAM_SOURCE (random|stockfish), STOCKFISH_BIN

# C. lichess_puzzle/Curriculum training (Chess-R1 puzzle data with μ-top-k VAM)
sbatch --export=ALL,WANDB_API_KEY=$WANDB_API_KEY \
    sbatch_vam_lichess_puzzle_smoke.slurm
# Knobs: VAM_K, VAM_ITERATIVE, VAM_SOURCE (mu_topk|random), PARQUETS

# D. Zero-shot baselines (no training)
sbatch sbatch_eval_chesslesson_base.slurm           # single-turn JSON paradigm
sbatch sbatch_eval_chesslesson_base_multiturn.slurm # multi-turn paradigm (matches HGPO training)
```

### VAM config block (Method B)

Either env can run with VAM enabled by passing config overrides:

```yaml
env.chess.vam.enable=True env.chess.vam.k=8 env.chess.vam.subset_source=random
env.chess.vam.iterative=False env.chess.vam.penalty=-1.0

env.lichess_puzzle.vam.enable=True env.lichess_puzzle.vam.k=8
env.lichess_puzzle.vam.subset_source=mu_topk    # uses precomputed Stockfish μ
```

`subset_source` options:
- `random` — uniform k-subset of legal moves
- `stockfish` — top-k by Stockfish MultiPV at `stockfish_depth` (chess env only)
- `mu_topk` — top-k by precomputed μ table (lichess_puzzle env only; paper-style)

### Env-level smoke (CPU only, no GPU)

```bash
cd verl-agent-vam-agent
python tests/test_vam_chess_env.py            # chess env, 6 unit tests
python tests/test_vam_lichess_puzzle_env.py   # lichess_puzzle env, 7 unit tests
```

---

## Results

### Method A puzzle pass@k (Chess-R1 held-out test set, 10k positions)

| Model | Pass@1 | Pass@8 | Train samples |
|---|---|---|---|
| Base Qwen2.5-3B-Instruct | 0.019 | 0.102 | — |
| Paper RL 3B (800 step) | ~0.20 | 0.425 | 1.64M |
| **OPD-distilled 3B (300 step)** | **0.213** | **0.476** | **307k** |
| Teacher 7B (640 step) | 0.220 | 0.514 | 1.31M |

Full report + charts: [`progress_report/chess_distill_summary.md`](progress_report/chess_distill_summary.md).

### Method B chesslesson baseline (no training)

Qwen2.5-7B-Instruct zero-shot on 175 tasks (single-turn paradigm):

```
overall acc      0.126   parse_ok_rate    0.971
single-move acc  0.083   multi-move acc   0.015
coord drill acc  ~0.283

By move budget (lessons):
  1 move:  8.3%   2 move: 0.0%   3 move: 20.0%   4-9: 0%
```

A paradigm-aligned multi-turn baseline (the same prompts the env uses during
HGPO rollout) is being measured (`sbatch_eval_chesslesson_base_multiturn.slurm`)
and will be the proper anchor for the post-training comparison.

---

## Caveats

- Method A pass@k is in-distribution (train + test from Chess-R1). Both
  trained models lose 0/50 to Stockfish at depth 1 in full-game eval —
  puzzle skill ≠ chess strength. This is part of why Method B exists.
- Method A distill-vs-teacher h2h: distill "wins" 56/80 of completed games,
  but ~49 wins came from teacher format failures (truncation mid-`<think>`).
  Pure-chess strength ratio is closer to 37/63.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Both `verl-vam-chess/` and `verl-agent-vam-agent/` are vendored from
upstream Apache-2.0 projects; original copyrights are preserved in their
respective `LICENSE` / `Notice.txt` files.
