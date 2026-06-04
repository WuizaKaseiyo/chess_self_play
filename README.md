# chess_self_play

Research repo for asymmetric self-play chess agents. This snapshot captures
**Phase 1**: a 7B chess-RL **teacher** and a 3B **on-policy distillation
student**, plus the evaluation harness used to compare them against
Stockfish and against each other. **Phase 2** (multi-turn self-play) is the
next step and not yet included.

> **Headline result**: a 3B student distilled from the 7B teacher reaches
> **pass@8 = 0.476** on held-out chess puzzles, beating the published 3B
> RL baseline (pass@8 = 0.425) while using **~16× fewer training samples**.
> Full numbers: [`progress_report/chess_distill_summary.md`](progress_report/chess_distill_summary.md).

---

## What's in this repo

```
chess_self_play/
├── README.md                                    ← this file
├── LICENSE                                      ← Apache 2.0
├── NOTICE                                       ← attribution / vendoring note
├── scripts/                                     ← sbatch launchers (sanitized)
│   ├── sbatch_train_teacher_7b.slurm           Stage A — train teacher
│   ├── sbatch_train_distill_3b.slurm           Stage B — on-policy distillation
│   ├── sbatch_eval_passk.slurm                 Stage C — puzzle pass@k eval
│   ├── sbatch_eval_fullgame.slurm              Stage C — full game vs Stockfish
│   └── sbatch_eval_h2h.slurm                   Stage C — head-to-head
├── vam-chess/                                   ← VENDORED training framework (Apache 2.0)
│   ├── recipe/chess/                              reward fn, prompt templates, SF scoring
│   ├── recipe/chess_distill/                      distill launcher
│   ├── verl/                                      RL framework (Pass@k, distill estimators)
│   ├── scripts/                                   eval_chess_passk.py, eval_chess_fullgame.py, ...
│   ├── train_chess.sh                             main training launcher
│   └── ...
├── results/                                     ← eval artifacts from our runs
│   ├── teacher_step640_passk.json              7B teacher final pass@k
│   ├── distill_step{50..300}_passk.json        3B distill learning curve
│   ├── student_baseline_passk.json             3B base pass@k (no training)
│   ├── distill_step300_fullgame/               50 games distill vs SF d5
│   ├── distill_s300_fullgame_d1/               50 games distill vs SF d1
│   ├── base3b_fullgame_d1/                     50 games base 3B vs SF d1
│   └── distill_vs_teacher_h2h/                 Distill vs teacher 100-game match
└── progress_report/
    ├── chess_distill_summary.md                5-section report (motivation/method/results)
    └── chess_distill_report.html               same content with SVG charts
```

---

## What this is NOT

- **Not** Phase 2. The asymmetric multi-turn self-play work (RAG opponent,
  trajectory-level reward, two-actor RL) is the next phase and is not in
  this repo yet — the present snapshot covers Phase 1 (puzzle-graded RL +
  on-policy distillation) only.

---

## Pipeline overview

```
            +--------------------------+
            |  Lichess puzzle dataset  |
            |  (Chess-R1 aligned;      |
            |   Stockfish-graded μ)    |
            +-------------+------------+
                          |
        +-----------------+------------------+
        |                                    |
        v                                    v
+----------------+                +-------------------+
| Stage A —      |                | Stage A reference |
|  Teacher RL    |  ===>          |  ckpt (HF format) |
| 7B + Pass@k    |                +---------+---------+
| (640 steps)    |                          |
+----------------+                          v
                                  +-------------------+
                                  | Stage B —         |
                                  |  Distill 3B from  |
                                  |  7B teacher       |
                                  | (per-token KL =   |
                                  |  advantage)       |
                                  +---------+---------+
                                            |
                                            v
                          +-----------------+-----------------+
                          | Stage C — Evaluation              |
                          |  • Puzzle pass@k (1, 2, 4, 8)      |
                          |  • Full-game vs Stockfish d=1..5   |
                          |  • Head-to-head distill vs teacher |
                          +-----------------------------------+
```

---

## Prerequisites

### 1. Training framework (bundled)

The training framework lives in [`vam-chess/`](vam-chess/)
inside this repo. It is a chess-specific recipe layered on top of an
upstream RL training library, redistributed here under Apache 2.0 (see
[NOTICE](NOTICE) and [`vam-chess/LICENSE`](vam-chess/LICENSE)).
Key pieces:

- **Pass@k GRPO** advantage estimator —
  [`vam-chess/verl/trainer/ppo/core_algos.py`](vam-chess/verl/trainer/ppo/core_algos.py)
  (`passk_advantages_max_subsets`)
- **Distill** advantage estimator — per-token reverse-KL, same file
  (`AdvantageEstimator.DISTILL`)
- **Stockfish-graded reward** — [`vam-chess/recipe/chess/reward_fn.py`](vam-chess/recipe/chess/reward_fn.py)
- **Online play / multi-step rollout hook** —
  [`vam-chess/verl/trainer/ppo/ray_trainer.py`](vam-chess/verl/trainer/ppo/ray_trainer.py)
  (`_build_self_play_train_batch`)

`UPSTREAM_DIR` defaults to `<repo_root>/vam-chess`, so the sbatch
launchers in `scripts/` work out of the box. To point them at a different
upstream tree, pass `--export=ALL,UPSTREAM_DIR=/path/to/your/clone,...` to
`sbatch`.

> Puzzle data is **not** bundled (the parquets are large). The launcher
> expects them at
> `${UPSTREAM_DIR}/data/chess_puzzles_chessr1_aligned_sharded_baseline/`.
> Run the upstream's `scripts/build_chessr1_aligned_dataset.py` (or the
> matching sbatch under `vam-chess/`) to materialise them from the
> public [Lichess puzzle database](https://database.lichess.org/).

### 2. Base models

```bash
# Used as the teacher base
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct

# Used as the distill student base
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir $HOME/models/Qwen2.5-3B-Instruct
```

### 3. Stockfish

Stockfish 16 with bmi2 build, installed at any path. Set
`STOCKFISH_BIN=/path/to/stockfish` when invoking the eval sbatch.

### 4. Conda env

The launchers assume `conda activate chess`. Build it via the upstream's
environment.yaml or pip-install equivalents (vLLM, transformers, ray,
python-chess, etc.).

### 5. WandB

```bash
export WANDB_API_KEY=...   # your own key; never commit it
```

---

## Stage A — Train the 7B teacher

**Goal**: Qwen2.5-7B-Instruct → chess RL teacher with Pass@k GRPO.

**Config snapshot** (encoded in [`scripts/sbatch_train_teacher_7b.slurm`](scripts/sbatch_train_teacher_7b.slurm)):

| Knob | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Advantage estimator | `grpo_passk` |
| Rollout `n` | 16 |
| Pass-k `k` | 4 |
| Train batch | 128 |
| Max prompt | 1536 tokens |
| Max response | 2000 tokens |
| Reward fn | `expected_score_wdl_vs_best` |
| Data | `data/chess_puzzles_chessr1_aligned_sharded_baseline` |
| KL loss | on (small coef) |
| Diversity / Filter-groups / Forced-prefix / Iterative-VAM | **off** |
| Save every | 80 steps |
| Eval every | 40 steps |
| Walltime budget | 5 days (we converged around step 640) |
| GPU | 4× H100 80GB, FSDP |

**Launch**:

```bash
cd /path/to/chess_self_play

sbatch \
    --export=ALL,WANDB_API_KEY=$WANDB_API_KEY \
    scripts/sbatch_train_teacher_7b.slurm
```

**Smoke test** (1 training step, fail-fast):

```bash
sbatch \
    --export=ALL,WANDB_API_KEY=$WANDB_API_KEY,SMOKE=1 \
    scripts/sbatch_train_teacher_7b.slurm
```

**Output**: `$HOME/chess_rl_outputs/teacher_7b_passk_<timestamp>/` containing
FSDP checkpoints under `actor/global_step_*/`.

**Convert FSDP ckpt to HF format** (required for Stage B + eval):

```bash
cd $UPSTREAM_DIR
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $HOME/chess_rl_outputs/teacher_7b_passk_<ts>/actor/global_step_640 \
    --target_dir $HOME/models/chess_teacher_7b_step640
```

Use the resulting `$HOME/models/chess_teacher_7b_step640` as `TEACHER_CKPT`
for Stage B.

**Expected results** (our run, step 640):

| Metric | Value |
|---|---|
| Pass@1 (val, in-distribution) | 0.220 |
| Pass@8 (val, in-distribution) | 0.514 |
| Walltime | ~3.5 days, 1× H100 4× GPU |

See `results/teacher_step640_passk.json`.

---

## Stage B — Distill 3B from 7B (on-policy distillation)

**Goal**: Recover ~93% of the 7B teacher's puzzle skill on a 3B student
using the Thinking Machines on-policy distillation recipe — per-token
reverse-KL between student and teacher used directly as the advantage
signal, no group normalization, no discount.

**Mathematical form** (from `recipe/chess_distill/`):

```
A_t  = log π_teacher(y_t | x, y<t) - log π_student(y_t | x, y<t)
loss = -A_t · log π_student_current(y_t)
```

Engineering trick: re-use the existing `ref_policy` worker slot in the RL
trainer to host the teacher (frozen). The `ref_log_prob` field naturally
carries the teacher's log-prob, so no new worker is written.

**Config snapshot** ([`scripts/sbatch_train_distill_3b.slurm`](scripts/sbatch_train_distill_3b.slurm)):

| Knob | Value |
|---|---|
| Student base | `Qwen/Qwen2.5-3B-Instruct` |
| Teacher | merged HF dir from Stage A |
| Advantage estimator | `distill` (per-token KL) |
| Rollout `n` | 8 |
| Train batch | 128 |
| Total steps | 300 |
| Save every | 50 steps |
| Eval every | 25 steps |
| Walltime budget | ~10 hours, 1× H100 4× GPU |

**Launch**:

```bash
sbatch \
    --export=ALL,\
\
TEACHER_CKPT=$HOME/models/chess_teacher_7b_step640,\
WANDB_API_KEY=$WANDB_API_KEY \
    scripts/sbatch_train_distill_3b.slurm
```

**Expected results** (our run, step 300):

| Metric | Base 3B | Paper RL 3B (800 step) | Distill 3B (this work, 300 step) | Teacher 7B (640 step) |
|---|---|---|---|---|
| Pass@1 | 0.019 | ~0.20 | **0.213** | 0.220 |
| Pass@8 | 0.102 | 0.425 | **0.476** | 0.514 |
| Train samples | — | 1.64M | **307k** (16× fewer) | 1.31M |

Sources:
- Distill curve: `results/distill_step{50,100,150,200,250,300}_passk.json`
- Baseline 3B: `results/student_baseline_passk.json`
- Teacher 7B: `results/teacher_step640_passk.json`

---

## Stage C — Evaluation

Three eval modes, each a separate sbatch. All write JSON / PGN artifacts to
the path you supply via `OUT_DIR` / `OUT_PATH`.

### C-1: Puzzle pass@k

```bash
sbatch --export=ALL,\
\
MODEL=$HOME/models/chess_teacher_7b_step640,\
OUT_PATH=results/my_teacher_passk.json \
    scripts/sbatch_eval_passk.slurm
```

Runs `eval_chess_passk.py` on the held-out test parquet with k=1..8 and dumps a summary JSON.

### C-2: Full-game vs Stockfish (50 games at depth `d`)

```bash
sbatch --export=ALL,\
\
MODEL=$HOME/models/chess_teacher_7b_step640,\
OUT_DIR=results/my_teacher_fullgame_d5,\
OPPONENT_DEPTHS=5,GAMES_PER_DEPTH=50,\
STOCKFISH_BIN=/path/to/stockfish \
    scripts/sbatch_eval_fullgame.slurm
```

Output:
- `games.pgn` — all 50 games in PGN format
- `summary.json` — W/D/L, average centipawn loss per move (ACPL), forfeit count

### C-3: Head-to-head (100 games, model A vs model B)

```bash
sbatch --export=ALL,\
\
MODEL_A=$HOME/models/chess_distill_3b_step300,\
MODEL_B=$HOME/models/chess_teacher_7b_step640,\
OUT_DIR=results/my_h2h \
    scripts/sbatch_eval_h2h.slurm
```

Each model occupies its own GPU at TP=1. Output:
- `moves.jsonl` — every move from every game (FEN, ply, model, move, parse status)
- Forfeit / format-failure attribution per model

**Note on h2h interpretation**: our distill vs teacher 100-game match
finished 80/100 games before walltime expired. Of the 56 wins by distill,
~49 came from teacher *format failures* (teacher 7B with rollout 2000
tokens occasionally hit truncation mid-`<think>`). The "pure chess-strength"
wins (checkmate or resignation) were 37% distill vs 63% teacher — the
distill is **not** actually stronger, the teacher is just leaky on long
games. See `results/distill_vs_teacher_h2h/`.

---

## Reproducing our exact eval results

The artifacts in `results/` came from these sbatch invocations:

```bash
# Teacher pass@k (after merging step 640 ckpt to HF)
sbatch --export=ALL,MODEL=$TEACHER_HF,\
       OUT_PATH=results/teacher_step640_passk.json \
       scripts/sbatch_eval_passk.slurm

# Distill curve: do this for each saved step in [50, 100, 150, 200, 250, 300]
for STEP in 50 100 150 200 250 300; do
    sbatch --export=ALL,\
MODEL=$HOME/models/chess_distill_3b_step${STEP},\
OUT_PATH=results/distill_step${STEP}_passk.json \
        scripts/sbatch_eval_passk.slurm
done

# Distill step 300 full-game vs SF d=5
sbatch --export=ALL,\
MODEL=$HOME/models/chess_distill_3b_step300,\
OUT_DIR=results/distill_step300_fullgame,\
OPPONENT_DEPTHS=5 \
       scripts/sbatch_eval_fullgame.slurm

# Distill vs teacher h2h
sbatch --export=ALL,\
MODEL_A=$HOME/models/chess_distill_3b_step300,\
MODEL_B=$HOME/models/chess_teacher_7b_step640,\
OUT_DIR=results/distill_vs_teacher_h2h \
       scripts/sbatch_eval_h2h.slurm
```

---

## Key findings

1. **On-policy distillation works on chess**. A 3B student recovers 92.6%
   of the 7B teacher's puzzle pass@8 with ~16× fewer training samples than
   from-scratch RL.
2. **Per-token reverse-KL is enough signal**, even though the long
   `<think>` block (~99% of response tokens) dilutes the chess-specific
   tokens (~5 UCI tokens). The `per_token_logp_gap` collapses 76% across
   training.
3. **Puzzle pass@k does not imply chess strength**. Both teacher and
   distill lose 50/50 to Stockfish depth=1 with high ACPL — single-step
   puzzle solving is a different skill than full-game play. This motivates
   Phase 2.

---

## Roadmap (Phase 2 — what comes next)

The Phase 1 work captured here trains a single-step puzzle solver. The
**asymmetric self-play** thesis envisions a different setup:

- Two trainable agents: an A-side model and a B-side model with **different
  observation channels** (e.g. one has retrieval access to a strategy
  database, one does not).
- **Multi-turn** rollouts — full chess games, not single moves.
- **Trajectory-level reward** — terminal win/draw/loss instead of per-step
  Stockfish μ-grading.
- **Joint training** with adversarial dynamics (LUPI-style).

Concrete work items:
1. Multi-turn rollout in the trainer (the upstream framework has a
   `self_play.enable=True` hook for online model-vs-Stockfish play that
   can be adapted)
2. Trajectory-level credit assignment (sparse terminal + bootstrap value
   or PPO-style GAE)
3. Strategy retrieval / RAG opponent
4. Curriculum following the [Lichess Practice studies](https://lichess.org/practice)
   pedagogy: checkmate patterns → fundamental tactics → advanced tactics
   → endgames → full game play

---

## Citation / acknowledgements

- Pass@k GRPO: chess RL paper (EMNLP submission, code under the `recipe/chess/`
  directory of the upstream framework).
- On-policy distillation recipe:
  [Thinking Machines — *On-policy distillation*](https://thinkingmachines.ai/blog/on-policy-distillation/)
- RL framework: [volcengine/verl](https://github.com/volcengine/verl)
- Puzzle data: [Lichess puzzle database](https://database.lichess.org/) +
  Chess-R1 aligned splits.

---

## Status

Phase 1 (this repo): **complete**. Phase 2: **planning**.
