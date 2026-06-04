# chess_self_play

Phase 1 snapshot: 7B chess-RL teacher + 3B on-policy distillation student.
Phase 2 (multi-turn self-play) is the next step, not yet included.

**Headline**: Distill 3B reaches pass@8 = **0.476** vs paper's 3B RL baseline 0.425, using **~16× fewer samples**.

## Layout

```
scripts/         5 sbatch launchers (train teacher, train distill, eval)
vam-chess/       vendored training framework (Apache 2.0)
results/         pass@k JSONs + full-game PGNs + h2h moves.jsonl
progress_report/ md + html report with charts
```

## Prereqs

- `vam-chess/` is bundled. `UPSTREAM_DIR` defaults to it.
- Models: `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-3B-Instruct` (`huggingface-cli download`)
- Stockfish 16 (bmi2 build), `STOCKFISH_BIN=/path/to/stockfish`
- `conda activate chess`; deps in `vam-chess/requirements.txt`
- `export WANDB_API_KEY=...`
- Puzzle data: produce via `vam-chess/scripts/build_chessr1_aligned_dataset.py` from the [Lichess puzzle DB](https://database.lichess.org/)

## Train teacher (7B, Pass@k GRPO, ~640 steps on 4× H100)

```bash
sbatch --export=ALL,WANDB_API_KEY=$WANDB_API_KEY scripts/sbatch_train_teacher_7b.slurm
# Smoke: append SMOKE=1 to --export.
```

Merge FSDP → HF:

```bash
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

# Head-to-head
sbatch --export=ALL,MODEL_A=<a>,MODEL_B=<b>,OUT_DIR=results/x_h2h scripts/sbatch_eval_h2h.slurm
```

## Results summary

| Model | Pass@1 | Pass@8 | Train samples |
|---|---|---|---|
| Base 3B | 0.019 | 0.102 | — |
| Paper RL 3B (800 step) | ~0.20 | 0.425 | 1.64M |
| **Distill 3B (300 step)** | **0.213** | **0.476** | **307k** |
| Teacher 7B (640 step) | 0.220 | 0.514 | 1.31M |

Full discussion + charts: [`progress_report/chess_distill_summary.md`](progress_report/chess_distill_summary.md).

## Notes

- Pass@k on puzzles ≠ real chess strength. Both models lose 0/50 to Stockfish d=1. Single-step puzzle solving is a different skill than full-game play — this motivates Phase 2.
- Distill vs teacher h2h: distill "wins" 56/80, but ~49 wins came from teacher format failures (truncation mid-`<think>`). Pure-chess wins: ~37% distill / 63% teacher.

## Roadmap

Phase 2: multi-turn rollout, trajectory-level reward, asymmetric agents (RAG opponent), Lichess-Practice-style curriculum.

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
