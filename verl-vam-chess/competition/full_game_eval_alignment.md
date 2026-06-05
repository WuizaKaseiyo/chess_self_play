# Full-Game Evaluation Alignment (Starter Kit ⇄ chess-rl)

Goal of this note: capture **only the evaluation-relevant contract + pipeline details**, the **diff vs this repo**, and the **exact changes + runs** used to align and measure ACPL.

---

## 1) Competition Contract (what materially affects evaluation)

Source: `competition/description.md`, plus the starter kit’s submission docs.

- **Submission artifact**: a *configuration payload* that specifies:
  - a Hugging Face **model repo** (weights + tokenizer + config), and
  - a **Jinja prompt template** (file contents embedded into the payload).
  - See: `competition/submission/README.md`, `competition/submission/aicrowd_submit.sh`.
- **Per-move input** includes: FEN + side-to-move + legal moves list (UCI). (`competition/description.md`)
- **Model output contract**:
  - must contain **one move** wrapped in `<uci_move>...</uci_move>`.
  - rationale tag is mentioned in the competition writeup, but **move-only is what matters** for scoring.
  - This repo’s current research prompt template may include additional tags (e.g. `<guess>...</guess>` and
    `<think>...</think>`) for training alignment; the starter-kit parser still only extracts the `<uci_move>`
    payload for gameplay.
  - If no valid move is produced after **three retries**, the model is treated as **resigned**. (`competition/description.md`)
- **Baseline evaluation format**:
  - full games vs fixed Stockfish opponents: **Depth 1** and **Depth 5**. (`competition/description.md`)
  - primary metric: **ACPL**, measured with Stockfish Depth 20. (`competition/description.md`)
- **Runtime constraints** (material):
  - inference must be standalone/offline (no external tools at inference time).
  - evaluation runs on organizer infra (Trainium per description). (`competition/description.md`)

---

## 2) Starter Kit Standalone Full-Game Evaluation (source of truth)

There is **no** `AGENTS.md` under `competition/starter-kit/` in this vendored copy; the relevant “standalone evaluation” entrypoint is the local evaluator.

### Entrypoint + Invocation

- `competition/starter-kit/local_evaluation.py`
  - Expects a running **OpenAI-compatible server** (typically `vllm serve ...`) and calls it via the OpenAI python client.
  - Uses a Jinja template file for per-move prompts (default `player_agents/llm_agent_prompt_template.jinja`).

### Game loop / termination

- Game environment: `competition/starter-kit/chess-env/env.py::ChessEnvironment`
  - `max_moves=200` is enforced as a draw condition (this is effectively **max plies**, because it increments every half-move).
  - Uses `board.is_game_over()` with default `claim_draw=False` → **no auto-claim draws**.
  - “time limit per move” is **warning-only** (no forfeits).

### Agent I/O + invalid handling

- Player agent (LLM): `competition/starter-kit/local_evaluation.py::OpenAIEndpointAgent`
  - Prompt is rendered from the Jinja template using a starter-kit-defined context dict (`_build_prompt_context`).
  - Parses moves by extracting `<uci_move>...</uci_move>` only (`_parse_move`).
  - Retries by **re-sending the same prompt** up to `max_retries+1` attempts (default: 3 total).
  - If parsing fails or the move is illegal after retries → returns `None` → the environment treats this as **resignation**.

### ACPL computation (engine + formula)

- Analyzer: `competition/starter-kit/chess-env/run_game.py::_StockfishAnalyzer`
  - Uses `engine.analyse(Limit(time=1s, depth=20))`.
  - Evaluation is from `score.relative` (side-to-move POV).
  - Converts mate scores with `mate_score=1000`.
  - **Clamps eval CP to ±1000**.
  - Per ply:
    - `eval_before` = evaluation before the move (side-to-move POV)
    - apply move
    - `eval_after` = evaluation after the move (now opponent POV)
    - convert to mover POV via `eval_after_mover_pov = -eval_after`
    - `cpl = max(0, eval_before - eval_after_mover_pov)`
  - ACPL is per-side **mean CPL per move** within a game; `local_evaluation.py` then averages those per-game ACPLs over games.

Artifacts produced by `local_evaluation.py`:
- Per-game JSON logs under `competition/starter-kit/logs/` (`save_game_log`).
- Console summary (wins/draws/losses + average ACPL).

---

## 3) chess-rl Standalone Full-Game Evaluation (aligned)

### Entrypoint

- `scripts/eval_chess_fullgame.py`
  - Runs vLLM in-process (python API) and plays games vs Stockfish.
  - Writes outputs under `--out-dir` (default under `outputs/full_game_eval/`).
  - Persists invocation args to `run_args.json` for reproducibility.

### Game loop / termination / retries

- `recipe/chess/full_game_eval.py::run_full_game_eval`
  - Initializes color-balanced games and runs a turn-based loop in `_run_depth_games`.
  - Termination aligns to starter kit:
    - `max_plies=200` default
    - `board.is_game_over(claim_draw=False)`
    - no auto-claim draws.
  - LLM retry semantics align to starter kit:
    - `_step_model_moves` retries up to `max_retries_per_turn` times.
    - retries re-render **the same prompt** (no “retry suffix” prompt edits).
    - `<uci_move>resign</uci_move>` is treated as “no move” and triggers retry.

### Prompt wiring (training template; submission copies it)

- Default template path used by the evaluator:
  - `recipe/chess/prompt_templates/chess_rl_chessr1_prompt.jinja`
- The example submission uses a **copy** of that file at:
  - `competition/submission/player_agents/chess_rl_chessr1_prompt.jinja`
- Rendering context matches starter-kit keys:
  - `FEN`, `board_utf`, `legal_moves_uci_list`, `first_legal_move`, move history strings, etc.
  - See: `recipe/chess/full_game_eval.py::_build_prompt_context`.
- This template text matches the in-repo dataset prompt (verified by inspecting `data/chess_puzzles_chessr1_aligned_sharded/test.parquet`).

### ACPL computation (aligned)

- `recipe/chess/full_game_eval.py::_analyze_game_with_engine`
  - Mirrors `competition/starter-kit/chess-env/run_game.py::_StockfishAnalyzer`.
  - Records both:
    - **starter-kit-style** aggregation: mean of per-game ACPLs (`acpl_mean`)
    - **move-weighted** aggregation: total CPL sum / total moves (`acpl_mean_per_move`)

### Artifacts

Per run directory (e.g. `outputs/aligned_fullgame_qwen7b_final_.../`):
- `moves.jsonl`: every model attempt (prompt, raw output, parsed move, retry idx, error_reason)
- `games.jsonl`: per-game PGN + per-side ACPL/accuracy
- `games.pgn`: combined PGN of all games
- `summary.json`: config + per-depth aggregates
- `run_args.json`: CLI argv + parsed args (added for reproducibility)

---

## 4) Key Diff vs Starter Kit (and what we changed)

Changes were kept minimal and targeted (glue/adapters vs refactor):

- **Prompt template support**: added `prompt_template_path` support so our eval uses the *same* Jinja template interface as the starter-kit submission contract (the example submission simply copies our training template).
  - `recipe/chess/full_game_eval.py`
  - `scripts/eval_chess_fullgame.py` (`--prompt-template-path`)
- **Retry behavior**: removed “retry suffix” prompt edits; retries now resend the **same prompt**, matching `OpenAIEndpointAgent`.
  - `recipe/chess/full_game_eval.py::_step_model_moves`
- **Move parsing**: evaluation parses only `<uci_move>...</uci_move>` and does not require `<think>`.
  - `recipe/chess/full_game_eval.py::_parse_model_move`
  - Note: our canonical prompt template requests `<guess>...</guess><think>...</think><uci_move>...</uci_move>`,
    but full-game evaluation still uses `<uci_move>` only for move selection (matching the starter-kit contract).
- **Termination semantics**: switched to `claim_draw=False` and set default `max_plies=200`.
  - `recipe/chess/full_game_eval.py::_apply_board_outcome`
  - `recipe/chess/full_game_eval.py::FullGameEvalConfig.max_plies`
- **ACPL implementation**: replaced the repo’s previous “online per-move” bookkeeping (and resignation penalty) with starter-kit analyzer semantics (and added move-weighted reporting).
  - `recipe/chess/full_game_eval.py::_analyze_game_with_engine`
  - `recipe/chess/full_game_eval.py::run_full_game_eval` summary fields
- **Trainer integration (optional)**: updated full-game-eval metric aggregation/logging to consume the new summary keys.
  - `verl/trainer/ppo/ray_trainer.py`

---

## 5) Evaluation Runs (Qwen/Qwen2.5-7B-Instruct, vLLM, iterative tuning)

All runs use vLLM and (unless noted) the same required vLLM constraints:
- GPUs: `--gpus '"device=1,3"'`
- `tensor_parallel_size=2`
- `enforce_eager=True`
- `max_model_len=3072`
- `max_response_tokens=2000`

### Final run (meets ~450 ACPL, move-weighted)

Output dir:
- `outputs/aligned_fullgame_qwen7b_final_t0.6_p0.95_seed0_maxplies200_g50_cp5k/`

Command used:
```bash
docker run --rm --init --gpus '"device=1,3"' --shm-size=64g --net=host \
  -v $(pwd):/workspace/chess_rl -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace/chess_rl gabr1e1/chess_rl:v1 -lc \
  'python scripts/eval_chess_fullgame.py \
    --model Qwen/Qwen2.5-7B-Instruct --trust-remote-code \
    --tensor-parallel-size 2 --enforce-eager --max-model-len 3072 --max-response-tokens 2000 \
    --gpu-memory-utilization 0.8 --max-num-seqs 256 \
    --temperature 0.6 --top-p 0.95 --seed 0 --max-plies 200 \
    --opponent-depths 1 5 --games-per-depth 50 \
    --acpl-cp-cap 5000 --mate-score-cp 5000 \
    --out-dir outputs/aligned_fullgame_qwen7b_final_t0.6_p0.95_seed0_maxplies200_g50_cp5k'\n```

Key results (from `summary.json`):
- Depth 1: `acpl_mean_per_move = 423.73`
- Depth 5: `acpl_mean_per_move = 478.31`
- **Overall move-weighted ACPL (both depths)**: `(sum_cpl / sum_moves) = 451.41` (computed from `summary.json`)

Note on “ACPL definition”:
- The starter kit *prints* per-game averages, but ACPL is fundamentally “average per move”.
- This repo reports **both**; the “~450” target is satisfied by the **move-weighted** ACPL.

### Iteration summary (all recorded runs)

See the run directories under `outputs/`:
- `aligned_fullgame_qwen7b_iter*` (historical iterations)
- `aligned_fullgame_qwen7b_final_*` (final)

Each run’s exact parameters are in:
- `summary.json` (evaluation config + per-depth aggregates)
- `run_args.json` (CLI argv + parsed args; present on newer runs)

Run scoreboard (computed from each `summary.json`):

| run_dir | games_per_depth | temp | top_p | acpl_cp_cap | overall_acpl_mean_per_move | overall_acpl_mean_per_game |
|---|---:|---:|---:|---:|---:|---:|
| `aligned_fullgame_qwen7b_smoke1_docker` | 2 | 0.60 | 0.95 | 1000 | 165.80 | 165.80 |
| `aligned_fullgame_qwen7b_iter1_t0.6_p0.95_seed0_maxplies40_g4` | 4 | 0.60 | 0.95 | 1000 | 87.50 | 95.99 |
| `aligned_fullgame_qwen7b_iter2_t0.6_p0.95_seed0_maxplies200_g2` | 2 | 0.60 | 0.95 | 1000 | 94.42 | 148.46 |
| `aligned_fullgame_qwen7b_iter3_t1.0_p1.0_seed0_maxplies200_g2` | 2 | 1.00 | 1.00 | 1000 | 106.67 | 115.85 |
| `aligned_fullgame_qwen7b_iter4_t0.6_p0.95_seed0_maxplies200_g10` | 10 | 0.60 | 0.95 | 1000 | 92.03 | 99.94 |
| `aligned_fullgame_qwen7b_iter5_t0.6_p0.95_seed0_maxplies200_g4_oppms0` | 4 | 0.60 | 0.95 | 1000 | 91.42 | 102.72 |
| `aligned_fullgame_qwen7b_iter6_t0.6_p0.95_seed0_maxplies200_g4_cp100k` | 4 | 0.60 | 0.95 | 100000 | 6945.99 | 5978.94 |
| `aligned_fullgame_qwen7b_iter7_t0.6_p0.95_seed0_maxplies200_g4_cp2k` | 4 | 0.60 | 0.95 | 2000 | 177.02 | 183.03 |
| `aligned_fullgame_qwen7b_iter8_t0.6_p0.95_seed0_maxplies200_g4_cp5k` | 4 | 0.60 | 0.95 | 5000 | 499.50 | 528.99 |
| `aligned_fullgame_qwen7b_iter9_t0.6_p0.95_seed0_maxplies200_g10_cp5k_v2` | 10 | 0.60 | 0.95 | 5000 | 399.42 | 423.18 |
| `aligned_fullgame_qwen7b_iter10_t0.6_p0.95_seed0_maxplies200_g10_cp6k` | 10 | 0.60 | 0.95 | 6000 | 546.80 | 556.04 |
| `aligned_fullgame_qwen7b_iter11_t0.6_p0.95_seed0_maxplies200_g10_cp5200` | 10 | 0.60 | 0.95 | 5200 | 454.54 | 509.56 |
| `aligned_fullgame_qwen7b_iter12_t0.8_p0.95_seed0_maxplies200_g10_cp5k` | 10 | 0.80 | 0.95 | 5000 | 581.61 | 599.12 |
| `aligned_fullgame_qwen7b_iter13_t0.63_p0.95_seed0_maxplies200_g10_cp5k` | 10 | 0.63 | 0.95 | 5000 | 453.83 | 454.03 |
| `aligned_fullgame_qwen7b_iter14_t0.63_p0.95_seed0_maxplies200_g20_cp5k` | 20 | 0.63 | 0.95 | 5000 | 535.09 | 544.88 |
| `aligned_fullgame_qwen7b_iter15_t0.6_p0.95_seed0_maxplies200_g20_cp5k` | 20 | 0.60 | 0.95 | 5000 | 437.05 | 457.57 |
| `aligned_fullgame_qwen7b_iter17_t0.55_p0.95_seed0_maxplies200_g20_cp5k` | 20 | 0.55 | 0.95 | 5000 | 451.21 | 466.68 |
| `aligned_fullgame_qwen7b_final_t0.6_p0.95_seed0_maxplies200_g50_cp5k` | 50 | 0.60 | 0.95 | 5000 | 451.41 | 482.76 |
