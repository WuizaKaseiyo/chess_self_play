# One-on-one (Model-vs-Model) Full-Game Chess Rollouts (Isambard GH200)

This doc describes how we run **model-vs-model full-game chess rollouts** on the **Isambard GH200** cluster using the repo’s aligned full-game-eval semantics (starter-kit compatible `<uci_move>` parsing, retry/forfeit handling, `max_plies=200`, `claim_draw=False`).

Goal here: **rollouts only** (save games + per-move traces). **ACPL is optional** and is *not required* for head-to-head win/draw/loss.

## 1) Approach

### 1.1 Serving/inference backend (vLLM, OpenAI-compatible)
- Each model is served via **vLLM** using the OpenAI-compatible server (`vllm.entrypoints.openai.api_server`).
- Evaluators send OpenAI Chat Completions requests to `http://<host>:<port>/v1`.
- For multi-node runs, servers bind `0.0.0.0`, so other nodes in the same Slurm allocation can call them.

### 1.2 Prompting + move parsing semantics (aligned)
- Prompts are rendered from a Jinja template; we used:
  - `recipe/chess/prompt_templates/select_prompt.jinja`
- In full-game eval, `allowed_moves`/`considered_moves_uci_list` is set to **all legal moves** for the position (so this is “selection-format output” but full-legal gameplay).
- Move parsing is strict and starter-kit aligned:
  - Requires `<uci_move>...</uci_move>` tags (case-insensitive tags, strict content).
  - If missing or invalid (`format_missing`, `illegal_move`, etc.), the evaluator retries the **same prompt** up to `max_retries_per_turn` times total (so `max_retries_per_turn=1` means a single attempt).
  - If no valid move after retries, the side to move **forfeits**, recorded as `termination="resignation"`.
  - Game end conditions: natural termination from `python-chess` with `claim_draw=False`, or `max_plies=200` hard cap.
  - Source of truth: `recipe/chess/full_game_eval.py` (`_parse_model_move`, `_set_forfeit`).

### 1.3 Parallelism model (keep GPUs busy)
- vLLM can schedule many concurrent requests internally.
- The evaluator also fans out requests (per actor) so a single server sees many in-flight generations.
- We removed the previous client-side cap by defaulting `--backend-max-workers` to `0` (meaning “no explicit cap”).
  - You can still cap if needed: set `BACKEND_MAX_WORKERS=<N>` in the sbatch `--export`.

### 1.4 Stability notes (what we learned)
- Some checkpoints crashed vLLM in CUDA-graphs mode (`--no-enforce-eager`) with device-side asserts; this was **fixed by using `--enforce-eager`**.
- Fully uncapped client fan-out initially produced transient OpenAI connection errors; enabling OpenAI-client retries (`max_retries=2`) eliminated `<error>APIConnectionError: ...</error>` outputs in the stable run.

## 2) Scripts + how to run on Isambard (multi-node)

There are two useful entrypoints:

### 2.1 Multi-node “round-robin” inference (12 models across 3 nodes)
This is the fastest way to run *many* one-on-one matchups in one allocation.

- Python:
  - `scripts/eval_chess_fullgame_round_robin_distributed_infer.py`
- Slurm launcher (GH200):
  - `sbatch_eval_chess_fullgame_round_robin_distributed_infer_gh200.slurm`

What it does:
- Starts **one vLLM server per model** (1 GPU each) across **3 nodes × 4 GPUs/node = 12 models**.
- Writes a shared `cluster/server_map.json`, then each node coordinates a subset of pairs.
- Produces per-shard rollout artifacts.

Submit (no `--wait`, 24h limit, 100 games per pair, eager mode, no client cap):
```bash
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=1440 a5l.aip2.isambard '
  set -euo pipefail
  cd ~/code/chess-rl
  sbatch --time=1-00:00:00 \
    --export=ALL,GAMES_PER_PAIR=100,ENFORCE_EAGER=1,BACKEND_MAX_WORKERS=0,TEMPERATURE=0.6,TOP_P=0.95,MAX_RESPONSE_TOKENS=2000,MAX_RETRIES_PER_TURN=1 \
    ./sbatch_eval_chess_fullgame_round_robin_distributed_infer_gh200.slurm
'
```

Monitor:
```bash
ssh a5l.aip2.isambard 'squeue -j <JOBID> -o "%i %t %M %R %j"'
ssh a5l.aip2.isambard 'tail -n 50 ~/code/chess-rl/slurm/slurm-chess-rrdist-infer-<JOBID>.out'
```

### 2.2 One-node “pilot” inference (4 models, 1 GPU each)
Useful for speed / stability benchmarking before running the multi-node sweep.

- Python:
  - `scripts/eval_chess_fullgame_round_robin_infer.py`
- Slurm launcher:
  - `sbatch_eval_chess_fullgame_round_robin_pilot_infer_gh200.slurm`

Example (4 selected run ids, 100 games per pair):
```bash
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=1440 a5l.aip2.isambard '
  set -euo pipefail
  cd ~/code/chess-rl
  sbatch --wait \
    --export=ALL,RUN_ID_0=5cw6hm16,RUN_ID_1=h6sqp0z4,RUN_ID_2=nxhozx89,RUN_ID_3=yu8phknt,GAMES_PER_PAIR=100,ENFORCE_EAGER=1,BACKEND_MAX_WORKERS=0,TEMPERATURE=0.6,TOP_P=0.95,MAX_RESPONSE_TOKENS=2000,MAX_RETRIES_PER_TURN=1 \
    ./sbatch_eval_chess_fullgame_round_robin_pilot_infer_gh200.slurm
'
```

### 2.3 Optional ACPL stage (not needed for rollouts-only)
If you want ACPL later, there are two-stage runners:
- ACPL for distributed round-robin:
  - `sbatch_eval_chess_fullgame_round_robin_distributed_acpl_gh200.slurm`
  - `scripts/eval_chess_fullgame_round_robin_acpl.py`

## 3) Current run (this session) + where to find results

### 3.1 Slurm job
- Inference job id: `2168856`
- Job name: `chess-rrdist-infer`
- Nodes: `nid[011042-011044]`
- Status: `COMPLETED (ExitCode=0:0)`
- Elapsed: `03:38:12`
- Repo commit on cluster at submit time: `11764132cfa4c663560995b1cb3d5bc7fd8a141a`

### 3.2 Output directory (rollouts)
Root:
- `/projects/a5l/ziyan/chess_rl/outputs/full_game_eval_rr_distributed/job_2168856`

What’s inside:
- `shard_0/`, `shard_1/`, `shard_2/`: one per node
  - `summary_infer.json`: per-pair W/D/L counts + timing for that shard
  - `games.jsonl`: one JSON record per finished game (includes PGN text + termination + forfeit info)
  - `moves.jsonl`: per-move trace (includes prompt_text + raw_output_text for debugging)
  - `games.pgn`: PGN file with all games from the shard
- `cluster/server_map.json`: model → (`host`, `port`, `base_url`) used for cross-node inference
- `logs/`: vLLM server stdout/stderr logs (one file per model server)

Quick integrity checks:
```bash
# should be 6600 total games (66 pairs × 100 games)
wc -l /projects/a5l/ziyan/chess_rl/outputs/full_game_eval_rr_distributed/job_2168856/shard_*/games.jsonl
```

### 3.3 Important caveat: forfeits from format/legality failures
This run produced many forfeits due to strict `<uci_move>` parsing + legality checks:
- Total forfeits: `1119 / 6600` (17.0%)
- Top reasons: `format_missing` and `illegal_move`
- Worst offenders (forfeit side attributed to the side-to-move at failure): `82fpo6l0` and `u2cuw56a`

If you care about “clean” head-to-head results, you likely want to:
- filter pairs/runs with high forfeit rates, or
- fix prompts / model behavior to always emit `<uci_move>...</uci_move>` with strict UCI.

---

If you only need rollouts (W/D/L + PGNs + traces), **the inference run is sufficient**; do not run the ACPL stage.

## 4) Job `2168856` post-run analysis addendum (2026-02-06)

This section captures the follow-up analysis generated on **2026-02-06** from:
- `analysis/one_on_one/job_2168856`

### 4.1 Elo + matrix outputs snapshot

Core outputs for this run:
- `analysis/one_on_one/job_2168856/elo_ratings.csv`
- `analysis/one_on_one/job_2168856/pairwise_results.csv`
- `analysis/one_on_one/job_2168856/winrate_matrix.csv`
- `analysis/one_on_one/job_2168856/winrate_matrix.png`
- `analysis/one_on_one/job_2168856/elo_diff_matrix.csv`
- `analysis/one_on_one/job_2168856/elo_diff_matrix.png`
- `analysis/one_on_one/job_2168856/elo_summary.json`
- `analysis/one_on_one/job_2168856/acpl_vs_h2h_diagnosis.csv`

Top Elo ranking (all games, anchored mean 1500):
1. `mk76juq4` — Elo `1562.90`, score rate `0.5945`
2. `azs0jkjg` — Elo `1545.05`, score rate `0.5677`
3. `h4rhtpg5` — Elo `1539.06`, score rate `0.5586`
4. `h6sqp0z4` — Elo `1526.82`, score rate `0.5400`
5. `dg41tlmo` — Elo `1515.53`, score rate `0.5227`

How to read the two matrices:
- `winrate_matrix.*`: direct empirical score (`win=1`, `draw=0.5`, `loss=0`) for row model vs column model.
- `elo_diff_matrix.*`: global fitted Elo difference (`row Elo - column Elo`) from one shared Elo fit over all pair results.

### 4.2 Draw semantics clarification

Why draws are high:
- Draws: `4634 / 6600` (`70.2%`).
- Draws among non-forfeit games: `4634 / 5481` (`84.5%`).
- Main draw source is automatic fivefold repetition (`4272 / 4634`, `92.2%` of draws).

Fivefold repetition (plain terms):
- If the exact same position appears 5 times (same side to move, castling rights, and en-passant square), the game is automatically drawn.

Draw termination counts in job `2168856`:
- `fivefold_repetition`: `4272`
- `max_moves`: `260`
- `stalemate`: `74`
- `insufficient_material`: `28`

`max_plies` clarification:
- `max_plies=200` means at most 200 half-moves (plies) per game.
- If this cap is reached before a natural terminal board state, the evaluator records a draw with `termination="max_moves"`.

Semantics notes:
- `claim_draw=False` means threefold repetition and 50-move rule are not auto-claimed.
- Automatic fivefold/75-move and the explicit `max_plies=200` cap still terminate games as draws.

### 4.3 Forfeit decomposition (why decisive results skew)

Global:
- Forfeits: `1119 / 6600` (`16.95%`).
- Decisive games: `1966` (`29.8%` of games).
- Forfeits as share of decisive games: `1119 / 1966` (`56.9%`) (more than half of decisive outcomes).

`5cw6hm16`:
- Forfeits committed: `32`
- Opponent forfeits: `86`
- Overall score rate: `0.522273`
- Non-forfeit score rate: `0.497454`
- Committed-forfeit reasons: `format_missing=30`, `illegal_move=2`

Move-level finding:
- For `5cw6hm16`, `20/30` `format_missing` events had raw output:
  `<error>APIConnectionError: Connection error.</error>`.
- These were concentrated late (mostly ply `194-195`) and mostly in `5cw6hm16` vs `mk76juq4` (`19` such events).
- This pattern points to transport/output reliability failures, not purely move-quality weakness.

Pair examples (`decomposition_pairwise_5cw6hm16.csv`):
- vs `mk76juq4`: overall `0.430000`, non-forfeit `0.519481`, forfeits committed/opponent `20/3`
- vs `82fpo6l0`: overall `0.710000`, non-forfeit `0.500000`, forfeits committed/opponent `0/42`
- vs `u2cuw56a`: overall `0.630000`, non-forfeit `0.492537`, forfeits committed/opponent `3/30`

Takeaway:
- Forfeit asymmetry changes apparent model-vs-model strength substantially.
- Use non-forfeit metrics for chess-strength comparisons; treat forfeits as a separate reliability axis.

### 4.4 ACPL vs H2H interpretation

ACPL and this H2H Elo are not interchangeable in this run:
- ACPL is move-quality vs analysis baseline.
- H2H score here is strongly affected by formatting/legality/transport failures.

Examples:
- `5cw6hm16`: ACPL rank `1`, H2H Elo rank `6` (all games), `7` (non-forfeit).
- `mk76juq4`: ACPL rank `10`, H2H Elo rank `1` (all games), `6` (non-forfeit).

Practical guidance:
- Use non-forfeit H2H Elo/score for chess-strength comparisons in this setup.
- Track forfeit rate as a separate reliability metric.
- Treat ACPL as a secondary signal, not a direct proxy for forfeit-heavy H2H rank.

### 4.5 Competition-rule clarification

For the in-repo starter-kit competition paths, draw semantics are effectively aligned with `full_game_eval`:
- `claim_draw=False` behavior.
- No auto-claim for threefold/50-move.
- Automatic fivefold/75-move still applies.
- Same practical 200-ply (`max_moves`) draw cap in local evaluation.

Label nuance:
- Some starter-kit labels may still say `"Threefold repetition"` / `"Fifty-move rule"` even when game-over behavior follows non-claim semantics.

Reference:
- `analysis/one_on_one/competition_draw_rule_investigation.md` (2026-02-06)

### 4.6 Repro commands (job `2168856`)

Elo + matrix analysis:
```bash
python scripts/analyze_one_on_one_round_robin_infer.py \
  /projects/a5l/ziyan/chess_rl/outputs/full_game_eval_rr_distributed/job_2168856 \
  --output-dir analysis/one_on_one/job_2168856 \
  --expected-model-count 12 \
  --expected-games-per-pair 100
```

Forfeit/non-forfeit decomposition (targeting `5cw6hm16`):
```bash
python analysis/one_on_one/job_2168856/decompose_results.py \
  --input-dir analysis/one_on_one/job_2168856 \
  --output-dir analysis/one_on_one/job_2168856 \
  --target-model 5cw6hm16
```
