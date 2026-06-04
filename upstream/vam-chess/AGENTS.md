# Chess RL (Start Here)

This repo’s current focus is **restricted-moves (“selection”) chess move prediction** framed as
**verbalized action masking**:
the model is shown an explicit candidate-move shortlist (`allowed_moves`) and must choose the best move
*from that list*.

If you are changing prompt templates, reward parsing, evaluators, or training launchers, read these first:
- `restricted_moves.md` (selection contract + evaluation + hard gates)
- `iterative.md` (current iterative training loop: `allowed_move_elim` + optional online play vs engine opponent)

---

## Working Style

- Execution venue policy:
  - Use the local environment for read-only inspection, formatting/static checks, unit tests, tiny CPU-only scripts, reward/parser tests, small data samples, and simple bash checks.
  - Use `isambard` for GPU, vLLM, Stockfish-heavy, training, smoke/debug training runs, and full evaluation runs.
- Local environment for lightweight work:
  - `conda activate verl`
  - `export CUDA_DEVICE_ORDER=PCI_BUS_ID`
  - `export CUDA_VISIBLE_DEVICES=1,3`
- Docs and comments can drift; **never treat any single document as the bible**. When in doubt, verify against the actual code paths, scripts, and configs.
- Generally ignore generated output folders (e.g., `outputs/`, W&B exports, caches) unless you are deliberately investigating a specific run/artifact.

---

## Selection Contract (Non-Negotiable)

Terminology / naming (kept for back-compat):
- **Prompt text**: calls the candidate list `allowed_moves` (what the model sees).
- **Jinja variable** (for `select_prompt.jinja`): `considered_moves_uci_list`.
- **Dataset field** (for selection training): `reward_model.considered_moves_uci`.

Hard gates:
- Output must be exactly: `<think>...</think><uci_move>...</uci_move>`.
- `<uci_move>` must be strict UCI (e.g., `e2e4`, `g1f3`, `a7a8q`) and **must match an element of `allowed_moves` exactly**.
- If `allowed_moves` contains exactly **one** move, the model must output that move.
- Do not emit any text or whitespace outside the two tags.
- The `<uci_move>` tag content must be only the UCI string, with no surrounding whitespace.
- Newlines are allowed inside `<think>...</think>`, but the safest final surface is exactly `</think><uci_move>` with no extra text between tags.

---

## Delegation Rules

This project supports delegating standalone tasks such as code implementation, code inspection,
documentation reading, debugging, online search, plotting, evaluation inspection, dataset
inspection, and writing/running unit tests.

Rules:

- Use subagents when they help with well-scoped delegated work. In every subagent prompt, explicitly instruct the subagent to think extremely hard and deeply before doing anything. The main agent must hold itself to the same standard.
- Only spawn subagents of type `coding_agent`, `researcher`, or `verifier`. This is mandatory; do not use any other agent type.
- If model and reasoning effort need to be specified for a subagent, use `gpt-5.5` with `xhigh` reasoning effort.
- Choose subagent type by task:
  - `coding_agent`: substantive coding tasks, codebase exploration, debugging, and implementation work
  - `researcher`: conceptually hard reasoning tasks with little/no implementation
  - `verifier`: read-only verification that checks whether proposed or implemented changes are justified from first principles and avoid unnecessary complexity
  - If unsure between `coding_agent` and `researcher`, use `coding_agent`. Use `verifier` for verification, not implementation.
- Provide each subagent with enough context to succeed: goal, constraints, relevant files/commands, and a clear definition of done.
- Whenever you delegate an implementation-oriented task to a subagent, also spawn a `verifier` agent for that task.
- If a `verifier` is spawned earlier to validate work in progress, spawn a `verifier` again after the task is done to verify the final result.
- The `verifier` must confirm that proposed or implemented changes are justified from first principles, tightly scoped to the real problem, and free of unnecessary abstraction, refactoring, or complexity.
- If the `verifier` concludes that extra complexity is warranted, explicitly report that to the user and explain why.
- Once a task is delegated, the main agent may inspect enough to integrate, review, and verify the result, but must not duplicate the delegated implementation or exploration.
- Subagents must never run persistent monitoring or relaunch loops. This includes infinite `for`/`while` loops, recurring polling, background watch processes, autonomous workflows that continue running after the subagent's main task should have ended, or logic that automatically resubmits or relaunches jobs.
- If monitoring is needed, it must be a bounded one-shot status check that exits immediately after reporting the current state.
- Both the main agent and all subagents must not use Codex background terminal execution or sessions, including detached or backgrounded command flow, especially for `sbatch --wait` and any other tasks that require waiting in the foreground.
- If you delegate work, wait for **all** subagents to finish and review their outputs before proceeding.
- Before moving on, fully close out delegated work: collect each final result, ensure the subagent is no longer running, and ensure no background monitoring, polling, or follow-up submissions remain active.
- If an agent was created for a single, self-contained task, verify completion and close it with the available close-agent mechanism. Only keep an agent open if it contains especially valuable context that would be difficult to recover.
- Use `timeout_ms=3600000` when waiting on subagents because some agent tasks might be long-running.
- Subagents should not launch remote `sbatch` jobs because that can create duplicate cluster runs. The only acceptable exception is a very quick smoke run with an expected runtime of at most 30 minutes.
- Any longer or non-smoke remote `sbatch` job must be launched by the main agent, not a subagent.
- Before the main agent launches any remote `sbatch` job, it must check for existing relevant jobs first, for example with `squeue`, to avoid duplicate submissions.

---

## Datasets (Current)

### Searchless-chess (puzzle-style)

Base source parquets (engine-scored legal moves):
- `data/chess_puzzles/train.parquet`
- `data/chess_puzzles/train_hard.parquet`
- `data/chess_puzzles/test.parquet`

Selection-framed full-legal parquets (recommended offline base; prompts rendered from `select_prompt.jinja`):
- `data/chess_puzzles_select_v4/{train,train_hard,test}.parquet`

Build v4 (full-legal only):
```bash
python scripts/build_chess_select_train_dataset_v4.py \
  --input_dir data/chess_puzzles \
  --output_dir data/chess_puzzles_select_v4 \
  --overwrite
```

### Chess‑R1 aligned (prompt-variant rewrites)

We keep Chess‑R1 aligned rows sharded on disk. To get a **selection/action-masking** prompt variant (with
`allowed_moves` and `reward_model.considered_moves_uci = legal_moves_uci`), rewrite prompts from the template:

```bash
python scripts/rewrite_chess_prompts_from_template.py \
  --input_dir data/chess_puzzles_chessr1_aligned_sharded \
  --output_dir data/chess_puzzles_chessr1_aligned_sharded_ours \
  --template_path recipe/chess/prompt_templates/select_prompt.jinja \
  --set_considered_moves_uci \
  --overwrite
```

Use for selection training/eval:
- `data/chess_puzzles_chessr1_aligned_sharded_ours/`

Final/reference W&B runs used for recent comparisons:
- GRPO reward-function runs `mk76juq4`, `iu768gtj`, and `yu8phknt` used
  `data/chess_puzzles_chessr1_aligned_sharded_baseline/{train_0,train_1}.parquet`
  with validation on `test.parquet` and `test_shuffled_legal_moves.parquet`.
- GRPO model-size run `82fpo6l0` used the same
  `data/chess_puzzles_chessr1_aligned_sharded_baseline/` dataset.
- VAM model-size runs `s0anl08n` and `h4rhtpg5` used
  `data/chess_puzzles_chessr1_aligned_sharded_ours/{train_0,train_1}.parquet`
  with validation on `test.parquet` and `test_shuffled_legal_moves.parquet`.
- These are intentionally different prompt/dataset variants: GRPO baseline references use the Chess-R1 baseline prompt data, while VAM uses the selection/action-masking rewrite.

---

## Workflows (Current)

### 1) Offline fixed-dataset training (parquet-driven)

Entrypoints:
- Wrapper: `train_chess.sh`
- Cluster launcher: `sbatch_train_chess_gh200.slurm`
- Prompt template: `recipe/chess/prompt_templates/select_prompt.jinja`
- Reward: `recipe/chess/reward_fn.py`

Recommended default datasets:
- Searchless: `CHESS_DATA_DIR=data/chess_puzzles_select_v4` (+ `USE_HARD_DATASET=True` for train_hard)
- Chess‑R1 aligned (selection prompt): `CHESS_DATA_DIR=data/chess_puzzles_chessr1_aligned_sharded_ours`
- Chess‑R1 baseline/GRPO reference runs: `CHESS_DATA_DIR=data/chess_puzzles_chessr1_aligned_sharded_baseline`
- Do not rely on the bare launcher defaults for current/reference runs: `train_chess.sh` and
  `sbatch_train_chess_gh200.slurm` still default to `data/chess_puzzles_select`, so set
  `CHESS_DATA_DIR` explicitly.

### 2) Online iterative training (engine opponent + iterative action masking)

This is the current “online” data source:
- Enable online play vs engine opponent: `CHESS_SELF_PLAY_ENABLE=True`
- Requires iterative sampler:
  - through `sbatch_train_chess_gh200.slurm`: `CHESS_ALLOWED_MOVE_ELIM_ENABLE=True`
  - when invoking `train_chess.sh` directly: `ALLOWED_MOVE_ELIM_ENABLE=True`

Allowed-move-elimination environment variable mapping:
- `sbatch_train_chess_gh200.slurm` accepts `CHESS_ALLOWED_MOVE_ELIM_*` variables as launcher-facing knobs.
- The Slurm launcher exports the corresponding unprefixed `ALLOWED_MOVE_ELIM_*` variables.
- `train_chess.sh` consumes the unprefixed `ALLOWED_MOVE_ELIM_*` variables directly.

Debugging:
- Inspect a dumped self-play batch with `scripts/inspect_self_play_batch.py`.

See `iterative.md` for the exact per-step semantics and knobs.

---

## Evaluation (Selection Framing)

Selection evaluation scripts (all are cached/resumable and enforce a one-candidate sanity gate):
- `scripts/eval_chess_select_pairs.py` (+ `scripts/analyze_chess_select_pairs.py`)
- `scripts/eval_chess_select_triples.py` (+ `scripts/analyze_chess_select_triples.py`)
- `scripts/eval_chess_select_subsets.py` (+ `scripts/analyze_chess_select_subsets.py`)
- `scripts/eval_chess_select_position_sweep.py` (+ `scripts/analyze_chess_select_position_sweep.py`)

See `restricted_moves.md` for the contract + how to run locally vs GH200.

---

## Full-game evaluation (competition-style)

Full-game eval (vLLM + Stockfish, ACPL-aligned to the starter kit) is implemented in:
- Runner: `scripts/eval_chess_fullgame.py`
- Core loop: `recipe/chess/full_game_eval.py`
- Alignment notes: `competition/full_game_eval_alignment.md`

Prompting:
- Default template for the standalone runner is `recipe/chess/prompt_templates/chess_rl_chessr1_prompt.jinja`.
- For current selection/action-masking prompts, pass:
  - `--prompt-template-path recipe/chess/prompt_templates/select_prompt.jinja`
  - (full-game eval sets `allowed_moves == legal_moves` via `considered_moves_uci_list`).

Training integration:
- `train_chess.sh` exposes `FULL_EVAL_*` knobs and picks a reasonable default prompt template based on whether
  `ALLOWED_MOVE_ELIM_ENABLE=True` is set (`CHESS_ALLOWED_MOVE_ELIM_ENABLE=True` through the Slurm wrapper).

---

## Isambard (GH200) quickstart (environment)

- Login: `ssh a5l.aip2.isambard`
- To run commands on the cluster, use `ssh a5l.aip2.isambard ' command '` (this is very important!)
- Repo location on cluster: typically `~/code/chess-rl` (or wherever you cloned it). Run `sbatch_*.slurm`
  scripts from the repo root so relative paths resolve.
- Sync policy: use Git only when the user intends or approves a commit/push. Do not bundle unrelated dirty work.
  For cluster runs, confirm the sync method before committing; the usual flow is commit + push locally, then
  `git pull` on-cluster.
- Slurm submission policy:
  - Before submitting, check existing relevant jobs with `squeue` to avoid duplicates.
  - For debug/smoke jobs (anything that is not a full run), or when explicitly told to wait, use `sbatch --wait`.
  - For full runs, sweeps, and other non-debug submissions, use `sbatch` without `--wait`.
  - For full submissions with multiple jobs, submit all intended jobs together after the `squeue` check: use one SSH command from local, or one shell block if already on Isambard. Do not submit one job, wait for it, then submit the next.

Storage / scratch policy (important):
- `$HOME` on the cluster is small (~50G). Prefer writing large artifacts under `/projects/...`.
- These GH200 launchers default to `/projects/a5l/ziyan/chess_rl` for logs + caches when writable:
  - override root via `CHESS_RL_PROJECTS_ROOT=/projects/<your_path>/chess_rl`
  - logs default to `/projects/a5l/ziyan/chess_rl/logs/slurm-<jobname>-<jobid>.out`

Useful Slurm commands:
- Cluster status: `sinfo`
- Your jobs: `squeue -u $USER -o "%i %t %M %R %j"`
- Cancel: `scancel <jobid>`
- Logs: `tail -n 200 /projects/a5l/ziyan/chess_rl/logs/slurm-<jobname>-<jobid>.out`
- If live log output is explicitly needed, bound it, for example:
  `timeout 60 tail -f /projects/a5l/ziyan/chess_rl/logs/slurm-<jobname>-<jobid>.out`

Scratch / tmpdir behavior (Apptainer/Singularity):
- GH200 nodes are `arm64/aarch64` (the `sbatch_*_gh200.slurm` scripts hard-fail on `x86_64`).
- Apptainer rootless image extraction can require extended attributes (xattrs); home filesystems may not
  support them.
- The training launcher auto-selects a writable tempdir, roughly preferring:
  1) `/projects/...` scratch (persistent; avoids `/tmp` tmpfs pressure),
  2) node-local `/local/user/1483802860/$USER/...` (if writable on **all** nodes),
  3) `SLURM_TMPDIR` / `/tmp/$USER/...`.
- You can pin these explicitly at submit time:
  - `CHESS_RL_SINGULARITY_TMPDIR=...` (Apptainer extraction/build tmp)
  - `CHESS_RL_TMPDIR=...` (general tmp / TMPDIR)
  - `CHESS_RL_APPTAINER_CACHEDIR=...`, `CHESS_RL_WANDB_DIR=...`

Multi-node Ray training (Isambard GH200):
- `sbatch_train_chess_gh200.slurm` will auto-start a Ray head + workers when `--nodes > 1`.
- Debug/smoke from local: wrap the `squeue` check and `sbatch --wait` command in SSH:
```bash
ssh a5l.aip2.isambard 'cd ~/code/chess-rl && \
  squeue -u $USER -o "%i %t %M %R %j" && \
  sbatch --wait --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    ./sbatch_train_chess_gh200.slurm'
```
- Debug/smoke if already on Isambard:
```bash
squeue -u $USER -o "%i %t %M %R %j"
sbatch --wait --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  ./sbatch_train_chess_gh200.slurm
```
- Full/non-debug from local: use `sbatch` without `--wait`; if launching more than one job, submit them in the same SSH command:
```bash
ssh a5l.aip2.isambard 'cd ~/code/chess-rl && \
  squeue -u $USER -o "%i %t %M %R %j" && \
  sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    ./sbatch_train_chess_gh200.slurm'
```

Selection eval launchers (4 GPUs; follow the Slurm submission policy above):
- Pairs: `sbatch_eval_chess_select_pairs_gh200.slurm`
- Triples: `sbatch_eval_chess_select_triples_gh200.slurm`
- Subsets: `sbatch_eval_chess_select_subsets_gh200.slurm`
- Position-sweep: `sbatch_eval_chess_select_position_sweep_gh200.slurm`
- For a full selection-eval sweep from local, submit all eval jobs together:
```bash
ssh a5l.aip2.isambard 'cd ~/code/chess-rl && \
  squeue -u $USER -o "%i %t %M %R %j" && \
  sbatch ./sbatch_eval_chess_select_pairs_gh200.slurm && \
  sbatch ./sbatch_eval_chess_select_triples_gh200.slurm && \
  sbatch ./sbatch_eval_chess_select_subsets_gh200.slurm && \
  sbatch ./sbatch_eval_chess_select_position_sweep_gh200.slurm'
```

---

## Competition Submission Prompt (Legacy / Out-of-Date)

The example competition submission prompt template is kept intact under:
- `competition/submission/player_agents/chess_rl_chessr1_prompt.jinja`

It is **not aligned** with the current selection/action-masking prompts in this repo; do not use it as a
reference for selection training or restricted-moves evaluation.
