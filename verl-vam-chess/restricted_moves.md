# Restricted Moves (“Selection”) Chess: Contract and Evaluation

This repo’s current research framing for chess is **restricted-moves (“selection”)** via **verbalized action
masking**:
the model is shown an explicit candidate-move shortlist (`allowed_moves`) and must select the best move
*from that list*.

This document is the source-of-truth for:
- the selection prompt + output contract (hard gates),
- what “correct” means when the globally-best move may be absent from the candidate set,
- the selection evaluation entrypoints (pairs/triples/subsets/position-sweep) and their sanity checks.

If you are editing prompt templates, reward parsing, or selection evaluators, read this end-to-end first.

---

## 1) Concept + naming: restricted moves = selection from a candidate subset

Each prompt provides:
- `FEN`: the current chess position,
- `legal_moves_uci_list`: the full legal move list (UCI),
- `allowed_moves`: the candidate list the model is allowed to choose from (UCI).

Naming in this repo (important for keeping training/eval consistent):
- **Prompt text**: calls the candidate list `allowed_moves`.
- **Jinja variable name** (when rendering the selection template): `considered_moves_uci_list`.
- **Dataset field name** (stored on selection datasets): `reward_model.considered_moves_uci`.

Small-legal note:
- The canonical selection prompt literally uses the text `allowed_moves`.
- The later `select_prompt_small_legal.jinja` variant does **not**; it presents only one move list labeled as
  legal moves while still relying on `reward_model.considered_moves_uci` as the operative candidate set at runtime.

The candidate list must always be a subset of legal moves:
`allowed_moves ⊆ legal_moves`.

---

## 2) Prompt + output contract (strict)

Canonical selection prompt template:
- `recipe/chess/prompt_templates/select_prompt.jinja`

### 2.1 Required output format

The model must output exactly:

```text
<think> ... </think><uci_move> ... </uci_move>
```

Rules (hard gates):
- The `<uci_move>` payload must be **strict UCI** (e.g., `e2e4`, `g1f3`, `a7a8q`).
- The `<uci_move>` payload must appear **verbatim** in `allowed_moves` (exact string match).
- If `allowed_moves` contains exactly one move, the model **must output that move**.
- Any output that violates the format or selects a move outside `allowed_moves` is treated as incorrect and is
  logged as a compliance failure.

Parsing / normalization:
- Reward parsing is implemented in `recipe/chess/reward_fn.py` (extract `<uci_move>...</uci_move>`, normalize to UCI).

---

## 3) Correctness definition (subset-relative μ-best)

Given a candidate set `S` (the `allowed_moves` list):

1) Define the target move as the best move **within the candidate set**:
   - `target = argmax_{m ∈ S} μ(m)`
2) Parse the model output:
   - Missing / malformed `<uci_move>` → incorrect.
   - Parsed move not in `S` → incorrect (out-of-subset).
   - Otherwise correct iff `pred == target`.

μ (“move quality”) sources:
- Preferred: `reward_model.move_expected_scores_json` (JSON `{uci: float}` in `[0, 1]`)
- Fallback: `reward_model.move_values_json` (JSON `{uci: float}` in `[0, 1]`)

Deterministic tie-break (used consistently when selecting μ-best):
1) higher μ wins
2) if μ ties, lexicographically smaller UCI wins

This definition is intentionally **subset-relative**:
if the globally-best legal move is absent from `S`, the task is still meaningful (“best among the candidates”).

---

## 4) Dataset contract (selection framing)

### 4.1 Searchless-chess (puzzle-style)

Base source parquets (engine-scored legal moves):
- `data/chess_puzzles/train.parquet`
- `data/chess_puzzles/train_hard.parquet`
- `data/chess_puzzles/test.parquet`

Selection-framed full-legal parquets (recommended offline base; `allowed_moves == legal_moves`):
- `data/chess_puzzles_select_v4/{train,train_hard,test}.parquet`

Build v4 selection parquets:
- Script: `scripts/build_chess_select_train_dataset_v4.py`

### 4.2 Chess‑R1 aligned (prompt-variant rewrites)

For Chess‑R1 aligned data, we treat “selection prompts” as a **prompt rewrite** (no rescoring):
- Input: `data/chess_puzzles_chessr1_aligned_sharded/`
- Output (selection prompt variant): `data/chess_puzzles_chessr1_aligned_sharded_ours/`
- Script: `scripts/rewrite_chess_prompts_from_template.py`

The “ours” variant sets:
- `reward_model.considered_moves_uci = reward_model.legal_moves_uci`
so selection enforcement is well-defined (`allowed_moves == legal_moves`).

### 4.3 Row schema (reward payload)

Selection rows are VERL-format dicts with:
- `prompt`: list of chat messages (usually one user message)
- `reward_model`: dict consumed by `recipe/chess/reward_fn.py`
- `extra_info`: traceability metadata

Minimum `reward_model` fields expected by selection reward/eval:
- `fen`: FEN string
- `ground_truth`: labeled move (uci, lowercase; used for diagnostics/coverage, not the μ-target)
- `legal_moves_uci`: ordered list[str] of legal moves (uci)
- `considered_moves_uci`: list[str] candidate list (uci; what the prompt calls `allowed_moves`)
- `move_expected_scores_json` and/or `move_values_json`: JSON string maps `{uci: float}`

Stable row id used throughout selection eval caching:
- `extra_info.index`

---

## 5) Reward function enforcement (why “in-subset” is a hard gate)

Reward code: `recipe/chess/reward_fn.py`

Always enforced (all reward modes):
- Exactly one `<uci_move>...</uci_move>` span must be present (missing/multiple = format error).
- Payload must normalize to strict UCI.

Selection-specific enforcement:
- For selection prompts (i.e., prompts that include `allowed_moves`), the reward function enforces that the
  predicted move must be in `reward_model.considered_moves_uci` (the candidate list).
- If prompt text is unavailable at runtime, the reward function falls back to treating the presence of
  `reward_model.considered_moves_uci` as the “selection prompt” signal.
- For the small-legal prompt variant, trainer-side `use_considered_moves_uci=True` wiring is what keeps reward
  enforcement aligned even though the prompt text omits the literal `allowed_moves` token.

Practical implication:
- If you update `recipe/chess/prompt_templates/select_prompt.jinja`, regenerate any stored training prompts
  (datasets that store `prompt`) so training and evaluation stay aligned.

---

## 6) Evaluation pipelines (selection framing)

All selection evaluators:
- regenerate prompts per `(row, candidate set)` from the selection template,
- enforce a **one-candidate sanity check** before running expensive evaluation,
- are cached/restartable (resume from existing `results_shard*.jsonl`),
- log compliance signals (format errors, out-of-subset, etc.).

### 6.1 k=2 exhaustive pairs (order bias)

Files:
- Eval: `scripts/eval_chess_select_pairs.py`
- Analysis: `scripts/analyze_chess_select_pairs.py`
- Cluster launcher: `sbatch_eval_chess_select_pairs_gh200.slurm`

### 6.2 k=3 sampled triples (permutation-controlled bias + gap)

Files:
- Eval: `scripts/eval_chess_select_triples.py`
- Analysis: `scripts/analyze_chess_select_triples.py`
- Cluster launcher: `sbatch_eval_chess_select_triples_gh200.slurm`

### 6.3 k>=1 sampled subsets (hardness vs pass@k)

Files:
- Eval: `scripts/eval_chess_select_subsets.py`
- Analysis: `scripts/analyze_chess_select_subsets.py`
- Cluster launcher: `sbatch_eval_chess_select_subsets_gh200.slurm`

### 6.4 Position-sweep (presentation-order probe)

Files:
- Eval: `scripts/eval_chess_select_position_sweep.py`
- Analysis: `scripts/analyze_chess_select_position_sweep.py`
- Cluster launcher: `sbatch_eval_chess_select_position_sweep_gh200.slurm`

---

## 7) How to run (sanity-first)

### 7.1 Local one-candidate sanity gate

Local sanity check example (adjust device selection as needed):
- conda env `verl`
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- set `CUDA_VISIBLE_DEVICES` to an available GPU (example below uses `3`)

Example (pairs sanity gate):
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 conda run -n verl \
  python -m scripts.eval_chess_select_pairs \
    --model Qwen/Qwen2.5-7B-Instruct \
    --parquet data/chess_puzzles/test.parquet \
    --template_path recipe/chess/prompt_templates/select_prompt.jinja \
    --limit_rows 5 --samples_per_pair 8 \
    --sanity_only
```

### 7.2 GH200 (Slurm) examples

Evaluate pairs:
```bash
sbatch --wait --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,LIMIT_ROWS=100,SAMPLES_PER_PAIR=8,SEED=0 \
  ./sbatch_eval_chess_select_pairs_gh200.slurm
```

---

## 8) Training pointers

For training (offline fixed dataset and online engine-opponent data source), see:
- `AGENTS.md` (workflow index)
- `iterative.md` (allowed-move elimination + online play vs engine opponent)
