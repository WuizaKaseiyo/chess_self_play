#!/usr/bin/env python3
from __future__ import annotations

"""
Build chess SFT datasets from the in-repo hard parquet.

This script produces *multi-turn* SFT parquets consumable by veRL's FSDP SFTTrainer
(`verl/trainer/fsdp_sft_trainer.py`) via `MultiTurnSFTDataset`:

  - output column: `messages` = prompt_messages + [{"role": "assistant", "content": <response>}]

The response is constructed to satisfy the chess reward function contract:
  <guess> ... </guess><think> ... </think><uci_move> ... </uci_move>

This script is a legacy SFT-ablation helper. The current RL recipe uses a response-side forced prefix
of `<guess> {move} </guess>` (without injecting `<think>`). For SFT target construction we keep
prefix templates that include `<think>` so we can splice in non-empty analysis text.

All variants keep the `<guess>` block because it is format-required by the reward parser.

Note: "success" here is defined exactly as the repo evaluates it today:
  `acc == 1.0` from `recipe/chess/reward_fn.py::compute_score_batch`.
We validate every constructed example with the reward function and drop any failures.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from recipe.chess.reward_fn import compute_score_batch


@dataclass(frozen=True)
class Variant:
    variant_id: str
    forced_prefix_template: str
    strip_phrase_template: str | None


VARIANTS: dict[str, Variant] = {
    # (A) forced_prefix="<guess> {move} </guess>\n<think>\n"
    #     keep the guess line in the target (guess is format-required)
    "A": Variant(
        variant_id="A",
        forced_prefix_template="<guess> {move} </guess>\n<think>\n",
        strip_phrase_template=None,
    ),
    # (B) forced_prefix="<guess> {move} </guess>\n<think>\n"
    #     keep the guess line in the target (same as A; kept for compatibility with older naming)
    "B": Variant(
        variant_id="B",
        forced_prefix_template="<guess> {move} </guess>\n<think>\n",
        strip_phrase_template=None,
    ),
    # (C) same as (A) for now (kept for compatibility with older ablation naming).
    "C": Variant(
        variant_id="C",
        forced_prefix_template="<guess> {move} </guess>\n<think>\n",
        strip_phrase_template=None,
    ),
    # (D) legacy slot: still produce a valid completion under the current required-guess reward gate.
    "D": Variant(
        variant_id="D",
        forced_prefix_template="<guess> {move} </guess>\n<think>\n",
        strip_phrase_template=None,
    ),
    # (E) keep the guess line and start the think block on the next line.
    "E": Variant(
        variant_id="E",
        forced_prefix_template="<guess> {move} </guess>\n<think>\n",
        strip_phrase_template=None,
    ),
}


def _json_load_maybe(x: Any) -> Any:
    if isinstance(x, str) and x:
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _stable_u32_seed(*parts: str) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:4], "little", signed=False)


def _select_move_map(reward_model: dict[str, Any]) -> tuple[dict[str, float], str]:
    """Select the move→value map exactly as in RL forced-move sampling.

    Mirrors `verl/trainer/ppo/ray_trainer.py::_sample_forced_move`:
      - Prefer `move_expected_scores_json` if truthy, else fall back to `move_values_json`.
      - Parse JSON (stored as a string in the parquet) into a dict[str, float].
    """
    raw_expected = reward_model.get("move_expected_scores_json") if isinstance(reward_model, dict) else None
    raw_values = reward_model.get("move_values_json") if isinstance(reward_model, dict) else None

    raw = raw_expected or raw_values
    source_key = "move_expected_scores_json" if raw_expected else ("move_values_json" if raw_values else "none")
    raw = _json_load_maybe(raw)

    if not isinstance(raw, dict) or not raw:
        return {}, "none"

    out: dict[str, float] = {}
    for m, v in raw.items():
        if m is None:
            continue
        mv = str(m).strip().lower()
        if not mv:
            continue
        try:
            out[mv] = float(v)
        except Exception:
            continue
    if not out:
        return {}, "none"
    return out, source_key


def _sample_forced_move(
    reward_model: dict[str, Any],
    *,
    seed: int,
    temperature: float,
) -> tuple[str, float, str]:
    """
    Sample a `{move}` consistent with `verl/trainer/ppo/ray_trainer.py::_sample_forced_move`.

    Returns (move_uci, value, source_key).
    """
    gt = str(reward_model.get("ground_truth", "") or "").strip().lower()
    move_map, source_key = _select_move_map(reward_model)
    gt_val = float(move_map.get(gt, 0.0)) if gt and gt in move_map else 0.0

    filtered: list[tuple[str, float]] = []
    for m, v in move_map.items():
        if v <= 0:
            continue
        filtered.append((m, float(v)))

    if not filtered:
        # Match RL: if the filter empties the map, fall back to GT (if present).
        return gt, gt_val, "ground_truth"

    moves, vals = zip(*filtered)
    moves_arr = np.asarray(moves, dtype=object)
    vals_arr = np.asarray(vals, dtype=np.float64)

    temp = max(float(temperature), 1e-6)
    weights = np.power(np.clip(vals_arr, 1e-6, None), 1.0 / temp)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        probs = np.full_like(weights, 1.0 / float(len(weights)), dtype=np.float64)
    else:
        probs = weights / weights.sum()

    rng = np.random.default_rng(int(seed))
    idx = int(rng.choice(len(moves_arr), p=probs))
    return str(moves_arr[idx]), float(vals_arr[idx]), source_key


def _sample_forced_moves(
    reward_model: dict[str, Any],
    *,
    seed: int,
    temperature: float,
    count: int,
) -> tuple[list[str], list[float], str]:
    """Multi-sample version of `_sample_forced_move` with RL-like uniqueness.

    Mirrors `verl/trainer/ppo/ray_trainer.py::_sample_forced_moves_for_group`:
      - Best-effort uniqueness: sample without replacement when possible.
      - Otherwise sample with replacement.
    """
    count = int(count or 0)
    if count <= 0:
        return [], [], "none"

    gt = str(reward_model.get("ground_truth", "") or "").strip().lower()
    move_map, source_key = _select_move_map(reward_model)
    gt_val = float(move_map.get(gt, 0.0)) if gt and gt in move_map else 0.0

    if not move_map:
        # Degenerate case: no distribution → force GT (duplicates unavoidable).
        if gt:
            return [gt] * count, [gt_val] * count, "ground_truth"
        return [""] * count, [0.0] * count, "none"

    # Remove zero/negative valued moves; fall back to GT if the filter empties the list.
    filtered = [(m, float(v)) for m, v in move_map.items() if float(v) > 0.0]
    if filtered:
        moves, vals = zip(*filtered)
    else:
        if gt:
            return [gt] * count, [gt_val] * count, source_key
        moves, vals = zip(*[(m, float(v)) for m, v in move_map.items()])

    moves_arr = np.asarray(moves, dtype=object)
    vals_arr = np.asarray(vals, dtype=np.float64)
    orig_vals = vals_arr.copy()

    temp = max(float(temperature), 1e-6)
    vals_arr = np.clip(vals_arr, 1e-6, None)
    weights = np.power(vals_arr, 1.0 / temp)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        probs = np.full_like(weights, 1.0 / float(len(weights)), dtype=np.float64)
    else:
        probs = weights / weights.sum()

    replace = bool(count > len(moves_arr))
    rng = np.random.default_rng(int(seed))
    idxs = rng.choice(len(moves_arr), size=count, replace=replace, p=probs)

    idxs_list = np.asarray(idxs).tolist()
    out_moves = [str(moves_arr[int(i)]) for i in idxs_list]
    out_vals = [float(orig_vals[int(i)]) for i in idxs_list]
    return out_moves, out_vals, source_key


def _softmax_weights(values: np.ndarray, beta: float) -> np.ndarray:
    beta = max(float(beta), 1e-6)
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return v
    v = np.where(np.isfinite(v), v, -np.inf)
    # Stable softmax.
    m = np.max(v)
    if not np.isfinite(m):
        return np.full_like(v, 1.0 / float(len(v)), dtype=np.float64)
    exps = np.exp((v - m) / beta)
    s = float(np.sum(exps))
    if not np.isfinite(s) or s <= 0.0:
        return np.full_like(v, 1.0 / float(len(v)), dtype=np.float64)
    return exps / s


def _build_full_response(
    *,
    ground_truth: str,
    forced_move: str,
    variant: Variant,
) -> str:
    # Keep the response short but non-empty inside <think>.
    reasoning_tail = (
        "We compare candidate moves, verify tactics, and choose the best continuation.\n"
        "Now we play the best move from the legal list."
    )

    gt = str(ground_truth or "").strip().lower()
    mv = str(forced_move or "").strip().lower()

    prefix = variant.forced_prefix_template.format(move=mv) if variant.forced_prefix_template else ""
    if not prefix:
        prefix = "<think>\n"

    pre, sep, post = prefix.partition("<think>")
    if not sep:
        raise ValueError(
            f"variant {variant.variant_id}: forced_prefix_template must contain '<think>', got {prefix!r}"
        )

    # Keep any leading content (e.g., a `<guess>...</guess>\n` line) exactly as-is.
    return f"{pre}<think>{post}{reasoning_tail}\n</think><uci_move> {gt} </uci_move>"


def _strip_boilerplate(
    *,
    full_response: str,
    forced_move: str,
    variant: Variant,
) -> str:
    phrase_tmpl = variant.strip_phrase_template
    if not phrase_tmpl:
        return full_response

    mv = str(forced_move or "").strip().lower()
    phrase = phrase_tmpl.format(move=mv)

    # Historical note: older variants stripped a leading guess line like:
    #   "<guess> {move} </guess>\n"
    # This is no longer allowed because `<guess>` is format-required by the reward parser.
    if full_response.startswith(phrase):
        return full_response[len(phrase) :]

    # Fallback: strip first occurrence anywhere (best-effort). This shouldn't happen
    # for our constructed responses, but keeps the script robust to minor formatting drift.
    return full_response.replace(phrase, "", 1)


def _iter_rows(path: str, limit: int | None) -> Iterable[dict[str, Any]]:
    table = pq.read_table(path)
    if limit is not None:
        table = table.slice(0, limit)
    for row in table.to_pylist():
        yield row


def _validate_success(
    reward_model: dict[str, Any],
    response: str,
) -> tuple[bool, dict[str, Any]]:
    scored = compute_score_batch([reward_model], [response], chess_reward_fn="winrate", logit_eps=1e-6)
    info = dict(scored[0] or {})
    ok = float(info.get("acc", 0.0)) == 1.0 and (info.get("penalty_reason", "") or "") == ""
    return ok, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_parquet", default="data/chess_puzzles/train_hard.parquet")
    parser.add_argument("--out_parquet", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS.keys()), required=True)
    parser.add_argument("--limit", type=int, default=None, help="Debug: only process the first N rows")
    parser.add_argument(
        "--num_move_samples",
        type=int,
        default=1,
        help="Number of `{move}` samples per prompt row. Use 8 to mirror RL rollout.n-style exploration groups.",
    )
    parser.add_argument(
        "--sample_ordering",
        choices=["shuffle", "no_shuffle"],
        default="no_shuffle",
        help="When num_move_samples > 1: shuffle fully mixes examples across prompts; "
        "no_shuffle keeps the per-prompt group contiguous.",
    )
    parser.add_argument(
        "--sft_weighting",
        choices=["awr", "uniform", "best_only"],
        default="uniform",
        help="Per-example weight scheme for multi-sample datasets. Output column is `sft_weight`.",
    )
    parser.add_argument(
        "--awr_beta",
        type=float,
        default=2.0,
        help="Beta for AWR-style per-prompt softmax weights. Mirrors forced_prefix.beta in RL.",
    )
    parser.add_argument(
        "--move_temperature",
        type=float,
        default=2.0,
        help="Forced move sampling temperature (power-law). Mirrors forced_prefix.move_temperature.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_examples", type=int, default=3)
    args = parser.parse_args()

    variant = VARIANTS[str(args.variant)]
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, Any]] = []
    shown = 0
    dropped = 0

    for row in _iter_rows(args.in_parquet, args.limit):
        prompt_msgs = row.get("prompt", None)
        reward_model = dict(row.get("reward_model") or {})

        if not isinstance(prompt_msgs, list) or not prompt_msgs:
            dropped += 1
            continue

        # Deterministic per-row seed: global seed + stable id from (fen, index).
        fen = str(reward_model.get("fen", "") or "")
        idx = str((row.get("extra_info") or {}).get("index", "") or "")

        gt = str(reward_model.get("ground_truth", "") or "").strip().lower()
        if not gt:
            dropped += 1
            continue

        num_samples = int(args.num_move_samples or 1)
        variant_base = (
            f"{variant.variant_id}_sample_{args.sample_ordering}" if num_samples > 1 else str(variant.variant_id)
        )
        if num_samples <= 1:
            row_seed = _stable_u32_seed(str(args.seed), fen, idx, "forced_move")
            forced_moves = []
            forced_values = []
            forced_src = "none"
            mv, mv_val, mv_src = _sample_forced_move(
                reward_model,
                seed=row_seed,
                temperature=float(args.move_temperature),
            )
            forced_moves.append(mv)
            forced_values.append(float(mv_val))
            forced_src = mv_src
        else:
            row_seed = _stable_u32_seed(str(args.seed), fen, idx, "forced_moves")
            forced_moves, forced_values, forced_src = _sample_forced_moves(
                reward_model,
                seed=row_seed,
                temperature=float(args.move_temperature),
                count=num_samples,
            )

        # Compute per-example weights for this prompt group.
        weights: list[float] = []
        weight_mode = str(args.sft_weighting or "uniform").strip().lower()
        if weight_mode == "uniform":
            weights = [1.0] * len(forced_moves)
        elif weight_mode == "best_only":
            move_map, _src = _select_move_map(reward_model)
            if move_map:
                best_v = max(move_map.values())
                # Include ties deterministically.
                best_moves = {m for m, v in move_map.items() if abs(float(v) - float(best_v)) <= 1e-12}
                weights = [1.0 if m in best_moves else 0.0 for m in forced_moves]
            else:
                weights = [1.0 if (m and m == gt) else 0.0 for m in forced_moves]
        elif weight_mode == "awr":
            vals = np.asarray(forced_values, dtype=np.float64)
            p = _softmax_weights(vals, beta=float(args.awr_beta))
            w = float(len(vals)) * p
            weights = [float(x) for x in w.tolist()]
        else:
            raise ValueError(f"Unknown sft_weighting: {args.sft_weighting!r}")

        for sample_idx, (forced_move, forced_value, sft_weight) in enumerate(zip(forced_moves, forced_values, weights)):
            full = _build_full_response(ground_truth=gt, forced_move=forced_move, variant=variant)
            stripped = _strip_boilerplate(full_response=full, forced_move=forced_move, variant=variant)

            ok, info = _validate_success(reward_model, stripped)
            if not ok:
                dropped += 1
                continue

            messages = list(prompt_msgs) + [{"role": "assistant", "content": stripped}]

            rows_out.append(
                {
                    "messages": messages,
                    "variant": variant.variant_id,
                    "variant_base": variant_base,
                    "forced_prefix_template": variant.forced_prefix_template,
                    "strip_phrase_template": variant.strip_phrase_template,
                    "forced_move": forced_move,
                    "forced_move_value": float(forced_value),
                    "forced_move_source": forced_src,
                    "forced_move_sample_idx": int(sample_idx),
                    "ground_truth": gt,
                    "move_temperature": float(args.move_temperature),
                    "num_move_samples": int(num_samples),
                    "sample_ordering": str(args.sample_ordering),
                    "sft_weighting": weight_mode,
                    "sft_weight": float(sft_weight),
                    "awr_beta": float(args.awr_beta),
                    "reward_debug": info,
                    # Grouping keys for debugging/analysis.
                    "group_fen": fen,
                    "group_index": int(row.get("extra_info", {}).get("index", -1) or -1),
                }
            )

            if shown < int(args.print_examples):
                shown += 1
                print("=" * 100)
                print(
                    f"[example {shown}] variant={variant.variant_id} forced_move={forced_move} gt={gt} "
                    f"sample_idx={sample_idx} sft_weight={float(sft_weight):.4f}"
                )
                print("[full_response]")
                print(full)
                if stripped != full:
                    print("-" * 100)
                    print("[stripped_response]")
                    print(stripped)

    # Optional global shuffle (after per-prompt expansion) for offline SFT variants.
    if int(args.num_move_samples or 1) > 1 and str(args.sample_ordering) == "shuffle" and rows_out:
        rng = np.random.default_rng(int(args.seed))
        perm = rng.permutation(len(rows_out)).tolist()
        rows_out = [rows_out[i] for i in perm]

    table = pa.Table.from_pylist(rows_out)
    pq.write_table(table, str(out_path))

    print(
        f"[build_chess_sft_prefix_dataset] in={args.in_parquet} variant={variant.variant_id} "
        f"rows_out={len(rows_out)} dropped={dropped} num_move_samples={int(args.num_move_samples or 1)} "
        f"ordering={args.sample_ordering} weighting={args.sft_weighting} -> {out_path}"
    )


if __name__ == "__main__":
    main()
