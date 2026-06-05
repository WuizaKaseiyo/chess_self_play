#!/usr/bin/env python3
"""
Build a restricted-moves ("selection") chess *training* parquet with per-row considered-moves sets.

Algorithm (per row; see `restricted_moves.md` + repo instructions):
1) Run N rollouts with considered_moves = full legal move list.
2) If the dataset ground-truth best move appears in any parsed rollout answer:
     - keep considered_moves = full legal list
   else:
     - let answer_moves be the unique parsed moves that are strict UCI and in legal_moves
     - rank answer_moves by μ (expected score preferred; fallback move_values) from low→high
     - construct considered_moves from answer_moves + move_best:
         - if k>5: sample 5 moves from answer_moves (deterministic per-row RNG), append move_best, shuffle
         - else: take all answer_moves, append move_best, shuffle
     - if k==0: fallback considered_moves = move_best + up to 5 random legal moves (≠ move_best), shuffle

Outputs:
- A VERL-format parquet where `prompt` is rendered from `select_prompt.jinja`.
- Stores `reward_model.considered_moves_uci` (list[str]) so reward code can enforce in-subset selection.
- Optionally writes a small JSONL sample for inspection.

Notes:
- This script uses vLLM for the rollout sampling step; it requires GPU availability.
- This script intentionally does not rely on stop sequences like `</uci_move>`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Ensure local namespace packages (e.g., `recipe/`) resolve when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipe.chess.reward_fn import _to_uci

_UCI_MOVE_TAG_RE = re.compile(r"<\s*uci_move\s*>(?P<ans>[\s\S]*?)<\s*/\s*uci_move\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class RowPayload:
    row_id: int
    fen: str
    legal_moves_uci: list[str]
    move_best_uci: str
    mu_map: dict[str, float]
    raw_row: dict[str, Any]


def _stable_int_hash(*parts: str, mod: int) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], "big") % int(mod)


def _normalize_moves(moves: Any) -> list[str]:
    if moves is None:
        return []
    if isinstance(moves, str):
        s = moves.strip().lower()
        return [s] if s else []
    out: list[str] = []
    try:
        for m in moves:
            s = str(m).strip().lower()
            if s:
                out.append(s)
    except Exception:
        return []
    return out


def _parse_mu_map(reward_model: dict[str, Any], *, row_id: int) -> dict[str, float]:
    mu_json = reward_model.get("move_expected_scores_json")
    if isinstance(mu_json, str) and mu_json.strip():
        mu_raw = json.loads(mu_json)
    else:
        mu_raw = mu_json
    mu_map: dict[str, float] = {}
    if isinstance(mu_raw, dict):
        for k, v in mu_raw.items():
            key = str(k).strip().lower()
            if not key:
                continue
            try:
                mu_map[key] = float(v)
            except Exception:
                continue

    if mu_map:
        return mu_map

    mv_json = reward_model.get("move_values_json")
    if isinstance(mv_json, str) and mv_json.strip():
        mv_raw = json.loads(mv_json)
    else:
        mv_raw = mv_json
    if isinstance(mv_raw, dict):
        for k, v in mv_raw.items():
            key = str(k).strip().lower()
            if not key:
                continue
            try:
                mu_map[key] = float(v)
            except Exception:
                continue

    if not mu_map:
        raise ValueError(f"Row {row_id}: empty mu_map (expected expected_scores or move_values).")
    return mu_map


def _load_rows(
    parquet_path: Path,
    *,
    limit_rows: Optional[int],
    num_shards: int,
    shard_idx: int,
) -> list[RowPayload]:
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if limit_rows is not None:
        if limit_rows < 0:
            raise ValueError("--limit_rows must be >= 0")
        rows = rows[: int(limit_rows)]

    out: list[RowPayload] = []
    for row in rows:
        rr = dict(row)
        rm = rr.get("reward_model") or {}
        if not isinstance(rm, dict):
            rm = {}
        ei = rr.get("extra_info") or {}
        if not isinstance(ei, dict):
            ei = {}

        row_id = int(ei.get("index"))
        if num_shards > 1 and (row_id % num_shards) != shard_idx:
            continue

        fen = str(rm.get("fen") or "").strip()
        legal_moves = _normalize_moves(rm.get("legal_moves_uci"))
        move_best = _to_uci(str(rm.get("ground_truth") or ""))

        if not fen:
            raise ValueError(f"Row {row_id}: empty FEN")
        if not legal_moves:
            raise ValueError(f"Row {row_id}: empty legal_moves_uci")
        if not move_best:
            raise ValueError(f"Row {row_id}: missing/invalid ground_truth UCI")
        if move_best not in set(legal_moves):
            raise ValueError(f"Row {row_id}: ground_truth {move_best} not in legal_moves_uci")

        mu_map = _parse_mu_map(rm, row_id=row_id)
        out.append(
            RowPayload(
                row_id=row_id,
                fen=fen,
                legal_moves_uci=legal_moves,
                move_best_uci=move_best,
                mu_map=mu_map,
                raw_row=rr,
            )
        )

    return out


def _load_template(template_path: Path) -> Any:
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    env = Environment(autoescape=False)
    return env.from_string(template_path.read_text(encoding="utf-8"))


def _render_prompt_text(template: Any, *, fen: str, legal_moves: list[str], considered_moves: list[str]) -> str:
    return str(
        template.render(
            FEN=fen,
            legal_moves_uci_list=legal_moves,
            considered_moves_uci_list=considered_moves,
        )
    )


def _parse_uci_move(text: str) -> tuple[Optional[str], str]:
    """Return (uci_or_none, reason).

    Reasons:
      - ""               : parsed ok with exactly one tag
      - "format_error"   : missing tag or multiple tags
      - "bad_move"       : tag present but payload not strict UCI
    """
    s = text or ""
    matches = list(_UCI_MOVE_TAG_RE.finditer(s))
    if not matches:
        return None, "format_error"
    ans = matches[0].group("ans") or ""
    pred = _to_uci(ans)
    if pred is None:
        return None, "bad_move"
    if len(matches) != 1:
        return pred, "format_error"
    return pred, ""


def _construct_considered_moves_from_rollouts(
    row: RowPayload,
    *,
    parsed_rollout_moves: list[Optional[str]],
    global_seed: int,
) -> tuple[list[str], dict[str, Any]]:
    """Return (considered_moves, stats)."""
    legal_set = set(row.legal_moves_uci)
    best = row.move_best_uci

    # Deterministic per-row RNG (stable across sharding / batch sizes).
    row_seed = _stable_int_hash(str(global_seed), str(row.row_id), "select-considered", mod=2**32)
    rng = np.random.default_rng(int(row_seed))

    # If the model ever produced the ground-truth best move, keep full legal (per spec).
    if any(mv == best for mv in parsed_rollout_moves if mv is not None):
        considered = list(row.legal_moves_uci)
        rng.shuffle(considered)
        return considered, {"considered_source": "full_legal_best_seen", "k_answer_moves": 0, "shuffled": True}

    answer_moves_set: set[str] = set()
    for mv in parsed_rollout_moves:
        if mv is None:
            continue
        if mv in legal_set:
            answer_moves_set.add(mv)

    answer_moves = sorted(answer_moves_set)
    k = len(answer_moves)

    missing_mu = 0

    def _mu(mv: str) -> float:
        nonlocal missing_mu
        if mv in row.mu_map:
            return float(row.mu_map[mv])
        missing_mu += 1
        # Expected-score maps should be dense over legal moves; treat missing as strictly-worst.
        return -1.0

    # Rank answer_moves by μ low→high (tie-break lexicographically).
    answer_moves_ranked = sorted(answer_moves, key=lambda m: (_mu(m), m))

    if k == 0:
        others = [m for m in row.legal_moves_uci if m != best]
        n_extra = min(5, len(others))
        sampled = rng.choice(others, size=n_extra, replace=False).tolist() if n_extra > 0 else []
        considered = sampled + [best]
        rng.shuffle(considered)
        return considered, {
            "considered_source": "fallback_random_legal",
            "k_answer_moves": 0,
            "missing_mu": int(missing_mu),
            "n_fallback_extra": int(n_extra),
        }

    if k > 5:
        sampled = rng.choice(answer_moves_ranked, size=5, replace=False).tolist()
        considered = sampled + [best]
        rng.shuffle(considered)
        return considered, {
            "considered_source": "sampled_answer_moves_plus_best",
            "k_answer_moves": int(k),
            "missing_mu": int(missing_mu),
            "sampled_k": 5,
        }

    considered = list(answer_moves_ranked) + [best]
    rng.shuffle(considered)
    return considered, {
        "considered_source": "all_answer_moves_plus_best",
        "k_answer_moves": int(k),
        "missing_mu": int(missing_mu),
        "sampled_k": int(k),
    }


def _atomic_write_parquet(table: pa.Table, *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(output_path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--template_path", default="recipe/chess/prompt_templates/select_prompt.jinja")

    ap.add_argument("--model", required=True, help="HF model id or local path for vLLM.")
    ap.add_argument("--num_rollouts", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--max_response_tokens", type=int, default=4096)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--max_num_seqs", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_rows", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sample_jsonl", default=None)
    ap.add_argument("--sample_n", type=int, default=20)
    args = ap.parse_args()

    input_path = Path(args.input_parquet)
    output_path = Path(args.output_parquet)
    template_path = Path(args.template_path)

    if not input_path.exists():
        raise SystemExit(f"Input parquet not found: {input_path}")
    if not template_path.exists():
        raise SystemExit(f"Template not found: {template_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {output_path} (pass --overwrite)")

    num_shards = int(args.num_shards)
    shard_idx = int(args.shard_idx)
    if num_shards <= 0:
        raise SystemExit("--num_shards must be > 0")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise SystemExit(f"--shard_idx must be in [0,{num_shards - 1}]")

    template = _load_template(template_path)

    rows = _load_rows(
        input_path,
        limit_rows=args.limit_rows,
        num_shards=num_shards,
        shard_idx=shard_idx,
    )
    if not rows:
        raise SystemExit("No rows to process (check --limit_rows/--num_shards/--shard_idx).")

    print(
        f"[load] rows={len(rows)} input={input_path} shard={shard_idx}/{num_shards} "
        f"limit_rows={args.limit_rows}"
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=int(args.tensor_parallel_size),
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_num_seqs=int(args.max_num_seqs),
    )

    base_sampling_kwargs = dict(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=-1,
        min_p=0.0,
        max_tokens=int(args.max_response_tokens),
        repetition_penalty=1.0,
        detokenize=True,
    )

    # Sample JSONL selection (deterministic but cheap).
    py_rng = random.Random(int(args.seed) + 1337 + 10_000 * shard_idx)
    sample_rows = set(py_rng.sample(range(len(rows)), k=min(int(args.sample_n), len(rows)))) if rows else set()
    samples: list[dict[str, Any]] = []

    stats = {
        "n_rows": 0,
        "n_full_legal_best_seen": 0,
        "n_all_answer_moves_plus_best": 0,
        "n_sampled_answer_moves_plus_best": 0,
        "n_fallback_random_legal": 0,
        "n_missing_mu_total": 0,
        "n_k0": 0,
    }
    parse_reason_counts: dict[str, int] = {}

    out_rows: list[dict[str, Any]] = []

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise SystemExit("--batch_size must be > 0")

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]

        # Rollout prompts always use full legal considered set.
        messages_list: list[list[dict[str, str]]] = []
        for r in batch_rows:
            prompt_text = _render_prompt_text(
                template,
                fen=r.fen,
                legal_moves=r.legal_moves_uci,
                considered_moves=r.legal_moves_uci,
            )
            messages_list.append([{"role": "user", "content": prompt_text}])

        prompt_token_ids: list[list[int]] = []
        for messages in messages_list:
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            if not isinstance(ids, list):
                raise TypeError(f"Expected token id list from apply_chat_template, got {type(ids)}")
            if len(ids) > int(args.max_prompt_tokens):
                raise ValueError(f"Prompt is {len(ids)} tokens (max={int(args.max_prompt_tokens)}).")
            prompt_token_ids.append([int(x) for x in ids])

        vllm_inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids]

        sampling_params: list[SamplingParams] = []
        for r in batch_rows:
            seed_i = _stable_int_hash(str(args.seed), str(r.row_id), "select-train-rollouts", mod=2**31)
            sampling_params.append(SamplingParams(seed=int(seed_i), n=int(args.num_rollouts), **base_sampling_kwargs))

        outs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
        if len(outs) != len(batch_rows):
            raise RuntimeError(f"Expected {len(batch_rows)} outputs, got {len(outs)}")

        for i, (r, out) in enumerate(zip(batch_rows, outs, strict=True)):
            if len(out.outputs) != int(args.num_rollouts):
                raise RuntimeError(f"Row {r.row_id}: expected n={int(args.num_rollouts)} outputs, got {len(out.outputs)}")

            parsed_moves: list[Optional[str]] = []
            for o in out.outputs:
                mv, reason = _parse_uci_move(o.text)
                if reason:
                    parse_reason_counts[reason] = parse_reason_counts.get(reason, 0) + 1
                    # Treat any format violation as an unqualified rollout for dataset construction.
                    # We still record the reason counts for debugging.
                    parsed_moves.append(None)
                else:
                    parsed_moves.append(mv)

            considered_moves, meta = _construct_considered_moves_from_rollouts(
                r,
                parsed_rollout_moves=parsed_moves,
                global_seed=int(args.seed),
            )
            stats["n_rows"] += 1
            stats["n_missing_mu_total"] += int(meta.get("missing_mu", 0) or 0)
            src = str(meta.get("considered_source") or "")
            stats_key = f"n_{src}"
            if stats_key in stats:
                stats[stats_key] += 1
            if src == "fallback_random_legal":
                stats["n_k0"] += 1

            # Invariants (hard asserts: keep the dataset clean).
            legal_set = set(r.legal_moves_uci)
            considered_set = set(considered_moves)
            if not considered_moves:
                raise ValueError(f"Row {r.row_id}: empty considered_moves after construction.")
            if not considered_set.issubset(legal_set):
                illegal = sorted(considered_set - legal_set)[:10]
                raise ValueError(f"Row {r.row_id}: considered_moves contains illegal moves: {illegal}")
            if r.move_best_uci not in considered_set:
                raise ValueError(f"Row {r.row_id}: move_best missing from considered_moves.")

            rr = dict(r.raw_row)
            rm = rr.get("reward_model") or {}
            if not isinstance(rm, dict):
                rm = {}
            rm["considered_moves_uci"] = list(considered_moves)
            rr["reward_model"] = rm

            rr["prompt"] = [
                {
                    "role": "user",
                    "content": _render_prompt_text(
                        template,
                        fen=r.fen,
                        legal_moves=r.legal_moves_uci,
                        considered_moves=list(considered_moves),
                    ),
                }
            ]
            out_rows.append(rr)

            if (start + i) in sample_rows:
                samples.append(
                    {
                        "row_id": r.row_id,
                        "fen": r.fen,
                        "move_best": r.move_best_uci,
                        "legal_moves_uci": r.legal_moves_uci,
                        "considered_moves_uci": list(considered_moves),
                        "parsed_rollout_moves": parsed_moves,
                        "considered_source": src,
                        "prompt": rr["prompt"][0]["content"],
                    }
                )

        if (start == 0) or ((start + batch_size) % (batch_size * 10) == 0):
            print(f"[progress] processed_rows={len(out_rows)}/{len(rows)}")

    out_table = pa.Table.from_pylist(out_rows)
    _atomic_write_parquet(out_table, output_path=output_path)
    print(f"[done] wrote={output_path} rows={out_table.num_rows} shard={shard_idx}/{num_shards}")
    print(f"[stats] {json.dumps(stats, sort_keys=True)}")
    if parse_reason_counts:
        print(f"[parse] {json.dumps(parse_reason_counts, sort_keys=True)}")

    if args.sample_jsonl:
        _write_jsonl(Path(args.sample_jsonl), samples)
        print(f"[sample] wrote={args.sample_jsonl} rows={len(samples)}")


if __name__ == "__main__":
    main()
