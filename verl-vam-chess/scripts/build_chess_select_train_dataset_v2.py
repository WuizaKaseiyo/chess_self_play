#!/usr/bin/env python3
"""
Build a restricted-moves ("selection") chess *training* parquet with per-row candidate sets
aligned to the AIcrowd / python-chess legal-move order.

v2 is a legacy mixture builder retained for reproduction/ablation.
For current workflows, prefer the full-legal v4 base dataset (`scripts/build_chess_select_train_dataset_v4.py`)
and apply iterative action masking during training (`iterative.md`).

v2 behavior:
- Candidate lists (`reward_model.considered_moves_uci`) are ALWAYS ordered as subsequences of
  `reward_model.legal_moves_uci` (no shuffling by default).
- The selection target is the μ-best move:
    μ source: `move_expected_scores_json` preferred; else `move_values_json`.
- Per original row, we generate a bounded set of derived rows:
  - full legal (required): considered_moves = legal_moves (same order)
  - hard negatives (optional): {target} ∪ (μ-near-best negatives) ∪ (optional model-mined negatives)
  - coverage blocks (optional): {target} ∪ one or more blocks of remaining moves (μ-stratified)

This script intentionally does *not* rely on stop sequences like `</uci_move>`.

Notes:
- This script is CPU-only unless model-mined negatives are enabled (`--mine_model` and `--K_mine>0`),
  in which case it uses vLLM and requires GPU availability.
- The output parquet uses stable derived row ids by overwriting `extra_info.index` and storing the original
  id under `extra_info.source_index`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment

# vLLM is optional (only needed for mined negatives).
try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    _VLLM_AVAILABLE = True
except Exception:
    AutoTokenizer = None  # type: ignore[assignment]
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
    _VLLM_AVAILABLE = False

# Ensure local namespace packages (e.g., `recipe/`) resolve when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipe.chess.reward_fn import _to_uci

_UCI_MOVE_TAG_RE = re.compile(r"<\s*uci_move\s*>(?P<ans>[\s\S]*?)<\s*/\s*uci_move\s*>", re.IGNORECASE)

BUILDER_VERSION = "select-v2-2026-01-16"


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


def _dedup_preserve_order(moves: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in moves:
        mm = str(m).strip().lower()
        if not mm:
            continue
        if mm in seen:
            continue
        seen.add(mm)
        out.append(mm)
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


def _best_move_by_mu(mu_map: dict[str, float], moves: Sequence[str]) -> str:
    best_move = ""
    best_mu = -float("inf")
    for mv in moves:
        key = str(mv).strip().lower()
        mu = float(mu_map.get(key, -float("inf")))
        if (mu > best_mu) or (mu == best_mu and (not best_move or key < best_move)):
            best_move = key
            best_mu = mu
    if not best_move:
        raise ValueError("Empty candidate set when selecting μ-target.")
    return best_move


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
        # Multiple tags are treated as format error; exclude from mining.
        return pred, "format_error"
    return pred, ""


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


@dataclass(frozen=True)
class BaseRow:
    source_index: int
    fen: str
    legal_moves_uci: list[str]  # ordered
    ground_truth_uci: str  # normalized; may differ from μ-best
    mu_map: dict[str, float]
    raw_row: dict[str, Any]


def _load_base_rows(
    parquet_path: Path,
    *,
    limit_rows: Optional[int],
    num_shards: int,
    shard_idx: int,
) -> list[BaseRow]:
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if limit_rows is not None:
        if limit_rows < 0:
            raise ValueError("--limit_rows must be >= 0")
        rows = rows[: int(limit_rows)]

    out: list[BaseRow] = []
    for row in rows:
        rr = dict(row)
        rm = rr.get("reward_model") or {}
        if not isinstance(rm, dict):
            rm = {}
        ei = rr.get("extra_info") or {}
        if not isinstance(ei, dict):
            ei = {}

        source_index = int(ei.get("index"))
        if num_shards > 1 and (source_index % num_shards) != shard_idx:
            continue

        fen = str(rm.get("fen") or "").strip()
        legal_moves = _normalize_moves(rm.get("legal_moves_uci"))
        gt = _to_uci(str(rm.get("ground_truth") or ""))

        if not fen:
            raise ValueError(f"Row {source_index}: empty FEN")
        if not legal_moves:
            raise ValueError(f"Row {source_index}: empty legal_moves_uci")
        if not gt:
            raise ValueError(f"Row {source_index}: missing/invalid ground_truth UCI")
        if gt not in set(legal_moves):
            raise ValueError(f"Row {source_index}: ground_truth {gt} not in legal_moves_uci")

        mu_map = _parse_mu_map(rm, row_id=source_index)
        out.append(
            BaseRow(
                source_index=source_index,
                fen=fen,
                legal_moves_uci=legal_moves,
                ground_truth_uci=gt,
                mu_map=mu_map,
                raw_row=rr,
            )
        )

    return out


def _legal_order_subsequence(legal_moves: list[str], subset_set: set[str]) -> list[str]:
    ordered = [m for m in legal_moves if m in subset_set]
    if len(ordered) != len(subset_set):
        missing = sorted(subset_set.difference(set(ordered)))
        raise ValueError(f"Subset moves missing from legal order: {missing[:10]}")
    return ordered


def _mu_sorted_desc(mu_map: dict[str, float], moves: list[str]) -> list[str]:
    def key(mv: str) -> tuple[float, str]:
        # Higher μ first; tie-break lexicographically for determinism.
        return (float(mu_map.get(mv, -float("inf"))), mv)

    return sorted([str(m).strip().lower() for m in moves], key=key, reverse=True)


def _partition_mu_stratified(
    moves: list[str],
    *,
    mu_map: dict[str, float],
    block_size: int,
) -> list[list[str]]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if not moves:
        return []
    n = len(moves)
    num_blocks = int(math.ceil(n / float(block_size)))
    num_blocks = max(1, num_blocks)

    moves_desc = _mu_sorted_desc(mu_map, moves)
    blocks: list[list[str]] = [[] for _ in range(num_blocks)]
    for i, mv in enumerate(moves_desc):
        blocks[i % num_blocks].append(mv)
    return [b for b in blocks if b]


def _sample_indices(n: int, k: int, *, seed: int) -> list[int]:
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=k, replace=False).tolist()
    return sorted([int(i) for i in idx])


def _make_derived_index(
    *,
    global_seed: int,
    source_index: int,
    variant: str,
    sub_id: int,
) -> int:
    # Use 63-bit positive ints to avoid awkward signed-overflow across runtimes.
    return _stable_int_hash(
        str(global_seed),
        str(source_index),
        str(variant),
        str(sub_id),
        BUILDER_VERSION,
        mod=(2**63 - 1),
    )


def _build_hard_negatives(
    row: BaseRow,
    *,
    global_target: str,
    L_mu: int,
    mined_negatives: list[str],
    max_negatives: int,
    ensure_ground_truth: bool,
) -> list[str]:
    legal = row.legal_moves_uci
    mu_map = row.mu_map

    # μ-near-best negatives (exclude target).
    mu_desc = _mu_sorted_desc(mu_map, [m for m in legal if m != global_target])
    mu_negs = mu_desc[: max(0, int(L_mu))]

    negs = _dedup_preserve_order(list(mu_negs) + list(mined_negatives))
    if max_negatives > 0:
        negs = negs[: int(max_negatives)]

    candidate_set = set(negs)
    candidate_set.add(global_target)
    if ensure_ground_truth:
        candidate_set.add(row.ground_truth_uci)

    return _legal_order_subsequence(legal, candidate_set)


def _build_coverage_sets(
    row: BaseRow,
    *,
    global_target: str,
    excluded_moves: set[str],
    block_size: int,
    num_blocks: int,
    global_seed: int,
    ensure_ground_truth: bool,
) -> list[list[str]]:
    legal = row.legal_moves_uci
    mu_map = row.mu_map

    remaining = [m for m in legal if (m not in excluded_moves) and (m != global_target)]
    blocks = _partition_mu_stratified(remaining, mu_map=mu_map, block_size=int(block_size))
    if not blocks:
        return []

    # Deterministic sampling of which blocks to include.
    row_seed = _stable_int_hash(str(global_seed), str(row.source_index), "coverage-blocks", mod=2**31)
    picked = _sample_indices(len(blocks), int(num_blocks), seed=int(row_seed))
    out: list[list[str]] = []
    for j, idx in enumerate(picked):
        block = blocks[int(idx)]
        candidate_set = set(block)
        candidate_set.add(global_target)
        if ensure_ground_truth:
            candidate_set.add(row.ground_truth_uci)
        out.append(_legal_order_subsequence(legal, candidate_set))
    return out


def _mine_negatives_vllm(
    rows: list[BaseRow],
    *,
    template: Any,
    model: str,
    tokenizer_name: Optional[str],
    global_seed: int,
    total_rollouts: int,
    temperature: float,
    top_p: float,
    max_prompt_tokens: int,
    max_response_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    batch_size: int,
    K_mine: int,
) -> dict[int, list[str]]:
    if not _VLLM_AVAILABLE:
        raise RuntimeError("vLLM dependencies are unavailable; install vllm+transformers or disable mining.")
    if total_rollouts <= 0 or K_mine <= 0:
        return {}

    tok_name = tokenizer_name or model
    tokenizer = AutoTokenizer.from_pretrained(str(tok_name))  # type: ignore[misc]
    llm = LLM(  # type: ignore[misc]
        model=str(model),
        tensor_parallel_size=1,
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_num_seqs=int(max_num_seqs),
        seed=int(global_seed),
        disable_log_stats=True,
    )

    base_sampling_kwargs = dict(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=-1,
        min_p=0.0,
        max_tokens=int(max_response_tokens),
        repetition_penalty=1.0,
        detokenize=True,
    )

    out: dict[int, list[str]] = {}

    for start in range(0, len(rows), int(batch_size)):
        batch_rows = rows[start : start + int(batch_size)]

        messages_list: list[list[dict[str, str]]] = []
        for r in batch_rows:
            # Mine on the full-legal prompt in legal order (AIcrowd-aligned).
            prompt_text = _render_prompt_text(template, fen=r.fen, legal_moves=r.legal_moves_uci, considered_moves=r.legal_moves_uci)
            messages_list.append([{"role": "user", "content": prompt_text}])

        prompt_token_ids: list[list[int]] = []
        for messages in messages_list:
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            if not isinstance(ids, list):
                raise TypeError(f"Expected token id list from apply_chat_template, got {type(ids)}")
            if len(ids) > int(max_prompt_tokens):
                raise ValueError(f"Prompt is {len(ids)} tokens (max={int(max_prompt_tokens)}).")
            prompt_token_ids.append([int(x) for x in ids])

        vllm_inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids]

        sampling_params: list[Any] = []
        for r in batch_rows:
            seed_i = _stable_int_hash(str(global_seed), str(r.source_index), "select-v2-mine", mod=2**31)
            sampling_params.append(SamplingParams(seed=int(seed_i), n=int(total_rollouts), **base_sampling_kwargs))  # type: ignore[misc]

        outs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
        if len(outs) != len(batch_rows):
            raise RuntimeError(f"Expected {len(batch_rows)} outputs, got {len(outs)}")

        for r, out_obj in zip(batch_rows, outs, strict=True):
            if len(out_obj.outputs) != int(total_rollouts):
                raise RuntimeError(
                    f"Row {r.source_index}: expected n={int(total_rollouts)} mined outputs, got {len(out_obj.outputs)}"
                )
            legal_set = set(r.legal_moves_uci)
            counts: dict[str, int] = {}
            for o in out_obj.outputs:
                mv, reason = _parse_uci_move(o.text)
                if reason:
                    continue
                if mv is None:
                    continue
                mv = str(mv).strip().lower()
                if mv not in legal_set:
                    continue
                counts[mv] = counts.get(mv, 0) + 1

            if not counts:
                out[r.source_index] = []
                continue

            # Select top-K by frequency (tie-break lexicographically).
            top = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
            mined = [mv for (mv, _) in top[: int(K_mine)]]
            out[r.source_index] = mined

    return out


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--template_path", default="recipe/chess/prompt_templates/select_prompt.jinja")

    ap.add_argument("--limit_rows", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)

    # v2 mixture knobs
    ap.add_argument(
        "--include_full_legal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a full-legal candidate-set row per position (required to match test-time distribution).",
    )
    ap.add_argument(
        "--include_hard_neg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include one hard-negative small-k row per position.",
    )
    ap.add_argument(
        "--include_coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include one or more coverage-block rows per position.",
    )

    ap.add_argument("--L_mu", type=int, default=8)
    ap.add_argument("--K_mine", type=int, default=0, help="Number of model-mined negatives to include (0 disables mining).")
    ap.add_argument("--max_negatives_hard", type=int, default=12)
    ap.add_argument("--block_size", type=int, default=8)
    ap.add_argument("--num_blocks", type=int, default=2)
    ap.add_argument(
        "--ensure_ground_truth_included",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep reward_model.ground_truth in considered_moves for compatibility (even if it differs from μ-best).",
    )

    # Mining model options (optional; requires GPU + vLLM)
    ap.add_argument("--mine_model", type=str, default=None)
    ap.add_argument("--mine_tokenizer", type=str, default=None)
    ap.add_argument("--mine_total_rollouts", type=int, default=0, help="Total generations per row for mining (e.g., 64).")
    ap.add_argument("--mine_temperature", type=float, default=0.6)
    ap.add_argument("--mine_top_p", type=float, default=0.95)
    ap.add_argument("--mine_max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--mine_max_response_tokens", type=int, default=512)
    ap.add_argument("--mine_max_model_len", type=int, default=4096)
    ap.add_argument("--mine_gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--mine_max_num_seqs", type=int, default=1024)
    ap.add_argument("--mine_batch_size", type=int, default=128)

    # Optional inspection sample
    ap.add_argument("--sample_jsonl", default=None)
    ap.add_argument("--sample_n", type=int, default=50)
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

    rows = _load_base_rows(
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

    # Optional mining pass (GPU/vLLM).
    mined_by_source: dict[int, list[str]] = {}
    if int(args.K_mine) > 0:
        if not args.mine_model:
            raise SystemExit("--K_mine > 0 requires --mine_model")
        if int(args.mine_total_rollouts) <= 0:
            raise SystemExit("--K_mine > 0 requires --mine_total_rollouts > 0")
        print(
            f"[mine] enabled model={args.mine_model} total_rollouts={int(args.mine_total_rollouts)} "
            f"K_mine={int(args.K_mine)} batch_size={int(args.mine_batch_size)}"
        )
        mined_by_source = _mine_negatives_vllm(
            rows,
            template=template,
            model=str(args.mine_model),
            tokenizer_name=str(args.mine_tokenizer) if args.mine_tokenizer else None,
            global_seed=int(args.seed),
            total_rollouts=int(args.mine_total_rollouts),
            temperature=float(args.mine_temperature),
            top_p=float(args.mine_top_p),
            max_prompt_tokens=int(args.mine_max_prompt_tokens),
            max_response_tokens=int(args.mine_max_response_tokens),
            max_model_len=int(args.mine_max_model_len),
            gpu_memory_utilization=float(args.mine_gpu_memory_utilization),
            max_num_seqs=int(args.mine_max_num_seqs),
            batch_size=int(args.mine_batch_size),
            K_mine=int(args.K_mine),
        )
        print(f"[mine] done rows_with_mined={len(mined_by_source)}")

    # Sample JSONL selection (deterministic but cheap).
    py_rng = random.Random(int(args.seed) + 1337 + 10_000 * shard_idx)
    sample_rows = set(py_rng.sample(range(len(rows)), k=min(int(args.sample_n), len(rows)))) if rows else set()
    samples: list[dict[str, Any]] = []

    out_rows: list[dict[str, Any]] = []

    # Counters for quick summaries.
    n_full = 0
    n_hard = 0
    n_cov = 0

    for i, r in enumerate(rows):
        legal = list(r.legal_moves_uci)
        legal_set = set(legal)
        if not legal:
            raise ValueError(f"Row {r.source_index}: empty legal moves")

        global_target = _best_move_by_mu(r.mu_map, legal)
        if global_target not in legal_set:
            raise ValueError(f"Row {r.source_index}: μ-best {global_target} not in legal set (unexpected)")

        # Full legal row.
        derived_plans: list[tuple[str, int, list[str]]] = []
        if bool(args.include_full_legal):
            derived_plans.append(("full_legal", 0, list(legal)))

        # Hard-negative row.
        hard_moves: list[str] = []
        if bool(args.include_hard_neg):
            mined = mined_by_source.get(r.source_index, [])
            hard_moves = _build_hard_negatives(
                r,
                global_target=global_target,
                L_mu=int(args.L_mu),
                mined_negatives=mined,
                max_negatives=int(args.max_negatives_hard),
                ensure_ground_truth=bool(args.ensure_ground_truth_included),
            )
            derived_plans.append(("hard_neg", 0, hard_moves))

        # Coverage blocks.
        if bool(args.include_coverage):
            excluded = set(hard_moves) if hard_moves else {global_target}
            cov_sets = _build_coverage_sets(
                r,
                global_target=global_target,
                excluded_moves=excluded,
                block_size=int(args.block_size),
                num_blocks=int(args.num_blocks),
                global_seed=int(args.seed),
                ensure_ground_truth=bool(args.ensure_ground_truth_included),
            )
            for j, moves in enumerate(cov_sets):
                derived_plans.append(("coverage_block", int(j), moves))

        for variant, sub_id, considered_moves in derived_plans:
            considered_moves = _dedup_preserve_order(considered_moves)
            if not considered_moves:
                raise ValueError(f"Row {r.source_index} variant={variant}: empty considered_moves")
            considered_set = set(considered_moves)
            if not considered_set.issubset(legal_set):
                illegal = sorted(list(considered_set - legal_set))[:10]
                raise ValueError(f"Row {r.source_index} variant={variant}: considered_moves contains illegal moves: {illegal}")
            if global_target not in considered_set:
                raise ValueError(f"Row {r.source_index} variant={variant}: missing μ-best target {global_target}")
            if bool(args.ensure_ground_truth_included) and (r.ground_truth_uci not in considered_set):
                raise ValueError(f"Row {r.source_index} variant={variant}: missing ground_truth {r.ground_truth_uci}")

            derived_index = _make_derived_index(
                global_seed=int(args.seed),
                source_index=int(r.source_index),
                variant=str(variant),
                sub_id=int(sub_id),
            )

            rr = dict(r.raw_row)
            # IMPORTANT: copy nested dicts so derived rows don't share mutable state.
            rm_raw = rr.get("reward_model") or {}
            rm = dict(rm_raw) if isinstance(rm_raw, dict) else {}
            rm["considered_moves_uci"] = list(considered_moves)
            rr["reward_model"] = rm

            ei_raw = rr.get("extra_info") or {}
            ei = dict(ei_raw) if isinstance(ei_raw, dict) else {}
            ei["source_index"] = int(r.source_index)
            ei["derived_variant"] = str(variant)
            ei["derived_sub_id"] = int(sub_id)
            ei["builder_version"] = str(BUILDER_VERSION)
            ei["index"] = int(derived_index)
            rr["extra_info"] = ei

            prompt_text = _render_prompt_text(template, fen=r.fen, legal_moves=legal, considered_moves=considered_moves)
            rr["prompt"] = [{"role": "user", "content": prompt_text}]

            out_rows.append(rr)

            if variant == "full_legal":
                n_full += 1
            elif variant == "hard_neg":
                n_hard += 1
            elif variant == "coverage_block":
                n_cov += 1

            if i in sample_rows:
                samples.append(
                    {
                        "source_index": int(r.source_index),
                        "derived_index": int(derived_index),
                        "derived_variant": str(variant),
                        "derived_sub_id": int(sub_id),
                        "fen": r.fen,
                        "ground_truth": r.ground_truth_uci,
                        "mu_target": global_target,
                        "n_legal": int(len(legal)),
                        "n_considered": int(len(considered_moves)),
                        "legal_moves_uci": legal,
                        "considered_moves_uci": list(considered_moves),
                    }
                )

    out_table = pa.Table.from_pylist(out_rows)
    _atomic_write_parquet(out_table, output_path=output_path)
    print(f"[done] wrote={output_path} rows={out_table.num_rows} shard={shard_idx}/{num_shards}")
    print(f"[stats] n_full={n_full} n_hard={n_hard} n_cov={n_cov}")

    if args.sample_jsonl:
        _write_jsonl(Path(args.sample_jsonl), samples)
        print(f"[sample] wrote={args.sample_jsonl} rows={len(samples)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
