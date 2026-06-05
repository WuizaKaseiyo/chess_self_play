#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pyarrow.parquet as pq
from jinja2 import Environment
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from recipe.chess.reward_fn import _to_uci
from verl.utils.prompt import (
    encode_prompt_from_messages,
    infer_use_chat_template_from_model_name,
    is_qwen3_base_model,
    render_prompt_from_messages,
)

_UCI_MOVE_TAG_RE = re.compile(r"<\s*uci_move\s*>(?P<ans>[\s\S]*?)<\s*/\s*uci_move\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class RowData:
    row_id: int
    fen: str
    legal_moves_uci: list[str]
    mu_map: dict[str, float]


@dataclass(frozen=True)
class SubsetPlan:
    row_id: int
    subset_id: str
    k: int
    candidate_moves_uci: list[str]
    target_move_uci: str
    mu_best: float
    mu_second: float
    top_margin: float
    h1: float
    includes_global_best: bool
    sampler_version: str
    sampler_seed: int


def _sanitize_model_name(model: str) -> str:
    safe = model.replace("/", "__").replace(":", "_")
    safe = "".join(ch if (ch.isalnum() or ch in "._-__") else "_" for ch in safe)
    return safe


def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _stable_int_hash(*parts: str, mod: int) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], "big") % int(mod)


def _load_template(template_path: str) -> Any:
    template_file = Path(template_path)
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")
    env = Environment(autoescape=False)
    return env.from_string(template_file.read_text(encoding="utf-8"))


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


def _load_rows(parquet_path: str, limit_rows: int) -> list[RowData]:
    table = pq.read_table(parquet_path, columns=["reward_model", "extra_info"])
    rows = table.to_pylist()
    if limit_rows < 0:
        raise ValueError("--limit_rows must be >= 0")
    rows = rows[:limit_rows] if limit_rows else []

    out: list[RowData] = []
    for row in rows:
        rm = row.get("reward_model") or {}
        ei = row.get("extra_info") or {}
        row_id = int(ei.get("index"))
        fen = str(rm.get("fen") or "").strip()
        legal_moves = _normalize_moves(rm.get("legal_moves_uci"))

        mu_json = rm.get("move_expected_scores_json")
        if isinstance(mu_json, str) and mu_json.strip():
            mu_raw = json.loads(mu_json)
        else:
            mu_raw = mu_json
        if not isinstance(mu_raw, dict):
            raise ValueError(f"Row {row_id}: move_expected_scores_json is not a dict")
        mu_map: dict[str, float] = {}
        for k, v in mu_raw.items():
            key = str(k).strip().lower()
            if not key:
                continue
            try:
                mu_map[key] = float(v)
            except Exception:
                continue

        # Fallback to move_values_json if expected scores are missing (shouldn't happen for searchless data).
        if not mu_map:
            mv_json = rm.get("move_values_json")
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

        if not legal_moves:
            raise ValueError(f"Row {row_id}: empty legal_moves_uci")
        if not mu_map:
            raise ValueError(f"Row {row_id}: empty mu_map (expected expected_scores or move_values)")

        out.append(RowData(row_id=row_id, fen=fen, legal_moves_uci=legal_moves, mu_map=mu_map))
    return out


def _best_move_by_mu(mu_map: dict[str, float], moves: Iterable[str]) -> tuple[str, float]:
    """Argmax μ with deterministic tie-break: highest μ, then lexicographic UCI."""
    best_move = ""
    best_mu = -float("inf")
    for mv in moves:
        key = str(mv).strip().lower()
        mu = float(mu_map.get(key, -float("inf")))
        if (mu > best_mu) or (mu == best_mu and key < best_move):
            best_move = key
            best_mu = mu
    if not best_move:
        raise ValueError("Empty move list when selecting best move.")
    return best_move, float(best_mu)


def _compute_subset_metrics(
    mu_map: dict[str, float],
    candidate_moves_uci: list[str],
    *,
    global_best_move: str,
    h1_eps: float = 1e-6,
) -> tuple[str, float, float, float, float, bool]:
    """Return (target_move, mu_best, mu_second, top_margin, h1, includes_global_best)."""
    if not candidate_moves_uci:
        raise ValueError("candidate_moves_uci must be non-empty")
    target, mu_best = _best_move_by_mu(mu_map, candidate_moves_uci)

    k = len(candidate_moves_uci)
    includes_global_best = global_best_move in set(candidate_moves_uci)

    if k == 1:
        # No second-best exists; keep metrics finite/non-NaN by using a sentinel μ_second=0.
        mu_second = 0.0
        top_margin = abs(float(mu_best) - float(mu_second))
        h1 = 0.0
        return target, float(mu_best), float(mu_second), float(top_margin), float(h1), bool(includes_global_best)

    # Second-best is the best among remaining moves after applying tie-break for the chosen target.
    second_candidates = [m for m in candidate_moves_uci if m != target]
    second_move, mu_second = _best_move_by_mu(mu_map, second_candidates)
    _ = second_move
    top_margin = float(mu_best) - float(mu_second)
    if top_margin < 0:
        # Should be impossible, but keep it safe.
        top_margin = abs(top_margin)

    # H1 complexity: sum 1 / Δ_i^2 over non-best moves, Δ_i = μ_best - μ_i.
    # For ties / Δ=0, clamp with eps to keep finite and numerically stable.
    eps = float(h1_eps)
    if eps <= 0:
        raise ValueError("h1_eps must be > 0")
    h1_acc = 0.0
    for mv in candidate_moves_uci:
        if mv == target:
            continue
        mu_i = float(mu_map.get(mv, -float("inf")))
        delta = float(mu_best) - mu_i
        if delta < eps:
            delta = eps
        h1_acc += 1.0 / (delta * delta)

    return target, float(mu_best), float(mu_second), float(top_margin), float(h1_acc), bool(includes_global_best)


def _choose_k_values(n: int) -> list[int]:
    base = [1, 2, 3, 4, 5, 8, 12, 16, 24, 32]
    ks = sorted({k for k in base if 1 <= k <= n} | {n})
    return ks


def _subset_id_from_ordered_moves(moves: list[str]) -> str:
    payload = "\n".join(moves)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _ordered_subset_from_legal(legal_moves_uci: list[str], subset_set: set[str]) -> list[str]:
    ordered = [m for m in legal_moves_uci if m in subset_set]
    # Defensive: ensure we didn't drop anything due to normalization mismatch.
    if len(ordered) != len(subset_set):
        missing = sorted(subset_set.difference(set(ordered)))
        raise ValueError(f"Subset moves missing from legal order: {missing[:10]}")
    return ordered


def _softmax_weights(values: np.ndarray, tau: float) -> np.ndarray:
    if values.size == 0:
        return values
    t = float(tau)
    if t <= 0:
        raise ValueError("tau must be > 0")
    z = (values - np.max(values)) / t
    w = np.exp(z)
    s = float(w.sum())
    if not math.isfinite(s) or s <= 0:
        return np.ones_like(values, dtype=np.float64) / float(values.size)
    return w / s


def _generate_subsets_for_row(
    row: RowData,
    *,
    max_subsets: int,
    seed: int,
    sampler_version: str,
) -> list[SubsetPlan]:
    moves = list(row.legal_moves_uci)
    n = len(moves)
    if n < 1:
        raise ValueError(f"Row {row.row_id}: no legal moves")
    if max_subsets <= 0:
        raise ValueError("--max_subsets_per_row must be > 0")

    # Global best by μ over full legal list.
    global_best_move, _ = _best_move_by_mu(row.mu_map, moves)

    ks = _choose_k_values(n)
    if not ks:
        raise ValueError(f"Row {row.row_id}: empty k list")

    rng = np.random.default_rng(int(seed))

    # Track unique subsets by subset_id (depends on candidate list order).
    out: list[SubsetPlan] = []
    seen: set[str] = set()

    def _try_add(subset_set: set[str]) -> None:
        ordered = _ordered_subset_from_legal(moves, subset_set)
        subset_id = _subset_id_from_ordered_moves(ordered)
        if subset_id in seen:
            return
        target, mu_best, mu_second, top_margin, h1, includes_best = _compute_subset_metrics(
            row.mu_map,
            ordered,
            global_best_move=global_best_move,
        )
        out.append(
            SubsetPlan(
                row_id=row.row_id,
                subset_id=subset_id,
                k=len(ordered),
                candidate_moves_uci=ordered,
                target_move_uci=target,
                mu_best=mu_best,
                mu_second=mu_second,
                top_margin=top_margin,
                h1=h1,
                includes_global_best=includes_best,
                sampler_version=sampler_version,
                sampler_seed=int(seed),
            )
        )
        seen.add(subset_id)

    # Always include the full legal move set (k=n) as a baseline.
    _try_add(set(moves))
    if len(out) >= max_subsets:
        return out[:max_subsets]

    # Allocate subset counts across k (excluding the full set, already included).
    ks_no_full = [k for k in ks if k != n]
    remaining_budget = max(0, max_subsets - len(out))
    if not ks_no_full or remaining_budget <= 0:
        return out

    weights = np.array([1.0 / math.sqrt(float(k)) for k in ks_no_full], dtype=np.float64)
    weights = weights / weights.sum()
    raw_counts = weights * float(remaining_budget)
    counts = np.floor(raw_counts).astype(int)
    # Distribute remainder to largest fractional parts (deterministic).
    remainder = int(remaining_budget - int(counts.sum()))
    if remainder > 0:
        frac = raw_counts - counts.astype(np.float64)
        for idx in np.argsort(-frac)[:remainder]:
            counts[int(idx)] += 1
    # Ensure we don't assign zero to tiny strata when budget allows.
    if remaining_budget >= len(ks_no_full):
        for i in range(len(counts)):
            if counts[i] == 0:
                counts[i] = 1

    # Precompute ranking pools.
    mu_vec = np.array([float(row.mu_map.get(m, -float("inf"))) for m in moves], dtype=np.float64)
    order_desc = list(np.argsort(-mu_vec, kind="stable"))
    order_asc = list(np.argsort(mu_vec, kind="stable"))
    moves_desc = [moves[i] for i in order_desc]
    moves_asc = [moves[i] for i in order_asc]

    # "Top" and "bottom" pools exclude the global best for convenience.
    top_pool = [m for m in moves_desc if m != global_best_move]
    bottom_pool = [m for m in moves_asc if m != global_best_move]

    for k, want in zip(ks_no_full, counts.tolist()):
        k = int(k)
        want = int(want)
        if want <= 0:
            continue

        # If the combinatorial space is small, enumerate all subsets for full coverage.
        # This is important for very small n (e.g., n<=6) so we don't spam duplicates.
        try:
            total_k = math.comb(n, k)
        except Exception:
            total_k = None
        if total_k is not None and total_k <= want and total_k <= 10_000:
            from itertools import combinations

            for comb in combinations(moves, k):
                _try_add(set(comb))
                if len(out) >= max_subsets:
                    return out[:max_subsets]
            continue

        # Split budget within each k across sampling modes.
        n_easy = int(round(0.30 * want))
        n_hard = int(round(0.30 * want))
        n_real = int(round(0.20 * want))
        n_excl = max(0, want - (n_easy + n_hard + n_real))

        def _sample_without_replacement(pool: list[str], size: int) -> list[str]:
            if size <= 0:
                return []
            if len(pool) < size:
                return list(pool)
            idx = rng.choice(len(pool), size=size, replace=False)
            return [pool[int(i)] for i in idx]

        # k=1 special case: just sample singletons (a mix of best and random other moves).
        if k == 1:
            # Ensure the global-best singleton is present.
            _try_add({global_best_move})
            # Fill remaining with random singletons (excluding best, if possible).
            other = [m for m in moves if m != global_best_move]
            rng.shuffle(other)
            for mv in other[: max(0, want - 1)]:
                _try_add({mv})
                if len(out) >= max_subsets:
                    return out[:max_subsets]
            continue

        # Mode 1: easy (include best + mostly low-μ distractors).
        for _ in range(n_easy):
            pool = bottom_pool
            subset = {global_best_move}
            subset.update(_sample_without_replacement(pool, k - 1))
            if len(subset) < k:
                subset.update(_sample_without_replacement([m for m in moves if m not in subset], k - len(subset)))
            _try_add(subset)
            if len(out) >= max_subsets:
                return out[:max_subsets]

        # Mode 2: hard (include best + mostly high-μ near-ties).
        for _ in range(n_hard):
            pool = top_pool[: min(len(top_pool), max(10, k * 4))]
            subset = {global_best_move}
            subset.update(_sample_without_replacement(pool, k - 1))
            if len(subset) < k:
                subset.update(_sample_without_replacement([m for m in moves if m not in subset], k - len(subset)))
            _try_add(subset)
            if len(out) >= max_subsets:
                return out[:max_subsets]

        # Mode 3: "realistic" weighted sampling (no forced inclusion of best).
        if n_real > 0:
            tau = 0.2
            w = _softmax_weights(mu_vec, tau=tau)
            for _ in range(n_real):
                idx = rng.choice(n, size=k, replace=False, p=w)
                subset = {moves[int(i)] for i in idx}
                _try_add(subset)
                if len(out) >= max_subsets:
                    return out[:max_subsets]

        # Mode 4: exclude global best (ensures non-trivial target when global-best absent).
        other_moves = [m for m in moves if m != global_best_move]
        for j in range(n_excl):
            # Alternate between "easy" and "hard" exclude-best subsets.
            hard = (j % 2) == 0
            if hard:
                pool = top_pool[: min(len(top_pool), max(10, k * 4))]
                base = _sample_without_replacement(pool, k)
                if len(base) < k:
                    base += _sample_without_replacement([m for m in other_moves if m not in set(base)], k - len(base))
                subset = set(base)
            else:
                # Include the best among the remaining moves + low-μ fillers.
                second_best, _ = _best_move_by_mu(row.mu_map, other_moves)
                subset = {second_best}
                filler_pool = [m for m in bottom_pool if m != second_best]
                subset.update(_sample_without_replacement(filler_pool, k - 1))
                if len(subset) < k:
                    subset.update(_sample_without_replacement([m for m in other_moves if m not in subset], k - len(subset)))
            _try_add(subset)
            if len(out) >= max_subsets:
                return out[:max_subsets]

    return out[:max_subsets]


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def _render_prompt_messages(template: Any, row: RowData, subset: SubsetPlan) -> list[dict[str, str]]:
    prompt_text = str(
        template.render(
            FEN=row.fen,
            legal_moves_uci_list=row.legal_moves_uci,
            considered_moves_uci_list=subset.candidate_moves_uci,
        )
    )
    return [{"role": "user", "content": prompt_text}]


def _build_prompt_token_ids(
    tokenizer: Any,
    messages_list: list[list[dict[str, str]]],
    *,
    max_prompt_tokens: int,
    use_chat_template: bool,
) -> list[list[int]]:
    prompt_token_ids: list[list[int]] = []
    for messages in messages_list:
        _, ids = encode_prompt_from_messages(
            tokenizer,
            messages,
            use_chat_template=use_chat_template,
            add_generation_prompt=True,
        )
        if len(ids) > max_prompt_tokens:
            raise ValueError(f"Prompt is {len(ids)} tokens (max={max_prompt_tokens}).")
        prompt_token_ids.append([int(x) for x in ids])
    return prompt_token_ids


def _iter_batches(n: int, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, n, batch_size):
        yield start, min(n, start + batch_size)


def _parse_completion(text: str) -> tuple[str, bool, str]:
    """Return (pred_move_uci_or_empty, format_ok, error_reason_or_empty)."""
    s = text or ""
    matches = list(_UCI_MOVE_TAG_RE.finditer(s))
    if not matches:
        return "", False, "format_error"

    # Best-effort: parse the first <uci_move>...</uci_move> span, but treat multiple spans as a format violation.
    ans_payload = matches[0].group("ans")
    pred = _to_uci(ans_payload)
    if pred is None:
        return "", False, "bad_move"
    if len(matches) != 1:
        return pred, False, "format_error"
    return pred, True, ""


def _stable_prompt_seed(global_seed: int, subset: SubsetPlan) -> int:
    # vLLM expects an int32 seed; keep it in range.
    h = hashlib.sha256()
    h.update(str(global_seed).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(subset.row_id).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(subset.subset_id).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big", signed=False)


def _load_or_build_plan(
    rows: list[RowData],
    *,
    out_dir: Path,
    parquet_path: str,
    max_subsets_per_row: int,
    seed: int,
    sampler_version: str,
) -> tuple[dict[int, RowData], list[SubsetPlan]]:
    rows_by_id = {r.row_id: r for r in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError("Duplicate row_id detected in loaded rows")

    plan_path = out_dir / "subsets.jsonl"
    rows_path = out_dir / "rows.jsonl"
    manifest_path = out_dir / "manifest.json"
    diagnostics_path = out_dir / "sampling_diagnostics.json"

    if plan_path.exists() and rows_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "parquet": str(manifest.get("parquet") or ""),
            "parquet_rows": int(manifest.get("parquet_rows") or 0),
            "max_subsets_per_row": int(manifest.get("max_subsets_per_row") or 0),
            "seed": int(manifest.get("seed") or 0),
            "sampler_version": str(manifest.get("sampler_version") or ""),
        }
        got = {
            "parquet": str(os.path.abspath(str(parquet_path))),
            "parquet_rows": len(rows),
            "max_subsets_per_row": int(max_subsets_per_row),
            "seed": int(seed),
            "sampler_version": str(sampler_version),
        }
        # Validate key plan-defining fields. Parquet path mismatch is allowed (e.g., different mount points),
        # but seed/sampler/max_subsets must match to keep subset IDs and caching consistent.
        if expected["max_subsets_per_row"] and expected["max_subsets_per_row"] != got["max_subsets_per_row"]:
            raise ValueError(f"Existing plan has max_subsets_per_row={expected['max_subsets_per_row']} but got {got['max_subsets_per_row']}")
        if expected["seed"] != got["seed"] or expected["sampler_version"] != got["sampler_version"]:
            raise ValueError(
                "Existing plan manifest mismatch:\n"
                f"  seed: {expected['seed']} vs {got['seed']}\n"
                f"  sampler_version: {expected['sampler_version']!r} vs {got['sampler_version']!r}\n"
                "Use a new --out_dir if you want a different plan."
            )
        plan: list[SubsetPlan] = []
        for rec in _read_jsonl(plan_path):
            plan.append(SubsetPlan(**rec))
        # Rows are optional for resuming; but ensure we can render prompts for all plan entries.
        if rows_by_id:
            missing = sorted({p.row_id for p in plan}.difference(rows_by_id.keys()))
            if missing:
                raise ValueError(f"Plan references missing row_ids: {missing[:10]}")
        return rows_by_id, plan

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write rows.jsonl first for reproducibility.
    _write_jsonl(
        rows_path,
        [
            {
                "row_id": r.row_id,
                "fen": r.fen,
                "legal_moves_uci": r.legal_moves_uci,
                "mu_map": r.mu_map,
                "mu_map_sha256": _sha256_text(json.dumps(r.mu_map, sort_keys=True)),
            }
            for r in rows
        ],
    )

    plan_records: list[dict[str, Any]] = []
    plan: list[SubsetPlan] = []
    for r in rows:
        row_seed = int(seed) + int(r.row_id)
        subsets = _generate_subsets_for_row(r, max_subsets=max_subsets_per_row, seed=row_seed, sampler_version=sampler_version)
        plan.extend(subsets)
        for s in subsets:
            plan_records.append(dataclasses.asdict(s))

    _write_jsonl(plan_path, plan_records)

    manifest = {
        "parquet": str(os.path.abspath(str(parquet_path))),
        "parquet_rows": len(rows),
        "max_subsets_per_row": int(max_subsets_per_row),
        "seed": int(seed),
        "sampler_version": str(sampler_version),
        "total_subsets": int(len(plan)),
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, manifest_path)

    # Sampling diagnostics (diversity checks).
    ks = sorted({int(s.k) for s in plan})
    k_counts: dict[str, int] = {}
    for s in plan:
        k_counts[str(int(s.k))] = k_counts.get(str(int(s.k)), 0) + 1

    def _quantiles(vals: list[float], qs: list[float]) -> dict[str, float]:
        if not vals:
            return {str(q): float("nan") for q in qs}
        arr = np.array(vals, dtype=np.float64)
        out_q = np.quantile(arr, qs, method="linear")
        return {str(q): float(v) for q, v in zip(qs, out_q)}

    per_k: dict[str, Any] = {}
    for k in ks:
        subset_k = [s for s in plan if int(s.k) == k]
        margins = [float(s.top_margin) for s in subset_k]
        h1s = [float(math.log1p(float(s.h1))) for s in subset_k]
        per_k[str(k)] = {
            "count": int(len(subset_k)),
            "top_margin_quantiles": _quantiles(margins, [0.0, 0.1, 0.5, 0.9, 1.0]),
            "log1p_h1_quantiles": _quantiles(h1s, [0.0, 0.1, 0.5, 0.9, 1.0]),
            "includes_global_best_frac": float(sum(1 for s in subset_k if s.includes_global_best) / max(1, len(subset_k))),
        }

    diagnostics = {
        "total_subsets": int(len(plan)),
        "k_counts": k_counts,
        "per_k": per_k,
    }
    tmp = diagnostics_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, sort_keys=True)
    os.replace(tmp, diagnostics_path)
    return rows_by_id, plan


def _run_sanity_check(
    *,
    llm: LLM,
    tokenizer: Any,
    template: Any,
    rows_by_id: dict[int, RowData],
    plan: list[SubsetPlan],
    n_sanity_rows: int,
    sanity_cases_per_row: int,
    samples_per_case: int,
    max_prompt_tokens: int,
    max_response_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    seed_mode: str,
    use_chat_template: bool,
) -> None:
    # Pick rows deterministically: ascending row_id.
    row_ids = sorted(rows_by_id.keys())[: int(n_sanity_rows)]
    if not row_ids:
        raise ValueError("No rows available for sanity check")

    # Build single-move subsets from each row: include the global best and some other moves.
    sanity_subsets: list[SubsetPlan] = []
    plan_by_row: dict[int, list[SubsetPlan]] = {}
    for s in plan:
        plan_by_row.setdefault(int(s.row_id), []).append(s)

    for rid in row_ids:
        row = rows_by_id[rid]
        # Build candidate singletons using the legal move list.
        legal = row.legal_moves_uci
        # Always test the first legal move and the (μ-)best legal move, plus some spread.
        global_best, _ = _best_move_by_mu(row.mu_map, legal)
        candidates: list[str] = []
        if legal:
            candidates.append(legal[0])
        if global_best and global_best not in candidates:
            candidates.append(global_best)
        # Add some additional deterministic picks from the legal list.
        stride = max(1, len(legal) // max(1, sanity_cases_per_row))
        for i in range(0, len(legal), stride):
            mv = legal[i]
            if mv not in candidates:
                candidates.append(mv)
            if len(candidates) >= sanity_cases_per_row:
                break
        candidates = candidates[: sanity_cases_per_row]

        for mv in candidates:
            # Construct a minimal SubsetPlan-like object for this singleton.
            subset_id = _subset_id_from_ordered_moves([mv])
            target, mu_best, mu_second, top_margin, h1, includes_best = _compute_subset_metrics(
                row.mu_map,
                [mv],
                global_best_move=global_best,
            )
            sanity_subsets.append(
                SubsetPlan(
                    row_id=rid,
                    subset_id=subset_id,
                    k=1,
                    candidate_moves_uci=[mv],
                    target_move_uci=target,
                    mu_best=mu_best,
                    mu_second=mu_second,
                    top_margin=top_margin,
                    h1=h1,
                    includes_global_best=includes_best,
                    sampler_version="sanity_singleton",
                    sampler_seed=int(seed),
                )
            )

    messages_list: list[list[dict[str, str]]] = []
    for s in sanity_subsets:
        row = rows_by_id[int(s.row_id)]
        messages_list.append(_render_prompt_messages(template, row, s))

    prompt_token_ids = _build_prompt_token_ids(
        tokenizer,
        messages_list,
        max_prompt_tokens=max_prompt_tokens,
        use_chat_template=use_chat_template,
    )
    vllm_inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids]

    if seed_mode not in ("engine", "per_prompt"):
        raise ValueError(f"Unknown seed_mode={seed_mode!r} (expected 'engine' or 'per_prompt')")

    base_sampling_kwargs = dict(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=-1,
        min_p=0.0,
        max_tokens=int(max_response_tokens),
        repetition_penalty=1.0,
        detokenize=True,
    )

    k_chunk = max(1, min(int(samples_per_case), int(getattr(llm, "max_num_seqs", 1024)) // max(1, len(sanity_subsets))))
    # For sanity checks, keep it simple: do one call if possible.
    k_chunk = int(samples_per_case) if (len(sanity_subsets) * int(samples_per_case) <= int(getattr(llm, "max_num_seqs", 1024))) else k_chunk

    all_sample_texts: list[list[str]] = [[] for _ in sanity_subsets]
    for chunk_idx, chunk_start in enumerate(range(0, int(samples_per_case), k_chunk)):
        n_gen = min(k_chunk, int(samples_per_case) - chunk_start)
        if seed_mode == "per_prompt":
            seed_stride = 1_000_000
            sampling_params = [
                SamplingParams(seed=_stable_prompt_seed(seed, s) + chunk_idx * seed_stride, n=n_gen, **base_sampling_kwargs)
                for s in sanity_subsets
            ]
            outs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
        else:
            outs = llm.generate(prompts=vllm_inputs, sampling_params=SamplingParams(n=n_gen, **base_sampling_kwargs), use_tqdm=False)

        if len(outs) != len(sanity_subsets):
            raise RuntimeError(f"Expected {len(sanity_subsets)} outputs, got {len(outs)}")
        for i, out in enumerate(outs):
            if len(out.outputs) != n_gen:
                raise RuntimeError(f"Sanity: expected n={n_gen} outputs, got {len(out.outputs)}")
            all_sample_texts[i].extend([o.text for o in out.outputs])

    # Hard gate: every sample must output the ONLY candidate move (under strict parsing).
    failures: list[dict[str, Any]] = []
    for s, sample_texts in zip(sanity_subsets, all_sample_texts):
        only_mv = s.candidate_moves_uci[0]
        cand_set = {only_mv}
        for j, txt in enumerate(sample_texts):
            pred, fmt_ok, err = _parse_completion(txt)
            in_subset = bool(pred) and (pred in cand_set)
            ok = fmt_ok and in_subset and (pred == only_mv)
            if not ok:
                row = rows_by_id[int(s.row_id)]
                prompt_text = render_prompt_from_messages(
                    tokenizer,
                    _render_prompt_messages(template, row, s),
                    use_chat_template=use_chat_template,
                    add_generation_prompt=True,
                )
                failures.append(
                    {
                        "row_id": int(s.row_id),
                        "only_move": only_mv,
                        "sample_idx": int(j),
                        "pred_move": pred,
                        "format_ok": bool(fmt_ok),
                        "in_subset": bool(in_subset),
                        "error": err,
                        "raw_output": txt,
                        "prompt_text": prompt_text,
                    }
                )
                break
        if failures:
            break

    if failures:
        msg = json.dumps(failures[0], indent=2, ensure_ascii=False)
        raise SystemExit(
            "\n".join(
                [
                    "[SANITY CHECK FAILED] One-candidate selection did not force the only move.",
                    "First failing case (for debugging):",
                    msg,
                ]
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--use_chat_template", dest="use_chat_template", action="store_true", default=None)
    ap.add_argument("--no_use_chat_template", dest="use_chat_template", action="store_false")
    ap.add_argument("--parquet", default="data/chess_puzzles/test.parquet")
    ap.add_argument("--template_path", default="recipe/chess/prompt_templates/select_prompt.jinja")
    ap.add_argument("--out_dir", default=None, help="Defaults to outputs/select_eval/<model>__<templatehash>_seed<seed>")

    ap.add_argument("--limit_rows", type=int, default=100)
    ap.add_argument("--max_subsets_per_row", type=int, default=1000)
    ap.add_argument("--samples_per_subset", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed_mode", choices=["engine", "per_prompt"], default="per_prompt")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--max_prompt_tokens", type=int, default=1024)
    ap.add_argument("--max_response_tokens", type=int, default=2000)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_num_seqs", type=int, default=1024)

    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument(
        "--no_resume",
        action="store_true",
        help="If set, ignore any existing cached results and recompute this shard (appends to results file).",
    )
    ap.add_argument(
        "--audit_max_records",
        type=int,
        default=200,
        help="Write at most this many audit records containing prompt+raw outputs for compliance debugging.",
    )

    ap.add_argument("--sanity_rows", type=int, default=5)
    ap.add_argument("--sanity_cases_per_row", type=int, default=3)
    ap.add_argument("--sanity_only", action="store_true", default=False)
    ap.add_argument(
        "--skip_sanity",
        action="store_true",
        default=False,
        help="Skip the one-candidate sanity check (only use this if you already ran --sanity_only successfully).",
    )

    args = ap.parse_args()

    sampler_version = "select_subset_sampler_v1"
    rows = _load_rows(str(args.parquet), limit_rows=int(args.limit_rows))
    if not rows:
        raise SystemExit(f"No rows loaded (parquet={args.parquet}, limit_rows={args.limit_rows})")

    template = _load_template(str(args.template_path))
    template_hash = _sha256_text(Path(args.template_path).read_text(encoding="utf-8"))[:12]

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / "select_eval" / f"{_sanitize_model_name(args.model)}__{template_hash}_seed{int(args.seed)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist plan (subsets.jsonl) for reproducibility.
    rows_by_id, plan = _load_or_build_plan(
        rows,
        out_dir=out_dir,
        parquet_path=str(args.parquet),
        max_subsets_per_row=int(args.max_subsets_per_row),
        seed=int(args.seed),
        sampler_version=sampler_version,
    )

    # Tokenizer.
    tokenizer_model = str(args.tokenizer) if args.tokenizer else str(args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_chat_template = (
        infer_use_chat_template_from_model_name(str(tokenizer_model), default=True)
        if args.use_chat_template is None
        else bool(args.use_chat_template)
    )
    if is_qwen3_base_model(str(tokenizer_model)) and use_chat_template:
        raise ValueError("Qwen3 base selection eval must use --no_use_chat_template.")

    # vLLM engine.
    llm = LLM(
        model=str(args.model),
        tokenizer=tokenizer_model,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        seed=int(args.seed),
        max_num_seqs=int(args.max_num_seqs),
    )

    # Evaluation config (for caching / reproducibility).
    eval_config_path = out_dir / "eval_config.json"
    current_eval_config = {
        "model": str(args.model),
        "tokenizer": str(tokenizer_model),
        "template_path": str(args.template_path),
        "template_sha256_12": str(template_hash),
        "parquet": str(os.path.abspath(str(args.parquet))),
        "limit_rows": int(args.limit_rows),
        "max_subsets_per_row": int(args.max_subsets_per_row),
        "samples_per_subset": int(args.samples_per_subset),
        "seed": int(args.seed),
        "seed_mode": str(args.seed_mode),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "max_response_tokens": int(args.max_response_tokens),
        "use_chat_template": bool(use_chat_template),
        "max_model_len": int(args.max_model_len),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "batch_size": int(args.batch_size),
        "max_num_seqs": int(args.max_num_seqs),
        "num_shards": int(args.num_shards),
        "sampler_version": str(sampler_version),
    }
    if eval_config_path.exists():
        previous = json.loads(eval_config_path.read_text(encoding="utf-8"))
        # Refuse to "resume" across different sampling settings; this would silently mix incompatible caches.
        key_fields = [
            "model",
            "tokenizer",
            "template_sha256_12",
            "samples_per_subset",
            "seed",
            "seed_mode",
            "temperature",
            "top_p",
            "sampler_version",
            "max_subsets_per_row",
            "limit_rows",
        ]
        mismatches = []
        for k in key_fields:
            if previous.get(k) != current_eval_config.get(k):
                mismatches.append((k, previous.get(k), current_eval_config.get(k)))
        if mismatches and not bool(args.no_resume):
            lines = ["Existing eval_config.json does not match current args (refusing to resume):"]
            for k, a, b in mismatches[:20]:
                lines.append(f"  {k}: {a!r} vs {b!r}")
            lines.append("Use a new --out_dir (recommended) or pass --no_resume to force recomputation.")
            raise SystemExit("\n".join(lines))
    else:
        tmp = eval_config_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(current_eval_config, f, indent=2, sort_keys=True)
        os.replace(tmp, eval_config_path)

    if args.sanity_only:
        # Hard gate: one-candidate sanity check.
        _run_sanity_check(
            llm=llm,
            tokenizer=tokenizer,
            template=template,
            rows_by_id=rows_by_id,
            plan=plan,
            n_sanity_rows=int(args.sanity_rows),
            sanity_cases_per_row=int(args.sanity_cases_per_row),
            samples_per_case=int(args.samples_per_subset),
            max_prompt_tokens=int(args.max_prompt_tokens),
            max_response_tokens=int(args.max_response_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=int(args.seed),
            seed_mode=str(args.seed_mode),
            use_chat_template=bool(use_chat_template),
        )
        print("[OK] sanity_only: one-candidate sanity check passed.")
        return

    if not bool(args.skip_sanity):
        # Hard gate: one-candidate sanity check before any large evaluation.
        _run_sanity_check(
            llm=llm,
            tokenizer=tokenizer,
            template=template,
            rows_by_id=rows_by_id,
            plan=plan,
            n_sanity_rows=int(args.sanity_rows),
            sanity_cases_per_row=int(args.sanity_cases_per_row),
            samples_per_case=int(args.samples_per_subset),
            max_prompt_tokens=int(args.max_prompt_tokens),
            max_response_tokens=int(args.max_response_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=int(args.seed),
            seed_mode=str(args.seed_mode),
            use_chat_template=bool(use_chat_template),
        )

    num_shards = int(args.num_shards)
    shard_idx = int(args.shard_idx)
    if num_shards <= 0:
        raise ValueError("--num_shards must be > 0")
    if not (0 <= shard_idx < num_shards):
        raise ValueError("--shard_idx must satisfy 0 <= shard_idx < num_shards")

    # Filter plan to this shard.
    shard_plan: list[SubsetPlan] = []
    for s in plan:
        h = _stable_int_hash(str(s.row_id), str(s.subset_id), mod=num_shards)
        if h == shard_idx:
            shard_plan.append(s)

    results_path = out_dir / f"results_shard{shard_idx:02d}of{num_shards:02d}.jsonl"
    done: set[tuple[int, str]] = set()
    resume = not bool(args.no_resume)
    if resume and results_path.exists():
        for rec in _read_jsonl(results_path):
            try:
                done.add((int(rec["row_id"]), str(rec["subset_id"])))
            except Exception:
                continue

    pending = [s for s in shard_plan if (int(s.row_id), str(s.subset_id)) not in done]
    print(f"[PLAN] rows={len(rows_by_id)} total_subsets={len(plan)} shard_subsets={len(shard_plan)} pending={len(pending)}")

    base_sampling_kwargs = dict(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=-1,
        min_p=0.0,
        max_tokens=int(args.max_response_tokens),
        repetition_penalty=1.0,
        detokenize=True,
    )

    n_per_prompt = int(args.samples_per_subset)
    if n_per_prompt <= 0:
        raise ValueError("--samples_per_subset must be > 0")

    batch_size = int(args.batch_size)
    max_num_seqs = int(args.max_num_seqs)
    # Ensure batch_size * n_gen <= max_num_seqs; chunk if needed.
    k_chunk = max(1, min(n_per_prompt, max_num_seqs // max(1, batch_size)))

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_batches = math.ceil(len(pending) / batch_size) if pending else 0
    with results_path.open("a", encoding="utf-8") as out_f:
        audit_path = out_dir / f"audit_shard{shard_idx:02d}of{num_shards:02d}.jsonl"
        audit_written = 0
        audit_max = int(args.audit_max_records)
        audit_f = audit_path.open("a", encoding="utf-8") if audit_max > 0 else None
        for batch_idx, (start, end) in enumerate(_iter_batches(len(pending), batch_size), start=1):
            batch = pending[start:end]
            messages_list = []
            for s in batch:
                row = rows_by_id[int(s.row_id)]
                messages_list.append(_render_prompt_messages(template, row, s))

            prompt_token_ids = _build_prompt_token_ids(
                tokenizer,
                messages_list,
                max_prompt_tokens=int(args.max_prompt_tokens),
                use_chat_template=bool(use_chat_template),
            )
            vllm_inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids]

            batch_sample_texts: list[list[str]] = [[] for _ in batch]
            for chunk_idx, chunk_start in enumerate(range(0, n_per_prompt, k_chunk)):
                n_gen = min(k_chunk, n_per_prompt - chunk_start)
                if args.seed_mode == "per_prompt":
                    seed_stride = 1_000_000
                    sampling_params = [
                        SamplingParams(seed=_stable_prompt_seed(int(args.seed), s) + chunk_idx * seed_stride, n=n_gen, **base_sampling_kwargs)
                        for s in batch
                    ]
                    outs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
                else:
                    outs = llm.generate(prompts=vllm_inputs, sampling_params=SamplingParams(n=n_gen, **base_sampling_kwargs), use_tqdm=False)

                if len(outs) != len(batch):
                    raise RuntimeError(f"Expected {len(batch)} outputs, got {len(outs)}")
                for i, out in enumerate(outs):
                    if len(out.outputs) != n_gen:
                        raise RuntimeError(f"Expected n={n_gen} outputs per prompt, got {len(out.outputs)}")
                    batch_sample_texts[i].extend([o.text for o in out.outputs])

            # Score + write results.
            for s, sample_texts in zip(batch, batch_sample_texts):
                cand = set(s.candidate_moves_uci)
                n_format_ok = 0
                n_in_subset = 0
                n_correct = 0
                n_bad_move = 0
                n_format_err = 0
                pred_move_counts: dict[str, int] = {}
                sample_summaries: list[dict[str, Any]] = []

                for txt in sample_texts:
                    pred, fmt_ok, err = _parse_completion(txt)
                    if fmt_ok:
                        n_format_ok += 1
                    if err == "bad_move":
                        n_bad_move += 1
                    if err == "format_error":
                        n_format_err += 1
                    if pred:
                        pred_move_counts[pred] = pred_move_counts.get(pred, 0) + 1
                        if pred in cand:
                            n_in_subset += 1
                        if fmt_ok and (pred == s.target_move_uci) and (pred in cand):
                            n_correct += 1
                    sample_summaries.append(
                        {
                            "pred_move_uci": pred,
                            "format_ok": bool(fmt_ok),
                            "error": err,
                            "in_subset": bool(pred) and (pred in cand),
                            "correct": bool(fmt_ok) and bool(pred) and (pred == s.target_move_uci) and (pred in cand),
                            "raw_output": txt,
                        }
                    )

                pass_at_k = 1 if n_correct > 0 else 0
                success_rate = float(n_correct) / float(n_per_prompt)
                most_common_pred = ""
                if pred_move_counts:
                    most_common_pred = sorted(pred_move_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

                rec = {
                    "row_id": int(s.row_id),
                    "subset_id": str(s.subset_id),
                    "k": int(s.k),
                    "candidate_moves_uci": list(s.candidate_moves_uci),
                    "target_move_uci": str(s.target_move_uci),
                    "includes_global_best": bool(s.includes_global_best),
                    "mu_best": float(s.mu_best),
                    "mu_second": float(s.mu_second),
                    "top_margin": float(s.top_margin),
                    "h1": float(s.h1),
                    "n_samples": int(n_per_prompt),
                    "n_format_ok": int(n_format_ok),
                    "n_in_subset": int(n_in_subset),
                    "n_correct": int(n_correct),
                    "pass_at_8": int(pass_at_k) if n_per_prompt == 8 else int(pass_at_k),
                    "success_rate": float(success_rate),
                    "n_bad_move": int(n_bad_move),
                    "n_format_error": int(n_format_err),
                    "most_common_pred_move": str(most_common_pred),
                    "seed": int(args.seed),
                    "sampler_version": str(s.sampler_version),
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Audit: keep a small, capped sample of compliance failures for debugging.
                if (
                    audit_f is not None
                    and audit_written < audit_max
                    and (
                        n_format_ok < n_per_prompt
                        or n_in_subset < n_per_prompt
                        or n_bad_move > 0
                        or n_format_err > 0
                    )
                ):
                    row = rows_by_id[int(s.row_id)]
                    messages = _render_prompt_messages(template, row, s)
                    try:
                        prompt_text = render_prompt_from_messages(
                            tokenizer,
                            messages,
                            use_chat_template=bool(use_chat_template),
                            add_generation_prompt=True,
                        )
                    except Exception:
                        _, ids = encode_prompt_from_messages(
                            tokenizer,
                            messages,
                            use_chat_template=bool(use_chat_template),
                            add_generation_prompt=True,
                        )
                        prompt_text = str(tokenizer.decode(ids, skip_special_tokens=False))
                    audit_rec = {
                        "row_id": int(s.row_id),
                        "subset_id": str(s.subset_id),
                        "k": int(s.k),
                        "fen": str(row.fen),
                        "candidate_moves_uci": list(s.candidate_moves_uci),
                        "target_move_uci": str(s.target_move_uci),
                        "n_samples": int(n_per_prompt),
                        "n_format_ok": int(n_format_ok),
                        "n_in_subset": int(n_in_subset),
                        "n_correct": int(n_correct),
                        "prompt_text": prompt_text,
                        "samples": sample_summaries,
                    }
                    audit_f.write(json.dumps(audit_rec, ensure_ascii=False) + "\n")
                    audit_f.flush()
                    audit_written += 1
            out_f.flush()

            elapsed = time.time() - t0
            print(f"[{batch_idx:>4}/{total_batches}] subsets {start}:{end} elapsed={elapsed/60:.1f}min")

        if audit_f is not None:
            audit_f.close()

    print(f"[DONE] wrote {results_path}")


if __name__ == "__main__":
    main()
