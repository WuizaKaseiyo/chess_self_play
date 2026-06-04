#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
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
class TriplePlan:
    row_id: int
    triple_unordered_id: str
    triple_id: str
    candidate_moves_uci: list[str]  # length 3, ordered
    target_move_uci: str  # argmax μ within the triple (tie-break lexicographic UCI)
    mu_best: float
    mu_second: float
    mu_third: float
    top_margin: float  # μ_best - μ_second >= 0
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

        # Fallback to move_values_json if expected scores are missing.
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


def _triple_unordered_id(moves: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(m).strip().lower() for m in moves))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _triple_id_from_ordered_moves(moves: list[str]) -> str:
    payload = "\n".join(moves)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


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


def _render_prompt_messages(template: Any, *, row: RowData, candidate_moves_uci: list[str]) -> list[dict[str, str]]:
    prompt_text = str(
        template.render(
            FEN=row.fen,
            legal_moves_uci_list=row.legal_moves_uci,
            considered_moves_uci_list=candidate_moves_uci,
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


def _stable_prompt_seed(global_seed: int, row_id: int, triple_id: str) -> int:
    # vLLM expects an int32 seed; keep it in range.
    h = hashlib.sha256()
    h.update(str(global_seed).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(row_id).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(triple_id).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big", signed=False)


def _compute_h1(mu_best: float, other_mus: list[float], *, eps: float) -> float:
    if eps <= 0:
        raise ValueError("eps must be > 0")
    acc = 0.0
    for mu_i in other_mus:
        delta = float(mu_best) - float(mu_i)
        if delta < eps:
            delta = eps
        acc += 1.0 / (delta * delta)
    return float(acc)


def _sample_unordered_triples_for_row(row: RowData, *, max_triples: int, seed: int) -> list[tuple[str, str, str]]:
    """Return a list of unordered triples (each sorted lexicographically)."""
    moves = list(row.legal_moves_uci)
    n = len(moves)
    if n < 3:
        return []
    if max_triples <= 0:
        raise ValueError("--max_triples_per_row must be > 0")

    mu_map = row.mu_map
    ranked = sorted(moves, key=lambda mv: (-float(mu_map.get(mv, -float("inf"))), mv))
    global_best = ranked[0]

    rng = np.random.default_rng(int(seed) + int(row.row_id))
    triples: set[tuple[str, str, str]] = set()

    # Always include a couple deterministic extremes (if possible).
    if len(ranked) >= 3:
        triples.add(tuple(sorted((ranked[0], ranked[1], ranked[2]))))
    if len(ranked) >= 3:
        triples.add(tuple(sorted((global_best, ranked[-1], ranked[-2]))))

    top_pool = [m for m in ranked[: min(10, n)] if m != global_best]
    bottom_pool = [m for m in ranked[max(0, n - min(12, n)) :] if m != global_best]
    other_pool = [m for m in moves if m != global_best]

    # Budget across categories (sum <= max_triples).
    n_easy = int(round(max_triples * 0.30))
    n_hard = int(round(max_triples * 0.30))
    n_excl = int(round(max_triples * 0.20))
    n_rand = max(0, int(max_triples) - n_easy - n_hard - n_excl)

    def add_easy() -> None:
        pool = bottom_pool if len(bottom_pool) >= 2 else other_pool
        if len(pool) < 2:
            return
        a, b = rng.choice(pool, size=2, replace=False).tolist()
        triples.add(tuple(sorted((global_best, str(a), str(b)))))

    def add_hard() -> None:
        pool = top_pool if len(top_pool) >= 2 else other_pool
        if len(pool) < 2:
            return
        a, b = rng.choice(pool, size=2, replace=False).tolist()
        triples.add(tuple(sorted((global_best, str(a), str(b)))))

    def add_excl_best() -> None:
        pool = top_pool if len(top_pool) >= 3 else other_pool
        if len(pool) < 3:
            return
        a, b, c = rng.choice(pool, size=3, replace=False).tolist()
        triples.add(tuple(sorted((str(a), str(b), str(c)))))

    def add_rand() -> None:
        if len(moves) < 3:
            return
        a, b, c = rng.choice(moves, size=3, replace=False).tolist()
        triples.add(tuple(sorted((str(a), str(b), str(c)))))

    # Populate with capped attempts per bucket (avoid pathological infinite loops with small n).
    def fill(target_count: int, add_fn) -> None:
        attempts = 0
        max_attempts = max(100, target_count * 50)
        while len(triples) < target_count and attempts < max_attempts:
            add_fn()
            attempts += 1

    start_count = len(triples)
    fill(start_count + n_easy, add_easy)
    fill(start_count + n_easy + n_hard, add_hard)
    fill(start_count + n_easy + n_hard + n_excl, add_excl_best)
    fill(start_count + n_easy + n_hard + n_excl + n_rand, add_rand)

    # Trim deterministically if we overshot.
    ordered = sorted(triples)
    return ordered[: int(max_triples)]


def _build_triple_plan_for_row(row: RowData, *, max_triples: int, seed: int, sampler_version: str) -> list[TriplePlan]:
    if len(row.legal_moves_uci) < 3:
        return []

    mu_map = row.mu_map
    legal_moves = list(row.legal_moves_uci)
    global_best_move, _mu_best_global = _best_move_by_mu(mu_map, legal_moves)
    global_best_move = str(global_best_move)

    triples = _sample_unordered_triples_for_row(row, max_triples=max_triples, seed=seed)
    sampler_seed = int(seed) + int(row.row_id)

    out: list[TriplePlan] = []
    for a, b, c in triples:
        unordered_moves = [a, b, c]
        unordered_id = _triple_unordered_id(unordered_moves)

        # Compute target + metrics (order-invariant).
        target, mu_best = _best_move_by_mu(mu_map, unordered_moves)
        remaining = [m for m in unordered_moves if m != target]
        second, mu_second = _best_move_by_mu(mu_map, remaining)
        third = remaining[0] if remaining[1] == second else remaining[1]
        mu_third = float(mu_map.get(third, -float("inf")))

        top_margin = float(mu_best) - float(mu_second)
        if top_margin < 0:
            top_margin = abs(top_margin)
        h1 = _compute_h1(float(mu_best), [float(mu_second), float(mu_third)], eps=1e-6)
        includes_global_best = bool(global_best_move in {a, b, c})

        # Evaluate all 6 prompt orders.
        for perm in itertools.permutations([a, b, c], 3):
            cand = [str(x) for x in perm]
            triple_id = _triple_id_from_ordered_moves(cand)
            out.append(
                TriplePlan(
                    row_id=int(row.row_id),
                    triple_unordered_id=str(unordered_id),
                    triple_id=str(triple_id),
                    candidate_moves_uci=list(cand),
                    target_move_uci=str(target),
                    mu_best=float(mu_best),
                    mu_second=float(mu_second),
                    mu_third=float(mu_third),
                    top_margin=float(top_margin),
                    h1=float(h1),
                    includes_global_best=bool(includes_global_best),
                    sampler_version=str(sampler_version),
                    sampler_seed=int(sampler_seed),
                )
            )
    return out


def _load_or_build_plan(
    rows: list[RowData],
    *,
    out_dir: Path,
    parquet_path: str,
    max_triples_per_row: int,
    seed: int,
    sampler_version: str,
) -> tuple[dict[int, RowData], list[TriplePlan], dict[str, Any]]:
    rows_by_id = {r.row_id: r for r in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError("Duplicate row_id detected in loaded rows")

    plan_path = out_dir / "triples.jsonl"
    rows_path = out_dir / "rows.jsonl"
    manifest_path = out_dir / "manifest.json"

    if plan_path.exists() and rows_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "parquet_rows": int(manifest.get("parquet_rows") or 0),
            "max_triples_per_row": int(manifest.get("max_triples_per_row") or 0),
            "seed": int(manifest.get("seed") or 0),
            "sampler_version": str(manifest.get("sampler_version") or ""),
        }
        got = {
            "parquet_rows": len(rows),
            "max_triples_per_row": int(max_triples_per_row),
            "seed": int(seed),
            "sampler_version": str(sampler_version),
        }
        if expected["parquet_rows"] and expected["parquet_rows"] != got["parquet_rows"]:
            raise ValueError(f"Existing plan has parquet_rows={expected['parquet_rows']} but got {got['parquet_rows']}. Use a new --out_dir.")
        if expected["max_triples_per_row"] and expected["max_triples_per_row"] != got["max_triples_per_row"]:
            raise ValueError(
                f"Existing plan has max_triples_per_row={expected['max_triples_per_row']} but got {got['max_triples_per_row']}. Use a new --out_dir."
            )
        if expected["seed"] != got["seed"] or expected["sampler_version"] != got["sampler_version"]:
            raise ValueError(
                "Existing plan manifest mismatch (seed/sampler_version):\n"
                f"  expected: {expected}\n"
                f"  got:      {got}\n"
                "Use a new --out_dir to avoid mixing caches."
            )
        plan: list[TriplePlan] = []
        for rec in _read_jsonl(plan_path):
            plan.append(TriplePlan(**rec))
        missing = sorted({p.row_id for p in plan}.difference(rows_by_id.keys()))
        if missing:
            raise ValueError(f"Plan references missing row_ids: {missing[:10]}")
        return rows_by_id, plan, manifest

    out_dir.mkdir(parents=True, exist_ok=True)
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
    plan: list[TriplePlan] = []
    total_unordered = 0

    for r in rows:
        triples = _sample_unordered_triples_for_row(r, max_triples=max_triples_per_row, seed=seed)
        total_unordered += int(len(triples))
        plan.extend(_build_triple_plan_for_row(r, max_triples=max_triples_per_row, seed=seed, sampler_version=sampler_version))

    # De-duplicate ordered triple_ids (defensive).
    seen: set[tuple[int, str]] = set()
    deduped: list[TriplePlan] = []
    for p in plan:
        key = (int(p.row_id), str(p.triple_id))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
        plan_records.append(dataclasses.asdict(p))
    plan = deduped

    _write_jsonl(plan_path, plan_records)

    manifest = {
        "parquet": str(os.path.abspath(str(parquet_path))),
        "parquet_rows": int(len(rows)),
        "rows": int(len(rows)),
        "max_triples_per_row": int(max_triples_per_row),
        "sampler_version": str(sampler_version),
        "seed": int(seed),
        "total_triples_unordered": int(total_unordered),
        "total_triples_ordered": int(len(plan)),
    }
    tmp = manifest_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, manifest_path)

    return rows_by_id, plan, manifest


def _run_one_candidate_sanity_check(
    *,
    llm: Any,
    tokenizer: Any,
    template: Any,
    rows: list[RowData],
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
    if n_sanity_rows <= 0:
        raise ValueError("n_sanity_rows must be > 0")
    if sanity_cases_per_row <= 0:
        raise ValueError("sanity_cases_per_row must be > 0")
    if samples_per_case <= 0:
        raise ValueError("samples_per_case must be > 0")

    rng = np.random.default_rng(int(seed))
    sanity_rows = rows[: min(len(rows), int(n_sanity_rows))]
    cases: list[tuple[RowData, str]] = []
    for r in sanity_rows:
        if not r.legal_moves_uci:
            continue
        moves = list(r.legal_moves_uci)
        rng.shuffle(moves)
        for mv in moves[: int(sanity_cases_per_row)]:
            cases.append((r, str(mv)))

    messages_list: list[list[dict[str, str]]] = []
    for r, mv in cases:
        messages_list.append(_render_prompt_messages(template, row=r, candidate_moves_uci=[mv]))

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

    max_num_seqs = int(getattr(llm, "max_num_seqs", 1024))
    k_chunk = max(1, min(int(samples_per_case), max_num_seqs // max(1, len(cases))))
    k_chunk = int(samples_per_case) if (len(cases) * int(samples_per_case) <= max_num_seqs) else k_chunk

    all_sample_texts: list[list[str]] = [[] for _ in cases]
    for chunk_idx, _chunk_start in enumerate(range(0, int(samples_per_case), k_chunk)):
        n_gen = min(k_chunk, int(samples_per_case) - chunk_idx * k_chunk)
        if seed_mode == "per_prompt":
            seed_stride = 1_000_000
            sampling_params = [
                SamplingParams(seed=_stable_prompt_seed(seed, r.row_id, mv) + chunk_idx * seed_stride, n=n_gen, **base_sampling_kwargs) for r, mv in cases
            ]
            outs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
        else:
            outs = llm.generate(prompts=vllm_inputs, sampling_params=SamplingParams(n=n_gen, **base_sampling_kwargs), use_tqdm=False)

        if len(outs) != len(cases):
            raise RuntimeError(f"Expected {len(cases)} outputs, got {len(outs)}")
        for i, out in enumerate(outs):
            if len(out.outputs) != n_gen:
                raise RuntimeError(f"Sanity: expected n={n_gen} outputs, got {len(out.outputs)}")
            all_sample_texts[i].extend([o.text for o in out.outputs])

    failures: list[dict[str, Any]] = []
    for (r, only_mv), sample_texts in zip(cases, all_sample_texts):
        cand_set = {only_mv}
        for j, txt in enumerate(sample_texts):
            pred, fmt_ok, err = _parse_completion(txt)
            in_subset = bool(pred) and (pred in cand_set)
            ok = fmt_ok and in_subset and (pred == only_mv)
            if not ok:
                prompt_text = render_prompt_from_messages(
                    tokenizer,
                    _render_prompt_messages(template, row=r, candidate_moves_uci=[only_mv]),
                    use_chat_template=use_chat_template,
                    add_generation_prompt=True,
                )
                failures.append(
                    {
                        "row_id": int(r.row_id),
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
    ap.add_argument(
        "--out_dir", default=None, help="Defaults to outputs/select_triples_eval/<model>__<templatehash>_seed<seed>_rows<limit_rows>_triples<max_triples_per_row>"
    )

    ap.add_argument("--limit_rows", type=int, default=100)
    ap.add_argument("--max_triples_per_row", type=int, default=200)
    ap.add_argument("--samples_per_triple", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed_mode", choices=["engine", "per_prompt"], default="per_prompt")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--max_prompt_tokens", type=int, default=1024)
    ap.add_argument("--max_response_tokens", type=int, default=512)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_num_seqs", type=int, default=1024)

    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--audit_max_records", type=int, default=200)

    ap.add_argument("--sanity_only", action="store_true")
    ap.add_argument("--skip_sanity", action="store_true")
    ap.add_argument("--sanity_rows", type=int, default=5)
    ap.add_argument("--sanity_cases_per_row", type=int, default=2)

    args = ap.parse_args()

    template = _load_template(str(args.template_path))
    template_hash = _sha256_text(Path(args.template_path).read_text(encoding="utf-8"))[:12]

    rows = _load_rows(str(args.parquet), int(args.limit_rows))
    if not rows:
        raise SystemExit("No rows loaded (check --limit_rows).")

    model_safe = _sanitize_model_name(str(args.model))
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            Path("outputs")
            / "select_triples_eval"
            / f"{model_safe}__{template_hash}_seed{int(args.seed)}_rows{int(args.limit_rows)}_triples{int(args.max_triples_per_row)}_samples{int(args.samples_per_triple)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    sampler_version = "select_triples_sampled_v1"
    rows_by_id, plan, plan_manifest = _load_or_build_plan(
        rows,
        out_dir=out_dir,
        parquet_path=str(args.parquet),
        max_triples_per_row=int(args.max_triples_per_row),
        seed=int(args.seed),
        sampler_version=str(sampler_version),
    )

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

    eval_config_path = out_dir / "eval_config.json"
    current_eval_config = {
        "model": str(args.model),
        "tokenizer": str(tokenizer_model),
        "template_path": str(args.template_path),
        "template_sha256_12": str(template_hash),
        "parquet": str(os.path.abspath(str(args.parquet))),
        "limit_rows": int(args.limit_rows),
        "max_triples_per_row": int(args.max_triples_per_row),
        "samples_per_triple": int(args.samples_per_triple),
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
        "plan_manifest": plan_manifest,
    }
    if eval_config_path.exists():
        previous = json.loads(eval_config_path.read_text(encoding="utf-8"))
        key_fields = [
            "model",
            "tokenizer",
            "template_sha256_12",
            "samples_per_triple",
            "seed",
            "seed_mode",
            "temperature",
            "top_p",
            "limit_rows",
            "max_triples_per_row",
            "sampler_version",
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

    if bool(args.sanity_only):
        _run_one_candidate_sanity_check(
            llm=llm,
            tokenizer=tokenizer,
            template=template,
            rows=rows,
            n_sanity_rows=int(args.sanity_rows),
            sanity_cases_per_row=int(args.sanity_cases_per_row),
            samples_per_case=int(args.samples_per_triple),
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
        _run_one_candidate_sanity_check(
            llm=llm,
            tokenizer=tokenizer,
            template=template,
            rows=rows,
            n_sanity_rows=int(args.sanity_rows),
            sanity_cases_per_row=int(args.sanity_cases_per_row),
            samples_per_case=int(args.samples_per_triple),
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

    shard_plan: list[TriplePlan] = []
    for p in plan:
        h = _stable_int_hash(str(p.row_id), str(p.triple_id), mod=num_shards)
        if h == shard_idx:
            shard_plan.append(p)

    results_path = out_dir / f"results_shard{shard_idx:02d}of{num_shards:02d}.jsonl"
    done: set[tuple[int, str]] = set()
    resume = not bool(args.no_resume)
    if resume and results_path.exists():
        for rec in _read_jsonl(results_path):
            try:
                done.add((int(rec["row_id"]), str(rec["triple_id"])))
            except Exception:
                continue

    pending = [p for p in shard_plan if (int(p.row_id), str(p.triple_id)) not in done]
    print(
        f"[PLAN] rows={len(rows_by_id)} total_triples_ordered={len(plan)} shard_triples={len(shard_plan)} pending={len(pending)}"
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

    n_per_prompt = int(args.samples_per_triple)
    if n_per_prompt <= 0:
        raise ValueError("--samples_per_triple must be > 0")

    batch_size = int(args.batch_size)
    max_num_seqs = int(args.max_num_seqs)
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
            messages_list: list[list[dict[str, str]]] = []
            for p in batch:
                row = rows_by_id[int(p.row_id)]
                messages_list.append(_render_prompt_messages(template, row=row, candidate_moves_uci=p.candidate_moves_uci))

            prompt_token_ids = _build_prompt_token_ids(
                tokenizer,
                messages_list,
                max_prompt_tokens=int(args.max_prompt_tokens),
                use_chat_template=bool(use_chat_template),
            )
            vllm_inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids]

            batch_sample_texts: list[list[str]] = [[] for _ in batch]
            for chunk_idx, _chunk_start in enumerate(range(0, n_per_prompt, k_chunk)):
                n_gen = min(k_chunk, n_per_prompt - chunk_idx * k_chunk)
                if args.seed_mode == "per_prompt":
                    seed_stride = 1_000_000
                    sampling_params = [
                        SamplingParams(
                            seed=_stable_prompt_seed(int(args.seed), int(p.row_id), str(p.triple_id)) + chunk_idx * seed_stride,
                            n=n_gen,
                            **base_sampling_kwargs,
                        )
                        for p in batch
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

            for p, sample_texts in zip(batch, batch_sample_texts):
                cand = list(p.candidate_moves_uci)
                if len(cand) != 3:
                    raise RuntimeError(f"Expected exactly 3 candidate moves, got {cand}")
                c0, c1, c2 = cand[0], cand[1], cand[2]
                cand_set = {c0, c1, c2}
                target_pos = int(cand.index(p.target_move_uci)) if p.target_move_uci in cand else -1

                n_format_ok = 0
                n_in_subset = 0
                n_correct = 0
                n_bad_move = 0
                n_format_err = 0
                n_pred_pos0 = 0
                n_pred_pos1 = 0
                n_pred_pos2 = 0
                n_pred_other = 0
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
                        if pred == c0:
                            n_pred_pos0 += 1
                        elif pred == c1:
                            n_pred_pos1 += 1
                        elif pred == c2:
                            n_pred_pos2 += 1
                        else:
                            n_pred_other += 1
                        if pred in cand_set:
                            n_in_subset += 1
                    if fmt_ok and pred and (pred == p.target_move_uci) and (pred in cand_set):
                        n_correct += 1
                    sample_summaries.append(
                        {
                            "pred_move_uci": pred,
                            "format_ok": bool(fmt_ok),
                            "error": err,
                            "in_subset": bool(pred) and (pred in cand_set),
                            "correct": bool(fmt_ok) and bool(pred) and (pred == p.target_move_uci) and (pred in cand_set),
                            "raw_output": txt,
                        }
                    )

                success_rate = float(n_correct) / float(n_per_prompt)
                pass_at_n = 1 if n_correct > 0 else 0
                most_common_pred = ""
                if pred_move_counts:
                    most_common_pred = sorted(pred_move_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

                rec = {
                    "row_id": int(p.row_id),
                    "triple_unordered_id": str(p.triple_unordered_id),
                    "triple_id": str(p.triple_id),
                    "candidate_moves_uci": list(p.candidate_moves_uci),
                    "target_move_uci": str(p.target_move_uci),
                    "target_pos": int(target_pos),
                    "includes_global_best": bool(p.includes_global_best),
                    "mu_best": float(p.mu_best),
                    "mu_second": float(p.mu_second),
                    "mu_third": float(p.mu_third),
                    "top_margin": float(p.top_margin),
                    "h1": float(p.h1),
                    "n_samples": int(n_per_prompt),
                    "n_format_ok": int(n_format_ok),
                    "n_in_subset": int(n_in_subset),
                    "n_correct": int(n_correct),
                    "pass_at_8": int(pass_at_n) if n_per_prompt == 8 else int(pass_at_n),
                    "success_rate": float(success_rate),
                    "n_bad_move": int(n_bad_move),
                    "n_format_error": int(n_format_err),
                    "n_pred_pos0": int(n_pred_pos0),
                    "n_pred_pos1": int(n_pred_pos1),
                    "n_pred_pos2": int(n_pred_pos2),
                    "n_pred_other": int(n_pred_other),
                    "most_common_pred_move": str(most_common_pred),
                    "seed": int(args.seed),
                    "sampler_version": str(p.sampler_version),
                    "sampler_seed": int(p.sampler_seed),
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if (
                    audit_f is not None
                    and audit_written < audit_max
                    and (
                        n_format_ok < n_per_prompt
                        or n_in_subset < n_per_prompt
                        or n_bad_move > 0
                        or n_format_err > 0
                        or n_pred_other > 0
                    )
                ):
                    row = rows_by_id[int(p.row_id)]
                    messages = _render_prompt_messages(template, row=row, candidate_moves_uci=p.candidate_moves_uci)
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
                        "row_id": int(p.row_id),
                        "triple_id": str(p.triple_id),
                        "triple_unordered_id": str(p.triple_unordered_id),
                        "candidate_moves_uci": list(p.candidate_moves_uci),
                        "target_move_uci": str(p.target_move_uci),
                        "target_pos": int(target_pos),
                        "prompt_text": prompt_text,
                        "sample_summaries": sample_summaries,
                    }
                    audit_f.write(json.dumps(audit_rec, ensure_ascii=False) + "\n")
                    audit_written += 1

            elapsed_min = (time.time() - t0) / 60.0
            if total_batches:
                print(f"[{batch_idx:>4}/{total_batches}] triples {start}:{end} elapsed={elapsed_min:.1f}min")

        if audit_f is not None:
            audit_f.close()

    print(f"[DONE] wrote {results_path}")


if __name__ == "__main__":
    main()
