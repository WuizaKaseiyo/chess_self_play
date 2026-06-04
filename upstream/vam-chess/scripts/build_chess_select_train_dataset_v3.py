#!/usr/bin/env python3
"""
Build a restricted-moves ("selection") chess *training* parquet with a curriculum over
the **position of the μ-best move** in the AIcrowd / python-chess legal-move order.

v3 intent (curriculum to fight presentation-order bias):
- Test-time (AIcrowd): allowed_moves == legal_moves in python-chess order (no shuffle).
- Training should learn to pick the μ-best move even when it appears late in that list.

We implement an *offline* curriculum by constructing a deterministic dataset where:
- During an "anneal" prefix of training, we only sample positions where the μ-best move
  appears within the first K moves of `legal_moves_uci` (K grows by stage).
- After anneal, we sample from the full distribution (no constraint).

This script supports two dataset flavors:
  1) full-legal-only: only `considered_moves_uci = legal_moves_uci`.
  2) mixture (v2-style): per position, include full-legal + (hard_neg) + (coverage_block) rows.

Notes:
- Candidate lists are ALWAYS ordered as subsequences of `legal_moves_uci` (no shuffle).
- The target is μ-best (expected-score preferred; else move_values).
- We overwrite `extra_info.index` with a stable derived id and store provenance under
  `extra_info.source_index`, plus curriculum metadata fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment

# Ensure local namespace packages (e.g., `recipe/`) resolve when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipe.chess.reward_fn import _to_uci

BUILDER_VERSION = "select-v3-2026-01-16"


def _stable_int_hash(*parts: str, mod: int) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], "big") % int(mod)


def _atomic_write_parquet(table: pa.Table, *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    pq.write_table(table, tmp)
    tmp.replace(output_path)


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


def _mu_sorted_desc(mu_map: dict[str, float], moves: Sequence[str]) -> list[str]:
    def _key(m: str) -> tuple[float, str]:
        mm = str(m).strip().lower()
        return (-float(mu_map.get(mm, -float("inf"))), mm)

    return sorted([str(m).strip().lower() for m in moves], key=_key)


def _legal_order_subsequence(legal: list[str], candidate_set: set[str]) -> list[str]:
    return [m for m in legal if m in candidate_set]


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
    stage: int,
    stage_pos: int,
    variant: str,
    sub_id: int,
) -> int:
    # Use 63-bit positive ints to avoid awkward signed-overflow across runtimes.
    return _stable_int_hash(
        str(global_seed),
        str(source_index),
        str(stage),
        str(stage_pos),
        str(variant),
        str(sub_id),
        BUILDER_VERSION,
        mod=(2**63 - 1),
    )


def _parse_fraction(text: str) -> Fraction:
    try:
        f = Fraction(str(text))
    except Exception as e:
        raise ValueError(f"Invalid fraction '{text}' (expected e.g. '0.5' or '1/2'): {e}") from e
    if f <= 0 or f >= 1:
        raise ValueError(f"anneal_frac must be in (0,1); got {f}")
    return f


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


def _load_base_rows(parquet_path: Path, *, limit_rows: int | None) -> list[BaseRow]:
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
        fen = str(rm.get("fen") or "").strip()
        legal_moves = _normalize_moves(rm.get("legal_moves_uci"))
        gt = _to_uci(str(rm.get("ground_truth") or ""))
        if gt is None:
            gt = ""
        mu_map = _parse_mu_map(rm, row_id=source_index)

        if not fen:
            raise ValueError(f"Row {source_index}: empty fen")
        if not legal_moves:
            raise ValueError(f"Row {source_index}: empty legal_moves_uci")

        out.append(
            BaseRow(
                source_index=source_index,
                fen=fen,
                legal_moves_uci=_dedup_preserve_order(legal_moves),
                ground_truth_uci=str(gt),
                mu_map=mu_map,
                raw_row=rr,
            )
        )
    return out


def _build_hard_negatives(
    row: BaseRow,
    *,
    global_target: str,
    L_mu: int,
    max_negatives: int,
    ensure_ground_truth: bool,
) -> list[str]:
    legal = row.legal_moves_uci
    mu_map = row.mu_map

    mu_desc = _mu_sorted_desc(mu_map, [m for m in legal if m != global_target])
    mu_negs = mu_desc[: max(0, int(L_mu))]

    negs = _dedup_preserve_order(list(mu_negs))
    if max_negatives > 0:
        negs = negs[: int(max_negatives)]

    candidate_set = set(negs)
    candidate_set.add(global_target)
    if ensure_ground_truth and row.ground_truth_uci:
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
    legal_set = set(legal)

    excluded = set(excluded_moves)
    excluded.add(global_target)
    if ensure_ground_truth and row.ground_truth_uci:
        excluded.add(row.ground_truth_uci)

    remaining = [m for m in legal if m not in excluded]
    blocks = _partition_mu_stratified(remaining, mu_map=row.mu_map, block_size=int(block_size))
    if not blocks:
        return []

    chosen_idx = _sample_indices(len(blocks), k=int(num_blocks), seed=int(global_seed) + 97 * int(row.source_index))
    out: list[list[str]] = []
    for bi in chosen_idx:
        group = blocks[int(bi)]
        cand_set = set(group)
        cand_set.add(global_target)
        if ensure_ground_truth and row.ground_truth_uci:
            cand_set.add(row.ground_truth_uci)
        cand_set = cand_set.intersection(legal_set)
        out.append(_legal_order_subsequence(legal, cand_set))
    return out


def _repeat_shuffle_sample(population: list[BaseRow], *, n: int, rng: random.Random) -> list[BaseRow]:
    if n <= 0:
        return []
    if not population:
        raise ValueError("Cannot sample from empty population")
    out: list[BaseRow] = []
    while len(out) < n:
        batch = list(population)
        rng.shuffle(batch)
        out.extend(batch)
    return out[:n]


def _parse_k_schedule(text: str) -> list[int]:
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("--k_schedule must be non-empty (e.g., '6,12,18')")
    ks: list[int] = []
    for p in parts:
        try:
            v = int(p)
        except Exception as e:
            raise ValueError(f"Invalid k_schedule value '{p}': {e}") from e
        if v <= 0:
            raise ValueError(f"k_schedule entries must be > 0; got {v}")
        ks.append(v)
    return ks


def _stage_sizes_source(
    *,
    base_size: int,
    total_passes: int,
    anneal_frac: Fraction,
    num_k_stages: int,
) -> list[int]:
    if base_size <= 0:
        raise ValueError("base_size must be > 0")
    if total_passes <= 0:
        raise ValueError("total_passes must be > 0")
    if num_k_stages <= 0:
        raise ValueError("num_k_stages must be > 0")

    total = int(base_size) * int(total_passes)
    anneal = int((total * anneal_frac.numerator) // anneal_frac.denominator)
    anneal = max(0, min(total, anneal))
    # Ensure we do *some* anneal rows; otherwise the curriculum is a no-op.
    if anneal <= 0:
        raise ValueError(
            f"anneal_frac={anneal_frac} produced anneal_size=0 with base_size={base_size} total_passes={total_passes}"
        )

    sizes = [anneal // num_k_stages] * num_k_stages
    sizes[-1] = anneal - sum(sizes[:-1])
    sizes.append(total - anneal)  # final full stage
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", type=Path, required=True)
    ap.add_argument("--output_parquet", type=Path, required=True)
    ap.add_argument("--template_path", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit_rows", type=int, default=None)

    ap.add_argument(
        "--dataset_flavor",
        type=str,
        choices=["full_legal_only", "mixture"],
        default="full_legal_only",
        help="Whether to emit only full-legal rows or a v2-style mixture (full_legal+hard_neg+coverage_block).",
    )

    # Curriculum knobs.
    ap.add_argument(
        "--k_schedule",
        type=str,
        default="6,12,18",
        help="Comma-separated K cutoffs for curriculum stages before the final full stage. "
        "K means μ-best rank < K in `legal_moves_uci` (0-indexed rank, so K=6 means positions 0..5).",
    )
    ap.add_argument("--total_passes", type=int, default=5)
    ap.add_argument(
        "--anneal_frac",
        type=str,
        default="1/2",
        help="Fraction of the dataset (in source-row samples) spent in the K-scheduled curriculum stages before switching to full.",
    )

    # Mixture knobs (only used when dataset_flavor=mixture).
    ap.add_argument("--include_hard_neg", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include_coverage", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--L_mu", type=int, default=8)
    ap.add_argument("--block_size", type=int, default=8)
    ap.add_argument("--num_blocks", type=int, default=2)
    ap.add_argument("--max_negatives_hard", type=int, default=12)
    ap.add_argument("--ensure_ground_truth_included", action=argparse.BooleanOptionalAction, default=True)

    args = ap.parse_args()

    input_path: Path = args.input_parquet
    output_path: Path = args.output_parquet
    template_path: Path = args.template_path

    if output_path.exists() and not bool(args.overwrite):
        raise SystemExit(f"Refusing to overwrite existing output_parquet={output_path}; pass --overwrite.")

    template = _load_template(template_path)
    rows = _load_base_rows(input_path, limit_rows=args.limit_rows)
    if not rows:
        raise SystemExit("No rows to process (check --limit_rows).")

    k_schedule = _parse_k_schedule(str(args.k_schedule))
    anneal_frac = _parse_fraction(str(args.anneal_frac))
    stage_sizes = _stage_sizes_source(
        base_size=len(rows),
        total_passes=int(args.total_passes),
        anneal_frac=anneal_frac,
        num_k_stages=len(k_schedule),
    )

    print(
        f"[load] rows={len(rows)} input={input_path} "
        f"dataset_flavor={args.dataset_flavor} total_passes={int(args.total_passes)} anneal_frac={anneal_frac} "
        f"k_schedule={k_schedule} stage_sizes={stage_sizes}"
    )

    out_rows: list[dict[str, Any]] = []

    # Precompute per-row global target + rank once; this is hot in stage filtering.
    per_source: dict[int, tuple[str, int]] = {}
    for r in rows:
        legal = list(r.legal_moves_uci)
        global_target = _best_move_by_mu(r.mu_map, legal)
        try:
            rank = legal.index(global_target)
        except ValueError as e:
            raise ValueError(f"Row {r.source_index}: μ-best target {global_target} not found in legal list") from e
        per_source[int(r.source_index)] = (global_target, int(rank))

    # Emit each stage sequentially so training can disable shuffling and realize the curriculum.
    num_full_stages = len(k_schedule) + 1
    for stage in range(num_full_stages):
        stage_size = int(stage_sizes[stage])
        if stage_size <= 0:
            continue

        if stage < len(k_schedule):
            k = int(k_schedule[stage])
            eligible = [r for r in rows if per_source[int(r.source_index)][1] < k]
            if not eligible:
                raise ValueError(
                    f"Stage {stage} K={k}: eligible set is empty. "
                    "Increase K_start or check target-rank computation."
                )
            stage_desc = f"K<{k}"
        else:
            k = -1
            eligible = list(rows)
            stage_desc = "FULL"

        stage_rng = random.Random(int(args.seed) + 10_000 * stage + 1337)
        sampled = _repeat_shuffle_sample(eligible, n=stage_size, rng=stage_rng)

        n_emit_before = len(out_rows)
        for stage_pos, r in enumerate(sampled):
            legal = list(r.legal_moves_uci)
            legal_set = set(legal)
            global_target, target_rank = per_source[int(r.source_index)]

            derived_plans: list[tuple[str, int, list[str]]] = []
            if args.dataset_flavor == "full_legal_only":
                derived_plans.append(("full_legal", 0, list(legal)))
            elif args.dataset_flavor == "mixture":
                # Always include at least one full-legal row (distribution match).
                derived_plans.append(("full_legal", 0, list(legal)))

                hard_moves: list[str] = []
                if bool(args.include_hard_neg):
                    hard_moves = _build_hard_negatives(
                        r,
                        global_target=global_target,
                        L_mu=int(args.L_mu),
                        max_negatives=int(args.max_negatives_hard),
                        ensure_ground_truth=bool(args.ensure_ground_truth_included),
                    )
                    derived_plans.append(("hard_neg", 0, hard_moves))

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
            else:
                raise AssertionError(f"Unexpected dataset_flavor: {args.dataset_flavor}")

            for variant, sub_id, considered_moves in derived_plans:
                considered_moves = _dedup_preserve_order(considered_moves)
                if not considered_moves:
                    raise ValueError(f"Row {r.source_index} stage={stage} variant={variant}: empty considered_moves")
                considered_set = set(considered_moves)
                if not considered_set.issubset(legal_set):
                    illegal = sorted(list(considered_set - legal_set))[:10]
                    raise ValueError(
                        f"Row {r.source_index} stage={stage} variant={variant}: "
                        f"considered_moves contains illegal moves: {illegal}"
                    )
                if global_target not in considered_set:
                    raise ValueError(
                        f"Row {r.source_index} stage={stage} variant={variant}: missing μ-best target {global_target}"
                    )
                if bool(args.ensure_ground_truth_included) and r.ground_truth_uci and (r.ground_truth_uci not in considered_set):
                    raise ValueError(
                        f"Row {r.source_index} stage={stage} variant={variant}: missing ground_truth {r.ground_truth_uci}"
                    )

                derived_index = _make_derived_index(
                    global_seed=int(args.seed),
                    source_index=int(r.source_index),
                    stage=int(stage),
                    stage_pos=int(stage_pos),
                    variant=str(variant),
                    sub_id=int(sub_id),
                )

                rr = dict(r.raw_row)
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
                ei["curriculum_stage"] = int(stage)
                ei["curriculum_k"] = int(k)
                ei["curriculum_stage_desc"] = str(stage_desc)
                ei["curriculum_stage_pos"] = int(stage_pos)
                ei["global_mu_target_uci"] = str(global_target)
                ei["global_mu_target_rank"] = int(target_rank)
                ei["index"] = int(derived_index)
                rr["extra_info"] = ei

                prompt_text = _render_prompt_text(
                    template, fen=r.fen, legal_moves=legal, considered_moves=considered_moves
                )
                rr["prompt"] = [{"role": "user", "content": prompt_text}]

                out_rows.append(rr)

        n_emit = len(out_rows) - n_emit_before
        print(f"[stage] {stage}/{num_full_stages-1} {stage_desc} sampled_source={len(sampled)} emitted_rows={n_emit}")

    out_table = pa.Table.from_pylist(out_rows)
    _atomic_write_parquet(out_table, output_path=output_path)
    print(f"[done] wrote={output_path} rows={out_table.num_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

