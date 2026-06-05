#!/usr/bin/env python3
"""
Rewrite VERL-format chess parquets so the stored prompt text matches a Jinja template.

Why
---
Training/evaluation consumes the `prompt` column stored in parquet. When we update
the canonical prompt template (e.g., to improve clarity or change formatting rules),
we must rewrite existing parquets so the model sees the updated prompt.

This script rewrites the `prompt` column and (optionally) ensures
`reward_model.considered_moves_uci` is present:
- For each row, render the template using `reward_model.fen`, `reward_model.legal_moves_uci`,
  and (when present) the considered-moves list.
- Replace the row's prompt with a single user message: [{"role":"user","content": rendered_prompt}].
- Optionally set `reward_model.considered_moves_uci` from the legal-move list.
- Leave other columns unchanged.

Usage
-----
Single parquet:
  python scripts/rewrite_chess_prompts_from_template.py \
    --input_parquet data/chess_puzzles_chessr1_aligned_sharded/test.parquet \
    --output_parquet data/chess_puzzles_chessr1_aligned_sharded/test.parquet \
    --overwrite

Directory of parquets:
  python scripts/rewrite_chess_prompts_from_template.py \
    --input_dir data/chess_puzzles_chessr1_aligned_sharded \
    --output_dir data/chess_puzzles_chessr1_aligned_sharded_ours \
    --template_path recipe/chess/prompt_templates/select_prompt.jinja \
    --set_considered_moves_uci \
    --overwrite

Shuffle legal moves in the prompt only (reward_model unchanged):
  python scripts/rewrite_chess_prompts_from_template.py \
    --input_parquet data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet \
    --output_parquet data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet \
    --template_path recipe/chess/prompt_templates/original_chessr1_prompt.jinja \
    --data_source_override local/chess_puzzles_shuffled \
    --shuffle_legal_moves --shuffle_seed 0 --overwrite

Shuffle legal moves in reward_model AND the prompt (selection prompts):
  python scripts/rewrite_chess_prompts_from_template.py \
    --input_parquet data/chess_puzzles_chessr1_aligned_sharded_ours/test.parquet \
    --output_parquet data/chess_puzzles_chessr1_aligned_sharded_ours/test_shuffled_legal_moves.parquet \
    --template_path recipe/chess/prompt_templates/select_prompt.jinja \
    --set_considered_moves_uci \
    --data_source_override local/chess_puzzles_shuffled \
    --shuffle_reward_model_legal_moves --shuffle_seed 0 --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment, Template


def _normalize_legal_moves(legal_moves_uci: Any) -> List[str]:
    if legal_moves_uci is None:
        return []
    if isinstance(legal_moves_uci, str):
        s = legal_moves_uci.strip()
        return [s] if s else []
    try:
        return [str(m).strip() for m in legal_moves_uci if str(m).strip()]
    except Exception:
        return []


def _render_prompt(
    template: Template,
    *,
    fen: str,
    legal_moves_uci_list: List[str],
    considered_moves_uci_list: List[str] | None = None,
) -> str:
    return str(
        template.render(
            FEN=fen,
            legal_moves_uci_list=legal_moves_uci_list,
            considered_moves_uci_list=considered_moves_uci_list or legal_moves_uci_list,
        )
    )


def _atomic_write_parquet(table: pa.Table, *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(output_path)


def _insert_considered_moves(
    reward_model: Dict[str, Any], considered_moves: List[str]
) -> Dict[str, Any]:
    """Insert/overwrite considered_moves_uci after legal_moves_uci when possible (order-preserving)."""
    if not isinstance(reward_model, dict):
        return {"considered_moves_uci": considered_moves}
    out: Dict[str, Any] = {}
    inserted = False
    for k, v in reward_model.items():
        if k == "considered_moves_uci":
            continue
        out[k] = v
        if k == "legal_moves_uci":
            out["considered_moves_uci"] = considered_moves
            inserted = True
    if not inserted:
        out["considered_moves_uci"] = considered_moves
    return out


def _build_output_schema(input_schema: pa.Schema, *, add_considered_moves: bool) -> pa.Schema:
    if not add_considered_moves:
        return input_schema
    try:
        rm_field = input_schema.field("reward_model")
    except KeyError:
        return input_schema
    if not isinstance(rm_field.type, pa.StructType):
        return input_schema

    if any(f.name == "considered_moves_uci" for f in rm_field.type):
        return input_schema

    rm_fields = list(rm_field.type)
    new_rm_fields: list[pa.Field] = []
    inserted = False
    for f in rm_fields:
        new_rm_fields.append(f)
        if f.name == "legal_moves_uci":
            new_rm_fields.append(pa.field("considered_moves_uci", pa.list_(pa.string())))
            inserted = True
    if not inserted:
        new_rm_fields.append(pa.field("considered_moves_uci", pa.list_(pa.string())))
    new_rm_type = pa.struct(new_rm_fields)

    new_fields = []
    for f in input_schema:
        if f.name == "reward_model":
            new_fields.append(pa.field("reward_model", new_rm_type))
        else:
            new_fields.append(f)
    return pa.schema(new_fields)


def _row_seed(base_seed: int, row_key: Any, row_idx: int) -> int:
    if row_key is None:
        return int(base_seed) + int(row_idx)
    payload = f"{base_seed}|{row_key}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _shuffle_moves_for_prompt(
    moves: List[str],
    *,
    base_seed: int,
    row_key: Any,
    row_idx: int,
) -> List[str]:
    if not moves:
        return []
    shuffled = list(moves)
    seed = _row_seed(base_seed, row_key, row_idx)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _apply_data_source_suffix(data_source: Any, suffix: str) -> str | Any:
    suffix = str(suffix or "")
    if not suffix:
        return data_source
    if data_source is None:
        return suffix
    if not isinstance(data_source, str):
        data_source = str(data_source)
    if data_source.endswith(suffix):
        return data_source
    return f"{data_source}{suffix}"


def _rewrite_table(
    table: pa.Table,
    *,
    template: Template,
    max_rows: int | None,
    set_considered_moves_uci: bool,
    shuffle_legal_moves: bool,
    shuffle_reward_model_legal_moves: bool,
    shuffle_seed: int,
    data_source_override: str,
    data_source_suffix: str,
    output_schema: pa.Schema | None,
) -> Tuple[pa.Table, int]:
    rows = table.to_pylist()
    if max_rows is not None:
        rows = rows[: int(max_rows)]

    rewritten = 0
    out_rows: List[Dict[str, Any]] = []
    for row_idx, r in enumerate(rows):
        rr = dict(r)
        rm = rr.get("reward_model") or {}
        if not isinstance(rm, dict):
            rm = {}

        fen = str(rm.get("fen") or "").strip()
        legal_moves = _normalize_legal_moves(rm.get("legal_moves_uci"))
        has_considered = "considered_moves_uci" in rm
        considered_moves = _normalize_legal_moves(rm.get("considered_moves_uci")) if has_considered else list(legal_moves)
        extra = rr.get("extra_info") or {}
        row_key = None
        if isinstance(extra, dict):
            if "index" in extra:
                row_key = extra.get("index")
            elif "chessr1_id" in extra:
                row_key = extra.get("chessr1_id")

        if shuffle_reward_model_legal_moves:
            legal_moves = _shuffle_moves_for_prompt(
                legal_moves,
                base_seed=shuffle_seed,
                row_key=row_key,
                row_idx=row_idx,
            )

            # Update legal_moves_uci in reward_model.
            rm = dict(rm)
            rm["legal_moves_uci"] = legal_moves

            # Reorder considered_moves to remain an order-preserving subsequence of the new legal order.
            if has_considered and considered_moves:
                considered_set = set(considered_moves)
                considered_moves = [m for m in legal_moves if m in considered_set]
                rm = _insert_considered_moves(rm, considered_moves)

        if set_considered_moves_uci:
            considered_moves = list(legal_moves)
            rm = _insert_considered_moves(rm, considered_moves)

        rr["reward_model"] = rm

        if data_source_override:
            rr["data_source"] = str(data_source_override)
        if data_source_suffix:
            rr["data_source"] = _apply_data_source_suffix(rr.get("data_source"), data_source_suffix)

        prompt_legal_moves = legal_moves
        if shuffle_legal_moves:
            prompt_legal_moves = _shuffle_moves_for_prompt(
                legal_moves,
                base_seed=shuffle_seed,
                row_key=row_key,
                row_idx=row_idx,
            )

        prompt_text = _render_prompt(
            template,
            fen=fen,
            legal_moves_uci_list=prompt_legal_moves,
            considered_moves_uci_list=considered_moves,
        )
        rr["prompt"] = [{"role": "user", "content": prompt_text}]
        out_rows.append(rr)
        rewritten += 1

    if output_schema is not None:
        out_table = pa.Table.from_pylist(out_rows, schema=output_schema)
    else:
        out_table = pa.Table.from_pylist(out_rows)
    return out_table, rewritten


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", default=None)
    ap.add_argument("--output_parquet", default=None)
    ap.add_argument("--input_dir", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument(
        "--template_path",
        default="recipe/chess/prompt_templates/chess_rl_chessr1_prompt.jinja",
        help="Jinja template path (default: canonical chess RL prompt template).",
    )
    ap.add_argument(
        "--set_considered_moves_uci",
        action="store_true",
        help="Set reward_model.considered_moves_uci to the legal-move list (order-preserving).",
    )
    ap.add_argument(
        "--shuffle_legal_moves",
        action="store_true",
        help="Shuffle legal moves in the rendered prompt only (reward_model remains unchanged).",
    )
    ap.add_argument(
        "--shuffle_reward_model_legal_moves",
        action="store_true",
        help=(
            "Shuffle reward_model.legal_moves_uci (and reorder considered_moves_uci to remain an "
            "order-preserving subsequence), and render the prompt from the shuffled lists."
        ),
    )
    ap.add_argument(
        "--shuffle_seed",
        type=int,
        default=0,
        help="Base seed for per-row legal-move shuffling (deterministic by row id).",
    )
    ap.add_argument(
        "--data_source_override",
        default="",
        help="Optional override for the top-level `data_source` field (wins over --data_source_suffix).",
    )
    ap.add_argument(
        "--data_source_suffix",
        default="",
        help="Optional suffix appended to the top-level `data_source` field (e.g., '_shuffled_legal_moves').",
    )
    ap.add_argument("--max_rows", type=int, default=None, help="Rewrite only the first N rows (smoke/test).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    input_parquet = Path(args.input_parquet) if args.input_parquet else None
    output_parquet = Path(args.output_parquet) if args.output_parquet else None
    input_dir = Path(args.input_dir) if args.input_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    template_path = Path(args.template_path)

    if input_parquet and input_dir:
        raise SystemExit("Pass either --input_parquet or --input_dir, not both.")
    if output_parquet and output_dir:
        raise SystemExit("Pass either --output_parquet or --output_dir, not both.")
    if not input_parquet and not input_dir:
        raise SystemExit("Missing input: pass --input_parquet or --input_dir.")
    if input_parquet and not output_parquet:
        raise SystemExit("Missing output: pass --output_parquet for --input_parquet mode.")
    if input_dir and not output_dir:
        raise SystemExit("Missing output: pass --output_dir for --input_dir mode.")

    if input_parquet and not input_parquet.exists():
        raise SystemExit(f"Input parquet not found: {input_parquet}")
    if input_dir and not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")
    if not template_path.exists():
        raise SystemExit(f"Template not found: {template_path}")

    template_text = template_path.read_text()
    env = Environment(autoescape=False)
    template = env.from_string(template_text)

    if args.data_source_override and args.data_source_suffix:
        raise SystemExit("Pass at most one of --data_source_override and --data_source_suffix.")
    if args.shuffle_legal_moves and args.shuffle_reward_model_legal_moves:
        raise SystemExit("Pass at most one of --shuffle_legal_moves and --shuffle_reward_model_legal_moves.")

    def _rewrite_single(input_path: Path, output_path: Path) -> None:
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing output: {output_path} (pass --overwrite)")
        input_schema = pq.read_schema(input_path)
        output_schema = _build_output_schema(input_schema, add_considered_moves=args.set_considered_moves_uci)
        table = pq.read_table(input_path)
        out_table, rewritten = _rewrite_table(
            table,
            template=template,
            max_rows=args.max_rows,
            set_considered_moves_uci=args.set_considered_moves_uci,
            shuffle_legal_moves=args.shuffle_legal_moves,
            shuffle_reward_model_legal_moves=args.shuffle_reward_model_legal_moves,
            shuffle_seed=args.shuffle_seed,
            data_source_override=args.data_source_override,
            data_source_suffix=args.data_source_suffix,
            output_schema=output_schema,
        )
        _atomic_write_parquet(out_table, output_path=output_path)
        print(f"Wrote {output_path} rows={out_table.num_rows} rows_rewritten={rewritten}")

    if input_parquet:
        _rewrite_single(input_parquet, output_parquet)
        return

    assert input_dir is not None and output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_paths = sorted(input_dir.glob("*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No parquet files found under {input_dir}")
    for path in parquet_paths:
        out_path = output_dir / path.name
        _rewrite_single(path, out_path)


if __name__ == "__main__":
    main()
