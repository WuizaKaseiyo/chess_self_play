#!/usr/bin/env python3
"""
Build a restricted-moves ("selection") chess dataset where prompts are rendered from
`recipe/chess/prompt_templates/select_prompt.jinja`.

Test-set builder:
- Read an existing VERL-format parquet (e.g., data/chess_puzzles/test.parquet).
- For each row, set `considered_moves_uci_list = legal_moves_uci_list` (full legal list).
- Optional: shuffle the full legal list per row to probe presentation-order bias
  (`--shuffle_legal_moves` + deterministic per-row seed).
- Render the selection prompt and store it in the `prompt` column as a single user message.
- Store the considered-moves list in `reward_model.considered_moves_uci` so reward code can enforce
  in-subset selection at training time.

This intentionally does *not* overwrite inputs unless `--overwrite` is provided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment


def _normalize_moves(moves: Any) -> List[str]:
    if moves is None:
        return []
    if isinstance(moves, str):
        s = moves.strip().lower()
        return [s] if s else []
    out: List[str] = []
    try:
        for m in moves:
            s = str(m).strip().lower()
            if s:
                out.append(s)
    except Exception:
        return []
    return out


def _stable_int_hash(*parts: str, mod: int) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], "big") % int(mod)


def _atomic_write_parquet(table: pa.Table, *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(output_path)


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
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
    ap.add_argument("--limit_rows", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--sample_jsonl",
        default=None,
        help="Optional path to write a small JSONL sample for human inspection.",
    )
    ap.add_argument("--sample_n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="Seed for sampling (and shuffle if --shuffle_seed unset).")
    ap.add_argument(
        "--shuffle_legal_moves",
        action="store_true",
        help="Shuffle the full legal move list per row (deterministic; uses --shuffle_seed + extra_info.index).",
    )
    ap.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="Seed for deterministic per-row shuffle. Defaults to --seed if unset.",
    )
    ap.add_argument(
        "--data_source_suffix",
        default=None,
        help="Optional suffix to append to each row's data_source (e.g., _shuffled_legal_moves).",
    )
    ap.add_argument(
        "--data_source_override",
        default=None,
        help="Optional override for data_source (wins over --data_source_suffix).",
    )
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

    env = Environment(autoescape=False)
    template = env.from_string(template_path.read_text(encoding="utf-8"))

    table = pq.read_table(input_path)
    rows = table.to_pylist()
    if args.limit_rows is not None:
        if args.limit_rows < 0:
            raise SystemExit("--limit_rows must be >= 0")
        rows = rows[: int(args.limit_rows)]

    rng = random.Random(int(args.seed))
    sample_rows = set(rng.sample(range(len(rows)), k=min(int(args.sample_n), len(rows)))) if rows else set()
    samples: List[Dict[str, Any]] = []
    shuffle_seed = int(args.shuffle_seed) if args.shuffle_seed is not None else int(args.seed)

    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        rr = dict(row)
        rm = rr.get("reward_model") or {}
        if not isinstance(rm, dict):
            rm = {}
        rr["reward_model"] = rm
        ei = rr.get("extra_info") or {}

        fen = str(rm.get("fen") or "").strip()
        legal_moves = _normalize_moves(rm.get("legal_moves_uci"))
        if not fen:
            raise ValueError("Empty FEN encountered.")
        if not legal_moves:
            raise ValueError("Empty legal_moves_uci encountered.")

        row_id = None
        if isinstance(ei, dict):
            row_id_val = ei.get("index")
            if row_id_val is not None:
                try:
                    row_id = int(row_id_val)
                except Exception:
                    row_id = None

        considered_moves = list(legal_moves)
        if args.shuffle_legal_moves:
            if row_id is None:
                raise ValueError("Missing extra_info.index required for deterministic shuffle.")
            row_seed = _stable_int_hash(
                str(shuffle_seed), str(row_id), "select-test-shuffle", mod=2**32
            )
            row_rng = random.Random(int(row_seed))
            row_rng.shuffle(considered_moves)
            if len(considered_moves) > 1 and considered_moves == legal_moves:
                swap_idx = (row_seed % (len(considered_moves) - 1)) + 1
                considered_moves[0], considered_moves[swap_idx] = considered_moves[swap_idx], considered_moves[0]
        rm["considered_moves_uci"] = considered_moves
        if args.data_source_override is not None:
            rr["data_source"] = args.data_source_override
        elif args.data_source_suffix:
            base_src = rr.get("data_source")
            base_src = "" if base_src is None else str(base_src)
            rr["data_source"] = f"{base_src}{args.data_source_suffix}" if base_src else str(args.data_source_suffix)

        prompt_text = str(
            template.render(
                FEN=fen,
                legal_moves_uci_list=legal_moves,
                considered_moves_uci_list=considered_moves,
            )
        )
        rr["prompt"] = [{"role": "user", "content": prompt_text}]
        out_rows.append(rr)

        if idx in sample_rows:
            samples.append(
                {
                    "row_id": row_id,
                    "data_source": rr.get("data_source"),
                    "fen": fen,
                    "move_best": str(rm.get("ground_truth") or "").strip().lower(),
                    "legal_moves_uci": legal_moves,
                    "considered_moves_uci": considered_moves,
                    "prompt": prompt_text,
                }
            )

    out_table = pa.Table.from_pylist(out_rows)
    _atomic_write_parquet(out_table, output_path=output_path)
    print(f"Wrote {output_path} rows={out_table.num_rows}")

    if args.sample_jsonl:
        _write_jsonl(Path(args.sample_jsonl), samples)
        print(f"Wrote sample JSONL: {args.sample_jsonl} rows={len(samples)}")


if __name__ == "__main__":
    main()
