#!/usr/bin/env python3
"""
Rewrite an existing VERL-format chess parquet from legacy `<answer>` tags to `<uci_move>`.

Why
---
This repo's current chess reward function (`recipe/chess/reward_fn.py`) enforces the strict output contract:

  <think> ... </think><uci_move> ... </uci_move>

Some older Searchless-derived parquets (e.g. `data/chess_puzzles/*.parquet`) instruct the model to output
the move in `<answer> ... </answer>` tags instead. When used with the strict `<uci_move>` reward parser,
these prompts lead to `format_error` and reward `-1` everywhere.

This script rewrites the `prompt` column (and prompt-related `extra_info` fields):
- Replaces `<answer>`/`</answer>` tags (case-insensitive, tolerant to whitespace) with
  `<uci_move>`/`</uci_move>` inside all chat messages' `content`.
- Also rewrites any `<answer>` tag mentions inside string fields of `extra_info`
  (e.g. `system_prompt`) to keep traceability consistent.
- Leaves `reward_model` untouched.

Usage
-----
  python scripts/rewrite_chess_prompts_to_uci_move.py \
    --input_parquet data/chess_puzzles/test.parquet \
    --output_parquet data/chess_puzzles_uci_move/test.parquet \
    --max_rows 1000 \
    --overwrite
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

ANSWER_OPEN_RE = re.compile(r"<\s*answer\s*>", flags=re.IGNORECASE)
ANSWER_CLOSE_RE = re.compile(r"<\s*/\s*answer\s*>", flags=re.IGNORECASE)


def _rewrite_text(text: str) -> str:
    if not text:
        return text
    # Replace closing tags first so we don't accidentally rewrite "<answer>" within a close tag.
    out = ANSWER_CLOSE_RE.sub("</uci_move>", text)
    out = ANSWER_OPEN_RE.sub("<uci_move>", out)
    return out


def _rewrite_prompt_messages(prompt: Any) -> List[Dict[str, Any]]:
    # `prompt` is typically a list[{"role": ..., "content": ...}], but can arrive as
    # a numpy array when round-tripping through pandas. Stick to duck-typing.
    if prompt is None:
        return []
    if isinstance(prompt, dict):
        prompt_iter = [prompt]
    else:
        try:
            prompt_iter = list(prompt)
        except TypeError:
            prompt_iter = []

    out: List[Dict[str, Any]] = []
    for msg in prompt_iter:
        if msg is None:
            continue
        try:
            msg_dict = dict(msg)
        except Exception:
            # Best-effort: skip malformed entries.
            continue
        msg_dict["content"] = _rewrite_text(str(msg_dict.get("content", "")))
        out.append(msg_dict)
    return out


def _rewrite_any(obj: Any) -> Any:
    """Recursively rewrite `<answer>` tags inside strings in a JSON-like structure."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return _rewrite_text(obj)
    if isinstance(obj, dict):
        return {k: _rewrite_any(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_any(v) for v in obj]
    if isinstance(obj, tuple):
        return [_rewrite_any(v) for v in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--max_rows", type=int, default=None, help="Rewrite only the first N rows (smoke/test).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input_parquet)
    output_path = Path(args.output_parquet)

    if not input_path.exists():
        raise SystemExit(f"Input parquet not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {output_path} (pass --overwrite)")

    table = pq.read_table(input_path)
    rows = table.to_pylist()
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]

    rewritten = 0
    out_rows = []
    for r in rows:
        rr = dict(r)
        before = rr.get("prompt")
        after = _rewrite_prompt_messages(before)
        rr["prompt"] = after
        if "extra_info" in rr and isinstance(rr["extra_info"], dict):
            rr["extra_info"] = _rewrite_any(rr["extra_info"])
        out_rows.append(rr)
        # Lightweight accounting: count rows that actually contained `<answer>`.
        try:
            before_str = str(before)
            if "<answer" in before_str.lower():
                rewritten += 1
        except Exception:
            pass

    out_table = pa.Table.from_pylist(out_rows)

    # If overwriting an existing parquet, preserve its exact schema to avoid downstream
    # Arrow struct/layout surprises.
    if output_path.exists():
        target_schema = pq.read_schema(output_path)
        out_table = out_table.cast(target_schema)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, output_path)

    print(f"Wrote {output_path} rows={out_table.num_rows} (rows_with_answer_tag~={rewritten})")


if __name__ == "__main__":
    main()
