#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _is_empty_prompt_text(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return True
    return value.strip() == ""


def _fmt_row_hint(row: Dict[str, Any]) -> str:
    parts = []
    for k in ("game_id", "ply", "opponent_depth", "retry_idx", "error_reason"):
        if k in row:
            parts.append(f"{k}={row.get(k)!r}")
    return " ".join(parts)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Validate full-game eval JSONL logs (e.g. moves.jsonl).")
    p.add_argument("jsonl_path", type=Path, help="Path to a JSONL file to validate.")
    p.add_argument("--field", default="prompt_text", help="Field name to check (default: prompt_text).")
    p.add_argument("--max-examples", type=int, default=10, help="Max bad-row examples to print.")
    args = p.parse_args(argv)

    total = 0
    parse_errors = 0
    missing = 0
    empty = 0
    bad_examples: list[str] = []

    with args.jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except Exception as e:
                parse_errors += 1
                if len(bad_examples) < args.max_examples:
                    bad_examples.append(f"line={line_idx} json_error={e!r}")
                continue

            if args.field not in row:
                missing += 1
                if len(bad_examples) < args.max_examples:
                    bad_examples.append(f"line={line_idx} missing_field {args.field!r} {_fmt_row_hint(row)}")
                continue

            if _is_empty_prompt_text(row.get(args.field)):
                empty += 1
                if len(bad_examples) < args.max_examples:
                    bad_examples.append(f"line={line_idx} empty_field {args.field!r} {_fmt_row_hint(row)}")

    print(f"[validate_fullgame_jsonl] path={args.jsonl_path}")
    print(f"[validate_fullgame_jsonl] total_rows={total}")
    print(f"[validate_fullgame_jsonl] parse_errors={parse_errors} missing_{args.field}={missing} empty_{args.field}={empty}")
    if bad_examples:
        print("[validate_fullgame_jsonl] examples:")
        for ex in bad_examples:
            print(f"  - {ex}")

    ok = parse_errors == 0 and missing == 0 and empty == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

