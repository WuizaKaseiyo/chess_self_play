#!/usr/bin/env python3
"""
Build a restricted-moves ("selection") chess dataset with FULL-LEGAL candidate sets only.

v4 spec (full-legal only):
- For every row, `reward_model.considered_moves_uci := reward_model.legal_moves_uci`.
- Prompts are rendered from `recipe/chess/prompt_templates/select_prompt.jinja`.
- No mixture variants or curriculum expansion; one row in, one row out.

This script is intended for training/eval datasets where the trainer will construct
harder subsets on-the-fly during training.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment

BUILDER_VERSION = "select-v4-fulllegal-2026-01-20"


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


def _parse_splits(spec: str) -> List[str]:
    parts = [p.strip() for p in (spec or "").split(",")]
    return [p for p in parts if p]


def _process_split(
    *,
    split: str,
    input_path: Path,
    output_path: Path,
    template,
    limit_rows: Optional[int],
    overwrite: bool,
    sample_jsonl: Optional[Path],
    sample_n: int,
    seed: int,
) -> None:
    if not input_path.exists():
        print(f"[WARN] Missing input parquet for split='{split}': {input_path}")
        return
    if output_path.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {output_path} (pass --overwrite)")

    table = pq.read_table(input_path)
    rows = table.to_pylist()
    if limit_rows is not None:
        if limit_rows < 0:
            raise SystemExit("--limit_rows must be >= 0")
        rows = rows[: int(limit_rows)]

    rng = random.Random(int(seed))
    sample_rows = set(rng.sample(range(len(rows)), k=min(int(sample_n), len(rows)))) if rows else set()
    samples: List[Dict[str, Any]] = []

    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        rr = dict(row)
        rm = rr.get("reward_model") or {}
        if not isinstance(rm, dict):
            rm = {}
        rr["reward_model"] = rm

        fen = str(rm.get("fen") or "").strip()
        legal_moves = _normalize_moves(rm.get("legal_moves_uci"))
        if not fen:
            raise ValueError(f"Empty FEN encountered in split='{split}'.")
        if not legal_moves:
            raise ValueError(f"Empty legal_moves_uci encountered in split='{split}'.")

        considered_moves = list(legal_moves)
        rm["legal_moves_uci"] = legal_moves
        rm["considered_moves_uci"] = considered_moves

        prompt_text = str(
            template.render(
                FEN=fen,
                legal_moves_uci_list=legal_moves,
                considered_moves_uci_list=considered_moves,
            )
        )
        rr["prompt"] = [{"role": "user", "content": prompt_text}]

        ei = rr.get("extra_info") or {}
        if isinstance(ei, dict):
            ei = dict(ei)
            ei["builder_version"] = BUILDER_VERSION
            ei.setdefault("derived_variant", "full_legal")
            rr["extra_info"] = ei

        out_rows.append(rr)

        if idx in sample_rows and sample_jsonl is not None:
            samples.append(
                {
                    "split": split,
                    "row_idx": idx,
                    "fen": fen,
                    "n_legal_moves": len(legal_moves),
                    "ground_truth": rm.get("ground_truth"),
                    "prompt_head": prompt_text[:200],
                    "considered_moves_head": considered_moves[:10],
                }
            )

    _atomic_write_parquet(pa.Table.from_pylist(out_rows), output_path=output_path)
    print(f"[OK] Wrote {len(out_rows)} rows -> {output_path}")

    if sample_jsonl is not None and samples:
        _write_jsonl(sample_jsonl, samples)
        print(f"[OK] Wrote sample JSONL -> {sample_jsonl}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/chess_puzzles")
    ap.add_argument("--output_dir", default="data/chess_puzzles_select_v4")
    ap.add_argument("--template_path", default="recipe/chess/prompt_templates/select_prompt.jinja")
    ap.add_argument("--splits", default="train,train_hard,test")
    ap.add_argument("--limit_rows", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sample_jsonl", default=None)
    ap.add_argument("--sample_n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    template_path = Path(args.template_path)

    if not template_path.exists():
        raise SystemExit(f"Template not found: {template_path}")

    env = Environment(autoescape=False)
    template = env.from_string(template_path.read_text(encoding="utf-8"))

    splits = _parse_splits(args.splits)
    if not splits:
        raise SystemExit("--splits must include at least one split name")

    for split in splits:
        input_path = input_dir / f"{split}.parquet"
        output_path = output_dir / f"{split}.parquet"
        sample_jsonl = Path(args.sample_jsonl) if args.sample_jsonl else None
        if sample_jsonl is not None and len(splits) > 1:
            sample_jsonl = sample_jsonl.with_name(f"{sample_jsonl.stem}_{split}{sample_jsonl.suffix}")

        _process_split(
            split=split,
            input_path=input_path,
            output_path=output_path,
            template=template,
            limit_rows=args.limit_rows,
            overwrite=args.overwrite,
            sample_jsonl=sample_jsonl,
            sample_n=args.sample_n,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
