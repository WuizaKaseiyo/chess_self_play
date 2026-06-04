#!/usr/bin/env python3
"""
Analyze alignment between per-move value maps in chess selection datasets.

Motivation
----------
In v4 (full-legal) selection training we have multiple per-move signals, e.g.:
  - move_expected_scores_json  (WDL expected score, preferred μ for selection)
  - move_values_json           (win-prob-like, often derived from centipawns)
  - move_cps_json              (centipawns)

This script quantifies:
  - how often their argmax moves agree (strict + tie-aware),
  - how tie/quantization-heavy each map is (unique value counts, best-tie counts),
  - how often dataset ground_truth matches each map's argmax.

It is designed for offline debugging of reward-shaping choices.

Example
-------
conda run -n verl python scripts/analyze_chess_reward_map_alignment.py \\
  --parquet data/chess_puzzles_select_v4/train_hard.parquet \\
  --out_dir outputs/reward_map_alignment/train_hard_v4
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _normalize_move(m: Any) -> str:
    return str(m or "").strip().lower()


def _parse_move_float_map(map_json: Any) -> dict[str, float]:
    if not map_json:
        return {}
    if isinstance(map_json, dict):
        obj = map_json
    elif isinstance(map_json, str):
        try:
            obj = json.loads(map_json)
        except Exception:
            return {}
    else:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in obj.items():
        key = _normalize_move(k)
        if not key:
            continue
        try:
            out[key] = float(v)
        except Exception:
            continue
    return out


@dataclass(frozen=True)
class BestByMap:
    best_move: str
    best_value: float
    best_tie_count: int
    unique_value_count: int
    second_best_value: float


def _best_by_map(value_map: dict[str, float], legal_moves: list[str]) -> Optional[BestByMap]:
    legal = [_normalize_move(m) for m in legal_moves or []]
    legal = [m for m in legal if m]
    if not legal or not value_map:
        return None

    # Restrict to values defined on the legal list (dataset contract usually ensures completeness).
    vals: list[tuple[str, float]] = []
    for m in legal:
        if m in value_map:
            v = value_map[m]
            if math.isfinite(v):
                vals.append((m, float(v)))
    if not vals:
        return None

    # Deterministic tie-break: higher value wins; on tie, lexicographically smaller UCI wins.
    best_move = ""
    best_val = -float("inf")
    for m, v in vals:
        if (v > best_val) or (v == best_val and (not best_move or m < best_move)):
            best_move = m
            best_val = v

    best_ties = [m for m, v in vals if v == best_val]
    uniq_vals = sorted({v for _, v in vals})
    second_best = float("nan")
    if len(uniq_vals) >= 2:
        second_best = uniq_vals[-2]

    return BestByMap(
        best_move=best_move,
        best_value=float(best_val),
        best_tie_count=int(len(best_ties)),
        unique_value_count=int(len(uniq_vals)),
        second_best_value=float(second_best),
    )


def _best_set(value_map: dict[str, float], legal_moves: list[str]) -> set[str]:
    bbm = _best_by_map(value_map, legal_moves)
    if bbm is None or not math.isfinite(bbm.best_value):
        return set()
    best_val = bbm.best_value
    out = set()
    for m in legal_moves or []:
        key = _normalize_move(m)
        if not key:
            continue
        v = value_map.get(key)
        if v is None:
            continue
        try:
            if float(v) == best_val:
                out.add(key)
        except Exception:
            continue
    return out


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Selection training parquet (e.g., v4 train_hard).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit_rows", type=int, default=None, help="Optional cap for quick iteration.")
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise SystemExit(f"Missing parquet: {parquet_path}")

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    table = pq.read_table(parquet_path, columns=["reward_model"])
    rows = table.to_pylist()
    if args.limit_rows is not None:
        rows = rows[: int(args.limit_rows)]
    print(f"[LOAD] rows={len(rows)} from {parquet_path}")

    per_row: list[dict[str, Any]] = []
    bad_rows = 0

    for i, row in enumerate(rows):
        rm = row.get("reward_model") or {}
        if not isinstance(rm, dict):
            bad_rows += 1
            continue

        fen = str(rm.get("fen") or "").strip()
        gt = _normalize_move(rm.get("ground_truth"))
        legal = rm.get("legal_moves_uci") or []
        legal_moves = [_normalize_move(m) for m in legal]
        legal_moves = [m for m in legal_moves if m]

        expected_map = _parse_move_float_map(rm.get("move_expected_scores_json"))
        winprob_map = _parse_move_float_map(rm.get("move_values_json"))
        cp_map = _parse_move_float_map(rm.get("move_cps_json"))

        bb_expected = _best_by_map(expected_map, legal_moves)
        bb_winprob = _best_by_map(winprob_map, legal_moves)
        bb_cp = _best_by_map(cp_map, legal_moves)

        if bb_expected is None:
            bad_rows += 1
            continue

        # Best sets for tie-aware comparisons.
        expected_best_set = _best_set(expected_map, legal_moves)
        winprob_best_set = _best_set(winprob_map, legal_moves) if winprob_map else set()
        cp_best_set = _best_set(cp_map, legal_moves) if cp_map else set()

        expected_best = bb_expected.best_move
        winprob_best = bb_winprob.best_move if bb_winprob is not None else ""
        cp_best = bb_cp.best_move if bb_cp is not None else ""

        per_row.append(
            {
                "row_idx": int(i),
                "fen": fen,
                "n_legal": int(len(legal_moves)),
                "ground_truth": gt,
                "expected_best": expected_best,
                "winprob_best": winprob_best,
                "cp_best": cp_best,
                "gt_eq_expected_best": bool(gt and gt == expected_best),
                "gt_eq_winprob_best": bool(gt and winprob_best and gt == winprob_best),
                "gt_eq_cp_best": bool(gt and cp_best and gt == cp_best),
                "expected_eq_winprob_best_strict": bool(expected_best and winprob_best and expected_best == winprob_best),
                "expected_eq_cp_best_strict": bool(expected_best and cp_best and expected_best == cp_best),
                "expected_in_winprob_best_set": bool(expected_best and expected_best in winprob_best_set)
                if winprob_best_set
                else None,
                "expected_in_cp_best_set": bool(expected_best and expected_best in cp_best_set) if cp_best_set else None,
                "gt_in_expected_best_set": bool(gt and gt in expected_best_set) if expected_best_set else None,
                "gt_in_winprob_best_set": bool(gt and gt in winprob_best_set) if winprob_best_set else None,
                "gt_in_cp_best_set": bool(gt and gt in cp_best_set) if cp_best_set else None,
                "expected_best_value": float(bb_expected.best_value),
                "expected_second_best_value": float(bb_expected.second_best_value),
                "expected_best_tie_count": int(bb_expected.best_tie_count),
                "expected_unique_value_count": int(bb_expected.unique_value_count),
                "winprob_best_value": float(bb_winprob.best_value) if bb_winprob is not None else float("nan"),
                "winprob_second_best_value": float(bb_winprob.second_best_value) if bb_winprob is not None else float("nan"),
                "winprob_best_tie_count": int(bb_winprob.best_tie_count) if bb_winprob is not None else 0,
                "winprob_unique_value_count": int(bb_winprob.unique_value_count) if bb_winprob is not None else 0,
                "cp_best_value": float(bb_cp.best_value) if bb_cp is not None else float("nan"),
                "cp_second_best_value": float(bb_cp.second_best_value) if bb_cp is not None else float("nan"),
                "cp_best_tie_count": int(bb_cp.best_tie_count) if bb_cp is not None else 0,
                "cp_unique_value_count": int(bb_cp.unique_value_count) if bb_cp is not None else 0,
            }
        )

    df = pd.DataFrame(per_row)
    df.to_parquet(out_dir / "per_row_alignment.parquet", index=False)
    df.to_csv(out_dir / "per_row_alignment.csv.gz", index=False, compression="gzip")

    def rate(col: str) -> float:
        if col not in df.columns:
            return float("nan")
        s = df[col].dropna()
        if s.empty:
            return float("nan")
        return float(s.mean())

    summary = {
        "parquet": str(parquet_path),
        "rows_loaded": int(len(rows)),
        "rows_analyzed": int(len(df)),
        "rows_skipped_bad_reward_model": int(bad_rows),
        "gt_eq_expected_best_rate": rate("gt_eq_expected_best"),
        "gt_eq_winprob_best_rate": rate("gt_eq_winprob_best"),
        "gt_eq_cp_best_rate": rate("gt_eq_cp_best"),
        "expected_eq_winprob_best_strict_rate": rate("expected_eq_winprob_best_strict"),
        "expected_eq_cp_best_strict_rate": rate("expected_eq_cp_best_strict"),
        "expected_in_winprob_best_set_rate": rate("expected_in_winprob_best_set"),
        "expected_in_cp_best_set_rate": rate("expected_in_cp_best_set"),
        "expected_best_tie_rate": float((df["expected_best_tie_count"] > 1).mean()) if not df.empty else float("nan"),
        "winprob_best_tie_rate": float((df["winprob_best_tie_count"] > 1).mean()) if not df.empty else float("nan"),
        "cp_best_tie_rate": float((df["cp_best_tie_count"] > 1).mean()) if not df.empty else float("nan"),
        "expected_unique_value_count_mean": float(df["expected_unique_value_count"].mean())
        if not df.empty
        else float("nan"),
        "winprob_unique_value_count_mean": float(df["winprob_unique_value_count"].mean())
        if not df.empty
        else float("nan"),
        "cp_unique_value_count_mean": float(df["cp_unique_value_count"].mean()) if not df.empty else float("nan"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[SUMMARY]")
    for k, v in summary.items():
        print(f"- {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

