#!/usr/bin/env python3
"""
Audit expected-score (μ) distributions by chess tactical subsets.

Motivation: In `expected_score_wdl_vs_best`, the raw reward is:
  expected_score(pred) - expected_score(best_legal)  (<= 0)

After GRPO-style group normalization, learning signal depends strongly on the
distribution of expected scores among candidate moves.

This script focuses on subsets that are relevant to "does the run learn checks?":
  - GT move gives check / does not give check
  - Candidate set has >=1 checking move
  - Candidate set has >=2 checking moves (multiple-check positions)

It uses only rule-based chess parsing (python-chess) and the dataset's stored
`move_expected_scores_json` (preferred) or `move_values_json` as fallback.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import chess
import numpy as np
import pyarrow.parquet as pq


EPS = 1e-9


def _isclose(a: float, b: float, eps: float = EPS) -> bool:
    return abs(a - b) <= eps


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _parse_move_float_map(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        out: dict[str, float] = {}
        for k, v in raw.items():
            vf = _safe_float(v)
            if vf is None:
                continue
            out[str(k).strip().lower()] = float(vf)
        return out
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
        except Exception:
            return {}
        return _parse_move_float_map(obj)
    return {}


class CheckComputer:
    def __init__(self) -> None:
        self._board_cache: dict[str, chess.Board] = {}
        self._move_cache: dict[tuple[str, str], tuple[bool, bool, bool]] = {}

    def _board_for_fen(self, fen: str) -> chess.Board:
        b = self._board_cache.get(fen)
        if b is None:
            b = chess.Board(fen)
            self._board_cache[fen] = b
        return b

    def info(self, fen: str, uci: str) -> tuple[bool, bool, bool]:
        fen_s = str(fen or "").strip()
        uci_s = str(uci or "").strip().lower()
        key = (fen_s, uci_s)
        cached = self._move_cache.get(key)
        if cached is not None:
            return cached

        try:
            move = chess.Move.from_uci(uci_s)
        except Exception:
            out = (False, False, False)
            self._move_cache[key] = out
            return out

        board = self._board_for_fen(fen_s)
        if not board.is_legal(move):
            out = (False, False, False)
            self._move_cache[key] = out
            return out

        board.push(move)
        try:
            is_check = bool(board.is_check())
            is_mate = bool(board.is_checkmate())
        finally:
            board.pop()

        out = (True, is_check, is_mate)
        self._move_cache[key] = out
        return out


def _quantiles(xs: list[float], qs: Iterable[float]) -> dict[float, float]:
    arr = np.asarray([x for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return {float(q): float("nan") for q in qs}
    out: dict[float, float] = {}
    for q in qs:
        out[float(q)] = float(np.quantile(arr, float(q)))
    return out


def _summary_line(name: str, xs: list[float]) -> str:
    arr = np.asarray([x for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return f"- {name}: n=0"
    q = _quantiles(xs, qs=[0.0, 0.1, 0.5, 0.9, 1.0])
    return (
        f"- {name}: n={arr.size} mean={float(arr.mean()):.4f} "
        f"median={q[0.5]:.4f} p10={q[0.1]:.4f} p90={q[0.9]:.4f} min={q[0.0]:.4f} max={q[1.0]:.4f}"
    )


@dataclass
class RowFeatures:
    gt_is_check: bool
    gt_is_mate: bool

    n_moves: int
    n_check_moves: int
    n_mate_moves: int

    mu_best: float
    mu_second: float
    mu_gap_best_second: float
    n_mu_best_ties: int

    frac_mu_eq_0: float
    frac_mu_eq_1: float
    frac_mu_mid: float  # 0 < mu < 1

    mu_best_check: float
    mu_second_check: float
    mu_gap_best_second_check: float
    n_mu_best_check_ties: int

    mu_best_noncheck: float
    mu_best_minus_mu_best_check: float


def _compute_row_features(
    *, fen: str, gt_uci: str, moves: list[str], mu_map: dict[str, float], checker: CheckComputer
) -> Optional[RowFeatures]:
    fen_s = str(fen or "").strip()
    gt_s = str(gt_uci or "").strip().lower()
    if not fen_s or not gt_s or not moves:
        return None

    ok_gt, gt_is_check, gt_is_mate = checker.info(fen_s, gt_s)
    if not ok_gt:
        return None

    mu_vals: list[float] = []
    check_mu: list[float] = []
    noncheck_mu: list[float] = []
    n_check_moves = 0
    n_mate_moves = 0

    for mv in moves:
        mv_s = str(mv or "").strip().lower()
        if not mv_s:
            continue
        v = mu_map.get(mv_s)
        vf = _safe_float(v)
        if vf is None:
            return None
        mu_vals.append(float(vf))

        ok, is_check, is_mate = checker.info(fen_s, mv_s)
        if ok and is_check:
            n_check_moves += 1
            check_mu.append(float(vf))
        else:
            noncheck_mu.append(float(vf))
        if ok and is_mate:
            n_mate_moves += 1

    if not mu_vals:
        return None

    mu_sorted = sorted(mu_vals, reverse=True)
    mu_best = float(mu_sorted[0])
    mu_second = float(mu_sorted[1]) if len(mu_sorted) >= 2 else float("nan")
    mu_gap = float(mu_best - mu_second) if len(mu_sorted) >= 2 else float("nan")
    n_best_ties = sum(1 for x in mu_vals if _isclose(float(x), mu_best))

    n_moves = len(mu_vals)
    frac0 = sum(1 for x in mu_vals if _isclose(float(x), 0.0)) / float(n_moves)
    frac1 = sum(1 for x in mu_vals if _isclose(float(x), 1.0)) / float(n_moves)
    fracmid = sum(1 for x in mu_vals if (float(x) > EPS and float(x) < 1.0 - EPS)) / float(n_moves)

    if check_mu:
        check_sorted = sorted(check_mu, reverse=True)
        mu_best_check = float(check_sorted[0])
        mu_second_check = float(check_sorted[1]) if len(check_sorted) >= 2 else float("nan")
        mu_gap_check = float(mu_best_check - mu_second_check) if len(check_sorted) >= 2 else float("nan")
        n_best_check_ties = sum(1 for x in check_mu if _isclose(float(x), mu_best_check))
    else:
        mu_best_check = float("nan")
        mu_second_check = float("nan")
        mu_gap_check = float("nan")
        n_best_check_ties = 0

    mu_best_noncheck = max(noncheck_mu) if noncheck_mu else float("nan")
    mu_best_minus_best_check = mu_best - mu_best_check if check_mu else float("nan")

    return RowFeatures(
        gt_is_check=bool(gt_is_check),
        gt_is_mate=bool(gt_is_mate),
        n_moves=int(n_moves),
        n_check_moves=int(n_check_moves),
        n_mate_moves=int(n_mate_moves),
        mu_best=float(mu_best),
        mu_second=float(mu_second),
        mu_gap_best_second=float(mu_gap),
        n_mu_best_ties=int(n_best_ties),
        frac_mu_eq_0=float(frac0),
        frac_mu_eq_1=float(frac1),
        frac_mu_mid=float(fracmid),
        mu_best_check=float(mu_best_check),
        mu_second_check=float(mu_second_check),
        mu_gap_best_second_check=float(mu_gap_check),
        n_mu_best_check_ties=int(n_best_check_ties),
        mu_best_noncheck=float(mu_best_noncheck) if noncheck_mu else float("nan"),
        mu_best_minus_mu_best_check=float(mu_best_minus_best_check),
    )


def _subset_name(f: RowFeatures) -> str:
    return "gt_check" if f.gt_is_check else "gt_noncheck"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", action="append", required=True, help="Parquet path to analyze (repeatable).")
    ap.add_argument("--limit_rows", type=int, default=-1)
    ap.add_argument("--out_md", type=str, default="", help="Optional markdown output path.")
    args = ap.parse_args()

    md_lines: list[str] = []
    md_lines.append("# Expected-score (μ) distribution audit\n")
    md_lines.append(
        "This report analyzes `move_expected_scores_json` (μ) distributions by chess tactical subsets "
        "using `python-chess` (no model inference).\n"
    )

    checker = CheckComputer()

    for parquet_path in args.parquet:
        path = Path(parquet_path)
        if not path.exists():
            raise SystemExit(f"Missing parquet: {path}")

        pf = pq.ParquetFile(str(path))
        n_total = int(pf.metadata.num_rows)
        n_read = n_total if args.limit_rows is None or args.limit_rows < 0 else min(n_total, int(args.limit_rows))

        # Read only the minimal columns.
        table = pq.read_table(
            str(path),
            columns=["reward_model", "extra_info"],
        )
        rows = table.to_pylist()[:n_read]

        features: list[RowFeatures] = []
        n_fail = 0
        for r in rows:
            rm = r.get("reward_model") or {}
            fen = (rm.get("fen") or "").strip()
            gt = (rm.get("ground_truth") or "").strip().lower()
            moves = rm.get("considered_moves_uci") or rm.get("legal_moves_uci") or []

            mu_map = _parse_move_float_map(rm.get("move_expected_scores_json"))
            if not mu_map:
                mu_map = _parse_move_float_map(rm.get("move_values_json"))

            feat = _compute_row_features(fen=fen, gt_uci=gt, moves=moves, mu_map=mu_map, checker=checker)
            if feat is None:
                n_fail += 1
                continue
            features.append(feat)

        md_lines.append(f"## Dataset: `{path}`\n")
        md_lines.append(f"- Rows: {n_total} (analyzed {len(features)}, failed {n_fail})\n")

        # Subset counts.
        n_gt_check = sum(1 for f in features if f.gt_is_check)
        n_gt_noncheck = sum(1 for f in features if not f.gt_is_check)
        n_check_avail = sum(1 for f in features if f.n_check_moves >= 1)
        n_multi_check = sum(1 for f in features if f.n_check_moves >= 2)
        n_gt_check_multi = sum(1 for f in features if f.gt_is_check and f.n_check_moves >= 2)
        n_gt_noncheck_check_avail = sum(1 for f in features if (not f.gt_is_check) and f.n_check_moves >= 1)

        md_lines.append("### Subset sizes\n")
        md_lines.append(f"- `gt_check`: {n_gt_check}\n")
        md_lines.append(f"- `gt_noncheck`: {n_gt_noncheck}\n")
        md_lines.append(f"- `check_available` (≥1 check in legal moves): {n_check_avail}\n")
        md_lines.append(f"- `multi_check` (≥2 checks in legal moves): {n_multi_check}\n")
        md_lines.append(f"- `gt_check & multi_check`: {n_gt_check_multi}\n")
        md_lines.append(f"- `gt_noncheck & check_available`: {n_gt_noncheck_check_avail}\n")

        def collect(filter_fn, attr: str) -> list[float]:
            out: list[float] = []
            for f in features:
                if not filter_fn(f):
                    continue
                out.append(float(getattr(f, attr)))
            return out

        md_lines.append("\n### Key μ margins (best vs second-best)\n")
        md_lines.append(_summary_line("μ_gap_best_second (gt_check)", collect(lambda f: f.gt_is_check, "mu_gap_best_second")) + "\n")
        md_lines.append(
            _summary_line("μ_gap_best_second (gt_noncheck)", collect(lambda f: not f.gt_is_check, "mu_gap_best_second")) + "\n"
        )

        md_lines.append("\n### Candidate score mass at μ=0 vs μ∈(0,1)\n")
        md_lines.append(_summary_line("frac_mu_eq_0 (gt_check)", collect(lambda f: f.gt_is_check, "frac_mu_eq_0")) + "\n")
        md_lines.append(_summary_line("frac_mu_eq_0 (gt_noncheck)", collect(lambda f: not f.gt_is_check, "frac_mu_eq_0")) + "\n")
        md_lines.append(_summary_line("frac_mu_mid (gt_check)", collect(lambda f: f.gt_is_check, "frac_mu_mid")) + "\n")
        md_lines.append(_summary_line("frac_mu_mid (gt_noncheck)", collect(lambda f: not f.gt_is_check, "frac_mu_mid")) + "\n")

        md_lines.append("\n### Multi-check positions: how good are the *other* checks?\n")
        md_lines.append(
            _summary_line(
                "μ_second_check (gt_check & multi_check)",
                collect(lambda f: f.gt_is_check and f.n_check_moves >= 2, "mu_second_check"),
            )
            + "\n"
        )
        md_lines.append(
            _summary_line(
                "μ_gap_best_second_check (gt_check & multi_check)",
                collect(lambda f: f.gt_is_check and f.n_check_moves >= 2, "mu_gap_best_second_check"),
            )
            + "\n"
        )

        # Tie counts among checks on multi-check positions.
        tie_counts: dict[int, int] = {}
        for f in features:
            if not (f.gt_is_check and f.n_check_moves >= 2):
                continue
            tie_counts[f.n_mu_best_check_ties] = tie_counts.get(f.n_mu_best_check_ties, 0) + 1
        if tie_counts:
            md_lines.append("- `n_mu_best_check_ties` distribution on `gt_check & multi_check`:\n")
            for k in sorted(tie_counts):
                md_lines.append(f"  - {k}: {tie_counts[k]}\n")

        md_lines.append("\n### GT-noncheck but checks exist: how close is the best check to the best overall move?\n")
        md_lines.append(
            _summary_line(
                "μ_best_minus_μ_best_check (gt_noncheck & check_available)",
                collect(lambda f: (not f.gt_is_check) and f.n_check_moves >= 1, "mu_best_minus_mu_best_check"),
            )
            + "\n"
        )

        md_lines.append("\n")

    if args.out_md:
        out_path = Path(args.out_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(md_lines), encoding="utf-8")
        print(f"[WRITE] {out_path}")
    else:
        print("".join(md_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
