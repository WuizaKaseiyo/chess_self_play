#!/usr/bin/env python3
"""
Local validation for GT-gated chess reward modes.

This script is intentionally lightweight and runs on a couple of real rows from the v4 selection dataset:
  - `CHESS_REWARD_FN=gt_gated`
  - `CHESS_REWARD_FN=gt_expected_threshold` (+ `gt_expected_score_diff_threshold`)

It asserts the key contracts:
  - format/out-of-subset penalties remain `score=-1.0`
  - gt_gated gives positive reward only on GT
  - gt_expected_threshold gives reward only within the expected-score band
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd


def _load_reward_module():
    repo_root = Path(__file__).resolve().parents[1]
    reward_path = repo_root / "recipe" / "chess" / "reward_fn.py"
    spec = importlib.util.spec_from_file_location("chess_reward_fn", reward_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load reward module from: {reward_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mk_solution(move: str) -> str:
    return f"<think>validate</think><uci_move>{move}</uci_move>"


def _pick_out_of_subset_move(considered_set: set[str]) -> str:
    # Prefer common valid UCI moves that are almost never in a random legal list.
    candidates = ["a1a2", "h1h2", "a8a7", "h8h7", "a2a3", "h2h3", "a7a8q", "b2b4"]
    for mv in candidates:
        if mv not in considered_set:
            return mv
    # Last resort: generate something that is syntactically valid and very unlikely to appear.
    for src_file in "abcdefgh":
        for src_rank in "12345678":
            for dst_file in "abcdefgh":
                for dst_rank in "12345678":
                    mv = f"{src_file}{src_rank}{dst_file}{dst_rank}"
                    if mv not in considered_set:
                        return mv
    raise RuntimeError("Failed to find an out-of-subset move (unexpected).")


def _as_set(moves: Any) -> set[str]:
    if moves is None:
        return set()
    if isinstance(moves, str):
        return {moves.strip().lower()} if moves.strip() else set()
    out: set[str] = set()
    try:
        for m in moves:
            s = str(m).strip().lower()
            if s:
                out.add(s)
    except Exception:
        return set()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default="data/chess_puzzles_select_v4/train_hard.parquet",
        help="Path to a v4 selection parquet (default: train_hard).",
    )
    parser.add_argument("--limit_rows", type=int, default=5, help="How many rows to scan for a suitable example.")
    args = parser.parse_args()

    reward_mod = _load_reward_module()
    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise SystemExit(f"Missing parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    if len(df) == 0:
        raise SystemExit(f"Empty parquet: {parquet_path}")

    scanned = 0
    validated = 0

    for _, row in df.head(max(1, int(args.limit_rows))).iterrows():
        scanned += 1
        rm = row.get("reward_model")
        if not isinstance(rm, dict):
            continue

        gt = str(rm.get("ground_truth") or "").strip().lower()

        considered_raw = rm.get("considered_moves_uci", None)
        considered_list = []
        if considered_raw is not None:
            try:
                considered_list = [str(m).strip().lower() for m in considered_raw if str(m).strip()]
            except Exception:
                considered_list = []

        if not considered_list:
            legal_raw = rm.get("legal_moves_uci", None)
            if legal_raw is not None:
                try:
                    considered_list = [str(m).strip().lower() for m in legal_raw if str(m).strip()]
                except Exception:
                    considered_list = []
        considered_set = set(considered_list)
        if not gt or gt not in considered_set:
            continue

        wrong_move = next((m for m in considered_list if m != gt), None)
        if not wrong_move:
            continue

        out_of_subset = _pick_out_of_subset_move(considered_set)

        # --- Penalty checks (must remain strict) ---
        res_pen = reward_mod.compute_score(
            data_source=rm, solution_str="no uci tag", ground_truth=gt, chess_reward_fn="gt_gated"
        )
        assert res_pen["penalty_applied"] is True
        assert res_pen["score"] == -1.0
        assert res_pen["penalty_reason"] == "format_error"

        res_oos = reward_mod.compute_score(
            data_source=rm, solution_str=_mk_solution(out_of_subset), ground_truth=gt, chess_reward_fn="gt_gated"
        )
        assert res_oos["penalty_applied"] is True
        assert res_oos["score"] == -1.0
        assert res_oos["penalty_reason"] == "out_of_subset"

        # --- Strict GT gating ---
        res_hit = reward_mod.compute_score(
            data_source=rm, solution_str=_mk_solution(gt), ground_truth=gt, chess_reward_fn="gt_gated"
        )
        assert res_hit["penalty_applied"] is False
        assert res_hit["score"] == 1.0
        assert res_hit["reward_reason"] == "gt_gated:hit"

        res_miss = reward_mod.compute_score(
            data_source=rm, solution_str=_mk_solution(wrong_move), ground_truth=gt, chess_reward_fn="gt_gated"
        )
        assert res_miss["penalty_applied"] is False
        assert res_miss["score"] == 0.0
        assert res_miss["reward_reason"] == "gt_gated:miss"

        # --- Relaxed expected-score threshold gating ---
        expected_map = reward_mod._parse_move_float_map(rm.get("move_expected_scores_json"))
        if not expected_map or gt not in expected_map:
            # v4 should usually have this, but keep the script robust.
            continue

        expected_gt = float(expected_map[gt])
        diffs = []
        for mv in considered_list:
            if mv == gt:
                continue
            if mv not in expected_map:
                continue
            diffs.append((abs(float(expected_map[mv]) - expected_gt), mv, float(expected_map[mv])))
        if not diffs:
            continue

        diffs.sort(key=lambda t: t[0])
        near_diff, near_move, near_expected = diffs[0]
        threshold = float(near_diff) + 1e-6

        res_near = reward_mod.compute_score(
            data_source=rm,
            solution_str=_mk_solution(near_move),
            ground_truth=gt,
            chess_reward_fn="gt_expected_threshold",
            gt_expected_score_diff_threshold=threshold,
        )
        assert res_near["penalty_applied"] is False
        assert res_near["score"] == near_expected
        assert res_near["reward_reason"].startswith("gt_expected_threshold:hit(")

        # Pick a move that should miss the band (if possible).
        far_candidate = None
        for far_diff, far_move, _far_expected in reversed(diffs):
            if far_diff > threshold:
                far_candidate = far_move
                break
        if far_candidate is not None:
            res_far = reward_mod.compute_score(
                data_source=rm,
                solution_str=_mk_solution(far_candidate),
                ground_truth=gt,
                chess_reward_fn="gt_expected_threshold",
                gt_expected_score_diff_threshold=threshold,
            )
            assert res_far["penalty_applied"] is False
            assert res_far["score"] == 0.0
            assert res_far["reward_reason"].startswith("gt_expected_threshold:miss(")

        validated += 1
        # Validate at most 2 rows; we only need a couple of real examples.
        if validated >= 2:
            break

    if validated == 0:
        raise SystemExit(
            f"Did not find a suitable row to validate in the first {scanned} rows of {parquet_path}"
        )

    print(
        f"[ok] validated {validated} row(s) from {parquet_path} (scanned {scanned}); "
        "GT-gated reward modes behave as expected."
    )


if __name__ == "__main__":
    main()
