#!/usr/bin/env python3
"""
Rescore the DeepMind/Searchless-Chess 10k Lichess puzzles with Stockfish CP.

Why this exists
---------------
Our current `data/puzzles.parquet` uses:
  - FENs from https://storage.googleapis.com/searchless_chess/data/puzzles.csv
  - but *non-official* `best_move_uci` + WDL-based `move_expectations_json`

This script regenerates a puzzles parquet where:
  - The ground-truth move comes from the official Lichess puzzle solution.
    In `puzzles.csv`, the `Moves` column is: <blunder> <solution> <...>.
    The puzzle position is AFTER applying the first move (the blunder), and the
    ground-truth is the second move (the first move of the solution).
  - All legal moves from that puzzle position are rescored with Stockfish 16's
    centipawn evaluation, converted to win-probability with:
      win% = 100 / (1 + exp(-0.00368208 * centipawns))
    We store win-probabilities in [0, 1].

Outputs a parquet compatible with `examples/data_preprocess/chess_puzzles.py`:
  system_prompt (str)
  user_prompt (str)
  best_move_uci (str)
  move_expectations_json (str)  # JSON {uci_move: float in [0,1]}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import chess
import chess.engine
import pandas as pd
from tqdm import tqdm

from recipe.chess.stockfish_scoring import analyse_all_legal_moves_multipv


K_CP_TO_WINPROB = 0.00368208


def _sigmoid_stable(z: float) -> float:
    # Numerically stable sigmoid.
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def centipawn_to_win_prob(cp: int) -> float:
    # win% = 100/(1 + exp(-k*cp)) -> win_prob = sigmoid(k*cp)
    return _sigmoid_stable(K_CP_TO_WINPROB * float(cp))


@dataclass(frozen=True)
class PuzzlePosition:
    puzzle_id: str
    rating: int
    fen_pre_blunder: str
    blunder_uci: str
    fen: str
    ground_truth_uci: str


def load_system_prompt(path: str) -> str:
    if os.path.exists(path):
        df = pd.read_parquet(path, columns=["system_prompt"])
        if "system_prompt" in df.columns and len(df) > 0:
            # It should be constant, but take the first.
            return str(df.iloc[0]["system_prompt"])

    # Repo hygiene: we often keep only `data/chess_puzzles/{train,test}.parquet`.
    # When the raw parquet is absent, fall back to the system prompt stored in
    # the VERL parquet extra_info.
    for fallback in ("data/chess_puzzles/train.parquet", "data/chess_puzzles/test.parquet"):
        if not os.path.exists(fallback):
            continue
        df = pd.read_parquet(fallback, columns=["extra_info"])
        if "extra_info" not in df.columns or len(df) == 0:
            continue
        extra = df.iloc[0]["extra_info"]
        if isinstance(extra, dict) and extra.get("system_prompt"):
            return str(extra["system_prompt"])

    raise ValueError(f"Could not load system_prompt from {path} or VERL fallbacks.")


def iter_puzzle_positions(
    puzzles_csv_path: str, limit: Optional[int] = None, offset: int = 0
) -> Iterable[PuzzlePosition]:
    df = pd.read_csv(puzzles_csv_path)
    if offset < 0:
        raise ValueError("--offset must be >= 0")

    end = None if limit is None else offset + limit
    df = df.iloc[offset:end]

    for _, row in df.iterrows():
        puzzle_id = str(row["PuzzleId"])
        rating = int(row["Rating"])
        fen_pre = str(row["FEN"]).strip()
        moves = str(row["Moves"]).strip().split()
        if len(moves) < 2:
            continue
        blunder_uci = moves[0].strip().lower()
        gt_uci = moves[1].strip().lower()

        board = chess.Board(fen_pre)
        try:
            board.push_uci(blunder_uci)
        except Exception:
            continue

        if chess.Move.from_uci(gt_uci) not in board.legal_moves:
            # If this ever triggers, the input csv is inconsistent.
            continue

        yield PuzzlePosition(
            puzzle_id=puzzle_id,
            rating=rating,
            fen_pre_blunder=fen_pre,
            blunder_uci=blunder_uci,
            fen=board.fen(),
            ground_truth_uci=gt_uci,
        )


def build_user_prompt(board: chess.Board) -> str:
    legal_san = [board.san(m) for m in board.legal_moves]
    return (
        "Current FEN string: "
        + board.fen()
        + "\n"
        + "Legal moves: "
        + ", ".join(legal_san)
        + "\n\n"
        + "Let's think step by step."
    )


@dataclass
class ScoreStats:
    rows: int = 0
    moves_total: int = 0
    moves_zero: int = 0
    rows_all_zero: int = 0
    rows_gt_missing: int = 0
    rows_gt_tied_best: int = 0
    rows_gt_strict_best: int = 0
    rows_best_tie_count_sum: int = 0

    @property
    def mismatch_rate_tie_ok(self) -> float:
        if self.rows == 0:
            return float("nan")
        return 1.0 - (self.rows_gt_tied_best / self.rows)

    @property
    def mismatch_rate_strict(self) -> float:
        if self.rows == 0:
            return float("nan")
        return 1.0 - (self.rows_gt_strict_best / self.rows)

    @property
    def move_zero_rate(self) -> float:
        if self.moves_total == 0:
            return float("nan")
        return self.moves_zero / self.moves_total

    @property
    def all_zero_rate(self) -> float:
        if self.rows == 0:
            return float("nan")
        return self.rows_all_zero / self.rows

    @property
    def best_tie_count_mean(self) -> float:
        if self.rows == 0:
            return float("nan")
        return self.rows_best_tie_count_sum / self.rows


def score_dataset(
    puzzles_csv_path: str,
    engine_path: str,
    depth: int,
    threads: int,
    hash_mb: int,
    limit: Optional[int],
    offset: int,
    verbose: bool,
) -> Tuple[List[dict], ScoreStats]:
    system_prompt = load_system_prompt("data/puzzles.parquet")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    stats = ScoreStats()
    out_rows: List[dict] = []

    try:
        engine.configure({"Threads": threads, "Hash": hash_mb, "UCI_ShowWDL": True})

        puzzles_iter = list(iter_puzzle_positions(puzzles_csv_path, limit=limit, offset=offset))
        for i, puzzle in enumerate(tqdm(puzzles_iter, desc="Rescoring raw puzzles", unit="pos"), start=1):
            board = chess.Board(puzzle.fen)
            move_to_cp, _move_to_expected = analyse_all_legal_moves_multipv(engine, board, depth=depth)
            if not move_to_cp:
                continue

            move_to_prob = {m: centipawn_to_win_prob(cp) for m, cp in move_to_cp.items()}
            # Canonicalize + sort keys for determinism.
            move_values_json = json.dumps(move_to_prob, sort_keys=True, separators=(",", ":"))

            gt = puzzle.ground_truth_uci
            gt_prob = move_to_prob.get(gt)

            stats.rows += 1
            stats.moves_total += len(move_to_prob)
            stats.moves_zero += sum(1 for v in move_to_prob.values() if v == 0.0)
            if all(v == 0.0 for v in move_to_prob.values()):
                stats.rows_all_zero += 1

            if gt_prob is None:
                stats.rows_gt_missing += 1
            else:
                best_prob = max(move_to_prob.values())
                # Tie-friendly: gt is \"best\" if it matches the best value.
                if gt_prob == best_prob:
                    stats.rows_gt_tied_best += 1

                # Strict: gt is \"best\" only if it's the unique argmax.
                best_moves = [m for m, v in move_to_prob.items() if v == best_prob]
                stats.rows_best_tie_count_sum += len(best_moves)
                if len(best_moves) == 1 and best_moves[0] == gt:
                    stats.rows_gt_strict_best += 1

            user_prompt = build_user_prompt(board)
            out_rows.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "best_move_uci": gt,
                    "move_expectations_json": move_values_json,
                }
            )

            if verbose and (i <= 5 or i % 200 == 0):
                best_move = max(move_to_prob.items(), key=lambda kv: kv[1])[0]
                best_prob = move_to_prob[best_move]
                print(
                    f"[{i:04d}] id={puzzle.puzzle_id} rating={puzzle.rating} "
                    f"gt={gt} best={best_move} best_prob={best_prob:.4f} "
                    f"gt_prob={move_to_prob.get(gt, float('nan')):.4f} "
                    f"n_moves={len(move_to_prob)}"
                )

    finally:
        engine.quit()

    return out_rows, stats


def rescore_rl_parquet(
    *,
    input_parquet_path: str,
    output_parquet_path: str,
    engine_path: str,
    depth: int,
    threads: int,
    hash_mb: int,
    desc: str,
) -> ScoreStats:
    df = pd.read_parquet(input_parquet_path)
    if "reward_model" not in df.columns:
        raise ValueError(f"{input_parquet_path} is missing `reward_model` column")

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    stats = ScoreStats()
    updated_reward_models: List[dict] = []
    try:
        engine.configure({"Threads": threads, "Hash": hash_mb, "UCI_ShowWDL": True})

        for rm in tqdm(df["reward_model"], total=len(df), desc=desc, unit="row"):
            fen = (rm.get("fen") or "").strip()
            if not fen:
                raise ValueError("Missing reward_model.fen; cannot rescore.")
            board = chess.Board(fen)

            move_to_cp, move_to_expected = analyse_all_legal_moves_multipv(engine, board, depth=depth)
            if not move_to_cp:
                raise ValueError(f"No legal moves returned for FEN:\n{fen}")
            move_to_prob = {m: centipawn_to_win_prob(cp) for m, cp in move_to_cp.items()}

            gt = str(rm.get("ground_truth") or "").strip().lower()

            stats.rows += 1
            stats.moves_total += len(move_to_prob)
            stats.moves_zero += sum(1 for v in move_to_prob.values() if v == 0.0)
            if all(v == 0.0 for v in move_to_prob.values()):
                stats.rows_all_zero += 1

            gt_prob = move_to_prob.get(gt)
            if gt_prob is None:
                stats.rows_gt_missing += 1
            else:
                best_prob = max(move_to_prob.values())
                if gt_prob == best_prob:
                    stats.rows_gt_tied_best += 1
                best_moves = [m for m, v in move_to_prob.items() if v == best_prob]
                stats.rows_best_tie_count_sum += len(best_moves)
                if len(best_moves) == 1 and best_moves[0] == gt:
                    stats.rows_gt_strict_best += 1

            updated = dict(rm)
            updated["move_values_json"] = json.dumps(move_to_prob, sort_keys=True, separators=(",", ":"))
            updated["move_cps_json"] = json.dumps(move_to_cp, sort_keys=True, separators=(",", ":"))
            if move_to_expected:
                updated["move_expected_scores_json"] = json.dumps(
                    move_to_expected, sort_keys=True, separators=(",", ":")
                )

            # Baselines for delta-style rewards.
            best_cp = max(move_to_cp.values()) if move_to_cp else None
            if best_cp is not None:
                updated["position_cp"] = int(best_cp)
                updated["position_win_prob"] = float(centipawn_to_win_prob(int(best_cp)))
                if move_to_expected:
                    updated["position_expected_score"] = float(max(move_to_expected.values()))

                best_moves = sorted([m for m, cp in move_to_cp.items() if cp == best_cp])
                if best_moves:
                    updated["best_move_uci"] = best_moves[0]
            updated_reward_models.append(updated)
    finally:
        engine.quit()

    df = df.copy()
    df["reward_model"] = updated_reward_models
    df.to_parquet(output_parquet_path, index=False)
    return stats


def run_converter(
    raw_parquet_path: str,
    output_dir: str,
    test_ratio: float,
    seed: int,
) -> None:
    cmd = [
        sys.executable,
        "examples/data_preprocess/chess_puzzles.py",
        "--local_dataset_path",
        raw_parquet_path,
        "--local_save_dir",
        output_dir,
        "--test_ratio",
        str(test_ratio),
        "--seed",
        str(seed),
    ]
    print("Running converter:", " ".join(cmd))
    subprocess.check_call(cmd)


def print_stats(depth: int, stats: ScoreStats, elapsed_s: float) -> None:
    rows = stats.rows
    print(f"\nDepth={depth} finished in {elapsed_s:.1f}s")
    print(f"  rows: {rows}")
    print(f"  mismatch_rate (tie-ok): {stats.mismatch_rate_tie_ok:.3%}")
    print(f"  mismatch_rate (strict): {stats.mismatch_rate_strict:.3%}")
    print(f"  best_tie_count_mean: {stats.best_tie_count_mean:.3f}")
    print(f"  all_zero_rate (row): {stats.all_zero_rate:.3%}")
    print(f"  zero_rate (move): {stats.move_zero_rate:.3%}")
    if stats.rows_gt_missing > 0:
        print(f"  gt_missing_rows: {stats.rows_gt_missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sweep = sub.add_parser("sweep", help="Depth sweep on a subset; prints mismatch/zero stats.")
    sweep.add_argument("--puzzles_csv", default=".third_party_cache/searchless_puzzles.csv")
    sweep.add_argument("--engine_path", default=".third_party_cache/stockfish/src/stockfish")
    sweep.add_argument("--depths", default="8,10,12,14,16")
    sweep.add_argument("--threads", type=int, default=4)
    sweep.add_argument("--hash_mb", type=int, default=256)
    sweep.add_argument("--limit", type=int, default=100)
    sweep.add_argument("--offset", type=int, default=0)

    gen = sub.add_parser("generate", help="Generate a parquet (and optionally regenerate VERL train/test).")
    gen.add_argument("--puzzles_csv", default=".third_party_cache/searchless_puzzles.csv")
    gen.add_argument("--engine_path", default=".third_party_cache/stockfish/src/stockfish")
    gen.add_argument("--depth", type=int, default=10)
    gen.add_argument("--threads", type=int, default=4)
    gen.add_argument("--hash_mb", type=int, default=256)
    gen.add_argument("--limit", type=int, default=None)
    gen.add_argument("--offset", type=int, default=0)
    gen.add_argument("--output_raw_parquet", default="data/puzzles.parquet")
    gen.add_argument("--overwrite", action="store_true")
    gen.add_argument("--run_converter", action="store_true")
    gen.add_argument("--converter_output_dir", default="data/chess_puzzles")
    gen.add_argument("--converter_test_ratio", type=float, default=0.1)
    gen.add_argument("--converter_seed", type=int, default=42)
    gen.add_argument("--verbose", action="store_true")

    rl = sub.add_parser(
        "rescore_rl",
        help=(
            "Rescore the existing VERL RL parquets in-place (train/test) "
            "using Stockfish CP->winprob, without regenerating from DeepMind."
        ),
    )
    rl.add_argument("--train_parquet", default="data/chess_puzzles/train.parquet")
    rl.add_argument("--test_parquet", default="data/chess_puzzles/test.parquet")
    rl.add_argument("--engine_path", default=".third_party_cache/stockfish/src/stockfish")
    rl.add_argument("--depth", type=int, default=10)
    rl.add_argument("--threads", type=int, default=4)
    rl.add_argument("--hash_mb", type=int, default=256)
    rl.add_argument("--overwrite", action="store_true")
    rl.add_argument("--output_train_parquet", default=None)
    rl.add_argument("--output_test_parquet", default=None)

    args = parser.parse_args()

    if args.cmd == "sweep":
        depths = [int(x.strip()) for x in str(args.depths).split(",") if x.strip()]
        for depth in depths:
            t0 = time.time()
            _, stats = score_dataset(
                puzzles_csv_path=args.puzzles_csv,
                engine_path=args.engine_path,
                depth=depth,
                threads=args.threads,
                hash_mb=args.hash_mb,
                limit=args.limit,
                offset=args.offset,
                verbose=False,
            )
            print_stats(depth, stats, elapsed_s=time.time() - t0)
        return

    if args.cmd == "generate":
        if os.path.exists(args.output_raw_parquet) and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite {args.output_raw_parquet}; pass --overwrite.")

        t0 = time.time()
        out_rows, stats = score_dataset(
            puzzles_csv_path=args.puzzles_csv,
            engine_path=args.engine_path,
            depth=args.depth,
            threads=args.threads,
            hash_mb=args.hash_mb,
            limit=args.limit,
            offset=args.offset,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0
        print_stats(args.depth, stats, elapsed_s=elapsed)

        os.makedirs(os.path.dirname(args.output_raw_parquet) or ".", exist_ok=True)
        df = pd.DataFrame(out_rows)
        df.to_parquet(args.output_raw_parquet, index=False)
        print(f"Wrote raw parquet: {args.output_raw_parquet} (rows={len(df)})")

        if args.run_converter:
            run_converter(
                raw_parquet_path=args.output_raw_parquet,
                output_dir=args.converter_output_dir,
                test_ratio=args.converter_test_ratio,
                seed=args.converter_seed,
            )
        return

    if args.cmd == "rescore_rl":
        out_train = args.output_train_parquet
        out_test = args.output_test_parquet
        if args.overwrite:
            out_train = args.train_parquet
            out_test = args.test_parquet
        if not out_train or not out_test:
            raise SystemExit(
                "Must pass --overwrite or provide both --output_train_parquet and --output_test_parquet."
            )

        t0 = time.time()
        train_stats = rescore_rl_parquet(
            input_parquet_path=args.train_parquet,
            output_parquet_path=out_train,
            engine_path=args.engine_path,
            depth=args.depth,
            threads=args.threads,
            hash_mb=args.hash_mb,
            desc=f"Rescoring train (depth={args.depth})",
        )
        print_stats(args.depth, train_stats, elapsed_s=time.time() - t0)

        t0 = time.time()
        test_stats = rescore_rl_parquet(
            input_parquet_path=args.test_parquet,
            output_parquet_path=out_test,
            engine_path=args.engine_path,
            depth=args.depth,
            threads=args.threads,
            hash_mb=args.hash_mb,
            desc=f"Rescoring test (depth={args.depth})",
        )
        print_stats(args.depth, test_stats, elapsed_s=time.time() - t0)
        return


if __name__ == "__main__":
    main()
