#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.pgn

# Make repo-root imports work when invoked as `python scripts/xxx.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import recipe.chess.full_game_eval as fge


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_pgn_text(pgn_text: str) -> chess.pgn.Game:
    fp = io.StringIO(pgn_text)
    g = chess.pgn.read_game(fp)
    if g is None:
        raise ValueError("Failed to parse PGN text")
    return g


def _replay_game_to_board(pgn_game: chess.pgn.Game) -> chess.Board:
    board = pgn_game.board()
    for mv in pgn_game.mainline_moves():
        board.push(mv)
    return board


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round-robin pilot (ACPL stage): compute ACPL/accuracy from stage-1 PGNs.")
    p.add_argument("--infer-out-dir", type=str, default="", help="Output directory produced by the infer stage.")
    p.add_argument(
        "--infer-out-root",
        type=str,
        default="",
        help="Root directory produced by the distributed infer stage (contains shard_*/games.jsonl).",
    )

    # Stockfish / ACPL settings (mirror aligned defaults).
    p.add_argument("--stockfish-path", type=str, default="/usr/local/bin/stockfish")
    p.add_argument("--stockfish-hash-mb", type=int, default=128)
    p.add_argument("--acpl-depth", type=int, default=20)
    p.add_argument("--acpl-movetime-ms", type=int, default=1000)
    p.add_argument("--acpl-cp-cap", type=int, default=1000)
    p.add_argument("--mate-score-cp", type=int, default=1000)
    p.add_argument("--resignation-cpl", type=int, default=1000)
    p.add_argument("--acpl-workers", type=int, default=72)
    p.add_argument("--acpl-threads", type=int, default=2)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.infer_out_dir) == bool(args.infer_out_root):
        raise ValueError("Provide exactly one of --infer-out-dir or --infer-out-root")

    out_dir = Path(args.infer_out_dir or args.infer_out_root)
    if not out_dir.exists():
        raise FileNotFoundError(f"Missing infer output directory: {out_dir}")

    games_paths: List[Path] = []
    if args.infer_out_dir:
        games_paths = [out_dir / "games.jsonl"]
    else:
        shards = sorted([p for p in out_dir.glob("shard_*") if p.is_dir()], key=lambda p: p.name)
        for shard in shards:
            gp = shard / "games.jsonl"
            if gp.exists():
                games_paths.append(gp)

    if not games_paths:
        raise FileNotFoundError(f"No stage-1 games.jsonl files found under: {out_dir}")

    rows: List[Dict[str, Any]] = []
    for gp in games_paths:
        part = _read_jsonl(gp)
        if not part:
            raise ValueError(f"No rows in {gp}")
        rows.extend(part)
    if not rows:
        raise ValueError("No rows loaded from infer outputs")

    run_args_path = out_dir / "run_args_acpl.json"
    run_args_path.write_text(
        json.dumps({"argv": sys.argv, "parsed_args": vars(args), "infer_games_jsonl": [str(p) for p in games_paths]}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"[rr][acpl] wrote run args: {run_args_path}", flush=True)

    # Reconstruct game states for the shared ACPL analyzer.
    games: List[fge._GameState] = []
    forfeit_color_by_game: Dict[str, chess.Color] = {}
    meta_by_game_id: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        game_id = str(r["game_id"])
        pair_id = str(r["pair_id"])
        run_id_a = str(r["run_id_a"])
        run_id_b = str(r["run_id_b"])
        white_run_id = str(r["white_run_id"])
        black_run_id = str(r["black_run_id"])
        pgn_text = str(r.get("pgn") or "").strip()
        if not pgn_text:
            raise ValueError(f"Missing pgn text for game_id={game_id}")

        pgn_game = _parse_pgn_text(pgn_text)
        board = _replay_game_to_board(pgn_game)

        a_is_white = run_id_a == white_run_id
        model_color = chess.WHITE if a_is_white else chess.BLACK

        g = fge._GameState(
            game_id=game_id,
            opponent_depth=0,
            model_color=model_color,
            board=board,
            pgn=pgn_game,
            pgn_node=pgn_game,
            round_idx=0,
            game_idx_in_round=0,
        )
        g.is_over = True
        g.termination = str(r.get("termination") or "")
        g.result = str(r.get("pgn_result") or r.get("result") or "")
        g.forfeit = bool(r.get("forfeit") or False)
        g.forfeit_reason = str(r.get("forfeit_reason") or "")
        g.result_str = str(r.get("result") or "")

        setattr(g, "pair_id", pair_id)
        setattr(g, "white_run_id", white_run_id)
        setattr(g, "black_run_id", black_run_id)
        setattr(g, "run_id_a", run_id_a)
        setattr(g, "run_id_b", run_id_b)

        forfeited_color = r.get("forfeit_color", None)
        if forfeited_color == "white":
            forfeit_color_by_game[game_id] = chess.WHITE
        elif forfeited_color == "black":
            forfeit_color_by_game[game_id] = chess.BLACK

        meta_by_game_id[game_id] = dict(r)
        games.append(g)

    cfg = fge.FullGameEvalConfig(
        opponent_depths=[0],
        games_per_depth=len(games),
        seed=0,
        resignation_cpl=int(args.resignation_cpl),
        acpl_eval_depth=int(args.acpl_depth),
        acpl_eval_movetime_ms=int(args.acpl_movetime_ms),
        acpl_eval_cp_cap=int(args.acpl_cp_cap),
        mate_score_cp=int(args.mate_score_cp),
        stockfish_opponent=fge.StockfishConfig(
            path=str(args.stockfish_path),
            threads=1,
            hash_mb=int(args.stockfish_hash_mb),
            skill_level=0,
        ),
        stockfish_eval=fge.StockfishConfig(
            path=str(args.stockfish_path),
            threads=int(args.acpl_threads),
            hash_mb=int(args.stockfish_hash_mb),
            skill_level=20,
        ),
        acpl_workers=int(args.acpl_workers),
        out_dir=out_dir,
    )

    t0 = time.time()
    analyses = fge._analyze_games_with_engine(cfg=cfg, eval_engine=None, games=games)
    t1 = time.time()
    acpl_time_s = float(t1 - t0)

    resignation_penalty = float(cfg.resignation_cpl or 0)

    games_acpl_path = out_dir / "games_acpl.jsonl"
    summary_path = out_dir / "summary_acpl.json"

    # Aggregate by pair.
    per_pair: Dict[str, Dict[str, Any]] = {}
    with open(games_acpl_path, "w", encoding="utf-8") as fp:
        for g in games:
            analysis = analyses.get(g.game_id, None) or {}
            white_acpl = float(analysis.get("white_acpl", float(cfg.acpl_eval_cp_cap)))
            black_acpl = float(analysis.get("black_acpl", float(cfg.acpl_eval_cp_cap)))
            white_accuracy_pct = float(analysis.get("white_accuracy_pct", 0.0))
            black_accuracy_pct = float(analysis.get("black_accuracy_pct", 0.0))
            white_moves = int(analysis.get("white_moves", 0))
            black_moves = int(analysis.get("black_moves", 0))
            white_cpl_sum = float(analysis.get("white_cpl_sum", 0.0))
            black_cpl_sum = float(analysis.get("black_cpl_sum", 0.0))

            resignation_penalty_white = 0.0
            resignation_penalty_black = 0.0
            if g.forfeit and resignation_penalty > 0:
                forfeited_color = forfeit_color_by_game.get(g.game_id, None)
                if forfeited_color == chess.WHITE:
                    resignation_penalty_white = float(resignation_penalty)
                    white_cpl_sum += resignation_penalty
                    if white_moves <= 0:
                        white_moves = 1
                    white_acpl = float(white_cpl_sum / white_moves)
                elif forfeited_color == chess.BLACK:
                    resignation_penalty_black = float(resignation_penalty)
                    black_cpl_sum += resignation_penalty
                    if black_moves <= 0:
                        black_moves = 1
                    black_acpl = float(black_cpl_sum / black_moves)

            pair_id = getattr(g, "pair_id")
            run_id_a = getattr(g, "run_id_a")
            run_id_b = getattr(g, "run_id_b")
            white_run_id = getattr(g, "white_run_id")
            black_run_id = getattr(g, "black_run_id")
            a_is_white = run_id_a == white_run_id

            model_a_game_acpl = float(white_acpl if a_is_white else black_acpl)
            model_b_game_acpl = float(black_acpl if a_is_white else white_acpl)
            model_a_game_moves = int(white_moves if a_is_white else black_moves)
            model_b_game_moves = int(black_moves if a_is_white else white_moves)
            model_a_game_cpl_sum = float(white_cpl_sum if a_is_white else black_cpl_sum)
            model_b_game_cpl_sum = float(black_cpl_sum if a_is_white else white_cpl_sum)

            if pair_id not in per_pair:
                per_pair[pair_id] = {
                    "run_id_a": run_id_a,
                    "run_id_b": run_id_b,
                    "games_total": 0,
                    "model_a": {"wins": 0, "losses": 0, "draws": 0, "acpl_sum": 0.0, "acpl_games": 0, "cpl_sum": 0.0, "moves": 0},
                    "model_b": {"wins": 0, "losses": 0, "draws": 0, "acpl_sum": 0.0, "acpl_games": 0, "cpl_sum": 0.0, "moves": 0},
                }

            per_pair[pair_id]["games_total"] += 1

            # Outcomes for model A/B (same attribution as infer stage).
            if g.result == "1/2-1/2":
                per_pair[pair_id]["model_a"]["draws"] += 1
                per_pair[pair_id]["model_b"]["draws"] += 1
            else:
                a_won = (g.result == "1-0" and a_is_white) or (g.result == "0-1" and (not a_is_white))
                if a_won:
                    per_pair[pair_id]["model_a"]["wins"] += 1
                    per_pair[pair_id]["model_b"]["losses"] += 1
                else:
                    per_pair[pair_id]["model_a"]["losses"] += 1
                    per_pair[pair_id]["model_b"]["wins"] += 1

            per_pair[pair_id]["model_a"]["acpl_sum"] += float(model_a_game_acpl)
            per_pair[pair_id]["model_a"]["acpl_games"] += 1
            per_pair[pair_id]["model_a"]["cpl_sum"] += float(model_a_game_cpl_sum)
            per_pair[pair_id]["model_a"]["moves"] += int(model_a_game_moves)

            per_pair[pair_id]["model_b"]["acpl_sum"] += float(model_b_game_acpl)
            per_pair[pair_id]["model_b"]["acpl_games"] += 1
            per_pair[pair_id]["model_b"]["cpl_sum"] += float(model_b_game_cpl_sum)
            per_pair[pair_id]["model_b"]["moves"] += int(model_b_game_moves)

            fge._jsonl_write(
                fp,
                {
                    "ts": time.time(),
                    "stage": "acpl",
                    "game_id": g.game_id,
                    "pair_id": pair_id,
                    "run_id_a": run_id_a,
                    "run_id_b": run_id_b,
                    "white_run_id": white_run_id,
                    "black_run_id": black_run_id,
                    "pgn_result": g.result,
                    "termination": g.termination,
                    "forfeit": bool(g.forfeit),
                    "forfeit_reason": g.forfeit_reason,
                    "forfeit_color": meta_by_game_id.get(g.game_id, {}).get("forfeit_color"),
                    "white_acpl": float(white_acpl),
                    "black_acpl": float(black_acpl),
                    "white_accuracy_pct": float(white_accuracy_pct),
                    "black_accuracy_pct": float(black_accuracy_pct),
                    "white_moves": int(white_moves),
                    "black_moves": int(black_moves),
                    "white_cpl_sum": float(white_cpl_sum),
                    "black_cpl_sum": float(black_cpl_sum),
                    "resignation_penalty_white": float(resignation_penalty_white),
                    "resignation_penalty_black": float(resignation_penalty_black),
                    "model_a_acpl": float(model_a_game_acpl),
                    "model_b_acpl": float(model_b_game_acpl),
                },
            )

    # Finalize per-pair aggregates (means).
    for pid, d in per_pair.items():
        for side in ("model_a", "model_b"):
            acpl_games = int(d[side]["acpl_games"])
            moves = int(d[side]["moves"])
            d[side]["acpl_mean"] = float(d[side]["acpl_sum"] / acpl_games) if acpl_games > 0 else float("nan")
            d[side]["acpl_mean_per_move"] = float(d[side]["cpl_sum"] / moves) if moves > 0 else float("nan")
            d[side]["acpl_moves"] = moves
            # Drop intermediate sums to keep summary compact.
            d[side].pop("acpl_sum", None)
            d[side].pop("acpl_games", None)
            d[side].pop("cpl_sum", None)
            d[side].pop("moves", None)

    summary = {
        "config": {
            "stage": "acpl",
            "infer_out_dir": str(out_dir),
            "infer_games_jsonl": [str(p) for p in games_paths],
            "stockfish_path": str(args.stockfish_path),
            "stockfish_hash_mb": int(args.stockfish_hash_mb),
            "acpl_depth": int(args.acpl_depth),
            "acpl_movetime_ms": int(args.acpl_movetime_ms),
            "acpl_cp_cap": int(args.acpl_cp_cap),
            "mate_score_cp": int(args.mate_score_cp),
            "resignation_cpl": int(args.resignation_cpl),
            "acpl_workers": int(args.acpl_workers),
            "acpl_threads": int(args.acpl_threads),
            "num_games": int(len(games)),
        },
        "timing": {"acpl_time_s": float(acpl_time_s)},
        "results": {"by_pair": per_pair},
        "paths": {
            "infer_games_jsonl": [str(p) for p in games_paths],
            "games_acpl_jsonl": str(games_acpl_path),
            "summary_acpl_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[rr][acpl] wrote: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
