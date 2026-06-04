from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import chess
import chess.engine

from recipe.chess.reward_fn import centipawn_to_win_prob

# Match `scripts/rescore_puzzles_cp.py` for consistency.
MATE_SCORE_CP = 100_000


def analyse_all_legal_moves_multipv(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    depth: int,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Analyze all legal moves with Stockfish, preferring a single MultiPV call.

    Returns:
      - move_to_cp: {uci -> centipawns} from side-to-move POV
      - move_to_expected_score: {uci -> expected score in [0,1]} when WDL is available

    This is intentionally aligned to `scripts/rescore_puzzles_cp.py::analyse_all_legal_moves_multipv`,
    including its fallback behavior when MultiPV omits some moves.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return {}, {}

    # python-chess will set the MultiPV option automatically when `multipv` is passed.
    info = engine.analyse(board, chess.engine.Limit(depth=int(depth)), multipv=len(legal_moves))
    info_list = info if isinstance(info, list) else [info]

    move_to_cp: Dict[str, int] = {}
    move_to_expected_score: Dict[str, float] = {}
    for entry in info_list:
        pv = entry.get("pv") or []
        if not pv:
            continue
        move_uci = pv[0].uci().lower()
        score = entry.get("score")
        if score is None:
            continue
        cp = score.pov(board.turn).score(mate_score=MATE_SCORE_CP)
        if cp is None:
            continue
        move_to_cp[move_uci] = int(cp)

        wdl = entry.get("wdl")
        if wdl is not None:
            try:
                wdl_pov = wdl.pov(board.turn)
                denom = max(1, int(wdl_pov.wins + wdl_pov.draws + wdl_pov.losses))
                move_to_expected_score[move_uci] = float((wdl_pov.wins + 0.5 * wdl_pov.draws) / denom)
            except Exception:
                pass

    # Fallback for any missing move (rare but possible).
    legal_uci = [m.uci().lower() for m in legal_moves]
    missing = [m for m in legal_uci if m not in move_to_cp]
    for move_uci in missing:
        mv = chess.Move.from_uci(move_uci)
        entry = engine.analyse(board, chess.engine.Limit(depth=int(depth)), root_moves=[mv])
        cp = entry["score"].pov(board.turn).score(mate_score=MATE_SCORE_CP)
        if cp is None:
            continue
        move_to_cp[move_uci] = int(cp)

        wdl = entry.get("wdl")
        if wdl is not None:
            try:
                wdl_pov = wdl.pov(board.turn)
                denom = max(1, int(wdl_pov.wins + wdl_pov.draws + wdl_pov.losses))
                move_to_expected_score[move_uci] = float((wdl_pov.wins + 0.5 * wdl_pov.draws) / denom)
            except Exception:
                pass

    return move_to_cp, move_to_expected_score


def score_position_all_legal_moves(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    depth: int,
) -> tuple[list[str], dict[str, int], dict[str, float], dict[str, float]]:
    """Compute per-move maps over *all* legal moves in python-chess order.

    Returns:
      - legal_moves_uci: ordered list[str]
      - move_cps: {uci -> cp} for all legal moves
      - move_winprobs: {uci -> win_prob} for all legal moves (cp -> sigmoid)
      - move_expected_scores: {uci -> expected_score} for all legal moves in [0,1]

    Notes on expected-score semantics:
      - When Stockfish WDL is available for *all* legal moves, expected scores are computed as:
          expected = (wins + 0.5 * draws) / (wins + draws + losses)
        from the POV of the side to move.
      - When WDL is not available (or incomplete), we fall back deterministically to the CP-derived
        win-probability map (same mapping as `centipawn_to_win_prob`). This keeps online/self-play
        reward payloads schema-compatible with offline datasets and avoids NaN rewards in reward modes
        that require `move_expected_scores_json`.
    """
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [m.uci().lower() for m in legal_moves]
    if not legal_moves_uci:
        return [], {}, {}, {}

    move_to_cp, move_to_expected = analyse_all_legal_moves_multipv(engine, board, depth=int(depth))

    # Ensure we cover every legal move; if not, fill missing with 0cp (neutral).
    # This should be rare due to the fallback path, but keep it robust.
    for mv in legal_moves_uci:
        if mv not in move_to_cp:
            move_to_cp[mv] = 0

    move_winprob: dict[str, float] = {mv: float(centipawn_to_win_prob(move_to_cp[mv])) for mv in legal_moves_uci}

    # Only treat WDL expected-score maps as "available" when they cover all legal moves.
    expected_complete = all(mv in move_to_expected for mv in legal_moves_uci)
    if expected_complete:
        move_expected_scores = {mv: float(move_to_expected[mv]) for mv in legal_moves_uci}
    else:
        # Deterministic fallback: use CP-derived winprob as a proxy expected score.
        move_expected_scores = dict(move_winprob)

    move_cp_all = {mv: int(move_to_cp[mv]) for mv in legal_moves_uci}
    return legal_moves_uci, move_cp_all, move_winprob, move_expected_scores


def dumps_compact_sorted(obj: Any) -> str:
    """Stable JSON formatting for reward payload fields (sorted keys, compact)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
