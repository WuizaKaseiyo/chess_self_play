"""chess_game package.

Three independent environments live here:
  * chess/WhiteVsRandom — full-game vs random Black (ray_envs / projection),
    optional Verbalized Action Masking via env.chess.vam.* config.
  * chesslesson/LichessLearn — single-turn Lichess "Learn chess" puzzles
    (chesslesson_envs), self-contained under chess_game/chesslesson/.
  * lichess_puzzle/Curriculum — single-turn Chess-R1 / SF μ-graded puzzles
    over chess-rl-C224 schema parquet (lichess_puzzle_envs), with optional
    VAM (subset_source='mu_topk' uses precomputed μ ranking).

The full-game imports are done lazily so the puzzle envs keep working even
when `python-chess` is not installed.
"""
from chess_game.chesslesson_envs import build_chesslesson_envs, chesslesson_projection
from chess_game.lichess_puzzle_envs import (
    build_lichess_puzzle_envs,
    lichess_puzzle_projection,
)

__all__ = [
    "chess_projection",
    "build_chess_envs",
    "train_chess",
    "build_chesslesson_envs",
    "chesslesson_projection",
    "build_lichess_puzzle_envs",
    "lichess_puzzle_projection",
]


def __getattr__(name):
    if name in ("chess_projection",):
        from chess_game.projection import chess_projection
        return chess_projection
    if name in ("build_chess_envs",):
        from chess_game.ray_envs import build_chess_envs
        return build_chess_envs
    if name in ("train_chess",):
        from chess_game.train_chess import train_chess
        return train_chess
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
