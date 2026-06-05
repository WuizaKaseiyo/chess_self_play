"""Single source of truth for rendering a chess board layout in a prompt.

Shared by both environments so the board *looks the same* everywhere:
  - chesslesson (single-turn, verl-chess): prepare_data.py bakes it into parquet
  - chess full-game (multi-turn, verl-agent): prompts_shared.py renders it online

Representations (`repr_`):
  ascii     : labelled grid, uppercase=White / lowercase=Black / "." = empty (default)
  fen       : nothing extra -- the caller's "FEN:" line already carries the layout
  piecelist : an explicit per-side piece list (compact; best for sparse positions)
  unicode   : a grid drawn with unicode chess glyphs

`render_board` returns the text that should follow the caller's "FEN: ..." line
(empty string for `fen`). Only `python-chess` is needed, and only for the
non-fen representations (imported lazily).

This file is bundled (kept byte-identical) in:
  verl-chess/recipe/chesslesson/lichess/board_render.py
  verl-agent/chess_game/chesslesson/board_render.py
"""
from __future__ import annotations

BOARD_REPRS = ("ascii", "fen", "piecelist", "unicode")

_UNICODE = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}


def _board_from_fen(fen):
    import chess  # lazy: only needed for non-fen representations

    return chess.Board(fen)


def _render_grid(board, glyphs=None):
    import chess

    rows = ["   a b c d e f g h"]
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            p = board.piece_at(chess.square(file, rank))
            if p is None:
                cells.append(".")
            elif glyphs:
                cells.append(glyphs[p.symbol()])
            else:
                cells.append(p.symbol())
        rows.append("%d |%s" % (rank + 1, " ".join(cells)))
    return "\n".join(rows)


def _render_piecelist(board):
    import chess

    def side(color):
        out = []
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == color:
                out.append("%s%s" % (p.symbol().upper(), chess.square_name(sq)))
        order = "KQRBNP"
        out.sort(key=lambda s: (order.index(s[0]), s[1:]))
        return out

    w, b = side(chess.WHITE), side(chess.BLACK)
    return "Pieces (UCI squares) -- White: %s | Black: %s" % (
        ", ".join(w) or "none",
        ", ".join(b) or "none",
    )


def render_board(fen, repr_="ascii"):
    """Text to place after the prompt's "FEN: ..." line (empty for `fen`)."""
    if repr_ == "fen":
        return ""
    if repr_ not in BOARD_REPRS:
        raise ValueError("unknown board repr: %s" % repr_)
    board = _board_from_fen(fen)
    if repr_ == "ascii":
        return ('Board (uppercase = White, lowercase = Black, "." = empty):\n'
                + _render_grid(board))
    if repr_ == "unicode":
        return ('Board (unicode glyphs, white pieces are outlined):\n'
                + _render_grid(board, _UNICODE))
    if repr_ == "piecelist":
        return _render_piecelist(board)
    raise ValueError("unknown board repr: %s" % repr_)  # unreachable
