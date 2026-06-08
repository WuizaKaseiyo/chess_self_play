"""Shared chess prompts for puzzle (chesslesson) and full-game (chess) environments."""
from __future__ import annotations

import os
import re
import sys
from typing import List, Optional, Tuple

# Shared board renderer (kept byte-identical with verl-chess's lichess bundle).
# Put the chesslesson bundle dir on sys.path so ``board_render`` resolves here
# and in every Ray worker, regardless of import order.
_LESSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chesslesson")
if _LESSON_DIR not in sys.path:
    sys.path.append(_LESSON_DIR)

from board_render import BOARD_REPRS, render_board  # noqa: E402

_EMPTY_FEN = "8/8/8/8/8/8/8/8 w - -"

POSITION_ENCODING = "The current board position is represented in FEN notation."

OUTPUT_FORMAT = """Reason step by step inside <think></think>, then put your answer in <action></action>.
For moves use UCI (e.g. e2e4). For coordinate tasks use the format the task asks for (e4, 3,5, or light/dark)."""

FULL_GAME_TASK = (
    "Win by checkmating the opponent. You play White; Black is played by the environment."
)

_GOAL_RE = re.compile(r"^Goal:\s*(.+)$", re.M)
_CONSTRAINTS_RE = re.compile(r"How this lesson works:\s*(.+?)(?:\n\n|\Z)", re.S)


def _marker_fen(square: str) -> str:
    try:
        import chess

        board = chess.Board(None)
        board.set_piece_at(
            chess.parse_square(square),
            chess.Piece(chess.KING, chess.WHITE),
        )
        return board.fen()
    except Exception:
        return _EMPTY_FEN


def _prompt_lines(
    task: str,
    *,
    player_to_move: Optional[str] = None,
    fen: Optional[str] = None,
    extra_lines: Optional[List[str]] = None,
) -> List[str]:
    lines = [POSITION_ENCODING, f"Task: {task}"]
    if player_to_move:
        lines.append(f"Player to move: {player_to_move}")
    if fen:
        lines.append(f"FEN: {fen}")
    if extra_lines:
        lines.extend(extra_lines)
    return lines


def _format_puzzle_body(task: dict) -> str:
    if task.get("kind") == "coordinate":
        return _format_coordinate_body(task)

    meta = task.get("meta") or {}
    user = task.get("user") or ""

    goal = meta.get("goal")
    if not goal:
        m = _GOAL_RE.search(user)
        goal = m.group(1).strip() if m else "Solve the chess puzzle."

    task_parts = [goal]
    cm = _CONSTRAINTS_RE.search(user)
    if cm:
        task_parts.append(cm.group(1).strip())

    nb_moves = meta.get("nbMoves")
    if nb_moves:
        task_parts.append(f"Use at most {nb_moves} move(s).")

    apples = meta.get("apples")
    if apples:
        task_parts.append(f"Visit target squares: {', '.join(apples)}.")

    task_text = " ".join(task_parts)

    return "\n".join(
        _prompt_lines(
            task_text,
            player_to_move=task.get("side_to_move"),
            fen=task.get("fen"),
        )
    )


def _render_coord_grid(marked_square: Optional[str], orientation: Optional[str]) -> str:
    """Render an 8x8 ASCII grid matching the chesslesson baked board layout.

    Marks `marked_square` with 'X' (for nameSquare mode), otherwise all '.'.
    From Black's view the file labels reverse (h..a) and ranks count 1..8 from top.
    Mirrors the layout used in coordinates.jsonl so the training prompt visually
    matches the baked data.
    """
    files = list("hgfedcba") if orientation == "black" else list("abcdefgh")
    ranks = list(range(1, 9)) if orientation == "black" else list(range(8, 0, -1))
    lines = ["   " + " ".join(files)]
    for r in ranks:
        cells = []
        for f in files:
            sq = f"{f}{r}"
            cells.append("X" if sq == (marked_square or "").lower() else ".")
        lines.append(f"{r} |" + " ".join(cells))
    return "\n".join(lines)


def _format_coordinate_body(task: dict) -> str:
    mode = task.get("mode", "")
    square = task.get("square", "")
    orient = task.get("orientation")
    view = f" ({orient} view)" if orient else ""

    if mode == "nameSquare":
        return "\n".join(
            _prompt_lines(
                f"Name the marked square{view}. The square is marked with X. "
                f"Answer with the square's algebraic name, e.g. <action>e4</action>.",
                extra_lines=[_render_coord_grid(square, orient)],
            )
        )
    if mode == "findSquare":
        return "\n".join(
            _prompt_lines(
                f"On an empty board{view}, where is square {square}? "
                f"Answer col,row (1-indexed from the top-left of the view), "
                f"e.g. <action>3,5</action>.",
                extra_lines=[_render_coord_grid(None, orient)],
            )
        )
    if mode == "squareColor":
        return "\n".join(
            _prompt_lines(
                f"Is square {square} a light square or a dark square? "
                f"Answer <action>light</action> or <action>dark</action>."
            )
        )

    first = (task.get("user") or "").strip().splitlines()[0]
    return "\n".join(_prompt_lines(first or "Answer the coordinate question."))


def build_puzzle_prompt(task: dict) -> str:
    return f"{_format_puzzle_body(task)}\n\n{OUTPUT_FORMAT}"


# --------------------------------------------------------------------------- #
# multi-turn ("Learn chess" lessons played one move per turn)
# --------------------------------------------------------------------------- #
_FEN_LINE_RE = re.compile(r"^FEN:.*$", re.M)


def _swap_fen_line(text: str, fen: str) -> str:
    """Replace the (first) ``FEN: ...`` line with the live position."""
    new, n = _FEN_LINE_RE.subn("FEN: " + fen, text, count=1)
    return new if n else text


def build_lesson_initial_obs(
    task: dict,
    render_fen: str,
    opening_opponent: Optional[List[str]] = None,
    *,
    board_repr: Optional[str] = None,
) -> str:
    """First turn of a multi-turn lesson: the full puzzle prompt re-pointed at the
    LIVE position (handles lessons that open with a scripted opponent move)."""
    body = _swap_fen_line(build_puzzle_prompt(task), render_fen)
    if board_repr:
        board_text = render_board(render_fen, board_repr)
        if board_text:
            body = _swap_fen_line(body, render_fen) + "\n" + board_text
    if opening_opponent:
        body = f"Opponent opened with: {' '.join(opening_opponent)}.\n" + body
    return body


def build_lesson_step_obs(
    render_fen: str,
    *,
    opponent_moves: Optional[List[str]] = None,
    apples_left: Optional[List[str]] = None,
    collected: Optional[str] = None,
    board_repr: Optional[str] = None,
) -> str:
    """Per-turn feedback after the player's move (+ any scripted opponent reply):
    new position, what was collected, remaining targets, ask for the next move."""
    lines: List[str] = []
    if opponent_moves:
        lines.append(f"Opponent replied: {' '.join(opponent_moves)}.")
    if collected:
        lines.append(f"You reached target {collected}.")
    lines.append("Position now:")
    lines.append(f"FEN: {render_fen}")
    if board_repr:
        board_text = render_board(render_fen, board_repr)
        if board_text:
            lines.append(board_text)
    if apples_left:
        lines.append(f"Remaining targets: {', '.join(apples_left)}.")
    lines.append("Play your next move inside <action></action>.")
    return "\n".join(lines)


def build_fullgame_position(fen: str, side_to_move: str = "White") -> str:
    """Snippet stored in env memory between steps (player + FEN only)."""
    return "\n".join(
        [
            f"Player to move: {side_to_move}",
            f"FEN: {fen}",
        ]
    )


def parse_fullgame_position(block: str) -> Tuple[str, str]:
    fen = ""
    side = "White"
    for line in (block or "").splitlines():
        if line.startswith("Player to move:"):
            side = line.split(":", 1)[1].strip()
        elif line.startswith("FEN:"):
            fen = line.split(":", 1)[1].strip()
    return fen, side


def build_fullgame_prompt(
    fen: str,
    side_to_move: str = "White",
    *,
    task_line: str = FULL_GAME_TASK,
    history_block: Optional[str] = None,
    board_repr: str = "ascii",
) -> str:
    lines = _prompt_lines(task_line, player_to_move=side_to_move, fen=fen)
    body = "\n".join(lines)
    # Render the board the same way chesslesson does (default: labelled ASCII grid).
    board_text = render_board(fen, board_repr)
    if board_text:
        body = f"{body}\n{board_text}"
    if history_block:
        body = f"{body}\n\n{history_block}"
    return f"{body}\n\n{OUTPUT_FORMAT}"


def build_fullgame_history_block(
    step_count: int,
    history_length: int,
    action_history: str,
    current_step: int,
) -> str:
    return (
        f"History ({history_length} recent step(s), you have moved {step_count} time(s), "
        f"now move {current_step}):\n{action_history}"
    )


def build_task_obs(task: dict) -> str:
    return build_puzzle_prompt(task)
