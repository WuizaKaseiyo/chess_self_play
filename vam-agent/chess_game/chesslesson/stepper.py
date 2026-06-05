"""Stateful, move-by-move driver for the lichess "Learn chess" lessons.

`reward.evaluate()` judges a *whole* move list in one shot (single-turn). For
multi-turn online RL the model plays one move per turn and must SEE the board
after the (optionally scripted) opponent reply before choosing the next move.

`LessonStepper` exposes exactly that as reset/step, while reproducing
`evaluate()`'s control flow line-for-line so the verdict is identical for any
given move sequence:

    st = LessonStepper(SPECS["enpassant-1"])
    obs = st.observation()          # FEN/board after any opening opponent move
    obs = st.step("c5d6")           # apply player move (+ scripted reply), observe
    ...                             # repeat until obs["done"]
    reward = 1 if obs["success"] else 0   # canonical lichess binary verdict

Both lesson families are handled by the same step():
  - scripted opponent (16/110 specs have a non-empty "scenario")
  - no opponent     (the other 94: the player keeps moving on one board)

This file is the source of truth; keep it byte-identical with any bundled copy.
"""
from __future__ import annotations

from reward import Chess, Scenario, decompose_uci, eval_spec, read_keys, _clean_uci  # noqa


class LessonStepper:
    def __init__(self, spec):
        self.spec = spec
        self.apple_keys = read_keys(spec.get("apples"))
        self.pc = "b" if spec["color"] == "black" else "w"
        self.chess = Chess(spec["fen"], self.apple_keys)
        self.scenario = Scenario(spec.get("scenario"), self.chess)
        self.items = set(self.apple_keys)
        self.vm = {"nbMoves": 0, "failed": False, "completed": False}
        self.data = {"chess": self.chess, "vm": self.vm, "scenario": self.scenario}
        self.success_spec = spec.get("success")
        self.failure_spec = spec.get("failure")
        self.done = False
        self.reason = None
        # mirror evaluate(): if the lesson opens on the opponent's turn, it moves first
        self.opening_opponent = []
        if self.chess.get_color() != self.pc:
            n = len(self.chess.history)
            self.scenario.opponent()
            self.opening_opponent = list(self.chess.history[n:])

    # --- introspection / rendering ------------------------------------------
    def board_fen(self):
        """Piece placement only (what board_render needs)."""
        return self.chess.fen()

    def render_fen(self):
        """A full FEN (placement + side-to-move) suitable for python-chess."""
        return "%s %s - -" % (self.chess.fen(), self.pc)

    def apples_left(self):
        return sorted(self.items)

    def _detect_capture(self):
        if not self.spec.get("detectCapture"):
            return False
        return self.chess.find_unprotected_capture() is not None

    def observation(self, *, legal=True, player_move=None, opponent_moves=None,
                    collected=None):
        return {
            "done": self.done,
            "success": bool(self.vm["completed"]),
            "failed": bool(self.vm["failed"]),
            "reason": self.reason,
            "legal": legal,
            "player_move": player_move,
            "opponent_moves": list(opponent_moves or []),
            "collected": collected,
            "apples_left": self.apples_left(),
            "nbMoves": self.vm["nbMoves"],
            "budget": self.spec.get("nbMoves"),
            "board_fen": self.board_fen(),
            "render_fen": self.render_fen(),
            "side_to_move": "white" if self.pc == "w" else "black",
        }

    # --- one player turn ----------------------------------------------------
    def step(self, uci):
        """Apply one player move; mirrors a single iteration of evaluate()'s loop."""
        if self.done:
            return self.observation(player_move=uci)

        mv = _clean_uci(uci)
        self.vm["nbMoves"] += 1

        if not mv:  # unparseable -> treat as illegal (evaluate skips empties up front)
            self.vm["failed"] = True
            self.reason = "no legal move parsed from turn #%d: %r" % (self.vm["nbMoves"], uci)
            self.done = True
            return self.observation(legal=False, player_move=uci)

        o, d, p = decompose_uci(mv)
        if not self.chess.move(o, d, p):
            self.vm["failed"] = True
            self.reason = "illegal / into-check move #%d: %s" % (self.vm["nbMoves"], mv)
            self.done = True
            return self.observation(legal=False, player_move=mv)

        collected = d if d in self.items else None
        if collected is not None:
            self.items.discard(d)

        n = len(self.chess.history)
        in_scenario = self.scenario.player(mv)   # also plays the scripted opponent reply
        opponent_moves = list(self.chess.history[n:])

        if not in_scenario:
            if self._detect_capture():
                self.vm["failed"] = True
                self.reason = "left a piece hanging"
            if self.failure_spec and eval_spec(self.failure_spec, self.data):
                self.vm["failed"] = True
                self.reason = self.reason or "failure condition met"

        ok_now = eval_spec(self.success_spec, self.data) if self.success_spec else (len(self.items) == 0)
        if not self.vm["failed"] and ok_now:
            self.vm["completed"] = True
            self.done = True
        elif self.vm["failed"]:
            self.done = True
        else:
            if not in_scenario:
                self.chess.set_color(self.pc)

        return self.observation(player_move=mv, opponent_moves=opponent_moves,
                                collected=collected)


def replay(spec, moves):
    """Drive a LessonStepper move-by-move; returns evaluate()-shaped verdict.

    Used to prove equivalence with reward.evaluate() for any move sequence.
    """
    st = LessonStepper(spec)
    obs = st.observation()
    for mv in (moves or []):
        if st.done:
            break
        obs = st.step(mv)
    return {
        "success": obs["success"],
        "reward": 1 if obs["success"] else 0,
        "details": {"failed": obs["failed"], "reason": obs["reason"],
                    "nbMoves": obs["nbMoves"]},
    }
