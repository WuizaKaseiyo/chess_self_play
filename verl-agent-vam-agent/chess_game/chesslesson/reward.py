"""Canonical reward / verifier for the lichess "Learn chess" tasks (Python port).

Mirrors reward.js exactly. The success/failure rules are lichess's own asserts,
dumped to reward_specs.json by dump_specs.js as serializable descriptor trees,
and interpreted here. The driver replicates ui/learn/src/levelCtrl.makeSendMove
(detectSuccess / detectFailure / detectCapture, move-into-check = fail, the
<2-kings => antichess no-check rule, apple collisions) and scenario.ts.

    from reward import evaluate_by_id
    evaluate_by_id("check1-1", ["a1e1"])
    # -> {"success": True, "reward": 1, ...}
"""
import json
import os
import re

import chesslib as CL

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "reward_specs.json")) as f:
    SPECS = json.load(f)

ROLE = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}


def decompose_uci(u):
    return (u[0:2], u[2:4], u[4:5] if len(u) > 4 else "")


def _clean_uci(m):
    """Normalize one move token to bare UCI (or '' if nothing usable)."""
    return re.sub(r"[^a-h1-8qrbn]", "", str(m).strip().lower())


def read_keys(keys):
    if isinstance(keys, str):
        return [k for k in keys.strip().split() if k]
    return keys or []


def _count_kings(board):
    w = b = 0
    for row in board:
        for p in row:
            if p == "K":
                w += 1
            elif p == "k":
                b += 1
    return w, b


class Chess:
    """Subset of lichess ChessCtrl over chesslib; antichess (no-check) when <2 kings."""

    def __init__(self, fen, apple_keys):
        self.s = CL.parse(fen)
        enemy_pawn = "p" if self.s["turn"] == "w" else "P"
        for key in apple_keys or []:
            r, c = CL.sq_to_rc(key)
            if not self.s["board"][r][c]:
                self.s["board"][r][c] = enemy_pawn
        wk, bk = _count_kings(self.s["board"])
        self.antichess = wk < 1 or bk < 1
        self.history = []

    def get(self, sq):
        r, c = CL.sq_to_rc(sq)
        p = self.s["board"][r][c]
        if not p:
            return None
        return {"role": ROLE[p.lower()], "color": "white" if CL.is_w(p) else "black"}

    def occupied_keys(self):
        out = []
        for r in range(8):
            for c in range(8):
                if self.s["board"][r][c]:
                    out.append(CL.rc_to_sq(r, c))
        return out

    def get_color(self):
        return self.s["turn"]

    def set_color(self, color):
        self.s["turn"] = color
        self.s["ep"] = None

    def fen(self):
        return CL.board_fen(self.s["board"])

    def is_check(self):
        return CL.in_check(self.s, self.s["turn"])

    def default_chess_mate(self):
        return CL.in_check(self.s, self.s["turn"]) and len(CL.legal_moves(self.s)) == 0

    def san_history(self):
        return self.history

    def move(self, orig, dest, prom):
        cands = [m for m in CL.pseudo(self.s) if m["from"] == orig and m["to"] == dest]
        if not cands:
            return None
        if prom:
            m = next((c for c in cands if c.get("promo") == prom), None)
        else:
            m = next((c for c in cands if c.get("promo") == "q"), cands[0])
        if not m:
            return None
        ns = CL.apply_move(self.s, m)
        if not self.antichess and CL.in_check(ns, self.s["turn"]):
            return None
        self.s = ns
        flag = m.get("flag")
        self.history.append("O-O" if flag == "O-O" else "O-O-O" if flag == "O-O-O" else orig + dest)
        return {"from": orig, "to": dest, "flag": flag}

    def find_unprotected_capture(self):
        caps = []
        for m in CL.legal_moves(self.s):
            r, c = CL.sq_to_rc(m["to"])
            if self.s["board"][r][c] or m.get("flag") == "ep":
                caps.append(m)
        for cap in caps:
            ns = CL.apply_move(self.s, cap)
            ns["turn"] = "b" if self.s["turn"] == "w" else "w"
            if not any(mm["to"] == cap["to"] for mm in CL.pseudo(ns)):
                return cap
        return None


# ---- assert interpreter (assert.ts ported, operating on spec dicts) ----
def _matcher(fp):
    return {"role": ROLE[fp.lower()], "color": "black" if fp.islower() else "white"}


def _piece_match(piece, m):
    return bool(piece) and piece["role"] == m["role"] and piece["color"] == m["color"]


def eval_spec(spec, data):
    """data = {'chess':Chess, 'vm':{'nbMoves':int}, 'scenario':Scenario}."""
    op, args = spec["op"], spec.get("args", [])
    chess, vm, scenario = data["chess"], data["vm"], data["scenario"]
    if op == "pieceOn":
        return _piece_match(chess.get(args[1]), _matcher(args[0]))
    if op == "pieceNotOn":
        return not _piece_match(chess.get(args[1]), _matcher(args[0]))
    if op == "noPieceOn":
        keys = read_keys(args[0])
        return any(sq not in keys for sq in chess.occupied_keys())
    if op == "whitePawnOnAnyOf":
        keys = read_keys(args[0])
        return any(_piece_match(chess.get(k), _matcher("P")) for k in keys)
    if op == "extinct":
        f = chess.fen().replace("/", "")
        return f == (f.lower() if args[0] == "white" else f.upper())
    if op == "check":
        return chess.is_check()
    if op == "mate":
        return chess.default_chess_mate()
    if op == "lastMoveSan":
        h = chess.san_history()
        return bool(h) and h[-1] == args[0]
    if op == "checkIn":
        return vm["nbMoves"] <= args[0] and chess.is_check()
    if op == "noCheckIn":
        return vm["nbMoves"] >= args[0] and not chess.is_check()
    if op == "not":
        return not eval_spec(args[0], data)
    if op == "and":
        return all(eval_spec(a, data) for a in args)
    if op == "or":
        return any(eval_spec(a, data) for a in args)
    if op == "scenarioComplete":
        return scenario.is_complete()
    if op == "scenarioFailed":
        return scenario.is_failed()
    raise ValueError("unknown assert op: " + op)


# ---- scenario.ts (synchronous) ----
class Scenario:
    def __init__(self, moves, chess):
        self.steps = list(moves or [])
        self.chess = chess
        self.it = 0
        self.failed = False

    def is_complete(self):
        return self.it == len(self.steps)

    def is_failed(self):
        return self.failed

    def opponent(self):
        if self.it >= len(self.steps):
            return
        o, d, p = decompose_uci(self.steps[self.it])
        if not self.chess.move(o, d, p):
            self.failed = True
            return
        self.it += 1

    def player(self, uci):
        if self.it >= len(self.steps):
            return False
        if self.steps[self.it] != uci:
            self.failed = True
            return False
        self.it += 1
        self.opponent()
        return True


# ---- driver (levelCtrl.makeSendMove, synchronous) ----
def evaluate(spec, moves):
    """spec = a reward_specs.json entry; moves = player UCI list."""
    moves = [_clean_uci(m) for m in (moves or [])]
    moves = [m for m in moves if m]
    apple_keys = read_keys(spec.get("apples"))
    pc = "b" if spec["color"] == "black" else "w"
    chess = Chess(spec["fen"], apple_keys)
    scenario = Scenario(spec.get("scenario"), chess)
    items = set(apple_keys)
    vm = {"nbMoves": 0, "failed": False, "completed": False}
    data = {"chess": chess, "vm": vm, "scenario": scenario}

    success_spec = spec.get("success")
    failure_spec = spec.get("failure")

    def detect_capture():
        if not spec.get("detectCapture"):
            return False
        return chess.find_unprotected_capture() is not None

    if chess.get_color() != pc:
        scenario.opponent()

    reason = None
    for i, mv in enumerate(moves):
        vm["nbMoves"] += 1
        o, d, p = decompose_uci(mv)
        if not chess.move(o, d, p):
            vm["failed"] = True
            reason = "illegal / into-check move #%d: %s" % (i + 1, mv)
            break
        if d in items:
            items.discard(d)
        in_scenario = scenario.player(mv)
        if not in_scenario:
            if detect_capture():
                vm["failed"] = True
                reason = "left a piece hanging"
            if failure_spec and eval_spec(failure_spec, data):
                vm["failed"] = True
                reason = reason or "failure condition met"
        ok_now = eval_spec(success_spec, data) if success_spec else (len(items) == 0)
        if not vm["failed"] and ok_now:
            vm["completed"] = True
            break
        if vm["failed"]:
            break
        if not in_scenario:
            chess.set_color(pc)

    ok = vm["completed"]
    return {
        "success": ok,
        "reward": 1 if ok else 0,        # canonical lichess verdict (binary)
        "details": {
            "color": spec["color"],
            "budget": spec.get("nbMoves"),
            "apples": len(apple_keys) or None,
            "nbMoves": vm["nbMoves"],
            "failed": vm["failed"],
            "reason": reason,
        },
    }


def evaluate_by_id(task_id, moves):
    if task_id not in SPECS:
        raise KeyError("unknown task id: " + task_id)
    return evaluate(SPECS[task_id], moves)


# ---- self-test ----
if __name__ == "__main__":
    STAGE_ORDER = ["rook", "bishop", "queen", "king", "knight", "pawn",
                   "capture", "protection", "combat", "check1", "outOfCheck", "checkmate1",
                   "setup", "castling", "enpassant", "stalemate", "value", "check2", "fork"]
    print("loaded %d level specs" % len(SPECS))
    print("\nspot-checks:")
    cases = [
        ("capture-2", ["f4c7", "c7f7"], True),
        ("castling-3", ["g1f3", "e1g1"], True),
        ("castling-8", ["f1h3", "e1g1"], True),
        ("pawn-7", ["e2e4", "e4e5", "e5d6"], True),
        ("check2-1", ["c5a5", "a5a8"], True),
        ("check2-1", ["c5c1", "c1c2"], False),
        ("checkmate1-1", ["h1h2"], False),
    ]
    for cid, mv, exp in cases:
        r = evaluate_by_id(cid, mv)
        mark = "OK " if r["success"] == exp else "XX "
        print("  %s%-12s %-22s -> success=%s %s"
              % (mark, cid, str(mv), r["success"], r["details"]["reason"] or ""))
