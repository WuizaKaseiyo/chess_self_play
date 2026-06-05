"""Smoke test for VAM-augmented chess env (chess_game/ray_envs.py).

Tests:
  1. Backward compat: vam.enable=False → obs identical to upstream (no `Allowed moves` line)
  2. VAM enabled (random source): obs has `Allowed moves` line, len = k
  3. VAM violation: pick a legal-but-out-of-subset move → reward = penalty, board NOT advanced
  4. VAM compliance: pick an in-subset move → board advances normally
  5. VAM iterative: prior pick removed from next subset
  6. Game seeding determinism: same seed → same allowed_moves (random source)

Run:
  cd $HOME/chess_self_play/vam-agent
  python tests/test_vam_chess_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from chess_game.ray_envs import ChessWorker


GREEN = "\033[92m✓\033[0m"
RED = "\033[91m✗\033[0m"


def _assert(cond: bool, msg: str) -> None:
    print(f"  {GREEN if cond else RED} {msg}")
    if not cond:
        raise AssertionError(msg)


def test_backward_compat():
    print("\n[1] Backward compat: vam.enable=False")
    w = ChessWorker({"max_agent_plies": 10})
    obs, info = w.reset(seed_for_reset=42)
    _assert("Allowed moves" not in obs, "no 'Allowed moves' line in obs")
    _assert("Legal moves" in obs, "still has 'Legal moves' line")
    _assert("FEN:" in obs, "still has FEN line")
    # Take a legal move
    obs2, r, done, info2 = w.step("e2e4")
    _assert(abs(r - (-0.01)) < 1e-6, f"legal move reward = -0.01 (got {r})")
    _assert(not done, "game not done after 1 ply")
    _assert(info2.get("action_is_effective"), "action was effective")


def test_vam_random_enable():
    print("\n[2] VAM enable (random source)")
    K = 5
    w = ChessWorker({
        "max_agent_plies": 10,
        "vam": {"enable": True, "k": K, "subset_source": "random"},
    })
    obs, info = w.reset(seed_for_reset=42)
    _assert("Allowed moves (UCI):" in obs, "obs has 'Allowed moves' line")
    line = next(l for l in obs.splitlines() if l.startswith("Allowed moves (UCI):"))
    moves = [m.strip() for m in line.split(":", 1)[1].split(",")]
    _assert(len(moves) == K, f"allowed_moves len = {K} (got {len(moves)})")
    # All allowed moves must be among legal moves
    legal_line = next(l for l in obs.splitlines() if l.startswith("Legal moves (UCI):"))
    legal = set(m.strip().rstrip(",") for m in legal_line.split(":", 1)[1].split(",") if m.strip() and m.strip() != "...")
    for m in moves:
        _assert(m in legal, f"allowed move {m!r} ∈ legal moves")


def test_vam_violation_does_not_advance_board():
    print("\n[3] VAM violation: out-of-subset move → -1, board NOT advanced")
    w = ChessWorker({
        "max_agent_plies": 10,
        "vam": {"enable": True, "k": 3, "subset_source": "random", "penalty": -1.0},
    })
    obs, _ = w.reset(seed_for_reset=42)
    line = next(l for l in obs.splitlines() if l.startswith("Allowed moves (UCI):"))
    allowed = [m.strip() for m in line.split(":", 1)[1].split(",")]
    legal_line = next(l for l in obs.splitlines() if l.startswith("Legal moves (UCI):"))
    legal = [m.strip().rstrip(",") for m in legal_line.split(":", 1)[1].split(",") if m.strip() and m.strip() != "..."]
    # Find a legal move NOT in allowed
    out_of_subset = next(m for m in legal if m not in allowed)
    print(f"     allowed = {allowed}")
    print(f"     trying out-of-subset legal move: {out_of_subset!r}")
    fen_before = w.board.fen()
    obs2, r, done, info = w.step(out_of_subset)
    fen_after = w.board.fen()
    _assert(abs(r - (-1.0)) < 1e-6, f"out-of-subset reward = -1.0 (got {r})")
    _assert(not done, "episode not terminated by VAM violation")
    _assert(info.get("vam_violation") is True, "info.vam_violation = True")
    _assert(fen_before == fen_after, "board NOT advanced on VAM violation")


def test_vam_compliance_advances():
    print("\n[4] VAM compliance: in-subset move → -0.01, board advances")
    w = ChessWorker({
        "max_agent_plies": 10,
        "vam": {"enable": True, "k": 5, "subset_source": "random"},
    })
    obs, _ = w.reset(seed_for_reset=42)
    line = next(l for l in obs.splitlines() if l.startswith("Allowed moves (UCI):"))
    allowed = [m.strip() for m in line.split(":", 1)[1].split(",")]
    print(f"     picking allowed move: {allowed[0]!r}")
    fen_before = w.board.fen()
    obs2, r, done, info = w.step(allowed[0])
    fen_after = w.board.fen()
    _assert(abs(r - (-0.01)) < 1e-6, f"in-subset move reward = -0.01 (got {r})")
    _assert(fen_before != fen_after, "board ADVANCED after compliant move")
    _assert(info.get("action_is_effective"), "action_is_effective=True")


def test_vam_iterative_removes_prior():
    print("\n[5] Iterative VAM: prior in-subset choice removed from next subset")
    K = 5
    w = ChessWorker({
        "max_agent_plies": 20,
        "vam": {"enable": True, "k": K, "subset_source": "random", "iterative": True},
    })
    obs0, _ = w.reset(seed_for_reset=42)
    line = next(l for l in obs0.splitlines() if l.startswith("Allowed moves (UCI):"))
    allowed0 = [m.strip() for m in line.split(":", 1)[1].split(",")]
    pick = allowed0[0]
    print(f"     ply 0 allowed = {allowed0}")
    print(f"     picked {pick!r}")
    obs1, r1, done, _ = w.step(pick)
    line1 = next(l for l in obs1.splitlines() if l.startswith("Allowed moves (UCI):"))
    allowed1 = [m.strip() for m in line1.split(":", 1)[1].split(",")]
    print(f"     ply 1 allowed = {allowed1}")
    # prior pick should NOT be in next allowed (board changed, but iterative state remembers)
    _assert(pick not in allowed1, f"prior choice {pick!r} removed from subsequent allowed_moves")


def test_seed_determinism():
    print("\n[6] Seed determinism for random source")
    cfg = {"max_agent_plies": 10, "vam": {"enable": True, "k": 4, "subset_source": "random"}}
    w1 = ChessWorker(cfg)
    w2 = ChessWorker(cfg)
    obs1, _ = w1.reset(seed_for_reset=99)
    obs2, _ = w2.reset(seed_for_reset=99)
    line1 = next(l for l in obs1.splitlines() if l.startswith("Allowed moves (UCI):"))
    line2 = next(l for l in obs2.splitlines() if l.startswith("Allowed moves (UCI):"))
    _assert(line1 == line2, "same seed → same allowed_moves")


def main():
    tests = [
        test_backward_compat,
        test_vam_random_enable,
        test_vam_violation_does_not_advance_board,
        test_vam_compliance_advances,
        test_vam_iterative_removes_prior,
        test_seed_determinism,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  {RED} FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"  {RED} ERROR: {type(e).__name__}: {e}")
    print()
    if failed == 0:
        print(f"{GREEN} all {len(tests)} tests passed")
    else:
        print(f"{RED} {failed}/{len(tests)} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
