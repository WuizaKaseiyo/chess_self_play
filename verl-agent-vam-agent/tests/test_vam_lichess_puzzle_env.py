"""Smoke test for LichessPuzzleWorker (chess-rl-C224 puzzle data → verl-agent).

Tests:
  1. Bare puzzle (vam.enable=False): obs has FEN + Legal moves, no Allowed
  2. Correct UCI → reward = μ from precomputed table, done=True
  3. Illegal UCI → reward = -1.0
  4. Bad format → reward = -1.0
  5. VAM mu_topk: Allowed = top-k by μ (deterministic)
  6. VAM violation: out-of-subset legal move → penalty
  7. Iterative VAM: prior pick removed from next reset on same puzzle id

Run:
  cd $HOME/chess_self_play/vam-agent
  python tests/test_vam_lichess_puzzle_env.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from chess_game.lichess_puzzle_envs import LichessPuzzleWorker

import os
# A real chess-rl-C224 parquet to use (any of stage1-4 or the legacy baseline).
# Override via TEST_PARQUET env var.
DEFAULT_PARQUET = os.environ.get(
    "TEST_PARQUET",
    os.path.expanduser("~/chess/chess-rl-C224/data/chess_puzzles_chessr1_aligned_sharded_baseline/train_0.parquet"),
)

GREEN = "\033[92m✓\033[0m"
RED = "\033[91m✗\033[0m"


def _assert(cond: bool, msg: str) -> None:
    print(f"  {GREEN if cond else RED} {msg}")
    if not cond:
        raise AssertionError(msg)


def _get_row(worker, task_idx=0):
    """Reset and return (obs, info, row, mu_table)."""
    obs, info = worker.reset(task_idx=task_idx)
    row = worker._cur_row
    rm = dict(row["reward_model"])
    mu = json.loads(rm.get("move_expected_scores_json") or "{}")
    return obs, info, row, mu


def test_bare_puzzle():
    print("\n[1] Bare puzzle (vam.enable=False)")
    w = LichessPuzzleWorker({"parquets": [DEFAULT_PARQUET]})
    obs, info = w.reset(task_idx=0)
    _assert("FEN:" in obs, "obs has FEN")
    _assert("Legal moves (UCI):" in obs, "obs has Legal moves")
    _assert("Allowed moves" not in obs, "obs has NO Allowed moves (vam disabled)")
    _assert("ground_truth" in info, "info has ground_truth")


def test_correct_move_reward():
    print("\n[2] Correct UCI → reward = μ from precomputed table, done=True")
    w = LichessPuzzleWorker({"parquets": [DEFAULT_PARQUET]})
    obs, info, row, mu = _get_row(w, task_idx=0)
    gt = info["ground_truth"]
    expected_mu = mu.get(gt, 0.0)
    obs2, r, done, info2 = w.step(gt)
    _assert(done, "episode done after 1 step")
    _assert(abs(r - expected_mu) < 1e-6, f"reward = μ[{gt}] = {expected_mu} (got {r})")
    _assert(info2.get("won") == True, "info.won = True (pred matches ground_truth)")


def test_illegal_move():
    print("\n[3] Illegal UCI → reward = -1.0")
    w = LichessPuzzleWorker({"parquets": [DEFAULT_PARQUET]})
    obs, info, row, mu = _get_row(w, task_idx=0)
    legal = set([str(m).lower() for m in row["reward_model"]["legal_moves_uci"]])
    # Pick a UCI that's syntactically valid but illegal in this position
    candidate = "a1h8"
    if candidate in legal:
        # extremely unlikely but fall back
        candidate = "b1h8"
    obs2, r, done, info2 = w.step(candidate)
    _assert(abs(r - (-1.0)) < 1e-6, f"illegal reward = -1.0 (got {r})")
    _assert(done, "episode terminated on illegal")
    _assert(info2.get("reason") == "illegal_move", "reason = illegal_move")


def test_bad_format():
    print("\n[4] Bad format → reward = -1.0")
    w = LichessPuzzleWorker({"parquets": [DEFAULT_PARQUET]})
    w.reset(task_idx=0)
    obs2, r, done, info2 = w.step("not_a_uci")
    _assert(abs(r - (-1.0)) < 1e-6, f"bad format reward = -1.0 (got {r})")
    _assert(done, "episode terminated on bad format")


def test_vam_mu_topk():
    print("\n[5] VAM mu_topk: Allowed = top-k by μ")
    K = 4
    w = LichessPuzzleWorker({
        "parquets": [DEFAULT_PARQUET],
        "vam": {"enable": True, "k": K, "subset_source": "mu_topk"},
    })
    obs, info, row, mu = _get_row(w, task_idx=0)
    _assert("Allowed moves (UCI):" in obs, "obs has Allowed moves line")
    line = next(l for l in obs.splitlines() if l.startswith("Allowed moves (UCI):"))
    allowed = [m.strip() for m in line.split(":", 1)[1].split(",")]
    _assert(len(allowed) == K, f"allowed_moves len = {K} (got {len(allowed)})")
    # The top-1 of allowed should equal the global μ-argmax
    mu_sorted = sorted(mu.items(), key=lambda x: (-x[1], x[0]))
    expected_top1 = mu_sorted[0][0]
    _assert(allowed[0] == expected_top1, f"top-1 allowed = μ-argmax {expected_top1!r} (got {allowed[0]!r})")
    # ground_truth (which is μ-argmax in chess-rl-C224) MUST be in allowed
    _assert(info["ground_truth"] in allowed, "ground_truth ∈ allowed (μ-best always present)")


def test_vam_violation():
    print("\n[6] VAM violation: out-of-subset legal move → penalty")
    w = LichessPuzzleWorker({
        "parquets": [DEFAULT_PARQUET],
        "vam": {"enable": True, "k": 3, "subset_source": "mu_topk", "penalty": -1.0},
    })
    obs, info, row, mu = _get_row(w, task_idx=0)
    line = next(l for l in obs.splitlines() if l.startswith("Allowed moves (UCI):"))
    allowed = [m.strip() for m in line.split(":", 1)[1].split(",")]
    legal = [str(m).lower() for m in row["reward_model"]["legal_moves_uci"]]
    # Find a legal move NOT in allowed
    out_of_subset = next((m for m in legal if m not in allowed), None)
    _assert(out_of_subset is not None, "found a legal-but-out-of-subset move to test")
    obs2, r, done, info2 = w.step(out_of_subset)
    _assert(abs(r - (-1.0)) < 1e-6, f"vam violation reward = -1.0 (got {r})")
    _assert(info2.get("vam_violation") == True, "info.vam_violation = True")


def test_iterative_vam():
    print("\n[7] Iterative VAM: prior pick removed from next reset on same puzzle")
    K = 4
    w = LichessPuzzleWorker({
        "parquets": [DEFAULT_PARQUET],
        "vam": {"enable": True, "k": K, "subset_source": "mu_topk", "iterative": True},
    })
    # First reset + step
    obs0, info0 = w.reset(task_idx=0)
    line0 = next(l for l in obs0.splitlines() if l.startswith("Allowed moves (UCI):"))
    allowed0 = [m.strip() for m in line0.split(":", 1)[1].split(",")]
    pick = allowed0[0]
    w.step(pick)
    # Reset same puzzle again — iterative state should remove pick
    obs1, info1 = w.reset(task_idx=0)
    line1 = next(l for l in obs1.splitlines() if l.startswith("Allowed moves (UCI):"))
    allowed1 = [m.strip() for m in line1.split(":", 1)[1].split(",")]
    _assert(pick not in allowed1, f"prior pick {pick!r} removed from re-reset allowed")
    _assert(info0["task_id"] == info1["task_id"], "same task id")


def main():
    if not Path(DEFAULT_PARQUET).exists():
        print(f"{RED} test parquet not found: {DEFAULT_PARQUET}")
        print("    set DEFAULT_PARQUET to a chess-rl-C224 train_0.parquet path and rerun")
        sys.exit(2)

    tests = [
        test_bare_puzzle,
        test_correct_move_reward,
        test_illegal_move,
        test_bad_format,
        test_vam_mu_topk,
        test_vam_violation,
        test_iterative_vam,
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
