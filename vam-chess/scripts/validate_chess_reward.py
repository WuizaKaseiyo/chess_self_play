#!/usr/bin/env python3
"""
Quick sanity check for the selection-framed chess reward (`recipe/chess/reward_fn.py`).

Usage:
  python scripts/validate_chess_reward.py \
    --parquet data/chess_puzzles/test.parquet \
    --limit 100

Checks:
- The reward uses the strict `<reason>...</reason><uci_move>...</uci_move>` parsing gate:
  - missing/multiple/malformed tags or extra text outside the two top-level tags
    => format_error => score=-1
  - invalid UCI payload => bad_move => score=-1
- For selection prompts (prompt contains `allowed_moves`), the predicted move must be within the candidate set
  (`considered_moves_uci` when present; else legal moves).
- For baseline prompts (no `allowed_moves`), out-of-subset is NOT penalized, but illegal moves still are.
- Target move is μ-based within the candidate set (expected scores preferred; fallback move_values):
  - correct target => score=1
  - wrong in-subset move => score=0
  - out-of-subset move => score=-1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import datasets

# Ensure local imports resolve when the script is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipe.chess.reward_fn import compute_score


def _best_move_by_mu(mu_map: dict[str, float], moves: list[str]) -> str:
    best_move = ""
    best_mu = -1e9
    for mv in moves:
        key = str(mv).strip().lower()
        mu = float(mu_map.get(key, float("-inf")))
        if (mu > best_mu) or (mu == best_mu and (not best_move or key < best_move)):
            best_move = key
            best_mu = mu
    if not best_move:
        raise ValueError("Empty move list when selecting μ-target.")
    return best_move


def _load_mu_map(rm: dict) -> dict[str, float]:
    mu_json = rm.get("move_expected_scores_json")
    if isinstance(mu_json, str) and mu_json.strip():
        raw = json.loads(mu_json)
    else:
        raw = mu_json
    if isinstance(raw, dict) and raw:
        return {str(k).strip().lower(): float(v) for k, v in raw.items() if str(k).strip()}

    mv_json = rm.get("move_values_json")
    if isinstance(mv_json, str) and mv_json.strip():
        raw = json.loads(mv_json)
    else:
        raw = mv_json
    if isinstance(raw, dict) and raw:
        return {str(k).strip().lower(): float(v) for k, v in raw.items() if str(k).strip()}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    ds = datasets.load_dataset("parquet", data_files=args.parquet)["train"]
    n = min(args.limit, len(ds))

    ok_target = ok_other_legal = ok_format = ok_invalid = ok_oos = 0
    ok_oos_baseline = 0
    checked_other_legal = checked_format = checked_invalid = checked_oos = 0
    checked_oos_baseline = 0

    for i in range(n):
        row = ds[i]
        rm = row.get("reward_model", {})
        mu_map = _load_mu_map(rm)
        if not mu_map:
            continue
        legal = rm.get("legal_moves_uci") or []
        if isinstance(legal, str):
            legal = [legal]
        legal = [str(m).lower() for m in legal]
        if not legal:
            continue

        target = _best_move_by_mu(mu_map, legal)
        resp = f"<reason>t</reason><uci_move>{target}</uci_move>"
        sel_extra = {"prompt_text": "allowed_moves"}  # selection prompt marker
        base_extra = {"prompt_text": "legal moves only"}  # baseline prompt (no allowed_moves)
        r = compute_score(
            data_source=rm,
            solution_str=resp,
            ground_truth=rm.get("ground_truth") or "",
            extra_info=sel_extra,
        )
        sc = r["score"] if isinstance(r, dict) else float(r)
        if sc == 1.0:
            ok_target += 1

        # try a different legal move
        alt = next((m for m in legal if m != target), None)
        if alt is not None:
            checked_other_legal += 1
            r2 = compute_score(
                data_source=rm,
                solution_str=f"<reason>t</reason><uci_move>{alt}</uci_move>",
                ground_truth=rm.get("ground_truth") or "",
                extra_info=sel_extra,
            )
            sc2 = r2["score"] if isinstance(r2, dict) else r2
            if sc2 == 0.0:
                ok_other_legal += 1

        # format gating check (score should be -1)
        checked_format += 1
        r_missing = compute_score(
            data_source=rm,
            solution_str="no uci tag",
            ground_truth=rm.get("ground_truth") or "",
            extra_info=sel_extra,
        )
        sc_missing = r_missing["score"] if isinstance(r_missing, dict) else r_missing
        r_wrong_tag = compute_score(
            data_source=rm,
            solution_str=f"<reason>t</reason><answer>{target}</answer>",
            ground_truth=rm.get("ground_truth") or "",
            extra_info=sel_extra,
        )
        sc_wrong_tag = r_wrong_tag["score"] if isinstance(r_wrong_tag, dict) else r_wrong_tag
        # Trailing text is not allowed under the strict two-tag contract.
        r_trailing = compute_score(
            data_source=rm,
            solution_str=f"<reason>t</reason><uci_move>{target}</uci_move> EXTRA",
            ground_truth=rm.get("ground_truth") or "",
            extra_info=sel_extra,
        )
        sc_trailing = r_trailing["score"] if isinstance(r_trailing, dict) else r_trailing
        r_multi_move = compute_score(
            data_source=rm,
            solution_str=f"<reason>a</reason><uci_move>{target}</uci_move><uci_move>{target}</uci_move>",
            ground_truth=rm.get("ground_truth") or "",
            extra_info=sel_extra,
        )
        sc_multi_move = r_multi_move["score"] if isinstance(r_multi_move, dict) else r_multi_move

        if sc_missing == -1.0 and sc_wrong_tag == -1.0 and sc_trailing == -1.0 and sc_multi_move == -1.0:
            ok_format += 1

        checked_invalid += 1
        r_invalid_move = compute_score(
            data_source=rm,
            solution_str="<reason>t</reason><uci_move>zzzz</uci_move>",
            ground_truth=rm.get("ground_truth") or "",
            extra_info=sel_extra,
        )
        sc_invalid = r_invalid_move["score"] if isinstance(r_invalid_move, dict) else r_invalid_move
        if sc_invalid == -1.0:
            ok_invalid += 1

        # Out-of-subset should be -1 for selection prompts when considered_moves_uci excludes the move.
        if len(legal) >= 2:
            checked_oos += 1
            rm_sub = dict(rm)
            rm_sub["considered_moves_uci"] = [legal[0]]
            oos_resp = f"<reason>t</reason><uci_move>{legal[1]}</uci_move>"
            r_oos = compute_score(
                data_source=rm_sub,
                solution_str=oos_resp,
                ground_truth=rm.get("ground_truth") or "",
                extra_info=sel_extra,
            )
            sc_oos = r_oos["score"] if isinstance(r_oos, dict) else r_oos
            if sc_oos == -1.0 and r_oos.get("penalty_reason") == "out_of_subset":
                ok_oos += 1

            # Baseline prompts should NOT apply out-of-subset for the same row/move.
            checked_oos_baseline += 1
            r_oos_base = compute_score(
                data_source=rm_sub,
                solution_str=oos_resp,
                ground_truth=rm.get("ground_truth") or "",
                extra_info=base_extra,
            )
            sc_oos_base = r_oos_base["score"] if isinstance(r_oos_base, dict) else r_oos_base
            if sc_oos_base != -1.0 and r_oos_base.get("penalty_reason") != "out_of_subset":
                ok_oos_baseline += 1

    print(f"Target-move rows (score=1) passed: {ok_target}/{n}")
    print(f"Other-legal rows (score=0) passed: {ok_other_legal}/{checked_other_legal}")
    print(f"Format gating (-1) passed: {ok_format}/{checked_format}")
    print(f"Unparsable move (-1) passed: {ok_invalid}/{checked_invalid}")
    if checked_oos:
        print(f"Out-of-subset (-1) passed: {ok_oos}/{checked_oos}")
    if checked_oos_baseline:
        print(f"Baseline no out-of-subset (-1) passed: {ok_oos_baseline}/{checked_oos_baseline}")


if __name__ == "__main__":
    main()
