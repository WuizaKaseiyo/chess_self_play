#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _best_move_by_mu(mu_map: dict[str, float], moves: list[str]) -> str:
    best_move = ""
    best_mu = -float("inf")
    for mv in moves:
        key = str(mv).strip().lower()
        mu = float(mu_map.get(key, -float("inf")))
        if (mu > best_mu) or (mu == best_mu and (not best_move or key < best_move)):
            best_move = key
            best_mu = mu
    return best_move


def _load_mu_map(rm: dict[str, Any]) -> tuple[str, dict[str, float]]:
    raw = rm.get("move_expected_scores_json") or rm.get("move_values_json") or ""
    src = "move_expected_scores_json" if rm.get("move_expected_scores_json") else "move_values_json"
    try:
        obj = json.loads(raw)
    except Exception as e:
        raise ValueError(f"Failed to parse {src}: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"{src} is not a JSON object")
    out: dict[str, float] = {}
    for k, v in obj.items():
        key = str(k).strip().lower()
        if not key:
            continue
        out[key] = float(v)
    return src, out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Inspect a dumped online play vs engine opponent (misnamed 'self-play') training batch JSONL for invariants."
    )
    p.add_argument("--jsonl", required=True, help="Path to the dumped JSONL (one row per line).")
    p.add_argument("--num", type=int, default=5, help="How many rows to print (default: 5).")
    args = p.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        print(f"Missing file: {path}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        print("No rows found.", file=sys.stderr)
        return 2

    bad = 0
    for i, row in enumerate(rows):
        rm = row.get("reward_model") or {}
        if not isinstance(rm, dict):
            print(f"[row {i}] reward_model missing/invalid: {type(rm)}", file=sys.stderr)
            bad += 1
            continue

        fen = str(rm.get("fen") or "").strip()
        legal = rm.get("legal_moves_uci")
        considered = rm.get("considered_moves_uci")
        gt = str(rm.get("ground_truth") or "").strip().lower()

        if not fen:
            print(f"[row {i}] empty fen", file=sys.stderr)
            bad += 1
        if not isinstance(legal, list) or not legal:
            print(f"[row {i}] legal_moves_uci missing/empty", file=sys.stderr)
            bad += 1
            continue
        legal = [str(x).strip().lower() for x in legal if str(x).strip()]
        if not legal:
            print(f"[row {i}] legal_moves_uci empty after normalization", file=sys.stderr)
            bad += 1
            continue

        if not isinstance(considered, list):
            print(f"[row {i}] considered_moves_uci missing/invalid", file=sys.stderr)
            bad += 1
            continue
        considered = [str(x).strip().lower() for x in considered if str(x).strip()]

        if considered != legal:
            print(f"[row {i}] considered_moves_uci != legal_moves_uci (expected full-legal init)", file=sys.stderr)
            bad += 1

        src, mu_map = _load_mu_map(rm)
        missing_mu = [mv for mv in legal if mv not in mu_map]
        if missing_mu:
            print(f"[row {i}] {src} missing {len(missing_mu)} legal moves (e.g., {missing_mu[:5]})", file=sys.stderr)
            bad += 1

        mu_best = _best_move_by_mu(mu_map, legal)
        if not gt:
            print(f"[row {i}] empty ground_truth", file=sys.stderr)
            bad += 1
        elif gt != mu_best:
            print(f"[row {i}] ground_truth != mu-best ({gt} != {mu_best})", file=sys.stderr)
            bad += 1

        # Prompt checks (optional but helpful).
        prompt = row.get("prompt") or []
        prompt_text = ""
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            prompt_text = str(prompt[0].get("content") or "")
        if "allowed_moves" not in prompt_text.lower():
            print(f"[row {i}] prompt missing 'allowed_moves' keyword", file=sys.stderr)
            bad += 1
        if "Allowed moves (UCI):" not in prompt_text:
            print(f"[row {i}] prompt missing 'Allowed moves (UCI):' line", file=sys.stderr)
            bad += 1

        if i < int(args.num):
            extra = row.get("extra_info") or {}
            row_id = extra.get("index", None) if isinstance(extra, dict) else None
            print(
                json.dumps(
                    {
                        "i": i,
                        "row_id": row_id,
                        "fen": fen,
                        "n_legal": len(legal),
                        "ground_truth": gt,
                        "mu_source": src,
                        "mu_best": mu_best,
                    },
                    ensure_ascii=False,
                )
            )

    if bad:
        print(f"FAILED: {bad} invariant violations across {len(rows)} rows", file=sys.stderr)
        return 1

    print(f"OK: {len(rows)} rows passed invariants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
