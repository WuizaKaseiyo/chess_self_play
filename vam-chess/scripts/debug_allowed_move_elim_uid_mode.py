#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _load_row_ids(parquet_path: Path, limit_rows: int) -> list[int | None]:
    table = pq.read_table(parquet_path, columns=["extra_info"])
    rows = table.slice(0, limit_rows).to_pylist()
    out: list[int | None] = []
    for row in rows:
        extra = row.get("extra_info") if isinstance(row, dict) else None
        if isinstance(extra, dict) and "index" in extra:
            try:
                out.append(int(extra["index"]))
            except Exception:
                out.append(None)
        else:
            out.append(None)
    return out


def _simulate_allowed_move_elim_batch(*, num_prompts: int, r_max: int, rollout_n: int, uid_mode: str):
    uid_mode = str(uid_mode).strip().lower()
    if uid_mode not in {"per_round", "per_prompt"}:
        raise ValueError(f"uid_mode must be per_round or per_prompt (got {uid_mode!r})")
    if num_prompts <= 0:
        raise ValueError("num_prompts must be > 0")
    if r_max <= 0:
        raise ValueError("r_max must be > 0")
    if rollout_n <= 0:
        raise ValueError("rollout_n must be > 0")

    base_uids = [f"uid_prompt{p}" for p in range(num_prompts)] if uid_mode == "per_prompt" else []

    prompt_idx: list[int] = []
    round_idx: list[int] = []
    uids: list[str] = []

    for r in range(1, r_max + 1):
        for p in range(num_prompts):
            uid = f"uid_prompt{p}_round{r}" if uid_mode == "per_round" else base_uids[p]
            for _ in range(rollout_n):
                prompt_idx.append(p)
                round_idx.append(r)
                uids.append(uid)

    prompt_idx_arr = np.asarray(prompt_idx, dtype=np.int64)
    round_idx_arr = np.asarray(round_idx, dtype=np.int64)
    uid_arr = np.asarray(uids, dtype=object)
    return prompt_idx_arr, round_idx_arr, uid_arr


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Local repro for iterative allowed-move elimination GRPO grouping.\n\n"
            "This script simulates an allowed_move_elim-expanded batch (R rounds × rollout_n samples/round), "
            "then reports how many unique GRPO uid-groups each original prompt produces.\n\n"
            "It is intentionally lightweight: it does not run model inference."
        )
    )
    p.add_argument(
        "--parquet",
        default="data/chess_puzzles_select_v4/test.parquet",
        help="Parquet used only to read a few row ids for sanity/context (default: v4 test split).",
    )
    p.add_argument("--limit_rows", type=int, default=3, help="How many base prompts to simulate (default: 3).")
    p.add_argument("--r_max", type=int, default=3, help="How many elimination rounds to simulate (default: 3).")
    p.add_argument("--rollout_n", type=int, default=4, help="GRPO rollout.n (samples per group per round).")
    p.add_argument(
        "--uid_mode",
        choices=["per_round", "per_prompt"],
        default="per_round",
        help="Grouping mode to simulate (matches algorithm.allowed_move_elim.uid_mode).",
    )
    args = p.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet: {parquet_path}")

    row_ids = _load_row_ids(parquet_path, int(args.limit_rows))
    prompt_idx_arr, round_idx_arr, uid_arr = _simulate_allowed_move_elim_batch(
        num_prompts=int(args.limit_rows),
        r_max=int(args.r_max),
        rollout_n=int(args.rollout_n),
        uid_mode=str(args.uid_mode),
    )

    uid_counts = RayPPOTrainer._allowed_move_elim_count_unique_uids_by_prompt(
        prompt_idx_arr=prompt_idx_arr,
        uid_arr=uid_arr,
    )
    round_counts = RayPPOTrainer._allowed_move_elim_count_unique_rounds_by_prompt(
        prompt_idx_arr=prompt_idx_arr,
        round_arr=round_idx_arr,
    )

    denom_counts = uid_counts if args.uid_mode == "per_round" else round_counts

    total_weight_by_prompt: dict[int, float] = defaultdict(float)
    for pidx in prompt_idx_arr:
        p_int = int(pidx)
        total_weight_by_prompt[p_int] += 1.0 / float(denom_counts.get(p_int, 1))

    payload = {
        "parquet": str(parquet_path),
        "row_ids_from_parquet": row_ids,
        "num_prompts": int(args.limit_rows),
        "r_max": int(args.r_max),
        "rollout_n": int(args.rollout_n),
        "uid_mode": str(args.uid_mode),
        "unique_uid_groups_per_prompt": {str(k): int(v) for k, v in sorted(uid_counts.items())},
        "unique_rounds_per_prompt": {str(k): int(v) for k, v in sorted(round_counts.items())},
        "loss_weight_denominator_per_prompt": {str(k): int(v) for k, v in sorted(denom_counts.items())},
        "total_loss_weight_per_prompt": {str(k): float(v) for k, v in sorted(total_weight_by_prompt.items())},
        "expected_total_loss_weight_per_prompt": float(args.rollout_n),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    expected_uid_groups = int(args.r_max) if args.uid_mode == "per_round" else 1
    ok = all(int(uid_counts.get(i, 0)) == expected_uid_groups for i in range(int(args.limit_rows)))
    ok = ok and all(abs(float(total_weight_by_prompt.get(i, 0.0)) - float(args.rollout_n)) < 1e-6 for i in range(int(args.limit_rows)))
    if not ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
