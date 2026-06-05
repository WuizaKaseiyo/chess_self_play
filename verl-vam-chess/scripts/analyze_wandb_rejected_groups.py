#!/usr/bin/env python3
"""
Analyze `filter_groups` rejected prompt-groups from locally exported W&B files.

This script expects the run export layout produced by `scripts/export_wandb_run.py`:
  <evidence_root>/
    files/
      rejected_group_summaries/*.jsonl

Each JSONL record (1 per rejected uid group) is produced by the optional rejected-group
logging added to the trainer and includes fields like:
  - step, uid, pred_move_unique
  - all_valid, all_best_move, all_suboptimal_move
  - considered_moves_uci, pred_moves
  - penalty_reasons

Outputs:
  <evidence_root>/investigation/
    - rejected_groups_by_step.csv
    - rejected_groups_detailed_by_step.csv
    - plots/rejected_groups_ratios.png
    - plots/rejected_groups_detailed.png

Example:
  python3 scripts/analyze_wandb_rejected_groups.py \
    --evidence-root outputs/wandb/rerun_full_2emjykpq
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def _savefig(path: Path, title: str) -> None:
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=12)
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


@dataclass
class StepAgg:
    rejected_groups: int = 0
    all_valid: int = 0
    not_all_valid: int = 0

    # all_valid breakdown
    all_best_move: int = 0
    all_suboptimal_move: int = 0
    other_or_tied: int = 0

    # all_valid determinism vs ties
    all_valid_det: int = 0
    all_valid_tie: int = 0

    # detailed
    det_best: int = 0
    det_subopt: int = 0
    det_other: int = 0
    tie_best: int = 0
    tie_subopt: int = 0
    tie_other: int = 0

    # order-bias proxies (deterministic groups only)
    det_rank0: int = 0
    det_rank0_best: int = 0
    det_rank0_subopt: int = 0


def _safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _move_rank_in_list(move: str, moves: list[str]) -> int | None:
    try:
        return moves.index(move)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-root", type=Path, required=True)
    ap.add_argument("--moving-average-window", type=int, default=7)
    args = ap.parse_args()

    evidence_root: Path = args.evidence_root
    files_root = evidence_root / "files"
    rejected_dir = files_root / "rejected_group_summaries"

    if not rejected_dir.exists():
        raise FileNotFoundError(
            f"Missing rejected-group summaries dir: {rejected_dir}\n"
            "This run likely did not enable rejected-group logging."
        )

    out_root = evidence_root / "investigation"
    plots_dir = out_root / "plots"
    _ensure_dir(out_root)
    _ensure_dir(plots_dir)

    # Accumulate per-step.
    step2agg: dict[int, StepAgg] = defaultdict(StepAgg)

    # Global counters (for a quick summary).
    global_counts = Counter()
    global_penalty_reasons = Counter()

    summary_files = sorted(rejected_dir.glob("*.jsonl"))
    if not summary_files:
        raise FileNotFoundError(f"No rejected_group_summaries JSONLs found under {rejected_dir}")

    for path in summary_files:
        for rec_idx, rec in _iter_jsonl(path):
            step = _safe_int(rec.get("step"))
            if step is None:
                raise RuntimeError(f"Missing/invalid step in {path} record {rec_idx}")
            agg = step2agg[step]
            agg.rejected_groups += 1
            global_counts["rejected_groups_total"] += 1

            all_valid = bool(rec.get("all_valid"))
            if all_valid:
                agg.all_valid += 1
                global_counts["all_valid"] += 1
            else:
                agg.not_all_valid += 1
                global_counts["not_all_valid"] += 1

            best = bool(rec.get("all_best_move"))
            subopt = bool(rec.get("all_suboptimal_move"))
            if all_valid:
                if best:
                    agg.all_best_move += 1
                elif subopt:
                    agg.all_suboptimal_move += 1
                else:
                    agg.other_or_tied += 1

            # Determinism vs ties only meaningful among all_valid.
            pred_move_unique = _safe_int(rec.get("pred_move_unique"))
            det = (pred_move_unique == 1)
            if all_valid and det:
                agg.all_valid_det += 1
            elif all_valid:
                agg.all_valid_tie += 1

            # detailed breakdown for all_valid only
            if all_valid:
                if det and best:
                    agg.det_best += 1
                elif det and subopt:
                    agg.det_subopt += 1
                elif det:
                    agg.det_other += 1
                elif best:
                    agg.tie_best += 1
                elif subopt:
                    agg.tie_subopt += 1
                else:
                    agg.tie_other += 1

            # penalty reasons (group-level): only track for not_all_valid.
            if not all_valid:
                reasons = [str(x or "").strip() for x in (rec.get("penalty_reasons") or [])]
                uniq = sorted({r for r in reasons if r})
                if not uniq:
                    global_penalty_reasons["<unknown>"] += 1
                else:
                    # In this repo's reward_fn, penalty_reason is a single string; so uniq should be size 1.
                    for r in uniq:
                        global_penalty_reasons[r] += 1

            # order bias proxies (deterministic groups only; all_valid required)
            if all_valid and det:
                pred_moves = rec.get("pred_moves") or []
                considered = rec.get("considered_moves_uci") or []
                if pred_moves and considered:
                    pred = str(pred_moves[0]).strip().lower()
                    considered = [str(m).strip().lower() for m in considered]
                    rank = _move_rank_in_list(pred, considered)
                    if rank == 0:
                        agg.det_rank0 += 1
                        if best:
                            agg.det_rank0_best += 1
                        elif subopt:
                            agg.det_rank0_subopt += 1

    # Build DataFrames.
    steps = sorted(step2agg.keys())
    by_step_rows = []
    detailed_rows = []
    for step in steps:
        a = step2agg[step]
        denom_best = a.all_best_move + a.all_suboptimal_move
        best_ratio = (a.all_best_move / denom_best) if denom_best else float("nan")
        pred_unique1_ratio = (a.all_valid_det / a.all_valid) if a.all_valid else float("nan")

        by_step_rows.append(
            {
                "step": step,
                "rejected_groups": a.rejected_groups,
                "all_valid": a.all_valid,
                "all_suboptimal_move": a.all_suboptimal_move,
                "pred_move_unique_gt1": a.all_valid_tie,
                "not_all_valid": a.not_all_valid,
                "pred_move_unique_1": a.all_valid_det,
                "best_ratio": float(best_ratio),
                "pred_move_unique1_ratio": float(pred_unique1_ratio),
                "all_best_move": a.all_best_move,
                "other_or_tied": a.other_or_tied,
            }
        )

        denom_det = a.det_best + a.det_subopt
        best_ratio_det = (a.det_best / denom_det) if denom_det else float("nan")
        det_frac_among_all_valid = (a.all_valid_det / a.all_valid) if a.all_valid else float("nan")
        det_rank0_frac_det = (a.det_rank0 / a.all_valid_det) if a.all_valid_det else float("nan")
        det_rank0_frac_subopt = (a.det_rank0_subopt / a.det_subopt) if a.det_subopt else float("nan")

        detailed_rows.append(
            {
                "step": step,
                "rejected_groups": a.rejected_groups,
                "all_valid": a.all_valid,
                "not_all_valid": a.not_all_valid,
                "all_valid_det": a.all_valid_det,
                "all_valid_tie": a.all_valid_tie,
                "det_best": a.det_best,
                "det_subopt": a.det_subopt,
                "det_other": a.det_other,
                "tie_best": a.tie_best,
                "tie_subopt": a.tie_subopt,
                "tie_other": a.tie_other,
                "det_rank0": a.det_rank0,
                "det_rank0_best": a.det_rank0_best,
                "det_rank0_subopt": a.det_rank0_subopt,
                "best_ratio_det": float(best_ratio_det),
                "det_frac_among_all_valid": float(det_frac_among_all_valid),
                "det_rank0_frac_det": float(det_rank0_frac_det),
                "det_rank0_frac_subopt": float(det_rank0_frac_subopt),
            }
        )

    by_step_df = pd.DataFrame(by_step_rows).sort_values("step").reset_index(drop=True)
    detailed_df = pd.DataFrame(detailed_rows).sort_values("step").reset_index(drop=True)

    by_step_path = out_root / "rejected_groups_by_step.csv"
    detailed_path = out_root / "rejected_groups_detailed_by_step.csv"
    by_step_df.to_csv(by_step_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)

    print(f"[OK] Wrote {by_step_path}")
    print(f"[OK] Wrote {detailed_path}")

    if global_penalty_reasons:
        print("[INFO] Penalty reasons among rejected groups (group-level counts):")
        for k, v in global_penalty_reasons.most_common():
            print(f"  {k}: {v}")

    # Plots
    window = int(args.moving_average_window)

    # Ratios: best_ratio + pred_move_unique1_ratio
    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(by_step_df["step"], by_step_df["rejected_groups"], label="rejected_groups", linewidth=1.4)
    ax1.plot(by_step_df["step"], by_step_df["all_valid"], label="all_valid", linewidth=1.2)
    ax1.set_xlabel("step")
    ax1.set_ylabel("count")
    ax1.set_title("Rejected groups per update")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(by_step_df["step"], by_step_df["best_ratio"], label="best_ratio (all_valid)", linewidth=1.4)
    ax2.plot(
        by_step_df["step"],
        _rolling_mean(by_step_df["best_ratio"], window),
        label=f"best_ratio MA{window}",
        linewidth=2.0,
    )
    ax2.plot(by_step_df["step"], by_step_df["pred_move_unique1_ratio"], label="pred_move_unique==1 ratio", linewidth=1.4)
    ax2.plot(
        by_step_df["step"],
        _rolling_mean(by_step_df["pred_move_unique1_ratio"], window),
        label=f"pred_move_unique==1 MA{window}",
        linewidth=2.0,
    )
    ax2.set_xlabel("step")
    ax2.set_ylabel("ratio")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2)
    _savefig(plots_dir / "rejected_groups_ratios.png", "Rejected groups: ratios and counts")

    # Detailed breakdown
    plt.figure(figsize=(12, 7))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(detailed_df["step"], detailed_df["det_best"], label="det_best", linewidth=1.4)
    ax1.plot(detailed_df["step"], detailed_df["det_subopt"], label="det_subopt", linewidth=1.4)
    ax1.plot(detailed_df["step"], detailed_df["tie_subopt"], label="tie_subopt", linewidth=1.4)
    ax1.plot(detailed_df["step"], detailed_df["not_all_valid"], label="not_all_valid", linewidth=1.0, linestyle="--")
    ax1.set_xlabel("step")
    ax1.set_ylabel("count")
    ax1.set_title("Rejected groups by category (all_valid + determinism/tie split)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(detailed_df["step"], detailed_df["det_frac_among_all_valid"], label="det_frac among all_valid", linewidth=1.4)
    ax2.plot(detailed_df["step"], detailed_df["best_ratio_det"], label="best_ratio among det", linewidth=1.4)
    ax2.plot(detailed_df["step"], detailed_df["det_rank0_frac_subopt"], label="rank0 frac among det_subopt", linewidth=1.4)
    ax2.set_xlabel("step")
    ax2.set_ylabel("ratio")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2)
    _savefig(plots_dir / "rejected_groups_detailed.png", "Rejected groups: detailed categories + ratios")

    print(f"[OK] Plots written under {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

