#!/usr/bin/env python3
"""
Compare multiple W&B runs from locally downloaded evidence.

This script is intentionally lightweight and only depends on:
  - <evidence_root>/<run_id>/history.parquet

It produces overlaid plots and a merged CSV under a gitignored `outputs/` dir.

Example
-------
conda run -n verl python scripts/compare_wandb_run_histories.py \\
  --evidence_root analysis/wandb_evidence \\
  --out_dir outputs/wandb_compare/am4_reward_fn_2026-01-21 \\
  --runs n1ihbyab:expected_score_wdl_vs_best 6g3tapxg:winrate_vs_best q58ea8sy:rank_among_moves
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_runs(items: list[str]) -> list[RunSpec]:
    out: list[RunSpec] = []
    for item in items:
        if ":" in item:
            run_id, label = item.split(":", 1)
        else:
            run_id, label = item, item
        out.append(RunSpec(run_id=run_id.strip(), label=label.strip()))
    return out


def _maybe_series(df: pd.DataFrame, y: str) -> pd.DataFrame:
    if "_step" not in df.columns or y not in df.columns:
        return pd.DataFrame(columns=["_step", y])
    sub = df[["_step", y]].dropna()
    if sub.empty:
        return pd.DataFrame(columns=["_step", y])
    return sub.sort_values("_step").reset_index(drop=True)


def _plot_overlay(
    *,
    histories: dict[str, pd.DataFrame],
    runs: list[RunSpec],
    y: str,
    out_path: Path,
    title: str,
    ylabel: Optional[str] = None,
    marker: Optional[str] = None,
) -> None:
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    for rs in runs:
        df = histories[rs.run_id]
        sub = _maybe_series(df, y)
        if sub.empty:
            continue
        ax.plot(sub["_step"], sub[y], label=rs.label, linewidth=1.6, marker=marker)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel or y)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _merge_histories_on_step(histories: dict[str, pd.DataFrame], runs: list[RunSpec], keys: list[str]) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for rs in runs:
        df = histories[rs.run_id].copy()
        keep = ["_step"] + [k for k in keys if k in df.columns]
        df = df[keep].copy()
        # Disambiguate by run label.
        rename = {k: f"{rs.label}/{k}" for k in keep if k != "_step"}
        df = df.rename(columns=rename)
        merged = df if merged is None else pd.merge(merged, df, on="_step", how="outer")
    if merged is None:
        return pd.DataFrame()
    return merged.sort_values("_step").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence_root", default="analysis/wandb_evidence")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="Run specs formatted as RUN_ID:LABEL")
    args = ap.parse_args()

    evidence_root = Path(args.evidence_root)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    _ensure_dir(out_dir)
    _ensure_dir(plots_dir)

    runs = _parse_runs(list(args.runs))

    histories: dict[str, pd.DataFrame] = {}
    for rs in runs:
        hist_path = evidence_root / rs.run_id / "history.parquet"
        if not hist_path.exists():
            raise FileNotFoundError(f"Missing history: {hist_path}")
        df = pd.read_parquet(hist_path)
        if "_step" not in df.columns:
            raise RuntimeError(f"history.parquet missing _step for run={rs.run_id}")
        histories[rs.run_id] = df.sort_values("_step").reset_index(drop=True)

    # Core metrics we want on one page for reward-function debugging.
    train_metrics = [
        "critic/score/mean",
        "actor/entropy",
        "actor/ppo_kl",
        "actor/pg_loss",
        "grpo/effective_batch_frac",
        "grpo/group_count",
        "selection_sampler/success_rate",
        "selection_sampler/r_max",
    ]
    val_metrics = [
        "val-core/local/chess_puzzles/acc/mean@1",
        "val-aux/local/chess_puzzles/coverage/mean@1",
        "val-aux/local/chess_puzzles/format_reward/mean@1",
        "val-aux/local/chess_puzzles/score/mean@1",
    ]

    # Plots (train metrics as lines; val as sparse markers).
    for y in train_metrics:
        _plot_overlay(
            histories=histories,
            runs=runs,
            y=y,
            out_path=plots_dir / f"{y.replace('/', '__')}.png",
            title=f"{y} (train)",
        )

    for y in val_metrics:
        _plot_overlay(
            histories=histories,
            runs=runs,
            y=y,
            out_path=plots_dir / f"{y.replace('/', '__')}.png",
            title=f"{y} (val)",
            marker="o",
        )

    # Merged CSV for quick grep / spreadsheet diffs.
    merged = _merge_histories_on_step(histories, runs, keys=train_metrics + val_metrics)
    merged.to_csv(out_dir / "merged_history.csv.gz", index=False, compression="gzip")

    # Small summary table.
    summary_rows: list[dict[str, Any]] = []
    for rs in runs:
        df = histories[rs.run_id]
        row: dict[str, Any] = {"run_id": rs.run_id, "label": rs.label, "max_step": int(df["_step"].max())}
        for k in train_metrics + val_metrics:
            if k not in df.columns:
                continue
            s = df[["_step", k]].dropna()
            if s.empty:
                continue
            row[f"{k}__first"] = float(s.iloc[0][k])
            row[f"{k}__last"] = float(s.iloc[-1][k])
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    print(f"[OK] wrote outputs under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
