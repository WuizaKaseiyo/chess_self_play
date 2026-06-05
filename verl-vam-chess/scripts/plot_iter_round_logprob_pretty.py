#!/usr/bin/env python3
"""Create a publication-quality round-vs-logprob plot from diagnostic artifacts.

Inputs:
  - round_summary.csv (mean/median per round)
  - round_records.jsonl (per-prompt round records for dispersion/CI)

Output:
  - round_vs_logprob_pretty.png (or a user-provided output path)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _load_logprobs_by_round(round_records_path: Path) -> dict[int, list[float]]:
    by_round: dict[int, list[float]] = defaultdict(list)
    with round_records_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {round_records_path} at line {line_num}") from exc

            round_idx = int(rec.get("round", 0))
            if round_idx <= 0:
                continue
            lp = _safe_float(rec.get("reference_logprob_sum_round0_prompt"))
            if math.isfinite(lp):
                by_round[round_idx].append(lp)
    return dict(by_round)


def _compute_round_stats(by_round: dict[int, list[float]], ci_z: float) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for round_idx in sorted(by_round):
        vals = np.asarray(by_round[round_idx], dtype=np.float64)
        n = int(vals.size)
        if n == 0:
            continue

        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))
        q25, q75 = np.percentile(vals, [25.0, 75.0])
        std_val = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        sem_val = std_val / math.sqrt(n) if n > 1 and math.isfinite(std_val) else float("nan")
        ci_lo = mean_val - ci_z * sem_val if math.isfinite(sem_val) else float("nan")
        ci_hi = mean_val + ci_z * sem_val if math.isfinite(sem_val) else float("nan")

        rows.append(
            {
                "round": int(round_idx),
                "n_finite": int(n),
                "mean_from_records": mean_val,
                "median_from_records": median_val,
                "q25": float(q25),
                "q75": float(q75),
                "std": std_val,
                "ci95_lo": float(ci_lo),
                "ci95_hi": float(ci_hi),
            }
        )
    return pd.DataFrame(rows)


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    run_dir = Path(args.run_dir).resolve()
    round_summary = Path(args.round_summary).resolve() if args.round_summary else run_dir / "round_summary.csv"
    round_records = Path(args.round_records).resolve() if args.round_records else run_dir / "round_records.jsonl"
    output = Path(args.output).resolve() if args.output else run_dir / "round_vs_logprob_pretty.png"
    return round_summary, round_records, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Diagnostic run artifact directory.")
    parser.add_argument("--round-summary", type=Path, default=None, help="Optional override for round_summary.csv.")
    parser.add_argument("--round-records", type=Path, default=None, help="Optional override for round_records.jsonl.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument(
        "--title",
        type=str,
        default="Round vs Reference Logprob (round-0 prompt context)",
        help="Plot title.",
    )
    parser.add_argument("--dpi", type=int, default=320, help="Output figure DPI.")
    parser.add_argument("--ci-z", type=float, default=1.96, help="Z-value for mean confidence interval.")
    args = parser.parse_args()

    round_summary_path, round_records_path, out_path = _resolve_paths(args)

    if not round_summary_path.exists():
        raise FileNotFoundError(f"Missing round summary CSV: {round_summary_path}")
    if not round_records_path.exists():
        raise FileNotFoundError(f"Missing round records JSONL: {round_records_path}")

    summary_df = pd.read_csv(round_summary_path)
    if "round" not in summary_df.columns:
        raise ValueError(f"'round' column not found in {round_summary_path}")
    if "mean_reference_logprob_sum" not in summary_df.columns:
        raise ValueError(f"'mean_reference_logprob_sum' column not found in {round_summary_path}")
    if "median_reference_logprob_sum" not in summary_df.columns:
        raise ValueError(f"'median_reference_logprob_sum' column not found in {round_summary_path}")
    summary_df["round"] = summary_df["round"].astype(int)

    by_round = _load_logprobs_by_round(round_records_path)
    disp_df = _compute_round_stats(by_round, ci_z=float(args.ci_z))
    if disp_df.empty:
        raise ValueError(f"No finite round logprob values found in {round_records_path}")
    disp_df["round"] = disp_df["round"].astype(int)

    plot_df = summary_df.merge(disp_df, on="round", how="left").sort_values("round").reset_index(drop=True)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    x = plot_df["round"].to_numpy(dtype=np.int64)
    mean_lp = plot_df["mean_reference_logprob_sum"].to_numpy(dtype=np.float64)
    median_lp = plot_df["median_reference_logprob_sum"].to_numpy(dtype=np.float64)
    q25 = plot_df["q25"].to_numpy(dtype=np.float64)
    q75 = plot_df["q75"].to_numpy(dtype=np.float64)
    ci_lo = plot_df["ci95_lo"].to_numpy(dtype=np.float64)
    ci_hi = plot_df["ci95_hi"].to_numpy(dtype=np.float64)

    mean_color = "#1f77b4"
    median_color = "#d95f02"

    ax.fill_between(x, q25, q75, color=median_color, alpha=0.16, linewidth=0.0, label="IQR (25-75%)")
    ax.fill_between(x, ci_lo, ci_hi, color=mean_color, alpha=0.14, linewidth=0.0, label="Mean 95% CI")

    ax.plot(
        x,
        mean_lp,
        color=mean_color,
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        label="Mean reference logprob sum",
    )
    ax.plot(
        x,
        median_lp,
        color=median_color,
        marker="s",
        markersize=5.0,
        linewidth=2.0,
        linestyle="--",
        label="Median reference logprob sum",
    )

    ax.set_title(args.title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Reference logprob sum")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.15)
    ax.minorticks_on()
    ax.legend(loc="best", frameon=True, framealpha=0.92)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)

    print(f"[DONE] round_summary={round_summary_path}")
    print(f"[DONE] round_records={round_records_path}")
    print(f"[DONE] output={out_path}")


if __name__ == "__main__":
    main()
