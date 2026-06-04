#!/usr/bin/env python3
"""Overlay round-vs-logprob curves for two iterative diagnostic runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

DEFAULT_ORIGINAL_RUN = (
    "artifacts/iter_round_logprob_diag/full25pct_submitall_genonly_u06_20260221_050113"
)
DEFAULT_NEW_RUN = (
    "artifacts/iter_round_logprob_diag/full10pct_submitall_genonly_u06_smalllegal_20260221_171746"
)
DEFAULT_OUTPUT = (
    "artifacts/iter_round_logprob_diag/compare_round_vs_logprob_original_vs_smalllegal.png"
)


@dataclass
class RunData:
    run_dir: Path
    run_id: str
    round_df: pd.DataFrame
    num_prompts: int | None
    sample_frac: float | None
    n_selected: int | None


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {}


def _as_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _load_run(run_dir: Path) -> RunData:
    round_summary_path = run_dir / "round_summary.csv"
    if not round_summary_path.exists():
        raise FileNotFoundError(f"Missing round summary CSV: {round_summary_path}")

    df = pd.read_csv(round_summary_path)
    required_columns = ["round", "mean_reference_logprob_sum", "median_reference_logprob_sum"]
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {round_summary_path}: {missing}")

    df = df[list(required_columns)].copy()
    df["round"] = df["round"].astype(int)
    df = df.sort_values("round").reset_index(drop=True)

    summary = _read_json_if_exists(run_dir / "summary.json")
    config = _read_json_if_exists(run_dir / "config.json")
    sample_info = config.get("sample_info", {}) if isinstance(config.get("sample_info"), dict) else {}

    num_prompts = _as_int(summary.get("num_prompts"))
    if num_prompts is None:
        num_prompts = _as_int(config.get("rows_loaded"))

    sample_frac = _as_float(config.get("sample_frac"))
    n_selected = _as_int(sample_info.get("n_selected"))
    if n_selected is None:
        n_selected = _as_int(num_prompts)

    return RunData(
        run_dir=run_dir,
        run_id=run_dir.name,
        round_df=df,
        num_prompts=num_prompts,
        sample_frac=sample_frac,
        n_selected=n_selected,
    )


def _format_run_meta(label: str, run: RunData) -> str:
    n_text = "n=?"
    if run.num_prompts is not None:
        n_text = f"n={run.num_prompts:,}"
    frac_text = "frac=?"
    if run.sample_frac is not None:
        frac_text = f"frac={run.sample_frac:.2f}"
    return f"{label}: {run.run_id} ({n_text}, {frac_text})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-run-dir",
        type=Path,
        default=Path(DEFAULT_ORIGINAL_RUN),
        help="Path to the original diagnostic run directory.",
    )
    parser.add_argument(
        "--new-run-dir",
        type=Path,
        default=Path(DEFAULT_NEW_RUN),
        help="Path to the new diagnostic run directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Round vs Reference Logprob: Original vs Small-Legal",
        help="Figure title.",
    )
    parser.add_argument("--dpi", type=int, default=320, help="Output figure DPI.")
    args = parser.parse_args()

    original = _load_run(args.original_run_dir.resolve())
    new = _load_run(args.new_run_dir.resolve())
    out_path = args.output.resolve()

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

    fig, ax = plt.subplots(figsize=(11.8, 6.8))

    orig_x = original.round_df["round"].to_numpy()
    orig_mean = original.round_df["mean_reference_logprob_sum"].to_numpy()
    orig_median = original.round_df["median_reference_logprob_sum"].to_numpy()
    new_x = new.round_df["round"].to_numpy()
    new_mean = new.round_df["mean_reference_logprob_sum"].to_numpy()
    new_median = new.round_df["median_reference_logprob_sum"].to_numpy()

    original_color = "#1f77b4"
    new_color = "#d95f02"

    ax.plot(
        orig_x,
        orig_mean,
        color=original_color,
        linewidth=2.4,
        marker="o",
        markersize=5.4,
        linestyle="-",
        label="Original mean",
    )
    ax.plot(
        orig_x,
        orig_median,
        color=original_color,
        linewidth=2.0,
        marker="s",
        markersize=4.9,
        linestyle="--",
        label="Original median",
    )
    ax.plot(
        new_x,
        new_mean,
        color=new_color,
        linewidth=2.4,
        marker="^",
        markersize=5.4,
        linestyle="-",
        label="New mean",
    )
    ax.plot(
        new_x,
        new_median,
        color=new_color,
        linewidth=2.0,
        marker="D",
        markersize=4.8,
        linestyle=":",
        label="New median",
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("Reference logprob sum")
    ax.set_title(args.title)
    all_rounds = sorted(set(int(x) for x in orig_x).union(int(x) for x in new_x))
    ax.set_xticks(all_rounds)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="major", alpha=0.34, linestyle="-")
    ax.minorticks_off()
    ax.legend(loc="best", frameon=True, framealpha=0.93, ncol=2)

    subtitle = (
        f"Original run {original.run_id} (n={original.num_prompts:,}, frac={original.sample_frac:.2f}) vs "
        f"New run {new.run_id} (n={new.num_prompts:,}, frac={new.sample_frac:.2f})"
        if (
            original.num_prompts is not None
            and original.sample_frac is not None
            and new.num_prompts is not None
            and new.sample_frac is not None
        )
        else f"Original run {original.run_id} vs New run {new.run_id}"
    )
    fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=10.2, color="#444444")

    annotation_lines = [
        _format_run_meta("Original", original),
        _format_run_meta("New", new),
    ]
    ax.text(
        0.015,
        0.985,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.6,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9, "boxstyle": "round,pad=0.3"},
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)

    print(f"[DONE] original_run={original.run_dir}")
    print(f"[DONE] new_run={new.run_dir}")
    print(f"[DONE] output={out_path}")


if __name__ == "__main__":
    main()
