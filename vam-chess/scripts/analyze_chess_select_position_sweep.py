#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--smooth_window", type=int, default=5)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir or (results_dir / "analysis")
    _ensure_dir(out_dir)

    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("results_shard*.jsonl")):
        for rec in _iter_jsonl(path):
            rows.append(rec)
    if not rows:
        raise SystemExit(f"No results found under {results_dir}")

    df = pd.DataFrame(rows)
    if "mode" not in df.columns:
        def _infer_mode(key: str) -> str:
            if isinstance(key, str) and "__original" in key:
                return "original_order"
            return "sweep_shuffle"
        df["mode"] = df["key"].apply(_infer_mode)
    df["k_ratio"] = df.apply(
        lambda r: (r["k_pos"] / (r["n_considered"] - 1)) if r["n_considered"] > 1 else 0.0,
        axis=1,
    )

    agg = df.groupby(["mode", "k_pos"]).agg(
        n_prompts=("row_id", "count"),
        pass_at_k=("pass_at_k", "mean"),
        success_rate=("success_rate", "mean"),
    ).reset_index()
    agg.to_csv(out_dir / "pass_by_k.csv", index=False)

    # Ratio bins
    bins = np.linspace(0.0, 1.0, 21)
    df["k_ratio_bin"] = pd.cut(df["k_ratio"], bins=bins, include_lowest=True)
    agg_ratio = df.groupby(["mode", "k_ratio_bin"]).agg(
        n_prompts=("row_id", "count"),
        pass_at_k=("pass_at_k", "mean"),
        success_rate=("success_rate", "mean"),
    ).reset_index()
    agg_ratio.to_csv(out_dir / "pass_by_ratio.csv", index=False)

    do_plots = not args.no_plots
    plt = None
    if do_plots:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[WARN] matplotlib not available ({exc}); skipping plots.")
            do_plots = False

    if do_plots and plt is not None:
        sweep = agg[agg["mode"] == "sweep_shuffle"].sort_values("k_pos")
        if not sweep.empty:
            plt.figure(figsize=(10, 4))
            ax = plt.gca()
            ax.plot(sweep["k_pos"], sweep["pass_at_k"], label="pass@32", alpha=0.7)
            ax.plot(sweep["k_pos"], sweep["success_rate"], label="pass@1", alpha=0.7)
            ax.set_xlabel("K position")
            ax.set_ylabel("pass rate")
            ax.set_title("Pass@1 and Pass@32 vs target position K (sweep_shuffle)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "pass_rates_by_k.png", dpi=180, bbox_inches="tight")
            plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
