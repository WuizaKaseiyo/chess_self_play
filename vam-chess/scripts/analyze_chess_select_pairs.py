#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CorrResult:
    n: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _corr(x: np.ndarray, y: np.ndarray) -> CorrResult:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x2 = x[mask]
    y2 = y[mask]
    if x2.size < 3:
        return CorrResult(n=int(x2.size), pearson_r=float("nan"), pearson_p=float("nan"), spearman_r=float("nan"), spearman_p=float("nan"))

    pearson_r, pearson_p = stats.pearsonr(x2, y2)
    spearman_r, spearman_p = stats.spearmanr(x2, y2)
    return CorrResult(
        n=int(x2.size),
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        spearman_r=float(spearman_r),
        spearman_p=float(spearman_p),
    )


def _format_corr(name: str, res: CorrResult) -> str:
    return (
        f"- {name}: n={res.n} | pearson r={res.pearson_r:+.3f} (p={res.pearson_p:.2e}) | "
        f"spearman ρ={res.spearman_r:+.3f} (p={res.spearman_p:.2e})"
    )


def _bootstrap_mean_ci(values: np.ndarray, *, rng: np.random.Generator, iters: int) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if iters <= 0:
        m = float(v.mean())
        return m, float("nan"), float("nan")

    n = int(v.size)
    means = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        sample = rng.integers(0, n, size=n)
        means[i] = float(v[sample].mean())
    means.sort()
    lo = float(np.quantile(means, 0.025))
    hi = float(np.quantile(means, 0.975))
    return float(v.mean()), lo, hi


def _load_results(input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {input_dir}")
    manifest = _read_json(manifest_path)

    shard_paths = sorted(input_dir.glob("results_shard*of*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No results_shard*.jsonl files found in {input_dir}")

    dfs: list[pd.DataFrame] = []
    for p in shard_paths:
        df = pd.read_json(p, lines=True)
        df["__shard_file__"] = p.name
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)

    required_cols = [
        "row_id",
        "pair_unordered_id",
        "pair_id",
        "candidate_moves_uci",
        "target_move_uci",
        "gap",
        "good_is_first",
        "n_samples",
        "n_correct",
        "n_pred_first",
        "n_pred_second",
        "n_pred_other",
        "n_in_subset",
        "n_format_ok",
        "n_format_error",
    ]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns in results: {missing}")

    # Types.
    for c in ("row_id", "n_samples", "n_correct", "n_pred_first", "n_pred_second", "n_pred_other", "n_in_subset", "n_format_ok", "n_format_error"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    out["gap"] = pd.to_numeric(out["gap"], errors="coerce").astype(float)
    out["good_is_first"] = out["good_is_first"].astype(bool)

    dup_mask = out.duplicated(subset=["row_id", "pair_id"], keep=False)
    if bool(dup_mask.any()):
        dup_n = int(dup_mask.sum())
        sample = out.loc[dup_mask, ["row_id", "pair_id", "__shard_file__"]].head(10).to_dict(orient="records")
        raise ValueError(f"Found {dup_n} duplicate (row_id, pair_id) records. Sample:\n{json.dumps(sample, indent=2)}")

    return out, manifest


def _binned_means(df: pd.DataFrame, x_col: str, y_cols: list[str], *, bins: int) -> pd.DataFrame:
    x = df[x_col].astype(float)
    try:
        q = pd.qcut(x, q=bins, duplicates="drop")
    except Exception:
        q = pd.cut(x, bins=bins)
    agg = {"x_mean": (x_col, "mean"), "n": (x_col, "size")}
    for y in y_cols:
        agg[f"{y}_mean"] = (y, "mean")
    out = df.assign(_bin=q).groupby("_bin", observed=True).agg(**agg).reset_index(drop=True).sort_values("x_mean", ascending=True)
    return out


def _make_plots(pair_df: pd.DataFrame, *, out_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # 1) Gap distribution
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(pair_df["gap"].astype(float), bins=80)
    ax.set_xlabel("μ gap = μ_good - μ_bad (within pair)")
    ax.set_ylabel("num pairs")
    ax.set_title("Gap distribution across 2-move pairs")
    fig.tight_layout()
    p = out_dir / "gap_hist.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["gap_hist"] = str(p)

    # 2) Binned probabilities vs gap
    b = _binned_means(
        pair_df,
        "gap",
        ["p_good_first", "p_good_second", "p_bad_when_bad_first"],
        bins=30,
    )
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(b["x_mean"], b["p_good_first_mean"], marker="o", markersize=3, linewidth=1, label="P(select Good | [Good,Bad])")
    ax.plot(b["x_mean"], b["p_good_second_mean"], marker="o", markersize=3, linewidth=1, label="P(select Good | [Bad,Good])")
    ax.plot(b["x_mean"], b["p_bad_when_bad_first_mean"], marker="o", markersize=3, linewidth=1, label="P(select Bad | [Bad,Good])")
    ax.set_xlabel("gap (binned by quantiles; x=mean gap in bin)")
    ax.set_ylabel("mean probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("2-move selection: probabilities vs μ gap (binned)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    p = out_dir / "probs_vs_gap_binned.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["probs_vs_gap_binned"] = str(p)

    # 3) Order effect vs gap
    b2 = _binned_means(pair_df, "gap", ["order_effect"], bins=30)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(b2["x_mean"], b2["order_effect_mean"], marker="o", markersize=3, linewidth=1)
    ax.set_xlabel("gap (binned; x=mean gap)")
    ax.set_ylabel("mean ΔP = P(Good|[Good,Bad]) - P(Good|[Bad,Good])")
    ax.set_title("Order effect vs gap (binned)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = out_dir / "order_effect_vs_gap_binned.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["order_effect_vs_gap_binned"] = str(p)

    # 4) Same plot but restricted to strictly non-tie pairs (gap > 0).
    df_pos = pair_df[pair_df["gap"].astype(float) > 0].copy()
    if len(df_pos) >= 500:
        b3 = _binned_means(
            df_pos,
            "gap",
            ["p_good_first", "p_good_second", "p_bad_when_bad_first"],
            bins=20,
        )
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(b3["x_mean"], b3["p_good_first_mean"], marker="o", markersize=3, linewidth=1, label="P(select Good | [Good,Bad])")
        ax.plot(b3["x_mean"], b3["p_good_second_mean"], marker="o", markersize=3, linewidth=1, label="P(select Good | [Bad,Good])")
        ax.plot(b3["x_mean"], b3["p_bad_when_bad_first_mean"], marker="o", markersize=3, linewidth=1, label="P(select Bad | [Bad,Good])")
        ax.set_xlabel("gap>0 only (binned by quantiles; x=mean gap in bin)")
        ax.set_ylabel("mean probability")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("2-move selection: probabilities vs μ gap (gap>0 only)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        p = out_dir / "probs_vs_gap_binned_gapgt0.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths["probs_vs_gap_binned_gapgt0"] = str(p)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing manifest.json + results_shard*.jsonl")
    ap.add_argument("--out_dir", default=None, help="Output directory for plots + report. Defaults under analysis/select_pairs_eval_reports/<run_id>/")
    ap.add_argument("--source_out_dir", default=None, help="Optional: original cluster path to include in the report.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap_iters", type=int, default=2000)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Missing input_dir: {input_dir}")

    run_id = input_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else Path("analysis") / "select_pairs_eval_reports" / run_id
    plots_dir = out_dir / "plots"

    df, manifest = _load_results(input_dir)

    # Basic derived rates (ordered-record level).
    df["p_good"] = df["n_correct"].astype(float) / df["n_samples"].astype(float)
    df["p_first"] = df["n_pred_first"].astype(float) / df["n_samples"].astype(float)
    df["p_in_subset"] = df["n_in_subset"].astype(float) / df["n_samples"].astype(float)
    df["p_format_ok"] = df["n_format_ok"].astype(float) / df["n_samples"].astype(float)

    # Pair-level join: for each unordered pair, we should have exactly two records (good first, good second).
    idx_cols = ["row_id", "pair_unordered_id"]
    df_first = df[df["good_is_first"]].set_index(idx_cols)
    df_second = df[~df["good_is_first"]].set_index(idx_cols)

    pair_df = df_first[["gap", "p_good", "p_first", "p_in_subset", "p_format_ok"]].rename(
        columns={
            "p_good": "p_good_first",
            "p_first": "p_first_when_good_first",
            "p_in_subset": "p_in_subset_good_first",
            "p_format_ok": "p_format_ok_good_first",
        }
    )
    pair_df = pair_df.join(
        df_second[["gap", "p_good", "p_first", "p_in_subset", "p_format_ok"]].rename(
            columns={
                "gap": "gap_2",
                "p_good": "p_good_second",
                "p_first": "p_bad_when_bad_first",
                "p_in_subset": "p_in_subset_bad_first",
                "p_format_ok": "p_format_ok_bad_first",
            }
        ),
        how="inner",
    )

    # Safety: gap should be identical across both orders.
    pair_df["gap"] = pair_df[["gap", "gap_2"]].max(axis=1)
    pair_df = pair_df.drop(columns=["gap_2"])

    pair_df["order_effect"] = pair_df["p_good_first"] - pair_df["p_good_second"]
    pair_df = pair_df.reset_index()

    total_rows = int(df["row_id"].nunique())
    total_ordered = int(len(df))
    total_pairs = int(len(pair_df))

    # Global correlations (pair-level).
    corr_good_first = _corr(pair_df["gap"].to_numpy(), pair_df["p_good_first"].to_numpy())
    corr_good_second = _corr(pair_df["gap"].to_numpy(), pair_df["p_good_second"].to_numpy())
    corr_bad_when_bad_first = _corr(pair_df["gap"].to_numpy(), pair_df["p_bad_when_bad_first"].to_numpy())
    corr_order_effect = _corr(pair_df["gap"].to_numpy(), pair_df["order_effect"].to_numpy())

    # Focused analysis: only non-tie pairs where μ_good != μ_bad (gap > 0).
    pair_pos = pair_df[pair_df["gap"].astype(float) > 0].copy()
    corr_pos_good_second = _corr(pair_pos["gap"].to_numpy(), pair_pos["p_good_second"].to_numpy())
    corr_pos_bad_when_bad_first = _corr(pair_pos["gap"].to_numpy(), pair_pos["p_bad_when_bad_first"].to_numpy())
    corr_pos_good_first = _corr(pair_pos["gap"].to_numpy(), pair_pos["p_good_first"].to_numpy())
    corr_pos_order_effect = _corr(pair_pos["gap"].to_numpy(), pair_pos["order_effect"].to_numpy())

    # Within-row correlations (control for row difficulty / μ scale).
    per_row: list[dict[str, Any]] = []
    for rid, g in pair_df.groupby(pair_df["row_id"].astype(int), observed=True):
        if len(g) < 50:
            continue
        per_row.append(
            {
                "row_id": int(rid),
                "n_pairs": int(len(g)),
                "spearman_gap_p_good_second": float(_corr(g["gap"].to_numpy(), g["p_good_second"].to_numpy()).spearman_r),
                "spearman_gap_p_bad_when_bad_first": float(_corr(g["gap"].to_numpy(), g["p_bad_when_bad_first"].to_numpy()).spearman_r),
                "spearman_gap_order_effect": float(_corr(g["gap"].to_numpy(), g["order_effect"].to_numpy()).spearman_r),
            }
        )
    per_row_df = pd.DataFrame(per_row)

    rng = np.random.default_rng(int(args.seed))
    iters = int(args.bootstrap_iters)
    per_row_ci: dict[str, Any] = {}
    if len(per_row_df) > 0:
        for k in ["spearman_gap_p_good_second", "spearman_gap_p_bad_when_bad_first", "spearman_gap_order_effect"]:
            mean, lo, hi = _bootstrap_mean_ci(per_row_df[k].to_numpy(), rng=rng, iters=iters)
            per_row_ci[k] = {"mean": float(mean), "ci95_lo": float(lo), "ci95_hi": float(hi), "bootstrap_iters": int(iters)}

    # Within-row correlations for gap>0 only (this matches the "Good vs Bad" framing better).
    per_row_pos: list[dict[str, Any]] = []
    for rid, g in pair_pos.groupby(pair_pos["row_id"].astype(int), observed=True):
        if len(g) < 20:
            continue
        per_row_pos.append(
            {
                "row_id": int(rid),
                "n_pairs_gap_gt0": int(len(g)),
                "spearman_gap_p_good_second": float(_corr(g["gap"].to_numpy(), g["p_good_second"].to_numpy()).spearman_r),
                "spearman_gap_p_bad_when_bad_first": float(_corr(g["gap"].to_numpy(), g["p_bad_when_bad_first"].to_numpy()).spearman_r),
                "spearman_gap_order_effect": float(_corr(g["gap"].to_numpy(), g["order_effect"].to_numpy()).spearman_r),
            }
        )
    per_row_pos_df = pd.DataFrame(per_row_pos)
    per_row_pos_ci: dict[str, Any] = {}
    if len(per_row_pos_df) > 0:
        for k in ["spearman_gap_p_good_second", "spearman_gap_p_bad_when_bad_first", "spearman_gap_order_effect"]:
            mean, lo, hi = _bootstrap_mean_ci(per_row_pos_df[k].to_numpy(), rng=rng, iters=iters)
            per_row_pos_ci[k] = {"mean": float(mean), "ci95_lo": float(lo), "ci95_hi": float(hi), "bootstrap_iters": int(iters)}

    # Quantile slices for effect sizes.
    q_lo = float(pair_df["gap"].quantile(0.10))
    q_hi = float(pair_df["gap"].quantile(0.90))
    low = pair_df[pair_df["gap"] <= q_lo]
    high = pair_df[pair_df["gap"] >= q_hi]

    slice_stats = {
        "gap_q10": q_lo,
        "gap_q90": q_hi,
        "p_good_second_mean_lowgap": float(low["p_good_second"].mean()) if len(low) else float("nan"),
        "p_good_second_mean_highgap": float(high["p_good_second"].mean()) if len(high) else float("nan"),
        "p_bad_when_bad_first_mean_lowgap": float(low["p_bad_when_bad_first"].mean()) if len(low) else float("nan"),
        "p_bad_when_bad_first_mean_highgap": float(high["p_bad_when_bad_first"].mean()) if len(high) else float("nan"),
        "order_effect_mean_lowgap": float(low["order_effect"].mean()) if len(low) else float("nan"),
        "order_effect_mean_highgap": float(high["order_effect"].mean()) if len(high) else float("nan"),
    }

    # Quantile slices on gap>0 only.
    if len(pair_pos) > 0:
        q_lo_pos = float(pair_pos["gap"].quantile(0.10))
        q_hi_pos = float(pair_pos["gap"].quantile(0.90))
        low_pos = pair_pos[pair_pos["gap"] <= q_lo_pos]
        high_pos = pair_pos[pair_pos["gap"] >= q_hi_pos]
        slice_stats_pos = {
            "gap_q10": q_lo_pos,
            "gap_q90": q_hi_pos,
            "p_good_second_mean_lowgap": float(low_pos["p_good_second"].mean()) if len(low_pos) else float("nan"),
            "p_good_second_mean_highgap": float(high_pos["p_good_second"].mean()) if len(high_pos) else float("nan"),
            "p_bad_when_bad_first_mean_lowgap": float(low_pos["p_bad_when_bad_first"].mean()) if len(low_pos) else float("nan"),
            "p_bad_when_bad_first_mean_highgap": float(high_pos["p_bad_when_bad_first"].mean()) if len(high_pos) else float("nan"),
            "p_good_first_mean_lowgap": float(low_pos["p_good_first"].mean()) if len(low_pos) else float("nan"),
            "p_good_first_mean_highgap": float(high_pos["p_good_first"].mean()) if len(high_pos) else float("nan"),
        }
    else:
        slice_stats_pos = {}

    plot_paths = _make_plots(pair_df, out_dir=plots_dir)

    summary = {
        "input_dir": str(input_dir),
        "source_out_dir": str(args.source_out_dir or ""),
        "total_rows": total_rows,
        "total_pairs_unordered": total_pairs,
        "total_pairs_ordered": total_ordered,
        "manifest": manifest,
        "overall_means": {
            "p_good_first_mean": float(pair_df["p_good_first"].mean()),
            "p_good_second_mean": float(pair_df["p_good_second"].mean()),
            "p_bad_when_bad_first_mean": float(pair_df["p_bad_when_bad_first"].mean()),
            "order_effect_mean": float(pair_df["order_effect"].mean()),
        },
        "gap_quantiles": {str(q): float(pair_df["gap"].quantile(q)) for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]},
        "gap_zero_fraction": float((pair_df["gap"].astype(float) == 0.0).mean()),
        "gap_gt0_count": int(len(pair_pos)),
        "gap_gt0_fraction": float(len(pair_pos) / float(len(pair_df))) if len(pair_df) else float("nan"),
        "correlations_pair_level": {
            "gap_vs_p_good_first": corr_good_first.__dict__,
            "gap_vs_p_good_second": corr_good_second.__dict__,
            "gap_vs_p_bad_when_bad_first": corr_bad_when_bad_first.__dict__,
            "gap_vs_order_effect": corr_order_effect.__dict__,
        },
        "correlations_pair_level_gap_gt0": {
            "gap_vs_p_good_first": corr_pos_good_first.__dict__,
            "gap_vs_p_good_second": corr_pos_good_second.__dict__,
            "gap_vs_p_bad_when_bad_first": corr_pos_bad_when_bad_first.__dict__,
            "gap_vs_order_effect": corr_pos_order_effect.__dict__,
        },
        "within_row_spearman_bootstrap_ci": per_row_ci,
        "within_row_spearman_bootstrap_ci_gap_gt0": per_row_pos_ci,
        "gap_slice_stats_q10_q90": slice_stats,
        "gap_gt0_slice_stats_q10_q90": slice_stats_pos,
        "plots": plot_paths,
    }
    _write_json(out_dir / "summary.json", summary)

    def fmt_pct(x: float) -> str:
        if not math.isfinite(x):
            return "nan"
        return f"{100.0 * x:.2f}%"

    report_lines: list[str] = []
    report_lines.extend(
        [
            "# 2-move selection analysis: does μ-gap reduce position bias?",
            "",
            "We evaluate every unordered 2-move pair from the first 100 positions, under both prompt orders:",
            "- `[Good, Bad]` (good move listed first)",
            "- `[Bad, Good]` (bad move listed first)",
            "",
            "## Artifacts",
            f"- Local input_dir: `{input_dir}`",
            f"- Original cluster out_dir: `{args.source_out_dir or ''}`",
            "",
            "## What is being measured?",
            "- Good/Bad are defined by μ = expected score from the dataset (within the pair).",
            "- gap = μ_good - μ_bad (>=0).",
            "- Probability is estimated empirically from `n_samples` stochastic generations per prompt.",
            "",
            "## Summary",
            f"- Rows: {total_rows}",
            f"- Unordered pairs: {total_pairs} (each evaluated in both orders → ordered prompts={total_ordered})",
            f"- Mean P(select Good | [Good,Bad]) = {pair_df['p_good_first'].mean():.4f}",
            f"- Mean P(select Good | [Bad,Good]) = {pair_df['p_good_second'].mean():.4f}",
            f"- Mean P(select Bad | [Bad,Good]) = {pair_df['p_bad_when_bad_first'].mean():.4f}",
            f"- Mean order effect Δ = P(Good|[Good,Bad]) - P(Good|[Bad,Good]) = {pair_df['order_effect'].mean():.4f}",
            "",
            "## Main question: does increasing gap help the model override list-order bias?",
            _format_corr("gap vs P(select Good | [Bad,Good])", corr_good_second),
            _format_corr("gap vs P(select Bad | [Bad,Good])", corr_bad_when_bad_first),
            "",
            f"Important note: this dataset has a huge mass of μ ties in 2-move pairs (gap=0) — frac(gap=0)={summary['gap_zero_fraction']:.3f}.",
            "Since 'Good vs Bad' only makes semantic sense when μ_good != μ_bad, we also report the same stats on gap>0 only.",
            "",
            "### gap>0 only (μ_good != μ_bad)",
            f"- Pairs with gap>0: n={summary['gap_gt0_count']} / {total_pairs} ({100.0*summary['gap_gt0_fraction']:.2f}%)",
            _format_corr("gap vs P(select Good | [Bad,Good]) (gap>0)", corr_pos_good_second),
            _format_corr("gap vs P(select Bad | [Bad,Good]) (gap>0)", corr_pos_bad_when_bad_first),
            "",
            "Effect sizes (gap quantiles, pair-level):",
            f"- gap q10={slice_stats['gap_q10']:.4f} → mean P(Good|[Bad,Good])={slice_stats['p_good_second_mean_lowgap']:.4f}",
            f"- gap q90={slice_stats['gap_q90']:.4f} → mean P(Good|[Bad,Good])={slice_stats['p_good_second_mean_highgap']:.4f}",
            f"- gap q10={slice_stats['gap_q10']:.4f} → mean P(Bad|[Bad,Good])={slice_stats['p_bad_when_bad_first_mean_lowgap']:.4f}",
            f"- gap q90={slice_stats['gap_q90']:.4f} → mean P(Bad|[Bad,Good])={slice_stats['p_bad_when_bad_first_mean_highgap']:.4f}",
            "",
            "Effect sizes (gap>0 only; gap quantiles within the non-tie subset):",
        ]
    )
    if slice_stats_pos:
        report_lines.extend(
            [
                f"- gap>0 q10={slice_stats_pos['gap_q10']:.4f} → mean P(Good|[Bad,Good])={slice_stats_pos['p_good_second_mean_lowgap']:.4f}",
                f"- gap>0 q90={slice_stats_pos['gap_q90']:.4f} → mean P(Good|[Bad,Good])={slice_stats_pos['p_good_second_mean_highgap']:.4f}",
                f"- gap>0 q10={slice_stats_pos['gap_q10']:.4f} → mean P(Bad|[Bad,Good])={slice_stats_pos['p_bad_when_bad_first_mean_lowgap']:.4f}",
                f"- gap>0 q90={slice_stats_pos['gap_q90']:.4f} → mean P(Bad|[Bad,Good])={slice_stats_pos['p_bad_when_bad_first_mean_highgap']:.4f}",
            ]
        )
    else:
        report_lines.append("- (No gap>0 slice stats; no non-tie pairs.)")

    report_lines.extend(
        [
            "## Controls: within-row correlations (Spearman; bootstrap CI over rows)",
        ]
    )
    if per_row_ci:
        for k, v in per_row_ci.items():
            report_lines.append(f"- {k}: mean={v['mean']:+.3f} CI95=[{v['ci95_lo']:+.3f}, {v['ci95_hi']:+.3f}] (rows used={len(per_row_df)})")
    else:
        report_lines.append("- (No per-row CI computed; insufficient per-row data.)")

    report_lines.append("")
    report_lines.append("Within-row correlations on gap>0 only:")
    if per_row_pos_ci:
        for k, v in per_row_pos_ci.items():
            report_lines.append(
                f"- (gap>0) {k}: mean={v['mean']:+.3f} CI95=[{v['ci95_lo']:+.3f}, {v['ci95_hi']:+.3f}] (rows used={len(per_row_pos_df)})"
            )
    else:
        report_lines.append("- (No per-row gap>0 CI computed; insufficient per-row non-tie pairs.)")

    report_lines.extend(
        [
            "",
            "## Additional correlations (sanity)",
            _format_corr("gap vs P(select Good | [Good,Bad])", corr_good_first),
            _format_corr("gap vs order_effect Δ", corr_order_effect),
            "",
            "## Compliance (should be ~100%; otherwise probability estimates are contaminated)",
            f"- Mean in-subset rate ([Good,Bad]) = {fmt_pct(pair_df['p_in_subset_good_first'].mean())}",
            f"- Mean in-subset rate ([Bad,Good]) = {fmt_pct(pair_df['p_in_subset_bad_first'].mean())}",
            f"- Mean strict-format-ok rate ([Good,Bad]) = {fmt_pct(pair_df['p_format_ok_good_first'].mean())}",
            f"- Mean strict-format-ok rate ([Bad,Good]) = {fmt_pct(pair_df['p_format_ok_bad_first'].mean())}",
            "",
            "## Plots",
            *(f"- {k}: `{v}`" for k, v in plot_paths.items()),
            "",
            "## Notes",
            "- This answers *exactly* the question you asked: we explicitly force the two candidate orders per pair.",
            "- If there is a strong positive relationship between gap and P(Good|[Bad,Good]), that indicates the model is",
            "  using (some) chess information to overcome position bias when the choice is obvious.",
            "",
        ]
    )

    _write_text(out_dir / "report.md", "\n".join(report_lines) + "\n")
    print(f"Wrote {out_dir/'report.md'}")
    print(f"Wrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
