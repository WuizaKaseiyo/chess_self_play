#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


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

    # Basic schema checks.
    required_cols = [
        "row_id",
        "subset_id",
        "k",
        "top_margin",
        "h1",
        "n_samples",
        "n_correct",
        "success_rate",
        "pass_at_8",
        "n_format_ok",
        "n_in_subset",
        "n_bad_move",
        "n_format_error",
    ]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns in results: {missing}")

    # Enforce types.
    for c in ("row_id", "k", "n_samples", "n_correct", "pass_at_8", "n_format_ok", "n_in_subset", "n_bad_move", "n_format_error"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    for c in ("success_rate", "top_margin", "h1"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)

    expected_total = int(manifest.get("total_subsets") or 0)
    if expected_total and len(out) != expected_total:
        raise ValueError(f"Expected total_subsets={expected_total} rows, but loaded {len(out)}")

    dup_mask = out.duplicated(subset=["row_id", "subset_id"], keep=False)
    if bool(dup_mask.any()):
        dup_n = int(dup_mask.sum())
        sample = out.loc[dup_mask, ["row_id", "subset_id", "__shard_file__"]].head(10).to_dict(orient="records")
        raise ValueError(f"Found {dup_n} duplicate (row_id, subset_id) records. Sample:\n{json.dumps(sample, indent=2)}")

    return out, manifest


def _corr(x: np.ndarray, y: np.ndarray) -> CorrResult:
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


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.nanmean(x))
    s = float(np.nanstd(x))
    if not math.isfinite(s) or s <= 0:
        return np.zeros_like(x, dtype=np.float64)
    return (x - m) / s


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _fit_logistic_irls(X: np.ndarray, y: np.ndarray, *, max_iter: int = 50, tol: float = 1e-8, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression via IRLS (Newton), with small ridge for stability.

    Returns (beta, se) where se is the approximate (Wald) std error from (X'WX)^-1.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    n, p = X.shape
    if n < p + 1:
        raise ValueError(f"Not enough samples for logistic regression: n={n}, p={p}")

    beta = np.zeros(p, dtype=np.float64)
    ridge = float(ridge)
    if ridge < 0:
        raise ValueError("ridge must be >= 0")

    for _ in range(max_iter):
        z = X @ beta
        p_hat = _sigmoid(z)
        w = p_hat * (1.0 - p_hat)
        # Avoid division by zero / perfect separation instability.
        w = np.clip(w, 1e-9, None)
        # Newton step: beta_new = beta + (X'WX)^-1 X'(y - p)
        XtW = X.T * w
        H = XtW @ X
        if ridge > 0:
            H = H + ridge * np.eye(p, dtype=np.float64)
        g = X.T @ (y - p_hat)
        step = np.linalg.solve(H, g)
        beta_new = beta + step
        if float(np.max(np.abs(beta_new - beta))) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Final covariance estimate.
    z = X @ beta
    p_hat = _sigmoid(z)
    w = np.clip(p_hat * (1.0 - p_hat), 1e-9, None)
    XtW = X.T * w
    H = XtW @ X
    if ridge > 0:
        H = H + ridge * np.eye(p, dtype=np.float64)
    cov = np.linalg.inv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return beta, se


def _format_corr(name: str, res: CorrResult) -> str:
    return (
        f"- {name}: n={res.n} | pearson r={res.pearson_r:+.3f} (p={res.pearson_p:.2e}) | "
        f"spearman ρ={res.spearman_r:+.3f} (p={res.spearman_p:.2e})"
    )


def _bucket_k(k: int) -> str:
    if k <= 4:
        return "k=2-4"
    if k <= 16:
        return "k=5-16"
    return "k>=17"


def _binned_means(df: pd.DataFrame, x_col: str, y_col: str, *, bins: int) -> pd.DataFrame:
    # Quantile bins (robust to skew).
    x = df[x_col].astype(float)
    try:
        q = pd.qcut(x, q=bins, duplicates="drop")
    except Exception:
        # Fall back to uniform bins.
        q = pd.cut(x, bins=bins)
    out = (
        df.assign(_bin=q)
        .groupby("_bin", observed=True)
        .agg(x_mean=(x_col, "mean"), y_mean=(y_col, "mean"), n=(y_col, "size"))
        .reset_index(drop=True)
        .sort_values("x_mean", ascending=True)
    )
    return out


def _make_plots(df: pd.DataFrame, *, out_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}

    # 1) k distribution
    k_counts = df["k"].value_counts().sort_index()
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(k_counts.index.astype(int), k_counts.values.astype(int), width=0.9)
    ax.set_xlabel("subset size k")
    ax.set_ylabel("num subsets")
    ax.set_title("Subset count by k")
    ax.set_yscale("log")
    fig.tight_layout()
    p = out_dir / "k_counts.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    plot_paths["k_counts"] = str(p)

    # 2) Hardness distributions
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.hist(df["top_margin"].astype(float), bins=50)
    ax1.set_title("Top margin distribution")
    ax1.set_xlabel("|μ_best - μ_2nd|")
    ax1.set_ylabel("count")
    ax2.hist(np.log1p(df["h1"].astype(float)), bins=50)
    ax2.set_title("log1p(H1) distribution")
    ax2.set_xlabel("log1p(H1)")
    ax2.set_ylabel("count")
    fig.tight_layout()
    p = out_dir / "hardness_hist.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    plot_paths["hardness_hist"] = str(p)

    # 3) Binned success_rate vs hardness, stratified by k bucket
    df2 = df[df["k"] >= 2].copy()
    df2["k_bucket"] = df2["k"].astype(int).map(_bucket_k)
    df2["log1p_h1"] = np.log1p(df2["h1"].astype(float))

    for x_col, name in [("top_margin", "margin"), ("log1p_h1", "logh1")]:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        for bucket in ["k=2-4", "k=5-16", "k>=17"]:
            sub = df2[df2["k_bucket"] == bucket]
            if len(sub) < 50:
                continue
            b = _binned_means(sub, x_col, "success_rate", bins=20)
            ax.plot(b["x_mean"], b["y_mean"], marker="o", markersize=3, linewidth=1, label=f"{bucket} (n={len(sub)})")
        ax.set_xlabel(x_col)
        ax.set_ylabel("mean success_rate (n_correct/8)")
        ax.set_title(f"Binned success_rate vs {x_col} (by k bucket)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        p = out_dir / f"binned_success_vs_{name}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths[f"binned_success_vs_{name}"] = str(p)

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        for bucket in ["k=2-4", "k=5-16", "k>=17"]:
            sub = df2[df2["k_bucket"] == bucket]
            if len(sub) < 50:
                continue
            b = _binned_means(sub, x_col, "pass_at_8", bins=20)
            ax.plot(b["x_mean"], b["y_mean"], marker="o", markersize=3, linewidth=1, label=f"{bucket} (n={len(sub)})")
        ax.set_xlabel(x_col)
        ax.set_ylabel("mean pass@8 (any correct in 8)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Binned pass@8 vs {x_col} (by k bucket)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        p = out_dir / f"binned_pass8_vs_{name}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths[f"binned_pass8_vs_{name}"] = str(p)

    # 4) pass@8 by target rank (position bias diagnostic)
    if "target_rank" in df2.columns:
        rank_counts = df2["target_rank"].value_counts().sort_index()
        # Cap ranks for readability.
        max_rank = 10
        keep = rank_counts.index.astype(int) <= max_rank
        ranks = rank_counts.index.astype(int)[keep]
        if len(ranks) > 0:
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)
            means = []
            ns = []
            for r in ranks:
                sub = df2[df2["target_rank"].astype(int) == int(r)]
                means.append(float(sub["pass_at_8"].mean()))
                ns.append(int(len(sub)))
            ax.bar([int(r) for r in ranks], means, width=0.8)
            ax.set_xlabel("target_rank in considered_moves (1=first)")
            ax.set_ylabel("mean pass@8")
            ax.set_ylim(0.0, 1.0)
            ax.set_title("Position bias diagnostic: pass@8 vs target_rank (k>=2)")
            ax.grid(True, axis="y", alpha=0.2)
            fig.tight_layout()
            p = out_dir / "pass8_by_target_rank.png"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            plot_paths["pass8_by_target_rank"] = str(p)

    return plot_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing manifest.json and results_shard*.jsonl")
    ap.add_argument("--out_dir", default=None, help="Output directory for plots + report. Defaults under analysis/select_eval_reports/<run_id>/")
    ap.add_argument(
        "--source_out_dir",
        default=None,
        help="Optional: original cluster path (string) to include in the report for reproducibility.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap_iters", type=int, default=2000)
    ap.add_argument("--min_subsets_per_row", type=int, default=50, help="Rows with fewer subsets are excluded from per-row correlation stats.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Missing input_dir: {input_dir}")

    run_id = input_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else Path("analysis") / "select_eval_reports" / run_id
    plots_dir = out_dir / "plots"

    df, manifest = _load_results(input_dir)

    # Derived metrics.
    df["format_ok_rate"] = df["n_format_ok"].astype(float) / df["n_samples"].astype(float)
    df["in_subset_rate"] = df["n_in_subset"].astype(float) / df["n_samples"].astype(float)
    df["bad_move_rate"] = df["n_bad_move"].astype(float) / df["n_samples"].astype(float)
    df["format_error_rate"] = df["n_format_error"].astype(float) / df["n_samples"].astype(float)
    df["log1p_h1"] = np.log1p(df["h1"].astype(float))

    # Rank / order diagnostics (position bias).
    def _first_move(x: Any) -> str:
        if isinstance(x, list) and x:
            return str(x[0])
        return ""

    def _index_1based(xs: Any, needle: str) -> int | None:
        if not isinstance(xs, list) or not needle:
            return None
        try:
            return int(xs.index(needle)) + 1
        except Exception:
            return None

    df["first_candidate_move"] = df["candidate_moves_uci"].apply(_first_move)
    df["target_rank"] = df.apply(lambda r: _index_1based(r.get("candidate_moves_uci"), str(r.get("target_move_uci") or "")), axis=1)
    df["pred_rank"] = df.apply(lambda r: _index_1based(r.get("candidate_moves_uci"), str(r.get("most_common_pred_move") or "")), axis=1)
    df["target_is_first"] = (df["target_rank"].astype("Int64") == 1)
    df["pred_is_first"] = (df["pred_rank"].astype("Int64") == 1)

    # Summary stats.
    total_subsets = int(len(df))
    total_rows = int(df["row_id"].nunique())
    per_row_counts = df.groupby("row_id").size().astype(int)
    k_counts = df["k"].value_counts().sort_index()

    n_correct_hist = df["n_correct"].value_counts().sort_index()

    summary = {
        "input_dir": str(input_dir),
        "source_out_dir": str(args.source_out_dir or ""),
        "total_subsets": total_subsets,
        "total_rows": total_rows,
        "subsets_per_row": {
            "min": int(per_row_counts.min()),
            "p50": int(per_row_counts.median()),
            "max": int(per_row_counts.max()),
            "mean": float(per_row_counts.mean()),
        },
        "k": {
            "unique_k": [int(x) for x in k_counts.index.astype(int).tolist()],
            "counts": {str(int(k)): int(v) for k, v in k_counts.items()},
        },
        "overall": {
            "success_rate_mean": float(df["success_rate"].mean()),
            "pass_at_8_mean": float(df["pass_at_8"].mean()),
            "format_ok_rate_mean": float(df["format_ok_rate"].mean()),
            "in_subset_rate_mean": float(df["in_subset_rate"].mean()),
            "bad_move_rate_mean": float(df["bad_move_rate"].mean()),
            "format_error_rate_mean": float(df["format_error_rate"].mean()),
            "pred_is_first_frac": float(df["pred_is_first"].mean()),
            "target_is_first_frac": float(df["target_is_first"].mean()),
            "baseline_first_pass_at_8": float(df["target_is_first"].mean()),
            "lift_over_first_baseline_pass_at_8": float(df["pass_at_8"].mean() - df["target_is_first"].mean()),
        },
        "n_correct_hist": {str(int(k)): int(v) for k, v in n_correct_hist.items()},
        "manifest": manifest,
    }

    # Optional sampling diagnostics (from plan builder).
    sampling_diag_path = input_dir / "sampling_diagnostics.json"
    if sampling_diag_path.exists():
        try:
            summary["sampling_diagnostics"] = _read_json(sampling_diag_path)
        except Exception:
            summary["sampling_diagnostics"] = {"error": f"Failed to read {sampling_diag_path}"}

    # Correlations (global, k>=2).
    df2 = df[df["k"].astype(int) >= 2].copy()
    corr_global = {
        "top_margin_vs_success_rate": _corr(df2["top_margin"].to_numpy(), df2["success_rate"].to_numpy()).__dict__,
        "log1p_h1_vs_success_rate": _corr(df2["log1p_h1"].to_numpy(), df2["success_rate"].to_numpy()).__dict__,
        "top_margin_vs_pass_at_8": _corr(df2["top_margin"].to_numpy(), df2["pass_at_8"].astype(float).to_numpy()).__dict__,
        "log1p_h1_vs_pass_at_8": _corr(df2["log1p_h1"].to_numpy(), df2["pass_at_8"].astype(float).to_numpy()).__dict__,
    }
    summary["correlations_global_kge2"] = corr_global

    # Stratify by k (to check interaction / dependence on k).
    per_k_rows: list[dict[str, Any]] = []
    for k, g in df2.groupby(df2["k"].astype(int), observed=True):
        if len(g) < 200:
            continue
        per_k_rows.append(
            {
                "k": int(k),
                "n": int(len(g)),
                "corr_margin_success_spearman": float(_corr(g["top_margin"].to_numpy(), g["success_rate"].to_numpy()).spearman_r),
                "corr_logh1_success_spearman": float(_corr(g["log1p_h1"].to_numpy(), g["success_rate"].to_numpy()).spearman_r),
                "success_rate_mean": float(g["success_rate"].mean()),
                "pass_at_8_mean": float(g["pass_at_8"].mean()),
            }
        )
    summary["per_k_summary_kge2"] = per_k_rows

    # Condition on target position (order bias diagnostic).
    # This is a known failure mode for list-selection prompting (LLMs often pick the first item).
    df2["target_is_first"] = df2["target_is_first"].astype(bool)
    df2_first = df2[df2["target_is_first"]]
    df2_not_first = df2[~df2["target_is_first"]]
    summary["target_position_effect_kge2"] = {
        "n_target_first": int(len(df2_first)),
        "n_target_not_first": int(len(df2_not_first)),
        "pass_at_8_target_first": float(df2_first["pass_at_8"].mean()) if len(df2_first) else float("nan"),
        "pass_at_8_target_not_first": float(df2_not_first["pass_at_8"].mean()) if len(df2_not_first) else float("nan"),
        "success_rate_target_first": float(df2_first["success_rate"].mean()) if len(df2_first) else float("nan"),
        "success_rate_target_not_first": float(df2_not_first["success_rate"].mean()) if len(df2_not_first) else float("nan"),
    }

    # Per-row correlations controlling for subset size k by demeaning within (row_id, k).
    # This is a lightweight way to control for both:
    # - position difficulty (row_id fixed effect)
    # - subset-size effects (k fixed effect within each row)
    df2["success_resid_rowk"] = df2["success_rate"] - df2.groupby(["row_id", "k"], observed=True)["success_rate"].transform("mean")
    df2["margin_resid_rowk"] = df2["top_margin"] - df2.groupby(["row_id", "k"], observed=True)["top_margin"].transform("mean")
    df2["logh1_resid_rowk"] = df2["log1p_h1"] - df2.groupby(["row_id", "k"], observed=True)["log1p_h1"].transform("mean")

    per_row: list[dict[str, Any]] = []
    min_subsets_per_row = int(args.min_subsets_per_row)
    for rid, g in df2.groupby(df2["row_id"].astype(int), observed=True):
        if len(g) < min_subsets_per_row:
            continue
        x1 = g["margin_resid_rowk"].to_numpy()
        x2 = g["logh1_resid_rowk"].to_numpy()
        y = g["success_resid_rowk"].to_numpy()
        if float(np.nanstd(x1)) <= 0 or float(np.nanstd(y)) <= 0:
            continue
        c1 = _corr(x1, y)
        c2 = _corr(x2, y)
        per_row.append(
            {
                "row_id": int(rid),
                "n_subsets": int(len(g)),
                "pearson_margin_success": float(c1.pearson_r),
                "spearman_margin_success": float(c1.spearman_r),
                "pearson_logh1_success": float(c2.pearson_r),
                "spearman_logh1_success": float(c2.spearman_r),
            }
        )
    per_row_df = pd.DataFrame(per_row)
    summary["per_row_corr_rows_used"] = int(len(per_row_df))

    rng = np.random.default_rng(int(args.seed))
    iters = int(args.bootstrap_iters)
    if len(per_row_df) > 0:
        # Bootstrap CIs over rows (position-level clusters).
        for key in [
            "pearson_margin_success",
            "spearman_margin_success",
            "pearson_logh1_success",
            "spearman_logh1_success",
        ]:
            mean, lo, hi = _bootstrap_mean_ci(per_row_df[key].to_numpy(), rng=rng, iters=iters)
            summary.setdefault("per_row_corr_bootstrap_ci", {})[key] = {
                "mean": float(mean),
                "ci95_lo": float(lo),
                "ci95_hi": float(hi),
                "bootstrap_iters": int(iters),
            }

    # Logistic models for pass@8 (simple, adjusted for k via log(k); no row FE).
    df_log = df2.copy()
    df_log = df_log[np.isfinite(df_log["log1p_h1"]) & np.isfinite(df_log["top_margin"])].copy()
    df_log["log_k"] = np.log(df_log["k"].astype(float))

    def _fit_logit_one(metric_col: str) -> dict[str, Any]:
        y = df_log["pass_at_8"].astype(float).to_numpy()
        x = _zscore(df_log[metric_col].astype(float).to_numpy())
        lk = _zscore(df_log["log_k"].astype(float).to_numpy())
        X = np.stack([np.ones_like(x), x, lk], axis=1)
        beta, se = _fit_logistic_irls(X, y, ridge=1e-6)
        # Convert coefficient to odds ratio per +1 SD in metric.
        b_metric = float(beta[1])
        se_metric = float(se[1])
        # Wald 95% CI.
        lo = float(b_metric - 1.96 * se_metric)
        hi = float(b_metric + 1.96 * se_metric)
        return {
            "n": int(len(y)),
            "beta_intercept": float(beta[0]),
            "beta_metric_z": b_metric,
            "beta_logk_z": float(beta[2]),
            "se_metric_z": se_metric,
            "odds_ratio_metric_per_1sd": float(math.exp(b_metric)),
            "odds_ratio_metric_ci95_lo": float(math.exp(lo)),
            "odds_ratio_metric_ci95_hi": float(math.exp(hi)),
        }

    summary["logit_pass8_adj_logk_kge2"] = {
        "top_margin": _fit_logit_one("top_margin"),
        "log1p_h1": _fit_logit_one("log1p_h1"),
    }

    # Global residual correlations controlling for row_id and k (more stable than averaging per-row).
    resid_corr = {
        "margin_resid_rowk_vs_success_resid_rowk": _corr(
            df2["margin_resid_rowk"].to_numpy(),
            df2["success_resid_rowk"].to_numpy(),
        ).__dict__,
        "logh1_resid_rowk_vs_success_resid_rowk": _corr(
            df2["logh1_resid_rowk"].to_numpy(),
            df2["success_resid_rowk"].to_numpy(),
        ).__dict__,
    }
    summary["correlations_residual_rowk_kge2"] = resid_corr

    plot_paths = _make_plots(df, out_dir=plots_dir)
    summary["plots"] = plot_paths

    _write_json(out_dir / "summary.json", summary)

    # Write report.md (compact but concrete).
    def fmt_pct(x: float) -> str:
        if not math.isfinite(x):
            return "nan"
        return f"{100.0 * x:.2f}%"

    # Correlation text blocks.
    global_lines = []
    global_lines.append(_format_corr("top_margin vs success_rate (k>=2)", CorrResult(**corr_global["top_margin_vs_success_rate"])))
    global_lines.append(_format_corr("log1p(h1) vs success_rate (k>=2)", CorrResult(**corr_global["log1p_h1_vs_success_rate"])))
    global_lines.append(_format_corr("top_margin vs pass@8 (k>=2)", CorrResult(**corr_global["top_margin_vs_pass_at_8"])))
    global_lines.append(_format_corr("log1p(h1) vs pass@8 (k>=2)", CorrResult(**corr_global["log1p_h1_vs_pass_at_8"])))

    per_row_ci = summary.get("per_row_corr_bootstrap_ci") or {}
    ci_lines = []
    if per_row_ci:
        ci_lines.append(
            "- Within-position, controlling for k by demeaning within (row_id,k), bootstrap CI over rows (95%):"
        )
        for k in [
            "spearman_margin_success",
            "spearman_logh1_success",
            "pearson_margin_success",
            "pearson_logh1_success",
        ]:
            v = per_row_ci.get(k) or {}
            if not v:
                continue
            ci_lines.append(f"  - {k}: mean={v.get('mean', float('nan')):+.3f} CI95=[{v.get('ci95_lo', float('nan')):+.3f}, {v.get('ci95_hi', float('nan')):+.3f}]")

    logit = summary["logit_pass8_adj_logk_kge2"]
    resid = summary["correlations_residual_rowk_kge2"]

    # Hard-gate sanity check evidence (k=1 subsets should be trivially perfect if the prompt enforces selection).
    df_k1 = df[df["k"].astype(int) == 1].copy()
    k1_n = int(len(df_k1))
    k1_success_mean = float(df_k1["success_rate"].mean()) if k1_n else float("nan")
    k1_frac_all_correct = float((df_k1["n_correct"].astype("Int64") == df_k1["n_samples"].astype("Int64")).mean()) if k1_n else float("nan")

    # Sampling diagnostics summary (from subset plan builder).
    sampling = summary.get("sampling_diagnostics") or {}
    sampling_k_counts = sampling.get("k_counts") or {}
    top_k_by_count: list[tuple[int, int]] = []
    if isinstance(sampling_k_counts, dict):
        for k_str, v in sampling_k_counts.items():
            try:
                k_int = int(k_str)
                top_k_by_count.append((k_int, int(v)))
            except Exception:
                continue
        top_k_by_count.sort(key=lambda kv: (-kv[1], kv[0]))
        top_k_by_count = top_k_by_count[:12]

    includes_best_frac_all = float(df["includes_global_best"].astype(bool).mean())
    includes_best_frac_kge2 = float(df2["includes_global_best"].astype(bool).mean()) if len(df2) else float("nan")

    report_lines: list[str] = []
    report_lines.extend(
        [
            "# Chess subset-selection eval (select_prompt): hardness vs pass@8",
            "",
            "This experiment reframes chess move prediction as *selection* from a candidate subset of legal moves.",
            "Each subset is evaluated independently: the model must output a move *from that subset*, and is scored",
            "against the subset-best move by μ (expected score).",
            "",
            "## Artifacts",
            f"- Local cache input_dir: `{input_dir}`",
            f"- Original cluster out_dir: `{args.source_out_dir or ''}`",
            "",
            "## Reproduce",
            "### Cluster eval (Isambard GH200, 4 GPUs)",
            "```bash",
            "sbatch --wait --export=ALL,\\",
            "  MODEL=Qwen/Qwen2.5-7B-Instruct,\\",
            "  LIMIT_ROWS=100,MAX_SUBSETS_PER_ROW=1000,SAMPLES_PER_SUBSET=8,\\",
            "  SEED=0,TEMPERATURE=0.6,TOP_P=0.95 \\",
            "  ./sbatch_eval_chess_select_subsets_gh200.slurm",
            "```",
            "",
            "### Local analysis (this script)",
            "```bash",
            "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \\",
            "  conda run -n verl python scripts/analyze_chess_select_subsets.py \\",
            f"    --input_dir {input_dir} \\",
            f"    --source_out_dir {args.source_out_dir or ''} \\",
            "    --seed 0 --bootstrap_iters 2000",
            "```",
            "",
            "## Prompt + parsing contract",
            "- Prompt template: `recipe/chess/prompt_templates/select_prompt.jinja`",
            "- Required output tags: `<think>...</think><uci_move>...</uci_move>` (no `<guess>` in selection prompt).",
            "- Parsing gate used by evaluator: first `<uci_move>...</uci_move>` span + `recipe/chess/reward_fn.py` (`_to_uci`).",
            "",
            "## Dataset schema (searchless test set)",
            "- Parquet: `data/chess_puzzles/test.parquet`",
            "- Row id: `extra_info.index` (0..999 in this dataset; we used the first 100 rows)",
            "- Position: `reward_model.fen`",
            "- Full legal move list: `reward_model.legal_moves_uci`",
            "- Per-move μ (expected score): `reward_model.move_expected_scores_json`",
            "",
            "## Correctness definition (selection framing)",
            "- μ source: `move_expected_scores_json` (fallback `move_values_json` if missing).",
            "- Target per subset: argmax μ within the candidate subset; tie-break: highest μ then lexicographic UCI.",
            "- Incorrect if: output is not strict-format, not a valid UCI move, or not in the candidate subset.",
            "",
            "## Subset sampling design (`select_subset_sampler_v1`)",
            "- k strata per row: `k ∈ {1,2,3,4,5,8,12,16,24,32,n_legal}` clipped to `[1,n_legal]`.",
            "- Always include the full legal list (k=n) as a baseline subset.",
            "- Allocate subset budget across k with weight ∝ `1/sqrt(k)` (bias towards small k).",
            "- Within each k (k>=2), sample a mix of:",
            "  - easy: include global-best + mostly low-μ distractors",
            "  - hard: include global-best + mostly high-μ near-ties",
            "  - realistic: weighted sampling by softmax(μ/τ), τ=0.2 (no forced inclusion of best)",
            "  - exclude-best: exclude global-best entirely to make target non-trivial when best is absent",
            "- Determinism: per-row seed = `seed + row_id`.",
            "- Candidate *order in the prompt* preserves the original legal-move order (important confound; see below).",
            "",
            "Sampling diagnostics (this run):",
            f"- Planned max_subsets_per_row: {int(manifest.get('max_subsets_per_row') or 0)}",
            f"- Actual subsets: {total_subsets} (avg {per_row_counts.mean():.1f}/row; min {per_row_counts.min()}; median {per_row_counts.median()}; max {per_row_counts.max()})",
            f"- includes_global_best fraction (all k): {includes_best_frac_all:.3f}",
            f"- includes_global_best fraction (k>=2): {includes_best_frac_kge2:.3f}",
        ]
    )
    if top_k_by_count:
        report_lines.append("- Most common k values (count):")
        report_lines.extend([f"  - k={k}: {c}" for k, c in top_k_by_count])
    report_lines.extend(
        [
            "",
            "## Hard gate: one-candidate sanity check",
            "- Requirement: if `considered_moves` contains exactly ONE move, the model must output that move.",
            f"- Evidence (from full eval results): k=1 subsets n={k1_n}, mean success_rate={k1_success_mean:.4f}, frac(all 8 correct)={k1_frac_all_correct:.4f}.",
            "",
            "## Run summary (k includes k=1 unless noted)",
            f"- Rows: {total_rows}",
            f"- Mean success_rate (n_correct/8): {df['success_rate'].mean():.4f}",
            f"- Mean pass@8 (any correct in 8): {df['pass_at_8'].mean():.4f}",
            "",
            "### n_correct distribution (how much pass@8 benefits from sampling)",
            "- This run uses n_samples=8 per subset. Empirically the model is near-deterministic per subset:",
            f"  - frac(n_correct=0) = {float((df['n_correct'].astype('Int64') == 0).mean()):.3f}",
            f"  - frac(n_correct=8) = {float((df['n_correct'].astype('Int64') == 8).mean()):.3f}",
            "",
            "## Compliance (non-compliant outputs are counted incorrect)",
            f"- Mean strict-format-ok rate: {fmt_pct(df['format_ok_rate'].mean())}",
            f"- Mean in-subset rate:        {fmt_pct(df['in_subset_rate'].mean())}",
            f"- Mean bad-move rate:         {fmt_pct(df['bad_move_rate'].mean())}",
            f"- Mean format-error rate:     {fmt_pct(df['format_error_rate'].mean())}",
            "",
            "## Key finding: severe list position bias",
            "This is a critical confound for any selection-from-list prompt: the model strongly prefers the first item.",
            f"- frac(predicted move is the FIRST candidate) = {float(df['pred_is_first'].mean()):.3f}",
            f"- frac(target move is the FIRST candidate)    = {float(df['target_is_first'].mean()):.3f}",
            f"- Baseline pass@8 if you always choose the first candidate = {float(df['target_is_first'].mean()):.3f}",
            f"- Actual pass@8 = {float(df['pass_at_8'].mean()):.3f} (lift over first-item baseline = {float(df['pass_at_8'].mean() - df['target_is_first'].mean()):+.3f})",
            f"- pass@8 | target_first (k>=2)     = {summary['target_position_effect_kge2']['pass_at_8_target_first']:.3f}",
            f"- pass@8 | target_not_first (k>=2) = {summary['target_position_effect_kge2']['pass_at_8_target_not_first']:.3f}",
            "",
            "## Hardness correlations",
            "### Global (k>=2; does NOT control for row difficulty)",
            *global_lines,
            "",
            "### Controls / robustness",
            f"- Rows used for within-position correlations (min_subsets_per_row={min_subsets_per_row}): {len(per_row_df)} / {total_rows}",
            *ci_lines,
            "",
            "Residual correlations (global; controls for row_id and k by demeaning within (row_id,k)):",
            _format_corr("margin_resid(row,k) vs success_resid(row,k)", CorrResult(**resid["margin_resid_rowk_vs_success_resid_rowk"])),
            _format_corr("logh1_resid(row,k) vs success_resid(row,k)", CorrResult(**resid["logh1_resid_rowk_vs_success_resid_rowk"])),
            "",
            "Logistic models for pass@8 (k>=2; adjusted for log(k); no row fixed effects):",
            f"- pass@8 ~ z(top_margin) + z(log(k)): OR(per +1 SD top_margin)={logit['top_margin']['odds_ratio_metric_per_1sd']:.3f} CI95=[{logit['top_margin']['odds_ratio_metric_ci95_lo']:.3f}, {logit['top_margin']['odds_ratio_metric_ci95_hi']:.3f}]",
            f"- pass@8 ~ z(log1p(h1)) + z(log(k)): OR(per +1 SD log1p(h1))={logit['log1p_h1']['odds_ratio_metric_per_1sd']:.3f} CI95=[{logit['log1p_h1']['odds_ratio_metric_ci95_lo']:.3f}, {logit['log1p_h1']['odds_ratio_metric_ci95_hi']:.3f}]",
            "",
            "## Plots",
            *(f"- {k}: `{v}`" for k, v in plot_paths.items()),
            "",
            "## Recommendation",
            "The current framing is *not yet* measuring chess skill cleanly: list position bias dominates.",
            "Before scaling to more rows, run a small eval variant that removes ordering confounds:",
            "- Deterministically shuffle the candidate move order in the prompt per (row_id, subset_id, seed),",
            "  and record the displayed order hash in the cache key.",
            "- Re-run (e.g., 25–50 rows) and re-check: (1) first-item bias, (2) hardness correlations, (3) pass@8 lift over random baseline 1/k.",
            "",
            "If we want pass@8 to be meaningful, also consider increasing diversity (temperature/top_p) or treating the task as",
            "pass@1, because the current output distribution is near-deterministic per subset.",
            "",
        ]
    )

    report = "\n".join(report_lines)
    _write_text(out_dir / "report.md", report + "\n")
    print(f"Wrote {out_dir/'report.md'}")
    print(f"Wrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
