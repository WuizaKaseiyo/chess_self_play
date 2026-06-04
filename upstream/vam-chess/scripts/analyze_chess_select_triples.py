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
        "triple_unordered_id",
        "triple_id",
        "candidate_moves_uci",
        "target_move_uci",
        "target_pos",
        "top_margin",
        "h1",
        "n_samples",
        "n_correct",
        "n_in_subset",
        "n_format_ok",
        "n_pred_pos0",
        "n_pred_pos1",
        "n_pred_pos2",
        "n_pred_other",
    ]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns in results: {missing}")

    for c in (
        "row_id",
        "target_pos",
        "n_samples",
        "n_correct",
        "n_in_subset",
        "n_format_ok",
        "n_pred_pos0",
        "n_pred_pos1",
        "n_pred_pos2",
        "n_pred_other",
    ):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    out["top_margin"] = pd.to_numeric(out["top_margin"], errors="coerce").astype(float)
    out["h1"] = pd.to_numeric(out["h1"], errors="coerce").astype(float)

    dup_mask = out.duplicated(subset=["row_id", "triple_id"], keep=False)
    if bool(dup_mask.any()):
        dup_n = int(dup_mask.sum())
        sample = out.loc[dup_mask, ["row_id", "triple_id", "__shard_file__"]].head(10).to_dict(orient="records")
        raise ValueError(f"Found {dup_n} duplicate (row_id, triple_id) records. Sample:\n{json.dumps(sample, indent=2)}")

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


def _make_plots(triple_df: pd.DataFrame, *, out_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(triple_df["top_margin"].astype(float), bins=80)
    ax.set_xlabel("top margin = μ_best - μ_second (within triple)")
    ax.set_ylabel("num triples")
    ax.set_title("Top-margin distribution across sampled 3-move triples")
    fig.tight_layout()
    p = out_dir / "top_margin_hist.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["top_margin_hist"] = str(p)

    # Binned success vs margin (averaged across permutations).
    b = _binned_means(triple_df, "top_margin", ["p_correct_avg"], bins=30)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(b["x_mean"], b["p_correct_avg_mean"], marker="o", markersize=3, linewidth=1)
    ax.set_xlabel("top_margin (binned; x=mean margin)")
    ax.set_ylabel("mean P(select subset-best)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("3-move selection: P(best) vs top-margin (averaged over 6 permutations)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = out_dir / "pbest_vs_margin_binned.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["pbest_vs_margin_binned"] = str(p)

    # Position-conditioned binned success (target pos 0/1/2).
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    for pos in (0, 1, 2):
        col = f"p_correct_pos{pos}"
        bpos = _binned_means(triple_df, "top_margin", [col], bins=30)
        ax.plot(bpos["x_mean"], bpos[f"{col}_mean"], marker="o", markersize=3, linewidth=1, label=f"P(best | best at pos {pos})")
    ax.set_xlabel("top_margin (binned; x=mean margin)")
    ax.set_ylabel("mean P(select subset-best)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("3-move selection: P(best) vs top-margin, conditioned on best position")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    p = out_dir / "pbest_vs_margin_by_pos_binned.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["pbest_vs_margin_by_pos_binned"] = str(p)

    # Position-bias metric vs margin: Δ = P(best | pos2) - P(best | pos0).
    b2 = _binned_means(triple_df, "top_margin", ["pos2_minus_pos0"], bins=30)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(b2["x_mean"], b2["pos2_minus_pos0_mean"], marker="o", markersize=3, linewidth=1)
    ax.set_xlabel("top_margin (binned; x=mean margin)")
    ax.set_ylabel("mean ΔP = P(best|pos2) - P(best|pos0)")
    ax.set_title("3-move: last-vs-first advantage vs margin (binned)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    p = out_dir / "pos2_minus_pos0_vs_margin_binned.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["pos2_minus_pos0_vs_margin_binned"] = str(p)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing manifest.json + results_shard*.jsonl")
    ap.add_argument("--out_dir", default=None, help="Output directory for plots + report. Defaults under analysis/select_triples_eval_reports/<run_id>/")
    ap.add_argument("--source_out_dir", default=None, help="Optional: original cluster path to include in the report.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap_iters", type=int, default=2000)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Missing input_dir: {input_dir}")

    run_id = input_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else Path("analysis") / "select_triples_eval_reports" / run_id
    plots_dir = out_dir / "plots"

    df, manifest = _load_results(input_dir)

    df["p_correct"] = df["n_correct"].astype(float) / df["n_samples"].astype(float)
    df["p_in_subset"] = df["n_in_subset"].astype(float) / df["n_samples"].astype(float)
    df["p_format_ok"] = df["n_format_ok"].astype(float) / df["n_samples"].astype(float)
    df["p_pred_pos0"] = df["n_pred_pos0"].astype(float) / df["n_samples"].astype(float)
    df["p_pred_pos1"] = df["n_pred_pos1"].astype(float) / df["n_samples"].astype(float)
    df["p_pred_pos2"] = df["n_pred_pos2"].astype(float) / df["n_samples"].astype(float)

    # Group across permutations for each unordered triple.
    key_cols = ["row_id", "triple_unordered_id"]
    g = df.groupby(key_cols, observed=True)

    triples = g.agg(
        top_margin=("top_margin", "first"),
        h1=("h1", "first"),
        n_orders=("triple_id", "size"),
        p_correct_avg=("p_correct", "mean"),
        p_format_ok_avg=("p_format_ok", "mean"),
        p_in_subset_avg=("p_in_subset", "mean"),
    ).reset_index()

    # Target-position specific averages.
    pos_avgs = []
    for pos in (0, 1, 2):
        sub = df[df["target_pos"].astype(int) == pos].groupby(key_cols, observed=True)["p_correct"].mean().reset_index()
        sub = sub.rename(columns={"p_correct": f"p_correct_pos{pos}"})
        pos_avgs.append(sub)
    for sub in pos_avgs:
        triples = triples.merge(sub, on=key_cols, how="left")

    # Position-bias metric: last vs first (best at pos2 vs pos0).
    triples["pos2_minus_pos0"] = triples["p_correct_pos2"] - triples["p_correct_pos0"]

    # Correlations: margin vs performance (averaged over permutations = controlled for order).
    corr_margin_pavg = _corr(triples["top_margin"].to_numpy(), triples["p_correct_avg"].to_numpy())
    corr_margin_pos0 = _corr(triples["top_margin"].to_numpy(), triples["p_correct_pos0"].to_numpy())
    corr_margin_pos1 = _corr(triples["top_margin"].to_numpy(), triples["p_correct_pos1"].to_numpy())
    corr_margin_pos2 = _corr(triples["top_margin"].to_numpy(), triples["p_correct_pos2"].to_numpy())
    corr_margin_posbias = _corr(triples["top_margin"].to_numpy(), triples["pos2_minus_pos0"].to_numpy())

    # Within-row correlations (control for per-position difficulty).
    per_row = []
    for rid, gr in triples.groupby(triples["row_id"].astype(int), observed=True):
        if len(gr) < 50:
            continue
        per_row.append(
            {
                "row_id": int(rid),
                "n_triples": int(len(gr)),
                "spearman_margin_pavg": float(_corr(gr["top_margin"].to_numpy(), gr["p_correct_avg"].to_numpy()).spearman_r),
                "spearman_margin_posbias": float(_corr(gr["top_margin"].to_numpy(), gr["pos2_minus_pos0"].to_numpy()).spearman_r),
            }
        )
    per_row_df = pd.DataFrame(per_row)

    rng = np.random.default_rng(int(args.seed))
    iters = int(args.bootstrap_iters)
    per_row_ci: dict[str, Any] = {}
    if len(per_row_df) > 0:
        for col in ("spearman_margin_pavg", "spearman_margin_posbias"):
            m, lo, hi = _bootstrap_mean_ci(per_row_df[col].to_numpy(), rng=rng, iters=iters)
            per_row_ci[col] = {"mean": float(m), "ci95_lo": float(lo), "ci95_hi": float(hi), "bootstrap_iters": int(iters)}

    plots = _make_plots(triples, out_dir=plots_dir)

    total_rows = int(df["row_id"].nunique())
    total_ordered = int(len(df))
    total_unordered = int(len(triples))
    mean_orders = float(triples["n_orders"].astype(float).mean()) if len(triples) else float("nan")

    # Summary stats.
    overall = {
        "p_correct_avg_mean": float(triples["p_correct_avg"].mean()),
        "p_correct_pos0_mean": float(triples["p_correct_pos0"].mean()),
        "p_correct_pos1_mean": float(triples["p_correct_pos1"].mean()),
        "p_correct_pos2_mean": float(triples["p_correct_pos2"].mean()),
        "pos2_minus_pos0_mean": float(triples["pos2_minus_pos0"].mean()),
        "p_in_subset_avg_mean": float(triples["p_in_subset_avg"].mean()),
        "p_format_ok_avg_mean": float(triples["p_format_ok_avg"].mean()),
    }

    summary = {
        "input_dir": str(input_dir),
        "source_out_dir": str(args.source_out_dir or ""),
        "manifest": manifest,
        "total_rows": total_rows,
        "total_triples_ordered": total_ordered,
        "total_triples_unordered": total_unordered,
        "mean_orders_per_unordered_triple": mean_orders,
        "overall_means": overall,
        "correlations_unordered_triple_level": {
            "margin_vs_p_correct_avg": corr_margin_pavg.__dict__,
            "margin_vs_p_correct_pos0": corr_margin_pos0.__dict__,
            "margin_vs_p_correct_pos1": corr_margin_pos1.__dict__,
            "margin_vs_p_correct_pos2": corr_margin_pos2.__dict__,
            "margin_vs_pos2_minus_pos0": corr_margin_posbias.__dict__,
        },
        "within_row_spearman_bootstrap_ci": per_row_ci,
        "plots": plots,
    }

    # Markdown report.
    def fmt_pct(x: float) -> str:
        if not math.isfinite(float(x)):
            return "nan"
        return f"{100.0 * float(x):.2f}%"

    report_lines: list[str] = []
    report_lines.extend(
        [
            "# Chess subset-selection eval (k=3 triples): gap vs position bias",
            "",
            "This experiment evaluates *3-move selection* (k=3) under the selection framing:",
            "given a FEN and a 3-move candidate list, the model must output the best move among the candidates.",
            "",
            "Each unordered triple is evaluated under all 6 permutations, so we can separate:",
            "- intrinsic difficulty (averaged over permutations), and",
            "- position/order bias (differences across permutations).",
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
            "  LIMIT_ROWS=100,MAX_TRIPLES_PER_ROW=200,SAMPLES_PER_TRIPLE=8,\\",
            "  SEED=0,TEMPERATURE=0.6,TOP_P=0.95 \\",
            "  ./sbatch_eval_chess_select_triples_gh200.slurm",
            "```",
            "",
            "### Local analysis (this script)",
            "```bash",
            "conda run -n verl python scripts/analyze_chess_select_triples.py \\",
            f"  --input_dir {input_dir} \\",
            f"  --source_out_dir {args.source_out_dir or ''} \\",
            "  --seed 0 --bootstrap_iters 2000",
            "```",
            "",
            "## Summary",
            f"- Rows: {total_rows}",
            f"- Unordered triples: {total_unordered} (mean orders observed per triple: {mean_orders:.2f}; expected 6.0)",
            "",
            "### Performance (averaged over permutations)",
            f"- Mean P(select subset-best) = {fmt_pct(overall['p_correct_avg_mean'])}",
            f"- Mean in-subset rate = {fmt_pct(overall['p_in_subset_avg_mean'])}",
            f"- Mean strict-format-ok rate = {fmt_pct(overall['p_format_ok_avg_mean'])}",
            "",
            "### Position bias (best-at-position)",
            f"- P(best | best at pos 0) = {fmt_pct(overall['p_correct_pos0_mean'])}",
            f"- P(best | best at pos 1) = {fmt_pct(overall['p_correct_pos1_mean'])}",
            f"- P(best | best at pos 2) = {fmt_pct(overall['p_correct_pos2_mean'])}",
            f"- Mean (pos2 - pos0) = {fmt_pct(overall['pos2_minus_pos0_mean'])}",
            "",
            "### Correlations (unordered-triple level)",
            _format_corr("top_margin vs P(best) (avg over permutations)", corr_margin_pavg),
            _format_corr("top_margin vs P(best | best at pos0)", corr_margin_pos0),
            _format_corr("top_margin vs P(best | best at pos1)", corr_margin_pos1),
            _format_corr("top_margin vs P(best | best at pos2)", corr_margin_pos2),
            _format_corr("top_margin vs (pos2 - pos0)", corr_margin_posbias),
            "",
        ]
    )
    if per_row_ci:
        report_lines.append("### Within-row (position-controlled) Spearman means (bootstrap 95% CI)")
        for k, v in per_row_ci.items():
            report_lines.append(f"- {k}: mean={v['mean']:+.3f} (95% CI [{v['ci95_lo']:+.3f}, {v['ci95_hi']:+.3f}])")
        report_lines.append("")
    report_lines.extend(
        [
            "## Plots",
            f"- top_margin histogram: `{plots.get('top_margin_hist','')}`",
            f"- P(best) vs margin: `{plots.get('pbest_vs_margin_binned','')}`",
            f"- P(best) vs margin by best position: `{plots.get('pbest_vs_margin_by_pos_binned','')}`",
            f"- (pos2 - pos0) vs margin: `{plots.get('pos2_minus_pos0_vs_margin_binned','')}`",
            "",
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_text(out_dir / "report.md", "\n".join(report_lines) + "\n")
    _write_json(out_dir / "summary.json", summary)
    print(f"Wrote {out_dir / 'report.md'}")
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
