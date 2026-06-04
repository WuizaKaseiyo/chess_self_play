#!/usr/bin/env python3
"""
Visualization-first plot suite for s0anl08n vs dg41tlmo investigation.

Reads tables from:
  analysis/investigation_s0_vs_dg/tables/

Writes figures to:
  analysis/investigation_s0_vs_dg/plots_visual/
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


BASELINE_COLOR = "#d97706"  # amber
ITER_COLOR = "#0f766e"  # teal
DELTA_POS = "#15803d"  # green
DELTA_NEG = "#b91c1c"  # red
GRID_ALPHA = 0.25


PAIR_COLORS = {
    "both_success": "#15803d",
    "iter_only": "#0ea5e9",
    "baseline_only": "#f97316",
    "both_fail": "#6b7280",
}

WINDOW_ORDER = ["early", "mid", "late"]
LEGAL_BIN_ORDER = ["01-15", "16-25", "26-35", "36-45", "46+"]
ROUND_ORDER = [1, 2, 3, 4]
PAIR_ORDER = ["both_success", "iter_only", "baseline_only", "both_fail"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    names = [
        "step_summary.csv",
        "paired_prompt_outcomes.csv",
        "iterative_round_group_summary.csv",
        "iterative_round_window_summary.csv",
        "learning_signal_summary.csv",
        "prompt_success_window_summary.csv",
        "prompt_success_legal_bin_summary.csv",
        "weighted_nonzero_signal_by_step.csv",
        "val_metrics_aligned_steps.csv",
        "fullgame_metrics_aligned_steps.csv",
        "comparability_matrix.csv",
        "hypothesis_tests.csv",
    ]
    out: dict[str, pd.DataFrame] = {}
    for n in names:
        out[n] = pd.read_csv(tables_dir / n)
    return out


def _plot_key_timeseries(step: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(step["step"], step["baseline_prompt_success"], marker="o", color=BASELINE_COLOR, label="baseline")
    ax.plot(step["step"], step["iterative_prompt_success"], marker="o", color=ITER_COLOR, label="iterative")
    ax.set_title("Prompt Success by Step")
    ax.set_xlabel("global_step")
    ax.set_ylabel("prompt success fraction")
    ax.grid(alpha=GRID_ALPHA)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(
        step["step"],
        step["baseline_effective_batch_frac_from_logs"],
        marker="o",
        color=BASELINE_COLOR,
        label="baseline",
    )
    ax.plot(
        step["step"],
        step["iterative_effective_batch_frac_from_logs"],
        marker="o",
        color=ITER_COLOR,
        label="iterative",
    )
    ax.set_title("Effective Batch (std(score)>0) by Step")
    ax.set_xlabel("global_step")
    ax.set_ylabel("effective batch fraction")
    ax.grid(alpha=GRID_ALPHA)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(
        step["step"],
        step["success_delta_iter_minus_base"],
        marker="o",
        color=DELTA_POS,
        label="coverage delta (iter-base)",
    )
    ax.plot(
        step["step"],
        step["effective_batch_delta_iter_minus_base"],
        marker="o",
        color=DELTA_NEG,
        label="signal delta (iter-base)",
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Coverage Gain vs Signal Loss")
    ax.set_xlabel("global_step")
    ax.set_ylabel("delta")
    ax.grid(alpha=GRID_ALPHA)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(
        step["step"],
        step["iterative_forced_prompt_frac"],
        marker="o",
        color="#7c3aed",
        label="forced prompt frac",
    )
    ax2 = ax.twinx()
    ax2.plot(
        step["step"],
        step["iterative_avg_rounds_used"],
        marker="o",
        color="#0f172a",
        label="avg rounds used",
    )
    ax.set_title("Iterative Dynamics")
    ax.set_xlabel("global_step")
    ax.set_ylabel("forced frac")
    ax2.set_ylabel("avg rounds")
    ax.grid(alpha=GRID_ALPHA)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")

    _save(fig, outdir / "01_key_timeseries.png")


def _stacked_bar_from_pairs(
    paired: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    title: str,
    ylabel: str,
    outpath: Path,
) -> None:
    tmp = (
        paired.groupby([group_col, "pair_category"], as_index=False, observed=False)
        .agg(count=("pair_category", "size"))
        .copy()
    )
    piv = tmp.pivot(index=group_col, columns="pair_category", values="count").fillna(0.0)
    piv = piv.reindex(index=group_order, columns=PAIR_ORDER).fillna(0.0)
    frac = piv.div(piv.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(frac), dtype=float)
    x = np.arange(len(frac))

    for cat in PAIR_ORDER:
        vals = frac[cat].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=PAIR_COLORS[cat], label=cat.replace("_", " "))
        bottom += vals

    for i, idx in enumerate(frac.index.tolist()):
        n = int(piv.loc[idx].sum())
        ax.text(i, 1.01, f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(frac.index.tolist())
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")
    _save(fig, outpath)


def _plot_first_success_round_heatmap(paired: pd.DataFrame, outdir: Path) -> None:
    tmp = paired.copy()
    tmp["first_round"] = tmp["first_success_round_iterative"].astype(int)
    tmp["round_label"] = tmp["first_round"].map(
        {0: "fail_all", 1: "r1", 2: "r2", 3: "r3", 4: "r4"}
    )
    piv = (
        tmp.groupby(["legal_bin", "round_label"], as_index=False, observed=False)
        .agg(count=("round_label", "size"))
        .pivot(index="legal_bin", columns="round_label", values="count")
        .fillna(0.0)
    )
    piv = piv.reindex(index=LEGAL_BIN_ORDER, columns=["fail_all", "r1", "r2", "r3", "r4"]).fillna(0.0)
    frac = piv.div(piv.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    mat = frac.to_numpy()
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.35, float(mat.max())))
    ax.set_xticks(np.arange(frac.shape[1]))
    ax.set_xticklabels(frac.columns.tolist())
    ax.set_yticks(np.arange(frac.shape[0]))
    ax.set_yticklabels(frac.index.tolist())
    ax.set_title("Iterative First-Success Round by Legal-Move Bin")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("fraction within legal bin")
    _save(fig, outdir / "04_first_success_round_heatmap_legalbin.png")


def _plot_round_dynamics(round_df: pd.DataFrame, outdir: Path) -> None:
    r = round_df.sort_values("allowed_move_elim_round").copy()
    x = r["allowed_move_elim_round"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.bar(x, r["groups"], color="#334155")
    ax.set_title("Iterative Groups per Round")
    ax.set_xlabel("round")
    ax.set_ylabel("groups")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    ax = axes[0, 1]
    ax.plot(x, r["success_group_frac"], marker="o", color=ITER_COLOR, label="success group frac")
    ax.plot(x, r["forced_accept_group_frac"], marker="o", color="#7c3aed", label="forced-accept frac")
    ax.plot(x, r["accepted_group_frac"], marker="o", color="#111827", label="accepted frac")
    ax.set_title("Acceptance Dynamics by Round")
    ax.set_xlabel("round")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=GRID_ALPHA)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(x, r["std_nonzero_frac"], marker="o", color="#0ea5e9", label="std_nonzero")
    ax2 = ax.twinx()
    ax2.plot(x, r["score_range_mean"], marker="o", color="#ef4444", label="mean score range")
    ax.set_title("Signal Strength by Round")
    ax.set_xlabel("round")
    ax.set_ylabel("std_nonzero frac")
    ax2.set_ylabel("mean score range")
    ax.grid(alpha=GRID_ALPHA)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")

    ax = axes[1, 1]
    ax.plot(x, r["mean_b_size"], marker="o", color="#0f172a")
    ax.set_title("Mean Candidate-Set Size by Round")
    ax.set_xlabel("round")
    ax.set_ylabel("mean b size")
    ax.grid(alpha=GRID_ALPHA)

    _save(fig, outdir / "05_round_dynamics_iterative.png")


def _plot_round_window_heatmaps(round_window_df: pd.DataFrame, outdir: Path) -> None:
    d = round_window_df.copy()
    d["window"] = pd.Categorical(d["window"], categories=WINDOW_ORDER, ordered=True)
    d = d.sort_values(["window", "allowed_move_elim_round"])

    p1 = d.pivot(index="window", columns="allowed_move_elim_round", values="success_group_frac").reindex(WINDOW_ORDER)
    p2 = d.pivot(index="window", columns="allowed_move_elim_round", values="std_nonzero_frac").reindex(WINDOW_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, piv, title in [
        (axes[0], p1, "Round Success Fraction (iterative)"),
        (axes[1], p2, "Round std_nonzero Fraction (iterative)"),
    ]:
        mat = piv.to_numpy()
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.55, float(np.nanmax(mat))))
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels([str(c) for c in piv.columns.tolist()])
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels(piv.index.tolist())
        ax.set_xlabel("round")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("fraction")
    _save(fig, outdir / "06_round_window_heatmaps.png")


def _plot_learning_signal_subsets(learning: pd.DataFrame, outdir: Path) -> None:
    d = learning.copy()
    d["window"] = pd.Categorical(d["window"], categories=WINDOW_ORDER, ordered=True)
    d = d.sort_values(["subset", "window", "run"])
    subset_order = ["all_groups", "successful_groups", "groups_from_successful_prompts"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    bar_w = 0.36
    x = np.arange(len(WINDOW_ORDER))

    for i, subset in enumerate(subset_order):
        sub = d[d["subset"] == subset]
        b = sub[sub["run"] == "baseline"].set_index("window").reindex(WINDOW_ORDER)
        it = sub[sub["run"] == "iterative"].set_index("window").reindex(WINDOW_ORDER)

        ax = axes[0, i]
        ax.bar(x - bar_w / 2, b["std_nonzero_frac"], width=bar_w, color=BASELINE_COLOR, label="baseline")
        ax.bar(x + bar_w / 2, it["std_nonzero_frac"], width=bar_w, color=ITER_COLOR, label="iterative")
        ax.set_title(subset)
        ax.set_ylabel("std_nonzero frac")
        ax.set_xticks(x)
        ax.set_xticklabels(WINDOW_ORDER)
        ax.grid(axis="y", alpha=GRID_ALPHA)
        if i == 0:
            ax.legend()

        ax2 = axes[1, i]
        ax2.bar(x - bar_w / 2, b["score_range_mean"], width=bar_w, color=BASELINE_COLOR)
        ax2.bar(x + bar_w / 2, it["score_range_mean"], width=bar_w, color=ITER_COLOR)
        ax2.set_ylabel("mean score range")
        ax2.set_xticks(x)
        ax2.set_xticklabels(WINDOW_ORDER)
        ax2.grid(axis="y", alpha=GRID_ALPHA)

    fig.suptitle("Learning-Signal Proxies by Subset and Window", y=1.02, fontsize=14)
    _save(fig, outdir / "07_learning_signal_subsets.png")


def _plot_eval_alignment(val_df: pd.DataFrame, full_df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    if not val_df.empty:
        v = val_df.sort_values("global_step")
        ax = axes[0]
        ax.plot(v["global_step"], v["dg_val_acc"], marker="o", color=BASELINE_COLOR, label="baseline")
        ax.plot(v["global_step"], v["s0_val_acc"], marker="o", color=ITER_COLOR, label="iterative")
        ax.bar(
            v["global_step"],
            v["delta_val_acc"],
            alpha=0.25,
            color=[DELTA_POS if x >= 0 else DELTA_NEG for x in v["delta_val_acc"]],
            label="delta (iter-base)",
        )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Puzzle Eval (Aligned Steps)")
        ax.set_xlabel("global_step")
        ax.set_ylabel("val acc")
        ax.grid(alpha=GRID_ALPHA)
        ax.legend()

    if not full_df.empty:
        f = full_df.sort_values("global_step")
        ax = axes[1]
        ax.plot(f["global_step"], f["dg_acpl_per_move"], marker="o", color=BASELINE_COLOR, label="baseline")
        ax.plot(f["global_step"], f["s0_acpl_per_move"], marker="o", color=ITER_COLOR, label="iterative")
        ax.bar(
            f["global_step"],
            f["delta_acpl_per_move"],
            alpha=0.25,
            color=[DELTA_POS if x <= 0 else DELTA_NEG for x in f["delta_acpl_per_move"]],
            label="delta (iter-base)",
        )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Full-Game Eval (ACPL/move, lower is better)")
        ax.set_xlabel("global_step")
        ax.set_ylabel("ACPL/move")
        ax.grid(alpha=GRID_ALPHA)
        ax.legend()

    _save(fig, outdir / "08_eval_alignment_and_deltas.png")


def _plot_comparability_heatmap(comp: pd.DataFrame, outdir: Path) -> None:
    d = comp.copy()
    d["same_int"] = d["same"].astype(int)
    # Most important dimensions first: differences on top.
    d = d.sort_values(["same_int", "dimension"], ascending=[True, True]).reset_index(drop=True)

    mat = d[["same_int"]].to_numpy()
    fig_h = max(7.5, 0.26 * len(d))
    fig, ax = plt.subplots(figsize=(7.8, fig_h))
    cmap = ListedColormap(["#fca5a5", "#86efac"])  # red=diff, green=same
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([0])
    ax.set_xticklabels(["same?"])
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["dimension"].tolist(), fontsize=9)
    ax.set_title("Comparability Map (green=same, red=different)")
    for i, v in enumerate(d["same_int"].tolist()):
        ax.text(0, i, "same" if v == 1 else "diff", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["different", "same"])
    _save(fig, outdir / "09_comparability_heatmap.png")


def _plot_hypothesis_verdicts(hyp: pd.DataFrame, outdir: Path) -> None:
    d = hyp.copy()
    mapping = {"supported": 1, "inconclusive": 0, "unsupported": -1}
    d["score"] = d["verdict"].map(mapping).fillna(0).astype(int)
    d["label"] = d["hypothesis_id"] + ": " + d["title"].map(lambda s: textwrap.shorten(str(s), width=64, placeholder="..."))
    colors = [DELTA_POS if s > 0 else (DELTA_NEG if s < 0 else "#64748b") for s in d["score"]]

    fig_h = max(4.6, 0.56 * len(d))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    y = np.arange(len(d))
    ax.barh(y, d["score"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=9)
    ax.set_xlim(-1.25, 1.25)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["unsupported", "inconclusive", "supported"])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Hypothesis Verdicts")
    ax.grid(axis="x", alpha=GRID_ALPHA)
    _save(fig, outdir / "10_hypothesis_verdicts.png")


def _plot_tradeoff_frontier(step: pd.DataFrame, outdir: Path) -> None:
    d = step.copy()
    color_map = {"early": "#3b82f6", "mid": "#f59e0b", "late": "#ef4444"}
    colors = d["window"].map(color_map).fillna("#6b7280")
    sizes = 140 + 520 * d["iterative_forced_prompt_frac"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    ax.scatter(
        d["success_delta_iter_minus_base"],
        d["effective_batch_delta_iter_minus_base"],
        c=colors,
        s=sizes,
        alpha=0.82,
        edgecolor="black",
        linewidth=0.6,
    )
    for _, r in d.iterrows():
        ax.text(
            r["success_delta_iter_minus_base"] + 0.002,
            r["effective_batch_delta_iter_minus_base"] + 0.004,
            str(int(r["step"])),
            fontsize=8,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("prompt-success delta (iter-base)")
    ax.set_ylabel("effective-batch delta (iter-base)")
    ax.set_title("Coverage-Signal Tradeoff Frontier by Step")
    ax.grid(alpha=GRID_ALPHA)

    # manual legend
    for w, c in color_map.items():
        ax.scatter([], [], c=c, s=100, label=w)
    ax.legend(title="window")
    _save(fig, outdir / "11_tradeoff_frontier.png")


def _plot_visual_abstract(step: pd.DataFrame, paired: pd.DataFrame, outdir: Path) -> None:
    overall_delta_success = float(step["success_delta_iter_minus_base"].mean())
    overall_delta_signal = float(step["effective_batch_delta_iter_minus_base"].mean())
    overall_forced = float(step["iterative_forced_prompt_frac"].mean())
    counts = paired["pair_category"].value_counts()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    ax = axes[0]
    vals = [step["baseline_prompt_success"].mean(), step["iterative_prompt_success"].mean()]
    ax.bar(["baseline", "iterative"], vals, color=[BASELINE_COLOR, ITER_COLOR])
    ax.set_ylim(0, 1)
    ax.set_title("Prompt Success")
    ax.text(0.5, max(vals) + 0.03, f"+{overall_delta_success:.3f}", ha="center", fontsize=12, color=DELTA_POS)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    ax = axes[1]
    vals = [step["baseline_effective_batch_frac_from_logs"].mean(), step["iterative_effective_batch_frac_from_logs"].mean()]
    ax.bar(["baseline", "iterative"], vals, color=[BASELINE_COLOR, ITER_COLOR])
    ax.set_ylim(0, 1)
    ax.set_title("Effective Batch (std>0)")
    sign = "+" if overall_delta_signal >= 0 else ""
    ax.text(0.5, max(vals) + 0.03, f"{sign}{overall_delta_signal:.3f}", ha="center", fontsize=12, color=DELTA_NEG)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    ax = axes[2]
    labels = ["both_success", "iter_only", "baseline_only", "both_fail"]
    vals = [int(counts.get(k, 0)) for k in labels]
    ax.bar(labels, vals, color=[PAIR_COLORS[k] for k in labels])
    ax.set_title("Paired Prompt Outcomes")
    ax.tick_params(axis="x", rotation=20)
    ax.text(
        0.98,
        0.95,
        f"iter forced prompt frac = {overall_forced:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    ax.grid(axis="y", alpha=GRID_ALPHA)

    fig.suptitle("Visual Abstract: Why Gains Look Marginal", y=1.02, fontsize=14)
    _save(fig, outdir / "12_visual_abstract.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables_dir", default="analysis/investigation_s0_vs_dg/tables")
    ap.add_argument("--out_dir", default="analysis/investigation_s0_vs_dg/plots_visual")
    args = ap.parse_args()

    tables_dir = Path(args.tables_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    t = _load_tables(tables_dir)
    step = t["step_summary.csv"].copy().sort_values("step")
    paired = t["paired_prompt_outcomes.csv"].copy()
    paired["window"] = pd.Categorical(paired["window"], categories=WINDOW_ORDER, ordered=True)
    paired["legal_bin"] = pd.Categorical(paired["legal_bin"], categories=LEGAL_BIN_ORDER, ordered=True)

    _plot_key_timeseries(step, out_dir)
    _stacked_bar_from_pairs(
        paired,
        group_col="window",
        group_order=WINDOW_ORDER,
        title="Paired Prompt Outcomes by Training Window",
        ylabel="fraction of prompts",
        outpath=out_dir / "02_pair_outcome_stacked_by_window.png",
    )
    _stacked_bar_from_pairs(
        paired,
        group_col="legal_bin",
        group_order=LEGAL_BIN_ORDER,
        title="Paired Prompt Outcomes by Legal-Move Bin",
        ylabel="fraction of prompts",
        outpath=out_dir / "03_pair_outcome_stacked_by_legal_bin.png",
    )
    _plot_first_success_round_heatmap(paired, out_dir)
    _plot_round_dynamics(t["iterative_round_group_summary.csv"], out_dir)
    _plot_round_window_heatmaps(t["iterative_round_window_summary.csv"], out_dir)
    _plot_learning_signal_subsets(t["learning_signal_summary.csv"], out_dir)
    _plot_eval_alignment(t["val_metrics_aligned_steps.csv"], t["fullgame_metrics_aligned_steps.csv"], out_dir)
    _plot_comparability_heatmap(t["comparability_matrix.csv"], out_dir)
    _plot_hypothesis_verdicts(t["hypothesis_tests.csv"], out_dir)
    _plot_tradeoff_frontier(step, out_dir)
    _plot_visual_abstract(step, paired, out_dir)

    print(f"[OK] Wrote visualization suite to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
