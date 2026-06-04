#!/usr/bin/env python3
"""
Targeted effective-batch diagnosis for:
  - baseline run: dg41tlmo
  - iterative run: s0anl08n

Outputs:
  - per-step dead-group composition/quality tables
  - diversity vs effective-batch tables
  - focused plots (13..17) under plots_visual/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collect_baseline(evidence_root: Path, steps: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for step in steps:
        fp = evidence_root / "dg41tlmo" / "files" / "rollout_logs" / f"{step}.jsonl"
        for row in _load_jsonl(fp):
            row["step"] = int(step)
            rows.append(row)
    if not rows:
        raise RuntimeError("No baseline rows loaded.")
    df = pd.DataFrame(rows)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["penalty_applied"] = df.get("penalty_applied", False).fillna(False).astype(bool)
    df["in_subset"] = df.get("in_subset", True).fillna(True).astype(bool)
    df["pred_move"] = df.get("pred_move", "").fillna("").astype(str)
    df["gt_uci"] = df.get("gt_uci", "").fillna("").astype(str)
    df["uid"] = df["uid"].astype(str)
    df["success_sample"] = (df["pred_move"] == df["gt_uci"]) & (~df["penalty_applied"]) & df["in_subset"]
    return df


def _collect_iterative(evidence_root: Path, steps: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for step in steps:
        for round_idx in (1, 2, 3, 4):
            fp = evidence_root / "s0anl08n" / "files" / "allowed_move_elim_rounds" / f"{step}_round{round_idx}.jsonl"
            for row in _load_jsonl(fp):
                row["step"] = int(step)
                rows.append(row)
    if not rows:
        raise RuntimeError("No iterative rows loaded.")
    df = pd.DataFrame(rows)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    for col, default in [
        ("penalty_applied", False),
        ("in_subset", True),
        ("allowed_move_elim_accepted", False),
        ("allowed_move_elim_success", False),
        ("allowed_move_elim_forced_accept", False),
    ]:
        df[col] = df.get(col, default).fillna(default).astype(bool)
    df["pred_move"] = df.get("pred_move", "").fillna("").astype(str)
    df["gt_uci"] = df.get("gt_uci", "").fillna("").astype(str)
    df["allowed_move_elim_round"] = pd.to_numeric(df["allowed_move_elim_round"], errors="coerce").astype(int)
    df["allowed_move_elim_prompt_idx"] = pd.to_numeric(df["allowed_move_elim_prompt_idx"], errors="coerce").astype(int)
    df["success_sample"] = (df["pred_move"] == df["gt_uci"]) & (~df["penalty_applied"]) & df["in_subset"]
    return df


def _group_baseline(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["step", "uid"], as_index=False)
        .agg(
            score_std=("score", "std"),
            score_min=("score", "min"),
            success_any=("success_sample", "max"),
            success_all=("success_sample", "min"),
            penalty_frac=("penalty_applied", "mean"),
            uniq_scores=("score", "nunique"),
            uniq_moves=("pred_move", "nunique"),
        )
        .copy()
    )
    g["score_std"] = g["score_std"].fillna(0.0)
    g["dead"] = g["score_std"] <= 0
    return g


def _group_iterative(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["step", "allowed_move_elim_round", "allowed_move_elim_prompt_idx"], as_index=False)
        .agg(
            score_std=("score", "std"),
            score_min=("score", "min"),
            success_any=("success_sample", "max"),
            success_all=("success_sample", "min"),
            penalty_frac=("penalty_applied", "mean"),
            accepted=("allowed_move_elim_accepted", "max"),
            forced=("allowed_move_elim_forced_accept", "max"),
            uniq_scores=("score", "nunique"),
            uniq_moves=("pred_move", "nunique"),
        )
        .copy()
    )
    g["score_std"] = g["score_std"].fillna(0.0)
    g["dead"] = g["score_std"] <= 0
    return g


def _classify_dead_constant(score_value: float) -> str:
    eps = 1e-12
    if pd.isna(score_value):
        return "nan"
    if abs(float(score_value) + 1.0) <= eps:
        return "all_-1"
    if abs(float(score_value)) <= eps:
        return "all_0"
    if -1.0 < float(score_value) < 0.0:
        return "all_neg_nonpen"
    if float(score_value) > 0.0:
        return "all_pos"
    return "other"


def _dead_class_by_step(group_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    dead = group_df[group_df["dead"]].copy()
    dead["dead_class"] = dead["score_min"].map(_classify_dead_constant)
    out = dead.groupby(["step", "dead_class"], as_index=False).size()
    totals = dead.groupby("step", as_index=False).size().rename(columns={"size": "dead_groups"})
    out = out.merge(totals, on="step", how="left")
    out["frac_within_dead"] = out["size"] / out["dead_groups"]
    out.to_csv(out_path, index=False)
    return out


def _dead_quality_by_step(group_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    dead = group_df[group_df["dead"]].copy()
    out = (
        dead.groupby("step", as_index=False)
        .agg(
            dead_groups=("dead", "size"),
            dead_success_any_frac=("success_any", "mean"),
            dead_success_all_frac=("success_all", "mean"),
            dead_penalty_sample_frac=("penalty_frac", "mean"),
            dead_constant_score_mean=("score_min", "mean"),
        )
        .copy()
    )
    out.to_csv(out_path, index=False)
    return out


def _merge_diagnosis(
    group_df: pd.DataFrame,
    dead_class_df: pd.DataFrame,
    dead_quality_df: pd.DataFrame,
    eff_col_name: str,
) -> pd.DataFrame:
    step_eff = (
        group_df.groupby("step", as_index=False)
        .agg(effective_batch_frac=("dead", lambda x: 1.0 - float(x.mean())))
        .rename(columns={"effective_batch_frac": eff_col_name})
    )

    class_piv = (
        dead_class_df.pivot(index="step", columns="dead_class", values="frac_within_dead")
        .fillna(0.0)
        .reset_index()
    )
    for cls in ("all_-1", "all_neg_nonpen", "all_0"):
        if cls not in class_piv.columns:
            class_piv[cls] = 0.0

    merged = step_eff.merge(dead_quality_df, on="step", how="left").merge(class_piv, on="step", how="left")
    dead_frac = 1.0 - merged[eff_col_name]
    merged["all_-1"] = merged["all_-1"] * dead_frac
    merged["all_neg_nonpen"] = merged["all_neg_nonpen"] * dead_frac
    merged["all_0"] = merged["all_0"] * dead_frac
    return merged.sort_values("step").reset_index(drop=True)


def _window_col(step_series: pd.Series) -> pd.Series:
    return pd.cut(step_series, bins=[0, 120, 240, 400], labels=["early", "mid", "late"])


def _plot_effective_decomposition(baseline_diag: pd.DataFrame, iterative_diag: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    for ax, df, title, eff_col in [
        (axes[0], baseline_diag, "Baseline (dg41tlmo)", "baseline_effective_batch_frac_from_logs"),
        (axes[1], iterative_diag, "Iterative (s0anl08n, all rounds)", "iterative_effective_batch_frac_from_logs"),
    ]:
        x = df["step"]
        ax.stackplot(
            x,
            df["all_-1"],
            df["all_neg_nonpen"],
            df["all_0"],
            labels=["dead all -1", "dead all neg(non-pen)", "dead all 0"],
            colors=["#b91c1c", "#f59e0b", "#0ea5e9"],
            alpha=0.62,
        )
        ax.plot(x, df[eff_col], marker="o", color="black", linewidth=2, label="effective_batch_frac (std>0)")
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("fraction of groups")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", ncol=2, fontsize=8)

    axes[1].set_xlabel("global_step")
    fig.suptitle("Effective-Batch Collapse Decomposition by Step", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_dead_makeup_within_dead(
    step_index: list[int],
    baseline_dead_cls: pd.DataFrame,
    iterative_dead_cls: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

    def _one(ax: plt.Axes, cls_df: pd.DataFrame, title: str) -> None:
        piv = cls_df.pivot(index="step", columns="dead_class", values="frac_within_dead").fillna(0.0)
        piv = piv.reindex(step_index).fillna(0.0)
        ax.plot(step_index, piv.get("all_-1", 0.0), marker="o", label="all -1", color="#b91c1c")
        ax.plot(step_index, piv.get("all_neg_nonpen", 0.0), marker="o", label="all neg(non-pen)", color="#f59e0b")
        ax.plot(step_index, piv.get("all_0", 0.0), marker="o", label="all 0", color="#0ea5e9")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("global_step")
        ax.set_ylabel("share within std=0 groups")
        ax.set_title(title)
        ax.grid(alpha=0.25)

    _one(axes[0], baseline_dead_cls, "Baseline dead-group makeup")
    _one(axes[1], iterative_dead_cls, "Iterative dead-group makeup (all rounds)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_iterative_accepted_dead_makeup(
    step_index: list[int],
    iterative_accepted_dead_cls: pd.DataFrame,
    out_path: Path,
) -> None:
    piv = iterative_accepted_dead_cls.pivot(index="step", columns="dead_class", values="frac_within_dead").fillna(0.0)
    piv = piv.reindex(step_index).fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(step_index, piv.get("all_-1", 0.0), marker="o", label="all -1", color="#b91c1c")
    ax.plot(step_index, piv.get("all_neg_nonpen", 0.0), marker="o", label="all neg(non-pen)", color="#f59e0b")
    ax.plot(step_index, piv.get("all_0", 0.0), marker="o", label="all 0", color="#0ea5e9")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("global_step")
    ax.set_ylabel("share within accepted std=0 groups")
    ax.set_title("Iterative ACCEPTED std=0 groups makeup")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_effective_vs_diversity(diversity_step: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharex=True)
    for ax, run, color in [
        (axes[0], "baseline", "#d97706"),
        (axes[1], "iterative", "#0f766e"),
    ]:
        d = diversity_step[diversity_step["run"] == run].sort_values("step")
        ax.plot(d["step"], d["effective_batch_frac"], marker="o", color="black", label="effective_batch_frac")
        ax2 = ax.twinx()
        ax2.plot(d["step"], d["uniq_scores_mean"], marker="o", color="#2563eb", label="mean unique scores/group")
        ax2.plot(d["step"], d["uniq_moves_mean"], marker="o", color="#9333ea", label="mean unique moves/group")
        ax.set_title(run)
        ax.set_xlabel("global_step")
        ax.set_ylabel("effective_batch_frac")
        ax2.set_ylabel("group diversity")
        ax.grid(alpha=0.25)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.suptitle("Effective Batch vs Group Diversity", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _analyze_entropy_vs_effective_batch(
    evidence_root: Path,
    out_tables_dir: Path,
    out_plots_dir: Path,
    step_min: int,
    step_max: int,
) -> None:
    run_map = {"baseline": "dg41tlmo", "iterative": "s0anl08n"}
    step_rows: list[pd.DataFrame] = []
    overall_rows: list[dict] = []
    by_window_rows: list[dict] = []

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharex=True)
    for ax, (run_label, run_id), color in zip(
        axes,
        run_map.items(),
        ["#d97706", "#0f766e"],
    ):
        h = pd.read_parquet(evidence_root / run_id / "history.parquet")
        need_cols = ["training/global_step", "actor/entropy", "grpo/effective_batch_frac"]
        for c in need_cols:
            if c not in h.columns:
                raise RuntimeError(f"Missing history column for {run_id}: {c}")
        h = (
            h[need_cols]
            .rename(columns={"training/global_step": "step"})
            .dropna(subset=["step", "actor/entropy", "grpo/effective_batch_frac"])
            .sort_values("step")
        )
        h = h[(h["step"] >= step_min) & (h["step"] <= step_max)].copy()
        if h.empty:
            raise RuntimeError(f"No history rows in requested range for {run_id}.")
        h["run"] = run_label
        h["window"] = _window_col(h["step"])
        step_rows.append(h)

        corr = float(np.corrcoef(h["actor/entropy"], h["grpo/effective_batch_frac"])[0, 1])
        overall_rows.append(
            {
                "run": run_label,
                "rows": int(len(h)),
                "entropy_effective_batch_corr": corr,
                "entropy_mean": float(h["actor/entropy"].mean()),
                "effective_batch_frac_mean": float(h["grpo/effective_batch_frac"].mean()),
            }
        )
        by_window = (
            h.groupby("window", as_index=False, observed=False)
            .agg(
                actor_entropy_mean=("actor/entropy", "mean"),
                effective_batch_frac_mean=("grpo/effective_batch_frac", "mean"),
            )
            .copy()
        )
        for _, row in by_window.iterrows():
            by_window_rows.append(
                {
                    "run": run_label,
                    "window": row["window"],
                    "actor_entropy_mean": float(row["actor_entropy_mean"]),
                    "effective_batch_frac_mean": float(row["effective_batch_frac_mean"]),
                }
            )

        ax.plot(h["step"], h["actor/entropy"], color="#2563eb", label="actor/entropy", linewidth=1.8)
        ax2 = ax.twinx()
        ax2.plot(
            h["step"],
            h["grpo/effective_batch_frac"],
            color="black",
            label="grpo/effective_batch_frac",
            linewidth=1.8,
        )
        ax.set_title(run_label)
        ax.set_xlabel("global_step")
        ax.set_ylabel("actor/entropy")
        ax2.set_ylabel("effective_batch_frac")
        ax.grid(alpha=0.25)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    fig.suptitle("Policy Entropy vs Effective Batch", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_plots_dir / "17_entropy_vs_effective_batch.png", dpi=180)
    plt.close(fig)

    pd.concat(step_rows, ignore_index=True).to_csv(
        out_tables_dir / "entropy_vs_effective_batch_by_step.csv", index=False
    )
    pd.DataFrame(overall_rows).to_csv(
        out_tables_dir / "entropy_effective_batch_overall.csv", index=False
    )
    pd.DataFrame(by_window_rows).to_csv(
        out_tables_dir / "entropy_effective_batch_by_window.csv", index=False
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence_root", default="analysis/wandb_evidence")
    ap.add_argument("--out_dir", default="analysis/investigation_s0_vs_dg")
    ap.add_argument("--steps_start", type=int, default=20)
    ap.add_argument("--steps_end", type=int, default=360)
    ap.add_argument("--steps_stride", type=int, default=20)
    args = ap.parse_args()

    evidence_root = Path(args.evidence_root)
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots_visual"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    steps = list(range(args.steps_start, args.steps_end + 1, args.steps_stride))

    baseline_samples = _collect_baseline(evidence_root=evidence_root, steps=steps)
    iterative_samples = _collect_iterative(evidence_root=evidence_root, steps=steps)

    baseline_groups = _group_baseline(baseline_samples)
    iterative_groups = _group_iterative(iterative_samples)
    iterative_groups_accepted = iterative_groups[iterative_groups["accepted"]].copy()

    baseline_dead_cls = _dead_class_by_step(
        baseline_groups, tables_dir / "baseline_dead_class_by_step.csv"
    )
    iterative_dead_cls = _dead_class_by_step(
        iterative_groups, tables_dir / "iterative_dead_class_by_step.csv"
    )
    iterative_accepted_dead_cls = _dead_class_by_step(
        iterative_groups_accepted, tables_dir / "iterative_accepted_dead_class_by_step.csv"
    )

    baseline_dead_quality = _dead_quality_by_step(
        baseline_groups, tables_dir / "baseline_dead_group_quality_by_step.csv"
    )
    iterative_dead_quality = _dead_quality_by_step(
        iterative_groups, tables_dir / "iterative_dead_group_quality_by_step.csv"
    )
    iterative_accepted_dead_quality = _dead_quality_by_step(
        iterative_groups_accepted, tables_dir / "iterative_accepted_dead_group_quality_by_step.csv"
    )

    baseline_diag = _merge_diagnosis(
        baseline_groups,
        baseline_dead_cls,
        baseline_dead_quality,
        eff_col_name="baseline_effective_batch_frac_from_logs",
    )
    iterative_diag = _merge_diagnosis(
        iterative_groups,
        iterative_dead_cls,
        iterative_dead_quality,
        eff_col_name="iterative_effective_batch_frac_from_logs",
    )
    baseline_diag.to_csv(tables_dir / "baseline_effective_batch_diagnosis_by_step.csv", index=False)
    iterative_diag.to_csv(tables_dir / "iterative_effective_batch_diagnosis_by_step.csv", index=False)

    diversity_step = pd.concat(
        [
            baseline_groups.groupby("step", as_index=False)
            .agg(
                effective_batch_frac=("dead", lambda x: 1.0 - float(x.mean())),
                dead_frac=("dead", "mean"),
                uniq_scores_mean=("uniq_scores", "mean"),
                uniq_moves_mean=("uniq_moves", "mean"),
            )
            .assign(run="baseline"),
            iterative_groups.groupby("step", as_index=False)
            .agg(
                effective_batch_frac=("dead", lambda x: 1.0 - float(x.mean())),
                dead_frac=("dead", "mean"),
                uniq_scores_mean=("uniq_scores", "mean"),
                uniq_moves_mean=("uniq_moves", "mean"),
            )
            .assign(run="iterative"),
        ],
        ignore_index=True,
    ).sort_values(["run", "step"])
    diversity_step.to_csv(tables_dir / "effective_batch_vs_group_diversity_by_step.csv", index=False)

    corr_rows: list[dict] = []
    for run, sub in diversity_step.groupby("run"):
        corr_rows.append(
            {
                "run": run,
                "corr_dead_vs_uniq_scores": float(np.corrcoef(sub["dead_frac"], sub["uniq_scores_mean"])[0, 1]),
                "corr_dead_vs_uniq_moves": float(np.corrcoef(sub["dead_frac"], sub["uniq_moves_mean"])[0, 1]),
            }
        )
    pd.DataFrame(corr_rows).to_csv(tables_dir / "effective_batch_diversity_correlations.csv", index=False)

    # A compact accepted-vs-unaccepted iterative breakdown by window.
    iter_acc_breakdown = iterative_groups.copy()
    iter_acc_breakdown["window"] = _window_col(iter_acc_breakdown["step"])
    (
        iter_acc_breakdown.groupby(["window", "accepted"], as_index=False, observed=False)
        .agg(
            groups=("dead", "size"),
            dead_frac=("dead", "mean"),
            uniq_scores_mean=("uniq_scores", "mean"),
            uniq_moves_mean=("uniq_moves", "mean"),
        )
        .to_csv(tables_dir / "iterative_accepted_vs_unaccepted_diversity.csv", index=False)
    )

    _plot_effective_decomposition(
        baseline_diag=baseline_diag,
        iterative_diag=iterative_diag,
        out_path=plots_dir / "13_effective_batch_decomposition.png",
    )
    _plot_dead_makeup_within_dead(
        step_index=steps,
        baseline_dead_cls=baseline_dead_cls,
        iterative_dead_cls=iterative_dead_cls,
        out_path=plots_dir / "14_dead_group_makeup_within_dead.png",
    )
    _plot_iterative_accepted_dead_makeup(
        step_index=steps,
        iterative_accepted_dead_cls=iterative_accepted_dead_cls,
        out_path=plots_dir / "15_iterative_accepted_dead_makeup.png",
    )
    _plot_effective_vs_diversity(
        diversity_step=diversity_step,
        out_path=plots_dir / "16_effective_batch_vs_group_diversity.png",
    )
    _analyze_entropy_vs_effective_batch(
        evidence_root=evidence_root,
        out_tables_dir=tables_dir,
        out_plots_dir=plots_dir,
        step_min=args.steps_start,
        step_max=args.steps_end,
    )

    # Write a one-row summary for report copy-paste safety.
    summary = {
        "baseline_eff_step40": float(
            baseline_diag.loc[baseline_diag["step"] == 40, "baseline_effective_batch_frac_from_logs"].iloc[0]
        ),
        "baseline_eff_step340": float(
            baseline_diag.loc[baseline_diag["step"] == 340, "baseline_effective_batch_frac_from_logs"].iloc[0]
        ),
        "iterative_eff_step40": float(
            iterative_diag.loc[iterative_diag["step"] == 40, "iterative_effective_batch_frac_from_logs"].iloc[0]
        ),
        "iterative_eff_step340": float(
            iterative_diag.loc[iterative_diag["step"] == 340, "iterative_effective_batch_frac_from_logs"].iloc[0]
        ),
        "baseline_dead_frac_mean": float(1.0 - baseline_diag["baseline_effective_batch_frac_from_logs"].mean()),
        "iterative_dead_frac_mean": float(1.0 - iterative_diag["iterative_effective_batch_frac_from_logs"].mean()),
    }
    (out_dir / "effective_batch_issue_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] wrote tables to: {tables_dir}")
    print(f"[OK] wrote plots to: {plots_dir}")
    print(f"[OK] wrote summary: {out_dir / 'effective_batch_issue_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
