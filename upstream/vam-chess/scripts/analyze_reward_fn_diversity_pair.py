#!/usr/bin/env python3
"""
Pair analysis for reward-function diversity effects on GRPO effective batch.

Default pair:
  - iu768gtj: winrate_vs_best
  - yu8phknt: rank_among_moves

Outputs:
  - matched-step rollout analysis tables
  - side-by-side plots using the same effective-batch diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_COLORS = {
    "run_a": "#1d4ed8",  # blue
    "run_b": "#dc2626",  # red
}

DEAD_COLORS = {
    "all_-1": "#b91c1c",
    "all_neg_nonpen": "#f59e0b",
    "all_0": "#0ea5e9",
    "all_pos": "#16a34a",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get_nested(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def _safe_bool(s: pd.Series | Any, default: bool) -> pd.Series:
    if isinstance(s, pd.Series):
        return s.fillna(default).astype(bool)
    return pd.Series([default]).astype(bool)


def _classify_dead_constant(v: float) -> str:
    eps = 1e-12
    if pd.isna(v):
        return "nan"
    v = float(v)
    if abs(v + 1.0) <= eps:
        return "all_-1"
    if abs(v) <= eps:
        return "all_0"
    if -1.0 < v < 0.0:
        return "all_neg_nonpen"
    if v > 0.0:
        return "all_pos"
    return "other"


def _collect_samples(evidence_root: Path, run_id: str, steps: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for step in steps:
        fp = evidence_root / run_id / "files" / "rollout_logs" / f"{step}.jsonl"
        if not fp.exists():
            raise FileNotFoundError(f"Missing rollout file: {fp}")
        for row in _load_jsonl(fp):
            row["step"] = int(step)
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rollout rows loaded for run={run_id}")
    df = pd.DataFrame(rows)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["uid"] = df["uid"].astype(str)
    df["penalty_applied"] = _safe_bool(df.get("penalty_applied"), default=False)
    in_subset = df.get("in_subset")
    if isinstance(in_subset, pd.Series):
        df["in_subset"] = in_subset.fillna(True).astype(bool)
    else:
        df["in_subset"] = True
    df["pred_move"] = df.get("pred_move", "").fillna("").astype(str)
    df["gt_uci"] = df.get("gt_uci", "").fillna("").astype(str)
    df["success_sample"] = (df["pred_move"] == df["gt_uci"]) & (~df["penalty_applied"]) & df["in_subset"]
    return df


def _summarize_groups(samples: pd.DataFrame) -> pd.DataFrame:
    g = (
        samples.groupby(["step", "uid"], as_index=False)
        .agg(
            n_samples=("score", "size"),
            score_std=("score", "std"),
            score_min=("score", "min"),
            score_max=("score", "max"),
            success_any=("success_sample", "max"),
            success_all=("success_sample", "min"),
            penalty_sample_frac=("penalty_applied", "mean"),
            uniq_scores=("score", "nunique"),
            uniq_moves=("pred_move", "nunique"),
            chess_reward_fn=("chess_reward_fn", "first"),
        )
        .copy()
    )
    g["score_std"] = g["score_std"].fillna(0.0)
    g["dead"] = g["score_std"] <= 0.0
    g["dead_class"] = np.where(g["dead"], g["score_min"].map(_classify_dead_constant), "non_dead")
    return g


def _step_metrics(group_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        group_df.groupby("step", as_index=False)
        .agg(
            groups=("uid", "nunique"),
            effective_batch_frac=("dead", lambda x: 1.0 - float(x.mean())),
            dead_frac=("dead", "mean"),
            prompt_success_frac=("success_any", "mean"),
            uniq_scores_mean=("uniq_scores", "mean"),
            uniq_moves_mean=("uniq_moves", "mean"),
            penalty_sample_frac_mean=("penalty_sample_frac", "mean"),
        )
        .copy()
    )

    dead = group_df[group_df["dead"]].copy()
    dead_cls = (
        dead.groupby(["step", "dead_class"], as_index=False)
        .size()
        .rename(columns={"size": "dead_class_count"})
    )
    dead_tot = dead.groupby("step", as_index=False).size().rename(columns={"size": "dead_groups"})
    dead_cls = dead_cls.merge(dead_tot, on="step", how="left")
    dead_cls["frac_within_dead"] = dead_cls["dead_class_count"] / dead_cls["dead_groups"]

    piv = dead_cls.pivot(index="step", columns="dead_class", values="frac_within_dead").fillna(0.0)
    for cls in ("all_-1", "all_neg_nonpen", "all_0", "all_pos"):
        if cls not in piv.columns:
            piv[cls] = 0.0
    piv = piv[["all_-1", "all_neg_nonpen", "all_0", "all_pos"]].reset_index()

    out = out.merge(piv, on="step", how="left").fillna(0.0)
    out["all_-1_overall"] = out["all_-1"] * out["dead_frac"]
    out["all_neg_nonpen_overall"] = out["all_neg_nonpen"] * out["dead_frac"]
    out["all_0_overall"] = out["all_0"] * out["dead_frac"]
    out["all_pos_overall"] = out["all_pos"] * out["dead_frac"]
    return out.sort_values("step").reset_index(drop=True)


def _dead_quality_by_step(group_df: pd.DataFrame) -> pd.DataFrame:
    dead = group_df[group_df["dead"]].copy()
    if dead.empty:
        return pd.DataFrame(
            columns=[
                "step",
                "dead_groups",
                "dead_success_any_frac",
                "dead_success_all_frac",
                "dead_penalty_sample_frac",
                "dead_constant_score_mean",
            ]
        )
    out = (
        dead.groupby("step", as_index=False)
        .agg(
            dead_groups=("dead", "size"),
            dead_success_any_frac=("success_any", "mean"),
            dead_success_all_frac=("success_all", "mean"),
            dead_penalty_sample_frac=("penalty_sample_frac", "mean"),
            dead_constant_score_mean=("score_min", "mean"),
        )
        .copy()
    )
    return out


def _dead_class_overall(group_df: pd.DataFrame) -> pd.DataFrame:
    dead = group_df[group_df["dead"]].copy()
    out = dead.groupby("dead_class", as_index=False).size().rename(columns={"size": "groups"})
    out["fraction"] = out["groups"] / out["groups"].sum() if len(out) else 0.0
    return out.sort_values("groups", ascending=False)


def _success_by_dead_class(group_df: pd.DataFrame) -> pd.DataFrame:
    dead = group_df[group_df["dead"]].copy()
    out = (
        dead.groupby("dead_class", as_index=False)
        .agg(
            groups=("dead", "size"),
            success_any_frac=("success_any", "mean"),
            success_all_frac=("success_all", "mean"),
            penalty_sample_frac=("penalty_sample_frac", "mean"),
            constant_score_mean=("score_min", "mean"),
        )
        .copy()
    )
    return out.sort_values("groups", ascending=False)


def _plot_effective_decomposition(step_a: pd.DataFrame, step_b: pd.DataFrame, label_a: str, label_b: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, df, title, line_color in [
        (axes[0], step_a, label_a, RUN_COLORS["run_a"]),
        (axes[1], step_b, label_b, RUN_COLORS["run_b"]),
    ]:
        x = df["step"]
        ax.stackplot(
            x,
            df["all_-1_overall"],
            df["all_neg_nonpen_overall"],
            df["all_0_overall"],
            df["all_pos_overall"],
            labels=["dead all -1", "dead all neg(non-pen)", "dead all 0", "dead all +score"],
            colors=[DEAD_COLORS["all_-1"], DEAD_COLORS["all_neg_nonpen"], DEAD_COLORS["all_0"], DEAD_COLORS["all_pos"]],
            alpha=0.62,
        )
        ax.plot(x, df["effective_batch_frac"], marker="o", color=line_color, linewidth=2, label="effective_batch_frac")
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("fraction of groups")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    axes[1].set_xlabel("global_step")
    fig.suptitle("Effective-Batch Decomposition (Matched Steps)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_dead_makeup_within_dead(step_a: pd.DataFrame, step_b: pd.DataFrame, label_a: str, label_b: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, df, title in [
        (axes[0], step_a, label_a),
        (axes[1], step_b, label_b),
    ]:
        x = df["step"]
        ax.plot(x, df["all_-1"], marker="o", color=DEAD_COLORS["all_-1"], label="all -1")
        ax.plot(x, df["all_neg_nonpen"], marker="o", color=DEAD_COLORS["all_neg_nonpen"], label="all neg(non-pen)")
        ax.plot(x, df["all_0"], marker="o", color=DEAD_COLORS["all_0"], label="all 0")
        ax.plot(x, df["all_pos"], marker="o", color=DEAD_COLORS["all_pos"], label="all +score")
        ax.set_title(title)
        ax.set_xlabel("global_step")
        ax.set_ylabel("share within std=0 groups")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_dead_quality(step_a: pd.DataFrame, step_b: pd.DataFrame, label_a: str, label_b: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for ax, df, label, color in [
        (axes[0], step_a, label_a, RUN_COLORS["run_a"]),
        (axes[1], step_b, label_b, RUN_COLORS["run_b"]),
    ]:
        ax.plot(df["step"], df["dead_success_any_frac"], marker="o", color=color, label="dead_success_any_frac")
        ax2 = ax.twinx()
        ax2.plot(
            df["step"],
            df["dead_penalty_sample_frac"],
            marker="o",
            color=DEAD_COLORS["all_-1"],
            label="dead_penalty_sample_frac",
        )
        ax.set_title(label)
        ax.set_xlabel("global_step")
        ax.set_ylabel("success fraction in dead groups")
        ax2.set_ylabel("penalty sample frac in dead groups")
        ax.grid(alpha=0.25)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.suptitle("Quality of Dead Groups", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_effective_vs_diversity(step_long: pd.DataFrame, label_a: str, label_b: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharex=True)
    for ax, metric, title in [
        (axes[0], "uniq_scores_mean", "Mean Unique Scores / Group"),
        (axes[1], "uniq_moves_mean", "Mean Unique Moves / Group"),
    ]:
        for run_key, label, color in [
            ("run_a", label_a, RUN_COLORS["run_a"]),
            ("run_b", label_b, RUN_COLORS["run_b"]),
        ]:
            d = step_long[step_long["run_key"] == run_key].sort_values("step")
            ax.plot(d["step"], d[metric], marker="o", color=color, label=f"{label}: {metric}")
            ax2 = ax.twinx()
            ax2.plot(
                d["step"],
                d["effective_batch_frac"],
                marker="o",
                linestyle="--",
                color=color,
                alpha=0.45,
                label=f"{label}: effective_batch_frac",
            )
        ax.set_title(title)
        ax.set_xlabel("global_step")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        for child in ax.figure.axes:
            if child is ax:
                continue
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    fig.suptitle("Effective Batch vs Group Diversity", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_entropy_vs_effective(entropy_df: pd.DataFrame, label_a: str, label_b: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for run_key, label, color in [
        ("run_a", label_a, RUN_COLORS["run_a"]),
        ("run_b", label_b, RUN_COLORS["run_b"]),
    ]:
        d = entropy_df[entropy_df["run_key"] == run_key].sort_values("step")
        ax.plot(d["step"], d["actor_entropy"], marker="o", color=color, label=f"{label}: actor/entropy")
    ax2 = ax.twinx()
    for run_key, label, color in [
        ("run_a", label_a, RUN_COLORS["run_a"]),
        ("run_b", label_b, RUN_COLORS["run_b"]),
    ]:
        d = entropy_df[entropy_df["run_key"] == run_key].sort_values("step")
        ax2.plot(
            d["step"],
            d["effective_batch_frac_history"],
            marker="o",
            linestyle="--",
            color=color,
            alpha=0.45,
            label=f"{label}: grpo/effective_batch_frac",
        )
    ax.set_xlabel("global_step")
    ax.set_ylabel("actor/entropy")
    ax2.set_ylabel("grpo/effective_batch_frac")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax.set_title("Entropy vs Effective Batch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_dead_vs_diversity_scatter(step_long: pd.DataFrame, label_a: str, label_b: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for run_key, label, color in [
        ("run_a", label_a, RUN_COLORS["run_a"]),
        ("run_b", label_b, RUN_COLORS["run_b"]),
    ]:
        d = step_long[step_long["run_key"] == run_key].sort_values("step")
        ax.scatter(
            d["dead_frac"],
            d["uniq_scores_mean"],
            s=70,
            alpha=0.8,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )
        for _, row in d.iterrows():
            ax.text(float(row["dead_frac"]) + 0.002, float(row["uniq_scores_mean"]) + 0.01, str(int(row["step"])), fontsize=7)
    ax.set_xlabel("dead group fraction")
    ax.set_ylabel("mean unique scores / group")
    ax.set_title("Dead Fraction vs Score Diversity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence_root", default="analysis/wandb_evidence")
    ap.add_argument("--run_a", default="iu768gtj")
    ap.add_argument("--run_b", default="yu8phknt")
    ap.add_argument("--out_dir", default="analysis/investigation_rewardfn_iu_vs_yu")
    ap.add_argument("--steps_start", type=int, default=20)
    ap.add_argument("--steps_end", type=int, default=360)
    ap.add_argument("--steps_stride", type=int, default=20)
    args = ap.parse_args()

    evidence_root = Path(args.evidence_root)
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    steps = list(range(args.steps_start, args.steps_end + 1, args.steps_stride))
    if not steps:
        raise ValueError("No steps selected.")

    # Run metadata/config
    cfg_a = _load_json(evidence_root / args.run_a / "config_api.json")
    cfg_b = _load_json(evidence_root / args.run_b / "config_api.json")
    summary_a = _load_json(evidence_root / args.run_a / "summary.json")
    summary_b = _load_json(evidence_root / args.run_b / "summary.json")
    meta_a = _load_json(evidence_root / args.run_a / "run_meta.json")
    meta_b = _load_json(evidence_root / args.run_b / "run_meta.json")
    md_a_path = evidence_root / args.run_a / "files" / "wandb-metadata.json"
    md_b_path = evidence_root / args.run_b / "files" / "wandb-metadata.json"
    md_a = _load_json(md_a_path) if md_a_path.exists() else {}
    md_b = _load_json(md_b_path) if md_b_path.exists() else {}

    reward_a = str(_get_nested(cfg_a, "custom_reward_function.reward_kwargs.chess_reward_fn"))
    reward_b = str(_get_nested(cfg_b, "custom_reward_function.reward_kwargs.chess_reward_fn"))
    label_a = f"{args.run_a} ({reward_a})"
    label_b = f"{args.run_b} ({reward_b})"

    # Load rollout samples
    samples_a = _collect_samples(evidence_root=evidence_root, run_id=args.run_a, steps=steps)
    samples_b = _collect_samples(evidence_root=evidence_root, run_id=args.run_b, steps=steps)

    sample_score_rows: list[dict[str, Any]] = []
    for run_key, run_id, sdf in [("run_a", args.run_a, samples_a), ("run_b", args.run_b, samples_b)]:
        ser = pd.to_numeric(sdf["score"], errors="coerce").dropna()
        q = ser.quantile([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        sample_score_rows.append(
            {
                "run_key": run_key,
                "run_id": run_id,
                "n_samples": int(len(ser)),
                "unique_score_count": int(ser.nunique()),
                "frac_score_eq_-1": float((ser == -1.0).mean()),
                "frac_score_eq_0": float((ser == 0.0).mean()),
                "frac_score_eq_1": float((ser == 1.0).mean()),
                "frac_score_in_{-1,0,1}": float(ser.round(12).isin([-1.0, 0.0, 1.0]).mean()),
                "q01": float(q.loc[0.01]),
                "q10": float(q.loc[0.10]),
                "q25": float(q.loc[0.25]),
                "q50": float(q.loc[0.50]),
                "q75": float(q.loc[0.75]),
                "q90": float(q.loc[0.90]),
                "q99": float(q.loc[0.99]),
            }
        )

    groups_a = _summarize_groups(samples_a)
    groups_b = _summarize_groups(samples_b)

    step_a = _step_metrics(groups_a)
    step_b = _step_metrics(groups_b)

    dead_quality_a = _dead_quality_by_step(groups_a)
    dead_quality_b = _dead_quality_by_step(groups_b)
    step_a = step_a.merge(dead_quality_a, on="step", how="left")
    step_b = step_b.merge(dead_quality_b, on="step", how="left")

    step_a["run_key"] = "run_a"
    step_b["run_key"] = "run_b"
    step_long = pd.concat([step_a, step_b], ignore_index=True)

    # History entropy/effective
    hist_frames: list[pd.DataFrame] = []
    entropy_summary_rows: list[dict[str, Any]] = []
    for run_key, run_id in [("run_a", args.run_a), ("run_b", args.run_b)]:
        hist = pd.read_parquet(evidence_root / run_id / "history.parquet")
        if not {"training/global_step", "actor/entropy", "grpo/effective_batch_frac"}.issubset(hist.columns):
            raise RuntimeError(f"Missing history columns in run={run_id}")
        h = (
            hist[["training/global_step", "actor/entropy", "grpo/effective_batch_frac"]]
            .rename(
                columns={
                    "training/global_step": "step",
                    "actor/entropy": "actor_entropy",
                    "grpo/effective_batch_frac": "effective_batch_frac_history",
                }
            )
            .dropna()
            .copy()
        )
        h = h[(h["step"] >= args.steps_start) & (h["step"] <= args.steps_end)].sort_values("step")
        h["run_key"] = run_key
        hist_frames.append(h)
        entropy_summary_rows.append(
            {
                "run_key": run_key,
                "run_id": run_id,
                "corr_entropy_vs_effective_batch_history": float(
                    np.corrcoef(h["actor_entropy"], h["effective_batch_frac_history"])[0, 1]
                ),
                "entropy_mean": float(h["actor_entropy"].mean()),
                "effective_batch_frac_history_mean": float(h["effective_batch_frac_history"].mean()),
            }
        )

    entropy_df = pd.concat(hist_frames, ignore_index=True)

    # Crosscheck: rollout-derived effective batch vs history effective batch at sampled steps.
    eff_cross = (
        step_long[["run_key", "step", "effective_batch_frac"]]
        .merge(entropy_df[["run_key", "step", "effective_batch_frac_history"]], on=["run_key", "step"], how="left")
        .copy()
    )
    eff_cross["abs_error"] = (eff_cross["effective_batch_frac"] - eff_cross["effective_batch_frac_history"]).abs()

    # Compare key configs (verify only reward function differs among main knobs).
    compare_keys = [
        "git.commit",
        "custom_reward_function.reward_kwargs.chess_reward_fn",
        "custom_reward_function.reward_kwargs.illegal_penalty",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.rollout.temperature",
        "actor_rollout_ref.rollout.top_k",
        "actor_rollout_ref.rollout.top_p",
        "actor_rollout_ref.rollout.do_sample",
        "algorithm.allowed_move_elim.enable",
        "algorithm.filter_groups.enable",
        "data.train_files",
        "data.val_files",
        "trainer.nnodes",
        "trainer.full_eval.prompt_template_path",
        "trainer.full_eval_freq",
    ]
    comp_rows: list[dict[str, Any]] = []
    for key in compare_keys:
        if key == "git.commit":
            a_val = _get_nested(md_a, "git.commit")
            b_val = _get_nested(md_b, "git.commit")
        else:
            a_val = _get_nested(cfg_a, key)
            b_val = _get_nested(cfg_b, key)
        comp_rows.append(
            {
                "dimension": key,
                "run_a_value": json.dumps(a_val, sort_keys=True, default=str),
                "run_b_value": json.dumps(b_val, sort_keys=True, default=str),
                "same": bool(a_val == b_val),
            }
        )
    comp_df = pd.DataFrame(comp_rows)

    # Dead class and class-quality tables
    dead_class_by_step_a = (
        groups_a[groups_a["dead"]]
        .groupby(["step", "dead_class"], as_index=False)
        .size()
        .rename(columns={"size": "groups"})
    )
    dead_class_by_step_b = (
        groups_b[groups_b["dead"]]
        .groupby(["step", "dead_class"], as_index=False)
        .size()
        .rename(columns={"size": "groups"})
    )
    dead_class_by_step_a["run_key"] = "run_a"
    dead_class_by_step_b["run_key"] = "run_b"
    dead_class_by_step = pd.concat([dead_class_by_step_a, dead_class_by_step_b], ignore_index=True)

    for run_key in ("run_a", "run_b"):
        mask = dead_class_by_step["run_key"] == run_key
        totals = dead_class_by_step.loc[mask].groupby("step")["groups"].transform("sum")
        dead_class_by_step.loc[mask, "frac_within_dead"] = dead_class_by_step.loc[mask, "groups"] / totals

    dead_class_overall = pd.concat(
        [
            _dead_class_overall(groups_a).assign(run_key="run_a"),
            _dead_class_overall(groups_b).assign(run_key="run_b"),
        ],
        ignore_index=True,
    )
    success_by_dead_class = pd.concat(
        [
            _success_by_dead_class(groups_a).assign(run_key="run_a"),
            _success_by_dead_class(groups_b).assign(run_key="run_b"),
        ],
        ignore_index=True,
    )

    # Run-level summaries
    summary_rows = []
    for run_key, run_id, reward_fn, groups_df, step_df in [
        ("run_a", args.run_a, reward_a, groups_a, step_a),
        ("run_b", args.run_b, reward_b, groups_b, step_b),
    ]:
        diversity_corr_scores = float(np.corrcoef(step_df["dead_frac"], step_df["uniq_scores_mean"])[0, 1])
        diversity_corr_moves = float(np.corrcoef(step_df["dead_frac"], step_df["uniq_moves_mean"])[0, 1])
        summary_rows.append(
            {
                "run_key": run_key,
                "run_id": run_id,
                "reward_fn": reward_fn,
                "effective_batch_mean": float(step_df["effective_batch_frac"].mean()),
                "dead_frac_mean": float(step_df["dead_frac"].mean()),
                "prompt_success_mean": float(step_df["prompt_success_frac"].mean()),
                "dead_minus1_overall_mean": float(step_df["all_-1_overall"].mean()),
                "dead_zero_overall_mean": float(step_df["all_0_overall"].mean()),
                "dead_neg_nonpen_overall_mean": float(step_df["all_neg_nonpen_overall"].mean()),
                "dead_pos_overall_mean": float(step_df["all_pos_overall"].mean()),
                "corr_dead_vs_uniq_scores": diversity_corr_scores,
                "corr_dead_vs_uniq_moves": diversity_corr_moves,
                "reward_fn_from_logs_unique": json.dumps(sorted(groups_df["chess_reward_fn"].dropna().astype(str).unique().tolist())),
            }
        )
    run_summary_df = pd.DataFrame(summary_rows)

    step_long["window"] = pd.cut(step_long["step"], bins=[0, 120, 240, 400], labels=["early", "mid", "late"])
    window_summary = (
        step_long.groupby(["run_key", "window"], as_index=False, observed=False)
        .agg(
            effective_batch_frac_mean=("effective_batch_frac", "mean"),
            dead_frac_mean=("dead_frac", "mean"),
            prompt_success_frac_mean=("prompt_success_frac", "mean"),
            uniq_scores_mean=("uniq_scores_mean", "mean"),
            uniq_moves_mean=("uniq_moves_mean", "mean"),
        )
        .copy()
    )

    run_config_summary = pd.DataFrame(
        [
            {
                "run_key": "run_a",
                "run_id": args.run_a,
                "run_name": meta_a.get("run_name"),
                "run_state": meta_a.get("run_state"),
                "reward_fn": reward_a,
                "git_commit": _get_nested(md_a, "git.commit"),
            },
            {
                "run_key": "run_b",
                "run_id": args.run_b,
                "run_name": meta_b.get("run_name"),
                "run_state": meta_b.get("run_state"),
                "reward_fn": reward_b,
                "git_commit": _get_nested(md_b, "git.commit"),
            },
        ]
    )

    fullgame_summary = pd.DataFrame(
        [
            {
                "run_key": "run_a",
                "run_id": args.run_a,
                "reward_fn": reward_a,
                "full_game_eval/overall/acpl_per_move": summary_a.get("full_game_eval/overall/acpl_per_move"),
                "full_game_eval/overall/win_rate": summary_a.get("full_game_eval/overall/win_rate"),
            },
            {
                "run_key": "run_b",
                "run_id": args.run_b,
                "reward_fn": reward_b,
                "full_game_eval/overall/acpl_per_move": summary_b.get("full_game_eval/overall/acpl_per_move"),
                "full_game_eval/overall/win_rate": summary_b.get("full_game_eval/overall/win_rate"),
            },
        ]
    )

    # Save tables
    comp_df.to_csv(tables_dir / "comparability_matrix.csv", index=False)
    step_long.to_csv(tables_dir / "step_metrics_long.csv", index=False)
    step_wide = (
        step_long.pivot(index="step", columns="run_key")
        .sort_index(axis=1)
        .copy()
    )
    step_wide.columns = [f"{a}_{b}" for a, b in step_wide.columns]
    step_wide = step_wide.reset_index()
    step_wide.to_csv(tables_dir / "step_metrics_wide.csv", index=False)
    dead_class_by_step.to_csv(tables_dir / "dead_class_by_step.csv", index=False)
    dead_class_overall.to_csv(tables_dir / "dead_class_overall.csv", index=False)
    success_by_dead_class.to_csv(tables_dir / "success_by_dead_class.csv", index=False)
    pd.concat(
        [
            dead_quality_a.assign(run_key="run_a"),
            dead_quality_b.assign(run_key="run_b"),
        ],
        ignore_index=True,
    ).to_csv(tables_dir / "dead_group_quality_by_step.csv", index=False)
    entropy_df.to_csv(tables_dir / "entropy_vs_effective_batch_by_step.csv", index=False)
    pd.DataFrame(entropy_summary_rows).to_csv(tables_dir / "entropy_effective_summary.csv", index=False)
    eff_cross.to_csv(tables_dir / "effective_batch_crosscheck.csv", index=False)
    run_summary_df.to_csv(tables_dir / "run_level_summary.csv", index=False)
    fullgame_summary.to_csv(tables_dir / "fullgame_summary.csv", index=False)
    window_summary.to_csv(tables_dir / "window_summary.csv", index=False)
    run_config_summary.to_csv(tables_dir / "run_config_summary.csv", index=False)
    sample_score_summary_df = pd.DataFrame(sample_score_rows)
    sample_score_summary_df.to_csv(tables_dir / "sample_score_distribution.csv", index=False)

    # JSON summary for easy consumption
    summary_json = {
        "run_a": {
            "id": args.run_a,
            "reward_fn": reward_a,
            "state": meta_a.get("run_state"),
            "url": meta_a.get("url"),
            "git_commit": _get_nested(md_a, "git.commit"),
        },
        "run_b": {
            "id": args.run_b,
            "reward_fn": reward_b,
            "state": meta_b.get("run_state"),
            "url": meta_b.get("url"),
            "git_commit": _get_nested(md_b, "git.commit"),
        },
        "steps": {"start": args.steps_start, "end": args.steps_end, "stride": args.steps_stride, "count": len(steps)},
        "key_metrics": run_summary_df.to_dict(orient="records"),
        "sample_score_distribution": sample_score_summary_df.to_dict(orient="records"),
        "crosscheck_max_abs_error": float(eff_cross["abs_error"].max()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    # Plots
    _plot_effective_decomposition(step_a, step_b, label_a, label_b, plots_dir / "13_effective_batch_decomposition.png")
    _plot_dead_makeup_within_dead(step_a, step_b, label_a, label_b, plots_dir / "14_dead_group_makeup_within_dead.png")
    _plot_dead_quality(dead_quality_a, dead_quality_b, label_a, label_b, plots_dir / "15_dead_group_quality.png")
    _plot_effective_vs_diversity(step_long, label_a, label_b, plots_dir / "16_effective_batch_vs_group_diversity.png")
    _plot_entropy_vs_effective(entropy_df, label_a, label_b, plots_dir / "17_entropy_vs_effective_batch.png")
    _plot_dead_vs_diversity_scatter(step_long, label_a, label_b, plots_dir / "18_dead_vs_diversity_scatter.png")

    print(f"[OK] wrote pair analysis tables to: {tables_dir}")
    print(f"[OK] wrote pair analysis plots to: {plots_dir}")
    print(f"[OK] summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
