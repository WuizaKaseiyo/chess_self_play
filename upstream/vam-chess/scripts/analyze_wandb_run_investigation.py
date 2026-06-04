#!/usr/bin/env python3
"""
Generate a lightweight, reproducible investigation bundle from locally-downloaded W&B evidence.

Inputs (from `scripts/download_wandb_run_evidence.py --download-files`):
  - <evidence_dir>/history.parquet
  - <evidence_dir>/config_api.yaml (optional; used for metadata)

Outputs (under --out_dir):
  - tables/*.csv (key metric slices + change points)
  - plots/*.png  (time-series plots with schedule overlays)

This is intentionally minimal and only depends on pandas + matplotlib + pyyaml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


def _get_path(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part, None)
        if cur is None:
            return None
    return cur


def _maybe_series(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if "_step" not in df.columns or key not in df.columns:
        return pd.DataFrame(columns=["_step", key])
    sub = df[["_step", key]].dropna()
    if sub.empty:
        return pd.DataFrame(columns=["_step", key])
    return sub.sort_values("_step").reset_index(drop=True)


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def _find_step_change_points(df: pd.DataFrame, key: str) -> pd.DataFrame:
    s = _maybe_series(df, key)
    if s.empty:
        return pd.DataFrame(columns=["_step", key])
    v = s[key].copy()
    if pd.api.types.is_float_dtype(v) or pd.api.types.is_integer_dtype(v):
        # Many “schedule-like” metrics are floats but represent integers.
        if np.all(np.isfinite(v)):
            v_int = v.round().astype(int)
            # Only collapse to int when it looks like an integer series.
            if np.allclose(v_int.astype(float), v.astype(float), atol=1e-6):
                s[key] = v_int
    change = s[s[key].diff().fillna(0) != 0].reset_index(drop=True)
    return change


def _plot_train_metrics_and_schedules(
    *,
    df: pd.DataFrame,
    out_path: Path,
    smooth_window: int,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # 1) Train reward (dense).
    m1 = "critic/score/mean"
    sub1 = _maybe_series(df, m1)
    if not sub1.empty:
        axes[0].plot(sub1["_step"], _rolling_mean(sub1[m1], smooth_window), color="C0", linewidth=1.6)
    axes[0].set_title(f"{title_prefix}train {m1} (smoothed)")
    axes[0].set_ylabel(m1)
    axes[0].grid(True, alpha=0.25)

    # 2) Selection success rate (dense).
    m2 = "selection_sampler/success_rate"
    sub2 = _maybe_series(df, m2)
    if not sub2.empty:
        axes[1].plot(sub2["_step"], _rolling_mean(sub2[m2], smooth_window), color="C2", linewidth=1.6)
    axes[1].set_title(f"{title_prefix}train {m2} (smoothed)")
    axes[1].set_ylabel(m2)
    axes[1].grid(True, alpha=0.25)

    # 3) Schedules: r_max + lr.
    s_rmax = _maybe_series(df, "selection_sampler/r_max")
    s_lr = _maybe_series(df, "actor/lr")
    ax_sched = axes[2]
    ax_sched.set_title(f"{title_prefix}schedules: selection_sampler/r_max and actor/lr")
    if not s_rmax.empty:
        ax_sched.step(
            s_rmax["_step"],
            s_rmax["selection_sampler/r_max"],
            where="post",
            color="C1",
            linewidth=1.6,
            label="selection_sampler/r_max",
        )
    ax_sched.set_ylabel("r_max")
    ax_sched.grid(True, alpha=0.25)

    ax_lr = ax_sched.twinx()
    if not s_lr.empty:
        ax_lr.plot(s_lr["_step"], s_lr["actor/lr"], color="C3", linewidth=1.2, alpha=0.8, label="actor/lr")
    ax_lr.set_ylabel("actor/lr")

    # Mark r_max change points (if any) as vertical lines throughout.
    cp = _find_step_change_points(df, "selection_sampler/r_max")
    for _, row in cp.iterrows():
        step = int(row["_step"])
        for ax in axes:
            ax.axvline(step, color="k", linestyle=":", linewidth=0.8, alpha=0.35)

    axes[-1].set_xlabel("step")
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_sparse_series(
    *,
    df: pd.DataFrame,
    y: str,
    out_path: Path,
    title: str,
    ylabel: str | None = None,
    marker: str = "o",
) -> None:
    sub = _maybe_series(df, y)
    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    if not sub.empty:
        ax.plot(sub["_step"], sub[y], marker=marker, linestyle="-", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel or y)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence_dir", required=True, help="Path to analysis/wandb_evidence/<run_id>")
    ap.add_argument("--out_dir", required=True, help="Output directory (plots/ + tables/)")
    ap.add_argument("--smooth_window", type=int, default=25, help="Rolling mean window for dense train metrics")
    ap.add_argument("--title_prefix", default="", help="Optional title prefix (e.g., 'et48q0cr — ')")
    args = ap.parse_args()

    evidence_dir = Path(args.evidence_dir)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    _ensure_dir(plots_dir)
    _ensure_dir(tables_dir)

    hist_path = evidence_dir / "history.parquet"
    cfg_path = evidence_dir / "config_api.yaml"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing history.parquet: {hist_path}")

    df = pd.read_parquet(hist_path).copy()
    if "_step" not in df.columns:
        raise RuntimeError("history.parquet missing required column: _step")
    df = df.sort_values("_step").reset_index(drop=True)

    cfg = _read_yaml(cfg_path)
    cfg_summary = {
        "run_id": evidence_dir.name,
        "data.self_play.enable": _get_path(cfg, "data.self_play.enable"),
        "algorithm.allowed_move_elim.enable": _get_path(cfg, "algorithm.allowed_move_elim.enable"),
        "algorithm.allowed_move_elim.r_max_start": _get_path(cfg, "algorithm.allowed_move_elim.r_max_start"),
        "algorithm.allowed_move_elim.r_max_end": _get_path(cfg, "algorithm.allowed_move_elim.r_max_end"),
        "algorithm.allowed_move_elim.anneal_frac": _get_path(cfg, "algorithm.allowed_move_elim.anneal_frac"),
        "algorithm.allowed_move_elim.no_success_policy": _get_path(cfg, "algorithm.allowed_move_elim.no_success_policy"),
        "rollout.n": _get_path(cfg, "actor_rollout_ref.rollout.n"),
        "data.train_batch_size": _get_path(cfg, "data.train_batch_size"),
        "data.gen_batch_size": _get_path(cfg, "data.gen_batch_size"),
        "data.max_prompt_length": _get_path(cfg, "data.max_prompt_length"),
        "data.max_response_length": _get_path(cfg, "data.max_response_length"),
        "custom_reward_function.reward_kwargs.chess_reward_fn": _get_path(cfg, "custom_reward_function.reward_kwargs.chess_reward_fn"),
        "data.train_files": _get_path(cfg, "data.train_files"),
        "data.val_files": _get_path(cfg, "data.val_files"),
    }
    (out_dir / "config_summary.json").write_text(json.dumps(cfg_summary, indent=2, sort_keys=True), encoding="utf-8")

    # Tables.
    # 1) ACPL points (sparse).
    acpl_key = "full_game_eval/overall/acpl"
    acpl = _maybe_series(df, acpl_key)
    if not acpl.empty:
        acpl.to_csv(tables_dir / "acpl_points.csv", index=False)

    # 2) Val points (sparse).
    val_keys = [
        "val-core/local/chess_puzzles/acc/mean@1",
        "val-core/local/chess_puzzles_shuffled/acc/mean@1",
        "val-aux/local/chess_puzzles/format_reward/mean@1",
        "val-aux/local/chess_puzzles_shuffled/format_reward/mean@1",
    ]
    keep = ["_step"] + [k for k in val_keys if k in df.columns]
    val = df[keep].dropna(how="all", subset=[k for k in keep if k != "_step"]).sort_values("_step")
    if not val.empty:
        val.to_csv(tables_dir / "val_points.csv", index=False)

    # 3) r_max change points.
    rmax_changes = _find_step_change_points(df, "selection_sampler/r_max")
    if not rmax_changes.empty:
        rmax_changes.to_csv(tables_dir / "r_max_change_points.csv", index=False)

    # Plots.
    _plot_train_metrics_and_schedules(
        df=df,
        out_path=plots_dir / "train_metrics_and_schedules.png",
        smooth_window=int(args.smooth_window),
        title_prefix=args.title_prefix,
    )

    _plot_sparse_series(
        df=df,
        y="full_game_eval/overall/acpl",
        out_path=plots_dir / "full_game_eval__overall__acpl.png",
        title=f"{args.title_prefix}full_game_eval/overall/acpl",
        marker="o",
    )

    # Val: accuracy + format validity over time.
    for y in (
        "val-core/local/chess_puzzles/acc/mean@1",
        "val-core/local/chess_puzzles_shuffled/acc/mean@1",
        "val-aux/local/chess_puzzles/format_reward/mean@1",
        "val-aux/local/chess_puzzles_shuffled/format_reward/mean@1",
    ):
        _plot_sparse_series(
            df=df,
            y=y,
            out_path=plots_dir / f"{y.replace('/', '__')}.png",
            title=f"{args.title_prefix}{y}",
            marker="o",
        )

    print(f"[OK] wrote investigation bundle to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

