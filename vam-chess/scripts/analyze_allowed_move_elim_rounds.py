#!/usr/bin/env python3
"""
Analyze iterative allowed-move elimination ("allowed_move_elim") JSONL logs.

This is designed to work offline against a locally-downloaded W&B run evidence
folder produced by `scripts/download_wandb_run_evidence.py`.

Inputs (expected):
  - <wandb_evidence_dir>/files/allowed_move_elim_rounds/<step>_round<r>.jsonl
  - Training dataset parquet (to recover μ maps) with unique FEN per row.

Outputs:
  - A per-(step, round, prompt_idx) table (parquet + csv.gz)
  - Per-step/per-round summaries
  - R_max - 1 counterfactual summaries
  - Plots under a gitignored `outputs/` directory
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROUND_FILE_RE = re.compile(r"^(?P<step>\d+)_round(?P<round>\d+)\.jsonl$")

# Prompt parsing:
# - Older selection prompts use "Current FEN string:".
# - Newer selection prompts (used by allowed_move_elim as of 2026-01) use "Position (FEN):".
# We support both, case-insensitively, and allow leading whitespace on the line.
FEN_RE = re.compile(
    r"^\s*(?:Current FEN string|Position\s*\(FEN\)):\s*(?P<fen>.+?)\s*$",
    flags=re.MULTILINE | re.IGNORECASE,
)
ALLOWED_RE = re.compile(
    r"^\s*Allowed moves\s*\(UCI\):\s*(?P<moves>.+?)\s*$",
    flags=re.MULTILINE | re.IGNORECASE,
)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _normalize_move(m: Any) -> str:
    return str(m or "").strip().lower()


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def parse_prompt_fen_and_allowed_moves(prompt_text: str) -> tuple[str, list[str]]:
    fen_m = FEN_RE.search(prompt_text or "")
    if not fen_m:
        raise ValueError("Could not parse FEN from prompt text.")
    fen = fen_m.group("fen").strip()

    allowed_m = ALLOWED_RE.search(prompt_text or "")
    if not allowed_m:
        raise ValueError("Could not parse allowed moves from prompt text.")
    raw = allowed_m.group("moves").strip()
    moves = [_normalize_move(x) for x in raw.split(",")]
    moves = [m for m in moves if m]
    moves = _dedupe_preserve_order(moves)
    return fen, moves


def load_fen_to_mu_map(dataset_parquet: Path) -> dict[str, dict[str, float]]:
    table = pq.read_table(dataset_parquet, columns=["reward_model"])
    rows = table.to_pylist()
    fen_to_mu: dict[str, dict[str, float]] = {}

    for row in rows:
        rm = row.get("reward_model") or {}
        if not isinstance(rm, dict):
            continue
        fen = str(rm.get("fen") or "").strip()
        if not fen:
            continue
        mu_raw = rm.get("move_expected_scores_json") or rm.get("move_values_json")
        if not mu_raw:
            raise ValueError(f"Missing mu map JSON for fen={fen!r}")
        try:
            mu_obj = json.loads(mu_raw)
        except Exception as e:
            raise ValueError(f"Failed to parse mu map JSON for fen={fen!r}: {e}") from e
        if not isinstance(mu_obj, dict):
            raise ValueError(f"mu map JSON is not a dict for fen={fen!r}")
        mu_map = {_normalize_move(k): float(v) for k, v in mu_obj.items()}
        fen_to_mu[fen] = mu_map

    # Sanity: unique FEN is assumed for joining logs -> dataset
    if len(fen_to_mu) != len(rows):
        raise ValueError(
            f"FEN is not unique in dataset (rows={len(rows)}, unique_fens={len(fen_to_mu)}). "
            "Joining logs by FEN would be ambiguous."
        )
    return fen_to_mu


def iter_round_files(logs_dir: Path) -> list[tuple[int, int, Path]]:
    out: list[tuple[int, int, Path]] = []
    for p in sorted(logs_dir.glob("*.jsonl")):
        m = ROUND_FILE_RE.match(p.name)
        if not m:
            continue
        step = int(m.group("step"))
        round_idx = int(m.group("round"))
        out.append((step, round_idx, p))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


@dataclass
class GroupAgg:
    step: int
    round_idx: int
    prompt_idx: int
    r_max: int
    b_size: int
    fen: str
    allowed_moves: list[str]
    target_move: str
    gt_uci: str

    n_seq: int = 0
    n_penalty: int = 0
    n_format_error: int = 0
    n_out_of_subset: int = 0
    n_in_subset: int = 0
    scores: list[float] = None  # type: ignore[assignment]
    pred_moves: list[str] = None  # type: ignore[assignment]
    pred_moves_valid: list[str] = None  # type: ignore[assignment]
    success_flag_any: bool = False
    forced_accept_any: bool = False
    accepted_any: bool = False

    def __post_init__(self) -> None:
        self.scores = []
        self.pred_moves = []
        self.pred_moves_valid = []

    def add_record(self, rec: dict[str, Any]) -> None:
        self.n_seq += 1

        score = _safe_float(rec.get("score"))
        self.scores.append(score)

        pred = _normalize_move(rec.get("pred_move"))
        self.pred_moves.append(pred)

        penalty_applied = bool(rec.get("penalty_applied", False))
        penalty_reason = str(rec.get("penalty_reason") or "")
        in_subset = bool(rec.get("in_subset", False))
        if in_subset:
            self.n_in_subset += 1
        if penalty_applied:
            self.n_penalty += 1
            if penalty_reason == "format_error":
                self.n_format_error += 1
            if penalty_reason == "out_of_subset":
                self.n_out_of_subset += 1
        else:
            if pred and in_subset:
                self.pred_moves_valid.append(pred)

        self.success_flag_any = self.success_flag_any or bool(rec.get("allowed_move_elim_success", False))
        self.forced_accept_any = self.forced_accept_any or bool(rec.get("allowed_move_elim_forced_accept", False))
        self.accepted_any = self.accepted_any or bool(rec.get("allowed_move_elim_accepted", False))

    def to_row(self, mu_map: dict[str, float]) -> dict[str, Any]:
        allowed_mus = [_safe_float(mu_map.get(m)) for m in self.allowed_moves]
        allowed_mus = [v for v in allowed_mus if np.isfinite(v)]
        if not allowed_mus:
            allowed_mu_mean = float("nan")
            allowed_mu_median = float("nan")
            allowed_mu_best = float("nan")
            allowed_mu_second_best = float("nan")
        else:
            allowed_mu_mean = float(np.mean(allowed_mus))
            allowed_mu_median = float(np.median(allowed_mus))
            allowed_mu_best = float(np.max(allowed_mus))
            uniq_sorted = sorted(set(allowed_mus), reverse=True)
            allowed_mu_second_best = float(uniq_sorted[1]) if len(uniq_sorted) >= 2 else float("nan")

        n_unique_pred_valid = len(set(self.pred_moves_valid))
        n_unique_pred_all = len(set(m for m in self.pred_moves if m))
        score_std = float(np.std(self.scores)) if self.scores else float("nan")
        n_unique_scores = len(set(self.scores))

        hit_target = any(m == self.target_move for m in self.pred_moves_valid)
        hit_gt = any(m == self.gt_uci for m in self.pred_moves_valid)

        return {
            "step": int(self.step),
            "round": int(self.round_idx),
            "prompt_idx": int(self.prompt_idx),
            "r_max": int(self.r_max),
            "b_size": int(self.b_size),
            "allowed_len_parsed": int(len(self.allowed_moves)),
            "fen": str(self.fen),
            "target_move": str(self.target_move),
            "gt_uci": str(self.gt_uci),
            "target_eq_gt": bool(self.target_move and self.target_move == self.gt_uci),
            "success": bool(self.success_flag_any),
            "forced_accept": bool(self.forced_accept_any),
            "accepted": bool(self.accepted_any),
            "n_seq": int(self.n_seq),
            "n_penalty": int(self.n_penalty),
            "n_format_error": int(self.n_format_error),
            "n_out_of_subset": int(self.n_out_of_subset),
            "n_in_subset": int(self.n_in_subset),
            "penalty_rate": float(self.n_penalty / self.n_seq) if self.n_seq else float("nan"),
            "in_subset_rate": float(self.n_in_subset / self.n_seq) if self.n_seq else float("nan"),
            "score_mean": float(np.mean(self.scores)) if self.scores else float("nan"),
            "score_std": score_std,
            "n_unique_scores": int(n_unique_scores),
            "n_unique_pred_moves_valid": int(n_unique_pred_valid),
            "n_unique_pred_moves_all": int(n_unique_pred_all),
            "hit_target_move": bool(hit_target),
            "hit_gt_uci": bool(hit_gt),
            "allowed_mu_mean": float(allowed_mu_mean),
            "allowed_mu_median": float(allowed_mu_median),
            "allowed_mu_best": float(allowed_mu_best),
            "allowed_mu_second_best": float(allowed_mu_second_best),
        }


def parse_round_file(
    jsonl_path: Path, *, step: int, round_idx: int, fen_to_mu: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    groups: dict[int, GroupAgg] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            prompt_idx = _safe_int(rec.get("allowed_move_elim_prompt_idx"), default=-1)
            if prompt_idx < 0:
                raise ValueError(f"Missing/invalid allowed_move_elim_prompt_idx in {jsonl_path}")

            if prompt_idx not in groups:
                prompt_text = str(rec.get("input") or "")
                fen, allowed_moves = parse_prompt_fen_and_allowed_moves(prompt_text)
                b_size = _safe_int(rec.get("allowed_move_elim_b_size"), default=-1)
                r_max = _safe_int(rec.get("allowed_move_elim_r_max"), default=-1)
                target_move = _normalize_move(rec.get("target_move"))
                gt_uci = _normalize_move(rec.get("gt_uci"))

                if fen not in fen_to_mu:
                    raise ValueError(f"FEN from logs not found in dataset: {fen!r}")
                groups[prompt_idx] = GroupAgg(
                    step=step,
                    round_idx=round_idx,
                    prompt_idx=prompt_idx,
                    r_max=r_max,
                    b_size=b_size,
                    fen=fen,
                    allowed_moves=allowed_moves,
                    target_move=target_move,
                    gt_uci=gt_uci,
                )

            groups[prompt_idx].add_record(rec)

    out_rows: list[dict[str, Any]] = []
    for prompt_idx, agg in groups.items():
        mu_map = fen_to_mu[agg.fen]
        row = agg.to_row(mu_map)
        out_rows.append(row)
    return out_rows


def plot_r_max(step_summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(step_summary["step"], step_summary["r_max"], linewidth=1.5)
    ax.set_title("R_max schedule (from logs)")
    ax.set_xlabel("global step")
    ax.set_ylabel("r_max")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "r_max_by_step.png", dpi=200)
    plt.close(fig)


def plot_step_round_lines(step_round: pd.DataFrame, *, metric: str, out_path: Path, title: str, ylabel: str) -> None:
    pivot = step_round.pivot(index="step", columns="round", values=metric)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for r in sorted(pivot.columns):
        ax.plot(pivot.index, pivot[r], linewidth=1.2, label=f"round{int(r)}")
    ax.set_title(title)
    ax.set_xlabel("global step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(pivot.columns) <= 6:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_hist(series: pd.Series, *, out_path: Path, title: str, xlabel: str, bins: int = 50) -> None:
    vals = series.dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(vals, bins=bins, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True, help="W&B run id (e.g. uq2t8uqw)")
    ap.add_argument("--wandb_evidence_dir", required=True)
    ap.add_argument("--dataset_parquet", required=True, help="Training parquet used by the run (for μ maps).")
    ap.add_argument("--max_steps", type=int, default=None, help="Optional cap on unique steps to parse.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    wandb_evidence_dir = Path(args.wandb_evidence_dir)
    logs_dir = wandb_evidence_dir / "files" / "allowed_move_elim_rounds"
    if not logs_dir.exists():
        raise SystemExit(f"Missing logs dir: {logs_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_parquet = Path(args.dataset_parquet)
    if not dataset_parquet.exists():
        raise SystemExit(f"Missing dataset parquet: {dataset_parquet}")

    print(f"[LOAD] dataset mu maps: {dataset_parquet}")
    fen_to_mu = load_fen_to_mu_map(dataset_parquet)
    print(f"[LOAD] dataset rows: {len(fen_to_mu)} (unique FENs)")

    round_files = iter_round_files(logs_dir)
    if not round_files:
        raise SystemExit(f"No allowed_move_elim_rounds JSONLs found under: {logs_dir}")

    if args.max_steps is not None:
        max_steps = int(args.max_steps)
        steps_seen: set[int] = set()
        filtered: list[tuple[int, int, Path]] = []
        for step, round_idx, p in round_files:
            if step not in steps_seen and len(steps_seen) >= max_steps:
                continue
            steps_seen.add(step)
            filtered.append((step, round_idx, p))
        round_files = filtered
        print(f"[INFO] max_steps={max_steps} -> parsing {len(steps_seen)} unique steps")

    print(f"[LOAD] parsing {len(round_files)} round files from {logs_dir}")

    group_rows: list[dict[str, Any]] = []
    for step, round_idx, p in round_files:
        print(f"[PARSE] step={step} round={round_idx} file={p.name}")
        group_rows.extend(parse_round_file(p, step=step, round_idx=round_idx, fen_to_mu=fen_to_mu))

    df = pd.DataFrame(group_rows)
    if df.empty:
        raise SystemExit("Parsed 0 group rows; unexpected.")

    df = df.sort_values(["step", "round", "prompt_idx"]).reset_index(drop=True)
    df.to_parquet(out_dir / "group_metrics.parquet", index=False)
    df.to_csv(out_dir / "group_metrics.csv.gz", index=False, compression="gzip")

    # Attach next-round metrics for advancing prompts (within the same global step).
    next_cols = ["step", "round", "prompt_idx", "b_size", "allowed_mu_mean", "allowed_mu_best", "allowed_len_parsed"]
    df_next = df[next_cols].copy()
    df_next["round"] = df_next["round"] - 1
    df_next = df_next.rename(
        columns={
            "b_size": "b_size_after",
            "allowed_mu_mean": "allowed_mu_mean_after",
            "allowed_mu_best": "allowed_mu_best_after",
            "allowed_len_parsed": "allowed_len_parsed_after",
        }
    )
    df_merged = df.merge(df_next, on=["step", "round", "prompt_idx"], how="left")
    df_merged["advanced"] = df_merged["b_size_after"].notna()
    df_merged["delta_b"] = df_merged["b_size"] - df_merged["b_size_after"]
    df_merged["delta_mu_mean"] = df_merged["allowed_mu_mean_after"] - df_merged["allowed_mu_mean"]
    df_merged.to_parquet(out_dir / "group_metrics_with_next.parquet", index=False)

    # Step/round summary table.
    step_round = (
        df_merged.groupby(["step", "round"], as_index=False)
        .agg(
            r_max=("r_max", "max"),
            prompt_count=("prompt_idx", "count"),
            success_count=("success", "sum"),
            forced_accept_count=("forced_accept", "sum"),
            accepted_count=("accepted", "sum"),
            advanced_count=("advanced", "sum"),
            avg_b=("b_size", "mean"),
            avg_score_std=("score_std", "mean"),
            total_seq=("n_seq", "sum"),
            total_penalty=("n_penalty", "sum"),
            total_format_error=("n_format_error", "sum"),
            total_out_of_subset=("n_out_of_subset", "sum"),
            total_in_subset=("n_in_subset", "sum"),
            mean_unique_pred_valid=("n_unique_pred_moves_valid", "mean"),
            mean_delta_b_adv=("delta_b", "mean"),
            mean_delta_mu_mean_adv=("delta_mu_mean", "mean"),
        )
        .sort_values(["step", "round"])
    )
    # Derived rates.
    step_round["advanced_frac"] = step_round["advanced_count"] / step_round["prompt_count"].clip(lower=1)
    step_round["success_frac"] = step_round["success_count"] / step_round["prompt_count"].clip(lower=1)
    step_round["penalty_rate_seq"] = step_round["total_penalty"] / step_round["total_seq"].clip(lower=1)
    step_round["format_error_rate_seq"] = step_round["total_format_error"] / step_round["total_seq"].clip(lower=1)
    step_round["out_of_subset_rate_seq"] = step_round["total_out_of_subset"] / step_round["total_seq"].clip(lower=1)
    step_round.to_csv(out_dir / "step_round_summary.csv", index=False)

    # Per-step summary (r_max schedule + overall group stats).
    step_summary = (
        step_round.groupby("step", as_index=False)
        .agg(r_max=("r_max", "max"), total_prompts=("prompt_count", "max"), total_groups=("prompt_count", "sum"))
        .sort_values("step")
    )
    step_summary.to_csv(out_dir / "step_summary.csv", index=False)

    plot_r_max(step_summary, out_dir)
    plot_step_round_lines(
        step_round,
        metric="prompt_count",
        out_path=out_dir / "prompt_count_by_round.png",
        title="Prompts entering each round (per step)",
        ylabel="prompt_count",
    )
    plot_step_round_lines(
        step_round,
        metric="success_frac",
        out_path=out_dir / "success_frac_by_round.png",
        title="Per-round success fraction (success_count / prompt_count)",
        ylabel="success_frac",
    )
    plot_step_round_lines(
        step_round,
        metric="advanced_frac",
        out_path=out_dir / "advanced_frac_by_round.png",
        title="Per-round advance fraction (advanced_count / prompt_count)",
        ylabel="advanced_frac",
    )
    plot_step_round_lines(
        step_round,
        metric="avg_b",
        out_path=out_dir / "avg_b_size_by_round.png",
        title="Average allowed-move set size entering each round",
        ylabel="avg |B_i|",
    )
    plot_step_round_lines(
        step_round,
        metric="mean_delta_b_adv",
        out_path=out_dir / "delta_b_adv_by_round.png",
        title="Mean |B_i| shrink for advancing prompts (round r -> r+1)",
        ylabel="Δ|B_i|",
    )
    plot_step_round_lines(
        step_round,
        metric="mean_delta_mu_mean_adv",
        out_path=out_dir / "delta_mu_mean_adv_by_round.png",
        title="Mean Δ mean(μ) for advancing prompts (round r -> r+1)",
        ylabel="Δ mean(μ)",
    )
    plot_step_round_lines(
        step_round,
        metric="penalty_rate_seq",
        out_path=out_dir / "penalty_rate_seq_by_round.png",
        title="Penalty-applied rate (sequence-level) by round",
        ylabel="penalty_rate_seq",
    )

    # How often GRPO would see a zero-variance group (score std == 0).
    zero_std = (
        df.groupby(["step", "round"], as_index=False)
        .agg(zero_score_std_frac=("score_std", lambda s: float((s == 0.0).mean())))
        .sort_values(["step", "round"])
    )
    plot_step_round_lines(
        zero_std,
        metric="zero_score_std_frac",
        out_path=out_dir / "zero_score_std_frac_by_round.png",
        title="Fraction of groups with score_std == 0 (no GRPO signal)",
        ylabel="zero_score_std_frac",
    )
    plot_hist(
        df["score_std"],
        out_path=out_dir / "score_std_hist.png",
        title="Distribution of per-group score_std",
        xlabel="score_std",
        bins=60,
    )
    plot_hist(
        df_merged.loc[df_merged["advanced"] == True, "delta_mu_mean"],  # noqa: E712
        out_path=out_dir / "delta_mu_mean_hist_advancing.png",
        title="Distribution of Δ mean(μ) for advancing prompts",
        xlabel="Δ mean(μ)",
        bins=60,
    )

    # Counterfactual: R_max - 1 (terminate one round earlier).
    # We compute "success round" as the minimum round where success==True.
    per_prompt = (
        df.groupby(["step", "prompt_idx"], as_index=False)
        .agg(
            r_max=("r_max", "max"),
            max_round_seen=("round", "max"),
            success_any=("success", "max"),
        )
        .sort_values(["step", "prompt_idx"])
    )
    success_rounds = (
        df[df["success"] == True]  # noqa: E712
        .groupby(["step", "prompt_idx"], as_index=False)["round"]
        .min()
        .rename(columns={"round": "success_round"})
    )
    per_prompt = per_prompt.merge(success_rounds, on=["step", "prompt_idx"], how="left")
    per_prompt["success_round"] = per_prompt["success_round"].fillna(0).astype(int)
    per_prompt["r_max_cf"] = per_prompt["r_max"].apply(lambda r: max(1, int(r) - 1))
    per_prompt["success_cf"] = per_prompt.apply(
        lambda r: bool(r["success_any"]) and int(r["success_round"]) > 0 and int(r["success_round"]) <= int(r["r_max_cf"]),
        axis=1,
    )
    per_prompt["lost_success_last_round"] = per_prompt.apply(
        lambda r: bool(r["success_any"]) and int(r["success_round"]) == int(r["r_max"]) and int(r["r_max"]) > int(r["r_max_cf"]),
        axis=1,
    )
    per_prompt["groups_base"] = per_prompt["max_round_seen"]
    per_prompt["groups_cf"] = per_prompt.apply(
        lambda r: min(int(r["groups_base"]), int(r["r_max_cf"])), axis=1
    )

    cf_summary = (
        per_prompt.groupby("step", as_index=False)
        .agg(
            r_max=("r_max", "max"),
            r_max_cf=("r_max_cf", "max"),
            prompts=("prompt_idx", "count"),
            success_base=("success_any", "sum"),
            success_cf=("success_cf", "sum"),
            lost_success_last_round=("lost_success_last_round", "sum"),
            total_groups_base=("groups_base", "sum"),
            total_groups_cf=("groups_cf", "sum"),
        )
        .sort_values("step")
    )
    cf_summary["success_rate_base"] = cf_summary["success_base"] / cf_summary["prompts"].clip(lower=1)
    cf_summary["success_rate_cf"] = cf_summary["success_cf"] / cf_summary["prompts"].clip(lower=1)
    cf_summary["groups_reduction_frac"] = 1.0 - (
        cf_summary["total_groups_cf"] / cf_summary["total_groups_base"].clip(lower=1)
    )
    cf_summary.to_csv(out_dir / "rmax_minus1_counterfactual_by_step.csv", index=False)

    overall = {
        "run_id": str(args.run_id),
        "dataset_parquet": str(dataset_parquet),
        "logs_dir": str(logs_dir),
        "parsed_group_rows": int(len(df)),
        "unique_steps": int(df["step"].nunique()),
        "unique_prompts_total": int(per_prompt.shape[0]),
        "overall_success_rate_base": float(per_prompt["success_any"].mean()),
        "overall_success_rate_cf_rmax_minus1": float(per_prompt["success_cf"].mean()),
        "overall_lost_success_last_round_frac": float(per_prompt["lost_success_last_round"].mean()),
        "overall_groups_reduction_frac": float(
            1.0 - (per_prompt["groups_cf"].sum() / max(1, per_prompt["groups_base"].sum()))
        ),
        "overall_mean_score_std": float(df["score_std"].mean()),
        "overall_zero_score_std_frac": float((df["score_std"] == 0.0).mean()),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"[OK] wrote outputs under: {out_dir}")
    print(json.dumps(overall, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
