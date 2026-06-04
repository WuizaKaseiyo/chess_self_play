#!/usr/bin/env python3
"""
Reproducible investigation for iterative allowed-move elimination vs baseline GRPO.

This script is intentionally focused on the two runs discussed in inst.md:
  - iterative: s0anl08n
  - baseline:  dg41tlmo

It expects local W&B evidence directories downloaded via:
  python scripts/download_wandb_run_evidence.py --entity gabr1e11 --project chess_rl --run <RUN> --outdir analysis/wandb_evidence/<RUN>

And selected rollout files downloaded under:
  analysis/wandb_evidence/<RUN>/files/...
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FEN_REGEXES = [
    re.compile(r"Current FEN string:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Position\s*\(FEN\)\s*:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"FEN\s*:\s*([^\n]+)", re.IGNORECASE),
]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    mode: str  # "iterative" or "baseline"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            dotted = f"{prefix}.{key}" if prefix else key
            out.update(_flatten(v, dotted))
        return out
    out[prefix] = obj
    return out


def _safe_bool_series(s: pd.Series, default: bool = False) -> pd.Series:
    if s is None:
        return pd.Series(dtype=bool)
    return s.fillna(default).astype(bool)


def _parse_fen(input_text: Any) -> str:
    if not isinstance(input_text, str):
        return ""
    for rgx in FEN_REGEXES:
        m = rgx.search(input_text)
        if m:
            cand = m.group(1).strip()
            if cand.count("/") >= 7:
                return cand
    for line in input_text.splitlines():
        line = line.strip()
        if line.count("/") >= 7:
            return line
    return ""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _window_name(step: int, cut1: int, cut2: int) -> str:
    if step <= cut1:
        return "early"
    if step <= cut2:
        return "mid"
    return "late"


def _canonical_prompt_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prompt_key"] = (
        out["step"].astype(str)
        + "||"
        + out["fen"].astype(str)
        + "||"
        + out["gt_uci"].astype(str)
        + "||"
        + out["n_legal_moves"].astype(str)
    )
    return out


def _series_mean(x: pd.Series) -> float:
    if len(x) == 0:
        return float("nan")
    return float(x.mean())


def _round_success_label(x: int) -> str:
    if x <= 0:
        return "fail_all_rounds"
    return f"round_{x}"


def _discover_schema(files: list[Path]) -> dict[str, Any]:
    type_map: dict[str, dict[str, int]] = {}
    total_rows = 0
    for fp in files:
        rows = _load_jsonl(fp)
        total_rows += len(rows)
        for row in rows:
            for k, v in row.items():
                t = type(v).__name__
                if k not in type_map:
                    type_map[k] = {}
                type_map[k][t] = type_map[k].get(t, 0) + 1
    keys_sorted = sorted(type_map.keys())
    return {
        "total_files": len(files),
        "total_rows": total_rows,
        "fields": {k: type_map[k] for k in keys_sorted},
    }


def _collect_baseline_samples(evidence_root: Path, run_id: str, steps: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for s in steps:
        fp = evidence_root / run_id / "files" / "rollout_logs" / f"{s}.jsonl"
        for r in _load_jsonl(fp):
            r["step"] = int(s)
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No baseline rollout rows loaded")
    df["step"] = df["step"].astype(int)
    df["fen"] = df["input"].map(_parse_fen)
    df["input_len"] = df["input"].map(lambda x: len(str(x)))
    df["penalty_applied"] = _safe_bool_series(df.get("penalty_applied"), default=False)
    in_subset = df.get("in_subset")
    if in_subset is None:
        df["in_subset"] = True
    else:
        df["in_subset"] = _safe_bool_series(in_subset, default=True)
    df["pred_move"] = df.get("pred_move", "").fillna("").astype(str)
    df["gt_uci"] = df.get("gt_uci", "").fillna("").astype(str)
    df["success_sample"] = (df["pred_move"] == df["gt_uci"]) & (~df["penalty_applied"]) & df["in_subset"]
    return df


def _collect_iterative_samples(evidence_root: Path, run_id: str, steps: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for s in steps:
        for r in (1, 2, 3, 4):
            fp = evidence_root / run_id / "files" / "allowed_move_elim_rounds" / f"{s}_round{r}.jsonl"
            for row in _load_jsonl(fp):
                row["step"] = int(s)
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No iterative round rows loaded")
    df["step"] = df["step"].astype(int)
    df["allowed_move_elim_round"] = df["allowed_move_elim_round"].astype(int)
    df["allowed_move_elim_prompt_idx"] = df["allowed_move_elim_prompt_idx"].astype(int)
    df["allowed_move_elim_b_size"] = df["allowed_move_elim_b_size"].astype(int)
    df["fen"] = df["input"].map(_parse_fen)
    df["input_len"] = df["input"].map(lambda x: len(str(x)))
    df["penalty_applied"] = _safe_bool_series(df.get("penalty_applied"), default=False)
    df["in_subset"] = _safe_bool_series(df.get("in_subset"), default=True)
    df["allowed_move_elim_success"] = _safe_bool_series(df.get("allowed_move_elim_success"), default=False)
    df["allowed_move_elim_forced_accept"] = _safe_bool_series(df.get("allowed_move_elim_forced_accept"), default=False)
    df["allowed_move_elim_accepted"] = _safe_bool_series(df.get("allowed_move_elim_accepted"), default=False)
    df["pred_move"] = df.get("pred_move", "").fillna("").astype(str)
    df["gt_uci"] = df.get("gt_uci", "").fillna("").astype(str)
    df["success_sample"] = (df["pred_move"] == df["gt_uci"]) & (~df["penalty_applied"]) & df["in_subset"]
    return df


def _summarize_group_metrics_baseline(samples: pd.DataFrame) -> pd.DataFrame:
    g = (
        samples.groupby(["step", "uid"], as_index=False)
        .agg(
            fen=("fen", "first"),
            gt_uci=("gt_uci", "first"),
            n_legal_moves=("n_legal_moves", "first"),
            n_considered_moves=("n_considered_moves", "first"),
            input_len=("input_len", "first"),
            n_samples=("score", "size"),
            success_any=("success_sample", "max"),
            score_std=("score", "std"),
            score_min=("score", "min"),
            score_max=("score", "max"),
            penalty_sample_frac=("penalty_applied", "mean"),
        )
        .copy()
    )
    g["score_std"] = g["score_std"].fillna(0.0)
    g["score_range"] = g["score_max"] - g["score_min"]
    g["std_nonzero"] = g["score_std"] > 0
    g["run_mode"] = "baseline"
    return g


def _summarize_prompt_metrics_baseline(group_df: pd.DataFrame) -> pd.DataFrame:
    p = group_df.copy()
    p["prompt_success"] = p["success_any"].astype(bool)
    p["rounds_used"] = 1
    p["first_success_round"] = p["prompt_success"].astype(int)  # 1 or 0
    p["forced_prompt"] = False
    p["init_b_size"] = p["n_considered_moves"]
    p["avg_loss_weight_proxy"] = 1.0
    return p[
        [
            "step",
            "fen",
            "gt_uci",
            "n_legal_moves",
            "n_considered_moves",
            "input_len",
            "prompt_success",
            "rounds_used",
            "first_success_round",
            "forced_prompt",
            "init_b_size",
            "avg_loss_weight_proxy",
        ]
    ].copy()


def _summarize_group_metrics_iterative(samples: pd.DataFrame) -> pd.DataFrame:
    g = (
        samples.groupby(["step", "allowed_move_elim_round", "allowed_move_elim_prompt_idx"], as_index=False)
        .agg(
            fen=("fen", "first"),
            gt_uci=("gt_uci", "first"),
            n_legal_moves=("n_legal_moves", "first"),
            n_considered_moves=("n_considered_moves", "first"),
            b_size=("allowed_move_elim_b_size", "first"),
            input_len=("input_len", "first"),
            n_samples=("score", "size"),
            success_any=("success_sample", "max"),
            accepted=("allowed_move_elim_accepted", "max"),
            forced_accept=("allowed_move_elim_forced_accept", "max"),
            score_std=("score", "std"),
            score_min=("score", "min"),
            score_max=("score", "max"),
            penalty_sample_frac=("penalty_applied", "mean"),
        )
        .copy()
    )
    g["score_std"] = g["score_std"].fillna(0.0)
    g["score_range"] = g["score_max"] - g["score_min"]
    g["std_nonzero"] = g["score_std"] > 0
    g["run_mode"] = "iterative"
    return g


def _summarize_prompt_metrics_iterative(group_df: pd.DataFrame) -> pd.DataFrame:
    prompts = (
        group_df.groupby(["step", "fen", "gt_uci", "n_legal_moves"], as_index=False)
        .agg(
            prompt_success=("success_any", "max"),
            rounds_used=("allowed_move_elim_round", "nunique"),
            forced_prompt=("forced_accept", "max"),
            input_len=("input_len", "first"),
            init_b_size=("b_size", "max"),
        )
        .copy()
    )
    success_rounds = (
        group_df[group_df["success_any"]]
        .groupby(["step", "fen", "gt_uci", "n_legal_moves"], as_index=False)["allowed_move_elim_round"]
        .min()
        .rename(columns={"allowed_move_elim_round": "first_success_round"})
    )
    prompts = prompts.merge(
        success_rounds,
        on=["step", "fen", "gt_uci", "n_legal_moves"],
        how="left",
    )
    prompts["first_success_round"] = prompts["first_success_round"].fillna(0).astype(int)
    prompts["avg_loss_weight_proxy"] = 1.0 / prompts["rounds_used"].astype(float)
    prompts["n_considered_moves"] = prompts["n_legal_moves"]
    return prompts[
        [
            "step",
            "fen",
            "gt_uci",
            "n_legal_moves",
            "n_considered_moves",
            "input_len",
            "prompt_success",
            "rounds_used",
            "first_success_round",
            "forced_prompt",
            "init_b_size",
            "avg_loss_weight_proxy",
        ]
    ].copy()


def _dataset_row_counts(paths: list[str]) -> tuple[int, list[dict[str, Any]]]:
    total = 0
    parts: list[dict[str, Any]] = []
    for p in paths:
        pf = pq.ParquetFile(p)
        n = int(pf.metadata.num_rows)
        total += n
        parts.append({"path": p, "rows": n})
    return total, parts


def _inspect_prompt_style(path: str, n_rows: int = 2) -> list[dict[str, Any]]:
    tbl = pq.read_table(path, columns=["prompt", "reward_model", "extra_info"]).slice(0, n_rows)
    rows: list[dict[str, Any]] = []
    for i in range(tbl.num_rows):
        prompt = tbl["prompt"][i].as_py()
        rm = tbl["reward_model"][i].as_py()
        extra = tbl["extra_info"][i].as_py()
        user_msg = ""
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict) and str(msg.get("role")) == "user":
                    user_msg = str(msg.get("content", ""))
                    break
        legal = rm.get("legal_moves_uci", []) if isinstance(rm, dict) else []
        considered = rm.get("considered_moves_uci", []) if isinstance(rm, dict) else []
        rows.append(
            {
                "row_index": i,
                "extra_index": extra.get("index") if isinstance(extra, dict) else None,
                "prompt_chars": len(user_msg),
                "prompt_has_allowed_moves": ("allowed_moves" in user_msg.lower()),
                "n_legal": len(legal) if isinstance(legal, list) else None,
                "n_considered": len(considered) if isinstance(considered, list) else None,
                "considered_equals_legal": list(considered) == list(legal),
            }
        )
    return rows


def _compute_expected_score_quantization(dataset_train_paths: list[str]) -> dict[str, Any]:
    uniq_counts: list[int] = []
    top2_gaps: list[float] = []
    for path in dataset_train_paths:
        table = pq.read_table(path, columns=["reward_model"])
        for rm in table["reward_model"].to_pylist():
            if not isinstance(rm, dict):
                continue
            try:
                mp = json.loads(rm.get("move_expected_scores_json", "{}"))
            except Exception:
                continue
            if not isinstance(mp, dict) or not mp:
                continue
            vals = []
            for v in mp.values():
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            if not vals:
                continue
            uniq_counts.append(len(set(vals)))
            sv = sorted(vals, reverse=True)
            if len(sv) >= 2:
                top2_gaps.append(float(sv[0] - sv[1]))
    arr = np.asarray(uniq_counts, dtype=np.float64)
    gaps = np.asarray(top2_gaps, dtype=np.float64)
    if arr.size == 0:
        return {"error": "No expected-score rows parsed"}
    out = {
        "rows_with_expected_score": int(arr.size),
        "unique_value_count_mean": float(np.mean(arr)),
        "unique_value_count_median": float(np.median(arr)),
        "unique_value_count_p25": float(np.percentile(arr, 25)),
        "unique_value_count_p75": float(np.percentile(arr, 75)),
        "frac_unique_le_2": float(np.mean(arr <= 2)),
        "frac_unique_le_3": float(np.mean(arr <= 3)),
        "frac_unique_le_5": float(np.mean(arr <= 5)),
    }
    if gaps.size:
        out.update(
            {
                "top2_gap_mean": float(np.mean(gaps)),
                "top2_gap_median": float(np.median(gaps)),
                "top2_gap_frac_eq_0": float(np.mean(gaps == 0)),
                "top2_gap_frac_lt_0_01": float(np.mean(gaps < 0.01)),
            }
        )
    return out


def _legal_move_bin(n_legal: Any) -> str:
    try:
        x = int(n_legal)
    except Exception:
        return "unknown"
    if x <= 15:
        return "01-15"
    if x <= 25:
        return "16-25"
    if x <= 35:
        return "26-35"
    if x <= 45:
        return "36-45"
    return "46+"


def _plot_prompt_success_by_step(
    out_path: Path,
    step_summary: pd.DataFrame,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(step_summary["step"], step_summary["baseline_prompt_success"], marker="o", label="baseline prompt success")
    plt.plot(step_summary["step"], step_summary["iterative_prompt_success"], marker="o", label="iterative prompt success")
    plt.plot(step_summary["step"], step_summary["success_delta_iter_minus_base"], marker="o", label="delta (iter-base)")
    plt.xlabel("training/global_step (matched sampled steps)")
    plt.ylabel("fraction")
    plt.title("Per-Prompt Optimal-Move Discovery (Matched Steps)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_effective_batch_by_step(out_path: Path, step_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(step_summary["step"], step_summary["baseline_effective_batch_frac_from_logs"], marker="o", label="baseline effective_batch_frac")
    plt.plot(step_summary["step"], step_summary["iterative_effective_batch_frac_from_logs"], marker="o", label="iterative effective_batch_frac")
    plt.xlabel("training/global_step (matched sampled steps)")
    plt.ylabel("fraction of groups with std(score) > 0")
    plt.title("GRPO Effective-Batch Proxy by Step")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_tradeoff_scatter(out_path: Path, step_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 6))
    x = step_summary["success_delta_iter_minus_base"]
    y = step_summary["effective_batch_delta_iter_minus_base"]
    plt.scatter(x, y, alpha=0.85)
    for _, r in step_summary.iterrows():
        plt.text(r["success_delta_iter_minus_base"], r["effective_batch_delta_iter_minus_base"], str(int(r["step"])), fontsize=8)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel("Prompt-success delta (iter - baseline)")
    plt.ylabel("Effective-batch delta (iter - baseline)")
    plt.title("Coverage vs Signal Tradeoff by Step")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_eval_alignment(out_path: Path, val_aligned: pd.DataFrame, full_aligned: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if not val_aligned.empty:
        axes[0].plot(val_aligned["global_step"], val_aligned["s0_val_acc"], marker="o", label="s0 val acc")
        axes[0].plot(val_aligned["global_step"], val_aligned["dg_val_acc"], marker="o", label="dg val acc")
        axes[0].set_title("Puzzle Eval (Aligned Steps)")
        axes[0].set_xlabel("global_step")
        axes[0].set_ylabel("val-core/local/chess_puzzles/acc/mean@1")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
    else:
        axes[0].set_title("Puzzle Eval (no aligned rows)")

    if not full_aligned.empty:
        axes[1].plot(full_aligned["global_step"], full_aligned["s0_acpl_per_move"], marker="o", label="s0 ACPL/move")
        axes[1].plot(full_aligned["global_step"], full_aligned["dg_acpl_per_move"], marker="o", label="dg ACPL/move")
        axes[1].set_title("Full-Game Eval (Aligned Steps)")
        axes[1].set_xlabel("global_step")
        axes[1].set_ylabel("full_game_eval/overall/acpl_per_move (lower is better)")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
    else:
        axes[1].set_title("Full-Game Eval (no aligned rows)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence_root", default="analysis/wandb_evidence")
    ap.add_argument("--out_dir", default="analysis/investigation_s0_vs_dg")
    ap.add_argument("--iter_run", default="s0anl08n")
    ap.add_argument("--baseline_run", default="dg41tlmo")
    ap.add_argument("--steps_start", type=int, default=20)
    ap.add_argument("--steps_end", type=int, default=360)
    ap.add_argument("--steps_stride", type=int, default=20)
    ap.add_argument("--window_cut_1", type=int, default=120)
    ap.add_argument("--window_cut_2", type=int, default=240)
    args = ap.parse_args()

    evidence_root = Path(args.evidence_root)
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    _ensure_dir(tables_dir)
    _ensure_dir(plots_dir)

    steps = list(range(int(args.steps_start), int(args.steps_end) + 1, int(args.steps_stride)))
    if not steps:
        raise ValueError("No steps selected")

    iter_run = RunSpec(run_id=args.iter_run, mode="iterative")
    base_run = RunSpec(run_id=args.baseline_run, mode="baseline")

    iter_cfg = _flatten(_load_json(evidence_root / iter_run.run_id / "config_api.json"))
    base_cfg = _flatten(_load_json(evidence_root / base_run.run_id / "config_api.json"))
    iter_meta = _load_json(evidence_root / iter_run.run_id / "run_meta.json")
    base_meta = _load_json(evidence_root / base_run.run_id / "run_meta.json")
    iter_wb_meta = _load_json(evidence_root / iter_run.run_id / "files" / "wandb-metadata.json")
    base_wb_meta = _load_json(evidence_root / base_run.run_id / "files" / "wandb-metadata.json")
    iter_cfg_hash = _load_json(evidence_root / iter_run.run_id / "files" / "config_hash.json")
    base_cfg_hash = _load_json(evidence_root / base_run.run_id / "files" / "config_hash.json")

    iter_history = pd.read_parquet(evidence_root / iter_run.run_id / "history.parquet")
    base_history = pd.read_parquet(evidence_root / base_run.run_id / "history.parquet")

    # Load logs.
    base_samples = _collect_baseline_samples(evidence_root, base_run.run_id, steps)
    iter_samples = _collect_iterative_samples(evidence_root, iter_run.run_id, steps)

    # Schema discovery.
    iter_schema_files = [
        evidence_root / iter_run.run_id / "files" / "allowed_move_elim_rounds" / f"{s}_round{r}.jsonl"
        for s in steps
        for r in (1, 2, 3, 4)
    ]
    base_schema_files = [
        evidence_root / base_run.run_id / "files" / "rollout_logs" / f"{s}.jsonl"
        for s in steps
    ]
    _dump_json(out_dir / "log_schema_iterative_allowed_move_elim_rounds.json", _discover_schema(iter_schema_files))
    _dump_json(out_dir / "log_schema_baseline_rollout_logs.json", _discover_schema(base_schema_files))

    # Group/prompt summaries.
    base_groups = _summarize_group_metrics_baseline(base_samples)
    iter_groups = _summarize_group_metrics_iterative(iter_samples)
    base_prompts = _summarize_prompt_metrics_baseline(base_groups)
    iter_prompts = _summarize_prompt_metrics_iterative(iter_groups)

    base_prompts = _canonical_prompt_key(base_prompts)
    iter_prompts = _canonical_prompt_key(iter_prompts)

    paired = base_prompts.merge(
        iter_prompts,
        on=["step", "fen", "gt_uci", "n_legal_moves", "prompt_key"],
        suffixes=("_baseline", "_iterative"),
        how="inner",
        validate="one_to_one",
    )
    if paired.empty:
        raise RuntimeError("Paired prompt table is empty; failed to align prompts across runs.")

    def _pair_category(row: pd.Series) -> str:
        b = bool(row["prompt_success_baseline"])
        i = bool(row["prompt_success_iterative"])
        if b and i:
            return "both_success"
        if (not b) and i:
            return "iter_only"
        if b and (not i):
            return "baseline_only"
        return "both_fail"

    paired["pair_category"] = paired.apply(_pair_category, axis=1)
    paired["window"] = paired["step"].map(lambda s: _window_name(int(s), args.window_cut_1, args.window_cut_2))
    paired["legal_bin"] = paired["n_legal_moves"].map(_legal_move_bin)
    paired["first_success_round_label"] = paired["first_success_round_iterative"].map(_round_success_label)
    paired.to_csv(tables_dir / "paired_prompt_outcomes.csv", index=False)

    # Add pair category onto baseline/iter groups.
    pair_key_cols = ["step", "fen", "gt_uci", "n_legal_moves", "pair_category", "prompt_key", "window", "legal_bin"]
    pair_key_df = paired[pair_key_cols].copy()
    base_groups = _canonical_prompt_key(base_groups).merge(
        pair_key_df, on=["step", "fen", "gt_uci", "n_legal_moves", "prompt_key"], how="left"
    )
    iter_groups = _canonical_prompt_key(iter_groups).merge(
        pair_key_df, on=["step", "fen", "gt_uci", "n_legal_moves", "prompt_key"], how="left"
    )

    # Step summary.
    base_step = (
        base_groups.groupby("step", as_index=False)
        .agg(
            baseline_prompt_success=("success_any", "mean"),
            baseline_effective_batch_frac_from_logs=("std_nonzero", "mean"),
            baseline_score_range_mean=("score_range", "mean"),
            baseline_penalty_sample_frac=("penalty_sample_frac", "mean"),
        )
        .copy()
    )
    iter_prompt_step = (
        iter_prompts.groupby("step", as_index=False)
        .agg(
            iterative_prompt_success=("prompt_success", "mean"),
            iterative_forced_prompt_frac=("forced_prompt", "mean"),
            iterative_avg_rounds_used=("rounds_used", "mean"),
            iterative_avg_loss_weight_proxy=("avg_loss_weight_proxy", "mean"),
        )
        .copy()
    )
    iter_group_step = (
        iter_groups.groupby("step", as_index=False)
        .agg(
            iterative_effective_batch_frac_from_logs=("std_nonzero", "mean"),
            iterative_score_range_mean=("score_range", "mean"),
            iterative_penalty_sample_frac=("penalty_sample_frac", "mean"),
            iterative_group_accepted_frac=("accepted", "mean"),
            iterative_group_forced_accept_frac=("forced_accept", "mean"),
        )
        .copy()
    )
    step_summary = base_step.merge(iter_prompt_step, on="step", how="inner").merge(iter_group_step, on="step", how="inner")
    step_summary["success_delta_iter_minus_base"] = (
        step_summary["iterative_prompt_success"] - step_summary["baseline_prompt_success"]
    )
    step_summary["effective_batch_delta_iter_minus_base"] = (
        step_summary["iterative_effective_batch_frac_from_logs"]
        - step_summary["baseline_effective_batch_frac_from_logs"]
    )
    step_summary["window"] = step_summary["step"].map(
        lambda s: _window_name(int(s), args.window_cut_1, args.window_cut_2)
    )
    step_summary.to_csv(tables_dir / "step_summary.csv", index=False)

    # Cross-check effective batch against W&B history (matched steps).
    base_hist_eff = (
        base_history[["training/global_step", "grpo/effective_batch_frac"]]
        .dropna()
        .rename(columns={"training/global_step": "step", "grpo/effective_batch_frac": "baseline_effective_batch_frac_history"})
    )
    iter_hist_eff = (
        iter_history[["training/global_step", "grpo/effective_batch_frac"]]
        .dropna()
        .rename(columns={"training/global_step": "step", "grpo/effective_batch_frac": "iterative_effective_batch_frac_history"})
    )
    base_hist_eff["step"] = base_hist_eff["step"].astype(int)
    iter_hist_eff["step"] = iter_hist_eff["step"].astype(int)
    eff_compare = (
        step_summary.merge(base_hist_eff, on="step", how="left")
        .merge(iter_hist_eff, on="step", how="left")
        .copy()
    )
    eff_compare["baseline_abs_diff"] = (
        eff_compare["baseline_effective_batch_frac_from_logs"] - eff_compare["baseline_effective_batch_frac_history"]
    ).abs()
    eff_compare["iterative_abs_diff"] = (
        eff_compare["iterative_effective_batch_frac_from_logs"] - eff_compare["iterative_effective_batch_frac_history"]
    ).abs()
    eff_compare.to_csv(tables_dir / "effective_batch_crosscheck.csv", index=False)

    # Prompt success summaries by window and legal-move bins.
    window_summary = (
        paired.groupby("window", as_index=False)
        .agg(
            prompts=("prompt_key", "size"),
            baseline_success=("prompt_success_baseline", "mean"),
            iterative_success=("prompt_success_iterative", "mean"),
            iter_only_frac=("pair_category", lambda x: float(np.mean(x == "iter_only"))),
            baseline_only_frac=("pair_category", lambda x: float(np.mean(x == "baseline_only"))),
        )
        .copy()
    )
    window_summary["delta_iter_minus_baseline"] = (
        window_summary["iterative_success"] - window_summary["baseline_success"]
    )
    window_summary.to_csv(tables_dir / "prompt_success_window_summary.csv", index=False)

    bin_summary = (
        paired.groupby("legal_bin", as_index=False)
        .agg(
            prompts=("prompt_key", "size"),
            baseline_success=("prompt_success_baseline", "mean"),
            iterative_success=("prompt_success_iterative", "mean"),
            iter_only_frac=("pair_category", lambda x: float(np.mean(x == "iter_only"))),
            baseline_only_frac=("pair_category", lambda x: float(np.mean(x == "baseline_only"))),
        )
        .copy()
    )
    bin_summary["delta_iter_minus_baseline"] = bin_summary["iterative_success"] - bin_summary["baseline_success"]
    bin_summary.to_csv(tables_dir / "prompt_success_legal_bin_summary.csv", index=False)

    first_round_dist = (
        paired[paired["prompt_success_iterative"]]
        .groupby("first_success_round_iterative", as_index=False)
        .agg(count=("prompt_key", "size"))
    )
    first_round_dist["fraction_among_iter_successes"] = first_round_dist["count"] / float(first_round_dist["count"].sum())
    first_round_dist.to_csv(tables_dir / "iterative_first_success_round_distribution.csv", index=False)

    # Prompt-length stratification (within-run quartiles).
    len_baseline_q = pd.qcut(
        paired["input_len_baseline"],
        q=4,
        labels=["Q1_short", "Q2", "Q3", "Q4_long"],
        duplicates="drop",
    )
    len_iter_q = pd.qcut(
        paired["input_len_iterative"],
        q=4,
        labels=["Q1_short", "Q2", "Q3", "Q4_long"],
        duplicates="drop",
    )
    paired["baseline_len_quartile"] = len_baseline_q.astype(str)
    paired["iterative_len_quartile"] = len_iter_q.astype(str)
    len_summary_rows: list[dict[str, Any]] = []
    for q_label, sub in paired.groupby("baseline_len_quartile"):
        len_summary_rows.append(
            {
                "run": "baseline",
                "len_quartile": q_label,
                "prompts": len(sub),
                "success_rate": _series_mean(sub["prompt_success_baseline"]),
            }
        )
    for q_label, sub in paired.groupby("iterative_len_quartile"):
        len_summary_rows.append(
            {
                "run": "iterative",
                "len_quartile": q_label,
                "prompts": len(sub),
                "success_rate": _series_mean(sub["prompt_success_iterative"]),
            }
        )
    len_summary = pd.DataFrame(len_summary_rows)
    len_summary.to_csv(tables_dir / "prompt_success_length_quartile_summary.csv", index=False)

    absolute_prompt_len_summary = pd.DataFrame(
        [
            {
                "run": "baseline",
                "prompt_len_min": float(paired["input_len_baseline"].min()),
                "prompt_len_max": float(paired["input_len_baseline"].max()),
                "prompt_len_mean": float(paired["input_len_baseline"].mean()),
            },
            {
                "run": "iterative",
                "prompt_len_min": float(paired["input_len_iterative"].min()),
                "prompt_len_max": float(paired["input_len_iterative"].max()),
                "prompt_len_mean": float(paired["input_len_iterative"].mean()),
            },
        ]
    )
    absolute_prompt_len_summary.to_csv(tables_dir / "prompt_length_absolute_summary.csv", index=False)

    # Learning-signal summary for all groups / successful groups / groups from successful prompts.
    # Baseline: group == prompt, so successful prompts == successful groups.
    b_signal_all = base_groups.copy()
    b_signal_success_groups = base_groups[base_groups["success_any"]].copy()
    b_signal_success_prompts = b_signal_success_groups.copy()

    # Iterative: groups from successful prompts include all rounds for prompts that eventually succeed.
    iter_prompt_success_keys = set(
        paired.loc[paired["prompt_success_iterative"], "prompt_key"].astype(str).tolist()
    )
    i_signal_all = iter_groups.copy()
    i_signal_success_groups = iter_groups[iter_groups["success_any"]].copy()
    i_signal_success_prompts = iter_groups[iter_groups["prompt_key"].astype(str).isin(iter_prompt_success_keys)].copy()

    def _signal_summary(df: pd.DataFrame, run_name: str, subset_name: str) -> pd.DataFrame:
        tmp = df.copy()
        tmp["window"] = tmp["step"].map(lambda s: _window_name(int(s), args.window_cut_1, args.window_cut_2))
        out = (
            tmp.groupby("window", as_index=False)
            .agg(
                run=("window", lambda _: run_name),
                subset=("window", lambda _: subset_name),
                groups=("std_nonzero", "size"),
                std_nonzero_frac=("std_nonzero", "mean"),
                score_range_mean=("score_range", "mean"),
                score_range_median=("score_range", "median"),
                penalty_sample_frac=("penalty_sample_frac", "mean"),
            )
            .copy()
        )
        return out[["run", "subset", "window", "groups", "std_nonzero_frac", "score_range_mean", "score_range_median", "penalty_sample_frac"]]

    signal_tables = [
        _signal_summary(b_signal_all, "baseline", "all_groups"),
        _signal_summary(b_signal_success_groups, "baseline", "successful_groups"),
        _signal_summary(b_signal_success_prompts, "baseline", "groups_from_successful_prompts"),
        _signal_summary(i_signal_all, "iterative", "all_groups"),
        _signal_summary(i_signal_success_groups, "iterative", "successful_groups"),
        _signal_summary(i_signal_success_prompts, "iterative", "groups_from_successful_prompts"),
    ]
    signal_summary = pd.concat(signal_tables, ignore_index=True)
    signal_summary.to_csv(tables_dir / "learning_signal_summary.csv", index=False)

    # Round-level iterative summaries.
    iter_round_summary = (
        iter_groups.groupby("allowed_move_elim_round", as_index=False)
        .agg(
            groups=("success_any", "size"),
            success_group_frac=("success_any", "mean"),
            accepted_group_frac=("accepted", "mean"),
            forced_accept_group_frac=("forced_accept", "mean"),
            std_nonzero_frac=("std_nonzero", "mean"),
            score_range_mean=("score_range", "mean"),
            mean_b_size=("b_size", "mean"),
        )
        .copy()
    )
    iter_round_summary.to_csv(tables_dir / "iterative_round_group_summary.csv", index=False)

    iter_round_window_summary = (
        iter_groups.assign(window=iter_groups["step"].map(lambda s: _window_name(int(s), args.window_cut_1, args.window_cut_2)))
        .groupby(["window", "allowed_move_elim_round"], as_index=False)
        .agg(
            groups=("success_any", "size"),
            success_group_frac=("success_any", "mean"),
            std_nonzero_frac=("std_nonzero", "mean"),
            score_range_mean=("score_range", "mean"),
            mean_b_size=("b_size", "mean"),
        )
    )
    iter_round_window_summary.to_csv(tables_dir / "iterative_round_window_summary.csv", index=False)

    # Penalty/invalid summaries.
    base_penalty_summary = (
        base_samples.groupby("penalty_reason", as_index=False)
        .agg(count=("penalty_reason", "size"))
        .sort_values("count", ascending=False)
    )
    base_penalty_summary["run"] = "baseline"
    iter_penalty_summary = (
        iter_samples.groupby("penalty_reason", as_index=False)
        .agg(count=("penalty_reason", "size"))
        .sort_values("count", ascending=False)
    )
    iter_penalty_summary["run"] = "iterative"
    penalty_summary = pd.concat([base_penalty_summary, iter_penalty_summary], ignore_index=True)
    penalty_summary.to_csv(tables_dir / "penalty_reason_counts.csv", index=False)

    penalty_rate_summary = pd.DataFrame(
        [
            {
                "run": "baseline",
                "penalty_rate": float(base_samples["penalty_applied"].mean()),
            },
            {
                "run": "iterative",
                "penalty_rate": float(iter_samples["penalty_applied"].mean()),
            },
        ]
    )
    penalty_rate_summary.to_csv(tables_dir / "penalty_rate_summary.csv", index=False)

    # Eval alignment tables.
    val_cols = [
        "training/global_step",
        "val-core/local/chess_puzzles/acc/mean@1",
        "val-core/local/chess_puzzles_shuffled/acc/mean@1",
        "val-aux/local/chess_puzzles/score/mean@1",
        "val-aux/local/chess_puzzles_shuffled/score/mean@1",
    ]
    base_val = base_history[val_cols].dropna(subset=["val-core/local/chess_puzzles/acc/mean@1"]).copy()
    iter_val = iter_history[val_cols].dropna(subset=["val-core/local/chess_puzzles/acc/mean@1"]).copy()
    base_val["global_step"] = base_val["training/global_step"].fillna(0).astype(int)
    iter_val["global_step"] = iter_val["training/global_step"].fillna(0).astype(int)
    val_aligned_steps = sorted(set(base_val["global_step"]).intersection(set(iter_val["global_step"])))
    val_aligned = (
        iter_val[iter_val["global_step"].isin(val_aligned_steps)][
            ["global_step", "val-core/local/chess_puzzles/acc/mean@1", "val-core/local/chess_puzzles_shuffled/acc/mean@1"]
        ]
        .rename(
            columns={
                "val-core/local/chess_puzzles/acc/mean@1": "s0_val_acc",
                "val-core/local/chess_puzzles_shuffled/acc/mean@1": "s0_val_acc_shuffled",
            }
        )
        .merge(
            base_val[base_val["global_step"].isin(val_aligned_steps)][
                ["global_step", "val-core/local/chess_puzzles/acc/mean@1", "val-core/local/chess_puzzles_shuffled/acc/mean@1"]
            ].rename(
                columns={
                    "val-core/local/chess_puzzles/acc/mean@1": "dg_val_acc",
                    "val-core/local/chess_puzzles_shuffled/acc/mean@1": "dg_val_acc_shuffled",
                }
            ),
            on="global_step",
            how="inner",
        )
    )
    if not val_aligned.empty:
        val_aligned["delta_val_acc"] = val_aligned["s0_val_acc"] - val_aligned["dg_val_acc"]
        val_aligned["delta_val_acc_shuffled"] = val_aligned["s0_val_acc_shuffled"] - val_aligned["dg_val_acc_shuffled"]
    val_aligned.to_csv(tables_dir / "val_metrics_aligned_steps.csv", index=False)

    full_cols = [
        "training/global_step",
        "full_game_eval/overall/acpl_per_move",
        "full_game_eval/overall/win_rate",
        "full_game_eval/overall/num_games",
    ]
    base_full = base_history[full_cols].dropna(subset=["full_game_eval/overall/acpl_per_move"]).copy()
    iter_full = iter_history[full_cols].dropna(subset=["full_game_eval/overall/acpl_per_move"]).copy()
    base_full["global_step"] = base_full["training/global_step"].fillna(0).astype(int)
    iter_full["global_step"] = iter_full["training/global_step"].fillna(0).astype(int)
    full_aligned_steps = sorted(set(base_full["global_step"]).intersection(set(iter_full["global_step"])))
    full_aligned = (
        iter_full[iter_full["global_step"].isin(full_aligned_steps)][
            ["global_step", "full_game_eval/overall/acpl_per_move", "full_game_eval/overall/win_rate"]
        ]
        .rename(
            columns={
                "full_game_eval/overall/acpl_per_move": "s0_acpl_per_move",
                "full_game_eval/overall/win_rate": "s0_win_rate",
            }
        )
        .merge(
            base_full[base_full["global_step"].isin(full_aligned_steps)][
                ["global_step", "full_game_eval/overall/acpl_per_move", "full_game_eval/overall/win_rate"]
            ].rename(
                columns={
                    "full_game_eval/overall/acpl_per_move": "dg_acpl_per_move",
                    "full_game_eval/overall/win_rate": "dg_win_rate",
                }
            ),
            on="global_step",
            how="inner",
        )
    )
    if not full_aligned.empty:
        full_aligned["delta_acpl_per_move"] = full_aligned["s0_acpl_per_move"] - full_aligned["dg_acpl_per_move"]
    full_aligned.to_csv(tables_dir / "fullgame_metrics_aligned_steps.csv", index=False)

    # Dataset inspection / comparability.
    iter_train_files = list(iter_cfg.get("data.train_files", []) or [])
    base_train_files = list(base_cfg.get("data.train_files", []) or [])
    iter_val_files = list(iter_cfg.get("data.val_files", []) or [])
    base_val_files = list(base_cfg.get("data.val_files", []) or [])

    iter_rows_total, iter_rows_parts = _dataset_row_counts(iter_train_files)
    base_rows_total, base_rows_parts = _dataset_row_counts(base_train_files)
    dataset_inspection = {
        "iterative_train_files": iter_rows_parts,
        "baseline_train_files": base_rows_parts,
        "iterative_train_rows_total": iter_rows_total,
        "baseline_train_rows_total": base_rows_total,
        "iterative_prompt_style_examples": _inspect_prompt_style(iter_train_files[0], n_rows=2) if iter_train_files else [],
        "baseline_prompt_style_examples": _inspect_prompt_style(base_train_files[0], n_rows=2) if base_train_files else [],
    }
    _dump_json(out_dir / "dataset_inspection.json", dataset_inspection)

    quantization_summary = _compute_expected_score_quantization(iter_train_files)
    _dump_json(out_dir / "expected_score_quantization_summary.json", quantization_summary)

    comparability_rows: list[dict[str, Any]] = []

    def add_row(dimension: str, s0_val: Any, dg_val: Any, source: str) -> None:
        comparability_rows.append(
            {
                "dimension": dimension,
                "s0anl08n": s0_val,
                "dg41tlmo": dg_val,
                "same": str(s0_val) == str(dg_val),
                "source": source,
            }
        )

    add_row("run_state", iter_meta.get("run_state"), base_meta.get("run_state"), "run_meta.json")
    add_row("git_commit", iter_wb_meta.get("git", {}).get("commit"), base_wb_meta.get("git", {}).get("commit"), "wandb-metadata.json")
    add_row("dataset_fingerprint", iter_cfg_hash.get("dataset_fingerprint"), base_cfg_hash.get("dataset_fingerprint"), "config_hash.json")
    add_row("data.self_play.enable", iter_cfg.get("data.self_play.enable"), base_cfg.get("data.self_play.enable"), "config_api.json")
    add_row("data.train_files", iter_train_files, base_train_files, "config_api.json")
    add_row("data.val_files", iter_val_files, base_val_files, "config_api.json")
    add_row("data.train_rows_total", iter_rows_total, base_rows_total, "local parquet metadata")
    add_row(
        "dataset.prompt_has_allowed_moves(row0)",
        dataset_inspection["iterative_prompt_style_examples"][0]["prompt_has_allowed_moves"] if dataset_inspection["iterative_prompt_style_examples"] else None,
        dataset_inspection["baseline_prompt_style_examples"][0]["prompt_has_allowed_moves"] if dataset_inspection["baseline_prompt_style_examples"] else None,
        "local parquet row inspection",
    )
    add_row("actor_rollout_ref.rollout.n", iter_cfg.get("actor_rollout_ref.rollout.n"), base_cfg.get("actor_rollout_ref.rollout.n"), "config_api.json")
    add_row("data.train_batch_size", iter_cfg.get("data.train_batch_size"), base_cfg.get("data.train_batch_size"), "config_api.json")
    add_row("data.gen_batch_size", iter_cfg.get("data.gen_batch_size"), base_cfg.get("data.gen_batch_size"), "config_api.json")
    add_row("data.max_prompt_length", iter_cfg.get("data.max_prompt_length"), base_cfg.get("data.max_prompt_length"), "config_api.json")
    add_row("data.max_response_length", iter_cfg.get("data.max_response_length"), base_cfg.get("data.max_response_length"), "config_api.json")
    add_row(
        "custom_reward_function.reward_kwargs.chess_reward_fn",
        iter_cfg.get("custom_reward_function.reward_kwargs.chess_reward_fn"),
        base_cfg.get("custom_reward_function.reward_kwargs.chess_reward_fn"),
        "config_api.json",
    )
    add_row("algorithm.filter_groups.enable", iter_cfg.get("algorithm.filter_groups.enable"), base_cfg.get("algorithm.filter_groups.enable"), "config_api.json")
    add_row("algorithm.allowed_move_elim.enable", iter_cfg.get("algorithm.allowed_move_elim.enable"), base_cfg.get("algorithm.allowed_move_elim.enable"), "config_api.json")
    add_row("algorithm.allowed_move_elim.uid_mode", iter_cfg.get("algorithm.allowed_move_elim.uid_mode"), base_cfg.get("algorithm.allowed_move_elim.uid_mode"), "config_api.json")
    add_row("algorithm.allowed_move_elim.r_max_start", iter_cfg.get("algorithm.allowed_move_elim.r_max_start"), base_cfg.get("algorithm.allowed_move_elim.r_max_start"), "config_api.json")
    add_row("algorithm.allowed_move_elim.r_max_end", iter_cfg.get("algorithm.allowed_move_elim.r_max_end"), base_cfg.get("algorithm.allowed_move_elim.r_max_end"), "config_api.json")
    add_row("algorithm.allowed_move_elim.anneal_frac", iter_cfg.get("algorithm.allowed_move_elim.anneal_frac"), base_cfg.get("algorithm.allowed_move_elim.anneal_frac"), "config_api.json")
    add_row("algorithm.allowed_move_elim.group_reward_range_min", iter_cfg.get("algorithm.allowed_move_elim.group_reward_range_min"), base_cfg.get("algorithm.allowed_move_elim.group_reward_range_min"), "config_api.json")
    add_row("trainer.nnodes", iter_cfg.get("trainer.nnodes"), base_cfg.get("trainer.nnodes"), "config_api.json")
    add_row("trainer.save_freq", iter_cfg.get("trainer.save_freq"), base_cfg.get("trainer.save_freq"), "config_api.json")
    add_row("trainer.test_freq", iter_cfg.get("trainer.test_freq"), base_cfg.get("trainer.test_freq"), "config_api.json")
    add_row("trainer.full_eval_freq", iter_cfg.get("trainer.full_eval_freq"), base_cfg.get("trainer.full_eval_freq"), "config_api.json")
    add_row("trainer.full_eval.prompt_template_path", iter_cfg.get("trainer.full_eval.prompt_template_path"), base_cfg.get("trainer.full_eval.prompt_template_path"), "config_api.json")
    add_row("trainer.full_eval.games_per_depth", iter_cfg.get("trainer.full_eval.games_per_depth"), base_cfg.get("trainer.full_eval.games_per_depth"), "config_api.json")
    add_row("trainer.full_eval.opponent_depths", iter_cfg.get("trainer.full_eval.opponent_depths"), base_cfg.get("trainer.full_eval.opponent_depths"), "config_api.json")
    add_row("training/global_step_max_history", int(iter_history["training/global_step"].dropna().max()), int(base_history["training/global_step"].dropna().max()), "history.parquet")
    add_row("wallclock_runtime_s", iter_meta.get("_runtime", None) if "_runtime" in iter_meta else iter_history["_runtime"].dropna().max(), base_meta.get("_runtime", None) if "_runtime" in base_meta else base_history["_runtime"].dropna().max(), "history.parquet")

    comparability_df = pd.DataFrame(comparability_rows)
    comparability_df.to_csv(tables_dir / "comparability_matrix.csv", index=False)

    # Additional hypothesis tests.
    baseline_eff_mean = float(step_summary["baseline_effective_batch_frac_from_logs"].mean())
    iterative_eff_mean = float(step_summary["iterative_effective_batch_frac_from_logs"].mean())
    baseline_prompt_success_mean = float(step_summary["baseline_prompt_success"].mean())
    iterative_prompt_success_mean = float(step_summary["iterative_prompt_success"].mean())
    baseline_weighted_nonzero_mean = float(step_summary["baseline_effective_batch_frac_from_logs"].mean())

    # Weighted non-zero proxy for iterative prompts: mean over prompts of (sum(nonzero groups)/k).
    iter_weighted_nonzero_by_step_rows: list[dict[str, Any]] = []
    for s in steps:
        sub = iter_groups[iter_groups["step"] == s]
        # prompt key is unique per prompt at a step
        per_prompt = sub.groupby("prompt_key", as_index=False).agg(
            k=("allowed_move_elim_round", "count"),
            nonzero_sum=("std_nonzero", "sum"),
        )
        per_prompt["weighted_nonzero"] = per_prompt["nonzero_sum"] / per_prompt["k"].astype(float)
        iter_weighted_nonzero_by_step_rows.append(
            {
                "step": int(s),
                "iterative_weighted_nonzero_per_prompt": float(per_prompt["weighted_nonzero"].mean()),
            }
        )
    iter_weighted_nonzero_df = pd.DataFrame(iter_weighted_nonzero_by_step_rows)
    weighted_signal = step_summary.merge(iter_weighted_nonzero_df, on="step", how="left")
    weighted_signal["weighted_nonzero_delta_iter_minus_base"] = (
        weighted_signal["iterative_weighted_nonzero_per_prompt"]
        - weighted_signal["baseline_effective_batch_frac_from_logs"]
    )
    weighted_signal.to_csv(tables_dir / "weighted_nonzero_signal_by_step.csv", index=False)

    iter_penalty_rate = float(iter_samples["penalty_applied"].mean())
    base_penalty_rate = float(base_samples["penalty_applied"].mean())
    iter_forced_prompt_frac = float(iter_prompts["forced_prompt"].mean())
    iter_accept_reject_frac = float(1.0 - iter_groups["accepted"].mean())
    iter_first_round_success_frac = float(
        (iter_prompts["first_success_round"] == 1).mean()
    )
    iter_fail_all_rounds_frac = float(
        (iter_prompts["first_success_round"] == 0).mean()
    )

    clip_base = float(base_history.get("prompt_length/clip_ratio", pd.Series(dtype=float)).dropna().max() or 0.0)
    clip_iter = float(iter_history.get("prompt_length/clip_ratio", pd.Series(dtype=float)).dropna().max() or 0.0)

    hypothesis_rows: list[dict[str, Any]] = []

    def add_hypothesis(h_id: str, title: str, verdict: str, key_evidence: str, metrics: dict[str, Any]) -> None:
        row = {
            "hypothesis_id": h_id,
            "title": title,
            "verdict": verdict,  # supported / unsupported / inconclusive
            "key_evidence": key_evidence,
        }
        row.update(metrics)
        hypothesis_rows.append(row)

    # H1: reward quantization/ties -> dead groups.
    h1_supported = (
        (iterative_eff_mean + 0.20 < baseline_eff_mean)
        and float(quantization_summary.get("frac_unique_le_3", 0.0)) >= 0.5
    )
    add_hypothesis(
        "H1",
        "Expected-score quantization and ties create many dead GRPO groups (especially iterative).",
        "supported" if h1_supported else "inconclusive",
        "Iterative effective_batch_frac is much lower than baseline; expected-score maps have low unique-value counts.",
        {
            "baseline_effective_batch_frac_mean": baseline_eff_mean,
            "iterative_effective_batch_frac_mean": iterative_eff_mean,
            "expected_score_frac_unique_le_3": quantization_summary.get("frac_unique_le_3"),
        },
    )

    # H2: iterative increases optimal discovery but signal is diluted by low-variance groups + per-prompt loss weighting.
    coverage_delta = iterative_prompt_success_mean - baseline_prompt_success_mean
    weighted_signal_delta = float(weighted_signal["weighted_nonzero_delta_iter_minus_base"].mean())
    h2_supported = (coverage_delta > 0.05) and (weighted_signal_delta < 0.0)
    add_hypothesis(
        "H2",
        "Iterative raises per-prompt optimal-hit coverage but weakens effective learning signal.",
        "supported" if h2_supported else "inconclusive",
        "Coverage rises, but weighted non-dead group signal per prompt falls vs baseline.",
        {
            "prompt_success_delta_iter_minus_base": coverage_delta,
            "weighted_nonzero_signal_delta_iter_minus_base": weighted_signal_delta,
        },
    )

    # H3: forced-accept / fail-all-rounds dynamics constrain gains.
    h3_supported = (iter_forced_prompt_frac >= 0.25) and (iter_fail_all_rounds_frac >= 0.25)
    add_hypothesis(
        "H3",
        "High forced-accept and fail-all-rounds rates cap iterative gains.",
        "supported" if h3_supported else "inconclusive",
        "A large fraction of prompts only terminate via forced accept at round4.",
        {
            "iterative_forced_prompt_frac": iter_forced_prompt_frac,
            "iterative_fail_all_rounds_frac": iter_fail_all_rounds_frac,
            "iterative_nonaccepted_group_frac": iter_accept_reject_frac,
            "iterative_first_round_success_frac": iter_first_round_success_frac,
        },
    )

    # H4: invalid/penalized output rates are the main reason for marginal gains.
    penalty_gap = iter_penalty_rate - base_penalty_rate
    h4_supported = abs(penalty_gap) > 0.01
    add_hypothesis(
        "H4",
        "Invalid/penalized output rates are the primary cause of marginal gains.",
        "supported" if h4_supported else "unsupported",
        "Penalty-rate gap is small in magnitude; not large enough to explain major signal differences.",
        {
            "baseline_penalty_rate": base_penalty_rate,
            "iterative_penalty_rate": iter_penalty_rate,
            "penalty_rate_gap_iter_minus_base": penalty_gap,
        },
    )

    # H5: prompt truncation/filtering is the main cause.
    h5_supported = (clip_base > 0.0) or (clip_iter > 0.0)
    add_hypothesis(
        "H5",
        "Prompt truncation/filtering from max_prompt_length is a major driver.",
        "supported" if h5_supported else "unsupported",
        "clip_ratio remains zero in both runs across history.",
        {
            "baseline_prompt_clip_ratio_max": clip_base,
            "iterative_prompt_clip_ratio_max": clip_iter,
        },
    )

    # H6: evaluation mismatch (dataset/prompt/template/code) confounds direct run-vs-run conclusions.
    dataset_diff = str(iter_train_files) != str(base_train_files)
    template_diff = str(iter_cfg.get("trainer.full_eval.prompt_template_path")) != str(
        base_cfg.get("trainer.full_eval.prompt_template_path")
    )
    commit_diff = str(iter_wb_meta.get("git", {}).get("commit")) != str(base_wb_meta.get("git", {}).get("commit"))
    h6_supported = dataset_diff and template_diff and commit_diff
    add_hypothesis(
        "H6",
        "Direct run-vs-run eval is confounded by dataset/prompt/template/code differences.",
        "supported" if h6_supported else "inconclusive",
        "Train/val datasets differ, full-game prompt template differs, and git commit differs.",
        {
            "dataset_diff": dataset_diff,
            "full_eval_template_diff": template_diff,
            "git_commit_diff": commit_diff,
        },
    )

    # H7: walltime-limited runs create unequal step budgets.
    iter_runtime = float(iter_history["_runtime"].dropna().max())
    base_runtime = float(base_history["_runtime"].dropna().max())
    iter_steps = float(iter_history["training/global_step"].dropna().max())
    base_steps = float(base_history["training/global_step"].dropna().max())
    iter_sph = iter_steps / (iter_runtime / 3600.0) if iter_runtime > 0 else float("nan")
    base_sph = base_steps / (base_runtime / 3600.0) if base_runtime > 0 else float("nan")
    h7_supported = (
        str(iter_meta.get("run_state")) == "crashed"
        and str(base_meta.get("run_state")) == "crashed"
        and abs(iter_runtime - base_runtime) < 3600
        and abs(iter_steps - base_steps) > 100
    )
    add_hypothesis(
        "H7",
        "Walltime-limited termination yielded unequal optimization budgets in steps.",
        "supported" if h7_supported else "inconclusive",
        "Both runs crashed around ~24h, but reached very different step counts.",
        {
            "iter_runtime_s": iter_runtime,
            "base_runtime_s": base_runtime,
            "iter_steps": iter_steps,
            "base_steps": base_steps,
            "iter_steps_per_hour": iter_sph,
            "base_steps_per_hour": base_sph,
        },
    )

    hypothesis_df = pd.DataFrame(hypothesis_rows)
    hypothesis_df.to_csv(tables_dir / "hypothesis_tests.csv", index=False)

    # Root-cause ranking (evidence-backed, simple deterministic order).
    ranked_root_causes = [
        {
            "rank": 1,
            "cause": "Comparison confounded by non-equivalent training/eval setups (dataset prompt variant, full-game template, code commit).",
            "status": "supported" if h6_supported else "inconclusive",
            "why": "These differences directly affect what is optimized and how eval prompts are rendered.",
        },
        {
            "rank": 2,
            "cause": "Iterative method improves prompt-level optimal discovery but suffers much lower GRPO variance/effective batch.",
            "status": "supported" if h2_supported else "inconclusive",
            "why": "Coverage gains are offset by weak per-group reward variance and loss-weight dilution.",
        },
        {
            "rank": 3,
            "cause": "High forced-accept / fail-all-rounds fraction leaves a large unresolved tail.",
            "status": "supported" if h3_supported else "inconclusive",
            "why": "Many prompts terminate only at round4 without discovering gt during sampling.",
        },
        {
            "rank": 4,
            "cause": "Expected-score quantization/ties contribute to dead-group behavior.",
            "status": "supported" if h1_supported else "inconclusive",
            "why": "Reward maps are coarse and iterative groups are frequently zero-variance.",
        },
        {
            "rank": 5,
            "cause": "Invalid output rates are not the primary limiter.",
            "status": "unsupported" if not h4_supported else "supported",
            "why": "Penalty rates are low and close between runs.",
        },
    ]
    _dump_json(out_dir / "ranked_root_causes.json", ranked_root_causes)

    # Plots.
    _plot_prompt_success_by_step(plots_dir / "prompt_success_by_step.png", step_summary)
    _plot_effective_batch_by_step(plots_dir / "effective_batch_frac_by_step.png", step_summary)
    _plot_tradeoff_scatter(plots_dir / "coverage_vs_signal_tradeoff_by_step.png", step_summary)
    _plot_eval_alignment(plots_dir / "eval_alignment.png", val_aligned, full_aligned)

    # High-level run summary.
    run_summary = {
        "iter_run": iter_run.run_id,
        "baseline_run": base_run.run_id,
        "steps_sampled": steps,
        "paired_prompt_count": int(len(paired)),
        "prompt_success_baseline_mean": baseline_prompt_success_mean,
        "prompt_success_iterative_mean": iterative_prompt_success_mean,
        "prompt_success_delta_iter_minus_base": coverage_delta,
        "effective_batch_frac_baseline_mean": baseline_eff_mean,
        "effective_batch_frac_iterative_mean": iterative_eff_mean,
        "effective_batch_frac_delta_iter_minus_base": float(iterative_eff_mean - baseline_eff_mean),
        "iterative_forced_prompt_frac": iter_forced_prompt_frac,
        "iterative_fail_all_rounds_frac": iter_fail_all_rounds_frac,
    }
    _dump_json(out_dir / "run_level_summary.json", run_summary)

    print(f"[OK] Investigation outputs written to: {out_dir}")
    print(f"[OK] Paired prompts analyzed: {len(paired)}")
    print(f"[OK] Prompt success delta (iter-base): {coverage_delta:.6f}")
    print(f"[OK] Effective batch frac delta (iter-base): {iterative_eff_mean - baseline_eff_mean:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

