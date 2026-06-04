#!/usr/bin/env python3
"""
Root-cause analysis for the Pass@k effective-batch discrepancy:
  - Baseline/global Pass@k runs (single-round GRPO groups keyed by uid)
  - Iterative allowed_move_elim runs (multi-round GRPO groups keyed by (prompt_idx, round))

This script is intended to be reproducible offline from locally downloaded W&B evidence:
  analysis/wandb_evidence/<run_id>/
    - history.parquet
    - config_api.json
    - files/rollout_logs/*.jsonl                    (baseline runs)
    - files/allowed_move_elim_rounds/*_round*.jsonl (iterative runs)

It produces:
  - CSVs under `analysis/custom_metrics/`
  - Plots under `reports/passk_effective_batch/`

The key custom metrics are not logged to W&B, e.g.:
  - mean unique predicted moves per GRPO group
  - mean unique reward values per GRPO group
  - dead-group breakdown (std(score)==0) by "has_optimal" vs "no_optimal"
  - counterfactual per-prompt effective fraction (aggregate across rounds)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _pick_files_root(evidence_dir: Path) -> Path:
    """Prefer <evidence>/files (full download); fall back to <evidence>/files_selected."""
    for cand in (evidence_dir / "files", evidence_dir / "files_selected"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Missing files root under {evidence_dir} (expected files/ or files_selected/)")


def _load_history(evidence_dir: Path) -> pd.DataFrame:
    hist_path = evidence_dir / "history.parquet"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing history.parquet: {hist_path}")
    df = pd.read_parquet(hist_path)
    if "training/global_step" in df.columns:
        df = df[df["training/global_step"].notna()].copy()
        df["step"] = df["training/global_step"].astype(int)
    elif "_step" in df.columns:
        df = df[df["_step"].notna()].copy()
        df["step"] = df["_step"].astype(int)
    else:
        # Fallback: monotonic index
        df = df.copy()
        df["step"] = np.arange(len(df), dtype=np.int64)
    return df


def _load_config(evidence_dir: Path) -> dict[str, Any]:
    cfg_path = evidence_dir / "config_api.json"
    if not cfg_path.exists():
        return {}
    try:
        obj = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _normalize_move(x: Any) -> str:
    return str(x or "").strip().lower()


def _is_success_sample(rec: dict[str, Any]) -> bool:
    # Mirrors sample_is_optimal logic (ray_trainer.compute_advantage) and reward_fn conventions.
    pm = _normalize_move(rec.get("pred_move"))
    gt = _normalize_move(rec.get("gt_uci"))
    if not pm or not gt:
        return False
    if bool(rec.get("penalty_applied", False)):
        return False
    if "in_subset" in rec and (not bool(rec.get("in_subset"))):
        return False
    return pm == gt


def _dead_class_from_constant_score(v: float) -> str:
    eps = 1e-12
    if not _is_finite(v):
        return "nan"
    if abs(float(v) + 1.0) <= eps:
        return "all_-1"
    if abs(float(v)) <= eps:
        return "all_0"
    if -1.0 < float(v) < 0.0:
        return "all_neg_nonpen"
    if float(v) > 0.0:
        return "all_pos"
    return "other"


@dataclass
class GroupAgg:
    scores: list[float]
    moves: set[str]
    n: int
    n_penalty: int
    has_optimal: bool

    @classmethod
    def empty(cls) -> "GroupAgg":
        return GroupAgg(scores=[], moves=set(), n=0, n_penalty=0, has_optimal=False)

    def add(self, rec: dict[str, Any]) -> None:
        self.n += 1
        score = rec.get("score")
        try:
            self.scores.append(float(score))
        except Exception:
            self.scores.append(float("nan"))
        self.moves.add(_normalize_move(rec.get("pred_move")))
        if bool(rec.get("penalty_applied", False)):
            self.n_penalty += 1
        if _is_success_sample(rec):
            self.has_optimal = True

    def score_std(self) -> float:
        arr = np.asarray([x for x in self.scores if _is_finite(x)], dtype=np.float32)
        if arr.size <= 1:
            return 0.0
        return float(np.std(arr))

    def uniq_scores(self) -> int:
        uniq = {float(x) for x in self.scores if _is_finite(x)}
        return int(len(uniq))

    def uniq_moves(self) -> int:
        uniq = {m for m in self.moves if m}
        return int(len(uniq))

    def penalty_frac(self) -> float:
        return float(self.n_penalty) / float(max(1, self.n))


def analyze_baseline_rollout_logs(*, evidence_dir: Path) -> pd.DataFrame:
    files_root = _pick_files_root(evidence_dir)
    log_dir = files_root / "rollout_logs"
    if not log_dir.exists():
        raise FileNotFoundError(f"Missing rollout_logs dir: {log_dir}")
    files = sorted(log_dir.glob("*.jsonl"), key=lambda p: int(p.stem))

    rows: list[dict[str, Any]] = []
    for fp in files:
        step = int(fp.stem)
        uid2agg: dict[str, GroupAgg] = defaultdict(GroupAgg.empty)
        for rec in _iter_jsonl(fp):
            uid = str(rec.get("uid") or "")
            if not uid:
                # Baseline logs should always have uid; skip if missing.
                continue
            uid2agg[uid].add(rec)

        total_groups = len(uid2agg)
        eff = 0
        det = 0
        dead_counts = Counter()
        uniq_moves = []
        uniq_scores = []
        pen_fracs = []
        success_any = []

        for agg in uid2agg.values():
            std = agg.score_std()
            if std > 0:
                eff += 1
            else:
                # classify by constant score (when possible)
                uniq = {x for x in agg.scores if _is_finite(x)}
                v = next(iter(uniq)) if len(uniq) == 1 else float("nan")
                dead_counts[_dead_class_from_constant_score(float(v))] += 1

            um = agg.uniq_moves()
            if um == 1:
                det += 1
            uniq_moves.append(um)
            uniq_scores.append(agg.uniq_scores())
            pen_fracs.append(agg.penalty_frac())
            success_any.append(1.0 if agg.has_optimal else 0.0)

        rows.append(
            {
                "step": step,
                "groups_total": int(total_groups),
                "effective_groups": int(eff),
                "effective_frac": float(eff / total_groups) if total_groups else float("nan"),
                "dead_groups": int(total_groups - eff),
                "det_group_frac": float(det / total_groups) if total_groups else float("nan"),
                "uniq_moves_mean": float(np.mean(uniq_moves)) if uniq_moves else float("nan"),
                "uniq_scores_mean": float(np.mean(uniq_scores)) if uniq_scores else float("nan"),
                "penalty_frac_mean": float(np.mean(pen_fracs)) if pen_fracs else float("nan"),
                "success_group_frac": float(np.mean(success_any)) if success_any else float("nan"),
                # dead-class counts
                "dead_all_-1": int(dead_counts.get("all_-1", 0)),
                "dead_all_0": int(dead_counts.get("all_0", 0)),
                "dead_all_neg_nonpen": int(dead_counts.get("all_neg_nonpen", 0)),
                "dead_other": int(dead_counts.get("other", 0)),
            }
        )

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return df


def _discover_allowed_move_elim_steps(log_dir: Path) -> list[int]:
    steps: set[int] = set()
    for p in log_dir.glob("*_round1.jsonl"):
        try:
            steps.add(int(p.name.split("_round")[0]))
        except Exception:
            continue
    return sorted(steps)


def analyze_iterative_allowed_move_elim_rounds(*, evidence_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (by_step, per_round, per_prompt_vs_per_round)."""
    files_root = _pick_files_root(evidence_dir)
    log_dir = files_root / "allowed_move_elim_rounds"
    if not log_dir.exists():
        raise FileNotFoundError(f"Missing allowed_move_elim_rounds dir: {log_dir}")

    steps = _discover_allowed_move_elim_steps(log_dir)
    if not steps:
        raise FileNotFoundError(f"No *_round1.jsonl files found under {log_dir}")

    by_step_rows: list[dict[str, Any]] = []
    per_round_rows: list[dict[str, Any]] = []
    per_prompt_rows: list[dict[str, Any]] = []

    for step in steps:
        # Group by (prompt_idx, round) for per_round stats, and by prompt_idx for counterfactual per_prompt stats.
        group_round: dict[tuple[int, int], GroupAgg] = defaultdict(GroupAgg.empty)
        group_prompt: dict[int, GroupAgg] = defaultdict(GroupAgg.empty)

        # Read up to 4 rounds (the default R_max). If the run logged fewer, we will just use what's available.
        round_files = sorted(log_dir.glob(f"{step}_round*.jsonl"), key=lambda p: p.name)
        for fp in round_files:
            for rec in _iter_jsonl(fp):
                pidx = rec.get("allowed_move_elim_prompt_idx")
                rnd = rec.get("allowed_move_elim_round")
                if pidx is None or rnd is None:
                    continue
                pidx_i = int(pidx)
                rnd_i = int(rnd)
                group_round[(pidx_i, rnd_i)].add(rec)
                group_prompt[pidx_i].add(rec)

        # ---- per-step aggregation over per-round groups ----
        total_groups = len(group_round)
        eff = 0
        dead_counts = Counter()

        n_has_opt = 0
        n_no_opt = 0
        dead_has_opt = 0
        dead_no_opt = 0

        uniq_moves_all = []
        uniq_scores_all = []
        uniq_moves_has_opt = []
        uniq_moves_no_opt = []
        uniq_scores_has_opt = []
        uniq_scores_no_opt = []
        pen_fracs = []

        for agg in group_round.values():
            std = agg.score_std()
            is_eff = std > 0
            if is_eff:
                eff += 1
            else:
                uniq = {x for x in agg.scores if _is_finite(x)}
                v = next(iter(uniq)) if len(uniq) == 1 else float("nan")
                dead_counts[_dead_class_from_constant_score(float(v))] += 1

            if agg.has_optimal:
                n_has_opt += 1
                if not is_eff:
                    dead_has_opt += 1
            else:
                n_no_opt += 1
                if not is_eff:
                    dead_no_opt += 1

            um = agg.uniq_moves()
            us = agg.uniq_scores()
            uniq_moves_all.append(um)
            uniq_scores_all.append(us)
            if agg.has_optimal:
                uniq_moves_has_opt.append(um)
                uniq_scores_has_opt.append(us)
            else:
                uniq_moves_no_opt.append(um)
                uniq_scores_no_opt.append(us)
            pen_fracs.append(agg.penalty_frac())

        by_step_rows.append(
            {
                "step": int(step),
                "groups_total": int(total_groups),
                "effective_groups": int(eff),
                "effective_frac": float(eff / total_groups) if total_groups else float("nan"),
                "dead_groups": int(total_groups - eff),
                "groups_has_optimal": int(n_has_opt),
                "groups_no_optimal": int(n_no_opt),
                "dead_has_optimal": int(dead_has_opt),
                "dead_no_optimal": int(dead_no_opt),
                "dead_frac_has_optimal": float(dead_has_opt / n_has_opt) if n_has_opt else float("nan"),
                "dead_frac_no_optimal": float(dead_no_opt / n_no_opt) if n_no_opt else float("nan"),
                "uniq_moves_mean": float(np.mean(uniq_moves_all)) if uniq_moves_all else float("nan"),
                "uniq_scores_mean": float(np.mean(uniq_scores_all)) if uniq_scores_all else float("nan"),
                "uniq_moves_mean_no_opt": float(np.mean(uniq_moves_no_opt)) if uniq_moves_no_opt else float("nan"),
                "uniq_moves_mean_has_opt": float(np.mean(uniq_moves_has_opt)) if uniq_moves_has_opt else float("nan"),
                "uniq_scores_mean_no_opt": float(np.mean(uniq_scores_no_opt)) if uniq_scores_no_opt else float("nan"),
                "uniq_scores_mean_has_opt": float(np.mean(uniq_scores_has_opt)) if uniq_scores_has_opt else float("nan"),
                "penalty_frac_mean": float(np.mean(pen_fracs)) if pen_fracs else float("nan"),
                # dead-class counts
                "dead_all_-1": int(dead_counts.get("all_-1", 0)),
                "dead_all_0": int(dead_counts.get("all_0", 0)),
                "dead_all_neg_nonpen": int(dead_counts.get("all_neg_nonpen", 0)),
                "dead_other": int(dead_counts.get("other", 0)),
            }
        )

        # ---- per-round breakdown (within the step) ----
        round2aggs: dict[int, list[GroupAgg]] = defaultdict(list)
        for (pidx_i, rnd_i), agg in group_round.items():
            round2aggs[int(rnd_i)].append(agg)

        for rnd_i, aggs in sorted(round2aggs.items()):
            n_groups = len(aggs)
            eff_r = 0
            no_opt_r = 0
            dead_no_opt_r = 0
            uniq_moves_r = []
            uniq_scores_r = []
            pen_fracs_r = []
            for agg in aggs:
                is_eff = agg.score_std() > 0
                if is_eff:
                    eff_r += 1
                if not agg.has_optimal:
                    no_opt_r += 1
                    if not is_eff:
                        dead_no_opt_r += 1
                uniq_moves_r.append(agg.uniq_moves())
                uniq_scores_r.append(agg.uniq_scores())
                pen_fracs_r.append(agg.penalty_frac())

            per_round_rows.append(
                {
                    "step": int(step),
                    "round": int(rnd_i),
                    "groups_total": int(n_groups),
                    "effective_groups": int(eff_r),
                    "effective_frac": float(eff_r / n_groups) if n_groups else float("nan"),
                    "no_opt_groups": int(no_opt_r),
                    "dead_frac_no_opt": float(dead_no_opt_r / no_opt_r) if no_opt_r else float("nan"),
                    "uniq_moves_mean": float(np.mean(uniq_moves_r)) if uniq_moves_r else float("nan"),
                    "uniq_scores_mean": float(np.mean(uniq_scores_r)) if uniq_scores_r else float("nan"),
                    "penalty_frac_mean": float(np.mean(pen_fracs_r)) if pen_fracs_r else float("nan"),
                }
            )

        # ---- counterfactual: per-prompt (aggregate across rounds) ----
        # This approximates what effective batch would look like under uid_mode='per_prompt'.
        total_prompt = len(group_prompt)
        eff_prompt = sum(agg.score_std() > 0 for agg in group_prompt.values())
        per_prompt_rows.append(
            {
                "step": int(step),
                "round_groups_total": int(total_groups),
                "round_effective": int(eff),
                "round_eff_frac": float(eff / total_groups) if total_groups else float("nan"),
                "prompts_total": int(total_prompt),
                "prompt_effective": int(eff_prompt),
                "prompt_eff_frac": float(eff_prompt / total_prompt) if total_prompt else float("nan"),
            }
        )

    by_step = pd.DataFrame(by_step_rows).sort_values("step").reset_index(drop=True)
    per_round = pd.DataFrame(per_round_rows).sort_values(["step", "round"]).reset_index(drop=True)
    per_prompt_vs_round = pd.DataFrame(per_prompt_rows).sort_values("step").reset_index(drop=True)
    return by_step, per_round, per_prompt_vs_round


def _plot_effective_batch_frac_from_history(
    *, run_specs: list[tuple[str, str]], evidence_root: Path, out_path: Path
) -> None:
    plt.figure(figsize=(10, 5))
    for run_id, label in run_specs:
        ed = evidence_root / run_id
        try:
            hist = _load_history(ed)
        except Exception:
            continue
        if "grpo/effective_batch_frac" not in hist.columns:
            continue
        plt.plot(hist["step"], hist["grpo/effective_batch_frac"], label=f"{run_id} ({label})", linewidth=1.8)
    plt.xlabel("training/global_step")
    plt.ylabel("grpo/effective_batch_frac")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.title("GRPO effective batch fraction (W&B history)")
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-root", type=Path, default=Path("analysis/wandb_evidence"))
    ap.add_argument("--metrics-out-dir", type=Path, default=Path("analysis/custom_metrics"))
    ap.add_argument("--plots-out-dir", type=Path, default=Path("reports/passk_effective_batch"))
    ap.add_argument("--baseline-runs", nargs="*", default=["82fpo6l0", "f5guq4ti", "u2cuw56a"])
    ap.add_argument("--iterative-runs", nargs="*", default=["s0anl08n", "xie1sbcg"])
    args = ap.parse_args()

    evidence_root: Path = args.evidence_root
    metrics_out_dir: Path = args.metrics_out_dir
    plots_out_dir: Path = args.plots_out_dir

    _ensure_dir(metrics_out_dir)
    _ensure_dir(plots_out_dir)

    # ---- Baseline runs ----
    for run_id in args.baseline_runs:
        evidence_dir = evidence_root / run_id
        if not evidence_dir.exists():
            print(f"[WARN] missing evidence dir: {evidence_dir}")
            continue
        try:
            cfg = _load_config(evidence_dir)
            rollout_n = cfg.get("actor_rollout_ref", {}).get("rollout", {}).get("n")
            passk = cfg.get("algorithm", {}).get("pass_k_training")
            print(f"[BASELINE] {run_id} rollout.n={rollout_n} pass_k_training={passk}")
            df = analyze_baseline_rollout_logs(evidence_dir=evidence_dir)
        except Exception as e:
            print(f"[WARN] baseline analysis failed for {run_id}: {e}")
            continue

        out_path = metrics_out_dir / f"{run_id}_baseline_group_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path} rows={len(df)} steps={df['step'].min()}..{df['step'].max()}")

    # ---- Iterative runs ----
    for run_id in args.iterative_runs:
        evidence_dir = evidence_root / run_id
        if not evidence_dir.exists():
            print(f"[WARN] missing evidence dir: {evidence_dir}")
            continue
        try:
            cfg = _load_config(evidence_dir)
            rollout_n = cfg.get("actor_rollout_ref", {}).get("rollout", {}).get("n")
            ame = (cfg.get("algorithm", {}) or {}).get("allowed_move_elim", {}) or {}
            cond_passk = ame.get("pass_k_when_no_optimal")
            print(f"[ITER] {run_id} rollout.n={rollout_n} allowed_move_elim={ame.get('enable')} cond_passk={cond_passk}")
            by_step, per_round, per_prompt = analyze_iterative_allowed_move_elim_rounds(evidence_dir=evidence_dir)
        except Exception as e:
            print(f"[WARN] iterative analysis failed for {run_id}: {e}")
            continue

        out_step = metrics_out_dir / f"{run_id}_iterative_group_metrics_by_step.csv"
        out_round = metrics_out_dir / f"{run_id}_per_round_metrics.csv"
        out_prompt = metrics_out_dir / f"{run_id}_per_prompt_vs_per_round_effective.csv"
        by_step.to_csv(out_step, index=False)
        per_round.to_csv(out_round, index=False)
        per_prompt.to_csv(out_prompt, index=False)
        print(f"[OK] wrote {out_step} rows={len(by_step)}")
        print(f"[OK] wrote {out_round} rows={len(per_round)}")
        print(f"[OK] wrote {out_prompt} rows={len(per_prompt)}")

    # ---- Plots (W&B history) ----
    run_specs = []
    for rid in args.baseline_runs:
        run_specs.append((rid, "baseline"))
    for rid in args.iterative_runs:
        run_specs.append((rid, "iterative"))

    _plot_effective_batch_frac_from_history(
        run_specs=run_specs,
        evidence_root=evidence_root,
        out_path=plots_out_dir / "effective_batch_frac_wandb_comparison.png",
    )
    print(f"[OK] wrote {plots_out_dir/'effective_batch_frac_wandb_comparison.png'}")

    # Iterative-specific plots (if xie1sbcg metrics are available)
    xie_step = metrics_out_dir / "xie1sbcg_iterative_group_metrics_by_step.csv"
    xie_round = metrics_out_dir / "xie1sbcg_per_round_metrics.csv"
    xie_prompt = metrics_out_dir / "xie1sbcg_per_prompt_vs_per_round_effective.csv"

    if xie_round.exists():
        df = pd.read_csv(xie_round)
        plt.figure(figsize=(10, 5))
        for rnd in sorted(df["round"].unique()):
            d = df[df["round"] == rnd]
            plt.plot(d["step"], d["effective_frac"], label=f"round {rnd}")
        plt.xlabel("step")
        plt.ylabel("effective_frac (std(score)>0)")
        plt.ylim(0, 1.02)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.title("xie1sbcg: per-round effective fraction (from allowed_move_elim_rounds logs)")
        plt.tight_layout()
        plt.savefig(plots_out_dir / "xie1sbcg_per_round_effective_frac.png", dpi=180)
        plt.close()
        print(f"[OK] wrote {plots_out_dir/'xie1sbcg_per_round_effective_frac.png'}")

    if xie_prompt.exists():
        df = pd.read_csv(xie_prompt)
        plt.figure(figsize=(10, 5))
        plt.plot(df["step"], df["round_eff_frac"], label="per_round groups (uid_mode=per_round)", linewidth=2)
        plt.plot(df["step"], df["prompt_eff_frac"], label="per_prompt aggregate (counterfactual)", linewidth=2)
        plt.xlabel("step")
        plt.ylabel("effective_frac (std(score)>0)")
        plt.ylim(0, 1.02)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.title("xie1sbcg: effective fraction by grouping semantics")
        plt.tight_layout()
        plt.savefig(plots_out_dir / "xie1sbcg_per_prompt_vs_per_round_effective_frac.png", dpi=180)
        plt.close()
        print(f"[OK] wrote {plots_out_dir/'xie1sbcg_per_prompt_vs_per_round_effective_frac.png'}")

    if xie_step.exists():
        df = pd.read_csv(xie_step)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df["step"], df["dead_frac_no_optimal"], color="red", label="dead_frac_no_optimal")
        ax1.set_ylim(0, 1.02)
        ax1.set_xlabel("step")
        ax1.set_ylabel("dead_frac_no_optimal", color="red")
        ax1.tick_params(axis="y", labelcolor="red")
        ax1.grid(alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(df["step"], df["uniq_moves_mean_no_opt"], color="blue", label="uniq_moves_mean_no_opt")
        ax2.set_ylabel("uniq_moves_mean_no_opt", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.title("xie1sbcg: no-opt groups collapse (dead frac vs move diversity)")
        plt.tight_layout()
        plt.savefig(plots_out_dir / "xie1sbcg_no_opt_dead_vs_diversity.png", dpi=180)
        plt.close(fig)
        print(f"[OK] wrote {plots_out_dir/'xie1sbcg_no_opt_dead_vs_diversity.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

