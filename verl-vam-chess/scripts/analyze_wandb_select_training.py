#!/usr/bin/env python3
"""
Analyze a restricted-moves ("selection") chess GRPO run from locally downloaded W&B files.

This is specialized for runs like `gabr1e11/chess_rl/runs/ymyvoypx` where:
  - W&B uploads `rollout_logs/<step>.jsonl` (train rollouts, 8 per prompt uid)
  - W&B uploads `validation_logs/<step>.jsonl` (val rollouts, typically n=1)
  - W&B history is available as `history.jsonl`

The key thing this script adds vs raw W&B metrics:
  - Recomputes per-rollout reward metadata (penalty_reason, pred_move, target_move, acc, etc.)
    by joining `FEN` from the prompt to the base dataset and injecting `considered_moves_uci`
    parsed from the prompt ("Allowed moves (UCI): ...").

Outputs are written under:
  <evidence_root>/investigation/
    - train_rollouts_by_step.csv
    - train_groups_by_step.csv
    - merged_train_history.csv
    - plots/*.png

Example:
  python3 scripts/analyze_wandb_select_training.py \\
    --evidence-root outputs/wandb/ymyvoypx \\
    --train-dataset data/chess_puzzles/train_hard.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# Ensure local imports resolve when the script is run directly.
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipe.chess.reward_fn import UCI_MOVE_ONLY_RE, compute_score


_FEN_RE = re.compile(r"Current FEN string:\s*(?P<fen>[^\r\n]+)", flags=re.IGNORECASE)
_LEGAL_RE = re.compile(r"Legal moves \(UCI\):\s*(?P<moves>[^\r\n]+)", flags=re.IGNORECASE)
_ALLOWED_RE = re.compile(r"Allowed moves \(UCI\):\s*(?P<moves>[^\r\n]+)", flags=re.IGNORECASE)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def _parse_moves_csv(moves_csv: str) -> list[str]:
    return [m.strip().lower() for m in (moves_csv or "").split(",") if m.strip()]


@dataclass(frozen=True)
class PromptFields:
    fen: str
    legal_moves: list[str]
    allowed_moves: list[str]


def _parse_prompt_fields(prompt_text: str) -> PromptFields:
    fen_m = _FEN_RE.search(prompt_text or "")
    legal_m = _LEGAL_RE.search(prompt_text or "")
    allowed_m = _ALLOWED_RE.search(prompt_text or "")
    if not fen_m:
        raise ValueError("Missing 'Current FEN string:' in prompt.")
    if not legal_m:
        raise ValueError("Missing 'Legal moves (UCI):' in prompt.")
    if not allowed_m:
        raise ValueError("Missing 'Allowed moves (UCI):' in prompt.")
    return PromptFields(
        fen=fen_m.group("fen").strip(),
        legal_moves=_parse_moves_csv(legal_m.group("moves")),
        allowed_moves=_parse_moves_csv(allowed_m.group("moves")),
    )


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def _savefig(path: Path, title: str) -> None:
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=12)
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _load_reward_models_by_fen(parquet_path: str) -> dict[str, dict[str, Any]]:
    dataset = ds.dataset(parquet_path, format="parquet")
    table = dataset.to_table(columns=["reward_model"])
    out: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        rm = row.get("reward_model") or {}
        fen = str(rm.get("fen") or "").strip()
        if not fen:
            continue
        # FENs are unique in these datasets; still guard to avoid silent overrides.
        if fen in out:
            raise RuntimeError(f"Duplicate FEN in dataset: {fen}")
        out[fen] = dict(rm)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-root", type=Path, required=True)
    ap.add_argument("--train-dataset", type=str, default="data/chess_puzzles/train_hard.parquet")
    ap.add_argument("--reward-fn", type=str, default="expected_score_wdl_vs_best")
    ap.add_argument("--moving-average-window", type=int, default=7)
    args = ap.parse_args()

    evidence_root: Path = args.evidence_root
    files_root = evidence_root / "files"
    rollout_dir = files_root / "rollout_logs"
    history_path = evidence_root / "history.jsonl"

    if not rollout_dir.exists():
        raise FileNotFoundError(f"Missing rollout logs: {rollout_dir}")
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.jsonl: {history_path}")

    out_root = evidence_root / "investigation"
    plots_dir = out_root / "plots"
    _ensure_dir(out_root)
    _ensure_dir(plots_dir)

    fen2rm = _load_reward_models_by_fen(args.train_dataset)
    print(f"[OK] Loaded base reward_model rows: {len(fen2rm)} from {args.train_dataset}")

    # Load W&B history into a DataFrame.
    hist_rows = [row for _, row in _iter_jsonl(history_path)]
    hist_df = pd.DataFrame(hist_rows)
    if "_step" not in hist_df.columns:
        raise RuntimeError("history.jsonl missing _step column")
    hist_df = hist_df.sort_values("_step").reset_index(drop=True)

    # Derive normalized filter_groups rates (per update).
    if "filter_groups/rejected_groups_total" in hist_df.columns and "filter_groups/kept_groups_total" in hist_df.columns:
        hist_df["filter_groups/generated_groups_total"] = (
            hist_df["filter_groups/rejected_groups_total"] + hist_df["filter_groups/kept_groups_total"]
        )
        hist_df["filter_groups/reject_rate"] = (
            hist_df["filter_groups/rejected_groups_total"] / hist_df["filter_groups/generated_groups_total"]
        )
        hist_df["filter_groups/kept_rate"] = (
            hist_df["filter_groups/kept_groups_total"] / hist_df["filter_groups/generated_groups_total"]
        )
        if "filter_groups/rejected_by_penalty/all_valid" in hist_df.columns:
            denom = hist_df["filter_groups/rejected_groups_total"].replace(0, np.nan)
            hist_df["filter_groups/rejected_frac_all_valid"] = (
                hist_df["filter_groups/rejected_by_penalty/all_valid"] / denom
            )
    else:
        print("[WARN] history missing filter_groups/* columns; skipping reject rate derivations.")

    # Process train rollouts, step by step.
    rollout_files = sorted(
        [p for p in rollout_dir.glob("*.jsonl") if p.is_file()],
        key=lambda p: int(p.stem),
    )

    per_step_rows: list[dict[str, Any]] = []
    per_step_group_rows: list[dict[str, Any]] = []
    per_step_allowed_dist_rows: list[dict[str, Any]] = []

    total_score_mismatches = 0
    total_legal_mismatches = 0

    for file_path in rollout_files:
        step = int(file_path.stem)
        uid2scores: dict[str, list[float]] = defaultdict(list)
        uid2pred_moves: dict[str, list[str]] = defaultdict(list)
        uid2pred_moves_valid: dict[str, list[str]] = defaultdict(list)
        uid2penalty: dict[str, list[bool]] = defaultdict(list)
        uid2allowed_size: dict[str, int] = {}
        uid2target_move: dict[str, str] = {}
        uid2gt_move: dict[str, str] = {}

        n = 0
        sum_score = 0.0
        sum_acc = 0.0
        sum_exact = 0.0
        sum_format_ok = 0.0
        sum_in_subset = 0.0
        sum_penalty_applied = 0.0

        penalty_reason_counts: Counter[str] = Counter()
        uci_tag_counts: Counter[str] = Counter()

        for rec_idx, rec in _iter_jsonl(file_path):
            prompt_text = str(rec.get("input") or "")
            output_text = str(rec.get("output") or "")
            logged_score = float(rec.get("score"))
            uid = str(rec.get("uid") or "")
            if not uid:
                raise RuntimeError(f"Missing uid in {file_path} record {rec_idx}")

            # Parse prompt, join to dataset by FEN.
            fields = _parse_prompt_fields(prompt_text)
            rm_base = fen2rm.get(fields.fen)
            if rm_base is None:
                raise KeyError(f"FEN not found in base dataset: {fields.fen} ({file_path} record {rec_idx})")

            # Cross-check legal moves match (helps detect wrong join).
            rm_legal = [str(m).strip().lower() for m in (rm_base.get("legal_moves_uci") or [])]
            if set(rm_legal) != set(fields.legal_moves):
                total_legal_mismatches += 1

            rm = dict(rm_base)
            rm["considered_moves_uci"] = list(fields.allowed_moves)

            # Recompute the reward metadata with the same reward semantics as training.
            res = compute_score(
                data_source=rm,
                solution_str=output_text,
                ground_truth=str(rm.get("ground_truth") or ""),
                chess_reward_fn=str(args.reward_fn),
            )

            computed_score = float(res["score"])
            if not math.isfinite(computed_score) or abs(computed_score - logged_score) > 1e-6:
                total_score_mismatches += 1

            # Tag-span diagnosis: missing vs multiple tags.
            spans = list(UCI_MOVE_ONLY_RE.finditer(output_text))
            if len(spans) == 0:
                uci_tag_counts["missing_uci_move_tag"] += 1
            elif len(spans) == 1:
                uci_tag_counts["one_uci_move_tag"] += 1
            else:
                uci_tag_counts["multiple_uci_move_tags"] += 1

            penalty_reason = str(res.get("penalty_reason") or "")
            penalty_reason_counts[penalty_reason or "<none>"] += 1

            uid2scores[uid].append(float(logged_score))
            pred_move = str(res.get("pred_move") or "")
            uid2pred_moves[uid].append(pred_move)
            penalty_applied = bool(res.get("penalty_applied") or False)
            uid2penalty[uid].append(penalty_applied)
            if pred_move and not penalty_applied:
                uid2pred_moves_valid[uid].append(pred_move)
            uid2allowed_size.setdefault(uid, int(res.get("n_considered_moves") or len(fields.allowed_moves)))
            uid2target_move.setdefault(uid, str(res.get("target_move") or ""))
            uid2gt_move.setdefault(uid, str(res.get("gt_uci") or "").strip().lower())

            sum_score += float(logged_score)
            sum_acc += float(res.get("acc") or 0.0)
            sum_exact += float(res.get("exact_match") or 0.0)
            sum_format_ok += float(res.get("format_reward") or 0.0)
            sum_in_subset += 1.0 if bool(res.get("in_subset")) else 0.0
            sum_penalty_applied += 1.0 if bool(res.get("penalty_applied")) else 0.0
            n += 1

        if n == 0:
            continue

        # Group-level metrics (within kept groups, score std must be > 0 by construction).
        group_stds = np.array([float(np.std(v)) for v in uid2scores.values()], dtype=np.float64)
        group_unique_scores = np.array([len(set(v)) for v in uid2scores.values()], dtype=np.float64)
        group_unique_moves = np.array([len(set(v)) for v in uid2pred_moves.values()], dtype=np.float64)
        group_any_penalty = np.array([any(v) for v in uid2penalty.values()], dtype=np.float64)
        allowed_sizes = np.array(list(uid2allowed_size.values()), dtype=np.float64)

        # "Mode move" diagnostics: majority vote over valid (non-penalty) moves.
        mode_target = 0
        mode_suboptimal = 0
        mode_penalty = 0
        mode_exact_gt = 0
        for uid in uid2scores.keys():
            preds = uid2pred_moves_valid.get(uid, [])
            target = uid2target_move.get(uid, "")
            gt = uid2gt_move.get(uid, "")
            if not preds:
                mode_penalty += 1
                continue
            mode_move, _ = Counter(preds).most_common(1)[0]
            if gt and mode_move == gt:
                mode_exact_gt += 1
            if target and mode_move == target:
                mode_target += 1
            else:
                mode_suboptimal += 1
        mode_denom = mode_target + mode_suboptimal
        mode_target_ratio_excl_penalty = float(mode_target) / float(mode_denom) if mode_denom else float("nan")

        per_step_rows.append(
            {
                "step": step,
                "n_records": n,
                "mean_score": sum_score / n,
                "mean_acc": sum_acc / n,
                "mean_exact_match": sum_exact / n,
                "format_ok_rate": sum_format_ok / n,
                "in_subset_rate": sum_in_subset / n,
                "penalty_applied_rate": sum_penalty_applied / n,
                # tag diagnostics
                "missing_uci_move_tag_rate": uci_tag_counts["missing_uci_move_tag"] / n,
                "multiple_uci_move_tags_rate": uci_tag_counts["multiple_uci_move_tags"] / n,
                # penalty reasons
                **{f"penalty_reason/{k}": int(v) for k, v in penalty_reason_counts.items()},
            }
        )

        per_step_group_rows.append(
            {
                "step": step,
                "n_prompt_groups": int(len(uid2scores)),
                "group_score_std_mean": float(np.mean(group_stds)) if group_stds.size else float("nan"),
                "group_score_std_median": float(np.median(group_stds)) if group_stds.size else float("nan"),
                "group_unique_scores_mean": float(np.mean(group_unique_scores)) if group_unique_scores.size else float("nan"),
                "group_unique_moves_mean": float(np.mean(group_unique_moves)) if group_unique_moves.size else float("nan"),
                "group_any_penalty_frac": float(np.mean(group_any_penalty)) if group_any_penalty.size else float("nan"),
                "allowed_moves_mean": float(np.mean(allowed_sizes)) if allowed_sizes.size else float("nan"),
                "allowed_moves_min": float(np.min(allowed_sizes)) if allowed_sizes.size else float("nan"),
                "allowed_moves_max": float(np.max(allowed_sizes)) if allowed_sizes.size else float("nan"),
                "group_mode_target_frac": float(mode_target) / float(len(uid2scores)) if uid2scores else float("nan"),
                "group_mode_suboptimal_frac": float(mode_suboptimal) / float(len(uid2scores)) if uid2scores else float("nan"),
                "group_mode_penalty_frac": float(mode_penalty) / float(len(uid2scores)) if uid2scores else float("nan"),
                "group_mode_target_ratio_excl_penalty": mode_target_ratio_excl_penalty,
                "group_mode_exact_gt_frac": float(mode_exact_gt) / float(len(uid2scores)) if uid2scores else float("nan"),
            }
        )

        # More detailed allowed-moves distribution (useful for diagnosing filter-induced distribution shifts).
        if allowed_sizes.size:
            per_step_allowed_dist_rows.append(
                {
                    "step": step,
                    "n_groups": int(len(uid2scores)),
                    "allowed_mean": float(np.mean(allowed_sizes)),
                    "allowed_p50": float(np.median(allowed_sizes)),
                    "allowed_p90": float(np.percentile(allowed_sizes, 90)),
                    "frac_ge16": float(np.mean(allowed_sizes >= 16)),
                    "frac_ge20": float(np.mean(allowed_sizes >= 20)),
                    "frac_le6": float(np.mean(allowed_sizes <= 6)),
                }
            )

    roll_df = pd.DataFrame(per_step_rows).sort_values("step")
    group_df = pd.DataFrame(per_step_group_rows).sort_values("step")
    allowed_dist_df = pd.DataFrame(per_step_allowed_dist_rows).sort_values("step")

    roll_df.to_csv(out_root / "train_rollouts_by_step.csv", index=False)
    group_df.to_csv(out_root / "train_groups_by_step.csv", index=False)
    if not allowed_dist_df.empty:
        allowed_dist_df.to_csv(out_root / "allowed_moves_dist_by_step.csv", index=False)

    # Merge rollouts-derived metrics into history (on step).
    merged = hist_df.merge(roll_df, left_on="_step", right_on="step", how="left").merge(
        group_df, on="step", how="left"
    )
    merged.to_csv(out_root / "merged_train_history.csv", index=False)

    print(f"[OK] Wrote {out_root / 'train_rollouts_by_step.csv'}")
    print(f"[OK] Wrote {out_root / 'train_groups_by_step.csv'}")
    if not allowed_dist_df.empty:
        print(f"[OK] Wrote {out_root / 'allowed_moves_dist_by_step.csv'}")
    print(f"[OK] Wrote {out_root / 'merged_train_history.csv'}")
    if total_score_mismatches:
        print(f"[WARN] score mismatches vs logged score: {total_score_mismatches}")
    if total_legal_mismatches:
        print(
            "[WARN] prompt legal_moves mismatch vs base dataset "
            f"(count={total_legal_mismatches}; join-by-FEN still succeeded)"
        )

    # Plots
    window = int(args.moving_average_window)
    if "critic/rewards/mean" in merged.columns:
        plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.plot(merged["_step"], merged["critic/rewards/mean"], label="critic/rewards/mean", linewidth=1.2)
        ax.plot(merged["_step"], _rolling_mean(merged["critic/rewards/mean"], window), label=f"MA{window}", linewidth=2.0)
        ax.axvline(40, color="gray", linestyle="--", linewidth=1)
        ax.axvline(60, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel("mean reward")
        ax.set_title("Train reward (W&B): critic/rewards/mean")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _savefig(plots_dir / "train_reward_mean.png", "Train reward (critic/rewards/mean)")

    if "filter_groups/reject_rate" in merged.columns:
        plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.plot(merged["_step"], merged["filter_groups/reject_rate"], label="reject_rate", linewidth=1.4)
        ax.plot(
            merged["_step"],
            _rolling_mean(merged["filter_groups/reject_rate"], window),
            label=f"reject_rate MA{window}",
            linewidth=2.0,
        )
        ax.axvline(40, color="gray", linestyle="--", linewidth=1)
        ax.axvline(60, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel("rejected / generated")
        ax.set_title("filter_groups reject rate (per update)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _savefig(plots_dir / "filter_groups_reject_rate.png", "filter_groups reject rate")

    if "mean_acc" in merged.columns:
        plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.plot(merged["step"], merged["mean_acc"], label="train rollouts mean acc", linewidth=1.4)
        ax.plot(merged["step"], _rolling_mean(merged["mean_acc"], window), label=f"MA{window}", linewidth=2.0)
        ax.axvline(40, color="gray", linestyle="--", linewidth=1)
        ax.axvline(60, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel("acc (target match rate)")
        ax.set_title("Train rollouts: selection acc (computed from logs)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _savefig(plots_dir / "train_rollout_acc.png", "Train rollouts acc")

    if "group_mode_target_ratio_excl_penalty" in merged.columns:
        plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.plot(
            merged["step"],
            merged["group_mode_target_ratio_excl_penalty"],
            label="mode(target) / (mode(target)+mode(suboptimal))",
            linewidth=1.4,
        )
        ax.plot(
            merged["step"],
            _rolling_mean(merged["group_mode_target_ratio_excl_penalty"], window),
            label=f"MA{window}",
            linewidth=2.0,
        )
        ax.axvline(40, color="gray", linestyle="--", linewidth=1)
        ax.axvline(60, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel("ratio")
        ax.set_title("Train groups: mode-target ratio (proxy)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _savefig(plots_dir / "train_group_mode_target_ratio.png", "Train groups: mode-target ratio")

    if "actor/entropy" in merged.columns and "filter_groups/reject_rate" in merged.columns:
        plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax.scatter(merged["actor/entropy"], merged["filter_groups/reject_rate"], s=18, alpha=0.8)
        ax.set_xlabel("actor/entropy")
        ax.set_ylabel("filter_groups reject_rate")
        ax.set_title("Entropy vs reject rate")
        ax.grid(True, alpha=0.3)
        _savefig(plots_dir / "entropy_vs_reject_rate.png", "Entropy vs reject rate")

    # Compliance trends (rollouts-derived).
    if "format_ok_rate" in merged.columns and "in_subset_rate" in merged.columns:
        plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.plot(merged["step"], merged["format_ok_rate"], label="format_ok_rate", linewidth=1.4)
        ax.plot(merged["step"], merged["in_subset_rate"], label="in_subset_rate", linewidth=1.4)
        ax.plot(merged["step"], merged["penalty_applied_rate"], label="penalty_applied_rate", linewidth=1.4)
        ax.axvline(40, color="gray", linestyle="--", linewidth=1)
        ax.axvline(60, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel("rate")
        ax.set_title("Train rollout compliance (computed)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _savefig(plots_dir / "train_rollout_compliance.png", "Train rollout compliance")

    print(f"[OK] Plots written under {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
