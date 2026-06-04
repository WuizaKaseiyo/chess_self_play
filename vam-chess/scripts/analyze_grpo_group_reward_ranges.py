#!/usr/bin/env python3
"""
Offline analysis: GRPO within-group reward range histograms.

This script is designed to work *offline* against a locally-downloaded W&B run
evidence folder produced by `scripts/download_wandb_run_evidence.py`.

We support two common logging layouts:

1) Iterative allowed-move elimination (selection sampler enabled):
   - <evidence>/files/allowed_move_elim_rounds/<step>_round<r>.jsonl
   - Group id for GRPO aggregation: (step, allowed_move_elim_round, allowed_move_elim_prompt_idx)

2) Standard rollout logging (selection sampler disabled):
   - <evidence>/files/rollout_logs/<step>.jsonl
   - Group id for GRPO aggregation: (step, uid)

Metric:
  For each GRPO group, compute the within-group reward range:
    score_range = max(score) - min(score)

We report two variants:
  - score_range_all: uses all samples in the group (finite scores only).
  - score_range_valid: uses only "valid in-subset" samples where:
        penalty_applied == False and in_subset == True
    If a log format does not include the validity fields, score_range_valid is reported as NaN.

Outputs (under --out_dir):
  - group_ranges.csv.gz: one row per (run, requested_step, used_step, group)
  - summary_by_run_step.csv: summary stats per run+requested_step
  - step_mapping.json: requested->used mapping per run
  - plots/*.png: histograms (consistent bins/xlim across all runs in the invocation)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALLOWED_MOVE_ELIM_FILE_RE = re.compile(r"^(?P<step>\d+)_round(?P<round>\d+)\.jsonl$")
ROLLOUT_FILE_RE = re.compile(r"^(?P<step>\d+)\.jsonl$")


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _parse_steps_arg(raw: str) -> list[int]:
    s = (raw or "").strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _choose_nearest_step(target: int, available_steps: list[int]) -> int:
    if not available_steps:
        raise ValueError("No available steps found in logs.")
    if target in available_steps:
        return target
    return min(available_steps, key=lambda s: (abs(int(s) - int(target)), int(s)))


@dataclass
class GroupRangeAgg:
    # Counts
    n_total: int = 0
    n_total_non_finite: int = 0
    n_penalty: int = 0

    n_valid: int = 0
    n_valid_non_finite: int = 0

    # Extrema (all)
    min_all: float = float("inf")
    max_all: float = -float("inf")

    # Extrema (valid)
    min_valid: float = float("inf")
    max_valid: float = -float("inf")

    # Diagnostics
    chess_reward_fn: str = ""

    def add(
        self,
        *,
        score: float,
        penalty_applied: Optional[bool],
        in_subset: Optional[bool],
        chess_reward_fn: Optional[str],
    ) -> None:
        self.n_total += 1
        if chess_reward_fn and not self.chess_reward_fn:
            self.chess_reward_fn = str(chess_reward_fn)

        if not _is_finite(score):
            self.n_total_non_finite += 1
        else:
            self.min_all = min(self.min_all, float(score))
            self.max_all = max(self.max_all, float(score))

        # Validity fields are optional (depends on which JSONLs are available).
        if penalty_applied is None or in_subset is None:
            return

        if bool(penalty_applied):
            self.n_penalty += 1
            return

        if not bool(in_subset):
            # Defensive: in our chess reward fn, in_subset==False should normally coincide with a penalty.
            return

        self.n_valid += 1
        if not _is_finite(score):
            self.n_valid_non_finite += 1
        else:
            self.min_valid = min(self.min_valid, float(score))
            self.max_valid = max(self.max_valid, float(score))

    def _range(self, lo: float, hi: float) -> float:
        if lo == float("inf") or hi == -float("inf"):
            return float("nan")
        return float(hi - lo)

    @property
    def score_range_all(self) -> float:
        return self._range(self.min_all, self.max_all)

    @property
    def score_range_valid(self) -> float:
        return self._range(self.min_valid, self.max_valid)


def _discover_log_layout(evidence_dir: Path) -> tuple[str, Path]:
    files_root = evidence_dir / "files"
    if not files_root.exists():
        raise FileNotFoundError(f"Missing evidence files dir: {files_root}")

    allowed_dir = files_root / "allowed_move_elim_rounds"
    if allowed_dir.exists():
        # Require at least one file matching the expected pattern.
        for p in allowed_dir.glob("*.jsonl"):
            if ALLOWED_MOVE_ELIM_FILE_RE.match(p.name):
                return "allowed_move_elim_rounds", allowed_dir

    rollout_dir = files_root / "rollout_logs"
    if rollout_dir.exists():
        for p in rollout_dir.glob("*.jsonl"):
            if ROLLOUT_FILE_RE.match(p.name):
                return "rollout_logs", rollout_dir

    raise FileNotFoundError(
        "Could not find supported log layout under evidence dir. "
        f"Tried: {allowed_dir} and {rollout_dir}"
    )


def _available_steps_allowed_move_elim(log_dir: Path) -> list[int]:
    steps: set[int] = set()
    for p in log_dir.glob("*.jsonl"):
        m = ALLOWED_MOVE_ELIM_FILE_RE.match(p.name)
        if not m:
            continue
        step = _safe_int(m.group("step"))
        if step is not None:
            steps.add(int(step))
    return sorted(steps)


def _available_steps_rollout(log_dir: Path) -> list[int]:
    steps: set[int] = set()
    for p in log_dir.glob("*.jsonl"):
        m = ROLLOUT_FILE_RE.match(p.name)
        if not m:
            continue
        step = _safe_int(m.group("step"))
        if step is not None:
            steps.add(int(step))
    return sorted(steps)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_allowed_move_elim_step(
    log_dir: Path, *, step: int, run_id: str, requested_step: int
) -> list[dict[str, Any]]:
    round_files: list[tuple[int, Path]] = []
    for p in log_dir.glob(f"{step}_round*.jsonl"):
        m = ALLOWED_MOVE_ELIM_FILE_RE.match(p.name)
        if not m:
            continue
        round_idx = _safe_int(m.group("round"))
        if round_idx is None:
            continue
        round_files.append((int(round_idx), p))
    round_files.sort(key=lambda x: x[0])
    if not round_files:
        raise FileNotFoundError(f"No allowed_move_elim_rounds files found for step={step} under {log_dir}")

    groups: dict[tuple[int, int], GroupRangeAgg] = {}
    any_valid_fields = False
    for round_idx, path in round_files:
        for rec in _iter_jsonl(path):
            prompt_idx = _safe_int(rec.get("allowed_move_elim_prompt_idx"))
            if prompt_idx is None:
                raise ValueError(f"Missing allowed_move_elim_prompt_idx in {path}")

            key = (int(round_idx), int(prompt_idx))
            agg = groups.get(key)
            if agg is None:
                agg = GroupRangeAgg()
                groups[key] = agg

            penalty_applied = rec.get("penalty_applied")
            in_subset = rec.get("in_subset")
            if penalty_applied is not None and in_subset is not None:
                any_valid_fields = True

            agg.add(
                score=_safe_float(rec.get("score")),
                penalty_applied=bool(penalty_applied) if penalty_applied is not None else None,
                in_subset=bool(in_subset) if in_subset is not None else None,
                chess_reward_fn=str(rec.get("chess_reward_fn") or ""),
            )

    rows: list[dict[str, Any]] = []
    for (round_idx, prompt_idx), agg in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rows.append(
            {
                "run_id": str(run_id),
                "log_layout": "allowed_move_elim_rounds",
                "requested_step": int(requested_step),
                "used_step": int(step),
                "group_key": f"round={round_idx},prompt_idx={prompt_idx}",
                "round": int(round_idx),
                "prompt_idx": int(prompt_idx),
                "uid": "",
                "rollout_n_expected": None,
                "n_total": int(agg.n_total),
                "n_total_non_finite": int(agg.n_total_non_finite),
                "n_penalty": int(agg.n_penalty),
                "n_valid": int(agg.n_valid) if any_valid_fields else None,
                "n_valid_non_finite": int(agg.n_valid_non_finite) if any_valid_fields else None,
                "score_min_all": float(agg.min_all) if agg.min_all != float("inf") else float("nan"),
                "score_max_all": float(agg.max_all) if agg.max_all != -float("inf") else float("nan"),
                "score_range_all": float(agg.score_range_all),
                "score_min_valid": float(agg.min_valid) if agg.min_valid != float("inf") else float("nan"),
                "score_max_valid": float(agg.max_valid) if agg.max_valid != -float("inf") else float("nan"),
                "score_range_valid": float(agg.score_range_valid) if any_valid_fields else float("nan"),
                "chess_reward_fn": str(agg.chess_reward_fn),
                "valid_fields_present": bool(any_valid_fields),
            }
        )
    return rows


def _load_rollout_logs_step(
    log_dir: Path, *, step: int, run_id: str, requested_step: int
) -> list[dict[str, Any]]:
    path = log_dir / f"{step}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing rollout log file: {path}")

    groups: dict[str, GroupRangeAgg] = {}
    any_valid_fields = False

    for rec in _iter_jsonl(path):
        uid = str(rec.get("uid") or "").strip()
        if not uid:
            raise ValueError(f"Missing uid in rollout_logs file: {path}")

        agg = groups.get(uid)
        if agg is None:
            agg = GroupRangeAgg()
            groups[uid] = agg

        penalty_applied = rec.get("penalty_applied")
        in_subset = rec.get("in_subset")
        if penalty_applied is not None and in_subset is not None:
            any_valid_fields = True

        agg.add(
            score=_safe_float(rec.get("score")),
            penalty_applied=bool(penalty_applied) if penalty_applied is not None else None,
            in_subset=bool(in_subset) if in_subset is not None else None,
            chess_reward_fn=str(rec.get("chess_reward_fn") or ""),
        )

    rows: list[dict[str, Any]] = []
    for uid, agg in sorted(groups.items(), key=lambda kv: kv[0]):
        rows.append(
            {
                "run_id": str(run_id),
                "log_layout": "rollout_logs",
                "requested_step": int(requested_step),
                "used_step": int(step),
                "group_key": str(uid),
                "round": None,
                "prompt_idx": None,
                "uid": str(uid),
                "rollout_n_expected": None,
                "n_total": int(agg.n_total),
                "n_total_non_finite": int(agg.n_total_non_finite),
                "n_penalty": int(agg.n_penalty),
                "n_valid": int(agg.n_valid) if any_valid_fields else None,
                "n_valid_non_finite": int(agg.n_valid_non_finite) if any_valid_fields else None,
                "score_min_all": float(agg.min_all) if agg.min_all != float("inf") else float("nan"),
                "score_max_all": float(agg.max_all) if agg.max_all != -float("inf") else float("nan"),
                "score_range_all": float(agg.score_range_all),
                "score_min_valid": float(agg.min_valid) if agg.min_valid != float("inf") else float("nan"),
                "score_max_valid": float(agg.max_valid) if agg.max_valid != -float("inf") else float("nan"),
                "score_range_valid": float(agg.score_range_valid) if any_valid_fields else float("nan"),
                "chess_reward_fn": str(agg.chess_reward_fn),
                "valid_fields_present": bool(any_valid_fields),
            }
        )
    return rows


def _percentile(vals: np.ndarray, q: float) -> float:
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def _summarize_ranges(values: pd.Series) -> dict[str, float]:
    arr = values.dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "frac_eq_0": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": _percentile(arr, 90),
        "p99": _percentile(arr, 99),
        "frac_eq_0": float(np.mean(arr == 0.0)),
    }


def _plot_hist_overlay(
    *,
    out_path: Path,
    title: str,
    all_vals: np.ndarray,
    valid_vals: Optional[np.ndarray],
    bin_edges: np.ndarray,
    x_min: float,
    x_max: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(all_vals, bins=bin_edges, alpha=0.75, label="all (includes penalties)")
    if valid_vals is not None and valid_vals.size > 0:
        ax.hist(valid_vals, bins=bin_edges, alpha=0.65, label="valid in-subset only")
    ax.set_title(title)
    ax.set_xlabel("score_range = max(score) - min(score)")
    ax.set_ylabel("GRPO group count")
    ax.set_xlim([x_min, x_max])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_url_from_meta(evidence_dir: Path) -> str:
    meta_path = evidence_dir / "run_meta.json"
    if not meta_path.exists():
        return ""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(meta.get("url") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as RUN_ID=EVIDENCE_DIR (repeatable).",
    )
    ap.add_argument(
        "--target_steps",
        default="20,40,60",
        help="Comma-separated list of target global steps (default: 20,40,60).",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--x_min", type=float, default=0.0)
    ap.add_argument(
        "--x_max",
        type=float,
        default=None,
        help="Optional fixed x-axis max for histograms. If omitted, inferred from data across all runs.",
    )
    args = ap.parse_args()

    target_steps = _parse_steps_arg(args.target_steps)
    if not target_steps:
        raise SystemExit("No target steps provided.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, Path]] = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"Invalid --run spec (expected RUN_ID=DIR): {spec!r}")
        run_id, dir_s = spec.split("=", 1)
        run_id = run_id.strip()
        evidence_dir = Path(dir_s.strip())
        if not run_id:
            raise SystemExit(f"Invalid --run spec (empty run_id): {spec!r}")
        if not evidence_dir.exists():
            raise SystemExit(f"Evidence dir not found for run {run_id}: {evidence_dir}")
        runs.append((run_id, evidence_dir))

    all_rows: list[dict[str, Any]] = []
    step_mapping: dict[str, dict[str, int]] = {}

    for run_id, evidence_dir in runs:
        layout, log_dir = _discover_log_layout(evidence_dir)
        if layout == "allowed_move_elim_rounds":
            available_steps = _available_steps_allowed_move_elim(log_dir)
        elif layout == "rollout_logs":
            available_steps = _available_steps_rollout(log_dir)
        else:
            raise AssertionError(f"Unexpected log layout: {layout}")
        if not available_steps:
            raise SystemExit(f"No steps found under {log_dir}")

        step_mapping[run_id] = {}
        for req_step in target_steps:
            used_step = _choose_nearest_step(req_step, available_steps)
            step_mapping[run_id][str(req_step)] = int(used_step)

            if layout == "allowed_move_elim_rounds":
                rows = _load_allowed_move_elim_step(
                    log_dir, step=used_step, run_id=run_id, requested_step=req_step
                )
            else:
                rows = _load_rollout_logs_step(log_dir, step=used_step, run_id=run_id, requested_step=req_step)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("Parsed 0 groups; unexpected.")

    df = df.sort_values(["run_id", "requested_step", "used_step", "group_key"]).reset_index(drop=True)

    # Persist raw per-group rows (small enough for offline repro; gzip for safety).
    df.to_csv(out_dir / "group_ranges.csv.gz", index=False, compression="gzip")

    # Compute a global x_max/bins across all runs in this invocation.
    x_min = float(args.x_min)
    if args.x_max is not None:
        x_max = float(args.x_max)
    else:
        max_seen = float(df["score_range_all"].replace([np.inf, -np.inf], np.nan).max())
        if not np.isfinite(max_seen):
            max_seen = 1.0
        # Round up slightly to make axis stable across floating jitter.
        if max_seen <= 1.0 + 1e-9:
            x_max = 1.0
        elif max_seen <= 2.0 + 1e-9:
            x_max = 2.0
        else:
            x_max = float(np.ceil(max_seen * 10.0) / 10.0)

    if x_max <= x_min:
        raise SystemExit(f"Invalid x-axis range: x_min={x_min} x_max={x_max}")

    bins = int(args.bins)
    if bins <= 0:
        raise SystemExit("--bins must be > 0")
    bin_edges = np.linspace(x_min, x_max, bins + 1)

    # Summary stats per run + requested step.
    summary_rows: list[dict[str, Any]] = []
    for (run_id, req_step), g in df.groupby(["run_id", "requested_step"], dropna=False):
        used_steps = sorted({int(s) for s in g["used_step"].dropna().tolist()})
        reward_fns = sorted({str(s) for s in g["chess_reward_fn"].dropna().tolist() if str(s)})

        s_all = _summarize_ranges(g["score_range_all"])
        s_valid = _summarize_ranges(g["score_range_valid"])
        frac_all_penalty_groups = float(np.mean((g["n_penalty"].fillna(0).astype(float) > 0.0).to_numpy()))
        frac_all_invalid_groups = float(np.mean((g["n_valid"].fillna(0).astype(float) == 0.0).to_numpy()))

        summary_rows.append(
            {
                "run_id": str(run_id),
                "run_url": _run_url_from_meta(next(d for r, d in runs if r == run_id)),
                "requested_step": int(req_step),
                "used_steps": ",".join(str(s) for s in used_steps),
                "log_layout": ",".join(sorted({str(x) for x in g["log_layout"].unique()})),
                "chess_reward_fns": ",".join(reward_fns),
                "groups_total": int(g.shape[0]),
                "groups_with_any_penalty_frac": float(frac_all_penalty_groups),
                "groups_with_zero_valid_frac": float(frac_all_invalid_groups),
                "score_range_all_count": int(s_all["count"]),
                "score_range_all_mean": float(s_all["mean"]),
                "score_range_all_median": float(s_all["median"]),
                "score_range_all_p90": float(s_all["p90"]),
                "score_range_all_p99": float(s_all["p99"]),
                "score_range_all_frac_eq_0": float(s_all["frac_eq_0"]),
                "score_range_valid_count": int(s_valid["count"]),
                "score_range_valid_mean": float(s_valid["mean"]),
                "score_range_valid_median": float(s_valid["median"]),
                "score_range_valid_p90": float(s_valid["p90"]),
                "score_range_valid_p99": float(s_valid["p99"]),
                "score_range_valid_frac_eq_0": float(s_valid["frac_eq_0"]),
            }
        )

        # Plots: one per run+requested step.
        all_vals = g["score_range_all"].dropna().to_numpy(dtype=float)
        all_vals = all_vals[np.isfinite(all_vals)]
        valid_vals = g["score_range_valid"].dropna().to_numpy(dtype=float)
        valid_vals = valid_vals[np.isfinite(valid_vals)]
        valid_vals_opt: Optional[np.ndarray] = valid_vals if valid_vals.size > 0 else None

        title = f"{run_id}  step {req_step} (used {used_steps})  score_range"
        out_path = out_dir / "plots" / f"{run_id}_step{int(req_step):04d}_score_range_hist.png"
        _plot_hist_overlay(
            out_path=out_path,
            title=title,
            all_vals=all_vals,
            valid_vals=valid_vals_opt,
            bin_edges=bin_edges,
            x_min=x_min,
            x_max=x_max,
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["run_id", "requested_step"]).reset_index(drop=True)
    summary_df.to_csv(out_dir / "summary_by_run_step.csv", index=False)

    _write_json(out_dir / "step_mapping.json", step_mapping)

    # Also write a lightweight human-readable summary.txt.
    lines: list[str] = []
    lines.append(f"out_dir: {out_dir}")
    lines.append(f"target_steps: {target_steps}")
    lines.append(f"x_range: [{x_min}, {x_max}] bins={bins}")
    lines.append("")
    for row in summary_rows:
        lines.append(
            f"{row['run_id']} step {row['requested_step']} (used {row['used_steps']}): "
            f"groups={row['groups_total']} "
            f"range_all mean={row['score_range_all_mean']:.6g} p90={row['score_range_all_p90']:.6g} "
            f"p99={row['score_range_all_p99']:.6g} frac0={row['score_range_all_frac_eq_0']:.3f}"
        )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {out_dir/'group_ranges.csv.gz'}")
    print(f"[OK] wrote: {out_dir/'summary_by_run_step.csv'}")
    print(f"[OK] wrote: {out_dir/'step_mapping.json'}")
    print(f"[OK] plots: {out_dir/'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
