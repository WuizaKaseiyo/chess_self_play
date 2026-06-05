#!/usr/bin/env python3
"""
Summarize evaluation-time format failures from locally-downloaded W&B evidence.

This focuses on:
  - `files/validation_logs/<step>.jsonl` (selection prompt validation)
  - `files/global_step_<step>/moves.jsonl` (full-game eval move traces)

Outputs CSVs + simple plots so investigations can directly attribute ACPL / eval dips
to format compliance vs policy quality.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v


VAL_RE = re.compile(r"^(?P<step>\d+)\.jsonl$")
GLOBAL_RE = re.compile(r"^global_step_(?P<step>\d+)$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence_dir", required=True, help="analysis/wandb_evidence/<run_id>")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title_prefix", default="", help="Optional plot title prefix (e.g. 'et48q0cr — ')")
    args = ap.parse_args()

    evidence_dir = Path(args.evidence_dir)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    _ensure_dir(plots_dir)
    _ensure_dir(tables_dir)

    files_dir = evidence_dir / "files"

    # 1) Validation logs.
    val_dir = files_dir / "validation_logs"
    val_rows: list[dict[str, Any]] = []
    if val_dir.exists():
        for p in sorted(val_dir.glob("*.jsonl")):
            m = VAL_RE.match(p.name)
            if not m:
                continue
            step = int(m.group("step"))
            n = 0
            n_pen = 0
            n_fmt0 = 0
            reasons = Counter()
            for rec in _iter_jsonl(p):
                n += 1
                if bool(rec.get("penalty_applied", False)):
                    n_pen += 1
                pr = str(rec.get("penalty_reason") or "")
                reasons[pr] += 1
                if _safe_float(rec.get("format_reward", 0.0), 0.0) < 1.0:
                    n_fmt0 += 1
            val_rows.append(
                {
                    "step": step,
                    "n": n,
                    "penalty_applied": n_pen,
                    "penalty_frac": (n_pen / n) if n else 0.0,
                    "format_reward_lt1": n_fmt0,
                    "format_error_frac": (n_fmt0 / n) if n else 0.0,
                    "penalty_reason_empty": reasons.get("", 0),
                    "penalty_reason_format_error": reasons.get("format_error", 0),
                    "penalty_reason_out_of_subset": reasons.get("out_of_subset", 0),
                    "penalty_reason_bad_move": reasons.get("bad_move", 0),
                }
            )

    val_df = pd.DataFrame(val_rows).sort_values("step") if val_rows else pd.DataFrame()
    if not val_df.empty:
        val_df.to_csv(tables_dir / "validation_format_failures_by_step.csv", index=False)
        plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.plot(val_df["step"], val_df["format_error_frac"], marker="o", linewidth=1.6)
        ax.set_title(f"{args.title_prefix}validation: format_error_frac (format_reward<1)")
        ax.set_xlabel("step")
        ax.set_ylabel("frac")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / "validation_format_error_frac.png", dpi=180, bbox_inches="tight")
        plt.close()

    # 2) Full-game eval move traces.
    global_rows: list[dict[str, Any]] = []
    for d in sorted(files_dir.glob("global_step_*")):
        if not d.is_dir():
            continue
        m = GLOBAL_RE.match(d.name)
        if not m:
            continue
        step = int(m.group("step"))
        moves_path = d / "moves.jsonl"
        if not moves_path.exists():
            continue
        n = 0
        n_format_missing = 0
        errors = Counter()
        for rec in _iter_jsonl(moves_path):
            n += 1
            if rec.get("format_ok") is False:
                n_format_missing += 1
            er = rec.get("error_reason")
            if er:
                errors[str(er)] += 1
        global_rows.append(
            {
                "step": step,
                "n_moves": n,
                "format_missing": n_format_missing,
                "format_missing_frac": (n_format_missing / n) if n else 0.0,
                "error_reason_format_missing": errors.get("format_missing", 0),
                "error_reason_forfeit_format_missing": errors.get("forfeit:format_missing", 0),
                "error_reason_illegal_move": errors.get("illegal_move", 0),
                "error_reason_forfeit_illegal_move": errors.get("forfeit:illegal_move", 0),
            }
        )

    global_df = pd.DataFrame(global_rows).sort_values("step") if global_rows else pd.DataFrame()
    if not global_df.empty:
        global_df.to_csv(tables_dir / "full_game_eval_format_failures_by_step.csv", index=False)
        plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.plot(global_df["step"], global_df["format_missing_frac"], marker="o", linewidth=1.6)
        ax.set_title(f"{args.title_prefix}full_game_eval: format_missing_frac (format_ok==False)")
        ax.set_xlabel("step")
        ax.set_ylabel("frac")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / "full_game_eval_format_missing_frac.png", dpi=180, bbox_inches="tight")
        plt.close()

    print(f"[OK] wrote tables/plots under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

