#!/usr/bin/env python3
"""
Diff config_api.yaml across multiple W&B runs from local evidence bundles.

This is meant to answer: "are these runs identical except for X?"

Inputs:
  analysis/wandb_evidence/<run_id>/config_api.yaml

Outputs (gitignored; under `outputs/`):
  - diff.csv
  - diff.md

Example
-------
conda run -n verl python scripts/diff_wandb_run_configs.py \\
  --evidence_root analysis/wandb_evidence \\
  --out_dir outputs/wandb_config_diff/am4_reward_fn_20260121 \\
  --runs n1ihbyab:expected_score 6g3tapxg:winrate q58ea8sy:rank
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd
import yaml


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_runs(items: list[str]) -> list[RunSpec]:
    out: list[RunSpec] = []
    for item in items:
        if ":" in item:
            run_id, label = item.split(":", 1)
        else:
            run_id, label = item, item
        out.append(RunSpec(run_id=run_id.strip(), label=label.strip()))
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            dotted = f"{prefix}.{key}" if prefix else key
            out.update(_flatten(v, dotted))
        return out
    if isinstance(obj, list):
        # Preserve list values as a whole (avoid exploding to per-index keys unless needed).
        out[prefix] = obj
        return out
    out[prefix] = obj
    return out


def _normalize_value(v: Any) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, default=str)
    if v is None:
        return "null"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence_root", default="analysis/wandb_evidence")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="Run specs formatted as RUN_ID:LABEL")
    args = ap.parse_args()

    evidence_root = Path(args.evidence_root)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    runs = _parse_runs(list(args.runs))
    flat_by_run: dict[str, dict[str, Any]] = {}

    for rs in runs:
        cfg_path = evidence_root / rs.run_id / "config_api.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config_api.yaml: {cfg_path}")
        cfg = _read_yaml(cfg_path)
        flat_by_run[rs.run_id] = _flatten(cfg)

    all_keys: set[str] = set()
    for flat in flat_by_run.values():
        all_keys.update(flat.keys())
    keys_sorted = sorted(all_keys)

    rows: list[dict[str, Any]] = []
    for k in keys_sorted:
        row: dict[str, Any] = {"key": k}
        vals = []
        for rs in runs:
            v = flat_by_run[rs.run_id].get(k, None)
            s = _normalize_value(v)
            row[rs.run_id] = s
            vals.append(s)
        row["all_equal"] = len(set(vals)) == 1
        rows.append(row)

    df = pd.DataFrame(rows)
    diff_df = df[df["all_equal"] == False].copy()  # noqa: E712
    diff_df.to_csv(out_dir / "diff.csv", index=False)

    md = []
    md.append("# W&B config_api.yaml diff (flattened)\n\n")
    md.append(f"Evidence root: `{evidence_root}`\n\n")
    md.append(f"Runs compared: {', '.join([rs.run_id for rs in runs])}\n\n")
    md.append(f"- total flattened keys: {len(df)}\n")
    md.append(f"- differing keys: {len(diff_df)}\n\n")
    md.append("## Differing keys\n\n")
    md.append("| key | " + " | ".join([rs.run_id for rs in runs]) + " |\n")
    md.append("|---" + "|---" * len(runs) + "|\n")
    for _, r in diff_df.iterrows():
        md.append("| " + str(r["key"]) + " | " + " | ".join([str(r[rs.run_id]) for rs in runs]) + " |\n")
    (out_dir / "diff.md").write_text("".join(md), encoding="utf-8")

    print(f"[OK] wrote config diff under: {out_dir}")
    print(f"[OK] differing keys: {len(diff_df)} / {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

