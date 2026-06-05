#!/usr/bin/env python3
"""Post-process judge-gating artifacts with the explicit contradiction backstop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diag_llm_gating_vllm import (
    _bucket_summary,
    _explicit_singleton_contradiction,
    _select_representatives,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cases_path = Path(args.cases_jsonl)
    if not cases_path.exists():
        raise SystemExit(f"Missing cases jsonl: {cases_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(cases_path)
    updated: list[dict[str, Any]] = []
    for row in rows:
        contradiction = _explicit_singleton_contradiction(
            visible_candidate_count=int(len(row.get("visible_considered_moves_uci") or [])),
            response_text=str(row.get("y_response_text") or ""),
            think_text=str(row.get("y_think_text") or ""),
        )
        row = dict(row)
        row["explicit_singleton_contradiction"] = contradiction
        row["judge_effective_verdict"] = (
            "REJECT" if (str(row.get("judge_verdict") or "").upper() == "REJECT" or contradiction) else "ACCEPT"
        )
        row["judge_effective_reason"] = (
            str(row.get("judge_reason") or "")
            if not contradiction
            else f"{contradiction['reason']}: {contradiction['matched_text']}"
        )
        updated.append(row)

    clean_records = [row for row in updated if str(row.get("bucket") or "") == "clean"]
    suspicious_records = [row for row in updated if str(row.get("bucket") or "") == "suspicious"]

    summary = {
        "source_cases_jsonl": str(cases_path),
        "clean_bucket": _bucket_summary(clean_records),
        "suspicious_bucket": _bucket_summary(suspicious_records),
        "clean_false_positives": [
            row["case_id"] for row in clean_records if str(row.get("judge_effective_verdict") or "") == "REJECT"
        ],
        "representative_case_ids": {
            "clean_accepts": [
                row["case_id"] for row in _select_representatives(clean_records, verdict="ACCEPT", limit=3)
            ],
            "clean_rejects": [
                row["case_id"] for row in _select_representatives(clean_records, verdict="REJECT", limit=3)
            ],
            "suspicious_rejects": [
                row["case_id"] for row in _select_representatives(suspicious_records, verdict="REJECT", limit=3)
            ],
            "suspicious_accepts": [
                row["case_id"] for row in _select_representatives(suspicious_records, verdict="ACCEPT", limit=3)
            ],
        },
        "reject_rate_gap": (
            _bucket_summary(suspicious_records)["reject_rate"] - _bucket_summary(clean_records)["reject_rate"]
        ),
    }
    representatives = {
        "clean_accepts": _select_representatives(clean_records, verdict="ACCEPT", limit=3),
        "clean_rejects": _select_representatives(clean_records, verdict="REJECT", limit=3),
        "suspicious_rejects": _select_representatives(suspicious_records, verdict="REJECT", limit=3),
        "suspicious_accepts": _select_representatives(suspicious_records, verdict="ACCEPT", limit=3),
    }

    _write_jsonl(out_dir / "cases_posthoc.jsonl", updated)
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "representatives.json", representatives)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
