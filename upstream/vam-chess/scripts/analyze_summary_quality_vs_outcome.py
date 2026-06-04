#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join summary-quality ratings to baseline/iterative outcomes for explored-path analysis."
    )
    parser.add_argument(
        "--scored-examples",
        default="analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval/summary_quality_scored_examples.jsonl",
    )
    parser.add_argument(
        "--baseline-traces-dir",
        default="analysis/math_explored_paths/cluster_job_3877433/raw_baseline_traces",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def bucket_quality(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 3.0:
        return "low_lt3"
    if value < 4.0:
        return "mid_3_to_lt4"
    return "high_ge4"


def problem_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["dataset_key"]), int(row["max_tokens"]), int(row["problem_index"]))


def build_problem_records(
    scored_examples: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    iterative_by_problem: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    baseline_by_problem: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)

    for row in scored_examples:
        iterative_by_problem[problem_key(row)].append(row)
    for row in baseline_rows:
        key = (str(row["dataset_key"]), int(row["max_tokens"]), int(row["problem_index"]))
        baseline_by_problem[key].append(row)

    all_keys = sorted(set(iterative_by_problem) | set(baseline_by_problem))
    problem_records: list[dict[str, Any]] = []
    for key in all_keys:
        iter_rows = sorted(iterative_by_problem.get(key, []), key=lambda row: int(row["attempt_index"]))
        base_rows = sorted(baseline_by_problem.get(key, []), key=lambda row: int(row["attempt_index"]))
        if not iter_rows:
            continue

        dataset_key, max_tokens, problem_index = key
        first = iter_rows[0]
        baseline_success = any(bool(row["correct"]) for row in base_rows)
        iterative_success = any(bool(row["correct"]) for row in iter_rows)
        if iterative_success and not baseline_success:
            outcome_bucket = "iterative_only_win"
        elif baseline_success and not iterative_success:
            outcome_bucket = "baseline_only_win"
        elif baseline_success and iterative_success:
            outcome_bucket = "both_win"
        else:
            outcome_bucket = "both_fail"

        ratings = [int(row["rating"]) for row in iter_rows]
        failed_rows = [row for row in iter_rows if not bool(row["correct"])]
        failed_ratings = [int(row["rating"]) for row in failed_rows]
        correct_rows = [row for row in iter_rows if bool(row["correct"])]
        first_success_attempt = min((int(row["attempt_index"]) for row in correct_rows), default=None)
        if first_success_attempt is None:
            pre_success_failed_rows = failed_rows
        else:
            pre_success_failed_rows = [
                row for row in failed_rows if int(row["attempt_index"]) < first_success_attempt
            ]

        pre_success_failed_ratings = [int(row["rating"]) for row in pre_success_failed_rows]

        record = {
            "dataset_key": dataset_key,
            "max_tokens": max_tokens,
            "problem_index": problem_index,
            "problem_id": first["problem_id"],
            "ground_truth": first["ground_truth"],
            "baseline_success": baseline_success,
            "iterative_success": iterative_success,
            "outcome_bucket": outcome_bucket,
            "first_iterative_success_attempt": first_success_attempt,
            "num_attempts": len(iter_rows),
            "all_summary_mean_rating": mean_or_none(ratings),
            "failed_summary_mean_rating": mean_or_none(failed_ratings),
            "pre_success_failed_summary_mean_rating": mean_or_none(pre_success_failed_ratings),
            "correct_attempt_summary_mean_rating": mean_or_none([int(row["rating"]) for row in correct_rows]),
            "low_quality_fraction": (sum(1 for rating in ratings if rating <= 2) / len(ratings)) if ratings else None,
            "good_quality_fraction": (sum(1 for rating in ratings if rating >= 4) / len(ratings)) if ratings else None,
            "quality_bin_failed_summary_mean": bucket_quality(mean_or_none(failed_ratings)),
            "quality_bin_pre_success_failed_mean": bucket_quality(mean_or_none(pre_success_failed_ratings)),
        }
        problem_records.append(record)
    return problem_records


def aggregate_records(
    rows: list[dict[str, Any]],
    group_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        record = {field: value for field, value in zip(group_fields, key)}
        record["count"] = len(items)
        record["iterative_success_rate"] = statistics.mean(1.0 if row["iterative_success"] else 0.0 for row in items)
        record["baseline_success_rate"] = statistics.mean(1.0 if row["baseline_success"] else 0.0 for row in items)
        record["mean_all_summary_rating"] = mean_or_none(
            [float(row["all_summary_mean_rating"]) for row in items if row["all_summary_mean_rating"] is not None]
        )
        record["mean_failed_summary_rating"] = mean_or_none(
            [float(row["failed_summary_mean_rating"]) for row in items if row["failed_summary_mean_rating"] is not None]
        )
        record["mean_pre_success_failed_summary_rating"] = mean_or_none(
            [
                float(row["pre_success_failed_summary_mean_rating"])
                for row in items
                if row["pre_success_failed_summary_mean_rating"] is not None
            ]
        )
        record["mean_low_quality_fraction"] = mean_or_none(
            [float(row["low_quality_fraction"]) for row in items if row["low_quality_fraction"] is not None]
        )
        record["mean_good_quality_fraction"] = mean_or_none(
            [float(row["good_quality_fraction"]) for row in items if row["good_quality_fraction"] is not None]
        )
        out.append(record)
    return out


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall_by_outcome_bucket"]
    baseline_fail = summary["baseline_fail_only_by_quality_bin"]
    baseline_fail_by_len = summary["baseline_fail_only_by_len_and_quality_bin"]
    baseline_fail_direct = summary["baseline_fail_direct_comparison"]

    lines = [
        "# Summary Quality vs Outcome",
        "",
        "This analysis joins per-attempt summary-quality ratings back to the per-problem baseline vs iterative outcomes.",
        "",
        "## Overall By Outcome Bucket",
        "",
        "| outcome | count | iterative success rate | baseline success rate | mean failed-summary rating | mean pre-success failed-summary rating |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall:
        lines.append(
            "| "
            f"{row['outcome_bucket']} | {row['count']} | {fmt(row['iterative_success_rate'])} | "
            f"{fmt(row['baseline_success_rate'])} | {fmt(row['mean_failed_summary_rating'])} | "
            f"{fmt(row['mean_pre_success_failed_summary_rating'])} |"
        )

    lines.extend(
        [
            "",
            "## Baseline-Fail Problems Only",
            "",
            "These rows directly test whether better failed-attempt summaries are associated with iterative recovery when baseline sampling failed.",
            "",
            "### By Failed-Summary Quality Bin",
            "",
            "| quality bin | count | iterative success rate | mean failed-summary rating |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in baseline_fail:
        lines.append(
            "| "
            f"{row['quality_bin_failed_summary_mean']} | {row['count']} | {fmt(row['iterative_success_rate'])} | "
            f"{fmt(row['mean_failed_summary_rating'])} |"
        )

    lines.extend(
        [
            "",
            "### By Failed-Summary Quality Bin And Output Length",
            "",
            "| max tokens | quality bin | count | iterative success rate | mean failed-summary rating |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in baseline_fail_by_len:
        lines.append(
            "| "
            f"{row['max_tokens']} | {row['quality_bin_failed_summary_mean']} | {row['count']} | "
            f"{fmt(row['iterative_success_rate'])} | {fmt(row['mean_failed_summary_rating'])} |"
        )

    lines.extend(
        [
            "",
            "### Direct Comparison Within Baseline-Fail Problems",
            "",
            "| outcome | count | mean failed-summary rating | mean pre-success failed-summary rating |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in baseline_fail_direct:
        lines.append(
            "| "
            f"{row['outcome_bucket']} | {row['count']} | {fmt(row['mean_failed_summary_rating'])} | "
            f"{fmt(row['mean_pre_success_failed_summary_rating'])} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    scored_path = Path(args.scored_examples)
    baseline_dir = Path(args.baseline_traces_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_examples = read_jsonl(scored_path)
    baseline_rows: list[dict[str, Any]] = []
    for path in sorted(baseline_dir.glob("*.jsonl")):
        baseline_rows.extend(read_jsonl(path))

    problem_records = build_problem_records(scored_examples, baseline_rows)
    overall_by_outcome_bucket = aggregate_records(problem_records, ["outcome_bucket"])
    baseline_fail_only = [row for row in problem_records if not row["baseline_success"]]
    baseline_fail_only_by_quality_bin = aggregate_records(baseline_fail_only, ["quality_bin_failed_summary_mean"])
    baseline_fail_only_by_len_and_quality_bin = aggregate_records(
        baseline_fail_only, ["max_tokens", "quality_bin_failed_summary_mean"]
    )
    baseline_fail_direct_comparison = aggregate_records(
        [row for row in baseline_fail_only if row["outcome_bucket"] in {"iterative_only_win", "both_fail"}],
        ["outcome_bucket"],
    )

    summary = {
        "num_problem_records": len(problem_records),
        "overall_by_outcome_bucket": overall_by_outcome_bucket,
        "baseline_fail_only_by_quality_bin": baseline_fail_only_by_quality_bin,
        "baseline_fail_only_by_len_and_quality_bin": baseline_fail_only_by_len_and_quality_bin,
        "baseline_fail_direct_comparison": baseline_fail_direct_comparison,
    }

    write_jsonl(out_dir / "summary_quality_vs_outcome_problem_level.jsonl", problem_records)
    write_csv(out_dir / "summary_quality_vs_outcome_problem_level.csv", problem_records)
    write_json(out_dir / "summary_quality_vs_outcome_summary.json", summary)
    (out_dir / "summary_quality_vs_outcome_summary.md").write_text(build_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
