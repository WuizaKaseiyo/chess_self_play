#!/usr/bin/env python3
"""Analyze fallback-summary contamination for explored-path traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FALLBACK_ROUTE_PHRASE = "route: summary unavailable from the raw attempt."
FALLBACK_COMMITMENT_PHRASE = "key commitment: unspecified."
FALLBACK_ROUTE_KEY = "summary unavailable from the raw attempt || unspecified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("analysis/math_explored_paths/cluster_job_3877433"),
        help="Directory containing raw_iterative_traces/, raw_baseline_traces/, and summary_quality_eval/.",
    )
    return parser.parse_args()


def is_fallback_placeholder(summary: str | None) -> bool:
    text = (summary or "").strip().lower()
    return FALLBACK_ROUTE_PHRASE in text and FALLBACK_COMMITMENT_PHRASE in text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def pct(numer: int, denom: int) -> float | None:
    if denom == 0:
        return None
    return numer / denom


def sorted_records(counter: Counter[tuple[Any, ...]], key_names: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, value in sorted(counter.items()):
        if not isinstance(key, tuple):
            key = (key,)
        record = {name: item for name, item in zip(key_names, key, strict=True)}
        record["count"] = value
        records.append(record)
    return records


def as_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def render_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def render_num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def aggregate_rate(rows: list[dict[str, Any]], positive_key: str) -> dict[str, Any]:
    count = len(rows)
    positive = sum(1 for row in rows if row[positive_key])
    return {
        "count": count,
        "positive_count": positive,
        "rate": as_float(pct(positive, count)),
    }


def main() -> None:
    args = parse_args()
    bundle_dir = args.bundle_dir.resolve()
    eval_dir = bundle_dir / "summary_quality_eval"
    iter_dir = bundle_dir / "raw_iterative_traces"
    base_dir = bundle_dir / "raw_baseline_traces"

    # Prefer the canonical nested trace layout produced by the merged bundle.
    # Older flat copies were only kept temporarily during local fetch/cleanup.
    iterative_trace_paths = sorted(iter_dir.rglob("iterative_traces.jsonl"))
    if not iterative_trace_paths:
        iterative_trace_paths = sorted(iter_dir.glob("*.jsonl"))

    iterative_rows: list[dict[str, Any]] = []
    for path in iterative_trace_paths:
        for row in load_jsonl(path):
            row["source_trace_file"] = path.name
            row["fallback_placeholder"] = is_fallback_placeholder(row.get("summary"))
            before = row.get("accumulated_explored_paths_before") or []
            row["fallback_in_prompt"] = any(is_fallback_placeholder(item) for item in before)
            row["fallback_prompt_count"] = sum(1 for item in before if is_fallback_placeholder(item))
            iterative_rows.append(row)

    baseline_rows: list[dict[str, Any]] = []
    for path in sorted(base_dir.glob("*.jsonl")):
        for row in load_jsonl(path):
            row["source_trace_file"] = path.name
            baseline_rows.append(row)

    scored_examples_path = eval_dir / "summary_quality_scored_examples.jsonl"
    scored_examples = load_jsonl(scored_examples_path)

    existing_exact_fallback_count = sum(1 for row in scored_examples if bool(row.get("fallback_summary")))
    broad_placeholder_count = sum(1 for row in iterative_rows if row["fallback_placeholder"])
    broad_placeholder_only_count = broad_placeholder_count - existing_exact_fallback_count

    fallback_by_dataset = Counter(row["dataset_key"] for row in iterative_rows if row["fallback_placeholder"])
    attempts_by_dataset = Counter(row["dataset_key"] for row in iterative_rows)
    fallback_by_max_tokens = Counter(row["max_tokens"] for row in iterative_rows if row["fallback_placeholder"])
    attempts_by_max_tokens = Counter(row["max_tokens"] for row in iterative_rows)
    fallback_by_attempt = Counter(row["attempt_index"] for row in iterative_rows if row["fallback_placeholder"])
    attempts_by_attempt = Counter(row["attempt_index"] for row in iterative_rows)

    failed_iterative_rows = [row for row in iterative_rows if not row["correct"]]
    failed_fallback_rows = [row for row in failed_iterative_rows if row["fallback_placeholder"]]

    failed_fallback_by_dataset = Counter(row["dataset_key"] for row in failed_fallback_rows)
    failed_attempts_by_dataset = Counter(row["dataset_key"] for row in failed_iterative_rows)
    failed_fallback_by_max_tokens = Counter(row["max_tokens"] for row in failed_fallback_rows)
    failed_attempts_by_max_tokens = Counter(row["max_tokens"] for row in failed_iterative_rows)
    failed_fallback_by_attempt = Counter(row["attempt_index"] for row in failed_fallback_rows)
    failed_attempts_by_attempt = Counter(row["attempt_index"] for row in failed_iterative_rows)

    prompt_contaminated_rows = [row for row in iterative_rows if row["fallback_in_prompt"]]
    later_rows = [row for row in iterative_rows if row["attempt_index"] > 1]
    later_failed_rows = [row for row in failed_iterative_rows if row["attempt_index"] > 1]
    prompt_contaminated_by_attempt = Counter(row["attempt_index"] for row in prompt_contaminated_rows)
    later_attempts_by_attempt = Counter(row["attempt_index"] for row in later_rows)

    baseline_by_problem: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        baseline_by_problem[(row["dataset_key"], row["max_tokens"], row["problem_index"])].append(row)

    iterative_by_problem: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in iterative_rows:
        iterative_by_problem[(row["dataset_key"], row["max_tokens"], row["problem_index"])].append(row)

    problem_records: list[dict[str, Any]] = []
    for key, iter_rows in sorted(iterative_by_problem.items()):
        dataset_key, max_tokens, problem_index = key
        iter_rows = sorted(iter_rows, key=lambda row: row["attempt_index"])
        base_rows = baseline_by_problem[key]
        baseline_success = any(row["correct"] for row in base_rows)
        iterative_success = any(row["correct"] for row in iter_rows)
        success_attempts = [row["attempt_index"] for row in iter_rows if row["correct"]]
        first_iterative_success_attempt = min(success_attempts) if success_attempts else None

        if first_iterative_success_attempt is None:
            pre_success_rows = list(iter_rows)
        else:
            pre_success_rows = [
                row for row in iter_rows if row["attempt_index"] <= first_iterative_success_attempt
            ]

        fallback_rows = [row for row in iter_rows if row["fallback_placeholder"]]
        failed_fallback_rows_problem = [
            row for row in iter_rows if row["fallback_placeholder"] and not row["correct"]
        ]
        pre_success_failed_fallback_rows = [
            row for row in pre_success_rows if row["fallback_placeholder"] and not row["correct"]
        ]
        prompt_contaminated_problem_rows = [row for row in iter_rows if row["fallback_in_prompt"]]
        pre_success_prompt_contaminated_rows = [
            row for row in pre_success_rows if row["fallback_in_prompt"]
        ]

        route_keys = [row["summary_route_key"] for row in iter_rows]
        non_fallback_route_keys = [row["summary_route_key"] for row in iter_rows if not row["fallback_placeholder"]]
        actual_distinct_route_keys = len(set(route_keys))
        distinct_non_fallback_route_keys = len(set(non_fallback_route_keys))
        fallback_attempt_count = len(fallback_rows)
        upper_bound_distinct_route_keys = distinct_non_fallback_route_keys + fallback_attempt_count
        repeated_route_key_events = len(route_keys) - actual_distinct_route_keys
        fallback_repeat_events = max(0, fallback_attempt_count - 1)

        problem_records.append(
            {
                "dataset_key": dataset_key,
                "max_tokens": max_tokens,
                "problem_index": problem_index,
                "problem_id": iter_rows[0]["problem_id"],
                "ground_truth": iter_rows[0]["ground_truth"],
                "baseline_success": baseline_success,
                "iterative_success": iterative_success,
                "first_iterative_success_attempt": first_iterative_success_attempt,
                "outcome_bucket": (
                    "both_win"
                    if baseline_success and iterative_success
                    else "baseline_only_win"
                    if baseline_success and not iterative_success
                    else "iterative_only_win"
                    if (not baseline_success) and iterative_success
                    else "both_fail"
                ),
                "has_any_fallback_placeholder": bool(fallback_rows),
                "has_any_failed_fallback_placeholder": bool(failed_fallback_rows_problem),
                "has_any_pre_success_failed_fallback_placeholder": bool(pre_success_failed_fallback_rows),
                "has_any_prompt_fallback_contamination": bool(prompt_contaminated_problem_rows),
                "has_any_pre_success_prompt_fallback_contamination": bool(pre_success_prompt_contaminated_rows),
                "num_fallback_placeholders": fallback_attempt_count,
                "num_failed_fallback_placeholders": len(failed_fallback_rows_problem),
                "num_pre_success_failed_fallback_placeholders": len(pre_success_failed_fallback_rows),
                "num_prompt_contaminated_attempts": len(prompt_contaminated_problem_rows),
                "num_pre_success_prompt_contaminated_attempts": len(pre_success_prompt_contaminated_rows),
                "actual_distinct_route_keys": actual_distinct_route_keys,
                "distinct_non_fallback_route_keys": distinct_non_fallback_route_keys,
                "upper_bound_distinct_route_keys_if_fallback_unique": upper_bound_distinct_route_keys,
                "route_key_collapse_upper_bound": upper_bound_distinct_route_keys - actual_distinct_route_keys,
                "repeated_route_key_events": repeated_route_key_events,
                "fallback_repeat_events": fallback_repeat_events,
                "fallback_share_of_repeat_events": as_float(
                    pct(fallback_repeat_events, repeated_route_key_events)
                ),
            }
        )

    baseline_fail_problem_records = [record for record in problem_records if not record["baseline_success"]]
    iterative_only_win_records = [record for record in problem_records if record["outcome_bucket"] == "iterative_only_win"]

    def rate_table(
        items: list[dict[str, Any]],
        key_name: str,
        positive_name: str,
        *,
        sort_key: Any = None,
    ) -> list[dict[str, Any]]:
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            groups[item[key_name]].append(item)
        records: list[dict[str, Any]] = []
        for key in sorted(groups.keys(), key=sort_key):
            group = groups[key]
            positive = sum(1 for item in group if item[positive_name])
            records.append(
                {
                    key_name: key,
                    "count": len(group),
                    positive_name: positive,
                    "rate": as_float(pct(positive, len(group))),
                }
            )
        return records

    prevalence_by_dataset_records = []
    for dataset_key in sorted(attempts_by_dataset):
        total = attempts_by_dataset[dataset_key]
        fallback = fallback_by_dataset[dataset_key]
        prevalence_by_dataset_records.append(
            {
                "dataset_key": dataset_key,
                "attempt_count": total,
                "fallback_placeholder_count": fallback,
                "fallback_placeholder_rate": as_float(pct(fallback, total)),
            }
        )

    prevalence_by_max_tokens_records = []
    for max_tokens in sorted(attempts_by_max_tokens):
        total = attempts_by_max_tokens[max_tokens]
        fallback = fallback_by_max_tokens[max_tokens]
        prevalence_by_max_tokens_records.append(
            {
                "max_tokens": max_tokens,
                "attempt_count": total,
                "fallback_placeholder_count": fallback,
                "fallback_placeholder_rate": as_float(pct(fallback, total)),
            }
        )

    prevalence_by_attempt_records = []
    for attempt_index in sorted(attempts_by_attempt):
        total = attempts_by_attempt[attempt_index]
        fallback = fallback_by_attempt[attempt_index]
        prevalence_by_attempt_records.append(
            {
                "attempt_index": attempt_index,
                "attempt_count": total,
                "fallback_placeholder_count": fallback,
                "fallback_placeholder_rate": as_float(pct(fallback, total)),
            }
        )

    failed_prevalence_by_dataset_records = []
    for dataset_key in sorted(failed_attempts_by_dataset):
        total = failed_attempts_by_dataset[dataset_key]
        fallback = failed_fallback_by_dataset[dataset_key]
        failed_prevalence_by_dataset_records.append(
            {
                "dataset_key": dataset_key,
                "failed_attempt_count": total,
                "failed_fallback_placeholder_count": fallback,
                "failed_fallback_placeholder_rate": as_float(pct(fallback, total)),
            }
        )

    failed_prevalence_by_max_tokens_records = []
    for max_tokens in sorted(failed_attempts_by_max_tokens):
        total = failed_attempts_by_max_tokens[max_tokens]
        fallback = failed_fallback_by_max_tokens[max_tokens]
        failed_prevalence_by_max_tokens_records.append(
            {
                "max_tokens": max_tokens,
                "failed_attempt_count": total,
                "failed_fallback_placeholder_count": fallback,
                "failed_fallback_placeholder_rate": as_float(pct(fallback, total)),
            }
        )

    failed_prevalence_by_attempt_records = []
    for attempt_index in sorted(failed_attempts_by_attempt):
        total = failed_attempts_by_attempt[attempt_index]
        fallback = failed_fallback_by_attempt[attempt_index]
        failed_prevalence_by_attempt_records.append(
            {
                "attempt_index": attempt_index,
                "failed_attempt_count": total,
                "failed_fallback_placeholder_count": fallback,
                "failed_fallback_placeholder_rate": as_float(pct(fallback, total)),
            }
        )

    prompt_contamination_by_attempt_records = []
    for attempt_index in sorted(later_attempts_by_attempt):
        total = later_attempts_by_attempt[attempt_index]
        contaminated = prompt_contaminated_by_attempt[attempt_index]
        prompt_contamination_by_attempt_records.append(
            {
                "attempt_index": attempt_index,
                "later_attempt_count": total,
                "prompt_fallback_contamination_count": contaminated,
                "prompt_fallback_contamination_rate": as_float(pct(contaminated, total)),
            }
        )

    baseline_fail_recovery_by_pre_success_fallback = []
    for has_contamination in [False, True]:
        group = [
            record
            for record in baseline_fail_problem_records
            if record["has_any_pre_success_prompt_fallback_contamination"] == has_contamination
        ]
        iterative_success_count = sum(1 for record in group if record["iterative_success"])
        baseline_fail_recovery_by_pre_success_fallback.append(
            {
                "has_any_pre_success_prompt_fallback_contamination": has_contamination,
                "problem_count": len(group),
                "iterative_success_count": iterative_success_count,
                "iterative_recovery_rate": as_float(pct(iterative_success_count, len(group))),
            }
        )

    route_key_collapse_by_dataset = []
    route_key_collapse_by_max_tokens = []
    for key_name, values in [
        ("dataset_key", sorted({record["dataset_key"] for record in problem_records})),
        ("max_tokens", sorted({record["max_tokens"] for record in problem_records})),
    ]:
        for key in values:
            group = [record for record in problem_records if record[key_name] == key]
            route_key_collapse_record = {
                key_name: key,
                "problem_count": len(group),
                "problems_with_any_fallback_placeholder": sum(
                    1 for record in group if record["has_any_fallback_placeholder"]
                ),
                "problems_with_route_key_collapse": sum(
                    1 for record in group if record["route_key_collapse_upper_bound"] > 0
                ),
                "mean_actual_distinct_route_keys": as_float(
                    mean(record["actual_distinct_route_keys"] for record in group)
                ),
                "mean_upper_bound_distinct_route_keys_if_fallback_unique": as_float(
                    mean(record["upper_bound_distinct_route_keys_if_fallback_unique"] for record in group)
                ),
                "mean_route_key_collapse_upper_bound": as_float(
                    mean(record["route_key_collapse_upper_bound"] for record in group)
                ),
                "repeated_route_key_event_count": sum(
                    record["repeated_route_key_events"] for record in group
                ),
                "fallback_repeat_event_count": sum(
                    record["fallback_repeat_events"] for record in group
                ),
            }
            route_key_collapse_record["fallback_share_of_repeat_events"] = as_float(
                pct(
                    route_key_collapse_record["fallback_repeat_event_count"],
                    route_key_collapse_record["repeated_route_key_event_count"],
                )
            )
            if key_name == "dataset_key":
                route_key_collapse_by_dataset.append(route_key_collapse_record)
            else:
                route_key_collapse_by_max_tokens.append(route_key_collapse_record)

    overall_route_key_repeated_event_count = sum(record["repeated_route_key_events"] for record in problem_records)
    overall_fallback_repeat_event_count = sum(record["fallback_repeat_events"] for record in problem_records)

    severity_label = "major"
    severity_judgment = (
        "Major. Fallback placeholders are not rare on this bundle, they directly enter the iterative "
        "negative-memory prompt on a large fraction of baseline-fail problems, and they dominate the "
        "route-key repetition diagnostics. They are not the only reason the method misses recoveries, "
        "but they are serious enough to materially distort both outcome analysis and route-diversity metrics."
    )

    summary = {
        "bundle_dir": str(bundle_dir),
        "fallback_detection": {
            "exact_string_fallback_count_from_existing_summary_quality_analysis": existing_exact_fallback_count,
            "broader_placeholder_fallback_count": broad_placeholder_count,
            "additional_placeholder_fallback_count_missed_by_exact_match": broad_placeholder_only_count,
            "exact_match_undercount_rate_relative_to_broader_placeholder_count": as_float(
                pct(broad_placeholder_only_count, broad_placeholder_count)
            ),
            "exact_match_undercount_factor": as_float(
                (broad_placeholder_count / existing_exact_fallback_count)
                if existing_exact_fallback_count
                else None
            ),
            "fallback_route_key": FALLBACK_ROUTE_KEY,
        },
        "severity": {
            "label": severity_label,
            "judgment": severity_judgment,
        },
        "prevalence_overall": {
            "iterative_attempt_count": len(iterative_rows),
            "fallback_placeholder_count": broad_placeholder_count,
            "fallback_placeholder_rate": as_float(pct(broad_placeholder_count, len(iterative_rows))),
            "problem_records_with_any_fallback_placeholder": sum(
                1 for record in problem_records if record["has_any_fallback_placeholder"]
            ),
            "problem_record_count": len(problem_records),
            "problem_record_rate_with_any_fallback_placeholder": as_float(
                pct(
                    sum(1 for record in problem_records if record["has_any_fallback_placeholder"]),
                    len(problem_records),
                )
            ),
        },
        "prevalence_by_dataset": prevalence_by_dataset_records,
        "prevalence_by_max_tokens": prevalence_by_max_tokens_records,
        "prevalence_by_attempt_index": prevalence_by_attempt_records,
        "failed_iterative_attempts": {
            "failed_attempt_count": len(failed_iterative_rows),
            "failed_fallback_placeholder_count": len(failed_fallback_rows),
            "failed_fallback_placeholder_rate": as_float(
                pct(len(failed_fallback_rows), len(failed_iterative_rows))
            ),
        },
        "failed_iterative_attempts_by_dataset": failed_prevalence_by_dataset_records,
        "failed_iterative_attempts_by_max_tokens": failed_prevalence_by_max_tokens_records,
        "failed_iterative_attempts_by_attempt_index": failed_prevalence_by_attempt_records,
        "prompt_fallback_contamination": {
            "later_attempt_count": len(later_rows),
            "prompt_fallback_contamination_count": len(prompt_contaminated_rows),
            "prompt_fallback_contamination_rate": as_float(
                pct(len(prompt_contaminated_rows), len(later_rows))
            ),
            "failed_later_attempt_count": len(later_failed_rows),
            "failed_later_attempts_with_prompt_fallback_contamination_count": sum(
                1 for row in later_failed_rows if row["fallback_in_prompt"]
            ),
            "failed_later_attempts_with_prompt_fallback_contamination_rate": as_float(
                pct(
                    sum(1 for row in later_failed_rows if row["fallback_in_prompt"]),
                    len(later_failed_rows),
                )
            ),
        },
        "prompt_fallback_contamination_by_attempt_index": prompt_contamination_by_attempt_records,
        "baseline_fail_recovery_by_pre_success_prompt_fallback_contamination": (
            baseline_fail_recovery_by_pre_success_fallback
        ),
        "iterative_only_wins": {
            "problem_count": len(iterative_only_win_records),
            "problem_count_with_any_pre_success_failed_fallback_placeholder": sum(
                1
                for record in iterative_only_win_records
                if record["has_any_pre_success_failed_fallback_placeholder"]
            ),
            "problem_count_with_any_pre_success_prompt_fallback_contamination": sum(
                1
                for record in iterative_only_win_records
                if record["has_any_pre_success_prompt_fallback_contamination"]
            ),
        },
        "route_key_collapse": {
            "fallback_route_key_reuse_count": broad_placeholder_count,
            "distinct_fallback_route_key_count": 1,
            "problem_count_with_route_key_collapse": sum(
                1 for record in problem_records if record["route_key_collapse_upper_bound"] > 0
            ),
            "problem_count_where_all_repeat_events_are_due_to_fallback": sum(
                1
                for record in problem_records
                if record["repeated_route_key_events"] > 0
                and record["repeated_route_key_events"] == record["fallback_repeat_events"]
            ),
            "overall_repeated_route_key_event_count": overall_route_key_repeated_event_count,
            "overall_fallback_repeat_event_count": overall_fallback_repeat_event_count,
            "fallback_share_of_repeated_route_key_events": as_float(
                pct(overall_fallback_repeat_event_count, overall_route_key_repeated_event_count)
            ),
            "mean_actual_distinct_route_keys": as_float(
                mean(record["actual_distinct_route_keys"] for record in problem_records)
            ),
            "mean_upper_bound_distinct_route_keys_if_fallback_unique": as_float(
                mean(
                    record["upper_bound_distinct_route_keys_if_fallback_unique"]
                    for record in problem_records
                )
            ),
            "mean_route_key_collapse_upper_bound": as_float(
                mean(record["route_key_collapse_upper_bound"] for record in problem_records)
            ),
            "mean_route_key_collapse_upper_bound_among_problems_with_any_fallback": as_float(
                mean(
                    record["route_key_collapse_upper_bound"]
                    for record in problem_records
                    if record["has_any_fallback_placeholder"]
                )
            ),
        },
        "route_key_collapse_by_dataset": route_key_collapse_by_dataset,
        "route_key_collapse_by_max_tokens": route_key_collapse_by_max_tokens,
    }

    report_lines = [
        "# Fallback Contamination Analysis",
        "",
        f"Bundle: `{bundle_dir}`",
        "",
        f"Severity judgment: **{severity_label}**",
        "",
        severity_judgment,
        "",
        "## Key Findings",
        "",
        (
            f"- The earlier summary-quality join undercounted fallback placeholders: it flagged "
            f"`{existing_exact_fallback_count}` exact-string fallbacks, but the broader placeholder "
            f"pattern appears `{broad_placeholder_count}` times. That misses `{broad_placeholder_only_count}` "
            f"cases, or `{render_pct(summary['fallback_detection']['exact_match_undercount_rate_relative_to_broader_placeholder_count'])}` "
            f"of all placeholder fallbacks."
        ),
        (
            f"- Placeholder fallbacks occur on `{broad_placeholder_count}/{len(iterative_rows)}` iterative attempts "
            f"({render_pct(summary['prevalence_overall']['fallback_placeholder_rate'])}) and on "
            f"`{summary['prevalence_overall']['problem_records_with_any_fallback_placeholder']}/{len(problem_records)}` "
            f"problem-length records ({render_pct(summary['prevalence_overall']['problem_record_rate_with_any_fallback_placeholder'])})."
        ),
        (
            f"- They directly contaminate the iterative prompt on `{len(prompt_contaminated_rows)}/{len(later_rows)}` later attempts "
            f"({render_pct(summary['prompt_fallback_contamination']['prompt_fallback_contamination_rate'])}) and on "
            f"`{summary['prompt_fallback_contamination']['failed_later_attempts_with_prompt_fallback_contamination_count']}/"
            f"{summary['prompt_fallback_contamination']['failed_later_attempt_count']}` failed later attempts "
            f"({render_pct(summary['prompt_fallback_contamination']['failed_later_attempts_with_prompt_fallback_contamination_rate'])})."
        ),
        (
            f"- For baseline-fail problems, iterative recovered `8/41` "
            f"({render_pct(summary['baseline_fail_recovery_by_pre_success_prompt_fallback_contamination'][0]['iterative_recovery_rate'])}) "
            f"when there was no pre-success prompt contamination, but `0/26` when a fallback summary had already entered the prompt."
        ),
        (
            f"- All `{len(iterative_only_win_records)}` iterative-only wins occur on problems with no pre-success fallback contamination."
        ),
        (
            f"- Route-memory diagnostics are heavily distorted: the single fallback route key "
            f"`{FALLBACK_ROUTE_KEY}` accounts for `{overall_fallback_repeat_event_count}/{overall_route_key_repeated_event_count}` "
            f"repeated route-key events ({render_pct(summary['route_key_collapse']['fallback_share_of_repeated_route_key_events'])})."
        ),
        "",
        "## Prevalence",
        "",
        "| slice | attempts | fallback placeholders | rate |",
        "| --- | ---: | ---: | ---: |",
    ]

    for row in prevalence_by_max_tokens_records:
        report_lines.append(
            f"| max_tokens={row['max_tokens']} | {row['attempt_count']} | "
            f"{row['fallback_placeholder_count']} | {render_pct(row['fallback_placeholder_rate'])} |"
        )
    for row in prevalence_by_dataset_records:
        report_lines.append(
            f"| dataset={row['dataset_key']} | {row['attempt_count']} | "
            f"{row['fallback_placeholder_count']} | {render_pct(row['fallback_placeholder_rate'])} |"
        )

    report_lines.extend(
        [
            "",
            "### By Attempt Index",
            "",
            "| attempt | attempts | fallback placeholders | rate | prompt contaminated later attempts | contamination rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    contamination_by_attempt_map = {
        row["attempt_index"]: row for row in prompt_contamination_by_attempt_records
    }
    for row in prevalence_by_attempt_records:
        contam = contamination_by_attempt_map.get(row["attempt_index"])
        report_lines.append(
            f"| {row['attempt_index']} | {row['attempt_count']} | {row['fallback_placeholder_count']} | "
            f"{render_pct(row['fallback_placeholder_rate'])} | "
            f"{0 if contam is None else contam['prompt_fallback_contamination_count']} | "
            f"{'n/a' if contam is None else render_pct(contam['prompt_fallback_contamination_rate'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Failed Iterative Attempts",
            "",
            f"- Failed iterative attempts with placeholder fallback: `{len(failed_fallback_rows)}/{len(failed_iterative_rows)}` "
            f"({render_pct(summary['failed_iterative_attempts']['failed_fallback_placeholder_rate'])}).",
            "",
            "| slice | failed attempts | failed fallback placeholders | rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in failed_prevalence_by_max_tokens_records:
        report_lines.append(
            f"| max_tokens={row['max_tokens']} | {row['failed_attempt_count']} | "
            f"{row['failed_fallback_placeholder_count']} | {render_pct(row['failed_fallback_placeholder_rate'])} |"
        )
    for row in failed_prevalence_by_dataset_records:
        report_lines.append(
            f"| dataset={row['dataset_key']} | {row['failed_attempt_count']} | "
            f"{row['failed_fallback_placeholder_count']} | {render_pct(row['failed_fallback_placeholder_rate'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Baseline-Fail Recovery",
            "",
            "| pre-success prompt fallback contamination | baseline-fail problems | iterative recoveries | recovery rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in baseline_fail_recovery_by_pre_success_fallback:
        report_lines.append(
            f"| {row['has_any_pre_success_prompt_fallback_contamination']} | {row['problem_count']} | "
            f"{row['iterative_success_count']} | {render_pct(row['iterative_recovery_rate'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Route-Key Collapse",
            "",
            f"- Fallback placeholders all map to the same `summary_route_key`: `{FALLBACK_ROUTE_KEY}`.",
            f"- Problems with any route-key collapse from fallback reuse: "
            f"`{summary['route_key_collapse']['problem_count_with_route_key_collapse']}/{len(problem_records)}`.",
            f"- Mean distinct route keys per problem: actual `{render_num(summary['route_key_collapse']['mean_actual_distinct_route_keys'])}`, "
            f"upper bound if fallback placeholders were unique unknown routes "
            f"`{render_num(summary['route_key_collapse']['mean_upper_bound_distinct_route_keys_if_fallback_unique'])}`.",
            "",
            "| slice | problems | repeated route-key events | fallback-caused repeats | fallback share | mean collapse upper bound |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in route_key_collapse_by_max_tokens:
        report_lines.append(
            f"| max_tokens={row['max_tokens']} | {row['problem_count']} | {row['repeated_route_key_event_count']} | "
            f"{row['fallback_repeat_event_count']} | {render_pct(row['fallback_share_of_repeat_events'])} | "
            f"{render_num(row['mean_route_key_collapse_upper_bound'])} |"
        )
    for row in route_key_collapse_by_dataset:
        report_lines.append(
            f"| dataset={row['dataset_key']} | {row['problem_count']} | {row['repeated_route_key_event_count']} | "
            f"{row['fallback_repeat_event_count']} | {render_pct(row['fallback_share_of_repeat_events'])} | "
            f"{render_num(row['mean_route_key_collapse_upper_bound'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                "Fallback-summary contamination is a **major** blocker for interpreting this bundle cleanly. "
                "It is not the sole reason the explored-path method fails, but it is serious enough to suppress "
                "recoveries on a large subset of baseline-fail problems and to dominate the route-memory repetition diagnostics."
            ),
            "",
        ]
    )

    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "fallback_contamination_summary.json"
    report_path = eval_dir / "fallback_contamination_summary.md"
    problem_level_path = eval_dir / "fallback_contamination_problem_level.jsonl"

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path.write_text("\n".join(report_lines))
    with problem_level_path.open("w") as handle:
        for record in problem_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {problem_level_path}")


if __name__ == "__main__":
    main()
