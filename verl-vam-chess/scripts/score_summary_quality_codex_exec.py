#!/usr/bin/env python3
"""Score explored-path summaries with local `codex exec`.

This script builds a machine-readable dataset from merged iterative traces,
judges each available summary against the original problem plus the full raw
attempt it summarizes, and aggregates the resulting 1-5 ratings.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any

from datasets import load_dataset


DATASET_SPECS: dict[str, dict[str, str]] = {
    "aime24": {
        "hf_name": "math-ai/aime24",
        "split": "test",
        "problem_field": "problem",
        "solution_field": "solution",
    },
    "aime25": {
        "hf_name": "math-ai/aime25",
        "split": "test",
        "problem_field": "problem",
        "solution_field": "answer",
    },
    "amc23": {
        "hf_name": "math-ai/amc23",
        "split": "test",
        "problem_field": "question",
        "solution_field": "answer",
    },
}

PROMPT_TEMPLATE = """You are evaluating whether a short summary faithfully captures a model's attempted solution to a math problem.

Problem:
{problem}

Full attempted solution:
{full_attempted_solution}

Candidate summary:
{summary}

Rate the summary on a scale from 1 to 5.

Scoring rubric:
1 = Bad: incorrect, misleading, or unrelated to the attempted solution
2 = Weak: partially relevant, but misses or distorts the main route
3 = Okay: captures some important parts, but is incomplete or vague
4 = Good: mostly accurate and useful, with only minor omissions or issues
5 = Excellent: accurate, faithful to the attempted solution, and captures the key route clearly

Judge the summary based on:
- faithfulness to the attempted solution
- coverage of the main route and important commitments
- clarity and usefulness as a compact negative-memory summary
- absence of fabricated or contradictory information

Do not judge whether the attempted solution is mathematically correct. Judge only whether the summary accurately captures the attempted solution shown above.

Return only a single integer: 1, 2, 3, 4, or 5.
"""

RATING_RE = re.compile(r"\b([1-5])\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("analysis/math_explored_paths/cluster_job_3877433/raw_iterative_traces"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/math_explored_paths/cluster_job_3877433/summary_quality_eval"),
    )
    parser.add_argument(
        "--codex-cwd",
        type=Path,
        default=Path("/usr0/home/zhichen3/codex_exec"),
    )
    parser.add_argument(
        "--codex-workspace",
        type=Path,
        default=Path("/usr0/home/zhichen3/chess-rl"),
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sleep-between-batches", type=float, default=15.0)
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def normalize_problem_id(value: Any) -> str:
    return str(value)


def load_problem_lookup(dataset_key: str) -> dict[str, dict[str, Any]]:
    spec = DATASET_SPECS[dataset_key]
    ds = load_dataset(spec["hf_name"], split=spec["split"])
    lookup: dict[str, dict[str, Any]] = {}
    for row in ds:
        lookup[normalize_problem_id(row["id"])] = dict(row)
    return lookup


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_trace_rows(trace_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(trace_dir.rglob("iterative_traces.jsonl")):
        with path.open() as f:
            for line in f:
                rows.append((path, json.loads(line)))
    return rows


def build_requests(trace_dir: Path) -> list[dict[str, Any]]:
    trace_rows = iter_trace_rows(trace_dir)
    dataset_keys = sorted({row["dataset_key"] for _, row in trace_rows})
    problem_lookups = {key: load_problem_lookup(key) for key in dataset_keys}
    requests: list[dict[str, Any]] = []

    for source_path, row in trace_rows:
        summary = row.get("summary")
        if summary is None or not str(summary).strip():
            continue

        dataset_key = str(row["dataset_key"])
        spec = DATASET_SPECS[dataset_key]
        problem_id = normalize_problem_id(row["problem_id"])
        dataset_row = problem_lookups[dataset_key].get(problem_id)
        if dataset_row is None:
            raise KeyError(f"Missing dataset row for {dataset_key} problem_id={problem_id}")

        problem_text = str(dataset_row[spec["problem_field"]])
        reference_solution = dataset_row.get(spec["solution_field"])
        full_attempted_solution = str(row["raw_response"])
        summary_str = str(summary).strip()
        uid = (
            f"{dataset_key}_len_{row['max_tokens']}_problem_{int(row['problem_index']):03d}"
            f"_attempt_{int(row['attempt_index']):02d}"
        )
        requests.append(
            {
                "uid": uid,
                "dataset_key": dataset_key,
                "max_tokens": int(row["max_tokens"]),
                "problem_index": int(row["problem_index"]),
                "problem_id": row["problem_id"],
                "attempt_index": int(row["attempt_index"]),
                "correct": bool(row["correct"]),
                "score": float(row["score"]),
                "ground_truth": row.get("ground_truth"),
                "problem_url": row.get("problem_url"),
                "prompt_variant": row.get("prompt_variant"),
                "request_seed": row.get("request_seed"),
                "summary": summary_str,
                "summary_route_key": row.get("summary_route_key"),
                "shown_explored_paths_before": row.get("accumulated_explored_paths_before", []),
                "problem_text": problem_text,
                "reference_solution": reference_solution,
                "reference_solution_source": spec["solution_field"],
                "full_attempted_solution": full_attempted_solution,
                "source_trace_file": str(source_path),
                "raw_response_char_len": len(full_attempted_solution),
                "raw_response_sha256": hashlib.sha256(full_attempted_solution.encode("utf-8")).hexdigest(),
            }
        )

    requests.sort(
        key=lambda row: (
            row["dataset_key"],
            row["max_tokens"],
            row["problem_index"],
            row["attempt_index"],
        )
    )
    return requests


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_jsonl_map(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    items: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            item_key = str(row[key])
            current = items.get(item_key)
            current_ok = current is not None and current.get("rating") in {1, 2, 3, 4, 5}
            row_ok = row.get("rating") in {1, 2, 3, 4, 5}
            if current_ok and not row_ok:
                continue
            items[item_key] = row
    return items


def render_prompt(row: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        problem=row["problem_text"],
        full_attempted_solution=row["full_attempted_solution"],
        summary=row["summary"],
    )


def parse_rating(text: str) -> int | None:
    match = RATING_RE.search(text.strip())
    if not match:
        return None
    return int(match.group(1))


def parse_rating_from_stdout(stdout_text: str) -> int | None:
    for line in stdout_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in {"1", "2", "3", "4", "5"}:
            return int(stripped)
        break
    return None


def score_one(
    row: dict[str, Any],
    args: argparse.Namespace,
    responses_dir: Path,
    failures_dir: Path,
) -> dict[str, Any]:
    prompt = render_prompt(row)
    response_path = (responses_dir / f"{row['uid']}.txt").resolve()
    if response_path.exists():
        response_path.unlink()

    cmd = [
        args.codex_bin,
        "exec",
        "--skip-git-repo-check",
        "--output-last-message",
        str(response_path),
        "-",
    ]

    last_error: dict[str, Any] | None = None
    for attempt_num in range(1, args.max_retries + 2):
        started_at = time.time()
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            cwd=args.codex_cwd,
            timeout=args.timeout_sec,
        )
        duration_sec = round(time.time() - started_at, 3)
        output_text = response_path.read_text().strip() if response_path.exists() else ""
        rating = parse_rating(output_text)
        if rating is None and proc.stdout:
            rating = parse_rating_from_stdout(proc.stdout)
            if rating is not None and not output_text:
                output_text = str(rating)
        if proc.returncode == 0 and rating is not None:
            return {
                "uid": row["uid"],
                "status": "ok",
                "rating": rating,
                "attempts_used": attempt_num,
                "duration_sec": duration_sec,
                "judge_output": output_text,
                "codex_exit_code": proc.returncode,
            }

        stdout_tail = proc.stdout[-4000:] if proc.stdout else ""
        stderr_tail = proc.stderr[-4000:] if proc.stderr else ""
        last_error = {
            "uid": row["uid"],
            "status": "error",
            "rating": None,
            "attempts_used": attempt_num,
            "duration_sec": duration_sec,
            "judge_output": output_text,
            "codex_exit_code": proc.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        failure_path = failures_dir / f"{row['uid']}_attempt_{attempt_num}.json"
        failure_path.write_text(json.dumps(last_error, indent=2, ensure_ascii=True) + "\n")
        if attempt_num < args.max_retries + 1:
            time.sleep(5)

    assert last_error is not None
    return last_error


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("rating") in {1, 2, 3, 4, 5}]
    ratings = [int(row["rating"]) for row in successful]
    histogram = {str(score): 0 for score in range(1, 6)}
    for score in ratings:
        histogram[str(score)] += 1

    def breakdown(group_key: str) -> dict[str, Any]:
        grouped: dict[str, list[int]] = {}
        for row in successful:
            key = str(row[group_key])
            grouped.setdefault(key, []).append(int(row["rating"]))
        return {
            key: summarize_ratings(value)
            for key, value in sorted(grouped.items(), key=lambda item: item[0])
        }

    stats = {
        "count_total": len(rows),
        "count_successful": len(successful),
        "count_failed": len(rows) - len(successful),
        "mean_rating": round(sum(ratings) / len(ratings), 6) if ratings else None,
        "median_rating": statistics.median(ratings) if ratings else None,
        "histogram": histogram,
        "by_dataset": breakdown("dataset_key"),
        "by_max_tokens": breakdown("max_tokens"),
        "by_correct": breakdown("correct"),
    }
    return stats


def summarize_ratings(ratings: list[int]) -> dict[str, Any]:
    histogram = {str(score): 0 for score in range(1, 6)}
    for score in ratings:
        histogram[str(score)] += 1
    return {
        "count": len(ratings),
        "mean": round(sum(ratings) / len(ratings), 6) if ratings else None,
        "median": statistics.median(ratings) if ratings else None,
        "histogram": histogram,
    }


def render_stats_markdown(stats: dict[str, Any]) -> str:
    lines = [
        "# Summary Quality Ratings",
        "",
        f"- Total rows: {stats['count_total']}",
        f"- Successful ratings: {stats['count_successful']}",
        f"- Failed ratings: {stats['count_failed']}",
        f"- Mean rating: {stats['mean_rating']}",
        f"- Median rating: {stats['median_rating']}",
        f"- Histogram 1-5: {stats['histogram']}",
        "",
        "## By Dataset",
        "",
    ]
    lines.extend(render_breakdown_lines(stats["by_dataset"]))
    lines.extend(["", "## By Max Tokens", ""])
    lines.extend(render_breakdown_lines(stats["by_max_tokens"]))
    lines.extend(["", "## By Correctness", ""])
    lines.extend(render_breakdown_lines(stats["by_correct"]))
    return "\n".join(lines) + "\n"


def render_breakdown_lines(breakdown: dict[str, Any]) -> list[str]:
    lines = [
        "| group | count | mean | median | histogram |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key, value in breakdown.items():
        mean_value = "n/a" if value["mean"] is None else value["mean"]
        median_value = "n/a" if value["median"] is None else value["median"]
        lines.append(
            f"| {key} | {value['count']} | {mean_value} | {median_value} | {value['histogram']} |"
        )
    return lines


def write_commands_manifest(path: Path, args: argparse.Namespace) -> None:
    script_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    codex_template = " ".join(
        shlex.quote(part)
        for part in [
            args.codex_bin,
            "exec",
            "--skip-git-repo-check",
            "--output-last-message",
            "<response_path>",
            "-",
        ]
    )
    lines = [
        "# Commands used for summary-quality scoring",
        "",
        f"# Script invocation",
        script_command,
        "",
        "# Per-example codex exec command template",
        f"(cd {shlex.quote(str(args.codex_cwd))} && {codex_template} < <prompt_file>)",
        "",
        f"# Workers: {args.workers}",
        f"# Sleep between batches: {args.sleep_between_batches}",
    ]
    path.write_text("\n".join(lines) + "\n")


def merge_requests_and_results(
    requests: list[dict[str, Any]], result_map: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in requests:
        result = result_map.get(row["uid"], {})
        merged.append({**row, **result})
    return merged


def main() -> None:
    args = parse_args()
    args.trace_dir = args.trace_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.codex_cwd = args.codex_cwd.resolve()
    args.codex_workspace = args.codex_workspace.resolve()

    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "responses")
    ensure_dir(args.output_dir / "failures")

    requests_path = args.output_dir / "summary_quality_requests.jsonl"
    results_path = args.output_dir / "summary_quality_results.jsonl"
    scored_path = args.output_dir / "summary_quality_scored_examples.jsonl"
    stats_json_path = args.output_dir / "summary_quality_stats.json"
    stats_md_path = args.output_dir / "summary_quality_stats.md"
    prompt_template_path = args.output_dir / "judge_prompt_template.txt"
    commands_path = args.output_dir / "commands_used.sh"

    requests = build_requests(args.trace_dir)
    if args.limit is not None:
        requests = requests[: args.limit]
    write_jsonl(requests_path, requests)
    prompt_template_path.write_text(PROMPT_TEMPLATE)
    write_commands_manifest(commands_path, args)

    result_map = load_jsonl_map(results_path, "uid")
    pending = [row for row in requests if result_map.get(row["uid"], {}).get("rating") not in {1, 2, 3, 4, 5}]
    print(
        f"[summary-quality] total={len(requests)} already_scored={len(requests) - len(pending)} pending={len(pending)}",
        flush=True,
    )

    if pending:
        with results_path.open("a") as results_file:
            for batch_start in range(0, len(pending), args.workers):
                batch = pending[batch_start : batch_start + args.workers]
                batch_num = batch_start // args.workers + 1
                batch_total = (len(pending) + args.workers - 1) // args.workers
                print(
                    f"[summary-quality] batch {batch_num}/{batch_total} size={len(batch)}",
                    flush=True,
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = [
                        executor.submit(
                            score_one,
                            row,
                            args,
                            args.output_dir / "responses",
                            args.output_dir / "failures",
                        )
                        for row in batch
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        results_file.write(json.dumps(result, ensure_ascii=True) + "\n")
                        results_file.flush()
                        result_map[result["uid"]] = result
                if batch_start + args.workers < len(pending):
                    print(
                        f"[summary-quality] sleeping {args.sleep_between_batches}s before next batch",
                        flush=True,
                    )
                    time.sleep(args.sleep_between_batches)

    scored = merge_requests_and_results(requests, result_map)
    write_jsonl(scored_path, scored)
    stats = aggregate(scored)
    stats_json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")
    stats_md_path.write_text(render_stats_markdown(stats))


if __name__ == "__main__":
    main()
