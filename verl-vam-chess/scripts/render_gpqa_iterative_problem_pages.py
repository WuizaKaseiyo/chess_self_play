#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one markdown page per GPQA Diamond problem from iterative explored-path traces."
    )
    parser.add_argument("--problems-path", required=True, help="Path to problems.jsonl.")
    parser.add_argument("--traces-path", required=True, help="Path to iterative_traces.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered markdown files.")
    parser.add_argument(
        "--ssh-host",
        default="",
        help="Optional SSH host. If set, both input paths are read remotely via `ssh <host> cat <path>`.",
    )
    return parser.parse_args()


def read_text(path: str, ssh_host: str) -> str:
    if ssh_host:
        result = subprocess.run(
            ["ssh", ssh_host, "cat", path],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    return Path(path).read_text(encoding="utf-8")


def read_jsonl(path: str, ssh_host: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in read_text(path, ssh_host).splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def choose_fence(text: str) -> str:
    runs = re.findall(r"`+", text)
    longest = max((len(run) for run in runs), default=0)
    return "`" * max(3, longest + 1)


def code_block(text: str, info: str = "text") -> str:
    fence = choose_fence(text)
    return f"{fence}{info}\n{text}\n{fence}"


def sanitize_problem_id(problem_id: Any) -> str:
    if isinstance(problem_id, int):
        return f"{problem_id}.md"
    if isinstance(problem_id, str) and problem_id.isdigit():
        return f"{problem_id}.md"
    raw = str(problem_id).strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    if not safe:
        safe = "unknown_problem"
    return f"{safe}.md"


def format_summaries(items: list[str]) -> str:
    if not items:
        return "None."
    lines = []
    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx}. {item}")
    return "\n".join(lines)


def render_attempt(trace: dict[str, Any]) -> str:
    parts = [
        f"## Attempt {trace['attempt_index']}",
        "",
        "### Explored-Path Summaries Before Attempt",
        format_summaries(trace.get("accumulated_explored_paths_before", [])),
        "",
        "### Solver Prompt",
        code_block(trace.get("solver_prompt", ""), info="text"),
        "",
        "### Model Solution",
        code_block(trace.get("raw_response", ""), info="text"),
        "",
        "### Attempt Outcome",
        f"- Extracted answer raw: `{format_scalar(trace.get('extracted_answer_raw'))}`",
        f"- Extracted answer normalized: `{format_scalar(trace.get('extracted_answer_normalized'))}`",
        f"- Score: `{format_scalar(trace.get('score'))}`",
        f"- Correct: `{format_scalar(trace.get('correct'))}`",
        "",
        "### Generated Summary",
        code_block(trace.get("summary", ""), info="text"),
    ]
    raw_summary_output = trace.get("raw_summary_output") or ""
    if raw_summary_output:
        info = "json" if raw_summary_output.lstrip().startswith("{") else "text"
        parts.extend(
            [
                "",
                "### Raw Summary Output",
                code_block(raw_summary_output, info=info),
            ]
        )
    summary_failure_reason = trace.get("summary_failure_reason")
    if summary_failure_reason is not None:
        parts.extend(
            [
                "",
                f"### Summary Failure Reason\n`{summary_failure_reason}`",
            ]
        )
    return "\n".join(parts)


def render_problem(problem: dict[str, Any], traces: list[dict[str, Any]]) -> str:
    traces = sorted(traces, key=lambda row: int(row["attempt_index"]))
    parts = [
        f"# GPQA Diamond Problem {problem['problem_id']}",
        "",
        f"- Problem id: `{problem['problem_id']}`",
        f"- Problem index: `{problem['problem_index']}`",
        f"- Ground truth: `{problem['ground_truth']}`",
        f"- Iterative attempts in run: `{len(traces)}`",
        "",
        "## Question",
        code_block(problem.get("question_raw", ""), info="text"),
        "",
        "## Original Prompt Context",
        code_block(problem.get("baseline_prompt", ""), info="text"),
        "",
    ]
    if not traces:
        parts.extend(["## Attempts", "No iterative attempts were found for this problem."])
        return "\n".join(parts).strip() + "\n"
    for trace in traces:
        parts.extend([render_attempt(trace), ""])
    return "\n".join(parts).strip() + "\n"


def main() -> None:
    args = parse_args()
    problems = read_jsonl(args.problems_path, args.ssh_host)
    traces = read_jsonl(args.traces_path, args.ssh_host)

    gpqa_problems = [row for row in problems if row.get("dataset_key") == "gpqa_diamond"]
    gpqa_traces = [
        row
        for row in traces
        if row.get("dataset_key") == "gpqa_diamond" and row.get("method") == "iterative"
    ]

    traces_by_problem: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in gpqa_traces:
        traces_by_problem[row["problem_id"]].append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for problem in sorted(gpqa_problems, key=lambda row: int(row["problem_index"])):
        path = output_dir / sanitize_problem_id(problem["problem_id"])
        path.write_text(
            render_problem(problem, traces_by_problem.get(problem["problem_id"], [])),
            encoding="utf-8",
        )
        rendered += 1

    print(json.dumps({"output_dir": str(output_dir), "rendered_problem_pages": rendered}, indent=2))


if __name__ == "__main__":
    main()
