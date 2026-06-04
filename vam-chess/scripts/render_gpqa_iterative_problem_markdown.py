#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Render one markdown file per GPQA Diamond problem from iterative explored-path traces."
    )
    ap.add_argument("--iterative-traces", type=Path, required=True, help="Path to iterative_traces.jsonl.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory for rendered markdown files.")
    ap.add_argument(
        "--source-run-dir",
        default="",
        help="Optional source run directory shown in the rendered metadata.",
    )
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def traces_by_problem(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["problem_id"])].append(row)
    for problem_rows in grouped.values():
        problem_rows.sort(key=lambda row: int(row["attempt_index"]))
    return dict(sorted(grouped.items(), key=lambda item: int(item[1][0]["problem_index"])))


def split_prompt(prompt_text: str) -> tuple[str, str]:
    if "\n\n" not in prompt_text:
        return prompt_text.strip(), ""
    question_block, instruction_block = prompt_text.rsplit("\n\n", 1)
    return question_block.strip(), instruction_block.strip()


def parse_options(question_block: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for line in question_block.splitlines():
        match = re.match(r"^([ABCD])\.\s+(.*)$", line.strip())
        if match:
            options[match.group(1)] = match.group(2).strip()
    return options


def fenced_block(text: str, info: str = "") -> str:
    body = text.rstrip("\n")
    suffix = info if info else ""
    if suffix:
        return f"~~~~{suffix}\n{body}\n~~~~"
    return f"~~~~\n{body}\n~~~~"


def details_block(summary: str, body: str) -> str:
    return f"<details>\n<summary>{summary}</summary>\n\n{body}\n\n</details>"


def outcome_label(row: dict[str, Any]) -> str:
    return "correct" if row.get("correct") else "incorrect"


def render_prior_summaries(summaries: list[str]) -> str:
    if not summaries:
        return "None."
    lines = []
    for idx, summary in enumerate(summaries, start=1):
        lines.append(f"{idx}. {summary}")
    return "\n".join(lines)


def first_correct_attempt(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        if row.get("correct"):
            return int(row["attempt_index"])
    return None


def render_attempt(row: dict[str, Any]) -> str:
    extracted_answer = row.get("extracted_answer_raw") or "none"
    summary_text = str(row.get("summary") or "").strip()
    sections: list[str] = [
        f"## Attempt {row['attempt_index']}",
        f"- Prompt variant: `{row.get('prompt_variant')}`",
        f"- Extracted answer: `{extracted_answer}`",
        f"- Outcome: `{outcome_label(row)}`",
        f"- Score: `{row.get('score')}`",
        "",
        "### Explored-path summaries shown before this attempt",
        render_prior_summaries(list(row.get("accumulated_explored_paths_before") or [])),
        "",
        "### Generated summary",
        summary_text or "(empty)",
        "",
    ]
    failure_reason = row.get("summary_failure_reason")
    if failure_reason:
        sections.extend(
            [
                "### Summary diagnostics",
                f"- Summary failure reason: `{failure_reason}`",
                "",
                details_block(
                    "Raw summary output",
                    fenced_block(str(row.get("raw_summary_output") or ""), info="json"),
                ),
                "",
            ]
        )
    sections.extend(
        [
            details_block(
                "Solution / raw response",
                fenced_block(str(row.get("raw_response") or ""), info="text"),
            ),
            "",
        ]
    )
    return "\n".join(sections).rstrip()


def render_problem_markdown(
    problem_id: str,
    rows: list[dict[str, Any]],
    *,
    source_trace_path: Path,
    source_run_dir: str,
) -> str:
    first_row = rows[0]
    question_block, instruction_block = split_prompt(str(first_row.get("solver_prompt") or ""))
    option_map = parse_options(question_block)
    ground_truth = str(first_row.get("ground_truth") or "")
    gold_text = option_map.get(ground_truth, "(option text unavailable)")
    first_success = first_correct_attempt(rows)

    sections: list[str] = [
        f"# GPQA Diamond Problem {problem_id}",
        "",
        f"- Problem index: `{first_row.get('problem_index')}`",
        f"- Problem id: `{problem_id}`",
        f"- Gold answer: `{ground_truth}`",
        f"- Gold option text: {gold_text}",
        f"- First correct iterative attempt: `{first_success if first_success is not None else 'none'}`",
        f"- Attempts rendered: `{len(rows)}`",
        f"- Source trace file: `{source_trace_path}`",
    ]
    if source_run_dir:
        sections.append(f"- Source run dir: `{source_run_dir}`")
    sections.extend(
        [
            "",
            "## Question",
            fenced_block(question_block, info="text"),
            "",
        ]
    )
    if instruction_block:
        sections.extend(
            [
                "## Answer instruction",
                fenced_block(instruction_block, info="text"),
                "",
            ]
        )
    for row in rows:
        sections.extend([render_attempt(row), ""])
    return "\n".join(sections).strip() + "\n"


def render_index(
    grouped_rows: dict[str, list[dict[str, Any]]],
    *,
    out_dir: Path,
    source_trace_path: Path,
    source_run_dir: str,
) -> str:
    lines = [
        "# GPQA Diamond Iterative Problem Markdown",
        "",
        f"- Problems rendered: `{len(grouped_rows)}`",
        f"- Source trace file: `{source_trace_path}`",
    ]
    if source_run_dir:
        lines.append(f"- Source run dir: `{source_run_dir}`")
    lines.extend(
        [
            "",
            "| problem_id | problem_index | first_correct_attempt | summary_failures | file |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for problem_id, rows in grouped_rows.items():
        first_row = rows[0]
        success = first_correct_attempt(rows)
        failure_count = sum(1 for row in rows if row.get("summary_failure_reason"))
        lines.append(
            "| "
            f"{problem_id} | "
            f"{first_row.get('problem_index')} | "
            f"{success if success is not None else 'none'} | "
            f"{failure_count} | "
            f"[{problem_id}.md]({problem_id}.md) |"
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.iterative_traces)
    if not rows:
        raise SystemExit(f"No rows found in {args.iterative_traces}")
    dataset_keys = {str(row.get("dataset_key") or "") for row in rows}
    method_keys = {str(row.get("method") or "") for row in rows}
    if dataset_keys != {"gpqa_diamond"}:
        raise SystemExit(f"Expected gpqa_diamond rows, found {sorted(dataset_keys)}")
    if method_keys != {"iterative"}:
        raise SystemExit(f"Expected iterative rows, found {sorted(method_keys)}")

    grouped = traces_by_problem(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for problem_id, problem_rows in grouped.items():
        output_path = args.out_dir / f"{problem_id}.md"
        output_path.write_text(
            render_problem_markdown(
                problem_id,
                problem_rows,
                source_trace_path=args.iterative_traces,
                source_run_dir=args.source_run_dir,
            ),
            encoding="utf-8",
        )

    (args.out_dir / "index.md").write_text(
        render_index(
            grouped,
            out_dir=args.out_dir,
            source_trace_path=args.iterative_traces,
            source_run_dir=args.source_run_dir,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
