#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explored_paths_summary_prompt import SUMMARY_PROMPT, SUMMARY_RESPONSE_SCHEMA
from recipe.open_math_reasoning.prepare_eval_dataset import instruction_following
from verl.utils.reward_score.math_dapo import normalize_final_answer
from verl.utils.reward_score.math_reward import (
    compute_score as boxed_compute_score,
    last_boxed_only_string,
    remove_boxed,
    strip_string,
)

GPQA_DIAMOND_INSTRUCTION = (
    "Please reason step by step, and put your final answer (only the choice letter) within \\boxed{}."
)
DEFAULT_DATASET_KEYS = ["aime24", "aime25", "amc23", "math500", "minerva", "olympiadbench"]

OLYMPIADBENCH_SINGLE_ANSWER_SUBSET_REASON = (
    "Excluded rows with is_multiple_answer=true because this harness currently scores a single extracted "
    "final answer per attempt; OlympiadBench multi-answer rows require benchmark-specific structured "
    "comparison to be faithful."
)
SUMMARY_RETRY_LIMIT = 5
SUMMARY_RETRY_SEED_STRIDE = 10_000_000
SUMMARY_RETRY_TEMPERATURE = 0.2


@dataclass(frozen=True)
class ProblemSpec:
    dataset_key: str
    row_index: int
    raw_id: Any
    url: str
    question_raw: str
    baseline_prompt: str
    ground_truth: str
    score_tolerance: Optional[float] = None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    hf_dataset: str
    config: str
    split: str
    question_field: str
    ground_truth_field: str
    id_field: Optional[str]
    url_field: Optional[str] = None
    context_field: Optional[str] = None
    extract_boxed_ground_truth: bool = False
    score_kind: str = "boxed"
    tolerance_field: Optional[str] = None


@dataclass(frozen=True)
class DatasetLoadResult:
    spec: DatasetSpec
    problems: list[ProblemSpec]
    raw_count: int
    filtered_count: int
    subset_metadata: dict[str, Any]


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict[str, str]]
    temperature: float
    top_p: float
    max_tokens: int
    n: int = 1
    seed: Optional[int] = None
    stop: Optional[list[str]] = None
    extra_body: Optional[dict[str, Any]] = None
    guided_decoding: Optional[GuidedDecodingParams] = None


@dataclass
class GenerationContext:
    llm: Any
    tokenizer: Any


@dataclass(frozen=True)
class SummaryAttemptResult:
    summary: str
    raw_summary_output: str


@dataclass(frozen=True)
class SummaryAttemptFailure:
    error: str
    attempts: int


@dataclass(frozen=True)
class SummaryBatchResult:
    results: list[Optional[SummaryAttemptResult]]
    failures: dict[int, SummaryAttemptFailure]


@dataclass(frozen=True)
class MethodRunResult:
    traces: list[dict[str, Any]]
    skipped_problems: list[dict[str, Any]]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "aime24": DatasetSpec(
        key="aime24",
        display_name="AIME24",
        hf_dataset="math-ai/aime24",
        config="default",
        split="test",
        question_field="problem",
        ground_truth_field="solution",
        id_field="id",
        url_field="url",
        extract_boxed_ground_truth=True,
        score_kind="boxed",
    ),
    "aime25": DatasetSpec(
        key="aime25",
        display_name="AIME25",
        hf_dataset="math-ai/aime25",
        config="default",
        split="test",
        question_field="problem",
        ground_truth_field="answer",
        id_field="id",
        score_kind="boxed",
    ),
    "amc23": DatasetSpec(
        key="amc23",
        display_name="AMC23",
        hf_dataset="math-ai/amc23",
        config="default",
        split="test",
        question_field="question",
        ground_truth_field="answer",
        id_field="id",
        url_field="url",
        score_kind="boxed",
    ),
    "math500": DatasetSpec(
        key="math500",
        display_name="MATH500",
        hf_dataset="math-ai/math500",
        config="default",
        split="test",
        question_field="problem",
        ground_truth_field="answer",
        id_field="unique_id",
        score_kind="boxed_normalized",
    ),
    "minerva": DatasetSpec(
        key="minerva",
        display_name="Minerva",
        hf_dataset="math-ai/minervamath",
        config="default",
        split="test",
        question_field="question",
        ground_truth_field="answer",
        id_field=None,
        score_kind="boxed_normalized",
    ),
    "olympiadbench": DatasetSpec(
        key="olympiadbench",
        display_name="OlympiadBench",
        hf_dataset="math-ai/olympiadbench",
        config="default",
        split="test",
        question_field="question",
        ground_truth_field="final_answer",
        id_field="id",
        context_field="context",
        score_kind="boxed_normalized",
        tolerance_field="error",
    ),
    "gpqa_diamond": DatasetSpec(
        key="gpqa_diamond",
        display_name="GPQA Diamond",
        hf_dataset="Idavidrein/gpqa",
        config="gpqa_diamond",
        split="train",
        question_field="Question",
        ground_truth_field="Correct Answer",
        id_field=None,
        score_kind="boxed_normalized",
    ),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare naive pass@k against iterative explored-path exclusion across math benchmarks."
    )
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS.keys()),
        default=DEFAULT_DATASET_KEYS,
        help="One or more benchmark datasets to evaluate.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional per-dataset subset size for smoke runs.")
    ap.add_argument("--attempts", type=int, default=8)
    ap.add_argument(
        "--solve-max-tokens",
        type=int,
        nargs="+",
        default=[8192, 32768],
        help="One or more max output lengths to evaluate.",
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.7)
    ap.add_argument("--summary-max-tokens", type=int, default=512)
    ap.add_argument("--summary-temperature", type=float, default=0.0)
    ap.add_argument("--summary-top-p", type=float, default=1.0)
    ap.add_argument("--base-seed", type=int, default=3407)
    ap.add_argument("--concurrency", type=int, default=32, help="Batch size for direct vLLM generate calls.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpus-per-node", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument(
        "--server-prompt-length",
        type=int,
        default=2048,
        help="Prompt-side token budget reserved for the local vLLM server context window.",
    )
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge per-shard outputs under <out-dir>/shards/shardXXofYY into <out-dir>.",
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        choices=["baseline", "iterative"],
        default=["baseline", "iterative"],
        help="Which evaluation methods to run. Existing artifacts for skipped methods are reused if present.",
    )
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args()


def ensure_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        model_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(args.model)).strip("_") or "model"
        out_dir = REPO_ROOT / "analysis" / "math_explored_paths" / f"{stamp}_{model_safe}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def ordered_methods(methods: list[str]) -> list[str]:
    return [method for method in ["baseline", "iterative"] if method in methods]


def olympiadbench_filter(row: dict[str, Any]) -> bool:
    return (
        row.get("modality") == "Text-only"
        and row.get("subject") == "Math"
        and row.get("language") == "English"
        and all(row.get(f"image_{i}") is None for i in range(1, 6))
    )


def olympiadbench_single_answer_filter(row: dict[str, Any]) -> bool:
    return not bool(row.get("is_multiple_answer"))


def normalize_ground_truth(raw_value: Any, *, extract_boxed: bool) -> str:
    if isinstance(raw_value, list):
        if not raw_value:
            return ""
        raw_value = raw_value[0]
    text = str(raw_value or "").strip()
    if not text:
        return ""
    if extract_boxed:
        try:
            boxed = last_boxed_only_string(text)
        except Exception:
            boxed = None
        if boxed:
            try:
                return remove_boxed(boxed).strip()
            except Exception:
                pass
    try:
        return remove_boxed(text).strip()
    except Exception:
        return text


def parse_optional_tolerance(raw_value: Any) -> Optional[float]:
    if raw_value in (None, "", "None"):
        return None
    try:
        return float(str(raw_value))
    except ValueError:
        return None


def build_question_text(raw_row: dict[str, Any], spec: DatasetSpec) -> str:
    question_raw = str(raw_row[spec.question_field]).strip()
    if spec.context_field:
        context = str(raw_row.get(spec.context_field) or "").strip()
        if context:
            return f"{context}\n\n{question_raw}"
    return question_raw


def answer_instruction_for_dataset(dataset_key: str) -> str:
    if dataset_key == "gpqa_diamond":
        return GPQA_DIAMOND_INSTRUCTION
    return instruction_following


def build_gpqa_problem_prompt(raw_row: dict[str, Any], row_index: int) -> tuple[str, str]:
    question = str(raw_row.get("Question") or "").strip()
    correct = str(raw_row.get("Correct Answer") or "").strip()
    incorrect_answers = [str(raw_row.get(f"Incorrect Answer {idx}") or "").strip() for idx in range(1, 4)]
    if not question or not correct or any(not answer for answer in incorrect_answers):
        raise ValueError(f"Malformed gpqa_diamond row {row_index}: {raw_row!r}")

    options = [(correct, True)] + [(answer, False) for answer in incorrect_answers]
    random.Random(f"gpqa_diamond:{row_index}").shuffle(options)

    option_lines: list[str] = []
    ground_truth = ""
    for label, (answer_text, is_correct) in zip("ABCD", options, strict=True):
        option_lines.append(f"{label}. {answer_text}")
        if is_correct:
            ground_truth = label
    if not ground_truth:
        raise RuntimeError(f"Failed to derive GPQA ground truth for row {row_index}.")

    question_raw = f"{question}\n" + "\n".join(option_lines)
    return question_raw, ground_truth


def load_problem_set(dataset_key: str, limit: Optional[int]) -> DatasetLoadResult:
    spec = DATASET_SPECS[dataset_key]
    dataset = datasets.load_dataset(spec.hf_dataset, spec.config, split=spec.split)
    raw_count = len(dataset)
    subset_metadata: dict[str, Any] = {}
    if dataset_key == "olympiadbench":
        dataset = dataset.filter(olympiadbench_filter)
        text_only_count = len(dataset)
        dataset = dataset.filter(olympiadbench_single_answer_filter)
        filtered_count = len(dataset)
        subset_metadata = {
            "applied": True,
            "base_subset": "Text-only English Math rows without images",
            "excluded_field": "is_multiple_answer",
            "excluded_multi_answer_count": text_only_count - filtered_count,
            "base_subset_count": text_only_count,
            "kept_single_answer_count": filtered_count,
            "reason": OLYMPIADBENCH_SINGLE_ANSWER_SUBSET_REASON,
        }
    else:
        filtered_count = len(dataset)
    problems: list[ProblemSpec] = []
    for idx, raw_row in enumerate(dataset):
        if limit is not None and len(problems) >= limit:
            break
        row_dict = dict(raw_row)
        if dataset_key == "gpqa_diamond":
            question_raw, ground_truth = build_gpqa_problem_prompt(row_dict, idx)
            baseline_prompt = f"{question_raw}\n\n{answer_instruction_for_dataset(dataset_key)}".strip()
        else:
            question_raw = build_question_text(row_dict, spec)
            baseline_prompt = f"{question_raw} {answer_instruction_for_dataset(dataset_key)}".strip()
            ground_truth = normalize_ground_truth(
                row_dict.get(spec.ground_truth_field),
                extract_boxed=spec.extract_boxed_ground_truth,
            )
        if not ground_truth:
            raise ValueError(f"Missing ground truth for {dataset_key} row {idx}: {row_dict!r}")
        problems.append(
            ProblemSpec(
                dataset_key=dataset_key,
                row_index=idx,
                raw_id=row_dict.get(spec.id_field) if spec.id_field else idx,
                url=str(row_dict.get(spec.url_field or "") or ""),
                question_raw=question_raw,
                baseline_prompt=baseline_prompt,
                ground_truth=ground_truth,
                score_tolerance=parse_optional_tolerance(row_dict.get(spec.tolerance_field))
                if spec.tolerance_field
                else None,
            )
        )
    return DatasetLoadResult(
        spec=spec,
        problems=problems,
        raw_count=raw_count,
        filtered_count=filtered_count,
        subset_metadata=subset_metadata,
    )


def build_iterative_prompt(problem: ProblemSpec, failed_summaries: list[str]) -> str:
    if not failed_summaries:
        return problem.baseline_prompt
    explored = "\n\n".join(
        f'<explore id="{i}">\n{summary_text_from_payload(parse_summary_text(summary))}\n</explore>'
        for i, summary in enumerate(failed_summaries, start=1)
    )
    instruction_text = answer_instruction_for_dataset(problem.dataset_key)
    return (
        "[Problem]\n"
        f"{problem.question_raw}\n\n"
        "[Explored paths]\n"
        f"{explored}\n\n"
        "[Instruction]\n"
        "The <explore> blocks above are failed solution paths. Treat them as wrong routes to falsify, not as facts, hints, evidence, or partial progress.\n"
        "- Assume the most recent failed route is wrong unless you find a new decisive observation that the failed route did not use.\n"
        "- Do not reuse a prior route signature or prior underlying mistake. No paraphrase, arithmetic tweak, local detail swap, or restatement counts as new.\n"
        "- Start from a different principle, representation, invariant, structural test, case split, option-by-option comparison, or derived quantity than any explored path used.\n"
        "- If the problem has explicit options or candidate conclusions, first identify the strongest route not already represented in the explored paths and test it directly.\n"
        "- If your reasoning falls back into a failed route for the same basic reason, discard it and restart from a different angle.\n\n"
        f"{instruction_text}"
    )


def collapse_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def extract_boxed_answer(response: str) -> tuple[Optional[str], Optional[str]]:
    try:
        boxed = last_boxed_only_string(response)
    except Exception:
        boxed = None
    if not boxed:
        return None, None
    try:
        raw = remove_boxed(boxed).strip()
    except Exception:
        return None, None
    if not raw:
        return None, None
    try:
        normalized = strip_string(raw)
    except Exception:
        normalized = raw
    return raw, normalized or raw


SCIENTIFIC_E_RE = re.compile(
    r"(?<![\w.])(?P<mantissa>[+-]?(?:\d+(?:\.\d*)?|\.\d+))[eE](?P<exp>[+-]?\d+)(?![\w])"
)
SCIENTIFIC_TEN_RE = re.compile(
    r"(?P<mantissa>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:\\times|\\cdot|\*)\s*10\^\{?(?P<exp>[+-]?\d+)\}?"
)


def canonicalize_scientific_notation(text: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        return f"{match.group('mantissa')}*10^{{{match.group('exp')}}}"

    text = SCIENTIFIC_TEN_RE.sub(replacer, text)
    return SCIENTIFIC_E_RE.sub(replacer, text)


def normalize_boxed_answer_for_scoring(text: str) -> str:
    text = text.replace("\\\\", "\\")
    normalized = canonicalize_scientific_notation(text)
    normalized = normalize_final_answer(normalized)
    try:
        normalized = strip_string(normalized)
    except Exception:
        pass
    return collapse_whitespace(normalized)


def score_boxed_normalized_response(response: str, ground_truth: str) -> float:
    # This intentionally trades away entropy_math's broader symbolic equivalence in favor of
    # a repo-local boxed-answer normalization path that is stable in the cluster container.
    extracted_answer_raw, _ = extract_boxed_answer(response)
    if not extracted_answer_raw:
        return 0.0
    normalized_pred = normalize_boxed_answer_for_scoring(extracted_answer_raw)
    normalized_gt = normalize_boxed_answer_for_scoring(ground_truth)
    return 1.0 if normalized_pred == normalized_gt else 0.0


def score_response(response: str, ground_truth: str, dataset_key: str) -> float:
    score_kind = DATASET_SPECS[dataset_key].score_kind
    if score_kind == "boxed":
        return float(boxed_compute_score(response, ground_truth))
    if score_kind == "boxed_normalized":
        return score_boxed_normalized_response(response, ground_truth)
    raise ValueError(f"Unsupported score kind: {score_kind}")


def try_parse_numeric(text: str) -> Optional[float]:
    cleaned = collapse_whitespace(text).replace("$", "").replace("\\%", "").replace("%", "").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def score_with_tolerance(
    response: str,
    ground_truth: str,
    *,
    dataset_key: str,
    score_tolerance: Optional[float],
) -> float:
    base_score = score_response(response, ground_truth, dataset_key)
    if base_score >= 1.0 or score_tolerance is None:
        return base_score
    extracted_answer_raw, extracted_answer_normalized = extract_boxed_answer(response)
    gt_numeric = try_parse_numeric(ground_truth)
    if gt_numeric is None:
        return base_score
    for candidate in (extracted_answer_raw, extracted_answer_normalized):
        if not candidate:
            continue
        pred_numeric = try_parse_numeric(candidate)
        if pred_numeric is None:
            continue
        if abs(pred_numeric - gt_numeric) <= score_tolerance:
            return 1.0
    return base_score


def normalize_summary_field(text: Any) -> str:
    return collapse_whitespace(redact_summary_answer_mentions(str(text or "").replace("\n", " ")))


SUMMARY_ANSWER_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\boption\s+[A-D]\b", re.IGNORECASE), "that option"),
    (re.compile(r"\bchoice\s+[A-D]\b", re.IGNORECASE), "that choice"),
    (re.compile(r"\banswer\s+[A-D]\b", re.IGNORECASE), "that answer"),
    (re.compile(r"\bselected\s+as\s+[A-D]\b", re.IGNORECASE), "selected as one candidate"),
    (
        re.compile(
            r"\b(?:pick|picks|picked|choose|chooses|chose|chosen|select|selects|selected)\s+(?:option\s+)?[A-D]\b",
            re.IGNORECASE,
        ),
        "select that option",
    ),
)


def redact_summary_answer_mentions(text: str) -> str:
    redacted = text
    for pattern, replacement in SUMMARY_ANSWER_REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def normalize_summary_list_field(
    values: Any,
    *,
    field_name: str,
    min_items: int,
    max_items: int,
) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list.")
    if len(values) < min_items or len(values) > max_items:
        raise ValueError(f"{field_name} must contain between {min_items} and {max_items} items.")
    normalized = [normalize_summary_field(value) for value in values]
    if any(not item for item in normalized):
        raise ValueError(f"{field_name} contains an empty item.")
    return normalized


def normalize_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Summary payload must be a JSON object.")
    expected_keys = {"route", "route_signature", "relationship_to_prior"}
    legacy_expected_keys = expected_keys | {"attempted_answer"}
    payload_keys = set(payload.keys())
    if payload_keys not in (expected_keys, legacy_expected_keys):
        raise ValueError(
            "Summary payload keys must be exactly "
            f"{sorted(expected_keys)} or {sorted(legacy_expected_keys)}."
        )

    route = normalize_summary_field(payload.get("route"))
    if not route:
        raise ValueError("route must be non-empty.")

    route_signature = normalize_summary_list_field(
        payload.get("route_signature"),
        field_name="route_signature",
        min_items=3,
        max_items=6,
    )

    relationship_to_prior = payload.get("relationship_to_prior")
    if not isinstance(relationship_to_prior, dict):
        raise ValueError("relationship_to_prior must be an object.")
    expected_relationship_keys = {"differences", "overlap"}
    if set(relationship_to_prior.keys()) != expected_relationship_keys:
        raise ValueError(
            f"relationship_to_prior keys must be exactly {sorted(expected_relationship_keys)}."
        )
    differences = normalize_summary_list_field(
        relationship_to_prior.get("differences"),
        field_name="relationship_to_prior.differences",
        min_items=0,
        max_items=5,
    )
    overlap = normalize_summary_list_field(
        relationship_to_prior.get("overlap"),
        field_name="relationship_to_prior.overlap",
        min_items=0,
        max_items=3,
    )
    return {
        "route": route,
        "route_signature": route_signature,
        "relationship_to_prior": {
            "differences": differences,
            "overlap": overlap,
        },
    }


def summary_text_from_payload(payload: dict[str, Any]) -> str:
    normalized_payload = normalize_summary_payload(payload)
    return json.dumps(normalized_payload, ensure_ascii=False)


def parse_summary_text(summary: str) -> dict[str, Any]:
    try:
        payload = json.loads(summary)
    except json.JSONDecodeError as exc:
        raise ValueError("Stored summary is not valid JSON.") from exc
    return normalize_summary_payload(payload)


def render_prior_summaries_for_prompt(prior_summaries: list[str]) -> str:
    if not prior_summaries:
        return "[]"
    prior_payloads = [parse_summary_text(summary) for summary in prior_summaries]
    return json.dumps(prior_payloads, ensure_ascii=False, indent=2)


def route_key_from_summary(summary: str) -> str:
    payload = parse_summary_text(summary)
    route = normalize_summary_field(payload["route"]).lower()
    signature = " | ".join(normalize_summary_field(tag).lower() for tag in payload["route_signature"])
    return f"{route} || {signature}"


def answer_key(extracted_answer_normalized: Optional[str]) -> str:
    return extracted_answer_normalized if extracted_answer_normalized is not None else "__none__"


def preview(text: str, limit: int = 260) -> str:
    clean = collapse_whitespace(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


async def start_server(args: argparse.Namespace) -> GenerationContext:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=False,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        disable_custom_all_reduce=True,
    )
    return GenerationContext(llm=llm, tokenizer=tokenizer)


async def stop_server(context: GenerationContext) -> None:
    llm_engine = getattr(context.llm, "llm_engine", None)
    if llm_engine is not None and hasattr(llm_engine, "shutdown"):
        llm_engine.shutdown()


def render_prompt(tokenizer: Any, messages: list[dict[str, str]], *, enable_thinking: Optional[bool]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


async def run_requests(
    context: GenerationContext,
    requests: list[ChatRequest],
    *,
    concurrency: int,
) -> list[list[str]]:
    results: list[list[str]] = []
    batch_size = max(1, concurrency)
    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        prompt_texts: list[str] = []
        sampling_params: list[SamplingParams] = []
        for request in batch:
            chat_kwargs = ((request.extra_body or {}).get("chat_template_kwargs") or {})
            enable_thinking = chat_kwargs.get("enable_thinking")
            prompt_texts.append(render_prompt(context.tokenizer, request.messages, enable_thinking=enable_thinking))
            sampling_params.append(
                SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    n=request.n,
                    seed=request.seed,
                    stop=request.stop,
                    guided_decoding=request.guided_decoding,
                )
            )
        batch_outputs = context.llm.generate(prompt_texts, sampling_params=sampling_params, use_tqdm=False)
        for request, output in zip(batch, batch_outputs, strict=True):
            texts = [sample.text for sample in output.outputs]
            if len(texts) != request.n:
                raise RuntimeError(f"Expected {request.n} outputs, got {len(texts)}")
            results.append(texts)
    return results


def parse_summary_attempt_output(problem: ProblemSpec, output: list[str]) -> SummaryAttemptResult:
    raw_text = output[0] if output and output[0] else ""
    raw_text_stripped = raw_text.strip()
    if not raw_text_stripped:
        raise RuntimeError(f"Empty summary output for {problem.dataset_key} problem {problem.row_index}.")
    try:
        payload = json.loads(raw_text_stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Summary JSON parse failed for "
            f"{problem.dataset_key} problem {problem.row_index}: {preview(raw_text, limit=400)}"
        ) from exc
    try:
        summary = summary_text_from_payload(payload)
    except Exception as exc:
        raise RuntimeError(
            "Summary payload validation failed for "
            f"{problem.dataset_key} problem {problem.row_index}: {preview(raw_text, limit=400)}"
        ) from exc
    return SummaryAttemptResult(summary=summary, raw_summary_output=raw_text)


async def summarize_attempts(
    problems: list[ProblemSpec],
    responses: list[str],
    prior_summaries: list[list[str]],
    *,
    context: GenerationContext,
    args: argparse.Namespace,
    seed_offset: int,
) -> SummaryBatchResult:
    results: list[Optional[SummaryAttemptResult]] = [None] * len(problems)
    failures: dict[int, SummaryAttemptFailure] = {}
    pending: list[tuple[int, ProblemSpec, str, list[str]]] = [
        (idx, problem, response, problem_prior_summaries)
        for idx, (problem, response, problem_prior_summaries) in enumerate(
            zip(problems, responses, prior_summaries, strict=True)
        )
    ]

    for summary_attempt in range(1, SUMMARY_RETRY_LIMIT + 1):
        if not pending:
            break
        retry_temperature = args.summary_temperature
        if summary_attempt > 1 and retry_temperature <= 0.0:
            retry_temperature = SUMMARY_RETRY_TEMPERATURE
        requests: list[ChatRequest] = []
        for original_idx, problem, response, problem_prior_summaries in pending:
            prior_summaries_text = render_prior_summaries_for_prompt(problem_prior_summaries)
            user_prompt = textwrap.dedent(
                f"""\
                PROBLEM:
                {problem.question_raw}

                PRIOR_SUMMARIES:
                {prior_summaries_text}

                CURRENT_ATTEMPT:
                {response.strip()}
                """
            )
            requests.append(
                ChatRequest(
                    messages=[
                        {"role": "system", "content": SUMMARY_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=retry_temperature,
                    top_p=args.summary_top_p,
                    max_tokens=args.summary_max_tokens,
                    seed=args.base_seed
                    + seed_offset
                    + original_idx
                    + (summary_attempt - 1) * SUMMARY_RETRY_SEED_STRIDE,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    guided_decoding=GuidedDecodingParams(json=SUMMARY_RESPONSE_SCHEMA),
                )
            )
        raw_outputs = await run_requests(context, requests, concurrency=args.concurrency)
        next_pending: list[tuple[int, ProblemSpec, str, list[str]]] = []
        for (original_idx, problem, response, problem_prior_summaries), output in zip(
            pending, raw_outputs, strict=True
        ):
            try:
                results[original_idx] = parse_summary_attempt_output(problem, output)
                failures.pop(original_idx, None)
            except RuntimeError as exc:
                failures[original_idx] = SummaryAttemptFailure(error=str(exc), attempts=summary_attempt)
                if summary_attempt < SUMMARY_RETRY_LIMIT:
                    next_pending.append((original_idx, problem, response, problem_prior_summaries))
        pending = next_pending

    return SummaryBatchResult(results=results, failures=failures)


def make_skipped_problem_record(
    problem: ProblemSpec,
    *,
    method: str,
    completed_attempts: int,
) -> dict[str, Any]:
    return {
        "dataset_key": problem.dataset_key,
        "problem_index": problem.row_index,
        "problem_id": problem.raw_id,
        "problem_url": problem.url,
        "method": method,
        "skip_stage": "summary_generation",
        "skip_reason": "summary_generation_failure",
        "completed_attempts": completed_attempts,
    }


def make_trace(
    *,
    method: str,
    prompt_variant: str,
    max_tokens: int,
    problem: ProblemSpec,
    attempt_index: int,
    prompt_text: str,
    response: str,
    extracted_answer_raw: Optional[str],
    extracted_answer_normalized: Optional[str],
    score: float,
    summary: str,
    raw_summary_output: str,
    accumulated_explored_paths_before: list[str],
    request_seed: int,
) -> dict[str, Any]:
    return {
        "dataset_key": problem.dataset_key,
        "method": method,
        "prompt_variant": prompt_variant,
        "max_tokens": max_tokens,
        "problem_index": problem.row_index,
        "problem_id": problem.raw_id,
        "problem_url": problem.url,
        "ground_truth": problem.ground_truth,
        "score_tolerance": problem.score_tolerance,
        "attempt_index": attempt_index,
        "request_seed": request_seed,
        "solver_prompt": prompt_text,
        "raw_response": response,
        "extracted_answer_raw": extracted_answer_raw,
        "extracted_answer_normalized": extracted_answer_normalized,
        "score": score,
        "correct": bool(score >= 1.0),
        "summary": summary,
        "raw_summary_output": raw_summary_output,
        "summary_route_key": route_key_from_summary(summary),
        "accumulated_explored_paths_before": list(accumulated_explored_paths_before),
    }


async def run_baseline(
    problems: list[ProblemSpec],
    *,
    context: GenerationContext,
    args: argparse.Namespace,
    max_tokens: int,
) -> MethodRunResult:
    solve_requests = [
        ChatRequest(
            messages=[{"role": "user", "content": problem.baseline_prompt}],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=max_tokens,
            n=args.attempts,
            seed=args.base_seed + problem.row_index,
        )
        for problem in problems
    ]
    solve_outputs = await run_requests(context, solve_requests, concurrency=args.concurrency)

    flat_problems: list[ProblemSpec] = []
    flat_responses: list[str] = []
    metadata: list[tuple[ProblemSpec, int]] = []
    for problem, responses in zip(problems, solve_outputs, strict=True):
        if len(responses) != args.attempts:
            raise RuntimeError(f"Expected {args.attempts} baseline responses for problem {problem.row_index}, got {len(responses)}")
        for attempt_offset, response in enumerate(responses, start=1):
            flat_problems.append(problem)
            flat_responses.append(response)
            metadata.append((problem, attempt_offset))

    summary_batch = await summarize_attempts(
        flat_problems,
        flat_responses,
        prior_summaries=[[] for _ in flat_problems],
        context=context,
        args=args,
        seed_offset=50_000 + max_tokens,
    )
    skipped_by_problem: dict[int, dict[str, Any]] = {}
    for flat_idx, failure in sorted(summary_batch.failures.items()):
        problem, attempt_index = metadata[flat_idx]
        row = skipped_by_problem.setdefault(
            problem.row_index,
            make_skipped_problem_record(problem, method="baseline", completed_attempts=0),
        )
        row.setdefault("failed_attempt_indices", []).append(attempt_index)
        row.setdefault("summary_failures", []).append(
            {
                "attempt_index": attempt_index,
                "summary_retry_attempts": failure.attempts,
                "error": failure.error,
            }
        )
    skipped_problem_indices = set(skipped_by_problem)
    skipped_problems = sort_skipped_problems(list(skipped_by_problem.values()))
    for row in skipped_problems:
        row["failed_attempt_indices"] = sorted(int(idx) for idx in row.get("failed_attempt_indices", []))
        row["summary_failures"] = sorted(
            row.get("summary_failures", []),
            key=lambda failure_row: int(failure_row["attempt_index"]),
        )
        print(
            f"[baseline] skipping dataset={row['dataset_key']} problem={row['problem_index']} "
            f"after summary-generation failures on attempts={row['failed_attempt_indices']}",
            flush=True,
        )

    traces: list[dict[str, Any]] = []
    for (problem, attempt_index), response, summary_result in zip(
        metadata, flat_responses, summary_batch.results, strict=True
    ):
        if problem.row_index in skipped_problem_indices:
            continue
        if summary_result is None:
            raise RuntimeError(
                f"Missing summary result for baseline {problem.dataset_key} problem {problem.row_index}."
            )
        extracted_answer_raw, extracted_answer_normalized = extract_boxed_answer(response)
        score = score_with_tolerance(
            response,
            problem.ground_truth,
            dataset_key=problem.dataset_key,
            score_tolerance=problem.score_tolerance,
        )
        traces.append(
            make_trace(
                method="baseline",
                prompt_variant="baseline_prompt",
                max_tokens=max_tokens,
                problem=problem,
                attempt_index=attempt_index,
                prompt_text=problem.baseline_prompt,
                response=response,
                extracted_answer_raw=extracted_answer_raw,
                extracted_answer_normalized=extracted_answer_normalized,
                score=score,
                summary=summary_result.summary,
                raw_summary_output=summary_result.raw_summary_output,
                accumulated_explored_paths_before=[],
                request_seed=args.base_seed + problem.row_index,
            )
        )
    return MethodRunResult(traces=traces, skipped_problems=skipped_problems)


async def run_iterative(
    problems: list[ProblemSpec],
    *,
    context: GenerationContext,
    args: argparse.Namespace,
    max_tokens: int,
) -> MethodRunResult:
    failed_summaries: list[list[str]] = [[] for _ in problems]
    traces: list[dict[str, Any]] = []
    skipped_problems: list[dict[str, Any]] = []
    active_indices = list(range(len(problems)))

    for round_idx in range(args.attempts):
        if not active_indices:
            break
        round_problems = [problems[idx] for idx in active_indices]
        prior_summaries_before_round = [list(failed_summaries[idx]) for idx in active_indices]
        prompt_texts = [
            build_iterative_prompt(problems[idx], failed_summaries[idx]) for idx in active_indices
        ]
        solve_requests = [
            ChatRequest(
                messages=[{"role": "user", "content": prompt_text}],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_tokens,
                n=1,
                seed=args.base_seed + 10_000 * (round_idx + 1) + problem.row_index,
            )
            for problem, prompt_text in zip(round_problems, prompt_texts, strict=True)
        ]
        solve_outputs = await run_requests(context, solve_requests, concurrency=args.concurrency)
        responses = [output[0] for output in solve_outputs]
        summary_batch = await summarize_attempts(
            round_problems,
            responses,
            prior_summaries_before_round,
            context=context,
            args=args,
            seed_offset=60_000 + max_tokens + 100 * round_idx,
        )
        next_active_indices: list[int] = []

        for local_idx, global_idx in enumerate(active_indices):
            problem = problems[global_idx]
            prompt_text = prompt_texts[local_idx]
            response = responses[local_idx]
            summary_result = summary_batch.results[local_idx]
            failure = summary_batch.failures.get(local_idx)
            if failure is not None:
                traces = [
                    row for row in traces if int(row["problem_index"]) != problem.row_index
                ]
                skipped_row = make_skipped_problem_record(
                    problem,
                    method="iterative",
                    completed_attempts=round_idx,
                )
                skipped_row["failed_attempt_index"] = round_idx + 1
                skipped_row["summary_retry_attempts"] = failure.attempts
                skipped_row["summary_failure_error"] = failure.error
                skipped_problems.append(skipped_row)
                print(
                    f"[iterative] skipping dataset={problem.dataset_key} problem={problem.row_index} "
                    f"at attempt={round_idx + 1} after summary-generation failures",
                    flush=True,
                )
                continue
            if summary_result is None:
                raise RuntimeError(
                    f"Missing summary result for iterative {problem.dataset_key} problem {problem.row_index}."
                )
            extracted_answer_raw, extracted_answer_normalized = extract_boxed_answer(response)
            score = score_with_tolerance(
                response,
                problem.ground_truth,
                dataset_key=problem.dataset_key,
                score_tolerance=problem.score_tolerance,
            )
            current_failed = list(prior_summaries_before_round[local_idx])
            prompt_variant = "explored_paths_prompt" if current_failed else "baseline_prompt"
            traces.append(
                make_trace(
                    method="iterative",
                    prompt_variant=prompt_variant,
                    max_tokens=max_tokens,
                    problem=problem,
                    attempt_index=round_idx + 1,
                    prompt_text=prompt_text,
                    response=response,
                    extracted_answer_raw=extracted_answer_raw,
                    extracted_answer_normalized=extracted_answer_normalized,
                    score=score,
                    summary=summary_result.summary,
                    raw_summary_output=summary_result.raw_summary_output,
                    accumulated_explored_paths_before=current_failed,
                    request_seed=args.base_seed + 10_000 * (round_idx + 1) + problem.row_index,
                )
            )
            if score < 1.0:
                failed_summaries[global_idx].append(summary_result.summary)
            next_active_indices.append(global_idx)
        active_indices = next_active_indices
    return MethodRunResult(traces=traces, skipped_problems=sort_skipped_problems(skipped_problems))


def traces_by_problem(traces: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for trace in traces:
        grouped.setdefault(int(trace["problem_index"]), []).append(trace)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["attempt_index"]))
    return grouped


def compute_metrics(
    traces: list[dict[str, Any]],
    *,
    total_problem_count: Optional[int] = None,
    skipped_problems: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    grouped = traces_by_problem(traces)
    skipped_problem_rows = skipped_problems or []
    pass1: list[float] = []
    passk: list[float] = []
    unique_answers: list[int] = []
    repeated_answer_flag: list[float] = []
    unique_routes: list[int] = []
    repeated_route_flag: list[float] = []

    for rows in grouped.values():
        scores = [float(row["score"]) for row in rows]
        answers = [answer_key(row["extracted_answer_normalized"]) for row in rows]
        routes = [str(row["summary_route_key"]) for row in rows]
        pass1.append(1.0 if scores[0] >= 1.0 else 0.0)
        passk.append(1.0 if any(score >= 1.0 for score in scores) else 0.0)
        unique_answers.append(len(set(answers)))
        repeated_answer_flag.append(1.0 if len(set(answers)) < len(answers) else 0.0)
        unique_routes.append(len(set(routes)))
        repeated_route_flag.append(1.0 if len(set(routes)) < len(routes) else 0.0)

    num_scored_problems = len(grouped)
    num_total_problems = total_problem_count if total_problem_count is not None else num_scored_problems
    return {
        "num_problems": num_scored_problems,
        "num_scored_problems": num_scored_problems,
        "num_total_problems": num_total_problems,
        "num_skipped_problems": len(skipped_problem_rows),
        "pass_at_1": statistics.mean(pass1) if pass1 else 0.0,
        "pass_at_k": statistics.mean(passk) if passk else 0.0,
        "avg_unique_final_answers": statistics.mean(unique_answers) if unique_answers else 0.0,
        "fraction_with_repeated_final_answers": statistics.mean(repeated_answer_flag) if repeated_answer_flag else 0.0,
        "avg_distinct_path_summaries": statistics.mean(unique_routes) if unique_routes else 0.0,
        "fraction_with_repeated_path_summaries": statistics.mean(repeated_route_flag) if repeated_route_flag else 0.0,
    }


def pick_baseline_repeat_case(
    problems: list[ProblemSpec],
    baseline_grouped: dict[int, list[dict[str, Any]]],
) -> Optional[tuple[ProblemSpec, list[dict[str, Any]]]]:
    for problem in problems:
        rows = baseline_grouped.get(problem.row_index, [])
        if not rows:
            continue
        if any(row["correct"] for row in rows):
            continue
        answers = [answer_key(row["extracted_answer_normalized"]) for row in rows]
        if len(set(answers)) < len(answers):
            return problem, rows
    return None


def pick_iterative_success_case(
    problems: list[ProblemSpec],
    baseline_grouped: dict[int, list[dict[str, Any]]],
    iterative_grouped: dict[int, list[dict[str, Any]]],
) -> Optional[tuple[ProblemSpec, list[dict[str, Any]], list[dict[str, Any]]]]:
    for problem in problems:
        irows = iterative_grouped.get(problem.row_index, [])
        brows = baseline_grouped.get(problem.row_index, [])
        if not irows or not brows:
            continue
        if not any(row["correct"] for row in irows):
            continue
        if irows[0]["correct"]:
            continue
        if any(row["correct"] for row in brows):
            continue
        return problem, brows, irows
    for problem in problems:
        irows = iterative_grouped.get(problem.row_index, [])
        if irows and any(row["correct"] for row in irows) and not irows[0]["correct"]:
            return problem, baseline_grouped.get(problem.row_index, []), irows
    return None


def format_attempt_lines(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"{row['attempt_index']}. answer={row['extracted_answer_raw'] or 'none'} "
            f"correct={'yes' if row['correct'] else 'no'} "
            f"summary={row['summary']}"
        )
    return "\n".join(lines)


def build_examples_markdown(
    problems: list[ProblemSpec],
    baseline_traces: list[dict[str, Any]],
    iterative_traces: list[dict[str, Any]],
    *,
    available_methods: Optional[list[str]] = None,
) -> str:
    baseline_grouped = traces_by_problem(baseline_traces)
    iterative_grouped = traces_by_problem(iterative_traces)
    dataset_label = DATASET_SPECS[problems[0].dataset_key].display_name if problems else "Dataset"
    available_method_set = set(available_methods or ["baseline", "iterative"])

    sections: list[str] = [f"# {dataset_label} Explored-Path Examples", ""]

    if "baseline" in available_method_set:
        baseline_case = pick_baseline_repeat_case(problems, baseline_grouped)
        if baseline_case is not None:
            problem, rows = baseline_case
            sections.extend(
                [
                    "## Baseline Repeated Failure",
                    f"- Problem index: `{problem.row_index}`",
                    f"- Problem id: `{problem.raw_id}`",
                    f"- Reference: {problem.url or '(none)'}",
                    f"- Prompt preview: `{preview(problem.baseline_prompt)}`",
                    "",
                    "```text",
                    format_attempt_lines(rows),
                    "```",
                    "",
                ]
            )
        else:
            sections.extend(["## Baseline Repeated Failure", "No clean repeated-failure example was found.", ""])

    if "iterative" in available_method_set:
        iterative_case = pick_iterative_success_case(problems, baseline_grouped, iterative_grouped)
        if iterative_case is not None:
            problem, brows, irows = iterative_case
            success_row = next(row for row in irows if row["correct"])
            sections.extend(
                [
                    "## Iterative Summary Evolution",
                    f"- Problem index: `{problem.row_index}`",
                    f"- Problem id: `{problem.raw_id}`",
                    f"- Reference: {problem.url or '(none)'}",
                    f"- Prompt preview: `{preview(problem.baseline_prompt)}`",
                    "",
                    "```text",
                    format_attempt_lines(irows),
                    "```",
                    "",
                    "## Iterative Success After Prior Exclusions",
                    f"- First successful attempt: `{success_row['attempt_index']}`",
                    f"- Successful answer: `{success_row['extracted_answer_raw'] or 'none'}`",
                    f"- Success summary: `{success_row['summary']}`",
                    "",
                ]
            )
            if brows:
                sections.extend(
                    [
                        "### Matching Baseline Attempts",
                        "",
                        "```text",
                        format_attempt_lines(brows),
                        "```",
                        "",
                    ]
                )
        else:
            sections.extend(
                [
                    "## Iterative Summary Evolution",
                    "No iterative success-after-failure example was found.",
                    "",
                ]
            )
    return "\n".join(sections).strip() + "\n"


def build_results_markdown(
    results_by_dataset: dict[str, dict[int, dict[str, Any]]],
    dataset_metadata: dict[str, dict[str, Any]],
    attempts: int,
) -> str:
    lines = [
        "# Math Explored-Path Results",
        "",
        f"| dataset | rows | max_tokens | method | pass@1 | pass@{attempts} | avg unique final answers | frac repeated final answers | avg distinct path summaries | frac repeated path summaries |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset_key, results_by_length in sorted(results_by_dataset.items()):
        rows = int(dataset_metadata.get(dataset_key, {}).get("num_problems", 0))
        dataset_label = str(dataset_metadata.get(dataset_key, {}).get("display_name", dataset_key))
        for max_tokens, method_map in sorted(results_by_length.items()):
            for method in ordered_methods(list(method_map)):
                metrics = method_map[method]
                lines.append(
                    "| "
                    f"{dataset_label} | {rows} | {max_tokens} | {method} | "
                    f"{metrics['pass_at_1']:.4f} | {metrics['pass_at_k']:.4f} | "
                    f"{metrics['avg_unique_final_answers']:.3f} | {metrics['fraction_with_repeated_final_answers']:.3f} | "
                    f"{metrics['avg_distinct_path_summaries']:.3f} | {metrics['fraction_with_repeated_path_summaries']:.3f} |"
                )
    dataset_notes: list[str] = []
    skip_notes: list[str] = []
    for dataset_key, metadata in sorted(dataset_metadata.items()):
        subset_metadata = metadata.get("subset_metadata")
        if not isinstance(subset_metadata, dict) or not subset_metadata.get("applied"):
            continue
        dataset_notes.append(
            f"- {metadata.get('display_name', dataset_key)}: kept "
            f"{subset_metadata.get('kept_single_answer_count', metadata.get('num_problems', 0))}/"
            f"{subset_metadata.get('base_subset_count', metadata.get('filtered_count', 0))} rows after excluding "
            f"{subset_metadata.get('excluded_multi_answer_count', 0)} multi-answer rows "
            f"(`{subset_metadata.get('excluded_field', 'unknown')}`); reason: {subset_metadata.get('reason', '')}"
        )
    for dataset_key, results_by_length in sorted(results_by_dataset.items()):
        dataset_label = str(dataset_metadata.get(dataset_key, {}).get("display_name", dataset_key))
        for max_tokens, method_map in sorted(results_by_length.items()):
            for method in ordered_methods(list(method_map)):
                metrics = method_map[method]
                num_skipped = int(metrics.get("num_skipped_problems", 0))
                if num_skipped <= 0:
                    continue
                num_scored = int(metrics.get("num_scored_problems", metrics.get("num_problems", 0)))
                skip_notes.append(
                    f"- {dataset_label}, max_tokens={max_tokens}, {method}: summary-generation failures affected "
                    f"{num_skipped} problem(s); metrics retained {num_scored} problem(s) with traces. "
                    f"See `{dataset_key}/len_{max_tokens}/{method}_skipped_problems.jsonl`."
                )
    if dataset_notes:
        lines.extend(["", "## Dataset Notes", ""] + dataset_notes)
    if skip_notes:
        lines.extend(["", "## Summary Failures", ""] + skip_notes)
    return "\n".join(lines) + "\n"


async def run_one_length(
    problems: list[ProblemSpec],
    *,
    context: GenerationContext,
    args: argparse.Namespace,
    max_tokens: int,
    out_dir: Path,
) -> dict[str, Any]:
    length_dir = out_dir / f"len_{max_tokens}"
    length_dir.mkdir(parents=True, exist_ok=True)

    baseline_traces_path = length_dir / "baseline_traces.jsonl"
    baseline_metrics_path = length_dir / "baseline_metrics.json"
    baseline_skipped_path = length_dir / "baseline_skipped_problems.jsonl"
    iterative_traces_path = length_dir / "iterative_traces.jsonl"
    iterative_metrics_path = length_dir / "iterative_metrics.json"
    iterative_skipped_path = length_dir / "iterative_skipped_problems.jsonl"
    requested_methods = set(args.methods)
    baseline_traces: list[dict[str, Any]] = []
    iterative_traces: list[dict[str, Any]] = []
    baseline_metrics: Optional[dict[str, Any]] = None
    iterative_metrics: Optional[dict[str, Any]] = None

    if "baseline" in requested_methods:
        baseline_result = await run_baseline(problems, context=context, args=args, max_tokens=max_tokens)
        baseline_traces = baseline_result.traces
        baseline_metrics = compute_metrics(
            baseline_traces,
            total_problem_count=len(problems),
            skipped_problems=baseline_result.skipped_problems,
        )
        write_jsonl(baseline_traces_path, baseline_traces)
        write_jsonl(baseline_skipped_path, baseline_result.skipped_problems)
        write_json(baseline_metrics_path, baseline_metrics)
    elif baseline_traces_path.exists() and baseline_metrics_path.exists():
        baseline_traces = read_jsonl(baseline_traces_path)
        baseline_metrics = read_json(baseline_metrics_path)

    if "iterative" in requested_methods:
        iterative_result = await run_iterative(
            problems,
            context=context,
            args=args,
            max_tokens=max_tokens,
        )
        iterative_traces = iterative_result.traces
        iterative_metrics = compute_metrics(
            iterative_traces,
            total_problem_count=len(problems),
            skipped_problems=iterative_result.skipped_problems,
        )
        write_jsonl(iterative_traces_path, iterative_traces)
        write_jsonl(iterative_skipped_path, iterative_result.skipped_problems)
        write_json(iterative_metrics_path, iterative_metrics)
    elif iterative_traces_path.exists() and iterative_metrics_path.exists():
        iterative_traces = read_jsonl(iterative_traces_path)
        iterative_metrics = read_json(iterative_metrics_path)

    metrics: dict[str, dict[str, Any]] = {}
    if baseline_metrics is not None:
        metrics["baseline"] = baseline_metrics
    if iterative_metrics is not None:
        metrics["iterative"] = iterative_metrics
    write_json(length_dir / "metrics.json", metrics)
    (length_dir / "examples.md").write_text(
        build_examples_markdown(
            problems,
            baseline_traces,
            iterative_traces,
            available_methods=ordered_methods(list(metrics)),
        ),
        encoding="utf-8",
    )
    return metrics


def save_problem_manifest(problems: list[ProblemSpec], out_dir: Path) -> None:
    rows = [
        {
            "dataset_key": problem.dataset_key,
            "problem_index": problem.row_index,
            "problem_id": problem.raw_id,
            "problem_reference": problem.url,
            "question_raw": problem.question_raw,
            "baseline_prompt": problem.baseline_prompt,
            "ground_truth": problem.ground_truth,
            "score_tolerance": problem.score_tolerance,
        }
        for problem in problems
    ]
    write_jsonl(out_dir / "problems.jsonl", rows)


def shard_output_dir(out_dir: Path, num_shards: int, shard_idx: int) -> Path:
    return out_dir / "shards" / f"shard{shard_idx:02d}of{num_shards:02d}"


def problems_from_manifest(rows: list[dict[str, Any]]) -> list[ProblemSpec]:
    problems: list[ProblemSpec] = []
    for row in rows:
        problems.append(
            ProblemSpec(
                dataset_key=str(row["dataset_key"]),
                row_index=int(row["problem_index"]),
                raw_id=row.get("problem_id"),
                url=str(row.get("problem_reference") or ""),
                question_raw=str(row["question_raw"]),
                baseline_prompt=str(row["baseline_prompt"]),
                ground_truth=str(row["ground_truth"]),
                score_tolerance=parse_optional_tolerance(row.get("score_tolerance")),
            )
        )
    return problems


def dedupe_problem_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_index[int(row["problem_index"])] = row
    return [by_index[idx] for idx in sorted(by_index)]


def dedupe_skipped_problem_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_index[int(row["problem_index"])] = row
    return sort_skipped_problems(list(by_index.values()))


def sort_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        traces,
        key=lambda row: (
            int(row["problem_index"]),
            int(row["attempt_index"]),
            str(row.get("method", "")),
            str(row.get("prompt_variant", "")),
        ),
    )


def sort_skipped_problems(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row["problem_index"]),
            int(row.get("failed_attempt_index", 0) or 0),
        ),
    )


def merge_dataset_metadata_rows(
    dataset_key: str,
    shard_dataset_metadata: list[dict[str, Any]],
    merged_problem_count: int,
    num_shards: int,
) -> dict[str, Any]:
    if not shard_dataset_metadata:
        raise FileNotFoundError(f"No dataset metadata found for dataset '{dataset_key}'.")
    merged = dict(shard_dataset_metadata[0])
    merged["num_problems"] = int(merged.get("num_problems_total", merged_problem_count))
    merged["merged_num_shards"] = num_shards
    merged.pop("num_problems_total", None)
    merged.pop("num_problems_shard", None)
    merged.pop("shard_idx", None)
    return merged


def merge_shard_outputs(args: argparse.Namespace, out_dir: Path) -> None:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    shard_dirs = [shard_output_dir(out_dir, args.num_shards, shard_idx) for shard_idx in range(args.num_shards)]
    missing_shards = [str(path) for path in shard_dirs if not path.exists()]
    if missing_shards:
        raise FileNotFoundError(f"Missing shard output directories: {missing_shards}")

    merge_start = time.time()
    shard_run_configs = [read_json(path / "run_config.json") for path in shard_dirs]
    shard_dataset_metadata = [read_json(path / "dataset_metadata.json") for path in shard_dirs]
    shard_overviews = [read_json(path / "results_overview.json") for path in shard_dirs]

    merged_results_by_dataset: dict[str, dict[int, dict[str, Any]]] = {}
    merged_dataset_metadata: dict[str, dict[str, Any]] = {}
    for dataset_key in args.datasets:
        dataset_dir = out_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)

        problem_rows: list[dict[str, Any]] = []
        per_shard_metadata: list[dict[str, Any]] = []
        for shard_dir, metadata_map in zip(shard_dirs, shard_dataset_metadata, strict=True):
            dataset_metadata_row = metadata_map.get(dataset_key)
            if dataset_metadata_row is not None:
                per_shard_metadata.append(dataset_metadata_row)
            problems_path = shard_dir / dataset_key / "problems.jsonl"
            if problems_path.exists():
                problem_rows.extend(read_jsonl(problems_path))
        merged_problem_rows = dedupe_problem_rows(problem_rows)
        write_jsonl(dataset_dir / "problems.jsonl", merged_problem_rows)
        problems = problems_from_manifest(merged_problem_rows)
        merged_dataset_metadata[dataset_key] = merge_dataset_metadata_rows(
            dataset_key,
            per_shard_metadata,
            len(merged_problem_rows),
            args.num_shards,
        )

        merged_results_by_dataset[dataset_key] = {}
        for max_tokens in args.solve_max_tokens:
            length_dir = dataset_dir / f"len_{max_tokens}"
            length_dir.mkdir(parents=True, exist_ok=True)

            baseline_traces: list[dict[str, Any]] = []
            iterative_traces: list[dict[str, Any]] = []
            merged_metrics: dict[str, dict[str, Any]] = {}
            requested_methods = set(args.methods)
            for method in ordered_methods(["baseline", "iterative"]):
                traces: list[dict[str, Any]] = []
                skipped_problem_rows: list[dict[str, Any]] = []
                found_any = False
                for shard_dir in shard_dirs:
                    shard_length_dir = shard_dir / dataset_key / f"len_{max_tokens}"
                    method_path = shard_length_dir / f"{method}_traces.jsonl"
                    if method_path.exists():
                        traces.extend(read_jsonl(method_path))
                        found_any = True
                    skipped_path = shard_length_dir / f"{method}_skipped_problems.jsonl"
                    if skipped_path.exists():
                        skipped_problem_rows.extend(read_jsonl(skipped_path))
                if not found_any:
                    if method in requested_methods:
                        raise FileNotFoundError(
                            f"Missing {method} shard artifacts for dataset={dataset_key} max_tokens={max_tokens}."
                        )
                    continue
                traces = sort_traces(traces)
                skipped_problem_rows = dedupe_skipped_problem_rows(skipped_problem_rows)
                method_metrics = compute_metrics(
                    traces,
                    total_problem_count=len(problems),
                    skipped_problems=skipped_problem_rows,
                )
                write_jsonl(length_dir / f"{method}_traces.jsonl", traces)
                write_jsonl(length_dir / f"{method}_skipped_problems.jsonl", skipped_problem_rows)
                write_json(length_dir / f"{method}_metrics.json", method_metrics)
                merged_metrics[method] = method_metrics
                if method == "baseline":
                    baseline_traces = traces
                else:
                    iterative_traces = traces

            write_json(length_dir / "metrics.json", merged_metrics)
            (length_dir / "examples.md").write_text(
                build_examples_markdown(
                    problems,
                    baseline_traces,
                    iterative_traces,
                    available_methods=ordered_methods(list(merged_metrics)),
                ),
                encoding="utf-8",
            )
            merged_results_by_dataset[dataset_key][max_tokens] = merged_metrics

    merged_run_config = dict(shard_run_configs[0])
    merged_run_config["merge_only"] = True
    merged_run_config["num_shards"] = args.num_shards
    merged_run_config.pop("shard_idx", None)
    write_json(out_dir / "run_config.json", merged_run_config)
    write_json(
        out_dir / "server_info.json",
        {
            "backend": "merged_shards",
            "num_shards": args.num_shards,
            "source_shards": [str(path) for path in shard_dirs],
        },
    )

    shard_wall_clock_seconds = [
        float(overview.get("wall_clock_seconds", 0.0)) for overview in shard_overviews if overview is not None
    ]
    overview = {
        "results_by_dataset": merged_results_by_dataset,
        "dataset_metadata": merged_dataset_metadata,
        "wall_clock_seconds": max(shard_wall_clock_seconds) if shard_wall_clock_seconds else 0.0,
        "merge_wall_clock_seconds": time.time() - merge_start,
    }
    write_json(out_dir / "dataset_metadata.json", merged_dataset_metadata)
    write_json(out_dir / "results_overview.json", overview)
    (out_dir / "results_overview.md").write_text(
        build_results_markdown(merged_results_by_dataset, merged_dataset_metadata, args.attempts),
        encoding="utf-8",
    )
    print(json.dumps(overview, indent=2), flush=True)


async def async_main(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_idx < args.num_shards):
        raise ValueError("--shard-idx must satisfy 0 <= shard_idx < num_shards")

    out_dir = ensure_out_dir(args)
    dataset_load_results: dict[str, DatasetLoadResult] = {}
    dataset_metadata: dict[str, dict[str, Any]] = {}
    for dataset_key in args.datasets:
        load_result = load_problem_set(dataset_key, args.limit)
        if not load_result.problems:
            raise RuntimeError(f"No problems loaded for dataset '{dataset_key}'.")
        dataset_dir = out_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_load_results[dataset_key] = load_result
        dataset_metadata[dataset_key] = {
            "display_name": load_result.spec.display_name,
            "hf_dataset": load_result.spec.hf_dataset,
            "config": load_result.spec.config,
            "split": load_result.spec.split,
            "raw_count": load_result.raw_count,
            "filtered_count": load_result.filtered_count,
            "num_problems": len(load_result.problems),
            "question_field": load_result.spec.question_field,
            "ground_truth_field": load_result.spec.ground_truth_field,
            "score_kind": load_result.spec.score_kind,
            "tolerance_field": load_result.spec.tolerance_field,
        }
        if load_result.subset_metadata:
            dataset_metadata[dataset_key]["subset_metadata"] = load_result.subset_metadata
        if args.num_shards > 1:
            sharded_problems = [
                problem for problem in load_result.problems if (problem.row_index % args.num_shards) == args.shard_idx
            ]
            dataset_load_results[dataset_key] = DatasetLoadResult(
                spec=load_result.spec,
                problems=sharded_problems,
                raw_count=load_result.raw_count,
                filtered_count=load_result.filtered_count,
                subset_metadata=load_result.subset_metadata,
            )
            dataset_metadata[dataset_key]["num_problems_total"] = len(load_result.problems)
            dataset_metadata[dataset_key]["num_problems_shard"] = len(sharded_problems)
            dataset_metadata[dataset_key]["num_shards"] = args.num_shards
            dataset_metadata[dataset_key]["shard_idx"] = args.shard_idx
        save_problem_manifest(dataset_load_results[dataset_key].problems, dataset_dir)

    config_payload = {
        "model": args.model,
        "datasets": list(args.datasets),
        "limit": args.limit,
        "attempts": args.attempts,
        "solve_max_tokens": list(args.solve_max_tokens),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "summary_max_tokens": args.summary_max_tokens,
        "summary_temperature": args.summary_temperature,
        "summary_top_p": args.summary_top_p,
        "base_seed": args.base_seed,
        "concurrency": args.concurrency,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpus_per_node": args.gpus_per_node,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "server_prompt_length": args.server_prompt_length,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
    }
    write_json(out_dir / "run_config.json", config_payload)
    write_json(out_dir / "dataset_metadata.json", dataset_metadata)

    context: Optional[GenerationContext] = None
    results_by_dataset: dict[str, dict[int, dict[str, Any]]] = {}
    start_time = time.time()
    try:
        context = await start_server(args)
        write_json(
            out_dir / "server_info.json",
            {
                "backend": "direct_vllm",
                "model": args.model,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        )
        print(f"[backend] direct_vllm model={args.model}", flush=True)
        for dataset_key in args.datasets:
            dataset_dir = out_dir / dataset_key
            problems = dataset_load_results[dataset_key].problems
            results_by_dataset[dataset_key] = {}
            print(f"[dataset] {dataset_key} rows={len(problems)}", flush=True)
            for max_tokens in args.solve_max_tokens:
                print(f"[run] dataset={dataset_key} max_tokens={max_tokens}", flush=True)
                results_by_dataset[dataset_key][max_tokens] = await run_one_length(
                    problems,
                    context=context,
                    args=args,
                    max_tokens=max_tokens,
                    out_dir=dataset_dir,
                )
    finally:
        if context is not None:
            await stop_server(context)

    total_seconds = time.time() - start_time
    overview = {
        "results_by_dataset": results_by_dataset,
        "dataset_metadata": dataset_metadata,
        "wall_clock_seconds": total_seconds,
    }
    write_json(out_dir / "results_overview.json", overview)
    (out_dir / "results_overview.md").write_text(
        build_results_markdown(results_by_dataset, dataset_metadata, args.attempts),
        encoding="utf-8",
    )
    print(json.dumps(overview, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_shard_outputs(args, ensure_out_dir(args))
        return
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
