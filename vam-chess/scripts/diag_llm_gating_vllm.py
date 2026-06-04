#!/usr/bin/env python3
"""Local judge-model gating sanity harness for chess selection traces.

This script is intentionally local-only and trainer-independent. It builds two
paired buckets from full-legal selection rows:

- `clean`: traces generated from the normal full-action-space prompt.
- `suspicious`: traces generated from a candidate-set size 1 prompt for the
  same row, while the judge is asked to assess plausibility under the original
  full-action-space context only.

The judge is not asked to score chess quality. Its task is only to decide
whether the response looks plausibly self-contained under the visible context
`x`, or whether it appears to depend on a hidden hint `h`.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import pyarrow.parquet as pq
from jinja2 import Environment
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.chess.reward_fn import compute_score

_THINK_TAG_RE = re.compile(r"<\s*think\s*>(?P<body>[\s\S]*?)<\s*/\s*think\s*>", re.IGNORECASE)
EXPLICIT_SINGLETON_HINT_PATTERNS = [
    re.compile(r"\bthe only allowed move\b", re.IGNORECASE),
    re.compile(r"\bonly allowed move\b", re.IGNORECASE),
    re.compile(r"\bthe allowed move is\s+[a-h][1-8][a-h][1-8][qrbn]?\b", re.IGNORECASE),
    re.compile(r"\ballowed moves?\s+(?:are|is)\s+limited\s+to\b", re.IGNORECASE),
    re.compile(
        r"\ballowed moves\b[^\n]{0,120}\b[a-h][1-8][a-h][1-8][qrbn]?\b\s+is the only (?:move|one)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bit is the only move in the allowed list\b", re.IGNORECASE),
    re.compile(r"\bis the only move in the allowed list\b", re.IGNORECASE),
    re.compile(r"\bthe only move in the allowed list\b", re.IGNORECASE),
    re.compile(r"\bit is the only one in the allowed list\b", re.IGNORECASE),
    re.compile(r"\bis the only one in the allowed list\b", re.IGNORECASE),
    re.compile(r"\bthe only one in the allowed list\b", re.IGNORECASE),
    re.compile(r"\bno other move in the allowed moves? list\b", re.IGNORECASE),
    re.compile(r"\bno other moves? in the allowed moves? list\b", re.IGNORECASE),
    re.compile(
        r"\bthe allowed move\s+[a-h][1-8][a-h][1-8][qrbn]?\s+is the only one in the list\b",
        re.IGNORECASE,
    ),
]

DEFAULT_PARQUETS = ["data/chess_puzzles_chessr1_aligned_sharded_ours/test.parquet"]
DEFAULT_TEMPLATE = "recipe/chess/prompt_templates/select_prompt.jinja"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

JUDGE_PROMPT = """You are filtering reasoning traces for training.

At student test time, the student sees only the visible context `x`, not the hidden hint `h`
(for example a reduced candidate set, legality filter, or other privileged restriction).

Judge whether `y` still looks like a plausible, self-contained response from a strong model that saw only `x`.
Do not judge chess quality or move correctness.

Important: false positives on clean traces are much worse than false negatives.
If you are unsure, ambiguous, or `y` could plausibly have been written from `x` alone, ACCEPT.

Use `x` as the authoritative visible context. Use `h` only to detect hidden-hint dependence.
Reject only for clear evidence that `y` depends on `h`.

Crucial distinction:
- It is usually ACCEPT if `y` simply discusses one chosen candidate from the visible list, even with wording like
  "the allowed move e1g3 is strong", "from the allowed list, e1g3 is best", or "the best move from the allowed
  moves is e1g3". Mentioning one move from the visible list is not enough for REJECT.
- It is also usually ACCEPT if `y` compares that move against other candidates on normal chess grounds, for
  example "other moves like g8g7 or g8g6 are more direct but less strategic". Strategic comparison is fine.
- It is usually REJECT if `y` describes the visible allowed-move list itself as smaller than it is in `x`, or if
  it treats the hidden shortlist in `h` as though it were the visible list.

Usually REJECT if `x` lists many candidates but `y` says or clearly implies things like:
- there was only one allowed/candidate/legal move;
- the allowed list was limited, short, narrow, or focused on just a few moves;
- the allowed moves were exactly a small set such as "e1e2 and f1f2";
- some other move was not in the allowed list;
- some move "is not an option";
- "the only allowed move is e2e4";
- "the allowed moves are e2e4 and g1f3";
- "the allowed moves are very limited";
- "other moves are not in the allowed list".

Do not reject merely because:
- `y` is short, confident, efficient, or omits alternatives;
- `y` names one move from the visible list as the best allowed move;
- `y` says "the allowed move X is strong" or "the move X from the allowed list is best" without making claims
  about what else was or was not on the list;
- the final move also appears in `h`;
- the final move is unusual, wrong, or invalid under `x`.

The final move alone is almost never enough evidence for REJECT.
If `x` lists many moves and `y` simply argues for one move on normal chess grounds without asserting that the
visible list was small, restricted, or enumerated by `h`, that is usually ACCEPT.

Examples:
- ACCEPT: "The allowed move c1c8 is a promotion move and the best move from the allowed list."
- ACCEPT: "Other moves like g8g7 or g8g6 are more direct, but e1g3 has better long-term value."
- REJECT: if `x` lists many candidates but `y` says "the allowed moves are f4e4 and f4d4."
- REJECT: if `x` lists many candidates but `y` says "the allowed moves are very limited, focusing on a single
  pawn move."
- REJECT: if `x` lists many candidates but `y` says "the allowed move h7h8 is not an option."

Return exactly one JSON object with:
- `verdict`: `ACCEPT` or `REJECT`
- `confidence`: `high`, `medium`, or `low`
- `reason`: a short explanation focused on hint dependence vs plausibility under `x`
"""

JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["ACCEPT", "REJECT"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class SourceRow:
    row_uid: str
    row_index: Optional[int]
    source_parquet: str
    row_in_parquet: int
    fen: str
    legal_moves_uci: list[str]
    considered_moves_uci: list[str]
    ground_truth: str
    mu_map: dict[str, float]
    reward_model: dict[str, Any]


@dataclass(frozen=True)
class Case:
    case_id: str
    pair_id: str
    bucket: str
    row_uid: str
    row_index: Optional[int]
    source_parquet: str
    row_in_parquet: int
    fen: str
    legal_moves_uci: list[str]
    visible_considered_moves_uci: list[str]
    generation_considered_moves_uci: list[str]
    visible_prompt_text: str
    generation_prompt_text: str
    singleton_move_uci: Optional[str]
    ground_truth: str
    visible_target_move: str
    generation_target_move: str
    reward_model_for_generation: dict[str, Any]
    extra_info_for_generation: dict[str, Any]


def _normalize_uci(move: Any) -> str:
    return str(move or "").strip().lower()


def _normalize_moves(moves: Any) -> list[str]:
    if moves is None:
        return []
    if isinstance(moves, str):
        s = _normalize_uci(moves)
        return [s] if s else []
    out: list[str] = []
    try:
        for move in moves:
            s = _normalize_uci(move)
            if s:
                out.append(s)
    except Exception:
        return []
    return out


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_move_map(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}
    obj = raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
        except Exception:
            return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in obj.items():
        move = _normalize_uci(key)
        if not move:
            continue
        try:
            score = float(value)
        except Exception:
            continue
        if math.isfinite(score):
            out[move] = score
    return out


def _best_move_by_mu(mu_map: dict[str, float], moves: list[str]) -> str:
    best_move = ""
    best_value = -float("inf")
    for move in moves:
        key = _normalize_uci(move)
        value = float(mu_map.get(key, -float("inf")))
        if (value > best_value) or (value == best_value and (not best_move or key < best_move)):
            best_move = key
            best_value = value
    if not best_move:
        raise ValueError("Empty move list when selecting μ-best move.")
    return best_move


def _render_prompt(template: Any, *, fen: str, legal_moves: list[str], considered_moves: list[str]) -> str:
    return str(
        template.render(
            FEN=fen,
            legal_moves_uci_list=legal_moves,
            considered_moves_uci_list=considered_moves,
        )
    )


def _load_template(template_path: str) -> Any:
    template_file = Path(template_path)
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")
    env = Environment(autoescape=False)
    return env.from_string(template_file.read_text(encoding="utf-8"))


def _to_optional_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _stable_int_hash(*parts: str, mod: int) -> int:
    h = hashlib.sha256()
    for part in parts:
        h.update((part or "").encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], "big") % int(mod)


def _json_safe(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return _json_safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False, sort_keys=False)
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _iter_batches(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _messages_to_plaintext_prompt(messages: list[dict[str, str]]) -> str:
    if len(messages) == 1 and str(messages[0].get("role") or "").strip().lower() == "user":
        return str(messages[0].get("content") or "")

    rendered_parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().upper() or "USER"
        content = str(message.get("content") or "")
        rendered_parts.append(f"{role}:\n{content}".rstrip())
    return "\n\n".join(rendered_parts) + "\n\nASSISTANT:\n"


def _messages_to_token_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    use_chat_template: bool,
) -> list[int]:
    if use_chat_template:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            raise TypeError(f"Expected token id list from apply_chat_template, got {type(ids)}")
        return [int(x) for x in ids]

    prompt_text = _messages_to_plaintext_prompt(messages)
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not isinstance(ids, list):
        raise TypeError(f"Expected token id list from tokenizer.encode, got {type(ids)}")
    return [int(x) for x in ids]


def _load_source_rows(
    *,
    parquet_paths: list[Path],
    template: Any,
    tokenizer: Any,
    use_chat_template: bool,
    max_prompt_tokens: int,
    limit_rows: int,
    row_seed: int,
) -> tuple[list[SourceRow], dict[str, Any]]:
    all_rows: list[SourceRow] = []
    stats = {
        "rows_total_seen": 0,
        "rows_loaded": 0,
        "rows_skipped_missing": 0,
        "rows_skipped_prompt_too_long": 0,
        "rows_skipped_considered_not_full_legal": 0,
        "rows_skipped_too_few_legal_moves": 0,
    }

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=["reward_model", "extra_info"])
        reward_models = table.column("reward_model").to_pylist()
        extra_infos = table.column("extra_info").to_pylist()
        for row_in_parquet, (rm_raw, ei_raw) in enumerate(zip(reward_models, extra_infos, strict=True)):
            stats["rows_total_seen"] += 1
            rm = rm_raw if isinstance(rm_raw, dict) else {}
            ei = ei_raw if isinstance(ei_raw, dict) else {}

            fen = str(rm.get("fen") or "").strip()
            ground_truth = _normalize_uci(rm.get("ground_truth"))
            legal_moves = _dedupe_preserve_order(_normalize_moves(rm.get("legal_moves_uci")))
            considered_moves = _dedupe_preserve_order(_normalize_moves(rm.get("considered_moves_uci")))
            if not fen or not ground_truth or not legal_moves:
                stats["rows_skipped_missing"] += 1
                continue
            if len(legal_moves) < 2:
                stats["rows_skipped_too_few_legal_moves"] += 1
                continue
            if considered_moves and considered_moves != legal_moves:
                stats["rows_skipped_considered_not_full_legal"] += 1
                continue

            mu_map = _parse_move_map(rm.get("move_expected_scores_json"))
            if not mu_map:
                mu_map = _parse_move_map(rm.get("move_values_json"))
            if not mu_map:
                stats["rows_skipped_missing"] += 1
                continue

            prompt_text = _render_prompt(
                template,
                fen=fen,
                legal_moves=legal_moves,
                considered_moves=legal_moves,
            )
            token_ids = _messages_to_token_ids(
                tokenizer,
                [{"role": "user", "content": prompt_text}],
                use_chat_template=use_chat_template,
            )
            if len(token_ids) > max_prompt_tokens:
                stats["rows_skipped_prompt_too_long"] += 1
                continue

            row_uid = f"{parquet_path.name}:{row_in_parquet}"
            all_rows.append(
                SourceRow(
                    row_uid=row_uid,
                    row_index=_to_optional_int(ei.get("index")),
                    source_parquet=str(parquet_path),
                    row_in_parquet=int(row_in_parquet),
                    fen=fen,
                    legal_moves_uci=list(legal_moves),
                    considered_moves_uci=list(legal_moves),
                    ground_truth=ground_truth,
                    mu_map=dict(mu_map),
                    reward_model={
                        "fen": fen,
                        "ground_truth": ground_truth,
                        "legal_moves_uci": list(legal_moves),
                        "considered_moves_uci": list(legal_moves),
                        "move_expected_scores_json": rm.get("move_expected_scores_json"),
                        "move_values_json": rm.get("move_values_json"),
                    },
                )
            )
            stats["rows_loaded"] += 1

    if not all_rows:
        raise SystemExit("No source rows loaded after filtering.")

    rng = random.Random(int(row_seed))
    rng.shuffle(all_rows)
    selected = all_rows[: min(len(all_rows), int(limit_rows))]
    stats["rows_selected_after_sampling"] = len(selected)
    stats["row_seed"] = int(row_seed)
    return selected, stats


def _choose_singleton_move(row: SourceRow, *, singleton_strategy: str, seed: int) -> str:
    moves = list(row.legal_moves_uci)
    if not moves:
        raise ValueError(f"Row {row.row_uid}: empty legal move list.")

    strategy = singleton_strategy.strip().lower()
    if strategy == "random_legal":
        idx = _stable_int_hash(str(seed), row.row_uid, mod=len(moves))
        return moves[idx]
    if strategy == "ground_truth":
        return row.ground_truth
    if strategy == "mu_best":
        return _best_move_by_mu(row.mu_map, moves)
    raise ValueError(f"Unknown singleton strategy: {singleton_strategy!r}")


def _build_cases(rows: list[SourceRow], *, template: Any, singleton_strategy: str, seed: int) -> list[Case]:
    cases: list[Case] = []
    for row in rows:
        pair_id = row.row_uid
        visible_prompt_text = _render_prompt(
            template,
            fen=row.fen,
            legal_moves=row.legal_moves_uci,
            considered_moves=row.legal_moves_uci,
        )
        visible_target = _best_move_by_mu(row.mu_map, row.legal_moves_uci)

        clean_reward_model = dict(row.reward_model)
        clean_reward_model["considered_moves_uci"] = list(row.legal_moves_uci)
        clean_extra = {"use_considered_moves_uci": True}
        cases.append(
            Case(
                case_id=f"{pair_id}:clean",
                pair_id=pair_id,
                bucket="clean",
                row_uid=row.row_uid,
                row_index=row.row_index,
                source_parquet=row.source_parquet,
                row_in_parquet=row.row_in_parquet,
                fen=row.fen,
                legal_moves_uci=list(row.legal_moves_uci),
                visible_considered_moves_uci=list(row.legal_moves_uci),
                generation_considered_moves_uci=list(row.legal_moves_uci),
                visible_prompt_text=visible_prompt_text,
                generation_prompt_text=visible_prompt_text,
                singleton_move_uci=None,
                ground_truth=row.ground_truth,
                visible_target_move=visible_target,
                generation_target_move=visible_target,
                reward_model_for_generation=clean_reward_model,
                extra_info_for_generation=clean_extra,
            )
        )

        singleton_move = _choose_singleton_move(row, singleton_strategy=singleton_strategy, seed=seed)
        suspicious_reward_model = dict(row.reward_model)
        suspicious_reward_model["considered_moves_uci"] = [singleton_move]
        suspicious_prompt_text = _render_prompt(
            template,
            fen=row.fen,
            legal_moves=row.legal_moves_uci,
            considered_moves=[singleton_move],
        )
        suspicious_extra = {"use_considered_moves_uci": True}
        cases.append(
            Case(
                case_id=f"{pair_id}:suspicious:{singleton_move}",
                pair_id=pair_id,
                bucket="suspicious",
                row_uid=row.row_uid,
                row_index=row.row_index,
                source_parquet=row.source_parquet,
                row_in_parquet=row.row_in_parquet,
                fen=row.fen,
                legal_moves_uci=list(row.legal_moves_uci),
                visible_considered_moves_uci=list(row.legal_moves_uci),
                generation_considered_moves_uci=[singleton_move],
                visible_prompt_text=visible_prompt_text,
                generation_prompt_text=suspicious_prompt_text,
                singleton_move_uci=singleton_move,
                ground_truth=row.ground_truth,
                visible_target_move=visible_target,
                generation_target_move=singleton_move,
                reward_model_for_generation=suspicious_reward_model,
                extra_info_for_generation=suspicious_extra,
            )
        )

    return cases


def _build_prompt_token_ids(
    tokenizer: Any,
    messages_list: list[list[dict[str, str]]],
    *,
    use_chat_template: bool,
    max_prompt_tokens: int,
) -> list[list[int]]:
    prompt_token_ids: list[list[int]] = []
    for messages in messages_list:
        ids = _messages_to_token_ids(
            tokenizer,
            messages,
            use_chat_template=use_chat_template,
        )
        if len(ids) > max_prompt_tokens:
            raise ValueError(f"Prompt is {len(ids)} tokens (max={max_prompt_tokens}).")
        prompt_token_ids.append(ids)
    return prompt_token_ids


def _extract_think_text(response_text: str) -> str:
    match = _THINK_TAG_RE.search(response_text or "")
    if not match:
        return ""
    return str(match.group("body") or "").strip()


def _judge_input_payload(case: Case, generation_record: dict[str, Any]) -> dict[str, Any]:
    if case.bucket == "clean":
        hidden_hint: Any = None
    else:
        hidden_hint = {
            "type": "restricted_candidate_set",
            "candidate_count": int(len(case.generation_considered_moves_uci)),
            "considered_moves_uci": list(case.generation_considered_moves_uci),
            "note": (
                "This restricted candidate list was visible during generation but will not "
                "be visible to the student at test time."
            ),
        }

    return {
        "x": {
            "prompt_text": case.visible_prompt_text,
            "fen": case.fen,
            "legal_moves_uci": list(case.legal_moves_uci),
            "visible_candidate_count": int(len(case.visible_considered_moves_uci)),
            "visible_candidate_moves_uci": list(case.visible_considered_moves_uci),
        },
        "h": hidden_hint,
        "y": {
            "reasoning_trace": str(generation_record.get("think_text") or generation_record["response_text"]),
            "has_explicit_think": bool(str(generation_record.get("think_text") or "").strip()),
        },
    }


def _coerce_json_object(raw_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:
        raise ValueError(f"Failed to parse judge JSON: {exc}: {raw_text!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected judge JSON object, got {type(parsed)}")
    return parsed


def _run_generation(
    *,
    llm: Any,
    tokenizer: Any,
    cases: list[Case],
    use_chat_template: bool,
    max_prompt_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    max_response_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    outputs_by_case_id: dict[str, dict[str, Any]] = {}
    base_sampling_kwargs = dict(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=-1,
        min_p=0.0,
        max_tokens=int(max_response_tokens),
        repetition_penalty=1.0,
        detokenize=True,
        stop=["</uci_move>"],
        include_stop_str_in_output=True,
    )

    for case_batch in _iter_batches(cases, batch_size):
        messages_list = [[{"role": "user", "content": case.generation_prompt_text}] for case in case_batch]
        prompt_token_ids = _build_prompt_token_ids(
            tokenizer,
            messages_list,
            use_chat_template=use_chat_template,
            max_prompt_tokens=max_prompt_tokens,
        )
        vllm_inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
        sampling_params = [
            SamplingParams(
                seed=_stable_int_hash(str(seed), case.case_id, mod=2**31 - 1),
                **base_sampling_kwargs,
            )
            for case in case_batch
        ]
        outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
        if len(outputs) != len(case_batch):
            raise RuntimeError(f"Expected {len(case_batch)} generation outputs, got {len(outputs)}")
        for case, output in zip(case_batch, outputs, strict=True):
            if len(output.outputs) != 1:
                raise RuntimeError(f"Expected exactly one sampled output for {case.case_id}, got {len(output.outputs)}")
            response_text = str(output.outputs[0].text or "")
            reward_info = compute_score(
                case.reward_model_for_generation,
                response_text,
                case.ground_truth,
                extra_info=case.extra_info_for_generation,
            )
            if not isinstance(reward_info, dict):
                raise TypeError(f"Expected dict from compute_score, got {type(reward_info)}")
            outputs_by_case_id[case.case_id] = {
                "case_id": case.case_id,
                "response_text": response_text,
                "think_text": _extract_think_text(response_text),
                "score": reward_info.get("score"),
                "acc": reward_info.get("acc"),
                "pred_move": reward_info.get("pred_move"),
                "target_move": reward_info.get("target_move"),
                "gt_uci": reward_info.get("gt_uci"),
                "in_subset": reward_info.get("in_subset"),
                "penalty_reason": reward_info.get("penalty_reason"),
                "penalty_applied": reward_info.get("penalty_applied"),
                "reward_reason": reward_info.get("reward_reason"),
                "format_reward": reward_info.get("format_reward"),
            }

    return [outputs_by_case_id[case.case_id] for case in cases]


def _run_judge(
    *,
    llm: Any,
    tokenizer: Any,
    cases: list[Case],
    generation_records: list[dict[str, Any]],
    judge_prompt: str,
    use_chat_template: bool,
    max_prompt_tokens: int,
    batch_size: int,
    judge_max_tokens: int,
) -> list[dict[str, Any]]:
    judge_outputs_by_case_id: dict[str, dict[str, Any]] = {}
    generation_by_case_id = {record["case_id"]: record for record in generation_records}
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=int(judge_max_tokens),
        detokenize=True,
        guided_decoding=GuidedDecodingParams(json=JUDGE_RESPONSE_SCHEMA),
    )

    for case_batch in _iter_batches(cases, batch_size):
        messages_list: list[list[dict[str, str]]] = []
        for case in case_batch:
            generation_record = generation_by_case_id[case.case_id]
            judge_input = _judge_input_payload(case, generation_record)
            user_text = (
                "Input JSON:\n"
                + json.dumps(_json_safe(judge_input), ensure_ascii=False, indent=2)
            )
            messages_list.append(
                [
                    {"role": "system", "content": judge_prompt},
                    {"role": "user", "content": user_text},
                ]
            )

        prompt_token_ids = _build_prompt_token_ids(
            tokenizer,
            messages_list,
            use_chat_template=use_chat_template,
            max_prompt_tokens=max_prompt_tokens,
        )
        vllm_inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
        outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
        if len(outputs) != len(case_batch):
            raise RuntimeError(f"Expected {len(case_batch)} judge outputs, got {len(outputs)}")
        for case, output in zip(case_batch, outputs, strict=True):
            if len(output.outputs) != 1:
                raise RuntimeError(f"Expected exactly one judge output for {case.case_id}, got {len(output.outputs)}")
            raw_text = str(output.outputs[0].text or "")
            parsed = _coerce_json_object(raw_text)
            verdict = str(parsed.get("verdict") or "").strip().upper()
            confidence = str(parsed.get("confidence") or "").strip().lower()
            if verdict not in {"ACCEPT", "REJECT"}:
                raise ValueError(f"Judge returned invalid verdict for {case.case_id}: {verdict!r}")
            if confidence not in {"high", "medium", "low"}:
                raise ValueError(f"Judge returned invalid confidence for {case.case_id}: {confidence!r}")
            judge_outputs_by_case_id[case.case_id] = {
                "case_id": case.case_id,
                "judge_raw_text": raw_text,
                "judge_parsed_output": parsed,
                "judge_verdict": verdict,
                "judge_confidence": confidence,
                "judge_reason": str(parsed.get("reason") or "").strip(),
            }

    return [judge_outputs_by_case_id[case.case_id] for case in cases]


def _bucket_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    reject_count = sum(1 for rec in records if rec["judge_effective_verdict"] == "REJECT")
    accept_count = total - reject_count
    confidences: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for rec in records:
        conf = str(rec.get("judge_confidence") or "").strip().lower()
        if conf in confidences:
            confidences[conf] += 1
    return {
        "n_total": int(total),
        "n_accept": int(accept_count),
        "n_reject": int(reject_count),
        "accept_rate": float(accept_count / float(max(1, total))),
        "reject_rate": float(reject_count / float(max(1, total))),
        "format_ok_rate": float(
            sum(1 for rec in records if float(rec.get("format_reward") or 0.0) >= 1.0) / float(max(1, total))
        ),
        "in_subset_rate": float(
            sum(1 for rec in records if bool(rec.get("in_subset"))) / float(max(1, total))
        ),
        "penalty_rate": float(
            sum(1 for rec in records if bool(rec.get("penalty_applied"))) / float(max(1, total))
        ),
        "think_nonempty_rate": float(
            sum(1 for rec in records if str(rec.get("y_think_text") or "").strip()) / float(max(1, total))
        ),
        "confidence_counts": confidences,
    }


def _select_representatives(records: list[dict[str, Any]], *, verdict: str, limit: int) -> list[dict[str, Any]]:
    verdict = verdict.strip().upper()
    filtered = [rec for rec in records if str(rec.get("judge_effective_verdict") or "").upper() == verdict]
    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    filtered.sort(
        key=lambda rec: (
            0 if float(rec.get("format_reward") or 0.0) >= 1.0 else 1,
            0 if bool(rec.get("in_subset")) else 1,
            0 if str(rec.get("y_think_text") or "").strip() else 1,
            confidence_rank.get(str(rec.get("judge_confidence") or "").lower(), 99),
            0 if not bool(rec.get("penalty_applied")) else 1,
            str(rec.get("case_id") or ""),
        )
    )
    return filtered[:limit]


def _explicit_singleton_contradiction(
    *,
    visible_candidate_count: int,
    response_text: str,
    think_text: str,
) -> Optional[dict[str, Any]]:
    if int(visible_candidate_count) <= 1:
        return None
    haystacks = [str(think_text or ""), str(response_text or "")]
    for pattern in EXPLICIT_SINGLETON_HINT_PATTERNS:
        for haystack in haystacks:
            match = pattern.search(haystack)
            if match:
                return {
                    "matched_text": match.group(0),
                    "pattern": pattern.pattern,
                    "reason": "explicit_singleton_claim_conflicts_with_visible_candidate_set",
                }
    return None


def _compose_case_record(case: Case, generation_record: dict[str, Any], judge_record: dict[str, Any]) -> dict[str, Any]:
    judge_input_payload = _judge_input_payload(case, generation_record)
    contradiction = _explicit_singleton_contradiction(
        visible_candidate_count=int(len(case.visible_considered_moves_uci)),
        response_text=str(generation_record["response_text"]),
        think_text=str(generation_record.get("think_text") or ""),
    )
    effective_verdict = "REJECT" if (judge_record["judge_verdict"] == "REJECT" or contradiction) else "ACCEPT"
    effective_reason = (
        str(judge_record["judge_reason"])
        if not contradiction
        else f"{contradiction['reason']}: {contradiction['matched_text']}"
    )
    return {
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "bucket": case.bucket,
        "row_uid": case.row_uid,
        "row_index": case.row_index,
        "source_parquet": case.source_parquet,
        "row_in_parquet": case.row_in_parquet,
        "fen": case.fen,
        "legal_moves_uci": list(case.legal_moves_uci),
        "visible_considered_moves_uci": list(case.visible_considered_moves_uci),
        "generation_considered_moves_uci": list(case.generation_considered_moves_uci),
        "singleton_move_uci": case.singleton_move_uci,
        "ground_truth": case.ground_truth,
        "visible_target_move": case.visible_target_move,
        "generation_target_move": case.generation_target_move,
        "visible_prompt_text": case.visible_prompt_text,
        "generation_prompt_text": case.generation_prompt_text,
        "x_visible_to_student": judge_input_payload["x"],
        "h_confidential_hint": judge_input_payload["h"],
        "y_candidate_trace": judge_input_payload["y"],
        "generation_metadata": {
            "reward_model_for_generation": case.reward_model_for_generation,
            "extra_info_for_generation": case.extra_info_for_generation,
        },
        "x_visible_prompt_text": case.visible_prompt_text,
        "h_hidden_hint": (
            None
            if case.bucket == "clean"
            else {
                "type": "restricted_candidate_set",
                "considered_moves_uci": list(case.generation_considered_moves_uci),
                "candidate_count": int(len(case.generation_considered_moves_uci)),
            }
        ),
        "y_response_text": generation_record["response_text"],
        "y_think_text": generation_record.get("think_text"),
        "score": generation_record.get("score"),
        "acc": generation_record.get("acc"),
        "pred_move": generation_record.get("pred_move"),
        "target_move": generation_record.get("target_move"),
        "gt_uci": generation_record.get("gt_uci"),
        "in_subset": generation_record.get("in_subset"),
        "penalty_reason": generation_record.get("penalty_reason"),
        "penalty_applied": generation_record.get("penalty_applied"),
        "reward_reason": generation_record.get("reward_reason"),
        "format_reward": generation_record.get("format_reward"),
        "judge_raw_text": judge_record["judge_raw_text"],
        "judge_parsed_output": judge_record.get("judge_parsed_output"),
        "judge_verdict": judge_record["judge_verdict"],
        "judge_confidence": judge_record["judge_confidence"],
        "judge_reason": judge_record["judge_reason"],
        "judge_effective_verdict": effective_verdict,
        "judge_effective_reason": effective_reason,
        "explicit_singleton_contradiction": contradiction,
    }


def _load_cases_and_generations_for_rejudge(cases_jsonl_path: Path) -> tuple[list[Case], list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(cases_jsonl_path)
    if not rows:
        raise ValueError(f"No rows found in reuse cases jsonl: {cases_jsonl_path}")

    cases: list[Case] = []
    generation_records: list[dict[str, Any]] = []
    pair_ids: set[str] = set()

    for row in rows:
        generation_meta = row.get("generation_metadata") or {}
        reward_model_for_generation = generation_meta.get("reward_model_for_generation") or {}
        extra_info_for_generation = generation_meta.get("extra_info_for_generation") or {}

        cases.append(
            Case(
                case_id=str(row["case_id"]),
                pair_id=str(row["pair_id"]),
                bucket=str(row["bucket"]),
                row_uid=str(row["row_uid"]),
                row_index=_to_optional_int(row.get("row_index")),
                source_parquet=str(row["source_parquet"]),
                row_in_parquet=int(row["row_in_parquet"]),
                fen=str(row["fen"]),
                legal_moves_uci=_dedupe_preserve_order(_normalize_moves(row.get("legal_moves_uci"))),
                visible_considered_moves_uci=_dedupe_preserve_order(
                    _normalize_moves(row.get("visible_considered_moves_uci"))
                ),
                generation_considered_moves_uci=_dedupe_preserve_order(
                    _normalize_moves(row.get("generation_considered_moves_uci"))
                ),
                visible_prompt_text=str(row["visible_prompt_text"]),
                generation_prompt_text=str(row["generation_prompt_text"]),
                singleton_move_uci=_normalize_uci(row.get("singleton_move_uci")) or None,
                ground_truth=_normalize_uci(row.get("ground_truth")),
                visible_target_move=_normalize_uci(row.get("visible_target_move")),
                generation_target_move=_normalize_uci(row.get("generation_target_move")),
                reward_model_for_generation=dict(reward_model_for_generation),
                extra_info_for_generation=dict(extra_info_for_generation),
            )
        )
        generation_records.append(
            {
                "case_id": str(row["case_id"]),
                "response_text": str(row.get("y_response_text") or ""),
                "think_text": str(row.get("y_think_text") or ""),
                "score": row.get("score"),
                "acc": row.get("acc"),
                "pred_move": row.get("pred_move"),
                "target_move": row.get("target_move"),
                "gt_uci": row.get("gt_uci"),
                "in_subset": row.get("in_subset"),
                "penalty_reason": row.get("penalty_reason"),
                "penalty_applied": row.get("penalty_applied"),
                "reward_reason": row.get("reward_reason"),
                "format_reward": row.get("format_reward"),
            }
        )
        pair_ids.add(str(row["pair_id"]))

    stats = {
        "reused_cases_jsonl": str(cases_jsonl_path),
        "n_reused_cases": len(cases),
        "n_reused_source_rows": len(pair_ids),
    }
    return cases, generation_records, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--parquets", nargs="+", default=DEFAULT_PARQUETS)
    ap.add_argument("--template_path", default=DEFAULT_TEMPLATE)
    ap.add_argument("--reuse_cases_jsonl", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit_rows", type=int, default=32)
    ap.add_argument("--row_seed", type=int, default=0)
    ap.add_argument("--singleton_strategy", choices=["random_legal", "ground_truth", "mu_best"], default="mu_best")
    ap.add_argument("--max_prompt_tokens", type=int, default=4096)
    ap.add_argument("--max_response_tokens", type=int, default=768)
    ap.add_argument("--judge_max_tokens", type=int, default=192)
    ap.add_argument("--generation_batch_size", type=int, default=24)
    ap.add_argument("--judge_batch_size", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.82)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--no_use_chat_template", action="store_true", default=False)
    ap.add_argument("--judge_prompt_path", default=None)
    ap.add_argument("--overwrite", action="store_true", default=False)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.overwrite):
        raise SystemExit(f"Output directory {out_dir} is not empty. Re-run with --overwrite to replace artifacts.")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    cases_path = out_dir / "cases.jsonl"
    representatives_path = out_dir / "representatives.json"
    config_path = out_dir / "config.json"
    judge_prompt_path = out_dir / "judge_prompt.txt"
    for path in [summary_path, cases_path, representatives_path, config_path, judge_prompt_path]:
        if path.exists():
            path.unlink()

    reuse_cases_jsonl = Path(str(args.reuse_cases_jsonl)) if args.reuse_cases_jsonl else None
    if reuse_cases_jsonl is not None and not reuse_cases_jsonl.exists():
        raise SystemExit(f"Missing reuse cases jsonl: {reuse_cases_jsonl}")
    judge_prompt_override = Path(str(args.judge_prompt_path)) if args.judge_prompt_path else None
    if judge_prompt_override is not None and not judge_prompt_override.exists():
        raise SystemExit(f"Missing judge prompt path: {judge_prompt_override}")

    parquet_paths = [Path(p) for p in args.parquets]
    if reuse_cases_jsonl is None:
        for parquet_path in parquet_paths:
            if not parquet_path.exists():
                raise SystemExit(f"Missing parquet: {parquet_path}")

    template = _load_template(str(args.template_path))
    tokenizer_name = str(args.tokenizer) if args.tokenizer else str(args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_chat_template = not bool(args.no_use_chat_template)
    judge_prompt_text = (
        judge_prompt_override.read_text(encoding="utf-8") if judge_prompt_override is not None else JUDGE_PROMPT
    )

    generation_records: list[dict[str, Any]] | None = None
    if reuse_cases_jsonl is None:
        source_rows, load_stats = _load_source_rows(
            parquet_paths=parquet_paths,
            template=template,
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
            max_prompt_tokens=int(args.max_prompt_tokens),
            limit_rows=int(args.limit_rows),
            row_seed=int(args.row_seed),
        )
        cases = _build_cases(
            source_rows,
            template=template,
            singleton_strategy=str(args.singleton_strategy),
            seed=int(args.seed),
        )
        n_source_rows = len(source_rows)
        print(
            f"[INIT] source_rows={n_source_rows} cases={len(cases)} "
            f"model={args.model} template={args.template_path}",
            flush=True,
        )
    else:
        cases, generation_records, load_stats = _load_cases_and_generations_for_rejudge(reuse_cases_jsonl)
        n_source_rows = int(load_stats["n_reused_source_rows"])
        print(
            f"[INIT] reusing_cases_jsonl={reuse_cases_jsonl} source_rows={n_source_rows} "
            f"cases={len(cases)} model={args.model}",
            flush=True,
        )

    config = {
        "model": str(args.model),
        "generation_model": str(args.model),
        "judge_model": str(args.model),
        "tokenizer": tokenizer_name,
        "reuse_cases_jsonl": str(reuse_cases_jsonl) if reuse_cases_jsonl is not None else None,
        "parquets": [str(p) for p in parquet_paths],
        "template_path": str(args.template_path),
        "limit_rows": int(args.limit_rows),
        "row_seed": int(args.row_seed),
        "singleton_strategy": str(args.singleton_strategy),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "max_response_tokens": int(args.max_response_tokens),
        "judge_max_tokens": int(args.judge_max_tokens),
        "generation_batch_size": int(args.generation_batch_size),
        "judge_batch_size": int(args.judge_batch_size),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "seed": int(args.seed),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "max_num_seqs": int(args.max_num_seqs),
        "use_chat_template": bool(use_chat_template),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "judge_prompt_path": str(judge_prompt_override) if judge_prompt_override is not None else None,
        "judge_prompt": judge_prompt_text,
        "judge_response_schema": JUDGE_RESPONSE_SCHEMA,
        "load_stats": load_stats,
        "n_source_rows": int(n_source_rows),
        "n_cases": len(cases),
    }
    _write_json(config_path, config)
    judge_prompt_path.write_text(judge_prompt_text, encoding="utf-8")

    t0 = time.time()
    llm = LLM(
        model=str(args.model),
        tokenizer=tokenizer_name,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        seed=int(args.seed),
        max_num_seqs=int(args.max_num_seqs),
    )

    if generation_records is None:
        print("[GEN] starting generation", flush=True)
        generation_records = _run_generation(
            llm=llm,
            tokenizer=tokenizer,
            cases=cases,
            use_chat_template=use_chat_template,
            max_prompt_tokens=int(args.max_prompt_tokens),
            batch_size=int(args.generation_batch_size),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_response_tokens=int(args.max_response_tokens),
            seed=int(args.seed),
        )
        print(f"[GEN] completed generation_records={len(generation_records)}", flush=True)
    else:
        print(f"[GEN] reusing generation_records={len(generation_records)} from {reuse_cases_jsonl}", flush=True)
    print("[JUDGE] starting judge pass", flush=True)
    judge_records = _run_judge(
        llm=llm,
        tokenizer=tokenizer,
        cases=cases,
        generation_records=generation_records,
        judge_prompt=judge_prompt_text,
        use_chat_template=use_chat_template,
        max_prompt_tokens=int(args.max_prompt_tokens),
        batch_size=int(args.judge_batch_size),
        judge_max_tokens=int(args.judge_max_tokens),
    )
    print(f"[JUDGE] completed judge_records={len(judge_records)}", flush=True)

    generation_by_case_id = {rec["case_id"]: rec for rec in generation_records}
    judge_by_case_id = {rec["case_id"]: rec for rec in judge_records}
    all_case_records = [
        _compose_case_record(case, generation_by_case_id[case.case_id], judge_by_case_id[case.case_id])
        for case in cases
    ]
    _write_jsonl(cases_path, all_case_records)

    clean_records = [rec for rec in all_case_records if rec["bucket"] == "clean"]
    suspicious_records = [rec for rec in all_case_records if rec["bucket"] == "suspicious"]

    summary = {
        "model": str(args.model),
        "generation_model": str(args.model),
        "judge_model": str(args.model),
        "reuse_cases_jsonl": str(reuse_cases_jsonl) if reuse_cases_jsonl is not None else None,
        "template_path": str(args.template_path),
        "parquets": [str(p) for p in parquet_paths],
        "elapsed_seconds": float(time.time() - t0),
        "n_source_rows": int(n_source_rows),
        "n_clean_cases": len(clean_records),
        "n_suspicious_cases": len(suspicious_records),
        "clean_bucket": _bucket_summary(clean_records),
        "suspicious_bucket": _bucket_summary(suspicious_records),
        "clean_false_positives": [
            rec["case_id"] for rec in clean_records if rec["judge_effective_verdict"] == "REJECT"
        ],
        "representative_case_ids": {
            "clean_accepts": [rec["case_id"] for rec in _select_representatives(clean_records, verdict="ACCEPT", limit=3)],
            "clean_rejects": [rec["case_id"] for rec in _select_representatives(clean_records, verdict="REJECT", limit=3)],
            "suspicious_rejects": [rec["case_id"] for rec in _select_representatives(suspicious_records, verdict="REJECT", limit=5)],
            "suspicious_accepts": [rec["case_id"] for rec in _select_representatives(suspicious_records, verdict="ACCEPT", limit=3)],
        },
        "reject_rate_gap": float(
            _bucket_summary(suspicious_records)["reject_rate"] - _bucket_summary(clean_records)["reject_rate"]
        ),
    }
    _write_json(summary_path, summary)

    representatives = {
        "clean_accepts": _select_representatives(clean_records, verdict="ACCEPT", limit=3),
        "clean_rejects": _select_representatives(clean_records, verdict="REJECT", limit=3),
        "suspicious_rejects": _select_representatives(suspicious_records, verdict="REJECT", limit=5),
        "suspicious_accepts": _select_representatives(suspicious_records, verdict="ACCEPT", limit=3),
    }
    _write_json(representatives_path, representatives)

    clean_reject_rate = summary["clean_bucket"]["reject_rate"]
    suspicious_reject_rate = summary["suspicious_bucket"]["reject_rate"]
    print(
        "[DONE] "
        f"clean_reject_rate={clean_reject_rate:.3f} "
        f"suspicious_reject_rate={suspicious_reject_rate:.3f} "
        f"clean_rejects={summary['clean_bucket']['n_reject']}/{summary['clean_bucket']['n_total']} "
        f"suspicious_rejects={summary['suspicious_bucket']['n_reject']}/{summary['suspicious_bucket']['n_total']}"
    )
    print(f"[ARTIFACTS] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
