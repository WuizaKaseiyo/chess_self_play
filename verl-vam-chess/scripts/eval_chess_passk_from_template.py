#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
from jinja2 import Environment
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from recipe.chess.reward_fn import compute_score_batch


@dataclass(frozen=True)
class EvalDefaults:
    # Hard requirements from the task prompt.
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    parquet: str = "data/chess_puzzles/test.parquet"
    k: int = 32
    max_response_tokens: int = 2000

    # Sensible defaults; can be overridden via env vars (no extra CLI flags).
    max_prompt_tokens: int = 1024
    max_model_len: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    seed: int = 0
    seed_mode: str = "per_prompt"  # deterministic across batch sizing
    batch_size: int = 32
    max_num_seqs: int = 1024
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8


def _die(msg: str, *, code: int = 2) -> None:
    raise SystemExit(msg if msg.endswith("\n") else msg + "\n")


def _read_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception as exc:
        raise ValueError(f"Invalid {name}={raw!r} (expected int)") from exc


def _read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception as exc:
        raise ValueError(f"Invalid {name}={raw!r} (expected float)") from exc


def _load_rows(parquet_path: str, limit: int | None) -> list[dict[str, Any]]:
    dataset = ds.dataset(parquet_path, format="parquet")
    table = dataset.to_table(columns=["reward_model"])
    rows = table.to_pylist()
    if limit is not None:
        return rows[:limit]
    return rows


def _parse_json_maps_in_reward_model(rm: dict[str, Any]) -> dict[str, Any]:
    # Avoid parsing the same JSON blobs once per completion (pass@k can be 32k+ completions).
    out = dict(rm)
    for key in ("move_values_json", "move_cps_json", "move_expected_scores_json"):
        v = out.get(key)
        if isinstance(v, str) and v:
            try:
                parsed = json.loads(v)
            except Exception:
                continue
            if isinstance(parsed, dict):
                out[key] = parsed
    return out


def _sanitize_filename_component(s: str) -> str:
    # Keep it simple and portable: [A-Za-z0-9._-]
    out = []
    for ch in (s or ""):
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("._-")
    return cleaned or "template"


def _normalize_legal_moves(legal_moves_uci: Any) -> list[str]:
    if legal_moves_uci is None:
        return []
    if isinstance(legal_moves_uci, str):
        s = legal_moves_uci.strip()
        return [s] if s else []
    try:
        return [str(m).strip() for m in legal_moves_uci if str(m).strip()]
    except Exception:
        return []


def _load_template(template_path: str) -> Any:
    template_file = Path(template_path)
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")
    env = Environment(autoescape=False)
    return env.from_string(template_file.read_text())


def _render_prompt_messages(template: Any, rm: dict[str, Any]) -> list[dict[str, str]]:
    fen = str(rm.get("fen") or "").strip()
    legal = _normalize_legal_moves(rm.get("legal_moves_uci"))
    prompt_text = str(template.render(FEN=fen, legal_moves_uci_list=legal))
    return [{"role": "user", "content": prompt_text}]


def _build_prompt_token_ids(tokenizer: Any, messages_batch: list[list[dict[str, str]]], *, max_prompt_tokens: int) -> list[list[int]]:
    prompt_token_ids: list[list[int]] = []
    for i, messages in enumerate(messages_batch):
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            raise TypeError(f"apply_chat_template returned {type(ids)} for prompt {i}")
        if len(ids) > max_prompt_tokens:
            raise ValueError(f"Prompt {i} is {len(ids)} tokens (max={max_prompt_tokens}).")
        prompt_token_ids.append([int(x) for x in ids])
    return prompt_token_ids


def _iter_batches(n: int, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, n, batch_size):
        yield start, min(n, start + batch_size)


def main() -> None:
    # Requirement: accept exactly one CLI argument (template path).
    import sys

    if len(sys.argv) != 2:
        _die("Usage: python3 -m scripts.eval_chess_passk_from_template <PROMPT_TEMPLATE_PATH>")
    template_path = str(sys.argv[1]).strip()
    if not template_path:
        _die("Template path is empty.")

    d = EvalDefaults()

    limit = _read_env_int("CHESS_EVAL_LIMIT", 0) or None
    batch_size = _read_env_int("CHESS_EVAL_BATCH_SIZE", d.batch_size)
    max_num_seqs = _read_env_int("CHESS_EVAL_MAX_NUM_SEQS", d.max_num_seqs)
    tensor_parallel_size = _read_env_int("CHESS_EVAL_TENSOR_PARALLEL_SIZE", d.tensor_parallel_size)
    gpu_memory_utilization = _read_env_float("CHESS_EVAL_GPU_MEMORY_UTILIZATION", d.gpu_memory_utilization)
    out_jsonl = os.environ.get("CHESS_EVAL_OUT_JSONL", "").strip() or None

    t0 = time.time()
    rows = _load_rows(d.parquet, limit)
    if not rows:
        _die(f"No rows loaded from parquet: {d.parquet}")

    reward_models: list[dict[str, Any]] = []
    for r in rows:
        rm = dict((r.get("reward_model") or {}))
        reward_models.append(_parse_json_maps_in_reward_model(rm))

    template = _load_template(template_path)

    tokenizer = AutoTokenizer.from_pretrained(d.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_f = None
    if out_jsonl:
        out_path = Path(out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")
        template_label = _sanitize_filename_component(Path(template_path).name)
        meta = {
            "type": "meta",
            "model": d.model,
            "parquet": d.parquet,
            "template_path": template_path,
            "template_name": template_label,
            "k": d.k,
            "max_response_tokens": d.max_response_tokens,
        }
        out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        out_f.flush()

    llm = LLM(
        model=d.model,
        tokenizer=d.model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=d.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        seed=d.seed,
        max_num_seqs=max_num_seqs,
    )

    if d.k <= 0:
        raise ValueError(f"k must be > 0, got {d.k}")

    if d.seed_mode not in ("engine", "per_prompt"):
        raise ValueError(f"Unknown seed_mode={d.seed_mode!r}; expected 'engine' or 'per_prompt'")

    # Automatically choose a chunk size that respects vLLM's scheduler cap.
    # vLLM schedules sequences, not prompts, so batch_size * n_per_prompt must fit.
    k_chunk = max(1, min(d.k, max_num_seqs // max(1, batch_size)))

    base_sampling_kwargs = dict(
        temperature=d.temperature,
        top_p=d.top_p,
        top_k=-1,
        min_p=0.0,
        max_tokens=d.max_response_tokens,
        repetition_penalty=1.0,
        detokenize=True,
    )

    n_prompts = len(reward_models)
    pass1_hits = 0
    pass32_hits = 0
    prompt_tokens_sum = 0
    gen_tokens_sum = 0
    gen_samples = 0

    pbar = tqdm(total=n_prompts, unit="prompt", desc="pass@k (template)", dynamic_ncols=True)
    total_batches = math.ceil(n_prompts / batch_size)
    for batch_idx, (start, end) in enumerate(_iter_batches(n_prompts, batch_size), start=1):
        batch_reward_models = reward_models[start:end]
        batch_messages = [_render_prompt_messages(template, rm) for rm in batch_reward_models]
        batch_prompt_token_ids = _build_prompt_token_ids(
            tokenizer,
            batch_messages,
            max_prompt_tokens=d.max_prompt_tokens,
        )
        prompt_tokens_sum += sum(len(x) for x in batch_prompt_token_ids)

        vllm_inputs = [{"prompt_token_ids": ids} for ids in batch_prompt_token_ids]
        batch_outputs: list[list[str]] = [[] for _ in range(end - start)]
        batch_gen_lens: list[list[int]] = [[] for _ in range(end - start)]

        for chunk_idx, chunk_start in enumerate(range(0, d.k, k_chunk)):
            n_gen = min(k_chunk, d.k - chunk_start)
            if d.seed_mode == "per_prompt":
                seed_stride = 1_000_000
                sampling_params_batch = [
                    SamplingParams(
                        seed=d.seed + (start + i) + chunk_idx * seed_stride,
                        n=n_gen,
                        **base_sampling_kwargs,
                    )
                    for i in range(end - start)
                ]
                request_outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params_batch, use_tqdm=False)
            else:
                sampling_params = SamplingParams(n=n_gen, **base_sampling_kwargs)
                request_outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

            if len(request_outputs) != (end - start):
                raise RuntimeError(f"Expected {end - start} vLLM outputs, got {len(request_outputs)}")

            for i, out in enumerate(request_outputs):
                if len(out.outputs) != n_gen:
                    raise RuntimeError(f"Expected n={n_gen} outputs per prompt, got {len(out.outputs)}")
                for sample in out.outputs:
                    token_ids = sample.token_ids
                    batch_gen_lens[i].append(len(token_ids))
                    batch_outputs[i].append(str(sample.text))

        for i, outs in enumerate(batch_outputs):
            if len(outs) != d.k:
                raise RuntimeError(f"Expected k={d.k} outputs per prompt, got {len(outs)}")
            gen_tokens_sum += int(sum(batch_gen_lens[i]))
            gen_samples += len(batch_gen_lens[i])

        # Score all samples via the canonical reward parser; derive strict pass@k from its format gate.
        flat_sources: list[dict[str, Any]] = []
        flat_solutions: list[str] = []
        for rm, outs in zip(batch_reward_models, batch_outputs):
            for s in outs:
                flat_sources.append(rm)
                flat_solutions.append(s)

        scored = compute_score_batch(flat_sources, flat_solutions, chess_reward_fn="winrate", logit_eps=1e-6)
        if len(scored) != (end - start) * d.k:
            raise RuntimeError(f"Scoring mismatch: expected {(end - start) * d.k} results, got {len(scored)}")

        for i in range(end - start):
            row_scored = scored[i * d.k : (i + 1) * d.k]
            strict_ok = [
                (float(s.get("format_reward", 0.0)) >= 1.0 and str(s.get("pred_move", "") or "") == str(s.get("gt_uci", "") or ""))
                for s in row_scored
            ]
            if strict_ok[0]:
                pass1_hits += 1
            if any(strict_ok):
                pass32_hits += 1

            if out_f is not None:
                rm = batch_reward_models[i]
                prompt_text = str(batch_messages[i][0]["content"])
                completions = []
                for j, (s, ok) in enumerate(zip(row_scored, strict_ok)):
                    completion = batch_outputs[i][j]
                    completions.append(
                        {
                            "sample_idx": j,
                            "text": completion,
                            "completion_tokens": int(batch_gen_lens[i][j]),
                            "pred_move": str(s.get("pred_move", "") or ""),
                            "gt_uci": str(s.get("gt_uci", "") or ""),
                            "format_reward": float(s.get("format_reward", 0.0)),
                            "acc": float(s.get("acc", 0.0)),
                            "penalty_reason": str(s.get("penalty_reason", "") or ""),
                            "strict_correct": bool(ok),
                        }
                    )
                record = {
                    "type": "row",
                    "row_idx": int(start + i),
                    "fen": str(rm.get("fen") or ""),
                    "legal_moves_uci": _normalize_legal_moves(rm.get("legal_moves_uci")),
                    "ground_truth": str(rm.get("ground_truth") or ""),
                    "prompt": prompt_text,
                    "completions": completions,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if out_f is not None:
            out_f.flush()

        processed = end
        pass1 = pass1_hits / processed
        pass32 = pass32_hits / processed
        avg_prompt_tok = prompt_tokens_sum / processed
        avg_gen_tok = gen_tokens_sum / max(gen_samples, 1)

        pbar.update(end - start)
        pbar.set_postfix(
            batch=f"{batch_idx}/{total_batches}",
            pass1=f"{pass1:.3f}",
            pass32=f"{pass32:.3f}",
            avg_gen_tok=f"{avg_gen_tok:.1f}",
        )

    pbar.close()

    elapsed_s = time.time() - t0
    metrics = {
        "model": d.model,
        "parquet": d.parquet,
        "template_path": template_path,
        "num_prompts": n_prompts,
        "k": d.k,
        # Pass@k under the strict `<guess>...</guess><think>...</think><uci_move>...</uci_move>` gate.
        "pass_at_1": pass1_hits / n_prompts,
        "pass_at_32": pass32_hits / n_prompts,
        # Token-length metrics (tokenizer counts).
        "avg_prompt_tokens": prompt_tokens_sum / n_prompts,
        "avg_completion_tokens": gen_tokens_sum / max(gen_samples, 1),
        "max_response_tokens": d.max_response_tokens,
        "elapsed_s": elapsed_s,
        "tensor_parallel_size": int(tensor_parallel_size),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "out_jsonl": out_jsonl or "",
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if out_f is not None:
        out_f.write(json.dumps({"type": "metrics", **metrics}, ensure_ascii=False) + "\n")
        out_f.close()


if __name__ == "__main__":
    main()
