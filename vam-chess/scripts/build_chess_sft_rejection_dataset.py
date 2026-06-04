#!/usr/bin/env python3
from __future__ import annotations

"""
Build an SFT dataset via *rejection sampling* on the Chess-R1-aligned **baseline** prompt dataset.

Goal
----
Given chess prompt rows (chat messages) with a ground-truth best move, sample a base model N times
per prompt and keep only "successful" samples whose predicted move matches ground truth.

This is intended for **self-training / self-imitation** on the baseline prompt format:
  - prompt: `recipe/chess/prompt_templates/original_chessr1_prompt.jinja`
  - output contract: the answer move is inside a single `<uci_move>...</uci_move>` tag, payload is strict UCI.

Dataset contract (this script's output)
--------------------------------------
Writes a **multi-turn** SFT parquet consumable by veRL's `MultiTurnSFTDataset`:

  - required column: `messages` = prompt_messages + [{"role": "assistant", "content": <assistant_text>}]
  - optional weight column: `sft_weight` (float, default 1.0)

We keep one output row *per accepted sample* (0..N per original prompt), because this is the
simplest format supported by the existing SFT trainer in this repo.

"Match" definition (strict)
---------------------------
We define a match as strict UCI equality after parsing the model output:

  1) Extract the payload of `<uci_move>...</uci_move>`.
  2) Require **exactly one** such span (missing or multiple spans are rejected).
  3) Normalize to strict UCI (e.g. `e7e8=Q` -> `e7e8q`).
  4) Accept iff `pred_uci == reward_model.ground_truth` (lowercase).

We default to strict tag-based parsing because it matches the repo's evaluation contract
(`recipe/chess/reward_fn.py`).

Implementation notes
--------------------
* Generation uses vLLM (same stack as `scripts/eval_chess_passk.py`) for throughput.
* The assistant target is **canonicalized** to:
    `<think> ... </think><uci_move> {pred_uci} </uci_move>`
  where the `<think>` body is best-effort extracted from the generated completion; this keeps the
  dataset clean (no extra text outside tags) while still retaining some model-generated reasoning.
"""

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


_UCI_MOVE_TAG_RE = re.compile(
    r"<\s*uci_move\s*>(?P<ans>[\s\S]*?)<\s*/\s*uci_move\s*>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_TAG_RE = re.compile(
    r"<\s*think\s*>(?P<body>[\s\S]*?)<\s*/\s*think\s*>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_STRIP_RE = re.compile(r"</?\s*(think|uci_move)\s*>", re.IGNORECASE)
_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)


def _to_uci(token: str) -> str | None:
    if not token:
        return None
    t = token.strip()
    t = t.strip("`'\"")
    t = t.rstrip(".!?;,:")
    # Normalize promotions with '=' (e7e8=Q -> e7e8q), then lowercase.
    if re.fullmatch(r"[a-h][1-8][a-h][1-8]=[QRBNqrbn]", t):
        return t.replace("=", "").lower()
    if _UCI_RE.fullmatch(t):
        return t.lower()
    return None


def _extract_single_uci_move(text: str) -> tuple[str | None, str]:
    """Return (pred_uci_or_none, reason).

    Reasons:
      - "" (success)
      - "missing_uci_move_tag"
      - "multiple_uci_move_tags"
      - "bad_uci_payload"
    """
    matches = list(_UCI_MOVE_TAG_RE.finditer(text or ""))
    if not matches:
        return None, "missing_uci_move_tag"
    if len(matches) != 1:
        return None, "multiple_uci_move_tags"
    payload = (matches[0].group("ans") or "").strip()
    uci = _to_uci(payload)
    if uci is None:
        return None, "bad_uci_payload"
    return uci, ""


def _canonicalize_completion(*, raw_text: str, pred_uci: str) -> str:
    """Convert decoded completion into strict `<think>...</think><uci_move>...</uci_move>`."""
    s = (raw_text or "").strip()
    think_body = ""

    m = _THINK_TAG_RE.search(s)
    if m:
        think_body = (m.group("body") or "").strip()
    else:
        # Fallback: drop any <uci_move> blocks and keep the rest as "analysis".
        think_body = _UCI_MOVE_TAG_RE.sub("", s).strip()

    # Avoid introducing extra tag occurrences inside the think block.
    think_body = _TAG_STRIP_RE.sub("", think_body).strip()
    if not think_body:
        think_body = "I will choose the best move."

    return f"<think>{think_body}</think><uci_move> {pred_uci} </uci_move>"


def _iter_parquet_rows(
    paths: list[str],
    *,
    columns: list[str],
    limit_rows: int | None,
) -> Iterable[tuple[str, int, dict[str, Any]]]:
    """Yield (source_path, global_row_idx, row_dict)."""
    global_idx = 0
    for path in paths:
        dataset = ds.dataset(path, format="parquet")
        scanner = dataset.scanner(columns=columns, batch_size=2048)
        for batch in scanner.to_batches():
            rows = batch.to_pylist()
            for row in rows:
                if limit_rows is not None and global_idx >= int(limit_rows):
                    return
                yield path, global_idx, row
                global_idx += 1


def _build_prompt_token_ids(
    tokenizer: Any,
    prompts: list[list[dict[str, str]]],
    *,
    max_prompt_length: int,
) -> list[list[int]]:
    prompt_token_ids: list[list[int]] = []
    for messages in prompts:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            raise TypeError(f"Expected token id list from apply_chat_template, got {type(ids)}")
        if len(ids) > int(max_prompt_length):
            raise ValueError(f"Prompt is {len(ids)} tokens (max={int(max_prompt_length)}).")
        prompt_token_ids.append([int(x) for x in ids])
    return prompt_token_ids


def _decode_outputs(
    tokenizer: Any,
    request_outputs: list[Any],
    *,
    expected_n: int,
) -> tuple[list[list[str]], list[list[int]]]:
    decoded: list[list[str]] = []
    lengths: list[list[int]] = []
    for out in request_outputs:
        sample_texts: list[str] = []
        sample_lens: list[int] = []
        if len(out.outputs) != expected_n:
            raise RuntimeError(f"Expected n={expected_n} outputs per prompt, got {len(out.outputs)}")
        for sample in out.outputs:
            token_ids = sample.token_ids
            sample_texts.append(tokenizer.decode(token_ids, skip_special_tokens=True))
            sample_lens.append(len(token_ids))
        decoded.append(sample_texts)
        lengths.append(sample_lens)
    return decoded, lengths


@dataclass(frozen=True)
class BuildConfig:
    model: str
    tokenizer: str | None
    in_parquets: list[str]
    out_parquet: str
    out_stats_json: str
    limit_rows: int | None
    shard_idx: int
    num_shards: int
    samples_per_prompt: int
    max_matches_per_prompt: int
    batch_size: int
    max_prompt_length: int
    max_response_length: int
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_num_seqs: int
    max_num_batched_tokens: int | None
    temperature: float
    top_p: float
    seed: int | None
    seed_mode: str
    k_chunk: int | None
    assistant_source: str


def _validate_config(cfg: BuildConfig) -> None:
    if not cfg.in_parquets:
        raise ValueError("--in-parquet must be set at least once")
    if cfg.samples_per_prompt <= 0:
        raise ValueError("--samples-per-prompt must be > 0")
    if cfg.max_matches_per_prompt < 0:
        raise ValueError("--max-matches-per-prompt must be >= 0")
    if cfg.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if cfg.max_num_batched_tokens is not None and cfg.max_num_batched_tokens <= 0:
        raise ValueError("--max-num-batched-tokens must be > 0 when set")
    if cfg.shard_idx < 0 or cfg.num_shards <= 0 or cfg.shard_idx >= cfg.num_shards:
        raise ValueError(f"Invalid shard params: shard_idx={cfg.shard_idx} num_shards={cfg.num_shards}")
    if cfg.seed_mode not in ("none", "engine", "per_prompt"):
        raise ValueError("--seed-mode must be one of {none, engine, per_prompt}")
    if cfg.seed_mode != "none" and cfg.seed is None:
        raise ValueError("--seed must be set when --seed-mode is engine or per_prompt")
    if cfg.assistant_source not in ("canonical", "raw"):
        raise ValueError("--assistant-source must be one of {canonical, raw}")


def parse_args() -> BuildConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, required=True, help="HF model id/path for vLLM generation.")
    p.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Optional tokenizer source (HF repo id or local path). Defaults to --model.",
    )
    p.add_argument(
        "--in-parquet",
        type=str,
        action="append",
        required=True,
        help="Input parquet path (repeatable). Expected columns: prompt, reward_model, extra_info.",
    )
    p.add_argument("--out-parquet", type=str, required=True, help="Output SFT parquet path.")
    p.add_argument(
        "--out-stats-json",
        type=str,
        default="",
        help="Optional: output stats JSON path. Defaults to <out-parquet>.stats.json",
    )
    p.add_argument("--limit-rows", type=int, default=0, help="Optional: cap the total input prompts (global, before sharding).")
    p.add_argument("--shard-idx", type=int, default=0, help="Shard index (0..num_shards-1).")
    p.add_argument("--num-shards", type=int, default=1, help="Number of shards for parallel runs.")
    p.add_argument("--samples-per-prompt", type=int, default=8, help="Number of generations per prompt.")
    p.add_argument(
        "--max-matches-per-prompt",
        type=int,
        default=-1,
        help="Cap accepted samples per prompt (default: keep all matches up to samples_per_prompt).",
    )

    # vLLM batching
    p.add_argument("--batch-size", type=int, default=128, help="Prompts per vLLM generate() call.")
    p.add_argument("--k-chunk", type=int, default=0, help="Optional: split samples_per_prompt into chunks (0 = no chunking).")
    p.add_argument("--max-prompt-length", type=int, default=1536)
    p.add_argument("--max-response-length", type=int, default=2000)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=1024)
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=0,
        help="Optional vLLM scheduler token cap (0 disables explicit override).",
    )

    # sampling
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Optional sampling seed (set >= 0 to enable deterministic seeding).",
    )
    p.add_argument(
        "--seed-mode",
        type=str,
        default="none",
        choices=["none", "engine", "per_prompt"],
        help="How to apply --seed. Use 'none' to avoid explicit seed controls.",
    )

    p.add_argument(
        "--assistant-source",
        type=str,
        default="canonical",
        choices=["canonical", "raw"],
        help="What to use as the assistant content in the SFT dataset.",
    )

    args = p.parse_args()

    limit_rows = int(args.limit_rows) if int(args.limit_rows) > 0 else None
    tokenizer = str(args.tokenizer).strip() or None
    out_stats_json = str(args.out_stats_json).strip()
    if not out_stats_json:
        out_stats_json = str(Path(args.out_parquet).with_suffix(Path(args.out_parquet).suffix + ".stats.json"))

    max_matches_per_prompt = int(args.max_matches_per_prompt)
    if max_matches_per_prompt < 0:
        max_matches_per_prompt = int(args.samples_per_prompt)

    k_chunk = int(args.k_chunk)
    k_chunk = k_chunk if k_chunk > 0 else None
    seed = int(args.seed)
    seed = seed if seed >= 0 else None

    cfg = BuildConfig(
        model=str(args.model),
        tokenizer=tokenizer,
        in_parquets=[str(x) for x in (args.in_parquet or [])],
        out_parquet=str(args.out_parquet),
        out_stats_json=out_stats_json,
        limit_rows=limit_rows,
        shard_idx=int(args.shard_idx),
        num_shards=int(args.num_shards),
        samples_per_prompt=int(args.samples_per_prompt),
        max_matches_per_prompt=max_matches_per_prompt,
        batch_size=int(args.batch_size),
        max_prompt_length=int(args.max_prompt_length),
        max_response_length=int(args.max_response_length),
        max_model_len=int(args.max_model_len),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_num_seqs=int(args.max_num_seqs),
        max_num_batched_tokens=(int(args.max_num_batched_tokens) if int(args.max_num_batched_tokens) > 0 else None),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        seed=seed,
        seed_mode=str(args.seed_mode),
        k_chunk=k_chunk,
        assistant_source=str(args.assistant_source),
    )
    _validate_config(cfg)
    return cfg


def main() -> None:
    cfg = parse_args()

    out_parquet = Path(cfg.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(cfg.out_stats_json)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer_model = cfg.tokenizer or cfg.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=cfg.model,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        max_num_seqs=cfg.max_num_seqs,
    )
    if cfg.seed is not None:
        llm_kwargs["seed"] = int(cfg.seed)
    if cfg.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = int(cfg.max_num_batched_tokens)
    llm = LLM(**llm_kwargs)

    base_sampling_kwargs = dict(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_response_length,
    )

    k_chunk = cfg.k_chunk or cfg.samples_per_prompt
    if k_chunk <= 0:
        raise ValueError("k_chunk must be > 0")

    total_input_prompts = 0
    total_prompts_used = 0  # after sharding and prompt-token-length checks
    total_samples = 0
    total_prompt_tokens = 0
    total_response_tokens = 0

    accepted_samples = 0
    accepted_prompts = 0

    # For stable acceptance-per-prompt stats when max_matches_per_prompt is set.
    accepted_per_prompt: dict[str, int] = {}

    writer: pq.ParquetWriter | None = None

    t0 = time.time()

    batch_rows: list[dict[str, Any]] = []

    def flush_rows() -> None:
        nonlocal writer, batch_rows
        if not batch_rows:
            return
        table = pa.Table.from_pylist(batch_rows)
        if writer is None:
            writer = pq.ParquetWriter(str(out_parquet), table.schema, compression="zstd")
        writer.write_table(table)
        batch_rows = []

    # Stream input rows so we don't hold the full dataset in memory.
    columns = ["prompt", "reward_model", "extra_info"]
    selected: list[tuple[str, int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    # Each element: (source_path, global_row_idx, prompt_msgs, reward_model, extra_info)

    def process_selected_batch(selected_batch: list[tuple[str, int, dict[str, Any], dict[str, Any], dict[str, Any]]]) -> None:
        nonlocal total_prompts_used, total_samples, total_prompt_tokens, total_response_tokens
        nonlocal accepted_samples, accepted_prompts

        if not selected_batch:
            return

        prompts: list[list[dict[str, str]]] = []
        metas: list[dict[str, Any]] = []
        for source_path, global_row_idx, prompt_msgs, reward_model, extra_info in selected_batch:
            if not isinstance(prompt_msgs, list):
                raise TypeError(f"prompt must be list[dict], got {type(prompt_msgs)} from {source_path}")
            prompts.append(prompt_msgs)
            metas.append(
                {
                    "source_parquet": source_path,
                    "global_row_idx": int(global_row_idx),
                    "reward_model": dict(reward_model or {}),
                    "extra_info": dict(extra_info or {}),
                }
            )

        prompt_token_ids = _build_prompt_token_ids(tokenizer, prompts, max_prompt_length=cfg.max_prompt_length)
        total_prompt_tokens += int(sum(len(x) for x in prompt_token_ids))
        total_prompts_used += len(prompts)

        vllm_inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids]

        # Per-prompt seeds (for seed_mode=per_prompt): mirror `scripts/eval_chess_passk.py`
        # by deriving seeds from the prompt's global row index (stable under sharding because
        # we compute global_row_idx before applying the modulo shard filter).
        prompt_seeds: list[int] = []
        if cfg.seed_mode == "per_prompt":
            base_seed = int(cfg.seed if cfg.seed is not None else 0)
            prompt_seeds = [base_seed + int(m.get("global_row_idx", 0)) for m in metas]

        # Track whether each prompt had at least one accepted sample.
        prompt_any_accept = [False] * len(prompts)

        for chunk_idx, chunk_start in enumerate(range(0, cfg.samples_per_prompt, k_chunk)):
            n_gen = min(k_chunk, cfg.samples_per_prompt - chunk_start)
            if cfg.seed_mode == "per_prompt":
                seed_stride = 1_000_000
                sampling_params_batch = [
                    SamplingParams(seed=int(prompt_seeds[i] + chunk_idx * seed_stride), n=n_gen, **base_sampling_kwargs)
                    for i in range(len(prompts))
                ]
                outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params_batch, use_tqdm=False)
            else:
                sampling_params = SamplingParams(n=n_gen, **base_sampling_kwargs)
                outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

            decoded, lengths = _decode_outputs(tokenizer, outputs, expected_n=n_gen)
            # lengths are generated token lengths (not prompt+response).
            total_response_tokens += int(sum(sum(int(x) for x in row) for row in lengths))
            total_samples += len(prompts) * n_gen

            # One prompt at a time: accept matches.
            for i, (meta, samples, sample_lens) in enumerate(zip(metas, decoded, lengths)):
                rm = meta["reward_model"]
                ei = meta["extra_info"]
                gt = str(rm.get("ground_truth") or "").strip().lower()
                fen = str(rm.get("fen") or "").strip()
                row_id = str(ei.get("index") or "")
                prompt_key = f"{meta.get('source_parquet')}||{row_id}||{fen}"

                accepted_here = accepted_per_prompt.get(prompt_key, 0)
                remaining = cfg.max_matches_per_prompt - accepted_here
                if remaining <= 0:
                    continue

                for j, (text, resp_len) in enumerate(zip(samples, sample_lens)):
                    pred_uci, reason = _extract_single_uci_move(text)
                    if pred_uci is None:
                        continue
                    if not gt or pred_uci != gt:
                        continue

                    raw_completion = text
                    canonical_completion = _canonicalize_completion(raw_text=text, pred_uci=pred_uci)
                    assistant_text = canonical_completion if cfg.assistant_source == "canonical" else raw_completion
                    messages = list(prompts[i]) + [{"role": "assistant", "content": assistant_text}]

                    batch_rows.append(
                        {
                            "messages": messages,
                            "sft_weight": 1.0,
                            "source_parquet": meta.get("source_parquet"),
                            "row_id": (int(ei.get("index")) if str(ei.get("index") or "").strip().isdigit() else -1),
                            "fen": fen,
                            "ground_truth": gt,
                            "pred_uci": pred_uci,
                            "sample_idx": int(chunk_start + j),
                            "prompt_tokens": int(len(prompt_token_ids[i])),
                            "response_tokens": int(resp_len),
                            "raw_completion": raw_completion,
                            "canonical_completion": canonical_completion,
                            "parse_reason": reason,
                        }
                    )

                    accepted_samples += 1
                    accepted_here += 1
                    accepted_per_prompt[prompt_key] = accepted_here
                    prompt_any_accept[i] = True

                    if accepted_here >= cfg.max_matches_per_prompt:
                        break

            flush_rows()

        # Count prompts with any accept (after all chunks).
        accepted_prompts += int(sum(1 for x in prompt_any_accept if x))

    # Main streaming loop.
    selected_batch: list[tuple[str, int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for source_path, global_row_idx, row in _iter_parquet_rows(cfg.in_parquets, columns=columns, limit_rows=cfg.limit_rows):
        total_input_prompts += 1
        if (global_row_idx % cfg.num_shards) != cfg.shard_idx:
            continue
        selected_batch.append((source_path, global_row_idx, row.get("prompt"), row.get("reward_model"), row.get("extra_info")))
        if len(selected_batch) >= cfg.batch_size:
            process_selected_batch(selected_batch)
            selected_batch = []

            elapsed = time.time() - t0
            prompts_done = total_prompts_used
            samples_done = total_samples
            acc = accepted_samples
            print(
                f"[progress] shard={cfg.shard_idx}/{cfg.num_shards} "
                f"prompts_used={prompts_done} samples={samples_done} accepted={acc} "
                f"elapsed_min={elapsed/60.0:.1f}",
                flush=True,
            )

    # Tail.
    if selected_batch:
        process_selected_batch(selected_batch)

    if writer is not None:
        writer.close()
    else:
        # Write an empty parquet with the expected schema (use an empty table with explicit schema).
        empty_schema = pa.schema(
            [
                ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
                ("sft_weight", pa.float32()),
                ("source_parquet", pa.string()),
                ("row_id", pa.int64()),
                ("fen", pa.string()),
                ("ground_truth", pa.string()),
                ("pred_uci", pa.string()),
                ("sample_idx", pa.int32()),
                ("prompt_tokens", pa.int32()),
                ("response_tokens", pa.int32()),
                ("raw_completion", pa.string()),
                ("canonical_completion", pa.string()),
                ("parse_reason", pa.string()),
            ]
        )
        pq.write_table(pa.Table.from_pylist([], schema=empty_schema), str(out_parquet), compression="zstd")

    elapsed = time.time() - t0
    prompts = total_prompts_used
    samples = total_samples
    accept_rate = (float(accepted_samples) / float(samples)) if samples > 0 else 0.0
    prompts_accept_rate = (float(accepted_prompts) / float(prompts)) if prompts > 0 else 0.0

    stats = {
        "config": asdict(cfg),
        "counts": {
            "total_input_prompts_scanned": int(total_input_prompts),
            "total_prompts_used": int(prompts),
            "total_samples_generated": int(samples),
            "accepted_samples": int(accepted_samples),
            "accepted_prompts": int(accepted_prompts),
            "accept_rate_per_sample": accept_rate,
            "accept_rate_per_prompt": prompts_accept_rate,
        },
        "tokens": {
            "prompt_tokens_total": int(total_prompt_tokens),
            "response_tokens_total": int(total_response_tokens),
            "prompt_tokens_mean": (float(total_prompt_tokens) / float(prompts)) if prompts > 0 else 0.0,
            "response_tokens_mean_per_sample": (float(total_response_tokens) / float(samples)) if samples > 0 else 0.0,
        },
        "throughput": {
            "elapsed_s": float(elapsed),
            "prompts_per_s": (float(prompts) / float(elapsed)) if elapsed > 0 else 0.0,
            "samples_per_s": (float(samples) / float(elapsed)) if elapsed > 0 else 0.0,
            "response_tokens_per_s": (float(total_response_tokens) / float(elapsed)) if elapsed > 0 else 0.0,
        },
        "paths": {
            "out_parquet": str(out_parquet),
            "out_stats_json": str(stats_path),
        },
    }

    tmp = stats_path.with_suffix(stats_path.suffix + ".tmp")
    tmp.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    tmp.replace(stats_path)

    print(f"[done] wrote {out_parquet} (rows={accepted_samples})", flush=True)
    print(f"[done] wrote {stats_path}", flush=True)


if __name__ == "__main__":
    main()
