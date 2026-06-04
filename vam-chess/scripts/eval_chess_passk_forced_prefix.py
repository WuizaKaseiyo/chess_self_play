#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from recipe.chess.reward_fn import compute_score_batch


@dataclass(frozen=True)
class EvalConfig:
    model: str
    tokenizer: str | None
    parquet: str
    prefix_template: str
    k: int
    batch_size: int
    limit: int | None
    max_prompt_length: int
    max_response_length: int
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_num_seqs: int
    temperature: float
    top_p: float
    seed: int
    seed_mode: str
    chess_reward_fn: str
    logit_eps: float


GUESS_PREFIX = "<guess>"


def _load_rows(parquet_path: str, limit: int | None) -> list[dict[str, Any]]:
    dataset = ds.dataset(parquet_path, format="parquet")
    columns = ["prompt", "reward_model", "extra_info"]
    table = dataset.to_table(columns=columns)
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]
    return rows


def _sanitize_model_name(model: str) -> str:
    safe = model.replace("/", "__").replace(":", "_")
    safe = "".join(ch if (ch.isalnum() or ch in "._-__") else "_" for ch in safe)
    return safe


def _tokenize_prefix(*, tokenizer: Any, template: str, move: str) -> list[int]:
    if "{move}" not in template:
        raise ValueError("prefix_template must include '{move}'")
    text = template.format(move=move)
    return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]


def _strip_leading_guess_block(text: str) -> tuple[str, str | None]:
    """
    Returns (stripped_text, extracted_guess_text_or_none).

    Only strips a single leading <guess>...</guess> block (plus surrounding whitespace).

    NOTE: This is for logging/analysis only. Scoring uses the full decoded output,
    since the reward function requires a leading <guess> block and ignores its payload for scoring.
    """
    s0 = text or ""
    s = s0.lstrip()
    low = s.lower()
    if not low.startswith(GUESS_PREFIX):
        return s0, None
    end_tag = "</guess>"
    end = low.find(end_tag)
    if end == -1:
        return s0, None
    guess_payload = s[len(GUESS_PREFIX) : end].strip()
    rest = s[end + len(end_tag) :]
    return rest.lstrip(), guess_payload


def _iter_batches(n: int, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, n, batch_size):
        yield start, min(n, start + batch_size)


def _score_outputs(
    reward_models: list[dict[str, Any]],
    decoded_outputs: list[list[str]],
    *,
    chess_reward_fn: str,
    logit_eps: float,
) -> list[list[dict[str, Any]]]:
    # Flatten for compute_score_batch.
    flat_sources: list[dict[str, Any]] = []
    flat_solutions: list[str] = []
    for rm, outs in zip(reward_models, decoded_outputs):
        for s in outs:
            flat_sources.append(rm)
            flat_solutions.append(s)

    scored = compute_score_batch(flat_sources, flat_solutions, chess_reward_fn=chess_reward_fn, logit_eps=logit_eps)

    # Unflatten back to per-prompt lists.
    n_prompts = len(reward_models)
    n_per = len(decoded_outputs[0]) if decoded_outputs else 0
    out: list[list[dict[str, Any]]] = []
    for i in range(n_prompts):
        out.append(scored[i * n_per : (i + 1) * n_per])
    return out


def run_eval(cfg: EvalConfig) -> dict[str, Any]:
    rows = _load_rows(cfg.parquet, cfg.limit)
    if not rows:
        raise ValueError(f"No rows loaded from {cfg.parquet}")

    tokenizer_model = cfg.tokenizer or cfg.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-tokenize prompts and forced prefixes.
    prompts_token_ids: list[list[int]] = []
    forced_prefix_token_ids: list[list[int]] = []
    reward_models: list[dict[str, Any]] = []
    extra_infos: list[dict[str, Any]] = []
    forced_prefix_texts: list[str] = []

    for row in rows:
        msgs = row.get("prompt")
        if not isinstance(msgs, list):
            raise TypeError(f"Expected row['prompt'] as list, got {type(msgs)}")
        prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if not isinstance(prompt_ids, list):
            raise TypeError(f"Expected token id list from apply_chat_template, got {type(prompt_ids)}")
        if len(prompt_ids) > cfg.max_prompt_length:
            raise ValueError(f"Prompt is {len(prompt_ids)} tokens (max={cfg.max_prompt_length}).")
        prompt_ids = [int(x) for x in prompt_ids]

        rm = dict(row.get("reward_model") or {})
        gt = str(rm.get("ground_truth") or "").strip().lower()
        if not gt:
            raise ValueError("Missing reward_model.ground_truth")
        fp_ids = _tokenize_prefix(tokenizer=tokenizer, template=cfg.prefix_template, move=gt)
        fp_text = cfg.prefix_template.format(move=gt)
        forced_prefix_texts.append(fp_text)
        forced_prefix_token_ids.append(fp_ids)

        prompts_token_ids.append(prompt_ids)
        reward_models.append(rm)
        extra_infos.append(dict(row.get("extra_info") or {}))

    llm = LLM(
        model=cfg.model,
        tokenizer=tokenizer_model,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        seed=cfg.seed,
        max_num_seqs=cfg.max_num_seqs,
    )

    base_sampling_kwargs = dict(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        detokenize=False,
        n=cfg.k,
    )
    if cfg.seed_mode not in ("engine", "per_prompt"):
        raise ValueError(f"Unknown seed_mode={cfg.seed_mode!r}; expected 'engine' or 'per_prompt'")

    n = len(prompts_token_ids)
    per_prompt_results: list[dict[str, Any]] = []

    t0 = time.time()
    total_batches = math.ceil(n / cfg.batch_size)
    for batch_idx, (start, end) in enumerate(_iter_batches(n, cfg.batch_size), start=1):
        batch_prompt_ids = prompts_token_ids[start:end]
        batch_prefix_ids = forced_prefix_token_ids[start:end]
        batch_reward_models = reward_models[start:end]
        batch_extra = extra_infos[start:end]
        batch_prefix_texts = forced_prefix_texts[start:end]

        vllm_inputs = []
        sampling_params_batch = []
        for i in range(end - start):
            # Condition generation on the forced prefix by appending it to the prompt IDs.
            prompt_plus_prefix = list(batch_prompt_ids[i]) + list(batch_prefix_ids[i])
            vllm_inputs.append({"prompt_token_ids": prompt_plus_prefix})

            # Match training-time truncation behavior: the stored response is
            # `forced_prefix + generated`, truncated to max_response_length.
            # We avoid wasted generation by reducing max_tokens accordingly.
            max_tokens = max(int(cfg.max_response_length) - len(batch_prefix_ids[i]), 1)
            if cfg.seed_mode == "per_prompt":
                sampling_params_batch.append(
                    SamplingParams(seed=cfg.seed + (start + i), max_tokens=max_tokens, **base_sampling_kwargs)
                )
            else:
                sampling_params_batch.append(
                    SamplingParams(max_tokens=max_tokens, **base_sampling_kwargs)
                )

        outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params_batch, use_tqdm=False)

        decoded_outputs: list[list[str]] = []
        for out, prefix_ids in zip(outputs, batch_prefix_ids):
            sample_texts: list[str] = []
            for sample in out.outputs:
                # Reconstruct the full response text (forced prefix + generated continuation).
                gen_ids = list(sample.token_ids)
                max_gen = max(int(cfg.max_response_length) - len(prefix_ids), 0)
                full_ids = list(prefix_ids) + gen_ids[:max_gen]
                sample_texts.append(tokenizer.decode(full_ids, skip_special_tokens=True))
            decoded_outputs.append(sample_texts)

        scored = _score_outputs(
            batch_reward_models,
            decoded_outputs,
            chess_reward_fn=cfg.chess_reward_fn,
            logit_eps=cfg.logit_eps,
        )

        # Per-prompt aggregation.
        for i in range(end - start):
            idx = batch_extra[i].get("index", None)
            gt = str(batch_reward_models[i].get("ground_truth") or "")
            sample_rows: list[dict[str, Any]] = []
            for s_text, s_scored in zip(decoded_outputs[i], scored[i]):
                stripped, guess_payload = _strip_leading_guess_block(s_text)
                sample_rows.append(
                    {
                        "output_text": s_text,
                        "output_text_stripped": stripped,
                        "guess_payload": guess_payload,
                        "pred_move": s_scored.get("pred_move", ""),
                        "acc": float(s_scored.get("acc", 0.0) or 0.0),
                        "total_reward": float(s_scored.get("total_reward", 0.0) or 0.0),
                        "penalty_reason": str(s_scored.get("penalty_reason", "") or ""),
                        "format_reward": float(s_scored.get("format_reward", 0.0) or 0.0),
                        "move_expected_score": s_scored.get("move_expected_score", None),
                    }
                )
            per_prompt_results.append(
                {
                    "index": idx,
                    "split": batch_extra[i].get("split", None),
                    "ground_truth": gt,
                    "forced_prefix_text": batch_prefix_texts[i],
                    "k": cfg.k,
                    "samples": sample_rows,
                }
            )

        elapsed = time.time() - t0
        print(f"[{batch_idx:>3}/{total_batches}] prompts {start}:{end} elapsed={elapsed/60:.1f}min")

    # Compute pass@k and diagnostics.
    pass_flags: list[float] = []
    valid_counts: list[int] = []
    unique_valid_moves: list[int] = []
    expected_score_sums: list[float] = []
    for row in per_prompt_results:
        samples = row.get("samples", [])
        accs = [float(s.get("acc", 0.0) or 0.0) for s in samples]
        pass_flags.append(1.0 if any(a > 0.0 for a in accs) else 0.0)

        valid = [1 if (str(s.get("penalty_reason", "") or "") == "") else 0 for s in samples]
        valid_counts.append(int(sum(valid)))

        mv_set: set[str] = set()
        for s, ok in zip(samples, valid):
            if not ok:
                continue
            mv = str(s.get("pred_move", "") or "").strip().lower()
            if mv:
                mv_set.add(mv)
        unique_valid_moves.append(len(mv_set))

        es_sum = 0.0
        for s, ok in zip(samples, valid):
            if not ok:
                continue
            x = s.get("move_expected_score", None)
            try:
                v = float(x)
            except Exception:
                continue
            if math.isfinite(v):
                es_sum += v
        expected_score_sums.append(float(es_sum))

    passk = float(np.mean(np.asarray(pass_flags, dtype=np.float32))) if pass_flags else 0.0
    summary = {
        "num_prompts": len(per_prompt_results),
        "k": cfg.k,
        "passk_acc_mean": passk,
        "valid_count_mean": float(np.mean(np.asarray(valid_counts, dtype=np.float32))) if valid_counts else 0.0,
        "unique_valid_moves_mean": float(np.mean(np.asarray(unique_valid_moves, dtype=np.float32))) if unique_valid_moves else 0.0,
        "expected_score_sum_mean": float(np.mean(np.asarray(expected_score_sums, dtype=np.float32))) if expected_score_sums else 0.0,
    }
    print("Summary:", json.dumps(summary, indent=2))

    return {
        "config": asdict(cfg),
        "summary": summary,
        "per_prompt": per_prompt_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--parquet", default="data/chess_puzzles/test.parquet")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_response_length", type=int, default=2000)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_num_seqs", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_mode", choices=["engine", "per_prompt"], default="engine")
    parser.add_argument("--chess_reward_fn", default="winrate")
    parser.add_argument("--logit_eps", type=float, default=1e-6)
    parser.add_argument(
        "--prefix_template",
        default="<guess> {move} </guess>",
        help="Prefix template (must include '{move}'). Default matches the RL forced-guess scheme.",
    )
    parser.add_argument("--out_json", required=True)
    parser.add_argument(
        "--out_jsonl_gz",
        default=None,
        help="Optional: write per-prompt results as JSONL.gz (one prompt per line).",
    )
    args = parser.parse_args()

    prefix_template = str(args.prefix_template)

    cfg = EvalConfig(
        model=str(args.model),
        tokenizer=(str(args.tokenizer) if args.tokenizer else None),
        parquet=str(args.parquet),
        prefix_template=prefix_template,
        k=int(args.k),
        batch_size=int(args.batch_size),
        limit=(int(args.limit) if args.limit is not None else None),
        max_prompt_length=int(args.max_prompt_length),
        max_response_length=int(args.max_response_length),
        max_model_len=int(args.max_model_len),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_num_seqs=int(args.max_num_seqs),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        seed=int(args.seed),
        seed_mode=str(args.seed_mode),
        chess_reward_fn=str(args.chess_reward_fn),
        logit_eps=float(args.logit_eps),
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = out_json.with_suffix(out_json.suffix + ".tmp")

    result = run_eval(cfg)

    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    tmp_json.replace(out_json)
    print(f"Wrote {out_json}")

    out_jsonl_gz = args.out_jsonl_gz
    if out_jsonl_gz:
        jsonl_path = Path(out_jsonl_gz)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_gz = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
        with gzip.open(tmp_gz, "wt", encoding="utf-8") as f:
            for row in result.get("per_prompt", []):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_gz.replace(jsonl_path)
        print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    main()
