#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from recipe.chess.reward_fn import compute_score_batch
from verl.utils.prompt import (
    encode_prompt_from_messages,
    infer_use_chat_template_from_model_name,
    is_qwen3_base_model,
    render_prompt_from_messages,
)


@dataclass(frozen=True)
class EvalConfig:
    model: str
    tokenizer: str | None
    parquet: str
    acc_key: str
    k_max: int
    k_chunk: int | None
    batch_size: int
    limit: int | None
    print_prompt_examples: int
    max_prompt_length: int
    max_response_length: int
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_num_seqs: int
    max_num_batched_tokens: int | None
    enforce_eager: bool
    temperature: float
    top_p: float
    do_sample: bool
    seed: int
    seed_mode: str
    use_chat_template: bool
    chess_reward_fn: str
    logit_eps: float
    out_rollouts_jsonl: str | None


def _sanitize_model_name(model: str) -> str:
    safe = model.replace("/", "__").replace(":", "_")
    safe = "".join(ch if (ch.isalnum() or ch in "._-__") else "_" for ch in safe)
    return safe


def _load_rows(
    parquet_path: str,
    limit: int | None,
    *,
    include_extra_info: bool,
) -> list[dict[str, Any]]:
    dataset = ds.dataset(parquet_path, format="parquet")
    columns = ["prompt", "reward_model"]
    if include_extra_info:
        columns.append("extra_info")
    table = dataset.to_table(columns=columns)
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]
    return rows


def _preparse_reward_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reward_models: list[dict[str, Any]] = []
    for row in rows:
        rm = dict(row["reward_model"] or {})
        mv = rm.get("move_values_json")
        if isinstance(mv, str) and mv:
            try:
                rm["move_values_json"] = json.loads(mv)
            except Exception:
                pass
        reward_models.append(rm)
    return reward_models


def _build_prompt_token_ids(
    tokenizer: Any,
    prompts: list[list[dict[str, str]]],
    *,
    max_prompt_length: int,
    use_chat_template: bool,
) -> list[list[int]]:
    prompt_token_ids: list[list[int]] = []
    for messages in prompts:
        _, ids = encode_prompt_from_messages(
            tokenizer,
            messages,
            use_chat_template=use_chat_template,
            add_generation_prompt=True,
        )
        if len(ids) > max_prompt_length:
            raise ValueError(f"Prompt is {len(ids)} tokens (max={max_prompt_length}).")
        prompt_token_ids.append([int(x) for x in ids])
    return prompt_token_ids


def _iter_batches(n: int, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, n, batch_size):
        yield start, min(n, start + batch_size)


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


def _score_batch(
    reward_models: list[dict[str, Any]],
    decoded_outputs: list[list[str]],
    *,
    acc_key: str,
    chess_reward_fn: str,
    logit_eps: float,
    return_scored: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[list[str]],
    list[list[dict[str, Any]]] | None,
]:
    if len(reward_models) != len(decoded_outputs):
        raise ValueError(f"Expected reward_models and decoded_outputs to have the same length, got {len(reward_models)} vs {len(decoded_outputs)}")
    flat_sources: list[dict[str, Any]] = []
    flat_solutions: list[str] = []
    # NOTE: Python 3.9 doesn't support `zip(..., strict=True)`.
    for rm, outs in zip(reward_models, decoded_outputs):
        for s in outs:
            flat_sources.append(rm)
            flat_solutions.append(s)

    scored = compute_score_batch(flat_sources, flat_solutions, chess_reward_fn=chess_reward_fn, logit_eps=logit_eps)
    flat_reward = np.array([float(x["total_reward"]) for x in scored], dtype=np.float32)
    flat_acc = np.array([float(x.get(acc_key, 0.0)) for x in scored], dtype=np.float32)
    flat_expected = np.array([float(x.get("move_expected_score", float("nan"))) for x in scored], dtype=np.float32)
    flat_valid = np.array([(1.0 if (x.get("penalty_reason", "") or "") == "" else 0.0) for x in scored], dtype=np.float32)
    flat_pred_moves: list[str] = [str(x.get("pred_move", "") or "") for x in scored]

    n_prompts = len(reward_models)
    n_per_prompt = len(decoded_outputs[0]) if decoded_outputs else 0
    rewards = flat_reward.reshape(n_prompts, n_per_prompt)
    accs = flat_acc.reshape(n_prompts, n_per_prompt)
    expected_scores = flat_expected.reshape(n_prompts, n_per_prompt)
    valid_mask = flat_valid.reshape(n_prompts, n_per_prompt)
    pred_moves: list[list[str]] = []
    for i in range(n_prompts):
        pred_moves.append(flat_pred_moves[i * n_per_prompt : (i + 1) * n_per_prompt])
    scored_matrix: list[list[dict[str, Any]]] | None = None
    if return_scored:
        scored_matrix = []
        for i in range(n_prompts):
            scored_matrix.append(scored[i * n_per_prompt : (i + 1) * n_per_prompt])
    return rewards, accs, expected_scores, valid_mask, pred_moves, scored_matrix


def _compute_passk_curves(
    rewards: np.ndarray,
    accs: np.ndarray,
    *,
    k_max: int,
) -> dict[str, list[float]]:
    if rewards.shape != accs.shape:
        raise ValueError(f"Shape mismatch: rewards {rewards.shape} vs accs {accs.shape}")
    if rewards.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got rewards.ndim={rewards.ndim}")

    if rewards.shape[1] != k_max:
        raise ValueError(f"Expected k_max={k_max} samples, got {rewards.shape[1]}")

    best_reward = np.maximum.accumulate(rewards, axis=1)
    pass_acc = np.maximum.accumulate(accs, axis=1)

    reward_mean = best_reward.mean(axis=0)
    acc_mean = pass_acc.mean(axis=0)

    return {
        "reward_best_mean": [float(x) for x in reward_mean.tolist()],
        "acc_pass_mean": [float(x) for x in acc_mean.tolist()],
    }


def _compute_valid_and_quality_metrics(
    valid_mask: np.ndarray,
    expected_scores: np.ndarray,
    pred_moves: list[list[str]],
    *,
    k_max: int,
) -> dict[str, Any]:
    """
    Additional metrics requested for this ablation:
      - "diversity": how many valid moves in the k samples (and optionally unique valid moves)
      - "quality": total expected score of valid moves in the k samples
    """
    if valid_mask.ndim != 2 or expected_scores.ndim != 2:
        raise ValueError("valid_mask and expected_scores must be 2D arrays")
    if valid_mask.shape != expected_scores.shape:
        raise ValueError(f"Shape mismatch: valid_mask {valid_mask.shape} vs expected_scores {expected_scores.shape}")
    if valid_mask.shape[1] != k_max:
        raise ValueError(f"Expected k_max={k_max} samples, got {valid_mask.shape[1]}")
    if len(pred_moves) != valid_mask.shape[0]:
        raise ValueError(f"pred_moves length {len(pred_moves)} != num prompts {valid_mask.shape[0]}")

    valid_b = valid_mask.astype(np.bool_)
    expected_f = expected_scores.astype(np.float32)
    finite_expected = np.isfinite(expected_f)
    valid_and_finite = valid_b & finite_expected

    valid_count = valid_b.sum(axis=1).astype(np.float32)
    valid_count_mean = float(valid_count.mean()) if valid_count.size else 0.0

    expected_sum = np.where(valid_and_finite, expected_f, 0.0).sum(axis=1).astype(np.float32)
    expected_sum_mean = float(expected_sum.mean()) if expected_sum.size else 0.0
    expected_sum_total = float(expected_sum.sum()) if expected_sum.size else 0.0

    expected_count = valid_and_finite.sum(axis=1).astype(np.float32)
    expected_mean_valid = np.divide(
        expected_sum,
        np.maximum(expected_count, 1.0),
        out=np.zeros_like(expected_sum, dtype=np.float32),
    )
    expected_mean_valid_mean = float(expected_mean_valid.mean()) if expected_mean_valid.size else 0.0

    # Unique valid moves as a diversity proxy.
    unique_counts: list[int] = []
    for mv_list, v_row in zip(pred_moves, valid_b):
        s: set[str] = set()
        for mv, ok in zip(mv_list, v_row):
            if ok and mv:
                s.add(str(mv))
        unique_counts.append(len(s))
    unique_counts_arr = np.array(unique_counts, dtype=np.float32)
    unique_valid_moves_mean = float(unique_counts_arr.mean()) if unique_counts_arr.size else 0.0

    # Curves (prefix averages across k) for debugging/analysis.
    valid_count_curve = np.cumsum(valid_b.astype(np.float32), axis=1).mean(axis=0).tolist()
    expected_sum_curve = np.cumsum(np.where(valid_and_finite, expected_f, 0.0), axis=1).mean(axis=0).tolist()

    return {
        "valid_count_mean": valid_count_mean,
        "valid_frac_mean": valid_count_mean / float(k_max),
        "valid_count_total": int(valid_count.sum()),
        "unique_valid_moves_mean": unique_valid_moves_mean,
        "expected_score_sum_mean": expected_sum_mean,
        "expected_score_sum_total": expected_sum_total,
        "expected_score_mean_valid_mean": expected_mean_valid_mean,
        "curves": {
            "valid_count_mean": [float(x) for x in valid_count_curve],
            "expected_score_sum_mean": [float(x) for x in expected_sum_curve],
        },
    }


def run_eval(cfg: EvalConfig) -> dict[str, Any]:
    if cfg.acc_key not in {"acc", "exact_match", "coverage"}:
        raise ValueError(f"Unsupported acc_key={cfg.acc_key!r}; expected one of: acc, exact_match, coverage")

    want_rollouts = bool(cfg.out_rollouts_jsonl)
    rows = _load_rows(cfg.parquet, cfg.limit, include_extra_info=want_rollouts)
    prompts = [row["prompt"] for row in rows]
    reward_models = _preparse_reward_models(rows)
    extra_infos = [row.get("extra_info") for row in rows] if want_rollouts else None

    if cfg.k_max <= 0:
        raise ValueError(f"k_max must be > 0, got {cfg.k_max}")
    if cfg.k_chunk is None:
        k_chunk = cfg.k_max
    else:
        k_chunk = int(cfg.k_chunk)
        if k_chunk <= 0:
            raise ValueError(f"k_chunk must be > 0, got {cfg.k_chunk}")
        if k_chunk > cfg.k_max:
            raise ValueError(f"k_chunk must be <= k_max, got k_chunk={k_chunk} k_max={cfg.k_max}")

    tokenizer_model = cfg.tokenizer or cfg.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.print_prompt_examples:
        n_show = min(int(cfg.print_prompt_examples), len(prompts))
        for i in range(n_show):
            try:
                prompt_text = render_prompt_from_messages(
                    tokenizer,
                    prompts[i],
                    use_chat_template=bool(cfg.use_chat_template),
                    add_generation_prompt=True,
                )
            except Exception:
                # Fallback: decode token IDs if the tokenizer doesn't support tokenize=False.
                _, ids = encode_prompt_from_messages(
                    tokenizer,
                    prompts[i],
                    use_chat_template=bool(cfg.use_chat_template),
                    add_generation_prompt=True,
                )
                prompt_text = tokenizer.decode(ids, skip_special_tokens=False)
            print("=" * 100)
            print(f"[prompt {i}]")
            print(prompt_text)

    prompt_token_ids = _build_prompt_token_ids(
        tokenizer,
        prompts,
        max_prompt_length=cfg.max_prompt_length,
        use_chat_template=bool(cfg.use_chat_template),
    )
    prompt_token_lens = np.array([len(x) for x in prompt_token_ids], dtype=np.int32)

    llm_kwargs = dict(
        model=cfg.model,
        tokenizer=tokenizer_model,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=bool(cfg.enforce_eager),
        disable_log_stats=True,
        seed=cfg.seed,
        max_num_seqs=cfg.max_num_seqs,
    )
    if cfg.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = int(cfg.max_num_batched_tokens)
    llm = LLM(**llm_kwargs)

    if cfg.do_sample:
        base_sampling_kwargs = dict(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=-1,
            min_p=0.0,
            max_tokens=cfg.max_response_length,
            repetition_penalty=1.0,
            detokenize=False,
        )
        if cfg.seed_mode not in ("engine", "per_prompt"):
            raise ValueError(f"Unknown seed_mode={cfg.seed_mode!r}; expected 'engine' or 'per_prompt'")
    else:
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            max_tokens=cfg.max_response_length,
            repetition_penalty=1.0,
            detokenize=False,
        )

    n = len(prompt_token_ids)
    all_rewards = np.empty((n, cfg.k_max), dtype=np.float32)
    all_accs = np.empty((n, cfg.k_max), dtype=np.float32)
    all_expected_scores = np.empty((n, cfg.k_max), dtype=np.float32)
    all_valid_mask = np.empty((n, cfg.k_max), dtype=np.float32)
    all_pred_moves: list[list[str]] = [([""] * cfg.k_max) for _ in range(n)]
    all_response_lens = np.empty((n, cfg.k_max), dtype=np.int32)

    rollouts_fp = None
    if cfg.out_rollouts_jsonl:
        out_path = Path(cfg.out_rollouts_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rollouts_fp = open(out_path, "w", encoding="utf-8")

    t0 = time.time()
    total_batches = math.ceil(n / cfg.batch_size)
    for batch_idx, (start, end) in enumerate(_iter_batches(n, cfg.batch_size), start=1):
        batch_prompt_token_ids = prompt_token_ids[start:end]
        batch_reward_models = reward_models[start:end]
        batch_extra_infos = (extra_infos[start:end] if extra_infos is not None else None)
        batch_prompts = prompts[start:end]

        vllm_inputs = [{"prompt_token_ids": ids} for ids in batch_prompt_token_ids]
        batch_rollouts: list[dict[str, Any]] | None = None
        if rollouts_fp is not None:
            batch_rollouts = []
            for i, (rm, prompt, ei) in enumerate(zip(batch_reward_models, batch_prompts, batch_extra_infos or [])):
                row_id = None
                if isinstance(ei, dict):
                    row_id = ei.get("index")
                batch_rollouts.append(
                    {
                        "prompt_idx": int(start + i),
                        "row_id": row_id,
                        "fen": (rm.get("fen") if isinstance(rm, dict) else None),
                        "ground_truth": (rm.get("ground_truth") if isinstance(rm, dict) else None),
                        "legal_moves_uci": (rm.get("legal_moves_uci") if isinstance(rm, dict) else None),
                        "acc_key": str(cfg.acc_key),
                        "prompt": prompt,
                        "samples": [None] * int(cfg.k_max),
                    }
                )

        if cfg.do_sample:
            # Potentially split pass@k into multiple vLLM calls (e.g., 128 prompts × 8 samples = 1024 seqs),
            # which can improve throughput while respecting `max_num_seqs`.
            for chunk_idx, chunk_start in enumerate(range(0, cfg.k_max, k_chunk)):
                n_gen = min(k_chunk, cfg.k_max - chunk_start)
                if cfg.seed_mode == "per_prompt":
                    # Ensure different seeds across chunks to avoid duplicate samples.
                    seed_stride = 1_000_000
                    sampling_params_batch = [
                        SamplingParams(
                            seed=cfg.seed + (start + i) + chunk_idx * seed_stride,
                            n=n_gen,
                            **base_sampling_kwargs,
                        )
                        for i in range(end - start)
                    ]
                    outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params_batch, use_tqdm=False)
                else:
                    sampling_params = SamplingParams(n=n_gen, **base_sampling_kwargs)
                    outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                decoded, lengths = _decode_outputs(tokenizer, outputs, expected_n=n_gen)
                rewards, accs, expected_scores, valid_mask, pred_moves, scored = _score_batch(
                    batch_reward_models,
                    decoded,
                    acc_key=cfg.acc_key,
                    chess_reward_fn=cfg.chess_reward_fn,
                    logit_eps=cfg.logit_eps,
                    return_scored=(rollouts_fp is not None),
                )

                chunk_end = chunk_start + n_gen
                all_rewards[start:end, chunk_start:chunk_end] = rewards
                all_accs[start:end, chunk_start:chunk_end] = accs
                all_expected_scores[start:end, chunk_start:chunk_end] = expected_scores
                all_valid_mask[start:end, chunk_start:chunk_end] = valid_mask
                all_response_lens[start:end, chunk_start:chunk_end] = np.array(lengths, dtype=np.int32)
                for row_i, mv_list in enumerate(pred_moves):
                    all_pred_moves[start + row_i][chunk_start:chunk_end] = mv_list
                if rollouts_fp is not None and batch_rollouts is not None and scored is not None:
                    for row_i in range(end - start):
                        for j in range(n_gen):
                            sample_idx = int(chunk_start + j)
                            sample_scored = scored[row_i][j]
                            batch_rollouts[row_i]["samples"][sample_idx] = {
                                "sample_idx": sample_idx,
                                "completion": decoded[row_i][j],
                                "response_len": int(lengths[row_i][j]),
                                "pred_move": str(sample_scored.get("pred_move", "") or ""),
                                "target_move": str(sample_scored.get("target_move", "") or ""),
                                "gt_uci": str(sample_scored.get("gt_uci", "") or ""),
                                "total_reward": float(sample_scored.get("total_reward", 0.0) or 0.0),
                                "reward_reason": str(sample_scored.get("reward_reason", "") or ""),
                                "penalty_reason": str(sample_scored.get("penalty_reason", "") or ""),
                                "acc": float(sample_scored.get("acc", 0.0) or 0.0),
                                "exact_match": float(sample_scored.get("exact_match", 0.0) or 0.0),
                                "coverage": float(sample_scored.get("coverage", 0.0) or 0.0),
                                "format_reward": float(sample_scored.get("format_reward", 0.0) or 0.0),
                            }
        else:
            outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
            decoded, lengths = _decode_outputs(tokenizer, outputs, expected_n=1)
            rewards, accs, expected_scores, valid_mask, pred_moves, scored = _score_batch(
                batch_reward_models,
                decoded,
                acc_key=cfg.acc_key,
                chess_reward_fn=cfg.chess_reward_fn,
                logit_eps=cfg.logit_eps,
                return_scored=(rollouts_fp is not None),
            )
            if cfg.k_max > 1:
                rewards = np.repeat(rewards, cfg.k_max, axis=1)
                accs = np.repeat(accs, cfg.k_max, axis=1)
                expected_scores = np.repeat(expected_scores, cfg.k_max, axis=1)
                valid_mask = np.repeat(valid_mask, cfg.k_max, axis=1)
                pred_moves = [mv * cfg.k_max for mv in pred_moves]
                lengths = [vals * cfg.k_max for vals in lengths]
                decoded = [outs * cfg.k_max for outs in decoded]
                if scored is not None:
                    scored = [row * cfg.k_max for row in scored]

            all_rewards[start:end, :] = rewards
            all_accs[start:end, :] = accs
            all_expected_scores[start:end, :] = expected_scores
            all_valid_mask[start:end, :] = valid_mask
            all_pred_moves[start:end] = pred_moves
            all_response_lens[start:end, :] = np.array(lengths, dtype=np.int32)
            if rollouts_fp is not None and batch_rollouts is not None and scored is not None:
                for row_i in range(end - start):
                    for j in range(cfg.k_max):
                        sample_scored = scored[row_i][j]
                        batch_rollouts[row_i]["samples"][j] = {
                            "sample_idx": int(j),
                            "completion": decoded[row_i][j],
                            "response_len": int(lengths[row_i][j]),
                            "pred_move": str(sample_scored.get("pred_move", "") or ""),
                            "target_move": str(sample_scored.get("target_move", "") or ""),
                            "gt_uci": str(sample_scored.get("gt_uci", "") or ""),
                            "total_reward": float(sample_scored.get("total_reward", 0.0) or 0.0),
                            "reward_reason": str(sample_scored.get("reward_reason", "") or ""),
                            "penalty_reason": str(sample_scored.get("penalty_reason", "") or ""),
                            "acc": float(sample_scored.get("acc", 0.0) or 0.0),
                            "exact_match": float(sample_scored.get("exact_match", 0.0) or 0.0),
                            "coverage": float(sample_scored.get("coverage", 0.0) or 0.0),
                            "format_reward": float(sample_scored.get("format_reward", 0.0) or 0.0),
                        }

        if rollouts_fp is not None and batch_rollouts is not None:
            for row in batch_rollouts:
                # Best-effort: drop missing samples (should not happen).
                samples = [s for s in (row.get("samples") or []) if s is not None]
                row["samples"] = samples
                rollouts_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            rollouts_fp.flush()

        elapsed = time.time() - t0
        print(
            f"[{batch_idx:>3}/{total_batches}] prompts {start}:{end} "
            f"elapsed={elapsed/60:.1f}min"
        )

    curves = _compute_passk_curves(all_rewards, all_accs, k_max=cfg.k_max)
    extra = _compute_valid_and_quality_metrics(
        all_valid_mask,
        all_expected_scores,
        all_pred_moves,
        k_max=cfg.k_max,
    )
    prompt_len_mean = float(prompt_token_lens.mean()) if n else 0.0
    response_len_mean = float(all_response_lens.mean()) if n else 0.0
    k_key_prefix = f"k{cfg.k_max}"
    summary = {
        "acc_key": str(cfg.acc_key),
        "k1_reward_mean": float(curves["reward_best_mean"][0]),
        "k1_acc_mean": float(curves["acc_pass_mean"][0]),
        "pass_at_1_mean": float(curves["acc_pass_mean"][0]),
        f"{k_key_prefix}_reward_mean": float(curves["reward_best_mean"][-1]),
        f"{k_key_prefix}_acc_mean": float(curves["acc_pass_mean"][-1]),
        "pass_at_k_mean": float(curves["acc_pass_mean"][-1]),
        "prompt_len_mean": prompt_len_mean,
        "response_len_mean": response_len_mean,
        f"{k_key_prefix}_valid_count_mean": float(extra["valid_count_mean"]),
        f"{k_key_prefix}_valid_frac_mean": float(extra["valid_frac_mean"]),
        f"{k_key_prefix}_unique_valid_moves_mean": float(extra["unique_valid_moves_mean"]),
        f"{k_key_prefix}_expected_score_sum_mean": float(extra["expected_score_sum_mean"]),
        f"{k_key_prefix}_expected_score_sum_total": float(extra["expected_score_sum_total"]),
        f"{k_key_prefix}_expected_score_mean_valid_mean": float(extra["expected_score_mean_valid_mean"]),
    }
    print("Summary:", json.dumps(summary, indent=2))
    if rollouts_fp is not None:
        rollouts_fp.close()

    return {
        "config": asdict(cfg),
        "num_prompts": n,
        "k": list(range(1, cfg.k_max + 1)),
        "curves": {**curves, **(extra.get("curves") or {})},
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer source (HF repo id or local path). Defaults to --model. "
        "Useful when --model is a local checkpoint missing tokenizer files or when local tokenizer loading is broken.",
    )
    parser.add_argument("--use_chat_template", dest="use_chat_template", action="store_true", default=None)
    parser.add_argument("--no_use_chat_template", dest="use_chat_template", action="store_false")
    parser.add_argument("--parquet", default="data/chess_puzzles/test.parquet")
    parser.add_argument(
        "--acc_key",
        choices=["acc", "exact_match", "coverage"],
        default="acc",
        help="Which scored field to treat as accuracy when computing pass@k curves. "
        "Use exact_match/coverage for strict ground-truth matching; acc is selection-target matching.",
    )
    parser.add_argument("--k_max", type=int, default=32)
    parser.add_argument(
        "--k_chunk",
        type=int,
        default=None,
        help="Optional: split pass@k sampling into chunks of size k_chunk per vLLM call. "
        "Example: k_max=32, k_chunk=8 runs 4 vLLM generate() calls per prompt batch. "
        "This is useful to increase prompt batch size while respecting max_num_seqs.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--print_prompt_examples",
        type=int,
        default=0,
        help="Debug: print the first N rendered prompts passed into vLLM.",
    )
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_response_length", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=5120)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=1024,
        help="vLLM scheduler limit; ensure batch_size * (k_chunk or k_max) <= max_num_seqs for best throughput.",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=None,
        help="Optional vLLM scheduler token cap. Set to a large value (e.g., 65536) for higher throughput.",
    )
    parser.add_argument(
        "--enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to force eager execution in vLLM. Default false to allow cudagraph capture.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed_mode",
        choices=["engine", "per_prompt"],
        default="engine",
        help="How to seed vLLM sampling when --do_sample is set.",
    )
    parser.add_argument("--out_json", default=None)
    parser.add_argument(
        "--chess_reward_fn",
        default="winrate",
        help="Reward shaping selector (see recipe/chess/reward_fn.py).",
    )
    parser.add_argument(
        "--logit_eps",
        type=float,
        default=1e-6,
        help="Clip epsilon for logit() when using logit-based or cp-inversion reward fns.",
    )
    parser.add_argument(
        "--out_rollouts_jsonl",
        default=None,
        help="Optional: write a JSONL file of per-prompt sampled completions and scored fields. "
        "One JSON object per prompt with a `samples` list of length k_max.",
    )

    args = parser.parse_args()
    tokenizer_model = str(args.tokenizer) if args.tokenizer else str(args.model)
    use_chat_template = (
        infer_use_chat_template_from_model_name(tokenizer_model, default=True)
        if args.use_chat_template is None
        else bool(args.use_chat_template)
    )
    if is_qwen3_base_model(tokenizer_model) and use_chat_template:
        raise ValueError("Qwen3 base pass@k eval must use --no_use_chat_template.")

    cfg = EvalConfig(
        model=args.model,
        tokenizer=(str(args.tokenizer) if args.tokenizer else None),
        parquet=args.parquet,
        acc_key=str(args.acc_key),
        k_max=args.k_max,
        k_chunk=(int(args.k_chunk) if args.k_chunk is not None else None),
        batch_size=args.batch_size,
        limit=args.limit,
        print_prompt_examples=int(args.print_prompt_examples),
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=(
            int(args.max_num_batched_tokens)
            if (args.max_num_batched_tokens is not None and int(args.max_num_batched_tokens) > 0)
            else None
        ),
        enforce_eager=bool(args.enforce_eager),
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=bool(args.do_sample),
        seed=args.seed,
        seed_mode=str(args.seed_mode),
        use_chat_template=bool(use_chat_template),
        chess_reward_fn=str(args.chess_reward_fn),
        logit_eps=float(args.logit_eps),
        out_rollouts_jsonl=(str(args.out_rollouts_jsonl) if args.out_rollouts_jsonl else None),
    )

    out_json = args.out_json
    if out_json is None:
        Path("plots").mkdir(parents=True, exist_ok=True)
        mode = "sample" if cfg.do_sample else "greedy"
        out_json = str(
            Path("plots")
            / f"passk_{_sanitize_model_name(cfg.model)}_{mode}_k{cfg.k_max}_{cfg.seed_mode}_seed{cfg.seed}.json"
        )

    result = run_eval(cfg)

    tmp_path = f"{out_json}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, out_json)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
