#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from recipe.chess.reward_fn import compute_score_batch


@dataclass(frozen=True)
class HardFilterConfig:
    model: str
    in_parquet: str
    out_parquet: str
    k: int
    pass_metric: str
    pass_threshold: float
    keep: str
    limit: int | None
    max_prompt_length: int
    max_response_length: int
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    enforce_eager: bool
    max_num_seqs: int
    batch_queries: int
    temperature: float
    top_p: float
    seed: int
    seed_mode: str
    chess_reward_fn: str
    logit_eps: float


def _load_table(path: str, limit: int | None) -> pa.Table:
    table = pq.read_table(path)
    if limit is not None:
        table = table.slice(0, limit)
    return table


def _preparse_reward_models(reward_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # `recipe/chess/reward_fn.py` accepts JSON strings, but parsing once per prompt
    # avoids repeated json.loads() cost across k samples.
    parsed: list[dict[str, Any]] = []
    for rm in reward_models:
        rm2 = dict(rm or {})
        for key in ("move_values_json", "move_cps_json", "move_expected_scores_json"):
            v = rm2.get(key)
            if isinstance(v, str) and v:
                try:
                    rm2[key] = json.loads(v)
                except Exception:
                    # Keep the original string; reward_fn will handle/ignore failures.
                    pass
        parsed.append(rm2)
    return parsed


def _build_prompt_token_ids(
    tokenizer: Any,
    prompts: list[list[dict[str, str]]],
    *,
    max_prompt_length: int,
) -> list[list[int]]:
    out: list[list[int]] = []
    for messages in prompts:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            raise TypeError(f"Expected token id list from apply_chat_template, got {type(ids)}")
        if len(ids) > max_prompt_length:
            raise ValueError(f"Prompt is {len(ids)} tokens (max={max_prompt_length}).")
        out.append([int(x) for x in ids])
    return out


def _decode_outputs(tokenizer: Any, request_outputs: list[Any], *, expected_n: int) -> list[list[str]]:
    decoded: list[list[str]] = []
    for out in request_outputs:
        if len(out.outputs) != expected_n:
            raise RuntimeError(f"Expected n={expected_n} outputs per prompt, got {len(out.outputs)}")
        sample_texts: list[str] = []
        for sample in out.outputs:
            sample_texts.append(tokenizer.decode(sample.token_ids, skip_special_tokens=True))
        decoded.append(sample_texts)
    return decoded


def _score_accs(
    reward_models: list[dict[str, Any]],
    decoded_outputs: list[list[str]],
    *,
    chess_reward_fn: str,
    logit_eps: float,
) -> np.ndarray:
    flat_sources: list[dict[str, Any]] = []
    flat_solutions: list[str] = []
    for rm, outs in zip(reward_models, decoded_outputs, strict=True):
        for s in outs:
            flat_sources.append(rm)
            flat_solutions.append(s)

    scored = compute_score_batch(flat_sources, flat_solutions, chess_reward_fn=chess_reward_fn, logit_eps=logit_eps)
    flat_acc = np.array([float(x.get("acc", 0.0)) for x in scored], dtype=np.float32)

    n_prompts = len(reward_models)
    n_per_prompt = len(decoded_outputs[0]) if decoded_outputs else 0
    return flat_acc.reshape(n_prompts, n_per_prompt)


def _compute_batch_prompts(*, k: int, batch_queries: int, max_num_seqs: int) -> int:
    if k <= 0:
        raise ValueError("--k must be > 0")
    if batch_queries <= 0:
        raise ValueError("--batch_queries must be > 0")
    if max_num_seqs <= 0:
        raise ValueError("--max_num_seqs must be > 0")

    # vLLM counts sequences, not prompts; when sampling with n=k, we get k sequences per prompt.
    by_queries = max(1, batch_queries // k)
    by_max_seqs = max(1, max_num_seqs // k)
    return min(by_queries, by_max_seqs)


def run(cfg: HardFilterConfig) -> None:
    table = _load_table(cfg.in_parquet, cfg.limit)
    n = table.num_rows
    if n == 0:
        raise ValueError(f"No rows found in {cfg.in_parquet!r}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=cfg.model,
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

    if cfg.seed_mode == "engine":
        sampling_params = SamplingParams(
            n=cfg.k,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=-1,
            min_p=0.0,
            max_tokens=cfg.max_response_length,
            repetition_penalty=1.0,
            detokenize=False,
        )
        sampling_params_batch = None
    elif cfg.seed_mode == "per_prompt":
        sampling_params = None
        sampling_params_batch = [
            SamplingParams(
                n=cfg.k,
                seed=cfg.seed + i,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=-1,
                min_p=0.0,
                max_tokens=cfg.max_response_length,
                repetition_penalty=1.0,
                detokenize=False,
            )
            for i in range(n)
        ]
    else:
        raise ValueError("--seed_mode must be 'engine' or 'per_prompt'")

    batch_prompts = _compute_batch_prompts(k=cfg.k, batch_queries=cfg.batch_queries, max_num_seqs=cfg.max_num_seqs)
    total_queries = n * cfg.k
    print(
        "[make_train_hard_passk] "
        f"rows={n} k={cfg.k} total_queries={total_queries} "
        f"batch_prompts={batch_prompts} (=> {batch_prompts*cfg.k} queries/batch) "
        f"max_num_seqs={cfg.max_num_seqs}"
    )

    keep_mask = np.zeros(n, dtype=bool)
    pbar = tqdm(total=n, desc="Filtering by pass@k", unit="prompt")
    metric_sum = 0.0

    for start in range(0, n, batch_prompts):
        end = min(n, start + batch_prompts)
        batch_table = table.slice(start, end - start)

        prompts = batch_table.column("prompt").to_pylist()
        reward_models_raw = batch_table.column("reward_model").to_pylist()
        reward_models = _preparse_reward_models(reward_models_raw)

        prompt_token_ids = _build_prompt_token_ids(
            tokenizer,
            prompts,
            max_prompt_length=cfg.max_prompt_length,
        )
        vllm_inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids]

        if cfg.seed_mode == "per_prompt":
            sp = sampling_params_batch[start:end]
            outputs = llm.generate(prompts=vllm_inputs, sampling_params=sp, use_tqdm=False)
        else:
            outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

        decoded = _decode_outputs(tokenizer, outputs, expected_n=cfg.k)
        accs = _score_accs(
            reward_models,
            decoded,
            chess_reward_fn=cfg.chess_reward_fn,
            logit_eps=cfg.logit_eps,
        )

        if cfg.pass_metric == "any":
            # Per-prompt pass@k is "any exact match among k samples".
            metric = accs.max(axis=1).astype(np.float32)
        elif cfg.pass_metric == "mean":
            # Mean exact-match rate across the k samples (e.g., for k=8 this is in {0, 0.125, ..., 1}).
            metric = accs.mean(axis=1).astype(np.float32)
        else:
            raise ValueError("--pass_metric must be 'any' or 'mean'")

        metric_sum += float(metric.sum())
        if cfg.keep == "below":
            keep = metric <= float(cfg.pass_threshold)
        elif cfg.keep == "above":
            keep = metric >= float(cfg.pass_threshold)
        else:
            raise ValueError("--keep must be 'below' or 'above'")

        keep_mask[start:end] = keep
        pbar.update(end - start)
        pbar.set_postfix(
            keep=int(keep_mask[:end].sum()),
            keep_frac=f"{keep_mask[:end].mean():.3f}",
            metric_mean=f"{metric_sum/max(1,end):.3f}",
        )

    pbar.close()

    kept = int(keep_mask.sum())
    metric_mean = float(metric_sum / max(1, n))
    print(f"[make_train_hard_passk] metric={cfg.pass_metric} mean={metric_mean:.6f} threshold={cfg.pass_threshold}")
    print(f"[make_train_hard_passk] keep={kept}/{n} ({kept/max(1,n):.3%}) -> {cfg.out_parquet}")

    out_path = Path(cfg.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filtered = table.filter(pa.array(keep_mask))
    pq.write_table(filtered, cfg.out_parquet)
    print(f"[make_train_hard_passk] wrote parquet rows={filtered.num_rows}")
    print("[make_train_hard_passk] done. config:", json.dumps(asdict(cfg), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a harder chess training parquet by filtering prompts based on a pass@k "
            "evaluation with vLLM sampling (exact-match on the labeled ground-truth move)."
        )
    )
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--in_parquet", default="data/chess_puzzles/train.parquet")
    parser.add_argument("--out_parquet", default="data/chess_puzzles/train_hard.parquet")
    parser.add_argument("--k", type=int, default=2, help="Number of samples per prompt (k in pass@k)")
    parser.add_argument(
        "--pass_metric",
        choices=["any", "mean"],
        default="any",
        help=(
            "How to turn per-sample exact-match scores into a per-prompt pass@k value. "
            "'any' = any exact match among k samples (0/1). "
            "'mean' = mean exact-match rate across the k samples (0..1 in steps of 1/k)."
        ),
    )
    parser.add_argument(
        "--pass_threshold",
        type=float,
        default=0.1,
        help="Hard filter threshold; with pass@k in {0,1} this usually means keep failures when < 0.1",
    )
    parser.add_argument(
        "--keep",
        choices=["below", "above"],
        default="below",
        help="Which side of the threshold to keep (below=hard; above=easy).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug: only process the first N rows")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_response_length", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=5120)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        default=False,
        help="Disable torch.compile / CUDA graphs for vLLM (often more reproducible, but slower).",
    )
    parser.add_argument("--max_num_seqs", type=int, default=1024)
    parser.add_argument(
        "--batch_queries",
        type=int,
        default=1024,
        help="Approx. (prompts * k) sequences per vLLM generate call.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_mode", choices=["engine", "per_prompt"], default="engine")
    parser.add_argument("--chess_reward_fn", default="winrate")
    parser.add_argument("--logit_eps", type=float, default=1e-6)
    args = parser.parse_args()

    cfg = HardFilterConfig(
        model=str(args.model),
        in_parquet=str(args.in_parquet),
        out_parquet=str(args.out_parquet),
        k=int(args.k),
        pass_metric=str(args.pass_metric),
        pass_threshold=float(args.pass_threshold),
        keep=str(args.keep),
        limit=args.limit,
        max_prompt_length=int(args.max_prompt_length),
        max_response_length=int(args.max_response_length),
        max_model_len=int(args.max_model_len),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        enforce_eager=bool(args.enforce_eager),
        max_num_seqs=int(args.max_num_seqs),
        batch_queries=int(args.batch_queries),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        seed=int(args.seed),
        seed_mode=str(args.seed_mode),
        chess_reward_fn=str(args.chess_reward_fn),
        logit_eps=float(args.logit_eps),
    )

    run(cfg)


if __name__ == "__main__":
    main()
