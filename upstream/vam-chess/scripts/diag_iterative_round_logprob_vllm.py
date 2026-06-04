#!/usr/bin/env python3
"""Iterative allowed-move elimination diagnostic with round-0 referenced logprobs.

This script mirrors the trainer's iterative allowed-move elimination behavior:
- Initialize allowed moves from full legal moves.
- Generate exactly one sample per active prompt per round.
- Stop a prompt when the selected success criterion is hit, else prune the sampled
  valid in-subset move from the candidate list.
- Force-accept unresolved prompts at round K.

For each generated response, it computes log P(response | round-0 prompt) using vLLM
prompt logprobs by scoring the concatenated token sequence:
    [round0_prompt_tokens + response_tokens]

Default data source is exactly:
  data/chess_puzzles_chessr1_aligned_sharded_ours/train_0.parquet
  data/chess_puzzles_chessr1_aligned_sharded_ours/train_1.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from jinja2 import Environment
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from recipe.chess.reward_fn import compute_score

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for environments without tqdm
    tqdm = None


DEFAULT_PARQUETS = [
    "data/chess_puzzles_chessr1_aligned_sharded_ours/train_0.parquet",
    "data/chess_puzzles_chessr1_aligned_sharded_ours/train_1.parquet",
]


@dataclass
class ExampleState:
    row_uid: str
    row_index: Optional[int]
    shard: str
    row_in_shard: int
    fen: str
    gt_uci: str
    legal_moves: list[str]
    allowed_moves: list[str]
    reward_model_payload: dict[str, Any]
    round0_prompt_text: str
    round0_prompt_model_text: str
    round0_prompt_token_ids: list[int]
    target_move_round0: str
    target_eq_gt_round0: bool
    initial_allowed_size: int
    done: bool = False
    rounds_used: int = 0
    forced_accept: bool = False
    stop_reason: str = ""
    first_success_round_gt: int = 0
    first_success_round_target: int = 0


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
        for m in moves:
            s = _normalize_uci(m)
            if s:
                out.append(s)
    except Exception:
        return []
    return out


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
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
    for k, v in obj.items():
        key = _normalize_uci(k)
        if not key:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            out[key] = fv
    return out


def _best_move_by_mu(mu_map: dict[str, float], moves: list[str]) -> str:
    best_move = ""
    best_val = -float("inf")
    for mv in moves:
        key = _normalize_uci(mv)
        val = float(mu_map.get(key, -float("inf")))
        if (val > best_val) or (val == best_val and (not best_move or key < best_move)):
            best_val = val
            best_move = key
    return best_move


def _batched_indices(n: int, batch_size: int) -> Iterator[tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    i = 0
    while i < n:
        j = min(n, i + batch_size)
        yield i, j
        i = j


def _render_prompt(template: Any, fen: str, legal: list[str], allowed: list[str]) -> str:
    return str(
        template.render(
            FEN=fen,
            legal_moves_uci_list=legal,
            considered_moves_uci_list=allowed,
        )
    )


def _to_model_prompt_and_ids(tokenizer: Any, prompt_text: str, use_chat_template: bool) -> tuple[str, list[int]]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        model_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return str(model_text), [int(x) for x in model_ids]
    model_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return prompt_text, [int(x) for x in model_ids]


def _to_optional_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _json_safe(obj: Any) -> Any:
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def _load_examples(
    *,
    parquet_paths: list[Path],
    limit_rows: Optional[int],
    template: Any,
    tokenizer: Any,
    use_chat_template: bool,
    max_model_len: int,
) -> tuple[list[ExampleState], dict[str, Any]]:
    out: list[ExampleState] = []
    stats = {
        "rows_total_seen": 0,
        "rows_loaded": 0,
        "rows_skipped_missing": 0,
        "rows_skipped_prompt_too_long": 0,
        "rows_considered_neq_legal": 0,
    }

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=["reward_model", "extra_info"])
        rm_col = table.column("reward_model").to_pylist()
        ei_col = table.column("extra_info").to_pylist()

        for row_in_shard, (rm_raw, ei_raw) in enumerate(zip(rm_col, ei_col, strict=True)):
            stats["rows_total_seen"] += 1
            if limit_rows is not None and len(out) >= limit_rows:
                return out, stats

            rm = rm_raw if isinstance(rm_raw, dict) else {}
            ei = ei_raw if isinstance(ei_raw, dict) else {}

            fen = str(rm.get("fen") or "").strip()
            gt_uci = _normalize_uci(rm.get("ground_truth"))
            legal_moves = _dedupe_preserve_order(_normalize_moves(rm.get("legal_moves_uci")))
            considered_moves = _dedupe_preserve_order(_normalize_moves(rm.get("considered_moves_uci")))

            if not fen or not gt_uci or not legal_moves:
                stats["rows_skipped_missing"] += 1
                continue

            if considered_moves and considered_moves != legal_moves:
                stats["rows_considered_neq_legal"] += 1

            mu_map = _parse_move_map(rm.get("move_expected_scores_json"))
            if not mu_map:
                mu_map = _parse_move_map(rm.get("move_values_json"))
            if not mu_map:
                stats["rows_skipped_missing"] += 1
                continue

            target_move_round0 = _best_move_by_mu(mu_map, legal_moves)
            if not target_move_round0:
                stats["rows_skipped_missing"] += 1
                continue

            round0_prompt_text = _render_prompt(template, fen, legal_moves, legal_moves)
            round0_prompt_model_text, round0_prompt_token_ids = _to_model_prompt_and_ids(
                tokenizer=tokenizer,
                prompt_text=round0_prompt_text,
                use_chat_template=use_chat_template,
            )
            if len(round0_prompt_token_ids) >= max_model_len:
                stats["rows_skipped_prompt_too_long"] += 1
                continue

            row_index = _to_optional_int((ei or {}).get("index"))
            shard_name = parquet_path.name
            row_uid = f"{shard_name}:{row_in_shard}"

            reward_payload = {
                "fen": fen,
                "ground_truth": gt_uci,
                "legal_moves_uci": list(legal_moves),
                "considered_moves_uci": list(legal_moves),
                "move_expected_scores_json": rm.get("move_expected_scores_json"),
                "move_values_json": rm.get("move_values_json"),
            }

            out.append(
                ExampleState(
                    row_uid=row_uid,
                    row_index=row_index,
                    shard=shard_name,
                    row_in_shard=int(row_in_shard),
                    fen=fen,
                    gt_uci=gt_uci,
                    legal_moves=list(legal_moves),
                    allowed_moves=list(legal_moves),
                    reward_model_payload=reward_payload,
                    round0_prompt_text=round0_prompt_text,
                    round0_prompt_model_text=round0_prompt_model_text,
                    round0_prompt_token_ids=list(round0_prompt_token_ids),
                    target_move_round0=target_move_round0,
                    target_eq_gt_round0=bool(target_move_round0 == gt_uci),
                    initial_allowed_size=len(legal_moves),
                )
            )
            stats["rows_loaded"] += 1

    return out, stats


def _finite_stats(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    return float(statistics.fmean(vals)), float(statistics.median(vals))


def _plot_round_vs_logprob(round_df: pd.DataFrame, out_path: Path) -> None:
    if round_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = round_df["round"].to_numpy(dtype=np.int64)
    mean_lp = round_df["mean_reference_logprob_sum"].to_numpy(dtype=np.float64)
    med_lp = round_df["median_reference_logprob_sum"].to_numpy(dtype=np.float64)
    ax.plot(x, mean_lp, marker="o", linewidth=1.6, label="mean logprob sum")
    ax.plot(x, med_lp, marker="s", linewidth=1.4, label="median logprob sum")
    ax.set_title("Round vs Reference Logprob (round-0 prompt context)")
    ax.set_xlabel("Round")
    ax.set_ylabel("logprob sum")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _deterministic_subsample_states(
    *,
    states: list[ExampleState],
    sample_frac: float,
    sample_seed: int,
    sample_n_max: Optional[int],
) -> tuple[list[ExampleState], dict[str, Any]]:
    n_total = len(states)
    if n_total == 0:
        return states, {"n_total": 0, "n_selected": 0, "sample_frac": sample_frac, "sample_seed": sample_seed}

    if not (0.0 < sample_frac <= 1.0):
        raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}")

    n_target = int(round(n_total * sample_frac))
    n_target = max(1, min(n_total, n_target))
    if sample_n_max is not None:
        n_target = max(1, min(n_target, int(sample_n_max)))

    if n_target >= n_total:
        return states, {
            "n_total": n_total,
            "n_selected": n_total,
            "sample_frac": sample_frac,
            "sample_seed": sample_seed,
            "sample_n_max": sample_n_max,
            "sampling_applied": False,
        }

    rng = random.Random(int(sample_seed))
    selected_indices = sorted(rng.sample(range(n_total), n_target))
    sampled = [states[i] for i in selected_indices]
    return sampled, {
        "n_total": n_total,
        "n_selected": n_target,
        "sample_frac": sample_frac,
        "sample_seed": sample_seed,
        "sample_n_max": sample_n_max,
        "sampling_applied": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquets", nargs="+", default=DEFAULT_PARQUETS)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--template_path", default="recipe/chess/prompt_templates/select_prompt.jinja")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit_rows", type=int, default=None)
    ap.add_argument("--k_max", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--score_batch_size", type=int, default=4)
    ap.add_argument(
        "--submit_all_per_round",
        action="store_true",
        default=False,
        help="Submit all active prompts in one request for generation only.",
    )
    ap.add_argument("--sample_frac", type=float, default=1.0)
    ap.add_argument("--sample_seed", type=int, default=None)
    ap.add_argument("--sample_n_max", type=int, default=None)
    ap.add_argument("--no_progress", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_output_tokens", type=int, default=2000)
    ap.add_argument("--max_model_length", type=int, default=4096)
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--stop_criterion", choices=["gt_uci", "target_move"], default="gt_uci")
    ap.add_argument("--use_chat_template", action="store_true", default=True)
    ap.add_argument("--no_use_chat_template", action="store_true")
    ap.add_argument("--save_raw_output", action="store_true", default=False)
    ap.add_argument("--overwrite", action="store_true", default=False)
    args = ap.parse_args()

    if args.no_use_chat_template:
        args.use_chat_template = False
    show_progress = not bool(args.no_progress)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    round_records_path = out_dir / "round_records.jsonl"
    round_summary_path = out_dir / "round_summary.csv"
    prompt_summary_path = out_dir / "prompt_summary.csv"
    summary_json_path = out_dir / "summary.json"
    plot_path = out_dir / "round_vs_logprob.png"
    config_path = out_dir / "config.json"

    if any(p.exists() for p in [round_records_path, round_summary_path, prompt_summary_path, summary_json_path, plot_path]) and not args.overwrite:
        raise SystemExit(
            f"Output files already exist in {out_dir}. Re-run with --overwrite to replace artifacts."
        )
    if args.overwrite:
        for p in [round_records_path, round_summary_path, prompt_summary_path, summary_json_path, plot_path]:
            if p.exists():
                p.unlink()

    parquet_paths = [Path(p) for p in args.parquets]
    for p in parquet_paths:
        if not p.exists():
            raise SystemExit(f"Missing parquet: {p}")

    template_path = Path(args.template_path)
    if not template_path.exists():
        raise SystemExit(f"Missing template: {template_path}")
    template = Environment(autoescape=False).from_string(template_path.read_text(encoding="utf-8"))

    tokenizer_model = str(args.tokenizer) if args.tokenizer else str(args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    states, load_stats = _load_examples(
        parquet_paths=parquet_paths,
        limit_rows=args.limit_rows,
        template=template,
        tokenizer=tokenizer,
        use_chat_template=bool(args.use_chat_template),
        max_model_len=int(args.max_model_length),
    )
    if not states:
        raise SystemExit("No rows loaded after filtering/validation.")

    sample_seed = int(args.sample_seed) if args.sample_seed is not None else int(args.seed)
    states, sample_info = _deterministic_subsample_states(
        states=states,
        sample_frac=float(args.sample_frac),
        sample_seed=sample_seed,
        sample_n_max=args.sample_n_max,
    )

    cfg = {
        "model": str(args.model),
        "tokenizer": tokenizer_model,
        "template_path": str(template_path),
        "parquets": [str(p) for p in parquet_paths],
        "rows_loaded": len(states),
        "limit_rows": args.limit_rows,
        "k_max": int(args.k_max),
        "batch_size": int(args.batch_size),
        "score_batch_size": int(args.score_batch_size),
        "submit_all_per_round": bool(args.submit_all_per_round),
        "sample_frac": float(args.sample_frac),
        "sample_seed": int(sample_seed),
        "sample_n_max": args.sample_n_max,
        "sample_info": sample_info,
        "seed": int(args.seed),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_output_tokens": int(args.max_output_tokens),
        "max_model_length": int(args.max_model_length),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_num_seqs": int(args.max_num_seqs),
        "stop_criterion": str(args.stop_criterion),
        "use_chat_template": bool(args.use_chat_template),
        "load_stats": load_stats,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    config_path.write_text(json.dumps(_json_safe(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[INIT] loaded_rows={len(states)} "
        f"sampled_from={sample_info.get('n_total', len(states))} "
        f"stop_criterion={args.stop_criterion} "
        f"k_max={args.k_max} "
        f"batch_size={args.batch_size} "
        f"score_batch_size={args.score_batch_size} "
        f"submit_all_per_round={bool(args.submit_all_per_round)}(generation_only)"
    )

    llm = LLM(
        model=str(args.model),
        tokenizer=tokenizer_model,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_length),
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        seed=int(args.seed),
        max_num_seqs=int(args.max_num_seqs),
    )

    gen_sampling = SamplingParams(
        n=1,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_output_tokens),
        logprobs=0,
    )
    score_sampling = SamplingParams(
        n=1,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=1,
        prompt_logprobs=0,
        detokenize=False,
    )

    round_summaries: list[dict[str, Any]] = []
    num_round_records = 0

    round_iter = range(1, int(args.k_max) + 1)
    round_bar = None
    if show_progress and tqdm is not None:
        round_bar = tqdm(round_iter, total=int(args.k_max), desc="Rounds", dynamic_ncols=True)
    else:
        round_bar = round_iter

    for round_idx in round_bar:
        active_indices = [i for i, st in enumerate(states) if not st.done]
        if not active_indices:
            print(f"[ROUND {round_idx}] no active prompts; stopping.")
            break

        print(f"[ROUND {round_idx}] active_prompts={len(active_indices)}")
        round_records: list[dict[str, Any]] = []

        # 1) Generation under current round prompts.
        if bool(args.submit_all_per_round):
            gen_spans = [(0, len(active_indices))]
        else:
            gen_spans = list(_batched_indices(len(active_indices), int(args.batch_size)))

        for start, end in gen_spans:
            chunk_state_indices = active_indices[start:end]
            prompts: list[str] = []
            prompt_texts: list[str] = []
            allowed_before_lists: list[list[str]] = []
            for state_idx in chunk_state_indices:
                st = states[state_idx]
                allowed_before = list(st.allowed_moves)
                prompt_text = _render_prompt(template, st.fen, st.legal_moves, allowed_before)
                prompt_model_text, _ = _to_model_prompt_and_ids(
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    use_chat_template=bool(args.use_chat_template),
                )
                prompts.append(prompt_model_text)
                prompt_texts.append(prompt_text)
                allowed_before_lists.append(allowed_before)

            outputs = llm.generate(prompts=prompts, sampling_params=gen_sampling, use_tqdm=False)
            if len(outputs) != len(chunk_state_indices):
                raise RuntimeError("vLLM output size mismatch in generation step.")

            for out_obj, state_idx, prompt_text, allowed_before in zip(
                outputs, chunk_state_indices, prompt_texts, allowed_before_lists, strict=True
            ):
                if not out_obj.outputs:
                    raise RuntimeError("Missing completion output from vLLM request.")
                out = out_obj.outputs[0]
                response_text = str(out.text or "")
                response_token_ids = [int(x) for x in (out.token_ids or [])]
                round_records.append(
                    {
                        "state_idx": int(state_idx),
                        "round": int(round_idx),
                        "round_prompt_text": prompt_text,
                        "allowed_moves_before": list(allowed_before),
                        "response_text": response_text,
                        "response_token_ids": response_token_ids,
                        "response_token_count": int(len(response_token_ids)),
                        "response_cumulative_logprob_round_prompt": float(out.cumulative_logprob)
                        if out.cumulative_logprob is not None
                        else float("nan"),
                    }
                )

        # 2) Reference logprob under round-0 prompt context.
        # Keep scoring chunked to avoid OOM from very large prompt_logprobs requests.
        score_spans = list(_batched_indices(len(round_records), int(args.score_batch_size)))

        for start, end in score_spans:
            chunk = round_records[start:end]
            score_inputs: list[dict[str, list[int]]] = []
            score_input_to_chunk_idx: list[int] = []

            for local_idx, rec in enumerate(chunk):
                st = states[int(rec["state_idx"])]
                base_ids = st.round0_prompt_token_ids
                response_ids = rec["response_token_ids"]
                # Fast path: when current prompt context is exactly the round-0 prompt
                # (i.e., allowed_moves still equals full legal list), generation
                # cumulative_logprob is already the desired reference score.
                if list(rec["allowed_moves_before"]) == list(st.legal_moves):
                    round_prompt_lp = float(rec["response_cumulative_logprob_round_prompt"])
                    if math.isfinite(round_prompt_lp):
                        rec["reference_logprob_sum_round0_prompt"] = float(round_prompt_lp)
                        rec["reference_logprob_mean_round0_prompt"] = (
                            float(round_prompt_lp / len(response_ids)) if response_ids else 0.0
                        )
                        rec["reference_logprob_missing_tokens"] = 0
                        rec["reference_logprob_status"] = "direct_round_prompt_match"
                        continue
                combined = list(base_ids) + list(response_ids)
                # max_tokens=1 for score call, so prompt must be strictly < max_model_length.
                if len(combined) >= int(args.max_model_length):
                    rec["reference_logprob_sum_round0_prompt"] = float("nan")
                    rec["reference_logprob_mean_round0_prompt"] = float("nan")
                    rec["reference_logprob_missing_tokens"] = int(len(response_ids))
                    rec["reference_logprob_status"] = "context_overflow"
                    continue
                score_inputs.append({"prompt_token_ids": combined})
                score_input_to_chunk_idx.append(local_idx)

            if score_inputs:
                score_outputs = llm.generate(prompts=score_inputs, sampling_params=score_sampling, use_tqdm=False)
                if len(score_outputs) != len(score_inputs):
                    raise RuntimeError("vLLM output size mismatch in scoring step.")

                for score_out, local_idx in zip(score_outputs, score_input_to_chunk_idx, strict=True):
                    rec = chunk[local_idx]
                    st = states[int(rec["state_idx"])]
                    base_len = len(st.round0_prompt_token_ids)
                    response_ids = rec["response_token_ids"]
                    prompt_logprobs = score_out.prompt_logprobs

                    lp_sum = 0.0
                    missing = 0
                    if prompt_logprobs is None:
                        missing = len(response_ids)
                    else:
                        for j, tok_id in enumerate(response_ids):
                            pos = base_len + j
                            if pos >= len(prompt_logprobs):
                                missing += 1
                                continue
                            entry = prompt_logprobs[pos]
                            if not entry or tok_id not in entry:
                                missing += 1
                                continue
                            lp_sum += float(entry[tok_id].logprob)

                    if missing > 0:
                        rec["reference_logprob_sum_round0_prompt"] = float("nan")
                        rec["reference_logprob_mean_round0_prompt"] = float("nan")
                        rec["reference_logprob_missing_tokens"] = int(missing)
                        rec["reference_logprob_status"] = "missing_tokens"
                    else:
                        rec["reference_logprob_sum_round0_prompt"] = float(lp_sum)
                        rec["reference_logprob_mean_round0_prompt"] = (
                            float(lp_sum / len(response_ids)) if response_ids else 0.0
                        )
                        rec["reference_logprob_missing_tokens"] = 0
                        rec["reference_logprob_status"] = "ok"

        # 3) Reward parse + prune/update + stopping.
        for rec_idx, rec in enumerate(round_records):
            st = states[int(rec["state_idx"])]
            allowed_before = list(rec["allowed_moves_before"])

            rm = dict(st.reward_model_payload)
            rm["legal_moves_uci"] = list(st.legal_moves)
            rm["considered_moves_uci"] = list(allowed_before)

            reward_info = compute_score(
                data_source=rm,
                solution_str=str(rec["response_text"]),
                ground_truth=st.gt_uci,
                extra_info={
                    "prompt_text": str(rec["round_prompt_text"]),
                    "use_considered_moves_uci": True,
                },
                chess_reward_fn="selection",
            )

            pred_move = _normalize_uci(reward_info.get("pred_move"))
            target_move = _normalize_uci(reward_info.get("target_move"))
            gt_uci = _normalize_uci(reward_info.get("gt_uci")) or st.gt_uci
            penalty_applied = bool(reward_info.get("penalty_applied", False))
            in_subset = bool(reward_info.get("in_subset", False))
            penalty_reason = str(reward_info.get("penalty_reason") or "")
            format_reward = float(reward_info.get("format_reward", float("nan")))
            acc = float(reward_info.get("acc", float("nan")))

            success_gt = bool((not penalty_applied) and pred_move and gt_uci and pred_move == gt_uci)
            success_target = bool((not penalty_applied) and pred_move and target_move and pred_move == target_move)
            success_selected = success_gt if args.stop_criterion == "gt_uci" else success_target

            if success_gt and st.first_success_round_gt == 0:
                st.first_success_round_gt = int(round_idx)
            if success_target and st.first_success_round_target == 0:
                st.first_success_round_target = int(round_idx)

            removed_move = ""
            forced_accept = False
            accepted = False
            stop_reason = ""

            if success_selected:
                st.done = True
                st.rounds_used = int(round_idx)
                st.forced_accept = False
                st.stop_reason = "success_gt_uci" if args.stop_criterion == "gt_uci" else "success_target_move"
                accepted = True
                stop_reason = st.stop_reason
            else:
                if (not penalty_applied) and in_subset and pred_move:
                    new_allowed = [m for m in st.allowed_moves if m != pred_move]
                    if new_allowed:
                        st.allowed_moves = new_allowed
                        removed_move = pred_move
                if round_idx == int(args.k_max):
                    st.done = True
                    st.rounds_used = int(round_idx)
                    st.forced_accept = True
                    st.stop_reason = "forced_accept_last_round"
                    forced_accept = True
                    accepted = True
                    stop_reason = st.stop_reason

            rec_out = {
                "row_uid": st.row_uid,
                "row_index": st.row_index,
                "shard": st.shard,
                "row_in_shard": st.row_in_shard,
                "round": int(round_idx),
                "k_max": int(args.k_max),
                "stop_criterion": str(args.stop_criterion),
                "allowed_size_before": int(len(allowed_before)),
                "allowed_size_after": int(len(st.allowed_moves)),
                "removed_move": removed_move,
                "pred_move": pred_move,
                "gt_uci": gt_uci,
                "target_move": target_move,
                "target_eq_gt_round0": bool(st.target_eq_gt_round0),
                "success_gt": bool(success_gt),
                "success_target": bool(success_target),
                "success_selected": bool(success_selected),
                "accepted": bool(accepted),
                "forced_accept": bool(forced_accept),
                "stop_reason": stop_reason,
                "penalty_applied": bool(penalty_applied),
                "penalty_reason": penalty_reason,
                "in_subset": bool(in_subset),
                "format_reward": float(format_reward),
                "acc": float(acc),
                "response_token_count": int(rec["response_token_count"]),
                "response_cumulative_logprob_round_prompt": float(
                    rec["response_cumulative_logprob_round_prompt"]
                ),
                "reference_logprob_sum_round0_prompt": float(rec["reference_logprob_sum_round0_prompt"]),
                "reference_logprob_mean_round0_prompt": float(rec["reference_logprob_mean_round0_prompt"]),
                "reference_logprob_status": str(rec["reference_logprob_status"]),
                "reference_logprob_missing_tokens": int(rec["reference_logprob_missing_tokens"]),
            }
            if args.save_raw_output:
                rec_out["response_text"] = str(rec["response_text"])
            round_records[rec_idx] = rec_out

        _write_jsonl(round_records_path, round_records)
        num_round_records += len(round_records)

        finite_lp = [
            float(r["reference_logprob_sum_round0_prompt"])
            for r in round_records
            if math.isfinite(float(r["reference_logprob_sum_round0_prompt"]))
        ]
        finite_lp_mean, finite_lp_median = _finite_stats(finite_lp)
        round_summary = {
            "round": int(round_idx),
            "active_prompts": int(len(round_records)),
            "success_gt_count": int(sum(1 for r in round_records if r["success_gt"])),
            "success_target_count": int(sum(1 for r in round_records if r["success_target"])),
            "success_selected_count": int(sum(1 for r in round_records if r["success_selected"])),
            "accepted_count": int(sum(1 for r in round_records if r["accepted"])),
            "forced_accept_count": int(sum(1 for r in round_records if r["forced_accept"])),
            "unresolved_after_round": int(sum(1 for st in states if not st.done)),
            "penalty_rate": float(
                sum(1 for r in round_records if r["penalty_applied"]) / max(1, len(round_records))
            ),
            "in_subset_rate": float(sum(1 for r in round_records if r["in_subset"]) / max(1, len(round_records))),
            "mean_allowed_size_before": float(
                statistics.fmean(float(r["allowed_size_before"]) for r in round_records)
            ),
            "mean_allowed_size_after": float(
                statistics.fmean(float(r["allowed_size_after"]) for r in round_records)
            ),
            "mean_response_tokens": float(
                statistics.fmean(float(r["response_token_count"]) for r in round_records)
            ),
            "mean_reference_logprob_sum": float(finite_lp_mean),
            "median_reference_logprob_sum": float(finite_lp_median),
            "mean_reference_logprob_mean": float(
                statistics.fmean(
                    float(r["reference_logprob_mean_round0_prompt"])
                    for r in round_records
                    if math.isfinite(float(r["reference_logprob_mean_round0_prompt"]))
                )
            )
            if any(math.isfinite(float(r["reference_logprob_mean_round0_prompt"])) for r in round_records)
            else float("nan"),
        }
        round_summaries.append(round_summary)
        print(
            f"[ROUND {round_idx}] accepted={round_summary['accepted_count']}/{round_summary['active_prompts']} "
            f"unresolved_after={round_summary['unresolved_after_round']} "
            f"mean_ref_lp={round_summary['mean_reference_logprob_sum']:.4f}"
        )
        if tqdm is not None and hasattr(round_bar, "set_postfix"):
            round_bar.set_postfix(
                {
                    "active": int(round_summary["active_prompts"]),
                    "accepted": int(round_summary["accepted_count"]),
                    "unresolved": int(round_summary["unresolved_after_round"]),
                    "mean_ref_lp": f"{float(round_summary['mean_reference_logprob_sum']):.1f}",
                }
            )

    prompt_rows: list[dict[str, Any]] = []
    for st in states:
        selected_success_round = (
            st.first_success_round_gt if args.stop_criterion == "gt_uci" else st.first_success_round_target
        )
        prompt_rows.append(
            {
                "row_uid": st.row_uid,
                "row_index": st.row_index,
                "shard": st.shard,
                "row_in_shard": st.row_in_shard,
                "gt_uci": st.gt_uci,
                "target_move_round0": st.target_move_round0,
                "target_eq_gt_round0": bool(st.target_eq_gt_round0),
                "initial_allowed_size": int(st.initial_allowed_size),
                "final_allowed_size": int(len(st.allowed_moves)),
                "rounds_used": int(st.rounds_used),
                "forced_accept": bool(st.forced_accept),
                "stop_reason": str(st.stop_reason),
                "first_success_round_gt": int(st.first_success_round_gt),
                "first_success_round_target": int(st.first_success_round_target),
                "first_success_round_selected": int(selected_success_round),
            }
        )

    round_df = pd.DataFrame(round_summaries).sort_values("round").reset_index(drop=True)
    prompt_df = pd.DataFrame(prompt_rows)
    round_df.to_csv(round_summary_path, index=False)
    prompt_df.to_csv(prompt_summary_path, index=False)
    _plot_round_vs_logprob(round_df, plot_path)

    overall = {
        "num_prompts": int(len(states)),
        "num_round_records": int(num_round_records),
        "num_prompts_forced_accept": int(sum(1 for st in states if st.forced_accept)),
        "num_prompts_success_gt": int(sum(1 for st in states if st.first_success_round_gt > 0)),
        "num_prompts_success_target": int(sum(1 for st in states if st.first_success_round_target > 0)),
        "num_prompts_target_eq_gt_round0": int(sum(1 for st in states if st.target_eq_gt_round0)),
        "stop_criterion": str(args.stop_criterion),
        "rows_loaded": int(len(states)),
        "load_stats": load_stats,
        "elapsed_sec": float(time.time() - t0),
        "artifacts": {
            "config_json": str(config_path),
            "round_records_jsonl": str(round_records_path),
            "round_summary_csv": str(round_summary_path),
            "prompt_summary_csv": str(prompt_summary_path),
            "round_vs_logprob_png": str(plot_path),
        },
    }
    summary_json_path.write_text(json.dumps(_json_safe(overall), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] elapsed_sec={overall['elapsed_sec']:.1f}")
    print(f"[DONE] round_records={round_records_path}")
    print(f"[DONE] round_summary={round_summary_path}")
    print(f"[DONE] prompt_summary={prompt_summary_path}")
    print(f"[DONE] summary_json={summary_json_path}")
    print(f"[DONE] plot={plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
