#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LogprobRow:
    index: int | None
    split: str | None
    ground_truth: str
    prefix_template: str
    prefix_text: str
    move_token_count: int
    move_logprob_sum: float
    move_logprob_mean: float
    prompt_token_count: int


def _load_rows(parquet_path: str, limit: int | None) -> list[dict[str, Any]]:
    dataset = ds.dataset(parquet_path, format="parquet")
    columns = ["prompt", "reward_model", "extra_info"]
    table = dataset.to_table(columns=columns)
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]
    return rows


def _tokenize_template_with_move_span(
    *,
    tokenizer: Any,
    template: str,
    move: str,
) -> tuple[list[int], int, int, str]:
    if "{move}" not in template:
        raise ValueError("prefix_template must include '{move}'")
    before, after = template.split("{move}", 1)
    before_ids = tokenizer.encode(before, add_special_tokens=False)
    move_ids = tokenizer.encode(move, add_special_tokens=False)
    after_ids = tokenizer.encode(after, add_special_tokens=False)
    token_ids = [int(x) for x in (before_ids + move_ids + after_ids)]
    move_start = len(before_ids)
    move_end = move_start + len(move_ids)
    prefix_text = template.format(move=move)
    return token_ids, int(move_start), int(move_end), prefix_text


def _iter_batches(n: int, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, n, batch_size):
        yield start, min(n, start + batch_size)


def _as_int_or_none(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _as_str_or_none(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x)
    return s if s else None


def _percentiles(values: list[float], ps: list[int]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    out: dict[str, float] = {}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(arr, p))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--parquet", default="data/chess_puzzles/test.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--prefix_template",
        default="<guess> {move} </guess>",
        help="Prefix template (must include '{move}'). Default matches the RL forced-guess scheme.",
    )
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_stats_json", default=None)
    args = parser.parse_args()

    prefix_template = str(args.prefix_template)

    if "{move}" not in prefix_template:
        raise ValueError("--prefix_template must include '{move}'")

    rows = _load_rows(str(args.parquet), args.limit)
    if not rows:
        raise ValueError(f"No rows loaded from {args.parquet}")

    tokenizer_model = str(args.tokenizer) if args.tokenizer else str(args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    model.to("cuda")

    t0 = time.time()
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    written: list[LogprobRow] = []

    # Pre-tokenize prompts/prefixes up front (CPU) so GPU work is pure forward passes.
    prompts_token_ids: list[list[int]] = []
    prefixes_token_ids: list[list[int]] = []
    move_spans: list[tuple[int, int]] = []
    prefix_texts: list[str] = []
    indices: list[int | None] = []
    splits: list[str | None] = []
    gts: list[str] = []

    for row in rows:
        msgs = row.get("prompt")
        if not isinstance(msgs, list):
            raise TypeError(f"Expected row['prompt'] as list, got {type(msgs)}")
        prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if not isinstance(prompt_ids, list):
            raise TypeError(f"Expected token id list from apply_chat_template, got {type(prompt_ids)}")
        if len(prompt_ids) > int(args.max_prompt_length):
            raise ValueError(f"Prompt is {len(prompt_ids)} tokens (max={args.max_prompt_length}).")
        prompt_ids = [int(x) for x in prompt_ids]

        rm = row.get("reward_model") or {}
        gt = str(rm.get("ground_truth") or "").strip().lower()
        if not gt:
            raise ValueError("Missing reward_model.ground_truth")

        prefix_ids, move_start, move_end, prefix_text = _tokenize_template_with_move_span(
            tokenizer=tokenizer,
            template=prefix_template,
            move=gt,
        )
        if move_end <= move_start:
            raise RuntimeError(f"Empty move span for gt={gt!r} template={prefix_template!r}")

        ei = row.get("extra_info") or {}
        indices.append(_as_int_or_none(ei.get("index")))
        splits.append(_as_str_or_none(ei.get("split")))
        gts.append(gt)
        prompts_token_ids.append(prompt_ids)
        prefixes_token_ids.append(prefix_ids)
        move_spans.append((move_start, move_end))
        prefix_texts.append(prefix_text)

    # Run teacher-forced logprob over batches.
    with torch.no_grad(), open(tmp_path, "w", encoding="utf-8") as f:
        for batch_idx, (start, end) in enumerate(_iter_batches(len(rows), int(args.batch_size)), start=1):
            batch_prompt_ids = prompts_token_ids[start:end]
            batch_prefix_ids = prefixes_token_ids[start:end]
            batch_spans = move_spans[start:end]

            # Build input_ids = prompt + prefix (ragged) and left-pad to max length.
            seqs: list[torch.Tensor] = []
            prompt_lens: list[int] = []
            for p_ids, fp_ids in zip(batch_prompt_ids, batch_prefix_ids):
                full = p_ids + fp_ids
                if len(full) > int(args.max_model_len):
                    raise ValueError(f"Sequence is {len(full)} tokens (max_model_len={args.max_model_len}).")
                seqs.append(torch.tensor(full, dtype=torch.long))
                prompt_lens.append(len(p_ids))

            pad_id = int(tokenizer.pad_token_id)
            input_ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id).to("cuda")
            attn_mask = (input_ids != pad_id).to(torch.long)

            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits  # (B, L, V)
            logprobs = torch.log_softmax(logits.to(torch.float32), dim=-1)

            for i in range(end - start):
                prompt_len = int(prompt_lens[i])
                move_start, move_end = batch_spans[i]
                move_token_count = int(move_end - move_start)
                abs_start = prompt_len + int(move_start)
                abs_end = prompt_len + int(move_end)
                if abs_start <= 0:
                    raise RuntimeError("Move tokens unexpectedly start at position 0.")

                # Gather token-wise logprobs for the move span.
                token_ids = input_ids[i, abs_start:abs_end]
                prev_positions = torch.arange(abs_start - 1, abs_end - 1, device=input_ids.device, dtype=torch.long)
                chosen_lp = logprobs[i, prev_positions, token_ids]
                lp_sum = float(chosen_lp.sum().item())
                lp_mean = lp_sum / float(move_token_count) if move_token_count > 0 else float("nan")

                row_out = LogprobRow(
                    index=indices[start + i],
                    split=splits[start + i],
                    ground_truth=gts[start + i],
                    prefix_template=prefix_template,
                    prefix_text=prefix_texts[start + i],
                    move_token_count=move_token_count,
                    move_logprob_sum=lp_sum,
                    move_logprob_mean=lp_mean,
                    prompt_token_count=prompt_len,
                )
                written.append(row_out)
                f.write(json.dumps(row_out.__dict__, ensure_ascii=False) + "\n")

            elapsed = time.time() - t0
            print(f"[{batch_idx:>3}] rows {start}:{end} elapsed={elapsed/60:.1f}min")

    tmp_path.replace(out_path)
    print(f"Wrote {out_path}")

    sums = [r.move_logprob_sum for r in written if math.isfinite(r.move_logprob_sum)]
    means = [r.move_logprob_mean for r in written if math.isfinite(r.move_logprob_mean)]
    stats = {
        "model": str(args.model),
        "parquet": str(args.parquet),
        "num_rows": len(written),
        "prefix_template": prefix_template,
        "move_logprob_sum": {
            "mean": float(statistics.fmean(sums)) if sums else float("nan"),
            "median": float(statistics.median(sums)) if sums else float("nan"),
            "stdev": float(statistics.pstdev(sums)) if len(sums) >= 2 else 0.0,
            **_percentiles(sums, [1, 5, 10, 25, 50, 75, 90, 95, 99]),
        },
        "move_logprob_mean": {
            "mean": float(statistics.fmean(means)) if means else float("nan"),
            "median": float(statistics.median(means)) if means else float("nan"),
            "stdev": float(statistics.pstdev(means)) if len(means) >= 2 else 0.0,
            **_percentiles(means, [1, 5, 10, 25, 50, 75, 90, 95, 99]),
        },
    }
    print("Stats:", json.dumps(stats, indent=2))

    out_stats_json = args.out_stats_json
    if out_stats_json:
        stats_path = Path(out_stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_stats = stats_path.with_suffix(stats_path.suffix + ".tmp")
        with open(tmp_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        tmp_stats.replace(stats_path)
        print(f"Wrote {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
