#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
from jinja2 import Environment
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from recipe.chess.reward_fn import compute_score_batch


@dataclass(frozen=True)
class SampleConfig:
    model: str
    tokenizer: str | None
    parquet: str
    limit: int
    n: int
    max_prompt_length: int
    max_response_length: int
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_num_seqs: int
    temperature: float
    top_p: float
    seed: int
    chess_reward_fn: str
    logit_eps: float
    template_path: str | None
    forced_guess_mode: str
    forced_prefix_template: str
    print_outputs: bool
    output_json: str | None


def _load_rows(parquet_path: str, limit: int) -> list[dict[str, Any]]:
    dataset = ds.dataset(parquet_path, format="parquet")
    table = dataset.to_table(columns=["prompt", "reward_model"])
    rows = table.to_pylist()
    return rows[:limit]


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


def _maybe_render_prompts(rows: list[dict[str, Any]], template_path: str | None) -> list[list[dict[str, str]]]:
    if not template_path:
        return [row["prompt"] for row in rows]

    template_file = Path(template_path)
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")
    env = Environment(autoescape=False)
    template = env.from_string(template_file.read_text())

    prompts: list[list[dict[str, str]]] = []
    for row in rows:
        rm = row.get("reward_model") or {}
        fen = str(rm.get("fen") or "").strip()
        legal = _normalize_legal_moves(rm.get("legal_moves_uci"))
        prompt_text = str(template.render(FEN=fen, legal_moves_uci_list=legal))
        prompts.append([{"role": "user", "content": prompt_text}])
    return prompts


def _sanitize_json_value(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_value(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_json_value(v) for k, v in obj.items()}
    # numpy scalars
    if hasattr(obj, "item"):
        try:
            return _sanitize_json_value(obj.item())
        except Exception:
            return str(obj)
    return str(obj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--parquet", default="data/chess_puzzles/train_hard.parquet")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--n", type=int, default=1, help="Number of samples per prompt (vLLM SamplingParams.n).")
    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--max_response_length", type=int, default=512)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    ap.add_argument("--max_num_seqs", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chess_reward_fn", default="expected_score_wdl_vs_best")
    ap.add_argument("--logit_eps", type=float, default=1e-6)
    ap.add_argument(
        "--print_outputs",
        action="store_true",
        help="Print each sample's full output. If unset, print only summary stats and per-sample metrics.",
    )
    ap.add_argument(
        "--forced_guess_mode",
        choices=["none", "ground_truth", "random_legal"],
        default="none",
        help="Optional: simulate forced-prefix injection by pre-pending a forced <guess> line to the generation prompt. "
        "This is useful to sanity-check that the model continues with <think>/<uci_move> even when the guess is forced.",
    )
    ap.add_argument(
        "--forced_prefix_template",
        default="<guess> {move} </guess>",
        help="Forced-prefix template to inject when --forced_guess_mode != none (must contain '{move}').",
    )
    ap.add_argument(
        "--template_path",
        default="recipe/chess/prompt_templates/chess_rl_chessr1_prompt.jinja",
        help="If set, render prompts from this Jinja template instead of using the stored parquet prompt column.",
    )
    ap.add_argument(
        "--output_json",
        default=None,
        help="Optional: write a JSON file containing inputs/outputs/scores for each sampled prompt.",
    )
    args = ap.parse_args()

    cfg = SampleConfig(
        model=str(args.model),
        tokenizer=(str(args.tokenizer) if args.tokenizer else None),
        parquet=str(args.parquet),
        limit=int(args.limit),
        n=int(args.n),
        max_prompt_length=int(args.max_prompt_length),
        max_response_length=int(args.max_response_length),
        max_model_len=int(args.max_model_len),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_num_seqs=int(args.max_num_seqs),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        seed=int(args.seed),
        chess_reward_fn=str(args.chess_reward_fn),
        logit_eps=float(args.logit_eps),
        template_path=str(args.template_path) if args.template_path else None,
        forced_guess_mode=str(args.forced_guess_mode),
        forced_prefix_template=str(args.forced_prefix_template),
        print_outputs=bool(args.print_outputs),
        output_json=(str(args.output_json) if args.output_json else None),
    )

    if cfg.n < 1:
        raise ValueError("--n must be >= 1")

    rows = _load_rows(cfg.parquet, cfg.limit)
    prompts = _maybe_render_prompts(rows, cfg.template_path)
    reward_models = [row["reward_model"] for row in rows]

    tokenizer_model = cfg.tokenizer or cfg.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_token_ids: list[list[int]] = []
    prompt_texts: list[str] = []
    forced_prefix_texts: list[str] = []
    rng = np.random.default_rng(cfg.seed)

    if cfg.forced_guess_mode != "none" and "{move}" not in cfg.forced_prefix_template:
        raise ValueError("--forced_prefix_template must contain '{move}' when --forced_guess_mode != none.")

    for i, (messages, rm) in enumerate(zip(prompts, reward_models)):
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            raise TypeError(f"apply_chat_template returned {type(ids)} for row {i}")
        prompt_texts.append(str(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)))

        forced_prefix_text = ""
        if cfg.forced_guess_mode == "ground_truth":
            forced_move = str((rm or {}).get("ground_truth") or "").strip()
            forced_prefix_text = cfg.forced_prefix_template.format(move=forced_move)
        elif cfg.forced_guess_mode == "random_legal":
            legal = _normalize_legal_moves((rm or {}).get("legal_moves_uci"))
            forced_move = str(rng.choice(legal)) if legal else str((rm or {}).get("ground_truth") or "").strip()
            forced_prefix_text = cfg.forced_prefix_template.format(move=forced_move)

        forced_prefix_texts.append(forced_prefix_text)
        if forced_prefix_text:
            forced_ids = tokenizer.encode(forced_prefix_text, add_special_tokens=False)
            if not isinstance(forced_ids, list):
                raise TypeError("tokenizer.encode returned non-list token ids for forced prefix")
            ids = list(ids) + [int(x) for x in forced_ids]

        if len(ids) > cfg.max_prompt_length:
            raise ValueError(f"Prompt {i} is {len(ids)} tokens (max={cfg.max_prompt_length})")
        prompt_token_ids.append([int(x) for x in ids])

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

    sampling = SamplingParams(
        n=cfg.n,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=-1,
        min_p=0.0,
        max_tokens=cfg.max_response_length,
        repetition_penalty=1.0,
        detokenize=False,
    )

    outputs = llm.generate(
        prompts=[{"prompt_token_ids": ids} for ids in prompt_token_ids],
        sampling_params=sampling,
        use_tqdm=False,
    )

    decoded: list[list[str]] = []
    gen_token_lens: list[list[int]] = []
    for out in outputs:
        if len(out.outputs) != cfg.n:
            raise RuntimeError(f"Expected {cfg.n} outputs per prompt, got {len(out.outputs)}")
        dec_group: list[str] = []
        len_group: list[int] = []
        for out_i in out.outputs:
            token_ids = out_i.token_ids
            len_group.append(int(len(token_ids)))
            dec_group.append(tokenizer.decode(token_ids, skip_special_tokens=True))
        decoded.append(dec_group)
        gen_token_lens.append(len_group)

    full_texts: list[str] = []
    forced_prefix_token_lens: list[int] = []
    for prefix in forced_prefix_texts:
        if prefix:
            forced_ids = tokenizer.encode(prefix, add_special_tokens=False)
            forced_prefix_token_lens.append(int(len(forced_ids)))
        else:
            forced_prefix_token_lens.append(0)

    flat_reward_models: list[dict[str, Any]] = []
    for rm in reward_models:
        for _ in range(cfg.n):
            flat_reward_models.append(rm)

    for prefix, gen_group in zip(forced_prefix_texts, decoded, strict=True):
        for gen in gen_group:
            full_texts.append(f"{prefix}{gen}")
    scored_flat = compute_score_batch(flat_reward_models, full_texts, chess_reward_fn=cfg.chess_reward_fn, logit_eps=cfg.logit_eps)

    scored: list[list[dict[str, Any]]] = []
    idx = 0
    for _ in range(len(rows)):
        scored.append(scored_flat[idx : idx + cfg.n])
        idx += cfg.n

    if cfg.output_json:
        output_path = Path(cfg.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        samples: list[dict[str, Any]] = []
        full_text_idx = 0
        for i, (row, score_group, gen_len_group, forced_len, prompt_ids, prompt_text) in enumerate(
            zip(rows, scored, gen_token_lens, forced_prefix_token_lens, prompt_token_ids, prompt_texts, strict=True)
        ):
            rm = row.get("reward_model") or {}
            completions: list[dict[str, Any]] = []
            for j, (score_info, gen_len) in enumerate(zip(score_group, gen_len_group, strict=True)):
                full_text = full_texts[full_text_idx]
                full_text_idx += 1
                completions.append(
                    {
                        "completion_index": j,
                        "output_text_full": full_text,
                        "token_lens": {
                            "forced_prefix": int(forced_len),
                            "generated": int(gen_len),
                            "total": int(gen_len + forced_len),
                        },
                        "score": score_info,
                    }
                )
            samples.append(
                {
                    "index": i,
                    "fen": (rm.get("fen") or "").strip(),
                    "legal_moves_uci": _normalize_legal_moves(rm.get("legal_moves_uci")),
                    "ground_truth": (rm.get("ground_truth") or "").strip(),
                    "prompt_user_content": prompts[i][0]["content"] if prompts[i] else "",
                    "prompt_chat_template": prompt_text,
                    "forced_prefix_text": forced_prefix_texts[i],
                    "token_lens": {"prompt": int(len(prompt_ids))},
                    "completions": completions,
                }
            )

        payload = _sanitize_json_value(
            {
                "config": cfg.__dict__,
                "samples": samples,
            }
        )
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    # Print one representative rendered prompt (template is shared across rows except for FEN/moves).
    try:
        prompt_preview = tokenizer.apply_chat_template(prompts[0], add_generation_prompt=True, tokenize=False)
    except Exception:
        prompt_preview = prompts[0][0]["content"]
    print("=" * 120)
    print("[PROMPT TEMPLATE PREVIEW (example 0)]")
    print(prompt_preview)
    print("=" * 120)

    penalty_counts: dict[str, int] = {}
    n_format_ok = 0
    n_any_format_ok = 0
    total_token_lens: list[int] = []
    gen_token_lens_flat: list[int] = []
    forced_prefix_token_lens_flat: list[int] = []

    full_idx = 0
    for i, (row, decoded_group, scored_group, gen_len_group, forced_len) in enumerate(
        zip(rows, decoded, scored, gen_token_lens, forced_prefix_token_lens, strict=True)
    ):
        rm = row.get("reward_model") or {}
        gt = (rm.get("ground_truth") or "").strip()

        group_format_ok = 0
        for j, (out_text, info, gen_len) in enumerate(zip(decoded_group, scored_group, gen_len_group, strict=True)):
            penalty_reason = (info.get("penalty_reason") or "").strip()
            penalty_counts[penalty_reason] = penalty_counts.get(penalty_reason, 0) + 1
            format_ok = float(info.get("format_reward") or 0.0) >= 1.0
            if format_ok:
                n_format_ok += 1
                group_format_ok += 1

            total_len = int(gen_len + forced_len)
            total_token_lens.append(total_len)
            gen_token_lens_flat.append(int(gen_len))
            forced_prefix_token_lens_flat.append(int(forced_len))

            forced_prefix = forced_prefix_texts[i]
            forced_hint = ""
            if forced_prefix:
                forced_hint = f" forced_prefix={forced_prefix!r}"
            if cfg.n == 1:
                label = f"[sample {i}]"
            else:
                label = f"[sample {i} completion {j}]"
            print(
                f"{label} gt={gt} gen_tokens={gen_len} forced_tokens={forced_len} total_tokens={total_len} "
                f"format_ok={int(format_ok)} penalty={penalty_reason!r}{forced_hint}"
            )
            if cfg.print_outputs:
                print("-" * 120)
                print(full_texts[full_idx])
                print("-" * 120)
            full_idx += 1
            print(
                "score:",
                json.dumps(
                    {k: info.get(k) for k in ["penalty_reason", "format_reward", "pred_move", "guess_present", "guess_uci", "acc", "move_expected_score", "total_reward"]},
                    ensure_ascii=False,
                ),
            )
        if group_format_ok > 0:
            n_any_format_ok += 1

    if total_token_lens:
        arr = np.array(total_token_lens, dtype=np.int32)
        gen_arr = np.array(gen_token_lens_flat, dtype=np.int32)
        forced_arr = np.array(forced_prefix_token_lens_flat, dtype=np.int32)
        print()
        print("Summary:")
        print(f"  format_ok_frac={n_format_ok}/{len(total_token_lens)} = {n_format_ok/len(total_token_lens):.3f}")
        print(f"  pass_at_n_format={n_any_format_ok}/{len(rows)} = {n_any_format_ok/len(rows):.3f}")
        print(f"  penalty_counts={json.dumps(penalty_counts, ensure_ascii=False, sort_keys=True)}")
        print(
            "  token_lengths_total:",
            json.dumps(
                {
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "p90": float(np.percentile(arr, 90)),
                    "min": int(arr.min()),
                    "max": int(arr.max()),
                },
                ensure_ascii=False,
            ),
        )
        print(
            "  token_lengths_generated:",
            json.dumps(
                {
                    "mean": float(gen_arr.mean()),
                    "median": float(np.median(gen_arr)),
                    "p90": float(np.percentile(gen_arr, 90)),
                    "min": int(gen_arr.min()),
                    "max": int(gen_arr.max()),
                },
                ensure_ascii=False,
            ),
        )
        print(
            "  token_lengths_forced_prefix:",
            json.dumps(
                {
                    "mean": float(forced_arr.mean()),
                    "median": float(np.median(forced_arr)),
                    "p90": float(np.percentile(forced_arr, 90)),
                    "min": int(forced_arr.min()),
                    "max": int(forced_arr.max()),
                },
                ensure_ascii=False,
            ),
        )
    else:
        print("Summary: no outputs decoded (empty batch?)")


if __name__ == "__main__":
    main()
