#!/usr/bin/env python3
"""
Position-sweep evaluation for selection framing.

For each test row (full legal list), we:
  - choose the global μ-best move as target
  - shuffle all other legal moves with a fixed per-row permutation
  - insert the target at position K (0..n-1)
  - run vLLM sampling and compute pass@1 (success_rate) and pass@N

This isolates the effect of target position in the candidate list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from jinja2 import Environment
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from recipe.chess.reward_fn import _to_uci

_UCI_MOVE_TAG_RE = re.compile(r"<\s*uci_move\s*>(?P<ans>[\s\S]*?)<\s*/\s*uci_move\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class RowData:
    row_id: int
    fen: str
    legal_moves_uci: list[str]
    mu_map: dict[str, float]


def _sanitize_model_name(model: str) -> str:
    safe = model.replace("/", "__").replace(":", "_")
    safe = "".join(ch if (ch.isalnum() or ch in "._-__") else "_" for ch in safe)
    return safe


def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _stable_int_hash(*parts: str, mod: int) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], "big") % int(mod)


def _load_template(template_path: str) -> Any:
    template_file = Path(template_path)
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")
    env = Environment(autoescape=False)
    return env.from_string(template_file.read_text(encoding="utf-8"))


def _normalize_moves(moves: Any) -> list[str]:
    if moves is None:
        return []
    if isinstance(moves, str):
        s = moves.strip().lower()
        return [s] if s else []
    out: list[str] = []
    try:
        for m in moves:
            s = str(m).strip().lower()
            if s:
                out.append(s)
    except Exception:
        return []
    return out


def _load_rows(parquet_path: str, limit_rows: int) -> list[RowData]:
    table = pq.read_table(parquet_path, columns=["reward_model", "extra_info"])
    rows = table.to_pylist()
    if limit_rows < 0:
        raise ValueError("--limit_rows must be >= 0")
    rows = rows[:limit_rows] if limit_rows else rows

    out: list[RowData] = []
    for row in rows:
        rm = row.get("reward_model") or {}
        ei = row.get("extra_info") or {}
        row_id = int(ei.get("index")) if ei.get("index") is not None else int(len(out))
        fen = str(rm.get("fen") or "").strip()
        legal_moves = _normalize_moves(rm.get("legal_moves_uci"))

        mu_json = rm.get("move_expected_scores_json")
        if isinstance(mu_json, str) and mu_json.strip():
            mu_raw = json.loads(mu_json)
        else:
            mu_raw = mu_json
        mu_map: dict[str, float] = {}
        if isinstance(mu_raw, dict):
            for k, v in mu_raw.items():
                key = str(k).strip().lower()
                if not key:
                    continue
                try:
                    mu_map[key] = float(v)
                except Exception:
                    continue
        if not mu_map:
            mv_json = rm.get("move_values_json")
            if isinstance(mv_json, str) and mv_json.strip():
                mv_raw = json.loads(mv_json)
            else:
                mv_raw = mv_json
            if isinstance(mv_raw, dict):
                for k, v in mv_raw.items():
                    key = str(k).strip().lower()
                    if not key:
                        continue
                    try:
                        mu_map[key] = float(v)
                    except Exception:
                        continue

        if not legal_moves:
            raise ValueError(f"Row {row_id}: empty legal_moves_uci")
        if not mu_map:
            raise ValueError(f"Row {row_id}: empty mu_map (expected expected_scores or move_values)")
        out.append(RowData(row_id=row_id, fen=fen, legal_moves_uci=legal_moves, mu_map=mu_map))
    return out


def _best_move_by_mu(mu_map: dict[str, float], moves: Iterable[str]) -> str:
    best_move = ""
    best_mu = -float("inf")
    for mv in moves:
        key = str(mv).strip().lower()
        mu = float(mu_map.get(key, -float("inf")))
        if (mu > best_mu) or (mu == best_mu and (not best_move or key < best_move)):
            best_move = key
            best_mu = mu
    if not best_move:
        raise ValueError("Empty move list when selecting best move")
    return best_move


def _parse_pred_move(text: str) -> tuple[str, str]:
    matches = list(_UCI_MOVE_TAG_RE.finditer(text or ""))
    if not matches:
        return "", "format_error"
    payload = matches[0].group("ans") or ""
    pred = _to_uci(payload)
    if pred is None:
        return "", "bad_move"
    if len(matches) != 1:
        return pred, "format_error"
    return pred, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--template_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit_rows", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples_per_prompt", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_prompt_tokens", type=int, default=1024)
    ap.add_argument("--max_response_tokens", type=int, default=256)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_num_seqs", type=int, default=2048)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--max_k", type=int, default=-1)
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--mode", choices=["sweep_shuffle", "original_order", "both"], default="both")
    ap.add_argument("--use_chat_template", action="store_true")
    args = ap.parse_args()

    rows = _load_rows(str(args.parquet), limit_rows=int(args.limit_rows))
    if not rows:
        raise SystemExit(f"No rows loaded (parquet={args.parquet}, limit_rows={args.limit_rows})")

    template = _load_template(str(args.template_path))
    template_hash = _sha256_text(Path(args.template_path).read_text(encoding="utf-8"))[:12]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"results_shard{int(args.shard_idx)}of{int(args.num_shards)}.jsonl"

    seen_keys: set[str] = set()
    if results_path.exists() and not args.no_resume:
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                key = str(rec.get("key") or "")
                if key:
                    seen_keys.add(key)

    # Tokenizer + vLLM
    tokenizer_model = str(args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=str(args.model),
        tokenizer=tokenizer_model,
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

    sampling_params = SamplingParams(
        n=int(args.samples_per_prompt),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_response_tokens),
    )

    do_sweep = args.mode in ("sweep_shuffle", "both")
    do_original = args.mode in ("original_order", "both")

    tasks: list[dict[str, Any]] = []
    for row in rows:
        if int(row.row_id) % int(args.num_shards) != int(args.shard_idx):
            continue
        legal = list(row.legal_moves_uci)
        target = _best_move_by_mu(row.mu_map, legal)
        if target not in legal:
            raise ValueError(f"Row {row.row_id}: target {target} not in legal list")

        others = [m for m in legal if m != target]
        row_seed = _stable_int_hash(str(args.seed), str(row.row_id), "pos-sweep", mod=2**31)
        rng = random.Random(int(row_seed))
        rng.shuffle(others)

        n = len(legal)
        if do_sweep:
            max_k = n - 1 if int(args.max_k) < 0 else min(int(args.max_k), n - 1)
            for k in range(max_k + 1):
                candidate = others[:k] + [target] + others[k:]
                key = f"{row.row_id}__sweep__k{k}"
                if key in seen_keys:
                    continue
                prompt_text = str(
                    template.render(
                        FEN=row.fen,
                        legal_moves_uci_list=legal,
                        considered_moves_uci_list=candidate,
                    )
                )
                if args.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                    messages = [{"role": "user", "content": prompt_text}]
                    prompt_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                tasks.append(
                    {
                        "key": key,
                        "row_id": int(row.row_id),
                        "mode": "sweep_shuffle",
                        "k_pos": int(k),
                        "n_considered": int(n),
                        "target_move": target,
                        "target_rank_legal": int(legal.index(target)) + 1,
                        "prompt": prompt_text,
                        "candidate": candidate,
                    }
                )
        # Original-order mode: use the canonical legal move order without shuffling.
        if do_original:
            orig_key = f"{row.row_id}__original"
            if orig_key not in seen_keys:
                orig_candidate = list(legal)
                orig_k = int(legal.index(target))
                orig_prompt = str(
                    template.render(
                        FEN=row.fen,
                        legal_moves_uci_list=legal,
                        considered_moves_uci_list=orig_candidate,
                    )
                )
                if args.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                    messages = [{"role": "user", "content": orig_prompt}]
                    orig_prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                tasks.append(
                    {
                        "key": orig_key,
                        "row_id": int(row.row_id),
                        "mode": "original_order",
                        "k_pos": orig_k,
                        "n_considered": int(n),
                        "target_move": target,
                        "target_rank_legal": int(legal.index(target)) + 1,
                        "prompt": orig_prompt,
                        "candidate": orig_candidate,
                    }
                )

    if not tasks:
        print("[OK] No new tasks to run (all cached).")
        return 0

    print(f"[RUN] shard {args.shard_idx}/{args.num_shards}: tasks={len(tasks)}")
    start_time = time.time()

    with results_path.open("a", encoding="utf-8") as f:
        for i in range(0, len(tasks), int(args.batch_size)):
            batch = tasks[i : i + int(args.batch_size)]
            prompts = [t["prompt"] for t in batch]
            outputs = llm.generate(prompts, sampling_params)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM output size mismatch")

            for task, out in zip(batch, outputs):
                pred_moves: list[str] = []
                n_format_error = 0
                n_bad_move = 0
                n_out_of_subset = 0
                n_correct = 0

                for sample in out.outputs:
                    pred, reason = _parse_pred_move(sample.text)
                    if not pred:
                        if reason == "bad_move":
                            n_bad_move += 1
                        else:
                            n_format_error += 1
                        continue
                    if pred not in task["candidate"]:
                        n_out_of_subset += 1
                        continue
                    pred_moves.append(pred)
                    if pred == task["target_move"]:
                        n_correct += 1

                n_samples = int(args.samples_per_prompt)
                success_rate = float(n_correct) / float(n_samples) if n_samples else 0.0
                pass_at_k = 1.0 if n_correct > 0 else 0.0

                rec = {
                    "key": task["key"],
                    "row_id": task["row_id"],
                    "mode": task.get("mode", ""),
                    "k_pos": task["k_pos"],
                    "n_considered": task["n_considered"],
                    "target_move": task["target_move"],
                    "target_rank_legal": task["target_rank_legal"],
                    "success_rate": success_rate,
                    "pass_at_k": pass_at_k,
                    "n_samples": n_samples,
                    "n_correct": n_correct,
                    "n_format_error": n_format_error,
                    "n_bad_move": n_bad_move,
                    "n_out_of_subset": n_out_of_subset,
                }
                f.write(json.dumps(rec) + "\n")
            if (i // int(args.batch_size)) % 10 == 0:
                f.flush()

    elapsed = time.time() - start_time
    print(f"[DONE] shard {args.shard_idx}: elapsed_sec={elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
