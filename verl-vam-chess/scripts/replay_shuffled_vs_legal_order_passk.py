#!/usr/bin/env python3
"""
Replay base-model sampling on a subset of `filter_groups` rejected groups to test
the effect of *Allowed-move order* in the prompt.

We target rejected groups that were logged by the rerun instrumentation:
  <evidence_root>/files/rejected_group_summaries/*.jsonl

and filter to groups that satisfy:
  - all_valid == True
  - all_suboptimal_move == True     (no rollout hit the μ-target)
  - pred_move_unique > 1            (multiple different predicted moves)
  - n_considered > 6 and n_considered == n_legal (full-legal candidate set)

For each sampled row_id, we evaluate pass@8 under two prompt variants:
  (A) "shuffled": Allowed moves list uses the dataset's considered_moves_uci order (shuffled).
  (B) "legal_order": Allowed moves list is in the same order as Legal moves.

This tests whether shuffling the Allowed list destroys a shortcut the model may be using
(e.g. implicit reliance on the engine's legal-move ordering).

Outputs:
  <evidence_root>/investigation/shuffle_vs_legal_order_replay/
    - sample_manifest.json
    - per_prompt_results.csv
    - summary.json

Local run (per repo policy):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 conda run -n verl \\
    python3 scripts/replay_shuffled_vs_legal_order_passk.py \\
      --evidence-root outputs/wandb/rerun_full_2emjykpq \\
      --train-dataset data/chess_puzzles/train_hard.parquet \\
      --limit 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from jinja2 import Environment
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Ensure local imports resolve when the script is run directly.
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipe.chess.reward_fn import compute_score


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def _stable_int_hash(*parts: str, mod: int) -> int:
    s = "||".join([str(p) for p in parts])
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % int(mod)


def _normalize_moves(moves: Any) -> list[str]:
    if moves is None:
        return []
    if isinstance(moves, str):
        s = moves.strip()
        return [s.lower()] if s else []
    try:
        return [str(m).strip().lower() for m in moves if str(m).strip()]
    except Exception:
        return []


def _load_reward_models_by_fen(parquet_path: str) -> dict[str, dict[str, Any]]:
    dataset = ds.dataset(parquet_path, format="parquet")
    table = dataset.to_table(columns=["reward_model"])
    out: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        rm = row.get("reward_model") or {}
        fen = str(rm.get("fen") or "").strip()
        if not fen:
            continue
        if fen in out:
            raise RuntimeError(f"Duplicate FEN in dataset: {fen}")
        out[fen] = dict(rm)
    return out


def _render_prompt_text(template_text: str, *, fen: str, legal: list[str], allowed: list[str]) -> str:
    env = Environment(autoescape=False)
    template = env.from_string(template_text)
    return str(template.render(FEN=fen, legal_moves_uci_list=legal, considered_moves_uci_list=allowed))


def _build_prompt_token_ids(tokenizer: Any, prompt_text: str, *, max_prompt_tokens: int) -> list[int]:
    messages = [{"role": "user", "content": prompt_text}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if not isinstance(ids, list):
        raise TypeError(f"apply_chat_template returned {type(ids)}")
    if len(ids) > int(max_prompt_tokens):
        raise ValueError(f"Prompt is {len(ids)} tokens (max={int(max_prompt_tokens)}).")
    return [int(x) for x in ids]


@dataclass(frozen=True)
class Example:
    row_id: int
    uid: str
    step: int
    fen: str
    legal_moves: list[str]
    allowed_moves_shuffled: list[str]


def _select_examples(rejected_dir: Path, *, limit: int, seed: int) -> list[Example]:
    # Sample unique row_ids to avoid repeats across epochs.
    seen_row_ids: set[int] = set()
    pool: list[Example] = []

    files = sorted(rejected_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No rejected_group_summaries JSONLs under {rejected_dir}")

    for path in files:
        for rec_idx, rec in _iter_jsonl(path):
            if not bool(rec.get("all_valid")):
                continue
            if not bool(rec.get("all_suboptimal_move")):
                continue
            if bool(rec.get("all_best_move")):
                continue
            try:
                pred_move_unique = int(rec.get("pred_move_unique"))
            except Exception:
                continue
            if pred_move_unique <= 1:
                continue  # not a tie

            try:
                row_id = int(rec.get("row_id"))
            except Exception:
                continue
            if row_id in seen_row_ids:
                continue

            legal = _normalize_moves(rec.get("legal_moves_uci"))
            allowed = _normalize_moves(rec.get("considered_moves_uci"))
            if not legal or not allowed:
                continue
            if len(allowed) <= 6:
                continue
            # Full legal (Case A in dataset construction), but allow order differences.
            if len(allowed) != len(legal):
                continue
            if set(allowed) != set(legal):
                continue

            fen = str(rec.get("fen") or "").strip()
            if not fen:
                continue

            pool.append(
                Example(
                    row_id=row_id,
                    uid=str(rec.get("uid") or ""),
                    step=int(rec.get("step") or 0),
                    fen=fen,
                    legal_moves=legal,
                    allowed_moves_shuffled=allowed,
                )
            )
            seen_row_ids.add(row_id)

    if len(pool) < int(limit):
        raise RuntimeError(f"Found only {len(pool)} eligible unique row_ids; need limit={int(limit)}.")

    rng = np.random.default_rng(int(seed))
    idxs = rng.choice(len(pool), size=int(limit), replace=False).tolist()
    return [pool[i] for i in idxs]


def _iter_batches(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        yield start, end


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _mcnemar_exact(n01: int, n10: int) -> float:
    # Two-sided exact McNemar via Binomial(n01+n10, 0.5) on the smaller tail.
    # Avoid scipy dependency.
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    from math import comb

    p_tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return float(min(1.0, 2.0 * p_tail))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-root", type=Path, required=True)
    ap.add_argument("--train-dataset", type=str, default="data/chess_puzzles/train_hard.parquet")
    ap.add_argument("--template-path", type=str, default="recipe/chess/prompt_templates/select_prompt.jinja")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_prompt_tokens", type=int, default=1024)
    ap.add_argument("--max_response_tokens", type=int, default=4096)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    ap.add_argument("--max_num_seqs", type=int, default=512)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--chess_reward_fn", type=str, default="expected_score_wdl_vs_best")
    args = ap.parse_args()

    evidence_root: Path = args.evidence_root
    rejected_dir = evidence_root / "files" / "rejected_group_summaries"
    if not rejected_dir.exists():
        raise FileNotFoundError(f"Missing rejected_group_summaries dir: {rejected_dir}")

    template_path = Path(args.template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    template_text = template_path.read_text(encoding="utf-8")

    out_dir = evidence_root / "investigation" / "shuffle_vs_legal_order_replay"
    _ensure_dir(out_dir)

    print(f"[LOAD] Loading reward_models by FEN from {args.train_dataset} ...")
    fen2rm = _load_reward_models_by_fen(str(args.train_dataset))
    print(f"[OK] Loaded {len(fen2rm)} unique FEN reward_models.")

    print(f"[SELECT] Sampling limit={int(args.limit)} examples from rejected_group_summaries ...")
    examples = _select_examples(rejected_dir, limit=int(args.limit), seed=int(args.seed))
    print(f"[OK] Selected {len(examples)} unique row_ids.")

    manifest = [
        {
            "row_id": e.row_id,
            "uid": e.uid,
            "step": e.step,
            "fen": e.fen,
            "n_legal": len(e.legal_moves),
            "n_allowed": len(e.allowed_moves_shuffled),
        }
        for e in examples
    ]
    (out_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {out_dir / 'sample_manifest.json'}")

    tokenizer_model = str(args.tokenizer or args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build tasks: two variants per example.
    tasks: list[dict[str, Any]] = []
    skipped = 0
    for e in examples:
        rm_base = fen2rm.get(e.fen)
        if rm_base is None:
            skipped += 1
            continue
        prompt_shuf = _render_prompt_text(
            template_text, fen=e.fen, legal=e.legal_moves, allowed=e.allowed_moves_shuffled
        )
        prompt_legal = _render_prompt_text(template_text, fen=e.fen, legal=e.legal_moves, allowed=e.legal_moves)
        try:
            ids_shuf = _build_prompt_token_ids(tokenizer, prompt_shuf, max_prompt_tokens=int(args.max_prompt_tokens))
            ids_legal = _build_prompt_token_ids(tokenizer, prompt_legal, max_prompt_tokens=int(args.max_prompt_tokens))
        except Exception as ex:
            print(f"[WARN] row_id={e.row_id}: prompt tokenization failed: {ex}; skipping.")
            skipped += 1
            continue
        tasks.append(
            {
                "row_id": e.row_id,
                "uid": e.uid,
                "step": e.step,
                "variant": "shuffled",
                "prompt_token_ids": ids_shuf,
                "legal_moves": e.legal_moves,
                "allowed_moves": e.allowed_moves_shuffled,
                "reward_model": rm_base,
            }
        )
        tasks.append(
            {
                "row_id": e.row_id,
                "uid": e.uid,
                "step": e.step,
                "variant": "legal_order",
                "prompt_token_ids": ids_legal,
                "legal_moves": e.legal_moves,
                "allowed_moves": e.legal_moves,
                "reward_model": rm_base,
            }
        )

    if skipped:
        print(f"[WARN] Skipped {skipped} examples (missing FEN or tokenization).")
    if not tasks:
        raise RuntimeError("No tasks to run after filtering.")

    n_per_prompt = int(args.n)
    if n_per_prompt <= 0:
        raise ValueError("--n must be > 0")

    # vLLM schedules sequences; enforce batch_size * n <= max_num_seqs.
    batch_size = max(1, min(64, int(args.max_num_seqs) // n_per_prompt))
    print(f"[RUN] prompts={len(tasks)} (2x{len(tasks)//2}), n={n_per_prompt}, batch_size={batch_size}")

    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        dtype=str(args.dtype),
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        seed=int(args.seed),
        max_num_seqs=int(args.max_num_seqs),
    )

    base_sampling_kwargs = dict(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=-1,
        min_p=0.0,
        max_tokens=int(args.max_response_tokens),
        repetition_penalty=1.0,
        detokenize=True,
    )

    results: list[dict[str, Any]] = []

    for start, end in _iter_batches(len(tasks), batch_size):
        batch = tasks[start:end]
        vllm_inputs = [{"prompt_token_ids": t["prompt_token_ids"]} for t in batch]
        sampling_params = []
        for t in batch:
            seed_i = _stable_int_hash(str(args.seed), str(t["row_id"]), str(t["variant"]), mod=2**31)
            sampling_params.append(SamplingParams(seed=int(seed_i), n=n_per_prompt, **base_sampling_kwargs))

        outs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
        if len(outs) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} outputs, got {len(outs)}")

        for t, out in zip(batch, outs, strict=True):
            if len(out.outputs) != n_per_prompt:
                raise RuntimeError(
                    f"row_id={t['row_id']} variant={t['variant']}: expected n={n_per_prompt}, got {len(out.outputs)}"
                )

            rm = dict(t["reward_model"])
            rm["considered_moves_uci"] = list(t["allowed_moves"])
            gt = str(rm.get("ground_truth") or "").strip().lower()

            pass8_target = False
            pass8_gt = False
            format_ok = 0
            in_subset = 0
            penalty_applied = 0
            pred_moves_valid: list[str] = []
            target_move: Optional[str] = None

            rollouts = []
            for o in out.outputs:
                res = compute_score(
                    data_source=rm,
                    solution_str=o.text,
                    ground_truth=gt,
                    chess_reward_fn=str(args.chess_reward_fn),
                )
                if target_move is None:
                    target_move = str(res.get("target_move") or "")

                acc = float(res.get("acc") or 0.0)
                exact = float(res.get("exact_match") or 0.0)
                pa = bool(res.get("penalty_applied") or False)
                if acc >= 1.0 and not pa:
                    pass8_target = True
                if exact >= 1.0 and not pa:
                    pass8_gt = True

                fr = float(res.get("format_reward") or 0.0)
                if fr >= 1.0:
                    format_ok += 1
                if bool(res.get("in_subset") or False):
                    in_subset += 1
                if pa:
                    penalty_applied += 1
                pred = str(res.get("pred_move") or "")
                if pred and not pa:
                    pred_moves_valid.append(pred)

                rollouts.append(
                    {
                        "pred_move": pred,
                        "acc": acc,
                        "exact_match": exact,
                        "score": float(res.get("score") or 0.0),
                        "penalty_reason": str(res.get("penalty_reason") or ""),
                    }
                )

            results.append(
                {
                    "row_id": int(t["row_id"]),
                    "uid": str(t["uid"]),
                    "step": int(t["step"]),
                    "variant": str(t["variant"]),
                    "n_legal": int(len(t["legal_moves"])),
                    "n_allowed": int(len(t["allowed_moves"])),
                    "target_move": str(target_move or ""),
                    "ground_truth": gt,
                    "pass8_target": int(pass8_target),
                    "pass8_ground_truth": int(pass8_gt),
                    "format_ok_rate": format_ok / n_per_prompt,
                    "in_subset_rate": in_subset / n_per_prompt,
                    "penalty_rate": penalty_applied / n_per_prompt,
                    "unique_valid_moves": int(len(set(pred_moves_valid)) if pred_moves_valid else 0),
                    "rollouts_json": json.dumps(rollouts, ensure_ascii=False),
                }
            )

    df = pd.DataFrame(results)
    out_csv = out_dir / "per_prompt_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv}")

    piv = df.pivot_table(
        index="row_id",
        columns="variant",
        values=["pass8_target", "pass8_ground_truth"],
        aggfunc="max",
        fill_value=0,
    )
    piv.columns = [f"{a}__{b}" for (a, b) in piv.columns]
    piv = piv.reset_index()

    def _paired_table(col_a: str, col_b: str) -> dict[str, int]:
        a = piv[col_a].astype(int).to_numpy()
        b = piv[col_b].astype(int).to_numpy()
        return {
            "n": int(len(piv)),
            "00": int(np.sum((a == 0) & (b == 0))),
            "01": int(np.sum((a == 0) & (b == 1))),
            "10": int(np.sum((a == 1) & (b == 0))),
            "11": int(np.sum((a == 1) & (b == 1))),
        }

    summary: dict[str, Any] = {
        "n_row_ids": int(len(piv)),
        "config": {
            "model": str(args.model),
            "n": int(n_per_prompt),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_prompt_tokens": int(args.max_prompt_tokens),
            "max_response_tokens": int(args.max_response_tokens),
            "seed": int(args.seed),
        },
    }

    for metric in ["pass8_target", "pass8_ground_truth"]:
        col_shuf = f"{metric}__shuffled"
        col_legal = f"{metric}__legal_order"
        tab = _paired_table(col_shuf, col_legal)
        hits_shuf = int(piv[col_shuf].sum())
        hits_legal = int(piv[col_legal].sum())
        summary[metric] = {
            "shuffled": {
                "hits": hits_shuf,
                "rate": hits_shuf / tab["n"] if tab["n"] else None,
                "wilson95": _wilson_ci(hits_shuf, tab["n"]),
            },
            "legal_order": {
                "hits": hits_legal,
                "rate": hits_legal / tab["n"] if tab["n"] else None,
                "wilson95": _wilson_ci(hits_legal, tab["n"]),
            },
            "paired_table": tab,
            "mcnemar_exact_p": _mcnemar_exact(tab["01"], tab["10"]),
            "delta_rate_legal_minus_shuffled": (hits_legal - hits_shuf) / tab["n"] if tab["n"] else None,
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {out_dir / 'summary.json'}")

    for metric in ["pass8_target", "pass8_ground_truth"]:
        s = summary[metric]
        tab = s["paired_table"]
        print(f"\n[SUMMARY] {metric}")
        print(f"  shuffled:    {s['shuffled']['hits']}/{summary['n_row_ids']} = {s['shuffled']['rate']:.3f}")
        print(f"  legal_order: {s['legal_order']['hits']}/{summary['n_row_ids']} = {s['legal_order']['rate']:.3f}")
        print(f"  paired (00/01/10/11): {tab['00']}/{tab['01']}/{tab['10']}/{tab['11']} (p={s['mcnemar_exact_p']:.4g})")
        print(f"  delta (legal - shuffled): {s['delta_rate_legal_minus_shuffled']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

