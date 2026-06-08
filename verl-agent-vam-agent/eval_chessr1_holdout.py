"""Held-out eval: Chess-R1 puzzles as 'unseen lesson-style' generalization test.

Treats each Chess-R1 single-step puzzle as a single-turn chess task using the
same <think>/<action> protocol as chesslesson training. Tasks come from
Chess-R1 sharded parquets (chess-rl-C224 data dirs), bucketed by stage:
  - stage1_mate         (mate puzzles)    → matches Phase 3+ tactics
  - stage2_fundamental  (fundamental)     → matches Phase 2+ pieces

This gives a 'true OOD same-format' eval: model trained on chesslesson lessons,
tested on Chess-R1 puzzles the model never saw. Action correctness = exact
UCI match against the puzzle's `best_move_uci` ground truth.

Usage:
  python eval_chessr1_holdout.py \
      --model $HOME/models/chesslesson_curriculum_stage/phase2 \
      --parquet $HOME/chess/chess-rl-C224/data/chess_puzzles_stage2_fundamental/train_0.parquet \
      --n-samples 200 \
      --output eval_results/phase2_chessr1_fundamental_holdout.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)


def parse_action(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if "<think>" not in text or "</think>" not in text:
        return ""
    m = _ACTION_RE.search(text)
    if not m:
        return ""
    return m.group(1).strip().lower().replace(" ", "")


def build_puzzle_user_msg(rm: dict, max_legal_shown: int = 64) -> str:
    """Render a Chess-R1 puzzle as a chesslesson-style single-turn prompt."""
    fen = rm["fen"]
    legal = list(rm["legal_moves_uci"])
    if len(legal) > max_legal_shown:
        legal = legal[:max_legal_shown]
    lines = [
        "The current board position is represented in FEN notation.",
        "Task: Find the single best move from the legal moves below. "
        "Output one UCI move, e.g. <action>e2e4</action>.",
        f"FEN: {fen}",
        f"Legal moves (UCI): {', '.join(legal)}",
        "",
        "Reason step by step inside <think></think>, then put your answer in <action></action>.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--parquet", required=True, help="Chess-R1 sharded parquet path")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of puzzles to evaluate (random sample)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    import pandas as pd
    print(f"[load] parquet={args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"  rows: {len(df)}")

    rng = __import__("random").Random(args.seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    indices = indices[:args.n_samples]
    rows = df.iloc[indices].to_dict("records")
    print(f"  sampled: {len(rows)}")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[load] model={args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=1,
    )

    # Render prompts
    prompts = []
    for r in rows:
        rm = r["reward_model"]
        if not isinstance(rm, dict):
            rm = json.loads(rm)
        # legal_moves_uci comes back as a numpy array sometimes; coerce
        if hasattr(rm["legal_moves_uci"], "tolist"):
            rm["legal_moves_uci"] = rm["legal_moves_uci"].tolist()
        user_msg = build_puzzle_user_msg(rm)
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append((rendered, rm, user_msg))

    print(f"[generate] {len(prompts)} prompts")
    outs = llm.generate([p[0] for p in prompts], sp)

    # Score
    results = []
    for (_, rm, user_msg), out in zip(prompts, outs):
        response = out.outputs[0].text
        action = parse_action(response)
        gold = str(rm["best_move_uci"]).strip().lower()
        legal = [m.lower() for m in rm["legal_moves_uci"]]
        is_legal = action in legal
        is_correct = action == gold
        parse_ok = bool(action)
        results.append({
            "fen": rm["fen"],
            "best_move": gold,
            "parsed_action": action,
            "parse_ok": parse_ok,
            "is_legal": is_legal,
            "is_correct": is_correct,
            "response_preview": response[:200],
        })

    n = len(results)
    parse_rate = sum(1 for r in results if r["parse_ok"]) / n
    legal_rate = sum(1 for r in results if r["is_legal"]) / n
    correct_rate = sum(1 for r in results if r["is_correct"]) / n
    legal_given_parsed = sum(1 for r in results if r["parse_ok"] and r["is_legal"]) / max(1, sum(1 for r in results if r["parse_ok"]))
    correct_given_legal = sum(1 for r in results if r["is_legal"] and r["is_correct"]) / max(1, sum(1 for r in results if r["is_legal"]))

    summary = {
        "config": vars(args),
        "n": n,
        "parse_rate": parse_rate,
        "legal_rate": legal_rate,
        "correct_rate": correct_rate,
        "legal_given_parsed": legal_given_parsed,
        "correct_given_legal": correct_given_legal,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "results": results}
    out_path.write_text(json.dumps(payload, indent=2))

    print()
    print("=" * 60)
    print(f"CHESS-R1 HOLD-OUT EVAL  (n={n})")
    print("=" * 60)
    print(f"  parse_rate (output has <action>):    {parse_rate:.3f}")
    print(f"  legal_rate (action is in legal):     {legal_rate:.3f}")
    print(f"  correct_rate (action == best_move):  {correct_rate:.3f}  ← key metric")
    print(f"  legal_given_parsed:                  {legal_given_parsed:.3f}")
    print(f"  correct_given_legal:                 {correct_given_legal:.3f}")
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
