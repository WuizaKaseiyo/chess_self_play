"""Zero-shot eval of a base model on chesslesson tasks.

Loads instructions.jsonl + coordinates.jsonl, runs the model with vLLM,
parses the JSON output, scores via reward.evaluate_by_id, and aggregates
break-downs by single-move vs multi-move, by stage, and by category.

Usage:
  python eval_chesslesson_base.py \
      --model $HOME/models/Qwen2.5-7B-Instruct \
      --output eval_results/base_qwen7b.json \
      --temperature 0.6 \
      --max-tokens 1024
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("$HOME/chess/chess-agent")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "chess_game" / "chesslesson"))

from reward import evaluate_by_id


def load_tasks(include_coord: bool = True):
    inst = [json.loads(l) for l in (REPO / "chess_game/chesslesson/instructions.jsonl").open()]
    for r in inst:
        r["kind"] = "lesson"
    coords = []
    if include_coord:
        coords = [json.loads(l) for l in (REPO / "chess_game/chesslesson/coordinates.jsonl").open()]
        for r in coords:
            r["kind"] = "coordinate"
    return inst, coords


def extract_json_from_response(text: str) -> dict | None:
    """Find the last {...} JSON object on the last non-empty line."""
    # Try every line from bottom up
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        # Try fenced or raw
        for candidate in (line, *re.findall(r"\{[^{}]*\}", line)):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    # Fallback: greedy regex over whole text
    matches = re.findall(r"\{[\s\S]*?\}", text)
    for m in reversed(matches):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def score_lesson(task: dict, response: str) -> dict:
    """Return dict with success/reward and parse details."""
    obj = extract_json_from_response(response)
    if obj is None:
        return {"parse_ok": False, "success": False, "reward": 0.0, "moves": []}
    moves = obj.get("moves")
    if not isinstance(moves, list):
        return {"parse_ok": False, "success": False, "reward": 0.0, "moves": []}
    moves = [str(m).strip().lower() for m in moves if m]
    try:
        verdict = evaluate_by_id(task["id"], moves)
        return {
            "parse_ok": True,
            "success": bool(verdict["success"]),
            "reward": float(verdict["reward"]),
            "moves": moves,
        }
    except Exception as exc:
        return {"parse_ok": False, "success": False, "reward": 0.0, "moves": moves, "error": str(exc)}


def score_coordinate(task: dict, response: str) -> dict:
    obj = extract_json_from_response(response)
    if obj is None or "answer" not in obj:
        return {"parse_ok": False, "success": False, "reward": 0.0}
    gold = str(task["meta"]["answer"]).strip().lower()
    pred = str(obj["answer"]).strip().lower()
    ok = pred == gold
    return {"parse_ok": True, "success": ok, "reward": 1.0 if ok else 0.0, "pred": pred, "gold": gold}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--include-coord", action="store_true", default=True)
    parser.add_argument("--n-samples", type=int, default=1, help="Pass@k (default greedy=1)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[loading] {args.model}")
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
        n=args.n_samples,
    )

    lessons, coords = load_tasks(include_coord=args.include_coord)
    print(f"[tasks] lessons={len(lessons)} coords={len(coords)}")

    all_tasks = [(t, "lesson") for t in lessons] + [(t, "coord") for t in coords]
    prompts = []
    for t, _ in all_tasks:
        messages = [
            {"role": "system", "content": t["system"]},
            {"role": "user", "content": t["user"]},
        ]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(rendered)

    print(f"[generating] {len(prompts)} prompts, temp={args.temperature}, n={args.n_samples}")
    outputs = llm.generate(prompts, sp)

    rows = []
    for (task, kind), out in zip(all_tasks, outputs):
        # Use the best of n samples (any success counts as pass@n)
        any_ok = False
        any_reward = 0.0
        best_score = None
        responses = []
        for sample in out.outputs:
            responses.append(sample.text)
            score = score_lesson(task, sample.text) if kind == "lesson" else score_coordinate(task, sample.text)
            if score["success"]:
                any_ok = True
            any_reward = max(any_reward, score["reward"])
            if best_score is None or score["reward"] > best_score["reward"]:
                best_score = score
        rows.append({
            "id": task["id"],
            "kind": kind,
            "stage_id": task.get("stage_id"),
            "stage_key": task.get("stage_key"),
            "stage_title": task.get("stage_title"),
            "category": task.get("category", "coord"),
            "level": task.get("level"),
            "nbMoves": task.get("meta", {}).get("nbMoves"),
            "is_multi_move": kind == "lesson" and task.get("meta", {}).get("nbMoves", 1) > 1,
            "success_any": any_ok,
            "best_reward": any_reward,
            "best_parse_ok": best_score.get("parse_ok"),
            "best_moves": best_score.get("moves") or best_score.get("pred"),
            "responses": responses,
        })

    # Aggregate
    def acc(filt):
        sub = [r for r in rows if filt(r)]
        if not sub:
            return None
        return sum(1 for r in sub if r["success_any"]) / len(sub)

    summary = {
        "model": args.model,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "total": len(rows),
        "overall_acc": acc(lambda r: True),
        "lesson_only_acc": acc(lambda r: r["kind"] == "lesson"),
        "coord_only_acc": acc(lambda r: r["kind"] == "coord"),
        "single_move_acc": acc(lambda r: r["kind"] == "lesson" and not r["is_multi_move"]),
        "multi_move_acc": acc(lambda r: r["kind"] == "lesson" and r["is_multi_move"]),
        "parse_ok_rate": sum(1 for r in rows if r["best_parse_ok"]) / len(rows),
        "by_stage": {},
        "by_category": {},
        "by_nbMoves": {},
    }
    for stage_key in sorted(set(r["stage_key"] for r in rows if r["stage_key"])):
        sub = [r for r in rows if r["stage_key"] == stage_key]
        summary["by_stage"][stage_key] = {
            "n": len(sub),
            "acc": acc(lambda r: r["stage_key"] == stage_key),
        }
    for cat in sorted(set(r["category"] for r in rows)):
        sub = [r for r in rows if r["category"] == cat]
        summary["by_category"][cat] = {
            "n": len(sub),
            "acc": acc(lambda r: r["category"] == cat),
        }
    for n in sorted(set(r["nbMoves"] for r in rows if r["nbMoves"] is not None), key=lambda x: x or 0):
        sub = [r for r in rows if r["nbMoves"] == n]
        summary["by_nbMoves"][str(n)] = {
            "n": len(sub),
            "acc": acc(lambda r: r["nbMoves"] == n),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    print(f"  total tasks:        {summary['total']}")
    print(f"  overall acc:        {summary['overall_acc']:.3f}")
    print(f"  parse_ok_rate:      {summary['parse_ok_rate']:.3f}")
    print(f"  lesson only acc:    {summary['lesson_only_acc']:.3f}")
    if summary["coord_only_acc"] is not None:
        print(f"  coord only acc:     {summary['coord_only_acc']:.3f}")
    print(f"  single-move acc:    {summary['single_move_acc']:.3f}")
    print(f"  multi-move acc:     {summary['multi_move_acc']:.3f}")
    print(f"\nBy nbMoves (lesson tasks):")
    for n, st in summary["by_nbMoves"].items():
        bar = "█" * int(st["acc"] * 30) if st["acc"] is not None else ""
        print(f"  {n} moves: n={st['n']:>3}  acc={st['acc']:.3f}  {bar}")
    print(f"\nBy stage:")
    for stage, st in summary["by_stage"].items():
        bar = "█" * int(st["acc"] * 30)
        print(f"  {stage:20s} n={st['n']:>3}  acc={st['acc']:.3f}  {bar}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
