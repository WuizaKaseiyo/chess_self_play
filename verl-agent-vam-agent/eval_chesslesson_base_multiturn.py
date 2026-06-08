"""Multi-turn zero-shot eval for chesslesson — paradigm-aligned with verl-agent training.

Each lesson is played one move per turn via `LessonStepper` (the same driver
used inside `ChessLessonWorker`). The chat history accumulates exactly as it
does during HGPO/PPO rollout, so this baseline is directly comparable to
post-training numbers.

For coordinate drills (which are inherently single-turn in the upstream env),
we issue one prompt + one response and judge string equality.

Batched generation: at each "turn step", we synchronously generate one
response per active episode using vLLM, then advance each episode by one
step. Finished episodes drop out of the pool. Continues until every episode
terminates or hits the per-task `nbMoves` budget (lesson) / 1 turn (coord).

Usage:
  python eval_chesslesson_base_multiturn.py \
      --model $HOME/models/Qwen2.5-7B-Instruct \
      --output eval_results/base_qwen7b_chesslesson_multiturn.json \
      --max-turns-cap 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "chess_game" / "chesslesson"))

from chess_game.prompts_shared import (
    build_puzzle_prompt,
    build_lesson_initial_obs,
    build_lesson_step_obs,
)
from stepper import LessonStepper
from reward import SPECS


_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)


def load_tasks(coord_path: str | None = None) -> tuple[list[dict], list[dict]]:
    lessons = [json.loads(l) for l in (REPO / "chess_game/chesslesson/instructions.jsonl").open()]
    for r in lessons:
        r["kind"] = "lesson"
    cp = Path(coord_path) if coord_path else (REPO / "chess_game/chesslesson/coordinates.jsonl")
    coords = [json.loads(l) for l in cp.open()]
    for r in coords:
        r["kind"] = "coord"
    return lessons, coords


def parse_action(text: str) -> str:
    """Extract the <action> payload (verl-agent convention)."""
    if not isinstance(text, str):
        return ""
    if "<think>" not in text or "</think>" not in text:
        return ""
    m = _ACTION_RE.search(text)
    if not m:
        return ""
    return m.group(1).strip().lower().replace(" ", "")


@dataclass
class Episode:
    """Live state for one task's rollout (mirrors ChessLessonWorker per-task state)."""
    task: dict
    kind: str                              # "lesson" or "coord"
    stepper: Optional[LessonStepper] = None
    messages: list = field(default_factory=list)  # accumulating chat history
    turns_taken: int = 0
    budget: int = 1
    apples_seen_init: list = field(default_factory=list)

    done: bool = False
    success: bool = False
    final_reward: float = 0.0
    parse_ok_last: bool = True
    last_action: str = ""

    def init_chat(self) -> bool:
        """Build the first user message exactly as verl-agent's
        ChessLessonWorker would after env.reset().
        Returns False if the lesson can't be initialised (missing spec)
        and should be skipped from the eval pool."""
        if self.kind == "lesson":
            spec = SPECS.get(self.task["id"])
            if spec is None:
                # Some lesson ids (e.g. fork-*) live in instructions.jsonl but
                # not in reward_specs.json — skip them rather than crash.
                self.done = True
                self.success = False
                self.final_reward = 0.0
                return False
            self.stepper = LessonStepper(spec)
            self.budget = int(self.task.get("meta", {}).get("nbMoves") or 1)
            self.apples_seen_init = list(self.task.get("meta", {}).get("apples") or [])
            initial = build_lesson_initial_obs(
                self.task,
                render_fen=self.stepper.board_fen(),
                opening_opponent=list(self.stepper.opening_opponent) or None,
            )
            self.messages = [{"role": "user", "content": initial}]
        else:  # coord
            self.budget = 1
            initial = build_puzzle_prompt(self.task)
            self.messages = [{"role": "user", "content": initial}]
        return True


def step_episode(ep: Episode, response_text: str, max_turns_cap: int) -> None:
    """Apply one model response to the episode."""
    ep.messages.append({"role": "assistant", "content": response_text})
    action = parse_action(response_text)
    ep.last_action = action
    ep.parse_ok_last = bool(action)

    if ep.kind == "coord":
        # Single-turn; judge string equality
        gold = str(ep.task["meta"]["answer"]).strip().lower()
        ep.done = True
        ep.success = (action == gold)
        ep.final_reward = 1.0 if ep.success else 0.0
        return

    # Lesson path
    ep.turns_taken += 1
    if not action:
        ep.done = True
        ep.success = False
        ep.final_reward = 0.0
        return

    pre_items = set(ep.stepper.items)
    pre_history_len = len(ep.stepper.chess.history)
    try:
        ep.stepper.step(action)
    except Exception:
        ep.done = True
        ep.success = False
        ep.final_reward = 0.0
        return

    if ep.stepper.done or ep.turns_taken >= ep.budget or ep.turns_taken >= max_turns_cap:
        ep.done = True
        ep.success = bool(ep.stepper.vm.get("completed"))
        ep.final_reward = 1.0 if ep.success else 0.0
        return

    # Build per-turn feedback (next user message) — mirror what env.step would emit
    post_items = set(ep.stepper.items)
    collected_this_turn = pre_items - post_items
    collected = next(iter(collected_this_turn)) if collected_this_turn else None
    new_history = ep.stepper.chess.history[pre_history_len:]
    # Heuristic: the player's own move is the first new entry; anything after is scripted opponent
    opp_moves = new_history[1:] if len(new_history) > 1 else None
    feedback = build_lesson_step_obs(
        render_fen=ep.stepper.board_fen(),
        opponent_moves=opp_moves,
        apples_left=sorted(post_items) if post_items else None,
        collected=collected,
    )
    ep.messages.append({"role": "user", "content": feedback})


def render_prompts(tok, episodes: list[Episode]) -> list[str]:
    out: list[str] = []
    for ep in episodes:
        s = tok.apply_chat_template(ep.messages, tokenize=False, add_generation_prompt=True)
        out.append(s)
    return out


def aggregate(episodes: list[Episode]) -> dict:
    n = len(episodes)
    if n == 0:
        return {}

    def acc(filt):
        sub = [e for e in episodes if filt(e)]
        if not sub:
            return None
        return sum(1 for e in sub if e.success) / len(sub)

    summary = {
        "total": n,
        "overall_acc": acc(lambda e: True),
        "parse_ok_rate_last": sum(1 for e in episodes if e.parse_ok_last) / n,
        "lesson_only_acc": acc(lambda e: e.kind == "lesson"),
        "coord_only_acc": acc(lambda e: e.kind == "coord"),
        "single_move_lesson_acc": acc(lambda e: e.kind == "lesson" and e.budget == 1),
        "multi_move_lesson_acc": acc(lambda e: e.kind == "lesson" and e.budget > 1),
        "by_nbMoves": {},
        "by_stage": {},
        "by_category": {},
    }
    # Per-bucket breakdowns
    for budget in sorted({e.budget for e in episodes if e.kind == "lesson"}):
        sub = [e for e in episodes if e.kind == "lesson" and e.budget == budget]
        summary["by_nbMoves"][str(budget)] = {
            "n": len(sub),
            "acc": sum(1 for e in sub if e.success) / max(len(sub), 1),
        }
    for stage_key in sorted({e.task.get("stage_key") for e in episodes if e.task.get("stage_key")}):
        sub = [e for e in episodes if e.task.get("stage_key") == stage_key]
        summary["by_stage"][stage_key] = {
            "n": len(sub),
            "acc": sum(1 for e in sub if e.success) / max(len(sub), 1),
        }
    for cat in sorted({e.task.get("category", "coord") for e in episodes}):
        sub = [e for e in episodes if e.task.get("category", "coord") == cat]
        summary["by_category"][cat] = {
            "n": len(sub),
            "acc": sum(1 for e in sub if e.success) / max(len(sub), 1),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-turns-cap", type=int, default=12,
                        help="Safety cap on lesson turns regardless of nbMoves budget.")
    parser.add_argument("--max-budget", type=int, default=None,
                        help="If set, only evaluate lessons with nbMoves <= this. "
                             "Coord drills are always included. Matches the curriculum "
                             "max_budget knob used during training.")
    parser.add_argument("--coord-path", type=str, default=None,
                        help="Override coord jsonl path (default: training set). "
                             "Use this to point at coordinates_holdout.jsonl for "
                             "generalization eval on unseen coord squares.")
    parser.add_argument("--skip-lessons", action="store_true",
                        help="Skip all lesson tasks (coord-only eval). Useful when "
                             "you only want to measure generalization on unseen "
                             "coord squares.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

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

    lessons, coords = load_tasks(coord_path=args.coord_path)
    if args.skip_lessons:
        print(f"[skip-lessons] dropping {len(lessons)} lesson tasks (coord-only eval)")
        lessons = []
    if args.max_budget is not None:
        before = len(lessons)
        lessons = [l for l in lessons if int((l.get("meta") or {}).get("nbMoves") or 1) <= args.max_budget]
        print(f"[filter] max_budget={args.max_budget}: lessons {before} → {len(lessons)} "
              f"(coord drills always kept)")
    if args.coord_path:
        print(f"[coord-source] {args.coord_path}")
    print(f"[tasks] lessons={len(lessons)} coords={len(coords)}")

    # Build episode pool
    episodes: list[Episode] = []
    skipped: list[str] = []
    for t in lessons:
        ep = Episode(task=t, kind="lesson")
        if not ep.init_chat():
            skipped.append(t["id"])
            continue
        episodes.append(ep)
    for t in coords:
        ep = Episode(task=t, kind="coord")
        ep.init_chat()
        episodes.append(ep)
    if skipped:
        print(f"[skip] {len(skipped)} lesson(s) missing reward spec: {skipped[:10]}{'...' if len(skipped)>10 else ''}")
    print(f"[pool] total episodes = {len(episodes)}")

    # Multi-turn synchronous batched rollout
    turn_step = 0
    while True:
        active = [e for e in episodes if not e.done]
        if not active:
            break
        turn_step += 1
        print(f"[turn {turn_step}] active episodes = {len(active)} / {len(episodes)}")
        prompts = render_prompts(tok, active)
        outs = llm.generate(prompts, sp)
        for ep, out in zip(active, outs):
            response = out.outputs[0].text
            step_episode(ep, response, max_turns_cap=args.max_turns_cap)
        n_done_now = sum(1 for e in episodes if e.done)
        print(f"           done so far: {n_done_now}")
        if turn_step > args.max_turns_cap + 4:
            print(f"[safety] hit turn cap, breaking")
            break

    # Aggregate
    summary = aggregate(episodes)

    # Per-episode rows
    rows = []
    for ep in episodes:
        rows.append({
            "id": ep.task.get("id"),
            "kind": ep.kind,
            "stage_key": ep.task.get("stage_key"),
            "stage_id": ep.task.get("stage_id"),
            "category": ep.task.get("category", "coord"),
            "level": ep.task.get("level"),
            "budget_nbMoves": ep.budget,
            "turns_taken": ep.turns_taken,
            "success": ep.success,
            "reward": ep.final_reward,
            "last_parsed_action": ep.last_action,
            "stepper_completed": bool(ep.stepper.vm.get("completed")) if ep.stepper else None,
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "summary": summary,
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print()
    print("=" * 60)
    print("MULTI-TURN BASELINE (paradigm-aligned with verl-agent training)")
    print("=" * 60)
    print(f"  total tasks:             {summary['total']}")
    print(f"  overall acc:             {summary['overall_acc']:.3f}")
    print(f"  parse_ok_rate (last):    {summary['parse_ok_rate_last']:.3f}")
    if summary.get("lesson_only_acc") is not None:
        print(f"  lesson-only acc:         {summary['lesson_only_acc']:.3f}")
    if summary.get("coord_only_acc") is not None:
        print(f"  coord-only acc:          {summary['coord_only_acc']:.3f}")
    if summary.get("single_move_lesson_acc") is not None:
        print(f"  single-move lesson acc:  {summary['single_move_lesson_acc']:.3f}")
    if summary.get("multi_move_lesson_acc") is not None:
        print(f"  multi-move lesson acc:   {summary['multi_move_lesson_acc']:.3f}")
    print()
    print("By nbMoves (lesson tasks):")
    for budget, st in summary["by_nbMoves"].items():
        bar = "█" * int((st["acc"] or 0) * 30)
        print(f"  {budget:>3} moves: n={st['n']:>3}  acc={st['acc']:.3f}  {bar}")
    print()
    print("By stage:")
    for stage, st in summary["by_stage"].items():
        bar = "█" * int((st["acc"] or 0) * 30)
        print(f"  {stage:18s} n={st['n']:>3}  acc={st['acc']:.3f}  {bar}")
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
