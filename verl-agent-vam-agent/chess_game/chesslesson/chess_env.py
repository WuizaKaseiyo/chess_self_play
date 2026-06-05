"""Single-turn RL environment over the lichess "Learn chess" tasks.

Framework-agnostic, pure Python, no deps. Wraps the canonical judge (reward.py)
and the coordinate grader (gen_coordinates.py) behind one interface that takes a
*raw model completion string* and returns a scalar reward — which is what an
online LLM-RL loop (GRPO/PPO/rejection-sampling) actually needs.

    from chess_env import ChessLearnEnv
    env = ChessLearnEnv()
    task = env.sample()                      # {id, system, user, kind, ...}
    out  = env.score(task, model_text)       # {reward, success, action, details}

The judge is stateless and fast (~0.25 ms/call), so score() is safe to call from
many worker processes in parallel.

A tiny gym-style adapter (reset/step) is provided for libraries that expect it,
but every episode is exactly one step (these are single-turn verifiable tasks).
"""
import json
import os
import random
import re

from reward import evaluate_by_id, SPECS
import gen_coordinates as _coord

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------- parsing model output -----------------------------
_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.I)


def parse_moves(text):
    """Extract a UCI move list from a raw completion.

    Preferred: the last JSON object containing a "moves" array (per the system
    prompt). Fallback: scan the last line, then the whole text, for UCI tokens.
    """
    if isinstance(text, (list, tuple)):
        return [str(m).strip().lower() for m in text]
    text = str(text or "")
    for m in reversed(list(re.finditer(r"\{[^{}]*\"moves\"[^{}]*\}", text, re.S))):
        try:
            obj = json.loads(m.group(0))
            mv = obj.get("moves")
            if isinstance(mv, list):
                return [str(x).strip().lower() for x in mv]
        except Exception:
            pass
    last = text.strip().splitlines()[-1] if text.strip() else ""
    found = _UCI_RE.findall(last) or _UCI_RE.findall(text)
    return [x.lower() for x in found]


def parse_answer(text):
    """Extract a coordinate answer ('e4' / '3,5' / 'light') from a completion."""
    if text is None:
        return ""
    text = str(text)
    for m in reversed(list(re.finditer(r"\{[^{}]*\"answer\"[^{}]*\}", text, re.S))):
        try:
            obj = json.loads(m.group(0))
            if "answer" in obj:
                return str(obj["answer"]).strip()
        except Exception:
            pass
    last = text.strip().splitlines()[-1] if text.strip() else ""
    m = re.search(r"\b([a-h][1-8])\b", last, re.I) or re.search(r"\b(\d\s*,\s*\d)\b", last) \
        or re.search(r"\b(light|dark)\b", last, re.I)
    return (m.group(1) if m else last).strip()


# ----------------------------------- the env ------------------------------------
class ChessLearnEnv:
    def __init__(self, include_chess=True, include_coord=True,
                 instructions=None, coordinates=None, seed=0):
        self.tasks = []
        if include_chess:
            path = instructions or os.path.join(HERE, "instructions.jsonl")
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line)
                    t["kind"] = "chess"
                    self.tasks.append(t)
        if include_coord:
            path = coordinates or os.path.join(HERE, "coordinates.jsonl")
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        t = json.loads(line)
                        t.setdefault("kind", "coordinate")
                        self.tasks.append(t)
        self._by_id = {t["id"]: t for t in self.tasks}
        self._rng = random.Random(seed)

    # --- dataset access ---
    def __len__(self):
        return len(self.tasks)

    def get(self, i):
        return self.tasks[i]

    def by_id(self, task_id):
        return self._by_id[task_id]

    def sample(self):
        return self._rng.choice(self.tasks)

    def prompt(self, task):
        """Convenience: the (system, user) pair to feed the model."""
        return task["system"], task["user"]

    # --- scoring (the bit online training calls) ---
    def score(self, task, completion):
        """Score a raw model completion string for `task`.

        Returns {reward, success, action, details}. `reward` is the canonical
        lichess binary 0/1 verdict.
        """
        if task["kind"] == "coordinate":
            ans = parse_answer(completion)
            g = _coord.grade(task, ans)
            return {"reward": float(g["reward"]), "success": g["success"],
                    "action": ans,
                    "details": {"expected": g["expected"]}}
        moves = parse_moves(completion)
        r = evaluate_by_id(task["id"], moves)
        return {"reward": float(r["reward"]), "success": r["success"],
                "action": moves,
                "details": r["details"]}

    def score_batch(self, tasks, completions):
        return [self.score(t, c) for t, c in zip(tasks, completions)]

    # --- optional gym-style single-step adapter ---
    def reset(self, task=None, task_id=None):
        if task_id is not None:
            self._cur = self.by_id(task_id)
        elif task is not None:
            self._cur = task
        else:
            self._cur = self.sample()
        return {"system": self._cur["system"], "user": self._cur["user"],
                "id": self._cur["id"], "kind": self._cur["kind"]}

    def step(self, completion):
        out = self.score(self._cur, completion)
        obs = None
        reward = out["reward"]
        done = True                      # single-turn task
        info = out
        return obs, reward, done, info


if __name__ == "__main__":
    env = ChessLearnEnv()
    print("loaded %d tasks (%d chess specs available)" % (len(env), len(SPECS)))

    # parser sanity: messy chatty completion -> correct moves -> reward 1
    demo = 'Let me think step by step...\nFinal answer: {"moves":["a1e1"],"explanation":"rook gives check"}'
    t = env.by_id("check1-1")
    print("parse_moves:", parse_moves(demo), "->", env.score(t, demo)["reward"])

    # coordinate parse: chatty completion
    ct = next(x for x in env.tasks if x["kind"] == "coordinate")
    good = 'The marked square is %s. {"answer":"%s"}' % (ct["meta"]["answer"], ct["meta"]["answer"])
    print("coord", ct["id"], "->", env.score(ct, good)["reward"])

    # gym-style
    obs = env.reset(task_id="checkmate1-1")
    _, r, done, info = env.step('{"moves":["h1h2"]}')   # illegal -> 0
    print("gym step reward=%s done=%s reason=%s" % (r, done, info["details"]["reason"]))

    # quick throughput note
    import time
    t = env.by_id("check2-1")
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        env.score(t, '{"moves":["c5a5","a5a8"]}')
    print("throughput: %.0f scores/s (single process)" % (n / (time.perf_counter() - t0)))
