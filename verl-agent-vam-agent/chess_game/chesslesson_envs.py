"""verl-agent environment for the lichess "Learn chess" tasks (multi-turn online).

Wraps the canonical lichess judge (reward.py / chesslib.py / reward_specs.json),
its stateful move-by-move driver (stepper.py / ``LessonStepper``), and the
coordinate grader (gen_coordinates.py) -- all self-contained under
``chess_game/chesslesson/``.

Two task families, one Worker:
  - chess      : TRUE multi-turn. The model plays ONE move per turn; the worker
                 applies it plus any scripted opponent reply, re-renders the
                 board, and returns the new position as the next observation --
                 until the lesson is solved / failed / the step budget runs out.
                 Reward is the canonical lichess binary verdict (1/0), delivered
                 ONLY on the terminal step (no per-step shaping), computed by
                 LessonStepper (proven identical to reward.evaluate).
  - coordinate : single turn (one answer), graded by gen_coordinates.grade.

Mirrors the structure of chess_game/ray_envs.py (Worker / MultiProcessEnv /
build_* / projection) so it slots into agent_system make_envs the same way. The
framework's rollout loop drives step() up to ``env.max_steps`` times, feeding
each observation back into the running conversation.
"""
import os
import re
import sys
from typing import List, Tuple

import gym
import numpy as np
import ray

# The bundled judge uses bare imports (``import chesslib``, ``import reward``).
# Put its directory on sys.path so those resolve here and in every Ray worker.
_LESSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chesslesson")
if _LESSON_DIR not in sys.path:
    sys.path.append(_LESSON_DIR)

from chess_env import ChessLearnEnv, parse_moves  # noqa: E402
from chess_game.prompts_shared import (  # noqa: E402
    build_lesson_initial_obs,
    build_lesson_step_obs,
    build_puzzle_prompt,
)


# --------------------------------------------------------------------------- #
# projection: model text -> action payload + validity
# --------------------------------------------------------------------------- #
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.S | re.I)
_TOKEN_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?|[a-h][1-8]|light|dark", re.I)


def chesslesson_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Extract the <action> payload from each completion.

    Requires both <think></think> and <action></action> (verl-agent convention).
    The payload is passed verbatim to the worker, which scores it with the
    bundled judge (its parser tolerates UCI/JSON/coordinate forms). Empty string
    => invalid / missing action.
    """
    out: List[str] = []
    valids = [0] * len(actions)
    for i, raw in enumerate(actions):
        text = raw if isinstance(raw, str) else str(raw)
        if "<think>" not in text or "</think>" not in text:
            out.append("")
            continue
        m = _ACTION_RE.search(text)
        if not m:
            out.append("")
            continue
        inner = m.group(1).strip()
        if inner and _TOKEN_RE.search(inner):
            out.append(inner)
            valids[i] = 1
        else:
            out.append("")
    return out, valids


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
class ChessLessonWorker:
    """One lichess "Learn chess" task at a time.

    chess tasks are driven move-by-move by a ``LessonStepper`` (multi-turn);
    coordinate tasks are graded in a single step. ``board_repr`` (env_kwargs)
    optionally appends an ASCII/unicode board to each observation; default
    None keeps the FEN-only style of the original single-turn prompts.
    """

    def __init__(self, env_kwargs):
        env_kwargs = env_kwargs or {}
        self.env = ChessLearnEnv(
            include_chess=env_kwargs.get("include_chess", True),
            include_coord=env_kwargs.get("include_coord", True),
        )
        self.board_repr = env_kwargs.get("board_repr") or None
        # Only chess tasks that actually have a reward spec are scoreable.
        from reward import SPECS  # noqa: E402
        self._specs = SPECS
        self.tasks = [
            t for t in self.env.tasks
            if t["kind"] != "chess" or t["id"] in SPECS
        ]
        self.n = len(self.tasks)
        self._cur = self.tasks[0]
        self._stepper = None  # set for chess tasks on reset

    def task_count(self):
        return self.n

    def reset(self, task_idx):
        from stepper import LessonStepper  # noqa: E402

        self._cur = self.tasks[int(task_idx) % self.n]
        info = {"won": False, "task_id": self._cur["id"], "kind": self._cur["kind"]}

        if self._cur["kind"] == "chess":
            st = LessonStepper(self._specs[self._cur["id"]])
            self._stepper = st
            obs = build_lesson_initial_obs(
                self._cur, st.render_fen(), st.opening_opponent,
                board_repr=self.board_repr,
            )
        else:
            self._stepper = None
            obs = build_puzzle_prompt(self._cur)
        return obs, info

    def step(self, action_text):
        if not isinstance(action_text, str):
            action_text = ""

        # coordinate: single-turn grade -> always done
        if self._stepper is None:
            out = self.env.score(self._cur, action_text)
            info = {
                "won": bool(out["success"]),
                "action_is_effective": bool(action_text.strip()),
                "task_id": self._cur["id"],
                "kind": self._cur["kind"],
                "parsed_action": out["action"],
            }
            return build_puzzle_prompt(self._cur), float(out["reward"]), True, info

        # chess: extract ONE clean UCI move from this turn, apply it; reward only
        # on the terminal step
        moves = parse_moves(action_text)
        mv = moves[0] if moves else ""
        obs = self._stepper.step(mv)
        done = bool(obs["done"])
        reward = 1.0 if obs["success"] else 0.0
        info = {
            "won": bool(obs["success"]),
            "action_is_effective": bool(obs["legal"]),
            "task_id": self._cur["id"],
            "kind": "chess",
            "parsed_action": obs["player_move"],
            "reason": obs["reason"],
        }
        # same feedback shape whether terminal or not (on terminal the framework
        # just won't ask for another move)
        text = build_lesson_step_obs(
            obs["render_fen"], opponent_moves=obs["opponent_moves"],
            apples_left=obs["apples_left"], collected=obs["collected"],
            board_repr=self.board_repr,
        )
        return text, reward, done, info


# --------------------------------------------------------------------------- #
# vectorised env (Ray)
# --------------------------------------------------------------------------- #
class ChessLessonMultiProcessEnv(gym.Env):
    def __init__(self, seed=0, env_num=1, group_n=1, resources_per_worker=None,
                 is_train=True, env_kwargs=None):
        super().__init__()
        if resources_per_worker is None:
            resources_per_worker = {"num_cpus": 0.1}
        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self._rng = np.random.RandomState(seed)
        self._val_cursor = 0

        env_kwargs = env_kwargs or {}
        worker = ray.remote(**resources_per_worker)(ChessLessonWorker)
        self.workers = [worker.remote(env_kwargs) for _ in range(self.num_processes)]
        self.num_tasks = int(ray.get(self.workers[0].task_count.remote()))

    def reset(self):
        if self.is_train:
            idx = self._rng.randint(0, self.num_tasks, size=self.env_num)
        else:
            idx = (self._val_cursor + np.arange(self.env_num)) % self.num_tasks
            self._val_cursor = (self._val_cursor + self.env_num) % self.num_tasks
        idx = np.repeat(idx, self.group_n).tolist()  # same task across a group

        results = ray.get([w.reset.remote(idx[i]) for i, w in enumerate(self.workers)])
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def step(self, actions):
        assert len(actions) == self.num_processes
        results = ray.get([w.step.remote(a) for w, a in zip(self.workers, actions)])
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def close(self):
        for w in self.workers:
            ray.kill(w)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def build_chesslesson_envs(seed=0, env_num=1, group_n=1, resources_per_worker=None,
                           is_train=True, env_kwargs=None):
    return ChessLessonMultiProcessEnv(
        seed=seed, env_num=env_num, group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train, env_kwargs=env_kwargs,
    )
