"""Single-turn Lichess puzzle env over Chess-R1 / SF μ-graded parquet data.

Wraps the chess-rl-C224 schema parquet (`prompt`, `reward_model{fen,
ground_truth, move_expected_scores_json, legal_moves_uci, ...}`) into a
verl-agent gym.Env so HGPO / GRPO can train on it.

Optional Verbalized Action Masking (VAM):
  - subset_source='mu_topk'  → top-k by precomputed Stockfish μ (paper-style,
                                most principled since μ table is ground truth)
  - subset_source='random'    → uniform random k-subset
  - iterative=True            → previously picked move removed from next reset
                                (per `puzzle id`; resets each new task)

Each episode is single-turn:
  reset() returns FEN + Legal moves [+ Allowed moves]
  step(action_text) parses UCI from <action>, looks up μ, returns reward & done
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

import gym
import numpy as np
import pandas as pd
import ray


_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)
_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Projection: model output → uci string
# --------------------------------------------------------------------------- #

def lichess_puzzle_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Parse UCI move from <action>...</action>. Returns (uci_list, validity_flags)."""
    valids = [0] * len(actions)
    out: List[str] = []
    for i, raw in enumerate(actions):
        text = raw if isinstance(raw, str) else str(raw)
        if "<think>" not in text or "</think>" not in text:
            out.append("")
            continue
        m = _ACTION_RE.search(text)
        if not m:
            out.append("")
            continue
        uci = m.group(1).strip().lower().replace(" ", "")
        if _UCI_RE.match(uci):
            out.append(uci)
            valids[i] = 1
        else:
            out.append("")
    return out, valids


# --------------------------------------------------------------------------- #
# VAM subset selection
# --------------------------------------------------------------------------- #

def _compute_allowed_moves(
    legal: List[str],
    mu_table: dict,
    *,
    k: int,
    source: str,
    rng: random.Random,
    already_chosen: Optional[set] = None,
) -> List[str]:
    pool = [m for m in legal if (not already_chosen or m not in already_chosen)]
    if not pool:
        return []
    k_eff = min(k, len(pool))

    if source == "mu_topk":
        scored = [(m, float(mu_table.get(m, 0.0))) for m in pool]
        scored.sort(key=lambda x: (-x[1], x[0]))  # μ desc, tie-break uci lex
        return [m for m, _ in scored[:k_eff]]

    # default: random
    out = pool[:]
    rng.shuffle(out)
    return out[:k_eff]


# --------------------------------------------------------------------------- #
# Obs builder
# --------------------------------------------------------------------------- #

def _build_puzzle_obs(
    fen: str,
    legal: List[str],
    *,
    allowed: Optional[List[str]] = None,
    side_to_move: Optional[str] = None,
) -> str:
    """Render a chess-rl-style single-turn puzzle obs."""
    side = side_to_move or ("White" if fen.split()[1] == "w" else "Black")
    lines = [
        f"FEN: {fen}",
        f"Side to move: {side}",
        "",
        f"Legal moves (UCI): {', '.join(legal)}",
    ]
    if allowed is not None:
        lines.append(f"Allowed moves (UCI): {', '.join(allowed)}")
        lines.append(
            "(You MUST pick your move from Allowed moves above. "
            "Out-of-subset picks are penalised.)"
        )
    lines.append("")
    lines.append(
        "Choose the single best legal UCI move. "
        "Reason inside <think></think> then output the move inside <action></action>."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

class LichessPuzzleWorker:
    """Single-turn puzzle env over Chess-R1 / SF μ-graded parquet."""

    def __init__(self, env_kwargs: dict):
        parquet_paths = env_kwargs.get("parquets") or []
        if isinstance(parquet_paths, str):
            parquet_paths = [parquet_paths]
        if not parquet_paths:
            raise ValueError("LichessPuzzleWorker requires env_kwargs['parquets'] (list of parquet paths)")

        dfs = [pd.read_parquet(p) for p in parquet_paths]
        self.df = pd.concat(dfs, ignore_index=True)
        self.n = len(self.df)
        if self.n == 0:
            raise ValueError("LichessPuzzleWorker: parquet input has 0 rows")

        # ---- VAM config (default = disabled, behavior == bare puzzle) ----
        vam = env_kwargs.get("vam") or {}
        self.vam_enable = bool(vam.get("enable", False))
        self.vam_k = int(vam.get("k", 8))
        self.vam_source = str(vam.get("subset_source", "mu_topk"))  # 'mu_topk' | 'random'
        self.vam_iterative = bool(vam.get("iterative", False))
        self.vam_penalty = float(vam.get("penalty", -1.0))

        # Per-puzzle state
        self._rng = random.Random(0)
        self._cur_idx: int = 0
        self._cur_row = None
        self._cur_mu: dict = {}
        self._cur_legal: List[str] = []
        self._allowed: List[str] = []
        # Iterative state keyed by puzzle id (so multi-rollout same puzzle exhausts subset)
        self._chosen_by_pid: dict[str, set] = {}

    # ---- helpers ----

    def _puzzle_id(self, row) -> str:
        ex = row.get("extra_info")
        if isinstance(ex, dict) and ex.get("chessr1_id"):
            return f"chessr1:{ex['chessr1_id']}"
        rm = row.get("reward_model") or {}
        fen = (rm.get("fen") if isinstance(rm, dict) else None) or ""
        return f"fen:{fen}"

    def _refresh_allowed(self) -> None:
        if not self.vam_enable:
            self._allowed = []
            return
        pid = self._puzzle_id(self._cur_row)
        chosen = self._chosen_by_pid.get(pid) if self.vam_iterative else None
        self._allowed = _compute_allowed_moves(
            self._cur_legal,
            self._cur_mu,
            k=self.vam_k,
            source=self.vam_source,
            rng=self._rng,
            already_chosen=chosen,
        )

    def _obs(self) -> str:
        rm = self._cur_row["reward_model"]
        rm = dict(rm) if not isinstance(rm, dict) else rm
        fen = rm.get("fen", "")
        return _build_puzzle_obs(
            fen,
            self._cur_legal,
            allowed=self._allowed if self.vam_enable else None,
        )

    # ---- gym API ----

    def reset(self, task_idx):
        idx = int(task_idx) % self.n
        self._rng.seed(idx)
        self._cur_idx = idx
        self._cur_row = self.df.iloc[idx]
        rm = self._cur_row["reward_model"]
        rm = dict(rm) if not isinstance(rm, dict) else rm

        # Legal moves
        legal_raw = rm.get("legal_moves_uci")
        if hasattr(legal_raw, "tolist"):
            legal_raw = legal_raw.tolist()
        self._cur_legal = [str(m).strip().lower() for m in (legal_raw or [])]

        # μ table
        try:
            self._cur_mu = json.loads(rm.get("move_expected_scores_json") or "{}")
        except Exception:
            self._cur_mu = {}

        self._refresh_allowed()

        info = {
            "task_id": self._puzzle_id(self._cur_row),
            "ground_truth": rm.get("ground_truth"),
            "won": False,
        }
        return self._obs(), info

    def step(self, action_text: str):
        rm = self._cur_row["reward_model"]
        rm = dict(rm) if not isinstance(rm, dict) else rm
        gt = str(rm.get("ground_truth") or "").strip().lower()

        # Parse UCI (action_text comes through projection already, but be defensive)
        uci = str(action_text or "").strip().lower().replace(" ", "")
        if not uci or not _UCI_RE.match(uci):
            # Bad / missing format
            return (
                self._obs(),
                -1.0,
                True,
                {
                    "won": False,
                    "action_is_effective": False,
                    "task_id": self._puzzle_id(self._cur_row),
                    "ground_truth": gt,
                    "pred": uci,
                    "reason": "format_error_or_missing_uci",
                },
            )

        if uci not in self._cur_legal:
            return (
                self._obs(),
                -1.0,
                True,
                {
                    "won": False,
                    "action_is_effective": False,
                    "task_id": self._puzzle_id(self._cur_row),
                    "ground_truth": gt,
                    "pred": uci,
                    "reason": "illegal_move",
                },
            )

        # VAM out-of-subset
        if self.vam_enable and self._allowed and uci not in self._allowed:
            return (
                self._obs(),
                self.vam_penalty,
                True,
                {
                    "won": False,
                    "action_is_effective": False,
                    "task_id": self._puzzle_id(self._cur_row),
                    "ground_truth": gt,
                    "pred": uci,
                    "reason": "vam_violation",
                    "vam_violation": True,
                },
            )

        # In-subset (or VAM disabled): score via μ table
        mu = float(self._cur_mu.get(uci, 0.0))

        # iterative VAM bookkeeping
        if self.vam_enable and self.vam_iterative:
            pid = self._puzzle_id(self._cur_row)
            self._chosen_by_pid.setdefault(pid, set()).add(uci)

        info = {
            "won": uci == gt,
            "action_is_effective": True,
            "task_id": self._puzzle_id(self._cur_row),
            "ground_truth": gt,
            "pred": uci,
            "mu": mu,
        }
        return self._obs(), mu, True, info

    def close(self):
        pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Vectorised env (Ray)
# --------------------------------------------------------------------------- #

class LichessPuzzleMultiProcessEnv(gym.Env):
    """Vectorised single-turn puzzle env via Ray actors."""

    def __init__(
        self,
        seed=0,
        env_num=1,
        group_n=1,
        resources_per_worker=None,
        is_train=True,
        env_kwargs=None,
    ):
        super().__init__()
        if resources_per_worker is None:
            resources_per_worker = {"num_cpus": 0.1}

        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        np.random.seed(seed)

        if env_kwargs is None:
            env_kwargs = {}

        # Discover total N from any one parquet so reset can sample legitimate indices.
        parquet_paths = env_kwargs.get("parquets") or []
        if isinstance(parquet_paths, str):
            parquet_paths = [parquet_paths]
        if not parquet_paths:
            raise ValueError("LichessPuzzleMultiProcessEnv: env_kwargs['parquets'] required")
        # Cheap row count: read pyarrow metadata.
        import pyarrow.parquet as pq
        n_total = sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_paths)
        self._n_total = max(n_total, 1)
        self._val_cursor = 0

        env_worker = ray.remote(**resources_per_worker)(LichessPuzzleWorker)
        self.workers = [env_worker.remote(env_kwargs) for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes
        futures = [w.step.remote(a) for w, a in zip(self.workers, actions)]
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        if self.is_train:
            base_idx = np.random.randint(0, self._n_total, size=self.env_num)
        else:
            base_idx = (self._val_cursor + np.arange(self.env_num)) % self._n_total
            self._val_cursor = (self._val_cursor + self.env_num) % self._n_total
        # Same task across each group (so group_n rollouts share initial conditions)
        idx_per_worker = np.repeat(base_idx, self.group_n).tolist()

        futures = [w.reset.remote(idx_per_worker[i]) for i, w in enumerate(self.workers)]
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def close(self):
        for w in self.workers:
            ray.kill(w)

    def __del__(self):
        self.close()


def build_lichess_puzzle_envs(
    seed=0,
    env_num=1,
    group_n=1,
    resources_per_worker=None,
    is_train=True,
    env_kwargs=None,
):
    return LichessPuzzleMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
