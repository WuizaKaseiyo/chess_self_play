"""Chess (White vs random Black) env for verl-agent, with optional VAM.

VAM = Verbalized Action Masking (chess-rl-C224 paper, EMNLP).

When enabled:
  - Each ply the env computes a subset of `vam.k` legal moves and exposes them
    as an `Allowed moves (UCI): ...` line in the observation.
  - If the model picks an in-subset move, it's pushed normally (-0.01).
  - If the model picks a legal-but-out-of-subset move, the env returns
    `vam.penalty` (default -1.0) and DOES NOT advance the board. The episode
    continues so the model can try again.
  - If `vam.iterative=True`, the model's previous in-subset choices are
    removed from future subsets (chess-rl-C224 allowed-move-elimination).

Subset sources:
  - "random"    — uniform random k-subset of legal moves (no SF dep).
  - "stockfish" — top-k by Stockfish MultiPV at `vam.stockfish_depth`.

The default (vam.enable=False) preserves the original behavior 1:1 — every
existing test / launcher works unchanged.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import chess
import chess.engine
import gym
import numpy as np
import ray


# --------------------------------------------------------------------------- #
# VAM helpers
# --------------------------------------------------------------------------- #

def _compute_allowed_moves(
    board: chess.Board,
    *,
    k: int,
    source: str,
    rng: random.Random,
    engine: Optional[chess.engine.SimpleEngine],
    stockfish_depth: int,
    already_chosen: Optional[set] = None,
) -> list[str]:
    """Return a k-subset of legal moves (UCI) for VAM exposure."""
    legal = [m.uci() for m in board.legal_moves]
    if not legal:
        return []
    if already_chosen:
        legal = [m for m in legal if m not in already_chosen]
        if not legal:
            return []
    k_eff = min(k, len(legal))

    if source == "stockfish" and engine is not None:
        try:
            info = engine.analyse(
                board,
                limit=chess.engine.Limit(depth=stockfish_depth),
                multipv=k_eff,
            )
            ranked: list[str] = []
            for entry in info:
                pv = entry.get("pv")
                if pv:
                    u = pv[0].uci()
                    if u in legal and u not in ranked:
                        ranked.append(u)
            if len(ranked) < k_eff:
                extras = [m for m in legal if m not in ranked]
                rng.shuffle(extras)
                ranked.extend(extras[: k_eff - len(ranked)])
            return ranked[:k_eff]
        except Exception:
            pass  # fall back to random

    # Random fallback (also the default source).
    sub = legal[:]
    rng.shuffle(sub)
    return sub[:k_eff]


def _board_text(
    board: chess.Board,
    *,
    allowed_moves: Optional[list[str]] = None,
) -> str:
    lines = [board.unicode(), "", f"FEN: {board.fen()}", ""]
    if board.turn == chess.WHITE:
        legal = [m.uci() for m in board.legal_moves]
        preview = ", ".join(legal[:48]) + (", ..." if len(legal) > 48 else "")
        lines.append("You play White. One legal UCI move in <action></action>, e.g. e2e4.")
        lines.append(f"Legal moves (UCI): {preview}")
        if allowed_moves is not None:
            lines.append(f"Allowed moves (UCI): {', '.join(allowed_moves)}")
            lines.append(
                "(You MUST pick your move from Allowed moves above. "
                "Picking a legal-but-out-of-subset move is penalised and does not advance the board.)"
            )
    else:
        lines.append("(Black to move — environment plays Black.)")
    return "\n".join(lines)


def _play_random_black(board: chess.Board, rng: random.Random) -> None:
    if board.turn != chess.BLACK or board.is_game_over():
        return
    moves = list(board.legal_moves)
    if moves:
        board.push(rng.choice(moves))


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

class ChessWorker:
    """White (policy) vs random Black; rules via python-chess. Optional VAM."""

    def __init__(self, env_kwargs):
        self.max_agent_plies = int(env_kwargs.get("max_agent_plies", 100))
        self._rng = random.Random(0)
        self.board = chess.Board()
        self._agent_plies = 0

        # ---- VAM config (default = disabled, behavior identical to upstream) ----
        vam_cfg = env_kwargs.get("vam", {}) or {}
        self.vam_enable = bool(vam_cfg.get("enable", False))
        self.vam_k = int(vam_cfg.get("k", 8))
        self.vam_iterative = bool(vam_cfg.get("iterative", False))
        self.vam_source = str(vam_cfg.get("subset_source", "random"))  # "random" | "stockfish"
        self.vam_penalty = float(vam_cfg.get("penalty", -1.0))
        self.vam_stockfish_path = str(vam_cfg.get("stockfish_path", "") or "").strip()
        self.vam_stockfish_depth = int(vam_cfg.get("stockfish_depth", 1))

        self._engine: Optional[chess.engine.SimpleEngine] = None
        if self.vam_enable and self.vam_source == "stockfish":
            if not self.vam_stockfish_path or not os.path.exists(self.vam_stockfish_path):
                raise RuntimeError(
                    f"VAM source=stockfish but binary missing at {self.vam_stockfish_path!r}. "
                    f"Set env.chess.vam.stockfish_path or switch to subset_source=random."
                )
            self._engine = chess.engine.SimpleEngine.popen_uci(self.vam_stockfish_path)

        # Per-game state (recomputed in reset)
        self._allowed_moves: list[str] = []
        self._chosen_moves: set[str] = set()

    # ---- VAM state helpers ----

    def _refresh_allowed(self) -> None:
        """Recompute allowed_moves for the current board position (if VAM enabled and White-to-move)."""
        if not self.vam_enable or self.board.turn != chess.WHITE:
            self._allowed_moves = []
            return
        already = self._chosen_moves if self.vam_iterative else None
        self._allowed_moves = _compute_allowed_moves(
            self.board,
            k=self.vam_k,
            source=self.vam_source,
            rng=self._rng,
            engine=self._engine,
            stockfish_depth=self.vam_stockfish_depth,
            already_chosen=already,
        )

    def _obs(self) -> str:
        return _board_text(
            self.board,
            allowed_moves=self._allowed_moves if self.vam_enable else None,
        )

    # ---- gym API ----

    def reset(self, seed_for_reset):
        self._rng.seed(seed_for_reset if seed_for_reset is not None else 0)
        self.board = chess.Board()
        self._agent_plies = 0
        self._chosen_moves = set()
        self._refresh_allowed()
        return self._obs(), {"won": False}

    def step(self, uci: str):
        # Always let env push pending Black moves before parsing the model's action.
        while self.board.turn == chess.BLACK and not self.board.is_game_over():
            _play_random_black(self.board, self._rng)

        # ---- input sanitization ----
        if not isinstance(uci, str):
            return self._obs(), -0.1, False, {"won": False, "action_is_effective": False}
        uci = uci.strip().lower().replace(" ", "")
        if not uci:
            return self._obs(), -0.1, False, {"won": False, "action_is_effective": False}

        # ---- UCI parse ----
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return self._obs(), -0.1, False, {"won": False, "action_is_effective": False}

        # ---- legality ----
        if move not in self.board.legal_moves:
            return self._obs(), -0.1, False, {"won": False, "action_is_effective": False}

        # ---- VAM subset check ----
        if self.vam_enable and self._allowed_moves and uci not in self._allowed_moves:
            return (
                self._obs(),
                self.vam_penalty,
                False,
                {"won": False, "action_is_effective": False, "vam_violation": True},
            )

        # ---- legal + (VAM ok) move: push board ----
        self._agent_plies += 1
        self.board.push(move)
        if self.vam_iterative:
            self._chosen_moves.add(uci)
        reward = -0.01
        info = {"won": False, "action_is_effective": True}

        outcome = self.board.outcome()
        if outcome is not None:
            return self._terminal(outcome, info)

        _play_random_black(self.board, self._rng)
        outcome = self.board.outcome()
        if outcome is not None:
            return self._terminal(outcome, info)

        if self._agent_plies >= self.max_agent_plies:
            return _board_text(self.board, allowed_moves=None), reward, True, {**info, "won": False}

        # Next-ply: refresh allowed_moves for the new position
        self._refresh_allowed()
        return self._obs(), reward, False, info

    def _terminal(self, outcome: chess.Outcome, info: dict):
        # Game over: no need to expose allowed_moves in terminal obs.
        obs = _board_text(self.board, allowed_moves=None)
        if outcome.winner == chess.WHITE:
            return obs, 1.0, True, {"won": True, "action_is_effective": info.get("action_is_effective", True)}
        if outcome.winner == chess.BLACK:
            return obs, -1.0, True, {"won": False, "action_is_effective": info.get("action_is_effective", True)}
        return obs, 0.0, True, {"won": False, "action_is_effective": info.get("action_is_effective", True)}

    def close(self):
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Vectorised env (Ray)
# --------------------------------------------------------------------------- #

class ChessMultiProcessEnv(gym.Env):
    """Vectorised chess via Ray; rules entirely delegated to python-chess."""

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

        env_worker = ray.remote(**resources_per_worker)(ChessWorker)
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
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)
        seeds = np.repeat(seeds, self.group_n).tolist()

        futures = [w.reset.remote(seeds[i]) for i, w in enumerate(self.workers)]
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def close(self):
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        self.close()


def build_chess_envs(
    seed=0,
    env_num=1,
    group_n=1,
    resources_per_worker=None,
    is_train=True,
    env_kwargs=None,
):
    return ChessMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
