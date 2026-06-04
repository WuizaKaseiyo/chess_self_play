from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import chess
import chess.pgn

# Ensure local namespace packages (e.g., `recipe/`) resolve under pytest.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipe.chess.full_game_eval import FullGameEvalConfig, _GameState, _step_model_moves


class _DummyBackend:
    def __init__(self, output: str):
        self._output = output

    def generate(
        self,
        prompts: List[List[Dict[str, str]]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seeds: Optional[List[int]] = None,
    ) -> List[str]:
        # Emit one response per prompt, intentionally invalid for move parsing.
        return [self._output for _ in prompts]


def test_full_game_eval_forfeit_rows_never_have_empty_prompt_text() -> None:
    # Force a single attempt, then forfeit. This matches the failure mode observed in
    # `competition/full_game.json`: forfeit rows were previously logged with `prompt_text=""`.
    cfg = FullGameEvalConfig(
        opponent_depths=[1],
        games_per_depth=1,
        seed=0,
        max_retries_per_turn=1,
    )

    pgn = chess.pgn.Game()
    game = _GameState(
        game_id="d1_g000",
        opponent_depth=1,
        model_color=chess.WHITE,
        board=chess.Board(),
        pgn=pgn,
        pgn_node=pgn,
    )

    moves_fp = io.StringIO()
    _step_model_moves(
        cfg=cfg,
        backend=_DummyBackend("no <uci_move> tag"),
        eval_engine=None,
        games=[game],
        moves_fp=moves_fp,
        prompt_template=None,
        prompt_template_vars=None,
    )

    records = [json.loads(line) for line in moves_fp.getvalue().splitlines() if line.strip()]
    assert len(records) == 2

    first, forfeit = records
    assert first["retry_idx"] == 0
    assert first["error_reason"] == "format_missing"
    assert first["prompt_text"]

    assert forfeit["retry_idx"] == cfg.max_retries_per_turn
    assert forfeit["forfeit"] is True
    assert forfeit["error_reason"].startswith("forfeit:")
    assert forfeit["prompt_text"]
    assert forfeit["prompt_text"] == first["prompt_text"]

