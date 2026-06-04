#!/usr/bin/env python3
"""Rewrite VERL chess datasets with Searchless Chess critic scores."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple

import chess
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.searchless_setup import prepare_searchless_src, pushd


def compute_critic_probs(
    engine: Any,
    board: chess.Board,
    move_order_fn: Callable[[chess.Board], Sequence[chess.Move]],
) -> Tuple[Iterable[str], np.ndarray]:
  """Return ordered legal moves and their critic win probabilities."""
  analysis = engine.analyse(board)
  log_probs = np.asarray(analysis['log_probs'])
  bucket_probs = np.exp(log_probs)
  win_probs = bucket_probs @ engine._return_buckets_values  # pylint: disable=protected-access
  ordered_moves = [move.uci() for move in move_order_fn(board)]
  return ordered_moves, win_probs.astype(np.float64)


def update_reward_with_critic(
    reward_model: Dict[str, Any],
    engine: Any,
    move_order_fn: Callable[[chess.Board], Sequence[chess.Move]],
    overwrite_ground_truth: bool,
) -> Dict[str, Any]:
  """Return a new reward_model dict scored by the neural critic."""
  fen = (reward_model.get('fen') or '').strip()
  if not fen:
    raise ValueError('Missing FEN in reward_model; cannot score position.')
  board = chess.Board(fen=fen)
  ordered_moves, critic_probs = compute_critic_probs(
      engine, board, move_order_fn
  )
  if len(ordered_moves) == 0:
    raise ValueError(f'No legal moves found for FEN:\n{fen}')
  move_values = {
      move: float(prob)
      for move, prob in zip(ordered_moves, critic_probs, strict=True)
  }
  best_move = max(move_values, key=move_values.get)
  updated = dict(reward_model)
  updated['move_values_json'] = json.dumps(
      move_values,
      sort_keys=True,
      separators=(',', ':'),
  )
  if overwrite_ground_truth:
    updated['ground_truth'] = best_move
  return updated


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          'Replace Stockfish scores with the Searchless Chess critic while'
          ' preserving the VERL parquet schema.'
      )
  )
  parser.add_argument(
      '--input-parquet',
      type=Path,
      required=True,
      help='Existing VERL-formatted parquet (e.g., data/chess_puzzles/train.parquet).',
  )
  parser.add_argument(
      '--output-parquet',
      type=Path,
      required=True,
      help='Path to write the critic-scored parquet.',
  )
  parser.add_argument(
      '--max-rows',
      type=int,
      default=None,
      help='Optional limit for debugging; processes the first N rows.',
  )
  parser.add_argument(
      '--overwrite-ground-truth',
      action='store_true',
      help=(
          'If set, replace reward_model.ground_truth with the critic argmax. '
          'By default, we preserve the existing ground truth (e.g., Lichess).'
      ),
  )
  args = parser.parse_args()

  input_path = args.input_parquet if args.input_parquet.is_absolute() else REPO_ROOT / args.input_parquet
  output_path = args.output_parquet if args.output_parquet.is_absolute() else REPO_ROOT / args.output_parquet
  output_path.parent.mkdir(parents=True, exist_ok=True)

  df = pd.read_parquet(input_path)
  if args.max_rows is not None:
    df = df.head(args.max_rows).copy()

  searchless_src = prepare_searchless_src()
  searchless_root = searchless_src.parent
  searchless_pkg_parent = searchless_root.parent
  if str(searchless_pkg_parent) not in sys.path:
    sys.path.insert(0, str(searchless_pkg_parent))
  engine_constants = importlib.import_module(
      'searchless_chess.src.engines.constants'
  )
  engine_lib = importlib.import_module('searchless_chess.src.engines.engine')

  print(f'Loaded {len(df)} rows from {input_path}. Building 270M critic...')
  with pushd(searchless_src):
    engine = engine_constants.ENGINE_BUILDERS['270M']()
  print('Critic ready. Re-scoring dataset...')

  updated_reward_models = []
  for reward_model in tqdm(
      df['reward_model'], total=len(df), desc='Relabeling', unit='row'
  ):
    updated_reward_models.append(
        update_reward_with_critic(
            reward_model,
            engine,
            engine_lib.get_ordered_legal_moves,
            overwrite_ground_truth=args.overwrite_ground_truth,
        )
    )

  df = df.copy()
  df['reward_model'] = updated_reward_models
  df.to_parquet(output_path, index=False)
  print(f'Wrote {len(df)} critic-scored rows to {output_path}')


if __name__ == '__main__':
  main()
