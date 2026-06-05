#!/usr/bin/env python3
"""Runs the 270M action-value critic and compares it to Stockfish scores."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import chess
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.searchless_setup import prepare_searchless_src, pushd


def load_samples(
    dataset_path: Path, num_samples: int, seed: int | None
) -> pd.DataFrame:
  if not dataset_path.is_absolute():
    dataset_path = REPO_ROOT / dataset_path
  df = pd.read_parquet(dataset_path)
  if num_samples <= 0 or num_samples > len(df):
    raise ValueError(
        f'num_samples must be in [1, {len(df)}], got {num_samples}.'
    )
  return df.sample(n=num_samples, random_state=seed).reset_index(drop=True)


def compute_critic_probs(
    engine: Any,
    board: chess.Board,
    move_order_fn: Callable[[chess.Board], Sequence[chess.Move]],
) -> tuple[list[str], np.ndarray]:
  analysis = engine.analyse(board)
  log_probs = np.asarray(analysis['log_probs'])
  bucket_probs = np.exp(log_probs)
  win_probs = bucket_probs @ engine._return_buckets_values  # pylint: disable=protected-access
  ordered_moves = [move.uci() for move in move_order_fn(board)]
  return ordered_moves, win_probs.astype(np.float64)


def compare_scores(
    sample: pd.Series,
    engine: Any,
    move_order_fn: Callable[[chess.Board], Sequence[chess.Move]],
) -> dict[str, Any]:
  fen = sample['reward_model']['fen']
  move_values = json.loads(sample['reward_model']['move_values_json'])
  board = chess.Board(fen=fen)
  ordered_moves, critic_probs = compute_critic_probs(
      engine, board, move_order_fn
  )
  stockfish_probs = np.array([move_values[move] for move in ordered_moves])
  diffs = critic_probs - stockfish_probs
  corr = (
      float(np.corrcoef(stockfish_probs, critic_probs)[0, 1])
      if len(stockfish_probs) > 1
      else float('nan')
  )
  best_stockfish_idx = int(np.argmax(stockfish_probs))
  best_critic_idx = int(np.argmax(critic_probs))
  per_move = []
  for move, s_prob, c_prob, delta in zip(
      ordered_moves, stockfish_probs, critic_probs, diffs
  ):
    per_move.append(
        {
            'move': move,
            'stockfish': float(s_prob),
            'critic': float(c_prob),
            'abs_diff': float(abs(delta)),
        }
    )
  per_move.sort(key=lambda item: item['stockfish'], reverse=True)
  return {
    'fen': fen,
    'sample_index': int(sample['extra_info'].get('index', -1)),
    'best_stockfish_move': ordered_moves[best_stockfish_idx],
    'best_stockfish_score': float(stockfish_probs[best_stockfish_idx]),
    'best_critic_move': ordered_moves[best_critic_idx],
    'best_critic_score': float(critic_probs[best_critic_idx]),
    'pearson_corr': corr,
    'mean_abs_diff': float(np.mean(np.abs(diffs))),
    'per_move': per_move,
  }


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          'Run the Searchless Chess 270M critic on dataset samples and'
          ' compare its win probabilities to Stockfish-derived scores.'
      )
  )
  parser.add_argument(
      '--dataset',
      type=Path,
      default=Path('data/chess_puzzles/train.parquet'),
      help='Path to the VERL-formatted parquet file.',
  )
  parser.add_argument(
      '--num-samples',
      type=int,
      default=5,
      help='Number of rows to evaluate.',
  )
  parser.add_argument(
      '--seed', type=int, default=0, help='Sampling seed for reproducibility.'
  )
  args = parser.parse_args()

  samples = load_samples(args.dataset, args.num_samples, args.seed)
  searchless_src = prepare_searchless_src()
  searchless_root = searchless_src.parent
  searchless_pkg_parent = searchless_root.parent
  if str(searchless_pkg_parent) not in sys.path:
    sys.path.insert(0, str(searchless_pkg_parent))
  engine_constants = importlib.import_module(
      'searchless_chess.src.engines.constants'
  )
  engine_lib = importlib.import_module('searchless_chess.src.engines.engine')

  print(
      f'Loaded {args.num_samples} samples from {args.dataset}. '
      'Building 270M critic...'
  )
  with pushd(searchless_src):
    engine = engine_constants.ENGINE_BUILDERS['270M']()
  print('Critic ready.')

  for idx, row in samples.iterrows():
    result = compare_scores(
        row, engine, move_order_fn=engine_lib.get_ordered_legal_moves
    )
    print('=' * 80)
    print(f'Sample #{idx} | data index: {result["sample_index"]}')
    print(f'FEN: {result["fen"]}')
    print(
        f'Stockfish best: {result["best_stockfish_move"]} '
        f'({result["best_stockfish_score"]:.3f})'
    )
    print(
        f'Critic best:    {result["best_critic_move"]} '
        f'({result["best_critic_score"]:.3f})'
    )
    print(
        f'Pearson corr: {result["pearson_corr"]:.3f} | '
        f'Mean |diff|: {result["mean_abs_diff"]:.3f}'
    )
    per_move_df = pd.DataFrame(result['per_move'])
    print(
        per_move_df.head(10).to_string(
            index=False,
            formatters={'stockfish': '{:.3f}'.format, 'critic': '{:.3f}'.format,
                        'abs_diff': '{:.3f}'.format},
        )
    )


if __name__ == '__main__':
  main()
