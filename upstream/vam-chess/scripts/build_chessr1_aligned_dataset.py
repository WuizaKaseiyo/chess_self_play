#!/usr/bin/env python3
"""
Build a VERL-format chess dataset aligned to Chess-R1 positions, scored with Stockfish.

Goal
----
Produce a dataset compatible with this repo's chess reward/training stack:
  - top-level keys: data_source, prompt, ability, reward_model, extra_info
  - reward_model contains per-legal-move eval maps (JSON strings) and baselines

Key behaviors
-------------
- Alignment anchor: Chess-R1 `board_fen` positions.
- Target move (ground_truth): by default, Chess-R1 `next_move_uci` when legal
  (fallback: engine best move by CP at the chosen depth).
- Scoring: Stockfish 16, depth=14 by default, evaluates *all legal moves*
  using MultiPV (with fallback for any missing moves).
- Output prompt contract: uses `<uci_move>...</uci_move>` tags (instead of `<answer>`).
- Parallel + restartable:
  - Work is sharded across `--workers` processes.
  - Each worker writes completed parquet parts and keeps a sqlite cache keyed by
    `(depth, threads, hash_mb, fen)` to skip repeated evaluations on resume.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import chess
    import chess.engine
except Exception as exc:  # pragma: no cover
    raise RuntimeError("python-chess is required for this script.") from exc

import pyarrow as pa
import pyarrow.parquet as pq

# Make repo-root imports work when invoked as `python scripts/xxx.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rescore_puzzles_cp import analyse_all_legal_moves_multipv, centipawn_to_win_prob


DEFAULT_CHESSR1_TRAIN_PARQUET = (
    ".third_party_cache/Chess-R1/data/"
    "lichess_db_puzzle_processed_qwen_instruct_reastemp_fen_legal_rule/train.parquet"
)


def _build_system_prompt_uci_move() -> str:
    # Mirror the in-repo prompt style, but swap the output tag to <uci_move>.
    return (
        "You are a helpful assistant who plays chess professionally.\n"
        "The assistant first thinks through the reasoning process internally and then provides the user with the best move.\n"
        "The reasoning process and the answer must be enclosed within <think> </think> and <uci_move> </uci_move> tags, respectively.\n"
        "The reasoning process should describe how you analyze the position and decide on the best move, including:\n"
        "  - A strategic evaluation of the position.\n"
        "  - A comparison of key candidate moves.\n"
        "  - For each candidate, consider the opponent's likely response and outcome.\n"
        "  - Conclude with a clear justification for your final choice.\n"
        "The answer must be in UCI notation, using the from-square and to-square (e.g., e2e4, g1f3, a7a8q).\n"
        "Now, the user provides a FEN string, and a list of legal moves for the given board.\n"
        "After analyzing the position, clearly state the best move in UCI notation within <uci_move> </uci_move> tags. "
        "i.e., <uci_move> e2e4 </uci_move>\n"
        "\n"
        "Reminder of chess rules:\n"
        "- Bishops move diagonally.\n"
        "- Rooks move horizontally or vertically.\n"
        "- Knights jump in an L-shape.\n"
        "- Queens combine rook and bishop movement.\n"
        "- Kings move one square in any direction.\n"
        "- Pawns move forward, capture diagonally, and can promote."
    )


def _build_user_prompt_uci_moves(fen: str, legal_moves_uci: List[str]) -> str:
    return (
        f"Current FEN string: {fen}\n"
        f"Legal moves (UCI): {', '.join(legal_moves_uci)}\n\n"
        "Let's think step by step."
        "\n\n"
        "IMPORTANT (format): Before your <think> block, output exactly one guess line:\n"
        "<guess> GUESS_UCI </guess>\n"
        "Then output the usual strict answer format:\n"
        "<think> ... </think><uci_move> ... </uci_move>\n"
        "Do not write any other text outside these tags.\n"
    )


def _canonicalize_uci(move: Optional[str]) -> Optional[str]:
    if not move:
        return None
    m = str(move).strip()
    m = m.strip("`'\"")
    m = m.rstrip(".!?;,:")
    m = m.replace("=", "").lower()
    try:
        chess.Move.from_uci(m)
        return m
    except Exception:
        return None


def _json_dumps_sorted(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ChessR1Row:
    index: int
    chessr1_id: str
    chessr1_rating: Optional[int]
    fen: str
    next_move_uci: Optional[str]
    next_move_san: Optional[str]
    prev_moves_uci: Optional[str]
    prev_moves_san: Optional[str]


class FenEvalCache:
    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), timeout=60)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_cache (
              cache_key TEXT PRIMARY KEY,
              move_cps_json TEXT NOT NULL,
              move_expected_scores_json TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get(self, cache_key: str) -> Optional[Tuple[Dict[str, int], Dict[str, float]]]:
        cur = self._conn.execute(
            "SELECT move_cps_json, move_expected_scores_json FROM eval_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cur.fetchone()
        if not row:
            return None
        move_cps = json.loads(row[0])
        move_expected = json.loads(row[1])
        return {k: int(v) for k, v in move_cps.items()}, {k: float(v) for k, v in move_expected.items()}

    def put(self, cache_key: str, move_cps: Dict[str, int], move_expected: Dict[str, float]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO eval_cache(cache_key, move_cps_json, move_expected_scores_json) VALUES (?, ?, ?)",
            (cache_key, _json_dumps_sorted(move_cps), _json_dumps_sorted(move_expected)),
        )

    def commit(self) -> None:
        self._conn.commit()


def _cache_key(*, depth: int, threads: int, hash_mb: int, fen: str) -> str:
    # Keep the key stable and explicit to avoid accidental cross-run reuse.
    return f"depth={depth}|threads={threads}|hash={hash_mb}|fen={fen}"


def _load_chessr1_rows(path: Path, max_rows: Optional[int]) -> List[ChessR1Row]:
    cols = ["id", "rating", "board_fen", "next_move_uci", "next_move_san", "prev_moves_uci", "prev_moves_san"]
    df = pd.read_parquet(path, columns=cols)
    if max_rows is not None:
        df = df.iloc[: max_rows]

    rows: List[ChessR1Row] = []
    for i, r in enumerate(df.itertuples(index=False), start=0):
        rows.append(
            ChessR1Row(
                index=i,
                chessr1_id=str(getattr(r, "id")),
                chessr1_rating=int(getattr(r, "rating")) if getattr(r, "rating") is not None else None,
                fen=str(getattr(r, "board_fen")).strip(),
                next_move_uci=str(getattr(r, "next_move_uci")).strip().lower() if getattr(r, "next_move_uci") else None,
                next_move_san=str(getattr(r, "next_move_san")).strip() if getattr(r, "next_move_san") else None,
                prev_moves_uci=str(getattr(r, "prev_moves_uci")).strip() if getattr(r, "prev_moves_uci") else None,
                prev_moves_san=str(getattr(r, "prev_moves_san")).strip() if getattr(r, "prev_moves_san") else None,
            )
        )
    return rows


def _eval_position(
    *,
    engine: chess.engine.SimpleEngine,
    cache: FenEvalCache,
    fen: str,
    depth: int,
    threads: int,
    hash_mb: int,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    key = _cache_key(depth=depth, threads=threads, hash_mb=hash_mb, fen=fen)
    cached = cache.get(key)
    if cached is not None:
        return cached

    board = chess.Board(fen)
    move_to_cp, move_to_expected = analyse_all_legal_moves_multipv(engine, board, depth=depth)
    if not move_to_cp:
        raise ValueError(f"No legal moves returned for FEN:\n{fen}")

    # Ensure we always store *some* expected-score map for schema stability.
    if not move_to_expected:
        move_to_expected = {m: 0.0 for m in move_to_cp}

    cache.put(key, move_to_cp, move_to_expected)
    return move_to_cp, move_to_expected


def _row_to_verl_dict(
    *,
    row: ChessR1Row,
    move_to_cp: Dict[str, int],
    move_to_expected: Dict[str, float],
    ground_truth_mode: str,
) -> Dict[str, Any]:
    fen = row.fen
    board = chess.Board(fen)
    legal_moves_uci = [m.uci().lower() for m in board.legal_moves]
    legal_set = set(legal_moves_uci)

    # Derived score maps.
    move_to_prob = {m: float(centipawn_to_win_prob(int(cp))) for m, cp in move_to_cp.items()}

    best_cp = max(move_to_cp.values()) if move_to_cp else None
    if best_cp is None:
        raise ValueError(f"Empty move_to_cp for FEN:\n{fen}")
    best_moves = sorted([m for m, cp in move_to_cp.items() if cp == best_cp])
    best_move_uci = best_moves[0] if best_moves else None

    # Choose ground truth.
    gt_label = _canonicalize_uci(row.next_move_uci)
    gt: Optional[str]
    if ground_truth_mode == "label_preserving":
        gt = gt_label if gt_label in legal_set else None
    elif ground_truth_mode == "engine_best":
        gt = None
    else:
        raise ValueError(f"Unknown ground_truth_mode={ground_truth_mode}")
    if gt is None:
        # Fallback to engine best if label is missing/illegal.
        gt = best_move_uci
    if gt is None:
        raise ValueError(f"Unable to determine ground_truth for FEN:\n{fen}")

    system_prompt = _build_system_prompt_uci_move()
    user_prompt = _build_user_prompt_uci_moves(fen, legal_moves_uci)
    prompt_text = f"{system_prompt}\n\n{user_prompt}".strip()

    # Position baselines for delta-style rewards.
    position_expected = max(move_to_expected.values()) if move_to_expected else 0.0

    reward_model = {
        "style": "rule",
        "fen": fen,
        "ground_truth": gt,
        "legal_moves_uci": legal_moves_uci,
        "move_values_json": _json_dumps_sorted({k: float(v) for k, v in move_to_prob.items()}),
        "move_cps_json": _json_dumps_sorted({k: int(v) for k, v in move_to_cp.items()}),
        "move_expected_scores_json": _json_dumps_sorted({k: float(v) for k, v in move_to_expected.items()}),
        "position_cp": int(best_cp),
        "position_win_prob": float(centipawn_to_win_prob(int(best_cp))),
        "position_expected_score": float(position_expected),
        "best_move_uci": str(best_move_uci or ""),
    }

    extra_info = {
        "split": "train",
        "index": int(row.index),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "legal_moves_uci": legal_moves_uci,
        # Chess-R1 traceability
        "chessr1_id": row.chessr1_id,
        "chessr1_rating": int(row.chessr1_rating) if row.chessr1_rating is not None else None,
        "chessr1_next_move_uci": row.next_move_uci,
        "chessr1_next_move_san": row.next_move_san,
        "chessr1_prev_moves_uci": row.prev_moves_uci,
        "chessr1_prev_moves_san": row.prev_moves_san,
        "ground_truth_mode": ground_truth_mode,
    }

    return {
        "data_source": "local/chessr1_aligned",
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "chess",
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def _write_parquet_part(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _load_progress(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return 0, 0
    obj = json.loads(path.read_text())
    return int(obj.get("next_task_idx", 0)), int(obj.get("next_part_idx", 0))


def _save_progress(path: Path, *, next_task_idx: int, next_part_idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"next_task_idx": next_task_idx, "next_part_idx": next_part_idx}))


def _worker_main(
    *,
    worker_id: int,
    rows: List[ChessR1Row],
    output_tmp_dir: Path,
    cache_path: Path,
    engine_path: str,
    depth: int,
    threads: int,
    hash_mb: int,
    ground_truth_mode: str,
    write_batch_size: int,
) -> None:
    output_tmp_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_tmp_dir / f"train_worker{worker_id}.progress.json"
    next_task_idx, next_part_idx = _load_progress(progress_path)

    cache = FenEvalCache(cache_path)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    t0 = time.time()
    n_since_commit = 0
    try:
        engine.configure({"Threads": threads, "Hash": hash_mb, "UCI_ShowWDL": True})

        batch: List[Dict[str, Any]] = []
        for local_i in range(next_task_idx, len(rows)):
            r = rows[local_i]
            try:
                move_to_cp, move_to_expected = _eval_position(
                    engine=engine,
                    cache=cache,
                    fen=r.fen,
                    depth=depth,
                    threads=threads,
                    hash_mb=hash_mb,
                )
                out = _row_to_verl_dict(
                    row=r,
                    move_to_cp=move_to_cp,
                    move_to_expected=move_to_expected,
                    ground_truth_mode=ground_truth_mode,
                )
                batch.append(out)
            except Exception as exc:
                # Research-mode: skip bad rows but keep a breadcrumb.
                print(f"[worker {worker_id}] skip index={r.index} id={r.chessr1_id}: {exc}", file=sys.stderr)

            n_since_commit += 1
            if n_since_commit >= 50:
                cache.commit()
                n_since_commit = 0

            if len(batch) >= write_batch_size:
                part_path = output_tmp_dir / f"train_worker{worker_id}.part{next_part_idx:05d}.parquet"
                _write_parquet_part(batch, part_path)
                next_part_idx += 1
                next_task_idx = local_i + 1
                _save_progress(progress_path, next_task_idx=next_task_idx, next_part_idx=next_part_idx)
                batch = []

        if batch:
            part_path = output_tmp_dir / f"train_worker{worker_id}.part{next_part_idx:05d}.parquet"
            _write_parquet_part(batch, part_path)
            next_part_idx += 1
            next_task_idx = len(rows)
            _save_progress(progress_path, next_task_idx=next_task_idx, next_part_idx=next_part_idx)
            batch = []

        cache.commit()
    finally:
        try:
            engine.quit()
        except Exception:
            pass
        cache.close()
    elapsed = time.time() - t0
    print(f"[worker {worker_id}] done in {elapsed:.1f}s ({len(rows)} assigned)")


def _merge_parts_to_single_parquet(parts: List[Path], out_path: Path) -> None:
    if not parts:
        raise ValueError("No parquet parts found to merge.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    writer: Optional[pq.ParquetWriter] = None
    try:
        for p in sorted(parts):
            table = pq.read_table(p)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _split_parquet_into_shards(
    *,
    input_path: Path,
    output_dir: Path,
    prefix: str,
    num_shards: int,
    overwrite: bool,
) -> List[Path]:
    if num_shards <= 1:
        raise ValueError(f"num_shards must be >=2, got {num_shards}")

    pf = pq.ParquetFile(input_path)
    meta = pf.metadata
    if meta is None:
        raise ValueError(f"Missing parquet metadata: {input_path}")

    schema = pf.schema_arrow
    compression = meta.row_group(0).column(0).compression if meta.num_row_groups > 0 else "snappy"

    total_rows = int(meta.num_rows)
    rows_per_shard = (total_rows + num_shards - 1) // num_shards

    out_paths = [output_dir / f"{prefix}_{i}.parquet" for i in range(num_shards)]
    for p in out_paths:
        if p.exists():
            if overwrite:
                p.unlink()
            else:
                raise FileExistsError(f"Refusing to overwrite {p}; pass --overwrite.")

    writers: List[pq.ParquetWriter] = []
    counts = [0 for _ in range(num_shards)]
    try:
        for p in out_paths:
            writers.append(pq.ParquetWriter(p, schema=schema, compression=compression))

        shard_idx = 0
        shard_rows = 0
        for rg in range(meta.num_row_groups):
            table = pf.read_row_group(rg)
            n = int(table.num_rows)

            # Advance to the next shard once we hit the target size (keep last shard as the sink).
            if shard_idx < (num_shards - 1) and shard_rows >= rows_per_shard:
                shard_idx += 1
                shard_rows = 0

            writers[shard_idx].write_table(table)
            counts[shard_idx] += n
            shard_rows += n
    finally:
        for w in writers:
            try:
                w.close()
            except Exception:
                pass

    print(f"Split {input_path} ({total_rows} rows) into {len(out_paths)} shards:")
    for p, n in zip(out_paths, counts):
        print(f"  - {p} rows={n}")
    return out_paths


def _rewrite_searchless_eval_to_uci_move(*, input_path: Path, output_path: Path) -> None:
    # Keep the eval set flexible:
    # - If `input_path` is a parquet file, rewrite just that file.
    # - If `input_path` is a directory, rewrite `train.parquet` + `test.parquet` from it
    #   (this matches "entire Searchless 10k" when pointing at `data/chess_puzzles/`).
    if input_path.is_dir():
        train_path = input_path / "train.parquet"
        test_path = input_path / "test.parquet"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "When --searchless_eval_parquet points to a directory, it must contain "
                f"`train.parquet` and `test.parquet`.\nGot: {input_path}"
            )
        train_table = pq.read_table(train_path)
        test_table = pq.read_table(test_path)
        try:
            table = pa.concat_tables([train_table, test_table], promote_options="default")
        except TypeError:  # pragma: no cover
            # Backwards-compat for older pyarrow.
            table = pa.concat_tables([train_table, test_table], promote=True)
    else:
        table = pq.read_table(input_path)
    rows = table.to_pylist()

    system_prompt = _build_system_prompt_uci_move()
    out_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        rm = r["reward_model"] or {}
        fen = str(rm.get("fen") or "").strip()
        legal_moves = [str(m).strip().lower() for m in (rm.get("legal_moves_uci") or []) if str(m).strip()]
        user_prompt = _build_user_prompt_uci_moves(fen, legal_moves)
        prompt_text = f"{system_prompt}\n\n{user_prompt}".strip()

        extra = dict(r.get("extra_info") or {})
        # For this eval parquet, treat every row as part of the eval split (even if it
        # originated from the Searchless "train" split in this repo).
        extra["split"] = "test"
        extra["index"] = int(i)
        extra["system_prompt"] = system_prompt
        extra["user_prompt"] = user_prompt
        extra["legal_moves_uci"] = legal_moves

        out_rows.append(
            {
                "data_source": r.get("data_source", "local/chess_puzzles"),
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": r.get("ability", "chess"),
                "reward_model": rm,
                "extra_info": extra,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_table = pa.Table.from_pylist(out_rows)

    # Preserve the exact schema of the existing eval parquet when overwriting, so downstream
    # code sees a stable Arrow struct layout.
    if output_path.exists():
        target_schema = pq.read_schema(output_path)
        out_table = out_table.cast(target_schema)

    pq.write_table(out_table, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chessr1_train_parquet", default=DEFAULT_CHESSR1_TRAIN_PARQUET)
    parser.add_argument(
        "--searchless_eval_parquet",
        default="data/chess_puzzles",
        help=(
            "Searchless eval source. If a parquet file: rewrite that file. "
            "If a directory: rewrite `train.parquet` + `test.parquet` inside it (10k total for `data/chess_puzzles/`)."
        ),
    )
    parser.add_argument("--output_dir", default="data/chess_puzzles_chessr1_aligned_sharded")
    parser.add_argument("--max_rows", type=int, default=None, help="Process only the first N Chess-R1 rows (smoke-test).")
    parser.add_argument("--engine_path", default=".third_party_cache/stockfish/src/stockfish")
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--hash_mb", type=int, default=256)
    parser.add_argument(
        "--train_shards",
        type=int,
        default=2,
        help="Write train set as `train_*.parquet` shards (default=2) to keep each file under GitHub's 100MB limit.",
    )
    parser.add_argument(
        "--ground_truth_mode",
        choices=["label_preserving", "engine_best"],
        default="label_preserving",
    )
    parser.add_argument("--write_batch_size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only_train", action="store_true")
    parser.add_argument("--only_eval", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_train_single = output_dir / "train.parquet"
    out_train_shards = [output_dir / f"train_{i}.parquet" for i in range(int(args.train_shards))]
    out_test = output_dir / "test.parquet"

    if not args.overwrite:
        if not args.only_eval:
            if int(args.train_shards) <= 1:
                if out_train_single.exists():
                    raise SystemExit(f"Refusing to overwrite {out_train_single}; pass --overwrite.")
            else:
                if any(p.exists() for p in out_train_shards):
                    raise SystemExit(f"Refusing to overwrite existing train shards under {output_dir}; pass --overwrite.")
        if not args.only_train and out_test.exists():
            raise SystemExit(f"Refusing to overwrite {out_test}; pass --overwrite.")

    if not args.only_eval:
        chessr1_rows = _load_chessr1_rows(Path(args.chessr1_train_parquet), args.max_rows)
        if not chessr1_rows:
            raise SystemExit("No Chess-R1 rows loaded.")

        # Deterministic sharding by row index.
        shards: List[List[ChessR1Row]] = [[] for _ in range(int(args.workers))]
        for r in chessr1_rows:
            shards[r.index % int(args.workers)].append(r)
        for s in shards:
            s.sort(key=lambda x: x.index)

        import multiprocessing as mp

        procs: List[mp.Process] = []
        for wid in range(int(args.workers)):
            worker_tmp = tmp_dir / f"worker{wid:02d}"
            cache_path = Path(f".third_party_cache/chessr1_aligned_depth{int(args.depth)}_stockfish_cache.worker{wid}.sqlite3")
            p = mp.Process(
                target=_worker_main,
                kwargs=dict(
                    worker_id=wid,
                    rows=shards[wid],
                    output_tmp_dir=worker_tmp,
                    cache_path=cache_path,
                    engine_path=str(args.engine_path),
                    depth=int(args.depth),
                    threads=int(args.threads),
                    hash_mb=int(args.hash_mb),
                    ground_truth_mode=str(args.ground_truth_mode),
                    write_batch_size=int(args.write_batch_size),
                ),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise SystemExit(f"Worker exited with code {p.exitcode}")

        parts = sorted(tmp_dir.glob("worker*/train_worker*.part*.parquet"))
        print(f"Merging {len(parts)} train parts -> {out_train_single}")
        _merge_parts_to_single_parquet(parts, out_train_single)

        if int(args.train_shards) > 1:
            _split_parquet_into_shards(
                input_path=out_train_single,
                output_dir=output_dir,
                prefix="train",
                num_shards=int(args.train_shards),
                overwrite=bool(args.overwrite),
            )
            out_train_single.unlink()

    if not args.only_train:
        print(f"Rewriting eval prompts -> {out_test}")
        _rewrite_searchless_eval_to_uci_move(
            input_path=Path(args.searchless_eval_parquet),
            output_path=out_test,
        )

    print("Done.")  # keep as a simple completion marker for logs


if __name__ == "__main__":
    main()
