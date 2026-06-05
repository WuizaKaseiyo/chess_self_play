#!/usr/bin/env python3
"""
Sanity-check validation parquet(s) for W&B metric namespace collisions.

Why this exists
---------------
`verl.trainer.ppo.ray_trainer` logs validation metrics with keys like:
  val-core/<data_source>/<var>/<metric>

If you validate on multiple parquet files in one run (via `data.val_files`) and the
rows in those parquets share the same top-level `data_source` value, W&B will show
only one `val-core/...` namespace because the metrics are intentionally aggregated
by `data_source`.

This script prints the `data_source` values per parquet and highlights collisions.

Example
-------
python3 scripts/check_val_parquet_namespaces.py \
  data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet \
  data/chess_puzzles_chessr1_aligned_sharded_baseline/test_shuffled_legal_moves.parquet \
  --print-example-keys
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def _read_unique_data_sources(path: Path) -> list[str]:
    pf = pq.ParquetFile(path)
    schema_cols = pf.schema_arrow.names
    if "data_source" not in schema_cols:
        return ["<missing:data_source>"]
    vals = pf.read(columns=["data_source"]).column("data_source").to_pylist()
    return sorted({str(v) for v in vals})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquets", nargs="+", help="One or more VERL-format parquet files.")
    ap.add_argument(
        "--print-example-keys",
        action="store_true",
        help="Print example `val-core/...` W&B keys implied by the discovered data_source values.",
    )
    ap.add_argument(
        "--fail-on-collision",
        action="store_true",
        help="Exit non-zero if multiple input parquets share any data_source value.",
    )
    args = ap.parse_args()

    source2files: dict[str, list[str]] = {}

    for p in args.parquets:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f"Parquet not found: {path}")

        pf = pq.ParquetFile(path)
        n_rows = pf.metadata.num_rows if pf.metadata is not None else None
        sources = _read_unique_data_sources(path)
        print(f"{path} rows={n_rows} unique_data_source={len(sources)} data_source={sources}")

        for s in sources:
            source2files.setdefault(s, []).append(str(path))

    collisions = {s: files for s, files in source2files.items() if len(files) > 1}
    if collisions:
        print("\n[WARN] data_source collisions detected (validation metrics will be merged):")
        for s, files in sorted(collisions.items()):
            print(f"- data_source={s}")
            for f in files:
                print(f"  - {f}")
        if args.fail_on_collision:
            return 2
    else:
        print("\nOK: no data_source collisions across the provided parquets.")

    if args.print_example_keys:
        print("\nExample W&B metric keys (prefixes):")
        for s in sorted(source2files.keys()):
            print(f"- val-core/{s}/...")
            print(f"- val-aux/{s}/...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

