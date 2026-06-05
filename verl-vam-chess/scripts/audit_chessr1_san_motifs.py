#!/usr/bin/env python3
"""
Audit SAN-explicit tactical motifs for Chess(-R1-aligned) datasets that store labeled moves in UCI.

Goal
----
Given a dataset row with:
  - a position (FEN), and
  - a labeled move in UCI (e.g. reward_model.ground_truth),

convert the labeled UCI move to SAN for that *pre-move* position and measure how often the move is:
  - checkmate (#), check (+),
  - capture (x), promotion (=), castling (O-O / O-O-O),

where SAN makes these motifs explicit but UCI does not (especially check/mate).

This script performs only rules-based parsing with python-chess (no model inference).
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import chess
import pyarrow.parquet as pq


def _as_str(x: Any) -> str:
    return str(x or "").strip()


def _norm_uci(x: Any) -> str:
    return _as_str(x).lower()


def _pct(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return 100.0 * float(n) / float(d)


def _shorten(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


@dataclass
class MoveAnalysis:
    uci: str
    san: str
    is_legal: bool
    is_capture: bool
    is_castling: bool
    is_promotion: bool
    is_check: bool
    is_checkmate: bool
    san_has_hash: bool
    san_has_plus: bool
    san_has_x: bool
    san_has_eq: bool
    san_is_castle: bool


def _analyze_uci_move(board: chess.Board, uci: str) -> MoveAnalysis:
    uci_norm = _norm_uci(uci)
    move = chess.Move.from_uci(uci_norm)
    is_legal = board.is_legal(move)
    if not is_legal:
        # Still fill fields deterministically.
        return MoveAnalysis(
            uci=uci_norm,
            san="",
            is_legal=False,
            is_capture=False,
            is_castling=False,
            is_promotion=move.promotion is not None,
            is_check=False,
            is_checkmate=False,
            san_has_hash=False,
            san_has_plus=False,
            san_has_x=False,
            san_has_eq=False,
            san_is_castle=False,
        )

    # SAN must be computed on the *pre-move* board.
    san = board.san(move)

    is_capture = board.is_capture(move)
    is_castling = board.is_castling(move)
    is_promotion = move.promotion is not None

    board.push(move)
    try:
        is_check = board.is_check()
        is_checkmate = board.is_checkmate()
    finally:
        board.pop()

    san_has_hash = "#" in san
    san_has_plus = "+" in san
    san_has_x = "x" in san
    san_has_eq = "=" in san
    # python-chess uses letter 'O' (not zero) for castling.
    san_is_castle = san.startswith("O-O")

    return MoveAnalysis(
        uci=uci_norm,
        san=san,
        is_legal=True,
        is_capture=bool(is_capture),
        is_castling=bool(is_castling),
        is_promotion=bool(is_promotion),
        is_check=bool(is_check),
        is_checkmate=bool(is_checkmate),
        san_has_hash=bool(san_has_hash),
        san_has_plus=bool(san_has_plus),
        san_has_x=bool(san_has_x),
        san_has_eq=bool(san_has_eq),
        san_is_castle=bool(san_is_castle),
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _iter_rows(
    parquet_path: Path,
    columns: list[str],
    batch_size: int,
    limit_rows: Optional[int],
) -> Iterable[dict[str, Any]]:
    pf = pq.ParquetFile(parquet_path)
    n_seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
        for row in batch.to_pylist():
            yield row
            n_seen += 1
            if limit_rows is not None and n_seen >= int(limit_rows):
                return


def _sample_append(buf: list[Any], item: Any, cap: int) -> None:
    if cap <= 0:
        return
    if len(buf) < cap:
        buf.append(item)


def _reservoir_update(buf: list[Any], item: Any, seen: int, cap: int) -> int:
    """Reservoir-sample up to `cap` items from a stream.

    Returns the updated `seen` counter (number of items observed for this bucket).
    """
    if cap <= 0:
        return seen + 1
    seen += 1
    if len(buf) < cap:
        buf.append(item)
        return seen
    j = random.randrange(seen)
    if j < cap:
        buf[j] = item
    return seen


def _markdown_escape_codeblock(s: str) -> str:
    # Prevent accidental triple-backtick sequences from terminating a fenced block.
    return s.replace("```", "``\\`")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet",
        nargs="+",
        default=["data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet"],
        help="One or more parquet files to audit (concatenated in-order).",
    )
    ap.add_argument(
        "--label_field",
        default="ground_truth",
        choices=["ground_truth", "best_move_uci"],
        help="Which reward_model field to treat as the labeled move.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Row batch size for parquet iteration.",
    )
    ap.add_argument("--limit_rows", type=int, default=None, help="Optional cap for quick iteration.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples_per_bucket", type=int, default=8)
    ap.add_argument("--title", default=None, help="Optional markdown report title override.")
    ap.add_argument("--out_md", default=None, help="Optional markdown output path (e.g. analysis/foo.md).")
    ap.add_argument("--out_json", default=None, help="Optional JSON output path (machine-readable summary).")
    args = ap.parse_args()

    random.seed(int(args.seed))

    # Prefer a portable repro command over an environment-specific sys.executable path.
    cmd_str = shlex.join(["python"] + sys.argv)

    parquet_paths = [Path(p) for p in args.parquet]
    for p in parquet_paths:
        if not p.exists():
            raise SystemExit(f"Missing parquet: {p}")

    per_file_rows: list[dict[str, Any]] = []
    total_rows = 0
    for p in parquet_paths:
        pf = pq.ParquetFile(p)
        n = pf.metadata.num_rows if pf.metadata is not None else None
        per_file_rows.append({"path": str(p), "rows": n})
        total_rows += int(n or 0)

    columns = ["reward_model", "extra_info"]

    counts: Dict[str, int] = {}

    def inc(key: str, delta: int = 1) -> None:
        counts[key] = counts.get(key, 0) + int(delta)

    error_examples: Dict[str, list[dict[str, Any]]] = {}

    def record_error(kind: str, row_id: Any, fen: Any, uci: Any, msg: str, source: str) -> None:
        inc(f"err/{kind}")
        buf = error_examples.setdefault(kind, [])
        _sample_append(
            buf,
            {
                "source": source,
                "row_id": row_id,
                "fen": _as_str(fen),
                "uci": _as_str(uci),
                "msg": msg,
            },
            cap=int(args.examples_per_bucket),
        )

    motif_examples: Dict[str, list[dict[str, Any]]] = {}
    motif_seen: Dict[str, int] = {}

    def record_example(kind: str, row_id: Any, fen: str, uci: str, san: str, source: str) -> None:
        buf = motif_examples.setdefault(kind, [])
        seen = motif_seen.get(kind, 0)
        motif_seen[kind] = _reservoir_update(
            buf,
            {"source": source, "row_id": row_id, "fen": fen, "uci": uci, "san": san},
            seen=seen,
            cap=int(args.examples_per_bucket),
        )

    # Track per-row mismatch between reward_model.ground_truth and reward_model.best_move_uci
    # for context: some datasets store both (and they sometimes differ).
    gt_best_mismatch_examples: list[dict[str, Any]] = []
    gt_best_mismatch_seen = 0

    # SAN vs state consistency diagnostics.
    mismatch_examples: Dict[str, list[dict[str, Any]]] = {}

    def record_mismatch(kind: str, row_id: Any, fen: str, uci: str, san: str, detail: str, source: str) -> None:
        inc(f"mismatch/{kind}")
        buf = mismatch_examples.setdefault(kind, [])
        _sample_append(
            buf,
            {
                "source": source,
                "row_id": row_id,
                "fen": fen,
                "uci": uci,
                "san": san,
                "detail": detail,
            },
            cap=int(args.examples_per_bucket),
        )

    analyzed_rows = 0
    ok_rows = 0

    # Motif counts for the chosen label field.
    limit_rows = int(args.limit_rows) if args.limit_rows is not None else None
    for parquet_path in parquet_paths:
        source = str(parquet_path)
        remaining = None if limit_rows is None else max(limit_rows - analyzed_rows, 0)
        if remaining == 0:
            break
        for row in _iter_rows(
            parquet_path=parquet_path,
            columns=columns,
            batch_size=int(args.batch_size),
            limit_rows=remaining,
        ):
            analyzed_rows += 1

            rm = row.get("reward_model") or {}
            extra = row.get("extra_info") or {}
            if not isinstance(rm, dict):
                record_error(
                    "reward_model_not_dict",
                    (extra.get("index") if isinstance(extra, dict) else None),
                    "",
                    "",
                    f"type={type(rm)}",
                    source=source,
                )
                continue
            if not isinstance(extra, dict):
                extra = {}

            row_id = extra.get("index")
            fen = _as_str(rm.get("fen"))
            if not fen:
                record_error("missing_fen", row_id, fen, "", "reward_model.fen missing/empty", source=source)
                continue

            gt = _norm_uci(rm.get("ground_truth"))
            best = _norm_uci(rm.get("best_move_uci"))
            if gt and best and gt != best:
                inc("ground_truth!=best_move_uci")
                gt_best_mismatch_seen = _reservoir_update(
                    gt_best_mismatch_examples,
                    {"source": source, "row_id": row_id, "fen": fen, "ground_truth": gt, "best_move_uci": best},
                    seen=gt_best_mismatch_seen,
                    cap=int(args.examples_per_bucket),
                )

            label_uci = _norm_uci(rm.get(str(args.label_field)))
            if not label_uci:
                record_error(
                    "missing_label_uci",
                    row_id,
                    fen,
                    label_uci,
                    f"missing reward_model.{args.label_field}",
                    source=source,
                )
                continue

            try:
                board = chess.Board(fen)
            except Exception as e:
                record_error("invalid_fen", row_id, fen, label_uci, repr(e), source=source)
                continue

            try:
                ma = _analyze_uci_move(board, label_uci)
            except Exception as e:
                record_error("move_parse_or_san_exception", row_id, fen, label_uci, repr(e), source=source)
                continue

            if not ma.is_legal:
                record_error("illegal_uci_move", row_id, fen, label_uci, "move not legal in position", source=source)
                continue

            ok_rows += 1

            # Core motifs.
            if ma.is_checkmate:
                inc("state/checkmate")
                record_example("checkmate", row_id, fen, ma.uci, ma.san, source=source)
            if ma.san_has_hash:
                inc("san/#")
            if ma.is_check and not ma.is_checkmate:
                inc("state/check_nonmate")
                record_example("check_nonmate", row_id, fen, ma.uci, ma.san, source=source)
            if ma.san_has_plus and not ma.san_has_hash:
                inc("san/+_nonmate")

            if ma.is_capture:
                inc("state/capture")
                record_example("capture", row_id, fen, ma.uci, ma.san, source=source)
            if ma.san_has_x:
                inc("san/x")

            if ma.is_castling:
                inc("state/castle")
                record_example("castle", row_id, fen, ma.uci, ma.san, source=source)
            if ma.san_is_castle:
                inc("san/O-O*")

            if ma.is_promotion:
                inc("state/promotion")
                record_example("promotion", row_id, fen, ma.uci, ma.san, source=source)
            if ma.san_has_eq:
                inc("san/=")

            if (not ma.is_check) and (not ma.is_checkmate):
                record_example("no_check_no_mate", row_id, fen, ma.uci, ma.san, source=source)

            # SAN/state consistency (debugging / validation).
            if ma.san_has_hash and not ma.is_checkmate:
                record_mismatch(
                    "san_hash_but_not_checkmate",
                    row_id,
                    fen,
                    ma.uci,
                    ma.san,
                    "SAN has #, state not mate",
                    source=source,
                )
            if ma.is_checkmate and not ma.san_has_hash:
                record_mismatch(
                    "checkmate_but_san_no_hash",
                    row_id,
                    fen,
                    ma.uci,
                    ma.san,
                    "state mate, SAN missing #",
                    source=source,
                )
            if (ma.san_has_plus and not ma.san_has_hash) and not ma.is_check:
                record_mismatch(
                    "san_plus_but_not_check",
                    row_id,
                    fen,
                    ma.uci,
                    ma.san,
                    "SAN has +, state not check",
                    source=source,
                )
            if (ma.is_check and not ma.is_checkmate) and not (ma.san_has_plus and not ma.san_has_hash):
                record_mismatch(
                    "check_nonmate_but_san_no_plus",
                    row_id,
                    fen,
                    ma.uci,
                    ma.san,
                    "state check (non-mate), SAN missing +",
                    source=source,
                )
            if ma.san_has_x and not ma.is_capture:
                record_mismatch(
                    "san_x_but_not_capture",
                    row_id,
                    fen,
                    ma.uci,
                    ma.san,
                    "SAN has x, state not capture",
                    source=source,
                )
            if ma.is_capture and not ma.san_has_x:
                record_mismatch(
                    "capture_but_san_no_x",
                    row_id,
                    fen,
                    ma.uci,
                    ma.san,
                    "state capture, SAN missing x",
                    source=source,
                )
            if ma.san_is_castle and not ma.is_castling:
                record_mismatch(
                    "san_castle_but_not_castle",
                    row_id,
                    fen,
                    ma.uci,
                    ma.san,
                    "SAN is O-O*, state not castle",
                    source=source,
                )
            if ma.is_castling and not ma.san_is_castle:
                record_mismatch(
                    "castle_but_san_no_castle",
                    row_id,
                    fen,
                    ma.uci,
                    ma.san,
                    "state castle, SAN missing O-O*",
                    source=source,
                )
            if ma.san_has_eq and not ma.is_promotion:
                record_mismatch(
                    "san_eq_but_not_promotion",
                    row_id,
                    fen,
                    ma.uci,
                    ma.san,
                    "SAN has =, state not promotion",
                    source=source,
                )
            if ma.is_promotion and not ma.san_has_eq:
                record_mismatch(
                    "promotion_but_san_no_eq",
                    row_id,
                    fen,
                    ma.uci,
                    ma.san,
                    "state promotion, SAN missing =",
                    source=source,
                )

    # Derived sanity counters.
    inc("total_rows", analyzed_rows)
    inc("ok_rows", ok_rows)
    inc("failed_rows", analyzed_rows - ok_rows)

    # Convenience aggregates (these are the "SAN-obvious but UCI-not-obvious" core of this audit).
    counts["state/check_or_mate"] = counts.get("state/checkmate", 0) + counts.get("state/check_nonmate", 0)
    counts["san/#_or_+"] = counts.get("san/#", 0) + counts.get("san/+_nonmate", 0)

    # Print summary to stdout.
    label_field = str(args.label_field)
    parq_str = ", ".join(str(p) for p in parquet_paths)
    print(f"[LOAD] parquets=[{parq_str}] total_rows_metadata={total_rows} analyzed_rows={analyzed_rows}")
    print(f"[LABEL] reward_model.{label_field} (UCI)")
    print(f"[CMD] {cmd_str}")
    print(f"[OK] ok_rows={ok_rows} failed_rows={analyzed_rows - ok_rows}")
    if analyzed_rows:
        print(f"  ok_pct={_pct(ok_rows, analyzed_rows):.2f}%")

    # Error breakdown.
    err_keys = sorted(k for k in counts.keys() if k.startswith("err/"))
    if err_keys:
        print("[ERRORS]")
        for k in err_keys:
            print(f"  {k}: {counts[k]}")
    else:
        print("[ERRORS] none")

    # Motif counts.
    print("[MOTIFS] (counts and percent of ok_rows)")
    for k in [
        "state/checkmate",
        "san/#",
        "state/check_nonmate",
        "san/+_nonmate",
        "state/check_or_mate",
        "san/#_or_+",
        "state/capture",
        "san/x",
        "state/castle",
        "san/O-O*",
        "state/promotion",
        "san/=",
        "ground_truth!=best_move_uci",
    ]:
        v = counts.get(k, 0)
        d = ok_rows if k.startswith(("state/", "san/")) else analyzed_rows
        print(f"  {k}: {v} ({_pct(v, d):.2f}%)")

    mismatch_keys = sorted(k for k in counts.keys() if k.startswith("mismatch/"))
    if mismatch_keys:
        print("[SAN<->STATE MISMATCHES]")
        for k in mismatch_keys:
            print(f"  {k}: {counts[k]}")
    else:
        print("[SAN<->STATE MISMATCHES] none")

    # Emit example rows for sanity checking.
    def _print_examples(title: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        print(f"[EXAMPLES] {title} (showing up to {len(rows)})")
        for ex in rows:
            print(
                "  "
                + json.dumps(
                    {
                        "row_id": ex.get("row_id"),
                        "source": ex.get("source"),
                        "uci": ex.get("uci"),
                        "san": ex.get("san"),
                        "fen": _shorten(_as_str(ex.get("fen")), 120),
                    },
                    ensure_ascii=False,
                )
            )

    _print_examples("checkmate", motif_examples.get("checkmate", []))
    _print_examples("check_nonmate", motif_examples.get("check_nonmate", []))
    _print_examples("no_check_no_mate", motif_examples.get("no_check_no_mate", []))
    _print_examples("capture", motif_examples.get("capture", []))
    _print_examples("castle", motif_examples.get("castle", []))
    _print_examples("promotion", motif_examples.get("promotion", []))

    if gt_best_mismatch_examples:
        print(f"[EXAMPLES] ground_truth!=best_move_uci (showing up to {len(gt_best_mismatch_examples)})")
        for ex in gt_best_mismatch_examples:
            print(
                "  "
                + json.dumps(
                    {
                        "row_id": ex.get("row_id"),
                        "source": ex.get("source"),
                        "ground_truth": ex.get("ground_truth"),
                        "best_move_uci": ex.get("best_move_uci"),
                        "fen": _shorten(_as_str(ex.get("fen")), 120),
                    },
                    ensure_ascii=False,
                )
            )

    # Write markdown report.
    now = datetime.now(timezone.utc).astimezone()
    report: Dict[str, Any] = {
        "timestamp": now.isoformat(),
        "parquets": [str(p) for p in parquet_paths],
        "per_file_rows": per_file_rows,
        "label_field": f"reward_model.{label_field}",
        "command": cmd_str,
        "args": vars(args),
        "python_chess_version": getattr(chess, "__version__", "unknown"),
        "title": (str(args.title).strip() if args.title else None),
        "counts": dict(sorted(counts.items())),
        "error_examples": error_examples,
        "motif_examples": motif_examples,
        "mismatch_examples": mismatch_examples,
        "ground_truth_best_mismatch_examples": gt_best_mismatch_examples,
    }

    if args.out_json:
        out_json = Path(args.out_json)
        _ensure_parent(out_json)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"[WRITE] {out_json}")

    if args.out_md:
        out_md = Path(args.out_md)
        _ensure_parent(out_md)

        ok = ok_rows
        analyzed = analyzed_rows
        mate = counts.get("state/checkmate", 0)
        mate_san = counts.get("san/#", 0)
        check_nm = counts.get("state/check_nonmate", 0)
        check_san = counts.get("san/+_nonmate", 0)
        check_or_mate = counts.get("state/check_or_mate", mate + check_nm)

        md_title = str(args.title).strip() if args.title else "SAN Motif Audit"

        lines: list[str] = []
        lines.append(f"# {md_title}")
        lines.append("")
        lines.append(f"- Timestamp: `{now.isoformat()}`")
        if len(parquet_paths) == 1:
            lines.append(f"- Parquet: `{parquet_paths[0]}`")
        else:
            lines.append("- Parquets:")
            for p in parquet_paths:
                lines.append(f"  - `{p}`")
        lines.append(f"- Label move: `{report['label_field']}` (UCI)")
        lines.append(f"- python-chess: `{report['python_chess_version']}`")
        lines.append(f"- Command: `{cmd_str}`")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Total rows analyzed: **{analyzed}**")
        lines.append(f"- Successfully parsed + legal move: **{ok}** ({_pct(ok, analyzed):.2f}%)")
        lines.append(f"- Failed rows: **{analyzed - ok}** ({_pct(analyzed - ok, analyzed):.2f}%)")
        lines.append("")
        lines.append("## Check / Mate (SAN-obvious, UCI-not-obvious)")
        lines.append("")
        lines.append(f"- Checkmate by board state (`is_checkmate()` after move): **{mate}** ({_pct(mate, ok):.2f}%)")
        lines.append(f"- Checkmate by SAN marker (`#` in SAN): **{mate_san}** ({_pct(mate_san, ok):.2f}%)")
        lines.append(f"- Check (non-mate) by board state (`is_check()` and not mate): **{check_nm}** ({_pct(check_nm, ok):.2f}%)")
        lines.append(f"- Check (non-mate) by SAN marker (`+` in SAN, excluding `#`): **{check_san}** ({_pct(check_san, ok):.2f}%)")
        lines.append(f"- Check **or** mate (combined): **{check_or_mate}** ({_pct(check_or_mate, ok):.2f}%)")
        lines.append("")
        lines.append("## Other SAN-explicit motifs (deducible from UCI+FEN, but not explicit in UCI)")
        lines.append("")
        for k, title in [
            ("state/capture", "Capture (`x` in SAN)"),
            ("state/castle", "Castling (`O-O` / `O-O-O` in SAN)"),
            ("state/promotion", "Promotion (`=Q` etc in SAN; promotion piece is explicit in UCI)"),
        ]:
            v = counts.get(k, 0)
            lines.append(f"- {title} by board state: **{v}** ({_pct(v, ok):.2f}%)")
        lines.append("")
        lines.append("## Label field consistency")
        lines.append("")
        v = counts.get("ground_truth!=best_move_uci", 0)
        lines.append(
            f"- Rows where `reward_model.ground_truth != reward_model.best_move_uci`: **{v}** ({_pct(v, analyzed):.2f}%)"
        )
        lines.append("")
        lines.append("## Errors")
        lines.append("")
        if err_keys:
            for k in err_keys:
                lines.append(f"- `{k}`: **{counts[k]}**")
        else:
            lines.append("- None")
        lines.append("")
        lines.append("## SAN vs board-state consistency checks")
        lines.append("")
        if mismatch_keys:
            for k in mismatch_keys:
                lines.append(f"- `{k}`: **{counts[k]}**")
        else:
            lines.append("- No mismatches detected for check/mate SAN markers vs `python-chess` state.")
        lines.append("")
        lines.append("## Example rows")
        lines.append("")

        def emit_bucket(name: str, bucket: list[dict[str, Any]]) -> None:
            if not bucket:
                return
            lines.append(f"### {name}")
            lines.append("")
            for ex in bucket:
                rid = ex.get("row_id")
                src = ex.get("source")
                fen_s = _as_str(ex.get("fen"))
                uci_s = _as_str(ex.get("uci"))
                san_s = _as_str(ex.get("san"))
                lines.append(f"- row_id={rid} | source=`{src}` | uci=`{uci_s}` | san=`{san_s}`")
                lines.append(f"  - fen=`{fen_s}`")
            lines.append("")

        emit_bucket("Checkmate (#)", motif_examples.get("checkmate", []))
        emit_bucket("Check (non-mate, +)", motif_examples.get("check_nonmate", []))
        emit_bucket("No check / no mate", motif_examples.get("no_check_no_mate", []))
        emit_bucket("Captures (x)", motif_examples.get("capture", []))
        emit_bucket("Castling (O-O / O-O-O)", motif_examples.get("castle", []))
        emit_bucket("Promotions (=)", motif_examples.get("promotion", []))

        if gt_best_mismatch_examples:
            lines.append("### ground_truth != best_move_uci")
            lines.append("")
            for ex in gt_best_mismatch_examples:
                src = ex.get('source')
                lines.append(
                    f"- row_id={ex.get('row_id')} | source=`{src}` | ground_truth=`{ex.get('ground_truth')}` | best_move_uci=`{ex.get('best_move_uci')}`"
                )
                lines.append(f"  - fen=`{_as_str(ex.get('fen'))}`")
            lines.append("")

        if mismatch_examples:
            lines.append("### SAN/state mismatches (debug)")
            lines.append("")
            for kind, bucket in mismatch_examples.items():
                lines.append(f"- {kind}:")
                for ex in bucket:
                    src = ex.get('source')
                    lines.append(
                        f"  - row_id={ex.get('row_id')} | source=`{src}` | uci=`{ex.get('uci')}` | san=`{ex.get('san')}` | detail={_markdown_escape_codeblock(_as_str(ex.get('detail')))}"
                    )
            lines.append("")

        out_md.write_text("\n".join(lines) + "\n")
        print(f"[WRITE] {out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
