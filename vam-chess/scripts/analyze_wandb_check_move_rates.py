#!/usr/bin/env python3
"""
Compute per-step check / mate rates from W&B-synced chess rollout logs.

This is designed for investigations like:
  "Does the model increasingly output checking moves as training progresses,
   and does that correlate with effective batch size?"

It operates purely offline on a folder created by:
  python scripts/download_wandb_run_evidence.py --download-files ...

Inputs (expected under <wandb_evidence_dir>/files/):
  - allowed_move_elim_rounds/<step>_round<r>.jsonl   (training-time iterative rounds)
  - validation_logs/<step>.jsonl                     (periodic validation rollouts)

Each JSONL record is assumed to include (at least):
  - input: prompt text containing a line like "Position (FEN): <fen>"
  - pred_move: predicted move in UCI (lowercase)
  - penalty_applied: bool (True => invalid / out-of-subset / format error)
  - gt_uci: dataset ground truth move (UCI)

We parse FEN from the prompt and use python-chess to determine whether a given
UCI move gives check / mate (post-move board state).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import chess
import pandas as pd


# Prompt parsing:
# - Older prompts: "Current FEN string:"
# - Newer prompts: "Position (FEN):"
FEN_RE = re.compile(
    r"^\s*(?:Current FEN string|Position\s*\(FEN\)):\s*(?P<fen>.+?)\s*$",
    flags=re.MULTILINE | re.IGNORECASE,
)

# Selection prompts include:
#   "Allowed moves (UCI): e2e4, d2d4, ..."
ALLOWED_MOVES_RE = re.compile(
    r"^\s*Allowed moves\s*\(UCI\)\s*:\s*(?P<moves>.+?)\s*$",
    flags=re.MULTILINE | re.IGNORECASE,
)

ROUND_FILE_RE = re.compile(r"^(?P<step>\d+)_round(?P<round>\d+)\.jsonl$")
STEP_FILE_RE = re.compile(r"^(?P<step>\d+)\.jsonl$")


def _norm_move(m: Any) -> str:
    return str(m or "").strip().lower()


def _safe_bool(x: Any) -> bool:
    return bool(x) if isinstance(x, bool) else bool(x)


def _parse_fen_from_prompt(prompt_text: str) -> str:
    m = FEN_RE.search(prompt_text or "")
    if not m:
        raise ValueError("Could not parse FEN from prompt text.")
    return m.group("fen").strip()


def _parse_allowed_moves_from_prompt(prompt_text: str) -> list[str]:
    m = ALLOWED_MOVES_RE.search(prompt_text or "")
    if not m:
        raise ValueError("Could not parse allowed moves from prompt text.")
    raw = (m.group("moves") or "").strip()
    if not raw:
        return []
    out: list[str] = []
    for tok in raw.split(","):
        t = _norm_move(tok)
        if t:
            out.append(t)
    return out


@dataclass
class MoveCheckResult:
    is_legal: bool
    is_check: bool
    is_checkmate: bool


class CheckComputer:
    def __init__(self) -> None:
        self._board_cache: dict[str, chess.Board] = {}
        self._move_cache: dict[tuple[str, str], MoveCheckResult] = {}

    def _board_for_fen(self, fen: str) -> chess.Board:
        b = self._board_cache.get(fen)
        if b is None:
            b = chess.Board(fen)
            self._board_cache[fen] = b
        return b

    def check(self, fen: str, uci: str) -> MoveCheckResult:
        fen_s = str(fen or "").strip()
        uci_s = _norm_move(uci)
        key = (fen_s, uci_s)
        cached = self._move_cache.get(key)
        if cached is not None:
            return cached

        board = self._board_for_fen(fen_s)
        try:
            move = chess.Move.from_uci(uci_s)
        except Exception:
            out = MoveCheckResult(is_legal=False, is_check=False, is_checkmate=False)
            self._move_cache[key] = out
            return out

        if not board.is_legal(move):
            out = MoveCheckResult(is_legal=False, is_check=False, is_checkmate=False)
            self._move_cache[key] = out
            return out

        board.push(move)
        try:
            is_check = board.is_check()
            is_mate = board.is_checkmate()
        finally:
            board.pop()

        out = MoveCheckResult(is_legal=True, is_check=bool(is_check), is_checkmate=bool(is_mate))
        self._move_cache[key] = out
        return out


@dataclass
class StepAgg:
    step: int
    round_idx: Optional[int] = None

    n_records: int = 0
    n_pred_present: int = 0
    n_penalty: int = 0
    n_valid: int = 0

    # Predicted-move checks/mates.
    pred_valid_check: int = 0
    pred_valid_mate: int = 0

    pred_any_check: int = 0  # among pred_present (even if penalty applied)
    pred_any_mate: int = 0
    pred_any_illegal: int = 0

    # Correctness (dataset-label exact match) among valid predictions.
    n_exact_match: int = 0
    pred_valid_check_exact: int = 0
    pred_valid_mate_exact: int = 0
    n_non_exact: int = 0
    pred_valid_check_non_exact: int = 0

    # Ground-truth checks/mates for the same sampled positions.
    gt_check: int = 0
    gt_mate: int = 0
    gt_illegal: int = 0

    # Conditional accuracy buckets by whether the *ground-truth* move is a check.
    # (Useful for testing: "Is the run getting better mostly on check tactics?")
    n_gt_check_pos: int = 0
    n_gt_noncheck_pos: int = 0
    n_valid_gt_check: int = 0
    n_valid_gt_noncheck: int = 0
    n_exact_gt_check: int = 0
    n_exact_gt_noncheck: int = 0

    # Predicted check/mate counts conditioned on whether the GT move is a check.
    pred_valid_check_gt_check: int = 0
    pred_valid_check_gt_noncheck: int = 0
    pred_valid_mate_gt_check: int = 0
    pred_valid_mate_gt_noncheck: int = 0

    # Optional baseline heuristic using the prompt's allowed-moves list.
    # Intended to test: "Could 'just output any check' explain the run's gains?"
    # Computed only for validation logs (where enabled).
    n_allowed_parsed: int = 0
    n_allowed_parse_fail: int = 0
    n_allowed_empty: int = 0
    n_allowed_gt_included: int = 0
    n_allowed_has_check: int = 0
    n_allowed_has_mate: int = 0
    n_allowed_has_distractor_check: int = 0
    n_allowed_has_distractor_mate: int = 0

    allowed_n_legal_sum: int = 0
    allowed_n_check_sum: int = 0
    allowed_n_mate_sum: int = 0

    baseline_n: int = 0  # baseline+GT both legal
    baseline_exact: int = 0
    baseline_check: int = 0
    baseline_mate: int = 0
    baseline_n_distractor_check: int = 0
    baseline_exact_distractor_check: int = 0
    baseline_n_distractor_check_gt_check: int = 0
    baseline_n_distractor_check_gt_noncheck: int = 0
    baseline_exact_distractor_check_gt_check: int = 0
    baseline_exact_distractor_check_gt_noncheck: int = 0
    baseline_n_gt_check: int = 0
    baseline_n_gt_noncheck: int = 0
    baseline_exact_gt_check: int = 0
    baseline_exact_gt_noncheck: int = 0
    baseline_check_gt_noncheck: int = 0

    # Model stats restricted to “distractor-check-present” positions.
    n_valid_distractor_check: int = 0
    n_exact_distractor_check: int = 0
    pred_valid_check_distractor_check: int = 0
    n_valid_distractor_check_gt_check: int = 0
    n_exact_distractor_check_gt_check: int = 0
    pred_valid_check_distractor_check_gt_check: int = 0
    n_valid_distractor_check_gt_noncheck: int = 0
    n_exact_distractor_check_gt_noncheck: int = 0
    pred_valid_check_distractor_check_gt_noncheck: int = 0

    # Prompt parsing failures.
    n_missing_input: int = 0
    n_bad_fen: int = 0

    def add_record(self, rec: dict[str, Any], *, checker: CheckComputer, compute_baseline: bool = False) -> None:
        self.n_records += 1

        prompt = rec.get("input")
        if not isinstance(prompt, str) or not prompt.strip():
            self.n_missing_input += 1
            return

        try:
            fen = _parse_fen_from_prompt(prompt)
        except Exception:
            self.n_bad_fen += 1
            return

        pred_move = _norm_move(rec.get("pred_move"))
        gt_move = _norm_move(rec.get("gt_uci") or rec.get("gts"))
        penalty_applied = _safe_bool(rec.get("penalty_applied", False))

        has_distractor_check = False
        has_distractor_mate = False

        gt_res: Optional[MoveCheckResult] = None
        if gt_move:
            gt_res = checker.check(fen, gt_move)
            if not gt_res.is_legal:
                self.gt_illegal += 1
            else:
                if gt_res.is_check:
                    self.gt_check += 1
                if gt_res.is_checkmate:
                    self.gt_mate += 1
                if gt_res.is_check:
                    self.n_gt_check_pos += 1
                else:
                    self.n_gt_noncheck_pos += 1

        pred_res: Optional[MoveCheckResult] = None
        if pred_move:
            self.n_pred_present += 1
            pred_res = checker.check(fen, pred_move)
            if not pred_res.is_legal:
                self.pred_any_illegal += 1
            else:
                if pred_res.is_check:
                    self.pred_any_check += 1
                if pred_res.is_checkmate:
                    self.pred_any_mate += 1

        if compute_baseline:
            try:
                allowed_moves = _parse_allowed_moves_from_prompt(prompt)
                self.n_allowed_parsed += 1
            except Exception:
                allowed_moves = []
                self.n_allowed_parse_fail += 1

            if not allowed_moves:
                self.n_allowed_empty += 1
            elif gt_res is not None and gt_res.is_legal:
                if gt_move and gt_move in allowed_moves:
                    self.n_allowed_gt_included += 1

                has_check = False
                has_mate = False
                first_legal: Optional[tuple[str, MoveCheckResult]] = None
                first_check: Optional[tuple[str, MoveCheckResult]] = None
                allowed_n_legal = 0
                allowed_n_check = 0
                allowed_n_mate = 0

                for mv in allowed_moves:
                    res = checker.check(fen, mv)
                    if not res.is_legal:
                        continue
                    allowed_n_legal += 1
                    if first_legal is None:
                        first_legal = (mv, res)
                    if res.is_check:
                        has_check = True
                        allowed_n_check += 1
                        if first_check is None:
                            first_check = (mv, res)
                        if gt_move and mv != gt_move:
                            has_distractor_check = True
                    if res.is_checkmate:
                        has_mate = True
                        allowed_n_mate += 1
                        if gt_move and mv != gt_move:
                            has_distractor_mate = True

                if has_check:
                    self.n_allowed_has_check += 1
                if has_mate:
                    self.n_allowed_has_mate += 1
                if has_distractor_check:
                    self.n_allowed_has_distractor_check += 1
                if has_distractor_mate:
                    self.n_allowed_has_distractor_mate += 1

                self.allowed_n_legal_sum += allowed_n_legal
                self.allowed_n_check_sum += allowed_n_check
                self.allowed_n_mate_sum += allowed_n_mate

                baseline_pick = first_check or first_legal
                if baseline_pick is not None:
                    baseline_move, baseline_res = baseline_pick
                    self.baseline_n += 1
                    if baseline_res.is_check:
                        self.baseline_check += 1
                    if baseline_res.is_checkmate:
                        self.baseline_mate += 1
                    if baseline_move == gt_move and gt_move:
                        self.baseline_exact += 1

                    if has_distractor_check:
                        self.baseline_n_distractor_check += 1
                        if baseline_move == gt_move and gt_move:
                            self.baseline_exact_distractor_check += 1
                        if gt_res.is_check:
                            self.baseline_n_distractor_check_gt_check += 1
                            if baseline_move == gt_move and gt_move:
                                self.baseline_exact_distractor_check_gt_check += 1
                        else:
                            self.baseline_n_distractor_check_gt_noncheck += 1
                            if baseline_move == gt_move and gt_move:
                                self.baseline_exact_distractor_check_gt_noncheck += 1

                    if gt_res.is_check:
                        self.baseline_n_gt_check += 1
                        if baseline_move == gt_move and gt_move:
                            self.baseline_exact_gt_check += 1
                    else:
                        self.baseline_n_gt_noncheck += 1
                        if baseline_move == gt_move and gt_move:
                            self.baseline_exact_gt_noncheck += 1
                        if baseline_res.is_check:
                            self.baseline_check_gt_noncheck += 1

        if penalty_applied:
            self.n_penalty += 1
            return

        # "Valid" here means "not penalized by the reward function".
        if pred_move:
            self.n_valid += 1
            if pred_res is None:
                pred_res = checker.check(fen, pred_move)
            if pred_res.is_legal:
                if pred_res.is_check:
                    self.pred_valid_check += 1
                if pred_res.is_checkmate:
                    self.pred_valid_mate += 1

                if gt_res is not None and gt_res.is_legal:
                    if gt_res.is_check:
                        if pred_res.is_check:
                            self.pred_valid_check_gt_check += 1
                        if pred_res.is_checkmate:
                            self.pred_valid_mate_gt_check += 1
                    else:
                        if pred_res.is_check:
                            self.pred_valid_check_gt_noncheck += 1
                        if pred_res.is_checkmate:
                            self.pred_valid_mate_gt_noncheck += 1

            # Exact-match tracking (only meaningful when pred_move is present and valid).
            exact_raw = rec.get("exact_match")
            is_exact = False
            try:
                is_exact = float(exact_raw) >= 0.999
            except Exception:
                is_exact = False

            # Extra deep-dive bucket: positions where allowed_moves contains a *checking* move
            # that is NOT the ground-truth (i.e., a “distractor check” is available).
            #
            # - For GT-noncheck positions, this is basically "any check exists in allowed_moves".
            # - For GT-check positions, this means there are multiple checks and the model must
            #   choose the right one.
            if compute_baseline and has_distractor_check and gt_res is not None and gt_res.is_legal:
                self.n_valid_distractor_check += 1
                if pred_res is not None and pred_res.is_legal and pred_res.is_check:
                    self.pred_valid_check_distractor_check += 1
                if is_exact:
                    self.n_exact_distractor_check += 1

                if gt_res.is_check:
                    self.n_valid_distractor_check_gt_check += 1
                    if pred_res is not None and pred_res.is_legal and pred_res.is_check:
                        self.pred_valid_check_distractor_check_gt_check += 1
                    if is_exact:
                        self.n_exact_distractor_check_gt_check += 1
                else:
                    self.n_valid_distractor_check_gt_noncheck += 1
                    if pred_res is not None and pred_res.is_legal and pred_res.is_check:
                        self.pred_valid_check_distractor_check_gt_noncheck += 1
                    if is_exact:
                        self.n_exact_distractor_check_gt_noncheck += 1
            if is_exact:
                self.n_exact_match += 1
                if pred_res.is_legal and pred_res.is_check:
                    self.pred_valid_check_exact += 1
                if pred_res.is_legal and pred_res.is_checkmate:
                    self.pred_valid_mate_exact += 1

                # Conditional exact-match by GT check-ness (only when GT is legal and known).
                if gt_res is not None and gt_res.is_legal:
                    if gt_res.is_check:
                        self.n_valid_gt_check += 1
                        self.n_exact_gt_check += 1
                    else:
                        self.n_valid_gt_noncheck += 1
                        self.n_exact_gt_noncheck += 1
            else:
                self.n_non_exact += 1
                if pred_res.is_legal and pred_res.is_check:
                    self.pred_valid_check_non_exact += 1

                if gt_res is not None and gt_res.is_legal:
                    if gt_res.is_check:
                        self.n_valid_gt_check += 1
                    else:
                        self.n_valid_gt_noncheck += 1

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)

        def frac(n: int, d: int) -> float:
            if d <= 0:
                return float("nan")
            return float(n) / float(d)

        row["pred_valid_check_frac"] = frac(self.pred_valid_check, self.n_valid)
        row["pred_valid_mate_frac"] = frac(self.pred_valid_mate, self.n_valid)
        row["pred_any_check_frac"] = frac(self.pred_any_check, self.n_pred_present)
        row["pred_any_mate_frac"] = frac(self.pred_any_mate, self.n_pred_present)
        row["exact_match_frac"] = frac(self.n_exact_match, self.n_valid)
        row["pred_valid_check_frac_exact"] = frac(self.pred_valid_check_exact, self.n_exact_match)
        row["pred_valid_mate_frac_exact"] = frac(self.pred_valid_mate_exact, self.n_exact_match)
        row["pred_valid_check_frac_non_exact"] = frac(self.pred_valid_check_non_exact, self.n_non_exact)
        row["exact_match_frac_gt_check"] = frac(self.n_exact_gt_check, self.n_valid_gt_check)
        row["exact_match_frac_gt_noncheck"] = frac(self.n_exact_gt_noncheck, self.n_valid_gt_noncheck)
        row["pred_valid_check_frac_gt_check"] = frac(self.pred_valid_check_gt_check, self.n_valid_gt_check)
        row["pred_valid_check_frac_gt_noncheck"] = frac(self.pred_valid_check_gt_noncheck, self.n_valid_gt_noncheck)
        row["pred_valid_mate_frac_gt_check"] = frac(self.pred_valid_mate_gt_check, self.n_valid_gt_check)
        row["pred_valid_mate_frac_gt_noncheck"] = frac(self.pred_valid_mate_gt_noncheck, self.n_valid_gt_noncheck)

        row["allowed_gt_included_frac"] = frac(self.n_allowed_gt_included, self.n_allowed_parsed)
        row["allowed_has_check_frac"] = frac(self.n_allowed_has_check, self.n_allowed_parsed)
        row["allowed_has_mate_frac"] = frac(self.n_allowed_has_mate, self.n_allowed_parsed)
        row["allowed_has_distractor_check_frac"] = frac(self.n_allowed_has_distractor_check, self.n_allowed_parsed)
        row["allowed_has_distractor_mate_frac"] = frac(self.n_allowed_has_distractor_mate, self.n_allowed_parsed)
        row["allowed_avg_legal_moves"] = frac(self.allowed_n_legal_sum, self.n_allowed_parsed)
        row["allowed_avg_check_moves"] = frac(self.allowed_n_check_sum, self.n_allowed_parsed)
        row["allowed_avg_mate_moves"] = frac(self.allowed_n_mate_sum, self.n_allowed_parsed)

        row["baseline_exact_match_frac"] = frac(self.baseline_exact, self.baseline_n)
        row["baseline_check_frac"] = frac(self.baseline_check, self.baseline_n)
        row["baseline_mate_frac"] = frac(self.baseline_mate, self.baseline_n)
        row["baseline_exact_match_frac_gt_check"] = frac(self.baseline_exact_gt_check, self.baseline_n_gt_check)
        row["baseline_exact_match_frac_gt_noncheck"] = frac(self.baseline_exact_gt_noncheck, self.baseline_n_gt_noncheck)
        row["baseline_check_frac_gt_noncheck"] = frac(self.baseline_check_gt_noncheck, self.baseline_n_gt_noncheck)
        row["baseline_exact_match_frac_distractor_check"] = frac(self.baseline_exact_distractor_check, self.baseline_n_distractor_check)
        row["baseline_exact_match_frac_distractor_check_gt_check"] = frac(
            self.baseline_exact_distractor_check_gt_check, self.baseline_n_distractor_check_gt_check
        )
        row["baseline_exact_match_frac_distractor_check_gt_noncheck"] = frac(
            self.baseline_exact_distractor_check_gt_noncheck, self.baseline_n_distractor_check_gt_noncheck
        )

        row["exact_match_frac_distractor_check"] = frac(self.n_exact_distractor_check, self.n_valid_distractor_check)
        row["pred_valid_check_frac_distractor_check"] = frac(self.pred_valid_check_distractor_check, self.n_valid_distractor_check)
        row["exact_match_frac_distractor_check_gt_check"] = frac(
            self.n_exact_distractor_check_gt_check, self.n_valid_distractor_check_gt_check
        )
        row["pred_valid_check_frac_distractor_check_gt_check"] = frac(
            self.pred_valid_check_distractor_check_gt_check, self.n_valid_distractor_check_gt_check
        )
        row["exact_match_frac_distractor_check_gt_noncheck"] = frac(
            self.n_exact_distractor_check_gt_noncheck, self.n_valid_distractor_check_gt_noncheck
        )
        row["pred_valid_check_frac_distractor_check_gt_noncheck"] = frac(
            self.pred_valid_check_distractor_check_gt_noncheck, self.n_valid_distractor_check_gt_noncheck
        )
        row["penalty_frac"] = frac(self.n_penalty, self.n_records)
        row["pred_present_frac"] = frac(self.n_pred_present, self.n_records)
        row["gt_check_frac"] = frac(self.gt_check, self.n_records)
        row["gt_mate_frac"] = frac(self.gt_mate, self.n_records)
        return row


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _load_history_metrics(history_parquet: Path) -> pd.DataFrame:
    if not history_parquet.exists():
        return pd.DataFrame()
    df = pd.read_parquet(history_parquet)
    # Normalize step column.
    if "_step" in df.columns:
        df = df.rename(columns={"_step": "step"})
    if "step" not in df.columns:
        return pd.DataFrame()
    keep = [c for c in ["step", "grpo/effective_batch_frac", "grpo/effective_batch_size", "selection_sampler/r_max"] if c in df.columns]
    return df[keep].copy()


def compute_train_allowed_move_elim_rates(evidence_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    logs_dir = evidence_dir / "files" / "allowed_move_elim_rounds"
    if not logs_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    checker = CheckComputer()
    per_round: dict[tuple[int, int], StepAgg] = {}

    for p in sorted(logs_dir.glob("*.jsonl")):
        m = ROUND_FILE_RE.match(p.name)
        if not m:
            continue
        step = int(m.group("step"))
        round_idx = int(m.group("round"))
        agg = per_round.get((step, round_idx))
        if agg is None:
            agg = StepAgg(step=step, round_idx=round_idx)
            per_round[(step, round_idx)] = agg
        for rec in _iter_jsonl(p):
            agg.add_record(rec, checker=checker)

    round_rows = [agg.to_row() for _, agg in sorted(per_round.items(), key=lambda kv: kv[0])]
    df_round = pd.DataFrame(round_rows).sort_values(["step", "round_idx"]).reset_index(drop=True)

    # Aggregate across rounds per step.
    per_step: dict[int, StepAgg] = {}
    for (_, _), agg in per_round.items():
        step = int(agg.step)
        out = per_step.get(step)
        if out is None:
            out = StepAgg(step=step, round_idx=None)
            per_step[step] = out
        # Manually sum numeric counters (keeping semantics consistent).
        for field in [
            "n_records",
            "n_pred_present",
            "n_penalty",
            "n_valid",
            "pred_valid_check",
            "pred_valid_mate",
            "pred_any_check",
            "pred_any_mate",
            "pred_any_illegal",
            "n_exact_match",
            "pred_valid_check_exact",
            "pred_valid_mate_exact",
            "n_non_exact",
            "pred_valid_check_non_exact",
            "gt_check",
            "gt_mate",
            "gt_illegal",
            "n_gt_check_pos",
            "n_gt_noncheck_pos",
            "n_valid_gt_check",
            "n_valid_gt_noncheck",
            "n_exact_gt_check",
            "n_exact_gt_noncheck",
            "pred_valid_check_gt_check",
            "pred_valid_check_gt_noncheck",
            "pred_valid_mate_gt_check",
            "pred_valid_mate_gt_noncheck",
            "n_allowed_parsed",
            "n_allowed_parse_fail",
            "n_allowed_empty",
            "n_allowed_has_check",
            "n_allowed_has_mate",
            "baseline_n",
            "baseline_exact",
            "baseline_check",
            "baseline_mate",
            "baseline_n_gt_check",
            "baseline_n_gt_noncheck",
            "baseline_exact_gt_check",
            "baseline_exact_gt_noncheck",
            "baseline_check_gt_noncheck",
            "n_missing_input",
            "n_bad_fen",
        ]:
            setattr(out, field, int(getattr(out, field)) + int(getattr(agg, field)))

    step_rows = [agg.to_row() for _, agg in sorted(per_step.items(), key=lambda kv: kv[0])]
    df_step = pd.DataFrame(step_rows).sort_values(["step"]).reset_index(drop=True)
    return df_step, df_round


def compute_validation_rates(evidence_dir: Path) -> pd.DataFrame:
    logs_dir = evidence_dir / "files" / "validation_logs"
    if not logs_dir.exists():
        return pd.DataFrame()

    checker = CheckComputer()
    per_step: dict[int, StepAgg] = {}

    for p in sorted(logs_dir.glob("*.jsonl")):
        m = STEP_FILE_RE.match(p.name)
        if not m:
            continue
        step = int(m.group("step"))
        agg = per_step.get(step)
        if agg is None:
            agg = StepAgg(step=step, round_idx=None)
            per_step[step] = agg
        for rec in _iter_jsonl(p):
            agg.add_record(rec, checker=checker, compute_baseline=True)

    rows = [agg.to_row() for _, agg in sorted(per_step.items(), key=lambda kv: kv[0])]
    return pd.DataFrame(rows).sort_values(["step"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_evidence_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    evidence_dir = Path(args.wandb_evidence_dir)
    if not evidence_dir.exists():
        raise SystemExit(f"Missing evidence dir: {evidence_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_step, train_round = compute_train_allowed_move_elim_rates(evidence_dir)
    val_step = compute_validation_rates(evidence_dir)
    history = _load_history_metrics(evidence_dir / "history.parquet")

    if not train_step.empty and not history.empty:
        train_step = train_step.merge(history, on="step", how="left")
    if not val_step.empty and not history.empty:
        val_step = val_step.merge(history, on="step", how="left")

    if not train_step.empty:
        train_step.to_csv(out_dir / "train_step_check_rates.csv", index=False)
    if not train_round.empty:
        train_round.to_csv(out_dir / "train_round_check_rates.csv", index=False)
    if not val_step.empty:
        val_step.to_csv(out_dir / "val_step_check_rates.csv", index=False)

    summary: Dict[str, Any] = {
        "wandb_evidence_dir": str(evidence_dir),
        "out_dir": str(out_dir),
        "train_steps": int(len(train_step)) if train_step is not None else 0,
        "val_steps": int(len(val_step)) if val_step is not None else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Small stdout summary.
    def _print_head_tail(df: pd.DataFrame, name: str) -> None:
        if df.empty:
            print(f"[{name}] empty")
            return
        cols = [
            "step",
            "n_valid",
            "pred_valid_check_frac",
            "pred_valid_mate_frac",
            "gt_check_frac",
            "pred_valid_check_frac_gt_noncheck",
            "baseline_exact_match_frac",
            "baseline_check_frac_gt_noncheck",
            "grpo/effective_batch_frac",
            "selection_sampler/r_max",
        ]
        cols = [c for c in cols if c in df.columns]
        print(f"[{name}] rows={len(df)} cols={len(df.columns)}")
        print("  head:")
        print(df[cols].head(5).to_string(index=False))
        print("  tail:")
        print(df[cols].tail(5).to_string(index=False))

    _print_head_tail(train_step, "TRAIN_STEP")
    _print_head_tail(val_step, "VAL_STEP")
    print(f"[WRITE] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
