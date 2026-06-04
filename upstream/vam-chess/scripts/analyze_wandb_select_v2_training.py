#!/usr/bin/env python3
"""
Analyze a restricted-moves ("selection") training run export with v2-derived variants.

This script:
  - Joins train rollouts to the v2 selection dataset to recover `extra_info.derived_variant`.
  - Recomputes reward metadata via recipe/chess/reward_fn.py::compute_score.
  - Aggregates per-(step, derived_variant) metrics (score, acc, compliance, candidate size).
  - Computes position-bias metrics vs candidate-list size (abs rank + ratio).
  - Computes validation position-bias metrics per step and validation dataset.

Outputs under:
  <evidence_root>/investigation/
    - run_config_summary.md
    - train_rollouts_by_step_variant.csv
    - train_position_bias_by_n_considered.csv
    - train_position_bias_by_n_considered_variant.csv
    - train_position_bias_by_n_considered_step_window.csv
    - validation_position_bias_by_step.csv
    - plots/*.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# Ensure local imports resolve when the script is run directly.
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipe.chess.reward_fn import compute_score


_FEN_RE = re.compile(r"Current FEN string:\s*(?P<fen>[^\r\n]+)", flags=re.IGNORECASE)
_LEGAL_RE = re.compile(r"Legal moves \(UCI\):\s*(?P<moves>[^\r\n]+)", flags=re.IGNORECASE)
_ALLOWED_RE = re.compile(r"Allowed moves \(UCI\):\s*(?P<moves>[^\r\n]+)", flags=re.IGNORECASE)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def _parse_moves_csv(moves_csv: str) -> List[str]:
    return [m.strip().lower() for m in (moves_csv or "").split(",") if m.strip()]


@dataclass(frozen=True)
class PromptFields:
    fen: str
    legal_moves: List[str]
    allowed_moves: List[str]


@dataclass(frozen=True)
class SelectRowInfo:
    reward_model: Dict[str, Any]
    derived_variant: str
    derived_sub_id: int
    source_index: int
    derived_index: int


def _parse_prompt_fields(prompt_text: str) -> PromptFields:
    fen_m = _FEN_RE.search(prompt_text or "")
    legal_m = _LEGAL_RE.search(prompt_text or "")
    allowed_m = _ALLOWED_RE.search(prompt_text or "")
    if not fen_m:
        raise ValueError("Missing 'Current FEN string:' in prompt.")
    if not legal_m:
        raise ValueError("Missing 'Legal moves (UCI):' in prompt.")
    legal_moves = _parse_moves_csv(legal_m.group("moves"))
    if allowed_m:
        allowed_moves = _parse_moves_csv(allowed_m.group("moves"))
    else:
        # Baseline prompts only list legal moves; treat allowed=legal.
        allowed_moves = list(legal_moves)
    return PromptFields(
        fen=fen_m.group("fen").strip(),
        legal_moves=legal_moves,
        allowed_moves=allowed_moves,
    )


def _resolve_dataset_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p
    if str(p).startswith("/workspace/chess_rl/"):
        alt = Path(str(p).replace("/workspace/chess_rl/", "", 1))
        if alt.exists():
            return alt
    # Try relative to repo root.
    alt = ROOT / path_str
    if alt.exists():
        return alt
    return p


def _load_config(evidence_root: Path) -> Dict[str, Any]:
    cfg_path = evidence_root / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _load_select_rows(
    select_parquet: Path,
) -> Tuple[
    Dict[Tuple[str, Tuple[str, ...]], List[SelectRowInfo]],
    Dict[Tuple[str, frozenset], List[SelectRowInfo]],
    int,
    int,
]:
    dataset = ds.dataset(str(select_parquet), format="parquet")
    table = dataset.to_table(columns=["reward_model", "extra_info"])
    by_key: Dict[Tuple[str, Tuple[str, ...]], List[SelectRowInfo]] = defaultdict(list)
    by_set: Dict[Tuple[str, frozenset], List[SelectRowInfo]] = defaultdict(list)
    for row in table.to_pylist():
        rm = row.get("reward_model") or {}
        ei = row.get("extra_info") or {}
        fen = str(rm.get("fen") or "").strip()
        considered = [str(m).strip().lower() for m in (rm.get("considered_moves_uci") or [])]
        considered_key = tuple(considered)
        if not fen or not considered_key:
            continue
        info = SelectRowInfo(
            reward_model=dict(rm),
            derived_variant=str(ei.get("derived_variant") or ""),
            derived_sub_id=int(ei.get("derived_sub_id") or 0),
            source_index=int(ei.get("source_index") or -1),
            derived_index=int(ei.get("index") or -1),
        )
        key = (fen, considered_key)
        by_key[key].append(info)
        by_set[(fen, frozenset(considered_key))].append(info)
    duplicate_keys = sum(1 for infos in by_key.values() if len(infos) > 1)
    multi_variant_keys = sum(1 for infos in by_key.values() if len({i.derived_variant for i in infos}) > 1)
    return by_key, by_set, duplicate_keys, multi_variant_keys


def _detect_selection_dataset(parquet_path: Path) -> bool:
    dataset = ds.dataset(str(parquet_path), format="parquet")
    table = dataset.to_table(columns=["reward_model"])
    for row in table.to_pylist()[:200]:
        rm = row.get("reward_model") or {}
        if isinstance(rm, dict) and "considered_moves_uci" in rm:
            return True
    return False


def _load_reward_models_by_fen(parquet_path: Path) -> Dict[str, Dict[str, Any]]:
    dataset = ds.dataset(str(parquet_path), format="parquet")
    table = dataset.to_table(columns=["reward_model"])
    out: Dict[str, Dict[str, Any]] = {}
    for row in table.to_pylist():
        rm = row.get("reward_model") or {}
        fen = str(rm.get("fen") or "").strip()
        if not fen:
            continue
        if fen not in out:
            out[fen] = dict(rm)
    return out


def _load_validation_lookup(
    val_files_config: Any,
) -> Tuple[Dict[Tuple[str, Tuple[str, ...]], str], int]:
    lookup: Dict[Tuple[str, Tuple[str, ...]], str] = {}
    conflicts = 0
    if not val_files_config:
        return lookup, conflicts
    paths: List[str]
    if isinstance(val_files_config, list):
        paths = [str(p) for p in val_files_config]
    else:
        paths = [str(val_files_config)]
    for path_str in paths:
        path = _resolve_dataset_path(path_str)
        if path is None or not path.exists():
            print(f"[WARN] Validation dataset not found for lookup: {path_str}")
            continue
        label = Path(path_str).stem
        dataset = ds.dataset(str(path), format="parquet")
        table = dataset.to_table(columns=["reward_model"])
        for row in table.to_pylist():
            rm = row.get("reward_model") or {}
            fen = str(rm.get("fen") or "").strip()
            if not fen:
                continue
            considered = rm.get("considered_moves_uci")
            if not considered:
                considered = rm.get("legal_moves_uci") or []
            considered_moves = [str(m).strip().lower() for m in considered if str(m).strip()]
            if not considered_moves:
                continue
            key = (fen, tuple(considered_moves))
            if key in lookup and lookup[key] != label:
                conflicts += 1
                continue
            lookup[key] = label
    return lookup, conflicts


def _parse_mu_map(reward_model: Dict[str, Any]) -> Dict[str, float]:
    mu_json = reward_model.get("move_expected_scores_json")
    if isinstance(mu_json, str) and mu_json.strip():
        try:
            mu_raw = json.loads(mu_json)
        except Exception:
            mu_raw = None
    else:
        mu_raw = mu_json
    mu_map: Dict[str, float] = {}
    if isinstance(mu_raw, dict):
        for k, v in mu_raw.items():
            key = str(k).strip().lower()
            if not key:
                continue
            try:
                mu_map[key] = float(v)
            except Exception:
                continue
    if mu_map:
        return mu_map
    mv_json = reward_model.get("move_values_json")
    if isinstance(mv_json, str) and mv_json.strip():
        try:
            mv_raw = json.loads(mv_json)
        except Exception:
            mv_raw = None
    else:
        mv_raw = mv_json
    if isinstance(mv_raw, dict):
        for k, v in mv_raw.items():
            key = str(k).strip().lower()
            if not key:
                continue
            try:
                mu_map[key] = float(v)
            except Exception:
                continue
    return mu_map


def _best_move_by_mu(mu_map: Dict[str, float], moves: List[str]) -> str:
    best_move = ""
    best_mu = -float("inf")
    for mv in moves:
        key = str(mv).strip().lower()
        mu = float(mu_map.get(key, -float("inf")))
        if (mu > best_mu) or (mu == best_mu and (not best_move or key < best_move)):
            best_move = key
            best_mu = mu
    return best_move


def _compute_target_move(reward_model: Dict[str, Any], considered_moves: List[str]) -> str:
    mu_map = _parse_mu_map(reward_model)
    if not mu_map or not considered_moves:
        return ""
    return _best_move_by_mu(mu_map, considered_moves)


def _load_legal_by_fen(parquet_path: Path) -> Dict[str, List[str]]:
    dataset = ds.dataset(str(parquet_path), format="parquet")
    table = dataset.to_table(columns=["reward_model"])
    out: Dict[str, List[str]] = {}
    for row in table.to_pylist():
        rm = row.get("reward_model") or {}
        fen = str(rm.get("fen") or "").strip()
        legal = [str(m).strip().lower() for m in (rm.get("legal_moves_uci") or [])]
        if fen and legal:
            out[fen] = legal
    return out


def _write_run_config_summary(
    out_root: Path,
    *,
    train_files: Optional[str],
    val_files: List[str],
    reward_fn: str,
    filter_groups: Dict[str, Any],
    data_shuffle: Optional[bool],
    train_batch_size: Optional[int],
    gen_batch_size: Optional[int],
    rollout_n: Optional[int],
) -> None:
    lines = ["# Run config summary", ""]
    lines.append(f"- train_files: `{train_files}`")
    if val_files:
        lines.append("- val_files:")
        for vf in val_files:
            lines.append(f"  - `{vf}`")
    else:
        lines.append("- val_files: (none)")
    lines.append(f"- chess_reward_fn: `{reward_fn}`")
    if filter_groups:
        lines.append("- filter_groups:")
        lines.append(f"  - enable: `{filter_groups.get('enable')}`")
        lines.append(f"  - metric: `{filter_groups.get('metric')}`")
        lines.append(f"  - max_num_gen_batches: `{filter_groups.get('max_num_gen_batches')}`")
    else:
        lines.append("- filter_groups: (missing)")
    lines.append(f"- data.shuffle: `{data_shuffle}`")
    lines.append(f"- data.train_batch_size: `{train_batch_size}`")
    lines.append(f"- data.gen_batch_size: `{gen_batch_size}`")
    lines.append(f"- rollout.n: `{rollout_n}`")
    out_path = out_root / "run_config_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _init_bias_agg() -> Dict[str, float]:
    return {"n": 0, "sum_rank_abs": 0.0, "sum_rank_ratio": 0.0}


def _init_rollout_agg() -> Dict[str, Any]:
    return {
        "n": 0,
        "sum_logged_score": 0.0,
        "sum_recomputed_score": 0.0,
        "sum_acc": 0.0,
        "sum_format_ok": 0.0,
        "sum_in_subset": 0.0,
        "sum_penalty": 0.0,
        "sum_n_considered": 0.0,
        "min_n_considered": float("inf"),
        "max_n_considered": -float("inf"),
        "sum_pred_rank_abs": 0.0,
        "sum_pred_rank_ratio": 0.0,
        "n_pred_rank": 0,
    }


def _init_val_agg() -> Dict[str, Any]:
    return {
        "n": 0,
        "sum_pred_rank_abs": 0.0,
        "sum_pred_rank_ratio": 0.0,
        "n_pred_rank": 0,
        "sum_target_rank_abs": 0.0,
        "sum_target_rank_ratio": 0.0,
        "n_target_rank": 0,
        "sum_n_considered": 0.0,
    }


def _rank_in_list(moves: List[str], move: str) -> Optional[int]:
    if not move:
        return None
    try:
        idx = moves.index(move)
    except ValueError:
        return None
    return idx + 1


def _ratio_from_rank(rank_abs: int, n_considered: int) -> float:
    if n_considered <= 1:
        return 0.0
    return float(rank_abs - 1) / float(n_considered - 1)


def _savefig(path: Path, title: str) -> None:
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=12)
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _load_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def _load_history_filter_groups(history_path: Path) -> pd.DataFrame:
    rows = []
    if not history_path.exists():
        return pd.DataFrame()
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "filter_groups/kept_groups_total" not in rec and "filter_groups/rejected_groups_total" not in rec:
                continue
            step = rec.get("training/global_step", rec.get("_step"))
            if step is None:
                continue
            kept = float(rec.get("filter_groups/kept_groups_total", 0.0) or 0.0)
            rejected = float(rec.get("filter_groups/rejected_groups_total", 0.0) or 0.0)
            rows.append(
                {
                    "step": int(step),
                    "kept_groups": kept,
                    "rejected_groups": rejected,
                    "rejected_all_valid": float(rec.get("filter_groups/rejected_by_penalty/all_valid", 0.0) or 0.0),
                    "rejected_mixed": float(rec.get("filter_groups/rejected_by_penalty/mixed", 0.0) or 0.0),
                    "rejected_out_of_subset": float(
                        rec.get("filter_groups/rejected_by_penalty/out_of_subset", 0.0) or 0.0
                    ),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("step").drop_duplicates(subset=["step"], keep="last")
    total = df["kept_groups"] + df["rejected_groups"]
    df["total_groups"] = total
    df["kept_frac"] = df["kept_groups"] / total.replace(0, np.nan)
    df["rejected_frac"] = df["rejected_groups"] / total.replace(0, np.nan)
    rejected = df["rejected_groups"].replace(0, np.nan)
    df["rejected_all_valid_frac"] = df["rejected_all_valid"] / rejected
    df["rejected_mixed_frac"] = df["rejected_mixed"] / rejected
    df["rejected_out_of_subset_frac"] = df["rejected_out_of_subset"] / rejected
    return df


def _smooth_values(values: List[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=float)
    return pd.Series(values).rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def _is_ambiguous_variant(variant: str) -> bool:
    return variant.startswith("ambiguous")


def _weighted_slope(xs: List[float], ys: List[float], ws: List[float]) -> float:
    wsum = float(sum(ws))
    if wsum <= 0.0:
        return float("nan")
    mean_x = sum(w * x for w, x in zip(ws, xs)) / wsum
    mean_y = sum(w * y for w, y in zip(ws, ys)) / wsum
    cov = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(ws, xs, ys))
    var_x = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    if var_x <= 0.0:
        return float("nan")
    return cov / var_x


def _markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    def _fmt(val: Any) -> str:
        if isinstance(val, float):
            if math.isfinite(val):
                return f"{val:.6g}"
            return "nan"
        return str(val)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-root", type=Path, required=True)
    ap.add_argument("--train-dataset", type=str, default=None)
    ap.add_argument("--reward-fn", type=str, default=None)
    ap.add_argument("--limit-rollout-files", type=int, default=None)
    ap.add_argument("--limit-validation-files", type=int, default=None)
    ap.add_argument("--ratio-bins", type=int, default=10)
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--compare-evidence-root", type=Path, default=None)
    ap.add_argument("--compare-label", type=str, default="baseline")
    args = ap.parse_args()

    evidence_root: Path = args.evidence_root
    files_root = evidence_root / "files"
    rollout_dir = files_root / "rollout_logs"
    val_dir = files_root / "validation_logs"
    if not rollout_dir.exists():
        raise FileNotFoundError(f"Missing rollout logs: {rollout_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Missing validation logs: {val_dir}")

    out_root = evidence_root / "investigation"
    plots_dir = out_root / "plots"
    _ensure_dir(out_root)
    _ensure_dir(plots_dir)

    cfg = _load_config(evidence_root)
    data_cfg = cfg.get("data") or {}
    alg_cfg = cfg.get("algorithm") or {}
    custom_reward = cfg.get("custom_reward_function") or {}

    train_files = args.train_dataset or data_cfg.get("train_files")
    reward_fn = args.reward_fn or (custom_reward.get("reward_kwargs") or {}).get("chess_reward_fn") or "selection"
    val_files_config = data_cfg.get("val_files") or []

    train_path = _resolve_dataset_path(train_files)
    if train_path is None or not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_files}")

    is_selection_dataset = _detect_selection_dataset(train_path)
    select_rows_by_key: Dict[Tuple[str, Tuple[str, ...]], List[SelectRowInfo]] = {}
    select_rows_by_set: Dict[Tuple[str, frozenset], List[SelectRowInfo]] = {}
    fen2rm: Dict[str, Dict[str, Any]] = {}
    if is_selection_dataset:
        select_rows_by_key, select_rows_by_set, duplicate_keys, multivar_keys = _load_select_rows(train_path)
        print(f"[OK] Loaded select rows: {len(select_rows_by_key)} from {train_path}")
        if duplicate_keys:
            print(f"[WARN] duplicate (fen, considered_moves) keys in dataset: {duplicate_keys}")
        if multivar_keys:
            print(f"[WARN] duplicate keys with multiple derived_variant labels: {multivar_keys}")
    else:
        fen2rm = _load_reward_models_by_fen(train_path)
        print(f"[OK] Loaded base reward_model rows: {len(fen2rm)} from {train_path}")

    # Validation dataset reference (canonical test legal-move order).
    canonical_test_path = _resolve_dataset_path("data/chess_puzzles_select_v2/test.parquet")
    legal_by_fen: Dict[str, List[str]] = {}
    rm_by_fen: Dict[str, Dict[str, Any]] = {}
    if canonical_test_path and canonical_test_path.exists():
        legal_by_fen = _load_legal_by_fen(canonical_test_path)
        rm_by_fen = _load_reward_models_by_fen(canonical_test_path)
        print(f"[OK] Loaded canonical test legal moves: {len(legal_by_fen)} rows")
    else:
        print("[WARN] Could not load canonical test dataset; validation dataset labels may be unknown.")

    val_lookup, val_lookup_conflicts = _load_validation_lookup(val_files_config)
    if val_lookup:
        print(f"[OK] Loaded validation lookup: {len(val_lookup)} entries")
    if val_lookup_conflicts:
        print(f"[WARN] validation lookup conflicts: {val_lookup_conflicts}")

    filter_groups_cfg = alg_cfg.get("filter_groups") or {}
    _write_run_config_summary(
        out_root,
        train_files=train_files,
        val_files=list(val_files_config) if isinstance(val_files_config, list) else [str(val_files_config)],
        reward_fn=str(reward_fn),
        filter_groups=dict(filter_groups_cfg) if isinstance(filter_groups_cfg, dict) else {},
        data_shuffle=data_cfg.get("shuffle"),
        train_batch_size=data_cfg.get("train_batch_size"),
        gen_batch_size=data_cfg.get("gen_batch_size"),
        rollout_n=((cfg.get("actor_rollout_ref") or {}).get("rollout") or {}).get("n"),
    )

    # Determine step windows for bias stratification.
    rollout_files = sorted([p for p in rollout_dir.glob("*.jsonl") if p.is_file()], key=lambda p: int(p.stem))
    if args.limit_rollout_files is not None:
        rollout_files = rollout_files[: int(args.limit_rollout_files)]
    steps = [int(p.stem) for p in rollout_files]
    steps_sorted = sorted(steps)
    if steps_sorted:
        q1_idx = max(0, int(len(steps_sorted) * 0.33) - 1)
        q2_idx = max(0, int(len(steps_sorted) * 0.66) - 1)
        q1 = steps_sorted[q1_idx]
        q2 = steps_sorted[q2_idx]
    else:
        q1 = q2 = 0

    def _step_window(step: int) -> str:
        if step <= q1:
            return "early"
        if step <= q2:
            return "mid"
        return "late"

    # Aggregators
    per_step_variant: Dict[Tuple[int, str], Dict[str, Any]] = defaultdict(_init_rollout_agg)
    per_step_total: Dict[int, int] = defaultdict(int)
    bias_by_n: Dict[int, Dict[str, float]] = defaultdict(_init_bias_agg)
    bias_by_n_variant: Dict[Tuple[int, str], Dict[str, float]] = defaultdict(_init_bias_agg)
    bias_by_n_window: Dict[Tuple[int, str], Dict[str, float]] = defaultdict(_init_bias_agg)
    train_dist_by_variant: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"rank_abs": [], "rank_ratio": []}
    )

    score_mismatch_count = 0
    score_mismatch_samples: List[Dict[str, Any]] = []
    max_score_diff = 0.0
    join_miss_count = 0
    join_set_fallback = 0
    ambiguous_variant_count = 0
    legal_mismatch_count = 0
    considered_mismatch_count = 0
    train_rank_abs_all: List[int] = []
    train_rank_ratio_all: List[float] = []
    train_n_considered_all: List[int] = []

    for file_path in rollout_files:
        step = int(file_path.stem)
        for rec_idx, rec in _iter_jsonl(file_path):
            prompt_text = str(rec.get("input") or "")
            output_text = str(rec.get("output") or "")
            logged_score = float(rec.get("score"))

            fields = _parse_prompt_fields(prompt_text)
            allowed_key = tuple(fields.allowed_moves)
            join_key = (fields.fen, allowed_key)
            if is_selection_dataset:
                infos = select_rows_by_key.get(join_key)
                if not infos:
                    # Fallback: match by set if unique (order should still match legal-order subsequence).
                    candidates = select_rows_by_set.get((fields.fen, frozenset(allowed_key))) or []
                    if len(candidates) == 1:
                        infos = candidates
                        join_set_fallback += 1
                    else:
                        join_miss_count += 1
                        continue

                variants = {info.derived_variant for info in infos}
                if len(variants) == 1:
                    variant = next(iter(variants))
                elif variants:
                    ambiguous_variant_count += 1
                    variant = "ambiguous_" + "+".join(sorted(v for v in variants if v))
                else:
                    ambiguous_variant_count += 1
                    variant = "ambiguous_<missing>"
                is_ambiguous = _is_ambiguous_variant(variant or "")

                row_info = infos[0]
                rm = dict(row_info.reward_model)
            else:
                rm_base = fen2rm.get(fields.fen)
                if rm_base is None:
                    join_miss_count += 1
                    continue
                variant = "full_legal"
                is_ambiguous = False
                rm = dict(rm_base)
            rm_considered = [str(m).strip().lower() for m in (rm.get("considered_moves_uci") or [])]
            if rm_considered != fields.allowed_moves:
                if is_selection_dataset:
                    considered_mismatch_count += 1
                rm["considered_moves_uci"] = list(fields.allowed_moves)

            rm_legal = [str(m).strip().lower() for m in (rm.get("legal_moves_uci") or [])]
            if set(rm_legal) != set(fields.legal_moves):
                legal_mismatch_count += 1

            use_logged_metrics = all(k in rec for k in ["pred_move", "acc", "format_reward"])
            if use_logged_metrics:
                pred_move = str(rec.get("pred_move") or "")
                in_subset = pred_move in fields.allowed_moves if pred_move else False
                res = {
                    "score": float(logged_score),
                    "pred_move": pred_move,
                    "acc": float(rec.get("acc") or 0.0),
                    "format_reward": float(rec.get("format_reward") or 0.0),
                    "in_subset": bool(rec.get("in_subset") or in_subset),
                    "penalty_applied": bool(rec.get("penalty_applied") or False),
                    "penalty_reason": str(rec.get("penalty_reason") or ""),
                    "n_considered_moves": int(rec.get("n_considered_moves") or len(fields.allowed_moves)),
                    "target_move": str(rec.get("target_move") or ""),
                }
                recomputed_score = float(logged_score)
            else:
                res = compute_score(
                    data_source=rm,
                    solution_str=output_text,
                    ground_truth=str(rm.get("ground_truth") or ""),
                    chess_reward_fn=str(reward_fn),
                )

                recomputed_score = float(res.get("score"))
                score_diff = abs(recomputed_score - logged_score)
                if not math.isfinite(recomputed_score) or score_diff > 1e-6:
                    score_mismatch_count += 1
                    max_score_diff = max(max_score_diff, score_diff)
                    if len(score_mismatch_samples) < 50:
                        score_mismatch_samples.append(
                            {
                                "step": step,
                                "record": rec_idx,
                                "logged_score": logged_score,
                                "recomputed_score": recomputed_score,
                                "diff": score_diff,
                            }
                        )

            variant = variant or "<missing>"
            agg = per_step_variant[(step, variant)]
            per_step_total[step] += 1

            n_considered = int(res.get("n_considered_moves") or len(fields.allowed_moves))
            agg["n"] += 1
            agg["sum_logged_score"] += float(logged_score)
            agg["sum_recomputed_score"] += float(recomputed_score)
            agg["sum_acc"] += float(res.get("acc") or 0.0)
            agg["sum_format_ok"] += float(res.get("format_reward") or 0.0)
            agg["sum_in_subset"] += 1.0 if bool(res.get("in_subset")) else 0.0
            agg["sum_penalty"] += 1.0 if bool(res.get("penalty_applied")) else 0.0
            agg["sum_n_considered"] += float(n_considered)
            agg["min_n_considered"] = min(agg["min_n_considered"], n_considered)
            agg["max_n_considered"] = max(agg["max_n_considered"], n_considered)

            # Position bias: only when pred_move is valid and in-subset.
            pred_move = str(res.get("pred_move") or "")
            if pred_move and bool(res.get("in_subset")):
                rank_abs = _rank_in_list(fields.allowed_moves, pred_move)
                if rank_abs is not None:
                    rank_ratio = _ratio_from_rank(rank_abs, n_considered)
                    agg["sum_pred_rank_abs"] += float(rank_abs)
                    agg["sum_pred_rank_ratio"] += float(rank_ratio)
                    agg["n_pred_rank"] += 1
                    if not is_ambiguous:
                        train_rank_abs_all.append(int(rank_abs))
                        train_rank_ratio_all.append(float(rank_ratio))
                        train_n_considered_all.append(int(n_considered))
                        train_dist_by_variant[variant]["rank_abs"].append(float(rank_abs))
                        train_dist_by_variant[variant]["rank_ratio"].append(float(rank_ratio))

                        bias = bias_by_n[n_considered]
                        bias["n"] += 1
                        bias["sum_rank_abs"] += float(rank_abs)
                        bias["sum_rank_ratio"] += float(rank_ratio)

                        bias_v = bias_by_n_variant[(n_considered, variant)]
                        bias_v["n"] += 1
                        bias_v["sum_rank_abs"] += float(rank_abs)
                        bias_v["sum_rank_ratio"] += float(rank_ratio)

                        win = _step_window(step)
                        bias_w = bias_by_n_window[(n_considered, win)]
                        bias_w["n"] += 1
                        bias_w["sum_rank_abs"] += float(rank_abs)
                        bias_w["sum_rank_ratio"] += float(rank_ratio)

    # Write mismatch sample if any.
    if score_mismatch_samples:
        sample_path = out_root / "score_mismatch_samples.jsonl"
        with sample_path.open("w", encoding="utf-8") as f:
            for row in score_mismatch_samples:
                f.write(json.dumps(row) + "\n")
        print(f"[WARN] score mismatches: {score_mismatch_count} (sample at {sample_path})")

    if join_miss_count:
        print(f"[WARN] join misses: {join_miss_count}")
    if join_set_fallback:
        print(f"[WARN] join set-fallback used: {join_set_fallback}")
    if ambiguous_variant_count:
        print(f"[WARN] ambiguous variant joins: {ambiguous_variant_count}")
    if legal_mismatch_count:
        print(f"[WARN] legal move mismatches: {legal_mismatch_count}")
    if considered_mismatch_count:
        print(f"[WARN] considered_moves mismatches: {considered_mismatch_count}")

    # Per-step, per-variant table
    step_rows: List[Dict[str, Any]] = []
    for (step, variant), agg in sorted(per_step_variant.items(), key=lambda x: (x[0][0], x[0][1])):
        n = agg["n"]
        if n == 0:
            continue
        total = per_step_total.get(step, n)
        step_rows.append(
            {
                "step": step,
                "derived_variant": variant,
                "n_records": n,
                "fraction_of_step": float(n) / float(total) if total else float("nan"),
                "mean_logged_score": agg["sum_logged_score"] / n,
                "mean_recomputed_score": agg["sum_recomputed_score"] / n,
                "mean_acc": agg["sum_acc"] / n,
                "format_ok_rate": agg["sum_format_ok"] / n,
                "in_subset_rate": agg["sum_in_subset"] / n,
                "penalty_applied_rate": agg["sum_penalty"] / n,
                "mean_n_considered": agg["sum_n_considered"] / n,
                "min_n_considered": agg["min_n_considered"],
                "max_n_considered": agg["max_n_considered"],
                "mean_pred_rank_abs": (agg["sum_pred_rank_abs"] / agg["n_pred_rank"]) if agg["n_pred_rank"] else float("nan"),
                "mean_pred_rank_ratio": (agg["sum_pred_rank_ratio"] / agg["n_pred_rank"]) if agg["n_pred_rank"] else float("nan"),
                "pred_rank_count": int(agg["n_pred_rank"]),
            }
        )

    train_step_df = pd.DataFrame(step_rows).sort_values(["step", "derived_variant"])
    train_step_path = out_root / "train_rollouts_by_step_variant.csv"
    train_step_df.to_csv(train_step_path, index=False)
    print(f"[OK] Wrote {train_step_path}")

    # Bias tables
    bias_rows = []
    for n_considered, agg in sorted(bias_by_n.items()):
        if agg["n"] == 0:
            continue
        bias_rows.append(
            {
                "n_considered": n_considered,
                "n_samples": int(agg["n"]),
                "mean_pred_rank_abs": agg["sum_rank_abs"] / agg["n"],
                "mean_pred_rank_ratio": agg["sum_rank_ratio"] / agg["n"],
            }
        )
    if bias_rows:
        bias_df = pd.DataFrame(bias_rows).sort_values("n_considered")
    else:
        bias_df = pd.DataFrame(columns=["n_considered", "n_samples", "mean_pred_rank_abs", "mean_pred_rank_ratio"])
    bias_path = out_root / "train_position_bias_by_n_considered.csv"
    bias_df.to_csv(bias_path, index=False)
    print(f"[OK] Wrote {bias_path}")

    bias_rows = []
    for (n_considered, variant), agg in sorted(bias_by_n_variant.items(), key=lambda x: (x[0][0], x[0][1])):
        if agg["n"] == 0:
            continue
        bias_rows.append(
            {
                "n_considered": n_considered,
                "derived_variant": variant,
                "n_samples": int(agg["n"]),
                "mean_pred_rank_abs": agg["sum_rank_abs"] / agg["n"],
                "mean_pred_rank_ratio": agg["sum_rank_ratio"] / agg["n"],
            }
        )
    if bias_rows:
        bias_variant_df = pd.DataFrame(bias_rows).sort_values(["n_considered", "derived_variant"])
    else:
        bias_variant_df = pd.DataFrame(
            columns=["n_considered", "derived_variant", "n_samples", "mean_pred_rank_abs", "mean_pred_rank_ratio"]
        )
    bias_variant_path = out_root / "train_position_bias_by_n_considered_variant.csv"
    bias_variant_df.to_csv(bias_variant_path, index=False)
    print(f"[OK] Wrote {bias_variant_path}")

    bias_rows = []
    for (n_considered, window), agg in sorted(bias_by_n_window.items(), key=lambda x: (x[0][0], x[0][1])):
        if agg["n"] == 0:
            continue
        bias_rows.append(
            {
                "n_considered": n_considered,
                "step_window": window,
                "n_samples": int(agg["n"]),
                "mean_pred_rank_abs": agg["sum_rank_abs"] / agg["n"],
                "mean_pred_rank_ratio": agg["sum_rank_ratio"] / agg["n"],
            }
        )
    if bias_rows:
        bias_window_df = pd.DataFrame(bias_rows).sort_values(["n_considered", "step_window"])
    else:
        bias_window_df = pd.DataFrame(
            columns=["n_considered", "step_window", "n_samples", "mean_pred_rank_abs", "mean_pred_rank_ratio"]
        )
    bias_window_path = out_root / "train_position_bias_by_n_considered_step_window.csv"
    bias_window_df.to_csv(bias_window_path, index=False)
    print(f"[OK] Wrote {bias_window_path}")

    # Train position distribution outputs (non-ambiguous variants only)
    ratio_bins = max(1, int(args.ratio_bins))
    ratio_edges = np.linspace(0.0, 1.0, ratio_bins + 1)
    if train_rank_abs_all:
        dist_df = (
            pd.Series(train_rank_abs_all, name="rank_abs")
            .value_counts()
            .sort_index()
            .rename_axis("rank_abs")
            .reset_index(name="count")
        )
        dist_df["fraction"] = dist_df["count"] / float(dist_df["count"].sum())
        dist_path = out_root / "train_pred_rank_abs_distribution.csv"
        dist_df.to_csv(dist_path, index=False)
        print(f"[OK] Wrote {dist_path}")

        plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.bar(dist_df["rank_abs"], dist_df["fraction"], width=0.8)
        ax.set_xlabel("pred_rank_abs")
        ax.set_ylabel("fraction")
        ax.set_title("Train pred rank (abs) distribution")
        ax.grid(True, alpha=0.3)
        _savefig(plots_dir / "train_pred_rank_abs_distribution.png", "Train pred rank (abs) distribution")

        hist, edges = np.histogram(train_rank_ratio_all, bins=ratio_edges)
        ratio_df = pd.DataFrame(
            {
                "ratio_bin_left": edges[:-1],
                "ratio_bin_right": edges[1:],
                "count": hist,
            }
        )
        ratio_df["fraction"] = ratio_df["count"] / float(max(1, hist.sum()))
        ratio_path = out_root / "train_pred_rank_ratio_distribution.csv"
        ratio_df.to_csv(ratio_path, index=False)
        print(f"[OK] Wrote {ratio_path}")

        plt.figure(figsize=(10, 4))
        ax = plt.gca()
        centers = (ratio_df["ratio_bin_left"] + ratio_df["ratio_bin_right"]) / 2.0
        ax.bar(centers, ratio_df["fraction"], width=1.0 / ratio_bins * 0.9)
        ax.set_xlabel("pred_rank_ratio bin")
        ax.set_ylabel("fraction")
        ax.set_title("Train pred rank ratio distribution")
        ax.grid(True, alpha=0.3)
        _savefig(plots_dir / "train_pred_rank_ratio_distribution.png", "Train pred rank ratio distribution")

        # Heatmap: n_considered x rank_abs (counts, log1p)
        dist_raw = pd.DataFrame(
            {
                "n_considered": train_n_considered_all,
                "rank_abs": train_rank_abs_all,
            }
        )
        heat_abs = dist_raw.pivot_table(
            index="n_considered", columns="rank_abs", values="rank_abs", aggfunc="size", fill_value=0
        ).sort_index()
        heat_abs_path = out_root / "train_pred_rank_abs_heatmap.csv"
        heat_abs.to_csv(heat_abs_path)
        print(f"[OK] Wrote {heat_abs_path}")
        heat_abs_norm = heat_abs.div(heat_abs.sum(axis=1), axis=0).fillna(0.0)

        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.imshow(heat_abs_norm.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("pred_rank_abs")
        ax.set_ylabel("n_considered_moves")
        ax.set_title("Train pred rank abs by candidate size (row-normalized)")
        x_vals = heat_abs.columns.tolist()
        y_vals = heat_abs.index.tolist()
        x_step = max(1, int(len(x_vals) / 12))
        y_step = max(1, int(len(y_vals) / 14))
        ax.set_xticks(list(range(0, len(x_vals), x_step)))
        ax.set_xticklabels([x_vals[i] for i in range(0, len(x_vals), x_step)], rotation=90, fontsize=7)
        ax.set_yticks(list(range(0, len(y_vals), y_step)))
        ax.set_yticklabels([y_vals[i] for i in range(0, len(y_vals), y_step)], fontsize=7)
        plt.colorbar(ax.images[0], ax=ax, shrink=0.8, label="fraction")
        _savefig(plots_dir / "train_pred_rank_abs_heatmap.png", "Train pred rank abs heatmap (normalized)")

        # Heatmap: n_considered x ratio_bin
        dist_raw = pd.DataFrame(
            {
                "n_considered": train_n_considered_all,
                "rank_ratio": train_rank_ratio_all,
            }
        )
        dist_raw["ratio_bin"] = pd.cut(dist_raw["rank_ratio"], bins=ratio_edges, include_lowest=True, labels=False)
        heat_ratio = dist_raw.pivot_table(
            index="n_considered", columns="ratio_bin", values="rank_ratio", aggfunc="size", fill_value=0
        ).sort_index()
        heat_ratio_path = out_root / "train_pred_rank_ratio_heatmap.csv"
        heat_ratio.to_csv(heat_ratio_path)
        print(f"[OK] Wrote {heat_ratio_path}")
        heat_ratio_norm = heat_ratio.div(heat_ratio.sum(axis=1), axis=0).fillna(0.0)

        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.imshow(heat_ratio_norm.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("pred_rank_ratio bin")
        ax.set_ylabel("n_considered_moves")
        ax.set_title("Train pred rank ratio by candidate size (row-normalized)")
        bin_labels = []
        for i in range(len(ratio_edges) - 1):
            bin_labels.append(f"{ratio_edges[i]:.2f}-{ratio_edges[i+1]:.2f}")
        x_vals = list(range(len(bin_labels)))
        x_step = max(1, int(len(x_vals) / 8))
        y_vals = heat_ratio.index.tolist()
        y_step = max(1, int(len(y_vals) / 14))
        ax.set_xticks(list(range(0, len(x_vals), x_step)))
        ax.set_xticklabels([bin_labels[i] for i in range(0, len(x_vals), x_step)], fontsize=7, rotation=45)
        ax.set_yticks(list(range(0, len(y_vals), y_step)))
        ax.set_yticklabels([y_vals[i] for i in range(0, len(y_vals), y_step)], fontsize=7)
        plt.colorbar(ax.images[0], ax=ax, shrink=0.8, label="fraction")
        _savefig(plots_dir / "train_pred_rank_ratio_heatmap.png", "Train pred rank ratio heatmap (normalized)")

        # Per-variant distributions (full_legal / hard_neg / coverage_block)
        for variant, dist in sorted(train_dist_by_variant.items()):
            if _is_ambiguous_variant(variant):
                continue
            if not dist["rank_abs"]:
                continue
            abs_counts = (
                pd.Series(dist["rank_abs"], name="rank_abs")
                .value_counts()
                .sort_index()
                .rename_axis("rank_abs")
                .reset_index(name="count")
            )
            abs_counts["fraction"] = abs_counts["count"] / float(abs_counts["count"].sum())
            abs_path = out_root / f"train_{variant}_pred_rank_abs_distribution.csv"
            abs_counts.to_csv(abs_path, index=False)

            plt.figure(figsize=(10, 4))
            ax = plt.gca()
            ax.bar(abs_counts["rank_abs"], abs_counts["fraction"], width=0.8)
            ax.set_xlabel("pred_rank_abs")
            ax.set_ylabel("fraction")
            ax.set_title(f"Train pred rank (abs) distribution: {variant}")
            ax.grid(True, alpha=0.3)
            _savefig(
                plots_dir / f"train_{variant}_pred_rank_abs_distribution.png",
                f"Train pred rank abs distribution ({variant})",
            )

            hist, edges = np.histogram(dist["rank_ratio"], bins=ratio_edges)
            ratio_df = pd.DataFrame(
                {
                    "ratio_bin_left": edges[:-1],
                    "ratio_bin_right": edges[1:],
                    "count": hist,
                }
            )
            ratio_df["fraction"] = ratio_df["count"] / float(max(1, hist.sum()))
            ratio_path = out_root / f"train_{variant}_pred_rank_ratio_distribution.csv"
            ratio_df.to_csv(ratio_path, index=False)

            plt.figure(figsize=(10, 4))
            ax = plt.gca()
            centers = (ratio_df["ratio_bin_left"] + ratio_df["ratio_bin_right"]) / 2.0
            ax.bar(centers, ratio_df["fraction"], width=1.0 / ratio_bins * 0.9)
            ax.set_xlabel("pred_rank_ratio bin")
            ax.set_ylabel("fraction")
            ax.set_title(f"Train pred rank ratio distribution: {variant}")
            ax.grid(True, alpha=0.3)
            _savefig(
                plots_dir / f"train_{variant}_pred_rank_ratio_distribution.png",
                f"Train pred rank ratio distribution ({variant})",
            )

        # Combined grid for per-variant distributions
        variants = [v for v in sorted(train_dist_by_variant.keys()) if not _is_ambiguous_variant(v)]
        variants = [v for v in variants if train_dist_by_variant[v]["rank_abs"]]
        if variants:
            fig, axes = plt.subplots(len(variants), 2, figsize=(12, 3 * len(variants)))
            if len(variants) == 1:
                axes = np.array([axes])
            for i, variant in enumerate(variants):
                dist = train_dist_by_variant[variant]
                abs_counts = (
                    pd.Series(dist["rank_abs"], name="rank_abs")
                    .value_counts()
                    .sort_index()
                    .rename_axis("rank_abs")
                    .reset_index(name="count")
                )
                abs_counts["fraction"] = abs_counts["count"] / float(abs_counts["count"].sum())
                ax = axes[i, 0]
                ax.bar(abs_counts["rank_abs"], abs_counts["fraction"], width=0.8)
                ax.set_title(f"{variant}: abs")
                ax.set_xlabel("pred_rank_abs")
                ax.set_ylabel("fraction")
                ax.grid(True, alpha=0.3)

                hist, edges = np.histogram(dist["rank_ratio"], bins=ratio_edges)
                ratio_df = pd.DataFrame(
                    {
                        "ratio_bin_left": edges[:-1],
                        "ratio_bin_right": edges[1:],
                        "count": hist,
                    }
                )
                ratio_df["fraction"] = ratio_df["count"] / float(max(1, hist.sum()))
                centers = (ratio_df["ratio_bin_left"] + ratio_df["ratio_bin_right"]) / 2.0
                ax = axes[i, 1]
                ax.bar(centers, ratio_df["fraction"], width=1.0 / ratio_bins * 0.9)
                ax.set_title(f"{variant}: ratio")
                ax.set_xlabel("pred_rank_ratio bin")
                ax.set_ylabel("fraction")
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _ensure_dir(plots_dir)
            fig.savefig(plots_dir / "train_pred_rank_distribution_by_variant_grid.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    # Validation position bias
    val_log_files = sorted([p for p in val_dir.glob("*.jsonl") if p.is_file()], key=lambda p: int(p.stem))
    if args.limit_validation_files is not None:
        val_log_files = val_log_files[: int(args.limit_validation_files)]

    # Dataset labels from config
    canonical_label = "val_canonical"
    shuffled_label = "val_shuffled"
    if isinstance(val_files_config, list):
        for vf in val_files_config:
            name = Path(str(vf)).stem
            if "shuffled" in str(vf):
                shuffled_label = name
            else:
                canonical_label = name

    val_agg: Dict[Tuple[int, str], Dict[str, Any]] = defaultdict(_init_val_agg)
    val_legal_mismatch = 0
    val_unknown_dataset = 0
    val_dist: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {
            "pred_rank_abs": [],
            "pred_rank_ratio": [],
            "target_rank_abs": [],
            "target_rank_ratio": [],
        }
    )
    val_k_agg: Dict[Tuple[int, str, int], Dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})

    for file_path in val_log_files:
        step = int(file_path.stem)
        for rec_idx, rec in _iter_jsonl(file_path):
            prompt_text = str(rec.get("input") or "")
            fields = _parse_prompt_fields(prompt_text)
            allowed = fields.allowed_moves
            legal = fields.legal_moves
            n_considered = int(rec.get("n_considered_moves") or len(allowed))

            dataset_label = "unknown"
            lookup_key = (fields.fen, tuple(allowed))
            if lookup_key in val_lookup:
                dataset_label = val_lookup[lookup_key]
            else:
                canonical_legal = legal_by_fen.get(fields.fen)
                if canonical_legal:
                    if canonical_legal != legal:
                        val_legal_mismatch += 1
                    is_full_legal = len(allowed) == len(canonical_legal) and set(allowed) == set(canonical_legal)
                    if is_full_legal:
                        if allowed == canonical_legal:
                            dataset_label = canonical_label
                        else:
                            dataset_label = shuffled_label
            if dataset_label == "unknown":
                val_unknown_dataset += 1

            agg = val_agg[(step, dataset_label)]
            agg["n"] += 1
            agg["sum_n_considered"] += float(n_considered)

            pred_move = str(rec.get("pred_move") or "").strip().lower()
            if "in_subset" in rec:
                in_subset = bool(rec.get("in_subset"))
            else:
                in_subset = pred_move in allowed if pred_move else False
            if pred_move and in_subset:
                rank_abs = _rank_in_list(allowed, pred_move)
                if rank_abs is not None:
                    agg["sum_pred_rank_abs"] += float(rank_abs)
                    agg["sum_pred_rank_ratio"] += float(_ratio_from_rank(rank_abs, n_considered))
                    agg["n_pred_rank"] += 1
                    val_dist[dataset_label]["pred_rank_abs"].append(float(rank_abs))
                    val_dist[dataset_label]["pred_rank_ratio"].append(float(_ratio_from_rank(rank_abs, n_considered)))

            target_move = str(rec.get("target_move") or "").strip().lower()
            if not target_move:
                rm_val = rm_by_fen.get(fields.fen)
                if rm_val:
                    target_move = str(_compute_target_move(rm_val, allowed) or "").strip().lower()
            if target_move:
                rank_abs = _rank_in_list(allowed, target_move)
                if rank_abs is not None:
                    agg["sum_target_rank_abs"] += float(rank_abs)
                    agg["sum_target_rank_ratio"] += float(_ratio_from_rank(rank_abs, n_considered))
                    agg["n_target_rank"] += 1
                    val_dist[dataset_label]["target_rank_abs"].append(float(rank_abs))
                    val_dist[dataset_label]["target_rank_ratio"].append(float(_ratio_from_rank(rank_abs, n_considered)))
                    k_pos = int(rank_abs - 1)
                    k_key = (step, dataset_label, k_pos)
                    val_k_agg[k_key]["n"] += 1
                    if pred_move and pred_move == target_move:
                        val_k_agg[k_key]["correct"] += 1

    val_rows: List[Dict[str, Any]] = []
    for (step, dataset_label), agg in sorted(val_agg.items(), key=lambda x: (x[0][0], x[0][1])):
        n = agg["n"]
        if n == 0:
            continue
        val_rows.append(
            {
                "step": step,
                "dataset": dataset_label,
                "n_records": n,
                "mean_n_considered": agg["sum_n_considered"] / n,
                "mean_pred_rank_abs": (agg["sum_pred_rank_abs"] / agg["n_pred_rank"]) if agg["n_pred_rank"] else float("nan"),
                "mean_pred_rank_ratio": (agg["sum_pred_rank_ratio"] / agg["n_pred_rank"]) if agg["n_pred_rank"] else float("nan"),
                "pred_rank_count": int(agg["n_pred_rank"]),
                "mean_target_rank_abs": (agg["sum_target_rank_abs"] / agg["n_target_rank"]) if agg["n_target_rank"] else float("nan"),
                "mean_target_rank_ratio": (agg["sum_target_rank_ratio"] / agg["n_target_rank"]) if agg["n_target_rank"] else float("nan"),
                "target_rank_count": int(agg["n_target_rank"]),
            }
        )

    val_df = pd.DataFrame(val_rows).sort_values(["step", "dataset"])
    val_path = out_root / "validation_position_bias_by_step.csv"
    val_df.to_csv(val_path, index=False)
    print(f"[OK] Wrote {val_path}")

    val_k_rows: List[Dict[str, Any]] = []
    for (step, dataset_label, k_pos), agg in sorted(val_k_agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        n = int(agg["n"])
        correct = int(agg["correct"])
        if n <= 0:
            continue
        val_k_rows.append(
            {
                "step": int(step),
                "dataset": dataset_label,
                "k_pos": int(k_pos),
                "n_records": n,
                "n_correct": correct,
                "pass_at1": float(correct) / float(n),
            }
        )
    val_k_df = pd.DataFrame(val_k_rows).sort_values(["step", "dataset", "k_pos"])
    val_k_path = out_root / "validation_pass1_by_target_k.csv"
    val_k_df.to_csv(val_k_path, index=False)
    print(f"[OK] Wrote {val_k_path}")

    val_bias_rows: List[Dict[str, Any]] = []
    if not val_k_df.empty:
        for (step, dataset_label), sub in val_k_df.groupby(["step", "dataset"]):
            xs = sub["k_pos"].astype(float).tolist()
            ns = sub["n_records"].astype(float).tolist()
            corrects = sub["n_correct"].astype(float).tolist()
            total_n = float(sum(ns))
            if total_n <= 0.0:
                continue
            total_correct = float(sum(corrects))
            mean_acc = total_correct / total_n
            mean_k = sum(k * n for k, n in zip(xs, ns)) / total_n
            mean_k2 = sum((k ** 2) * n for k, n in zip(xs, ns)) / total_n
            mean_k_correct = sum(k * c for k, c in zip(xs, corrects)) / total_n
            var_k = mean_k2 - mean_k ** 2
            var_y = mean_acc - mean_acc ** 2
            cov_ky = mean_k_correct - mean_k * mean_acc
            if var_k > 0.0 and var_y > 0.0:
                corr_ky = cov_ky / math.sqrt(var_k * var_y)
            else:
                corr_ky = float("nan")
            accs = [c / n if n > 0 else 0.0 for c, n in zip(corrects, ns)]
            slope = _weighted_slope(xs, accs, ns)
            val_bias_rows.append(
                {
                    "step": int(step),
                    "dataset": dataset_label,
                    "n_records": int(total_n),
                    "mean_pass_at1": mean_acc,
                    "mean_k_pos": mean_k,
                    "k_pass1_slope": slope,
                    "k_pass1_corr": corr_ky,
                }
            )
    val_bias_df = pd.DataFrame(val_bias_rows).sort_values(["step", "dataset"])
    val_bias_path = out_root / "validation_pass1_by_target_k_bias.csv"
    val_bias_df.to_csv(val_bias_path, index=False)
    print(f"[OK] Wrote {val_bias_path}")

    if val_legal_mismatch:
        print(f"[WARN] validation legal-move mismatches: {val_legal_mismatch}")
    if val_unknown_dataset:
        print(f"[WARN] validation records with unknown dataset label: {val_unknown_dataset}")

    # Validation position distributions (aggregated over steps)
    for dataset_label, dist in sorted(val_dist.items()):
        if dist["pred_rank_abs"]:
            abs_counts = (
                pd.Series(dist["pred_rank_abs"], name="rank_abs")
                .value_counts()
                .sort_index()
                .rename_axis("rank_abs")
                .reset_index(name="count")
            )
            abs_counts["fraction"] = abs_counts["count"] / float(abs_counts["count"].sum())
            abs_path = out_root / f"validation_{dataset_label}_pred_rank_abs_distribution.csv"
            abs_counts.to_csv(abs_path, index=False)

            plt.figure(figsize=(10, 4))
            ax = plt.gca()
            ax.bar(abs_counts["rank_abs"], abs_counts["fraction"], width=0.8)
            ax.set_xlabel("pred_rank_abs")
            ax.set_ylabel("fraction")
            ax.set_title(f"Validation pred rank (abs) distribution: {dataset_label}")
            ax.grid(True, alpha=0.3)
            _savefig(
                plots_dir / f"validation_{dataset_label}_pred_rank_abs_distribution.png",
                f"Validation pred rank abs distribution ({dataset_label})",
            )

            hist, edges = np.histogram(dist["pred_rank_ratio"], bins=ratio_edges)
            ratio_df = pd.DataFrame(
                {
                    "ratio_bin_left": edges[:-1],
                    "ratio_bin_right": edges[1:],
                    "count": hist,
                }
            )
            ratio_df["fraction"] = ratio_df["count"] / float(max(1, hist.sum()))
            ratio_path = out_root / f"validation_{dataset_label}_pred_rank_ratio_distribution.csv"
            ratio_df.to_csv(ratio_path, index=False)

            plt.figure(figsize=(10, 4))
            ax = plt.gca()
            centers = (ratio_df["ratio_bin_left"] + ratio_df["ratio_bin_right"]) / 2.0
            ax.bar(centers, ratio_df["fraction"], width=1.0 / ratio_bins * 0.9)
            ax.set_xlabel("pred_rank_ratio bin")
            ax.set_ylabel("fraction")
            ax.set_title(f"Validation pred rank ratio distribution: {dataset_label}")
            ax.grid(True, alpha=0.3)
            _savefig(
                plots_dir / f"validation_{dataset_label}_pred_rank_ratio_distribution.png",
                f"Validation pred rank ratio distribution ({dataset_label})",
            )

        if dist["target_rank_abs"]:
            abs_counts = (
                pd.Series(dist["target_rank_abs"], name="rank_abs")
                .value_counts()
                .sort_index()
                .rename_axis("rank_abs")
                .reset_index(name="count")
            )
            abs_counts["fraction"] = abs_counts["count"] / float(abs_counts["count"].sum())
            abs_path = out_root / f"validation_{dataset_label}_target_rank_abs_distribution.csv"
            abs_counts.to_csv(abs_path, index=False)

            plt.figure(figsize=(10, 4))
            ax = plt.gca()
            ax.bar(abs_counts["rank_abs"], abs_counts["fraction"], width=0.8)
            ax.set_xlabel("target_rank_abs")
            ax.set_ylabel("fraction")
            ax.set_title(f"Validation target rank (abs) distribution: {dataset_label}")
            ax.grid(True, alpha=0.3)
            _savefig(
                plots_dir / f"validation_{dataset_label}_target_rank_abs_distribution.png",
                f"Validation target rank abs distribution ({dataset_label})",
            )

            hist, edges = np.histogram(dist["target_rank_ratio"], bins=ratio_edges)
            ratio_df = pd.DataFrame(
                {
                    "ratio_bin_left": edges[:-1],
                    "ratio_bin_right": edges[1:],
                    "count": hist,
                }
            )
            ratio_df["fraction"] = ratio_df["count"] / float(max(1, hist.sum()))
            ratio_path = out_root / f"validation_{dataset_label}_target_rank_ratio_distribution.csv"
            ratio_df.to_csv(ratio_path, index=False)

            plt.figure(figsize=(10, 4))
            ax = plt.gca()
            centers = (ratio_df["ratio_bin_left"] + ratio_df["ratio_bin_right"]) / 2.0
            ax.bar(centers, ratio_df["fraction"], width=1.0 / ratio_bins * 0.9)
            ax.set_xlabel("target_rank_ratio bin")
            ax.set_ylabel("fraction")
            ax.set_title(f"Validation target rank ratio distribution: {dataset_label}")
            ax.grid(True, alpha=0.3)
            _savefig(
                plots_dir / f"validation_{dataset_label}_target_rank_ratio_distribution.png",
                f"Validation target rank ratio distribution ({dataset_label})",
            )

    # Validation pass@1 vs target position K (per step)
    if not val_k_df.empty:
        dataset_order: List[str] = []
        if isinstance(val_files_config, list):
            dataset_order = [Path(str(vf)).stem for vf in val_files_config]
        for label in sorted(val_k_df["dataset"].unique().tolist()):
            if label not in dataset_order:
                dataset_order.append(label)

        for step, step_df in val_k_df.groupby("step"):
            fig, ax = plt.subplots(figsize=(10, 4.8))
            for dataset_label in dataset_order:
                sub = step_df[step_df["dataset"] == dataset_label]
                if sub.empty:
                    continue
                ax.plot(
                    sub["k_pos"],
                    sub["pass_at1"],
                    marker="o",
                    linewidth=1.2,
                    alpha=0.7,
                    label=dataset_label,
                )
            ax.set_xlabel("target position K (0-based in allowed_moves)")
            ax.set_ylabel("pass@1")
            ax.set_title(f"Validation pass@1 vs target position K (step {step})")
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(plots_dir / f"validation_pass1_by_k_step_{step}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

        # Combined view: step 0 vs step 240 on the same axes, with distinct styling.
        step_a, step_b = 0, 240
        if step_a in val_k_df["step"].unique() and step_b in val_k_df["step"].unique():
            fig, ax = plt.subplots(figsize=(10.5, 4.8))
            step_style = {
                step_a: {"color": "#1f77b4", "linestyle": "-", "alpha": 0.9},
                step_b: {"color": "#d62728", "linestyle": "--", "alpha": 0.9},
            }
            marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]
            for idx, dataset_label in enumerate(dataset_order):
                marker = marker_cycle[idx % len(marker_cycle)]
                sub_a = val_k_df[(val_k_df["step"] == step_a) & (val_k_df["dataset"] == dataset_label)]
                sub_b = val_k_df[(val_k_df["step"] == step_b) & (val_k_df["dataset"] == dataset_label)]
                if not sub_a.empty:
                    ax.plot(
                        sub_a["k_pos"],
                        sub_a["pass_at1"],
                        marker=marker,
                        linewidth=1.8,
                        color=step_style[step_a]["color"],
                        linestyle=step_style[step_a]["linestyle"],
                        alpha=step_style[step_a]["alpha"],
                        label=f"step {step_a} · {dataset_label}",
                    )
                if not sub_b.empty:
                    ax.plot(
                        sub_b["k_pos"],
                        sub_b["pass_at1"],
                        marker=marker,
                        linewidth=1.8,
                        color=step_style[step_b]["color"],
                        linestyle=step_style[step_b]["linestyle"],
                        alpha=step_style[step_b]["alpha"],
                        label=f"step {step_b} · {dataset_label}",
                    )
            ax.set_xlabel("target position K (0-based in allowed_moves)")
            ax.set_ylabel("pass@1")
            ax.set_title("Validation pass@1 vs target position K (step 0 vs step 240)")
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right", ncol=2)
            fig.tight_layout()
            fig.savefig(
                plots_dir / "validation_pass1_by_k_step0_vs_step240.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)

    # Plots: train metrics by variant (single panel)
    smooth_window = int(args.smooth_window)

    if not train_step_df.empty:
        metrics = [
            ("mean_recomputed_score", "mean score"),
            ("mean_acc", "mean acc"),
            ("penalty_applied_rate", "penalty rate"),
        ]
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 9), sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        for (metric, label), ax in zip(metrics, axes):
            for variant, sub_df in train_step_df.groupby("derived_variant"):
                if _is_ambiguous_variant(str(variant)):
                    continue
                ax.plot(sub_df["step"], sub_df[metric], label=variant, linewidth=1.2, alpha=0.5)
                if smooth_window > 1:
                    y_smooth = _smooth_values(sub_df[metric].tolist(), smooth_window)
                    ax.plot(sub_df["step"], y_smooth, linewidth=2.2)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("step")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))
        fig.suptitle("Train metrics by derived_variant", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(plots_dir / "train_metrics_by_variant.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # Variant composition by step (fraction of step)
        comp_df = train_step_df[
            (~train_step_df["derived_variant"].astype(str).apply(_is_ambiguous_variant))
        ].copy()
        comp_df = comp_df[comp_df["derived_variant"].isin(["full_legal", "hard_neg", "coverage_block"])]
        if not comp_df.empty:
            pivot = (
                comp_df.pivot_table(
                    index="step",
                    columns="derived_variant",
                    values="fraction_of_step",
                    aggfunc="sum",
                )
                .fillna(0.0)
                .sort_index()
            )
            ordered_cols = [c for c in ["full_legal", "hard_neg", "coverage_block"] if c in pivot.columns]
            colors = ["#0072B2", "#D55E00", "#009E73"]  # blue, vermillion, green
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.stackplot(
                pivot.index,
                [pivot[c].values for c in ordered_cols],
                labels=ordered_cols,
                colors=colors[: len(ordered_cols)],
                alpha=0.85,
            )
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("step")
            ax.set_ylabel("fraction of batch")
            ax.set_title("Train batch composition by derived_variant")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(loc="upper right", ncol=min(3, len(ordered_cols)))
            fig.tight_layout()
            fig.savefig(plots_dir / "train_variant_composition_by_step.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    # Filter-groups composition (kept vs rejected, rejection reasons)
    history_path = evidence_root / "history.jsonl"
    filter_df = _load_history_filter_groups(history_path)
    if not filter_df.empty:
        filter_csv = out_root / "filter_groups_by_step.csv"
        filter_df.to_csv(filter_csv, index=False)
        print(f"[OK] Wrote {filter_csv}")

        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.stackplot(
            filter_df["step"],
            filter_df["kept_frac"].fillna(0.0),
            filter_df["rejected_frac"].fillna(0.0),
            labels=["kept", "rejected"],
            colors=["#0072B2", "#D55E00"],
            alpha=0.85,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("step")
        ax.set_ylabel("fraction of groups")
        ax.set_title("Filter-groups: kept vs rejected (fraction)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(plots_dir / "filter_groups_kept_rejected_fraction.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.stackplot(
            filter_df["step"],
            filter_df["rejected_all_valid_frac"].fillna(0.0),
            filter_df["rejected_mixed_frac"].fillna(0.0),
            filter_df["rejected_out_of_subset_frac"].fillna(0.0),
            labels=["all_valid", "mixed", "out_of_subset"],
            colors=["#56B4E9", "#E69F00", "#009E73"],
            alpha=0.85,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("step")
        ax.set_ylabel("fraction of rejected groups")
        ax.set_title("Filter-groups: rejected composition by penalty reason")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", ncol=3)
        fig.tight_layout()
        fig.savefig(plots_dir / "filter_groups_rejected_composition.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Plots: bias vs n_considered (combined)
    if not bias_df.empty:
        df = bias_df.sort_values("n_considered")
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})
        ax1, ax2, ax3 = axes
        ax1.plot(df["n_considered"], df["mean_pred_rank_abs"], marker="o", linewidth=1.2, alpha=0.5)
        if smooth_window > 1:
            y_smooth = _smooth_values(df["mean_pred_rank_abs"].tolist(), smooth_window)
            ax1.plot(df["n_considered"], y_smooth, linewidth=2.2)
        ax1.set_ylabel("mean pred rank (abs)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(df["n_considered"], df["mean_pred_rank_ratio"], marker="o", linewidth=1.2, alpha=0.5)
        if smooth_window > 1:
            y_smooth = _smooth_values(df["mean_pred_rank_ratio"].tolist(), smooth_window)
            ax2.plot(df["n_considered"], y_smooth, linewidth=2.2)
        ax2.set_ylabel("mean pred rank ratio")
        ax2.grid(True, alpha=0.3)

        ax3.bar(df["n_considered"], df["n_samples"], width=0.8)
        ax3.set_ylabel("count")
        ax3.set_xlabel("n_considered_moves")
        fig.suptitle("Train position bias vs candidate size", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(plots_dir / "train_position_bias_by_n_considered.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Plots: validation position bias by step (combined)
    if not val_df.empty:
        # Pred metrics
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for dataset_label, sub_df in val_df.groupby("dataset"):
            axes[0].plot(sub_df["step"], sub_df["mean_pred_rank_abs"], label=dataset_label, linewidth=1.2, alpha=0.5)
            axes[1].plot(sub_df["step"], sub_df["mean_pred_rank_ratio"], label=dataset_label, linewidth=1.2, alpha=0.5)
            if smooth_window > 1:
                y0 = _smooth_values(sub_df["mean_pred_rank_abs"].tolist(), smooth_window)
                y1 = _smooth_values(sub_df["mean_pred_rank_ratio"].tolist(), smooth_window)
                axes[0].plot(sub_df["step"], y0, linewidth=2.2)
                axes[1].plot(sub_df["step"], y1, linewidth=2.2)
        axes[0].set_ylabel("mean pred rank (abs)")
        axes[1].set_ylabel("mean pred rank ratio")
        axes[1].set_xlabel("step")
        axes[0].grid(True, alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))
        fig.suptitle("Validation pred rank by step", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(plots_dir / "validation_pred_rank_by_step.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # Target metrics
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for dataset_label, sub_df in val_df.groupby("dataset"):
            axes[0].plot(sub_df["step"], sub_df["mean_target_rank_abs"], label=dataset_label, linewidth=1.2, alpha=0.5)
            axes[1].plot(sub_df["step"], sub_df["mean_target_rank_ratio"], label=dataset_label, linewidth=1.2, alpha=0.5)
            if smooth_window > 1:
                y0 = _smooth_values(sub_df["mean_target_rank_abs"].tolist(), smooth_window)
                y1 = _smooth_values(sub_df["mean_target_rank_ratio"].tolist(), smooth_window)
                axes[0].plot(sub_df["step"], y0, linewidth=2.2)
                axes[1].plot(sub_df["step"], y1, linewidth=2.2)
        axes[0].set_ylabel("mean target rank (abs)")
        axes[1].set_ylabel("mean target rank ratio")
        axes[1].set_xlabel("step")
        axes[0].grid(True, alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))
        fig.suptitle("Validation target rank by step", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(plots_dir / "validation_target_rank_by_step.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Comparison plots (optional)
    if args.compare_evidence_root is not None:
        compare_root = Path(args.compare_evidence_root)
        compare_label = str(args.compare_label)
        comp_val = _load_csv_if_exists(compare_root / "investigation" / "validation_position_bias_by_step.csv")
        comp_train = _load_csv_if_exists(compare_root / "investigation" / "train_rollouts_by_step_variant.csv")

        if comp_val is not None and not val_df.empty:
            def _pick_dataset(df: pd.DataFrame) -> str:
                labels = df["dataset"].unique().tolist()
                if "test" in labels:
                    return "test"
                return labels[0]

            cur_label = _pick_dataset(val_df)
            comp_label = _pick_dataset(comp_val)
            cur = val_df[val_df["dataset"] == cur_label]
            comp = comp_val[comp_val["dataset"] == comp_label]

            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axes[0].plot(cur["step"], cur["mean_pred_rank_abs"], label=f"{evidence_root.name}:{cur_label}", alpha=0.5)
            axes[1].plot(cur["step"], cur["mean_pred_rank_ratio"], label=f"{evidence_root.name}:{cur_label}", alpha=0.5)
            axes[0].plot(comp["step"], comp["mean_pred_rank_abs"], label=f"{compare_label}:{comp_label}", alpha=0.5)
            axes[1].plot(comp["step"], comp["mean_pred_rank_ratio"], label=f"{compare_label}:{comp_label}", alpha=0.5)
            if smooth_window > 1:
                axes[0].plot(cur["step"], _smooth_values(cur["mean_pred_rank_abs"].tolist(), smooth_window), linewidth=2.2)
                axes[1].plot(cur["step"], _smooth_values(cur["mean_pred_rank_ratio"].tolist(), smooth_window), linewidth=2.2)
                axes[0].plot(comp["step"], _smooth_values(comp["mean_pred_rank_abs"].tolist(), smooth_window), linewidth=2.2)
                axes[1].plot(comp["step"], _smooth_values(comp["mean_pred_rank_ratio"].tolist(), smooth_window), linewidth=2.2)
            axes[0].set_ylabel("mean pred rank (abs)")
            axes[1].set_ylabel("mean pred rank ratio")
            axes[1].set_xlabel("step")
            axes[0].grid(True, alpha=0.3)
            axes[1].grid(True, alpha=0.3)
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=2)
            fig.suptitle("Validation pred rank comparison", y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(plots_dir / "compare_validation_pred_rank_by_step.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

        if comp_train is not None and not train_step_df.empty:
            def _pick_variant(df: pd.DataFrame) -> str:
                labels = [v for v in df["derived_variant"].unique().tolist() if not _is_ambiguous_variant(str(v))]
                if "full_legal" in labels:
                    return "full_legal"
                return labels[0] if labels else df["derived_variant"].unique().tolist()[0]

            cur_variant = _pick_variant(train_step_df)
            comp_variant = _pick_variant(comp_train)
            cur = train_step_df[train_step_df["derived_variant"] == cur_variant]
            comp = comp_train[comp_train["derived_variant"] == comp_variant]

            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axes[0].plot(cur["step"], cur["mean_acc"], label=f"{evidence_root.name}:{cur_variant}", alpha=0.5)
            axes[1].plot(cur["step"], cur["mean_recomputed_score"], label=f"{evidence_root.name}:{cur_variant}", alpha=0.5)
            axes[0].plot(comp["step"], comp["mean_acc"], label=f"{compare_label}:{comp_variant}", alpha=0.5)
            axes[1].plot(comp["step"], comp["mean_recomputed_score"], label=f"{compare_label}:{comp_variant}", alpha=0.5)
            if smooth_window > 1:
                axes[0].plot(cur["step"], _smooth_values(cur["mean_acc"].tolist(), smooth_window), linewidth=2.2)
                axes[1].plot(cur["step"], _smooth_values(cur["mean_recomputed_score"].tolist(), smooth_window), linewidth=2.2)
                axes[0].plot(comp["step"], _smooth_values(comp["mean_acc"].tolist(), smooth_window), linewidth=2.2)
                axes[1].plot(comp["step"], _smooth_values(comp["mean_recomputed_score"].tolist(), smooth_window), linewidth=2.2)
            axes[0].set_ylabel("mean acc")
            axes[1].set_ylabel("mean score")
            axes[1].set_xlabel("step")
            axes[0].grid(True, alpha=0.3)
            axes[1].grid(True, alpha=0.3)
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=2)
            fig.suptitle("Train full-legal comparison", y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(plots_dir / "compare_train_full_legal.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    print(f"[OK] Plots written under {plots_dir}")

    # Summary markdown
    summary_lines: List[str] = []
    summary_lines.append("# v2 selection training analysis summary")
    summary_lines.append("")
    summary_lines.append(f"Evidence root: `{evidence_root}`")
    summary_lines.append(f"Train dataset: `{train_path}`")
    summary_lines.append(f"Reward fn: `{reward_fn}`")
    summary_lines.append("")
    summary_lines.append("## Run config")
    summary_lines.append("")
    summary_lines.append(f"See: `{out_root / 'run_config_summary.md'}`")
    summary_lines.append("")

    # Aggregate train metrics per variant
    if not train_step_df.empty:
        summary_lines.append("## Train rollout summary (per derived_variant)")
        summary_lines.append("")
        rows = []
        total_non_amb = 0
        for variant, sub_df in train_step_df.groupby("derived_variant"):
            if _is_ambiguous_variant(str(variant)):
                continue
            total_non_amb += sub_df["n_records"].sum()
        for variant, sub_df in train_step_df.groupby("derived_variant"):
            if _is_ambiguous_variant(str(variant)):
                continue
            n = sub_df["n_records"].sum()
            if n <= 0:
                continue
            def wmean(col: str) -> float:
                return float((sub_df[col] * sub_df["n_records"]).sum() / n)

            rows.append(
                [
                    variant,
                    int(n),
                    float(n) / float(total_non_amb) if total_non_amb else float("nan"),
                    wmean("mean_recomputed_score"),
                    wmean("mean_acc"),
                    wmean("penalty_applied_rate"),
                    wmean("mean_n_considered"),
                    wmean("mean_pred_rank_abs"),
                    wmean("mean_pred_rank_ratio"),
                ]
            )
        rows = sorted(rows, key=lambda r: r[0])
        summary_lines.append(
            _markdown_table(
                [
                    "derived_variant",
                    "n_records",
                    "fraction",
                    "mean_score",
                    "mean_acc",
                    "penalty_rate",
                    "mean_n_considered",
                    "mean_pred_rank_abs",
                    "mean_pred_rank_ratio",
                ],
                rows,
            )
        )
        summary_lines.append("")
        summary_lines.append(
            "Note: ambiguous variants are excluded from this summary and from all plots/distributions."
        )
        summary_lines.append("")

    # Validation summary per dataset
    if not val_df.empty:
        summary_lines.append("## Validation position-bias summary (per dataset)")
        summary_lines.append("")
        rows = []
        for dataset_label, sub_df in val_df.groupby("dataset"):
            n_total = sub_df["n_records"].sum()
            if n_total <= 0:
                continue
            n_steps = sub_df["step"].nunique()
            per_step = float(n_total) / float(max(1, n_steps))
            def wmean(col: str) -> float:
                return float((sub_df[col] * sub_df["n_records"]).sum() / n_total)
            rows.append(
                [
                    dataset_label,
                    int(n_steps),
                    per_step,
                    int(n_total),
                    wmean("mean_pred_rank_abs"),
                    wmean("mean_pred_rank_ratio"),
                    wmean("mean_target_rank_abs"),
                    wmean("mean_target_rank_ratio"),
                ]
            )
        summary_lines.append(
            _markdown_table(
                [
                    "dataset",
                    "n_steps",
                    "n_records_per_step",
                    "n_records_total",
                    "mean_pred_rank_abs",
                    "mean_pred_rank_ratio",
                    "mean_target_rank_abs",
                    "mean_target_rank_ratio",
                ],
                sorted(rows, key=lambda r: r[0]),
            )
        )
        summary_lines.append("")
        summary_lines.append(
            "Target-rank metrics are computed against the **allowed_moves order** in each prompt. "
            "For the canonical test set this is the AIcrowd legal order; for the shuffled set it is the per-row shuffle."
        )
        summary_lines.append("")

    if not bias_df.empty:
        summary_lines.append("## Train position bias by candidate size (top sizes by count)")
        summary_lines.append("")
        top_bias = bias_df.sort_values("n_samples", ascending=False).head(10)
        rows = []
        for _, r in top_bias.iterrows():
            rows.append(
                [
                    int(r["n_considered"]),
                    int(r["n_samples"]),
                    float(r["mean_pred_rank_abs"]),
                    float(r["mean_pred_rank_ratio"]),
                ]
            )
        summary_lines.append(
            _markdown_table(
                ["n_considered", "n_samples", "mean_pred_rank_abs", "mean_pred_rank_ratio"],
                rows,
            )
        )
        summary_lines.append("")
        summary_lines.append(
            "Note: mean_pred_rank_ratio is averaged per-record; with varying candidate sizes it will not\n"
            "equal (mean_pred_rank_abs - 1) / (mean_n_considered - 1)."
        )
        summary_lines.append("")

    summary_lines.append("## Plots")
    summary_lines.append("")
    summary_lines.append("### Train metrics by variant")
    summary_lines.append("")
    summary_lines.append(f"![train metrics](plots/train_metrics_by_variant.png)")
    summary_lines.append("")
    if (plots_dir / "train_variant_composition_by_step.png").exists():
        summary_lines.append("### Train batch composition")
        summary_lines.append("")
        summary_lines.append("![train composition](plots/train_variant_composition_by_step.png)")
        summary_lines.append("")
    if (plots_dir / "filter_groups_kept_rejected_fraction.png").exists():
        summary_lines.append("### Filter-groups composition")
        summary_lines.append("")
        summary_lines.append(
            "Filter-groups metrics come from `history.jsonl` and report **kept vs rejected** group counts.\n"
            "They do not include per-variant labels for rejected groups; only the overall rejection reasons."
        )
        summary_lines.append("")
        summary_lines.append("![filter kept vs rejected](plots/filter_groups_kept_rejected_fraction.png)")
        summary_lines.append("")
    if (plots_dir / "filter_groups_rejected_composition.png").exists():
        summary_lines.append("![filter rejected composition](plots/filter_groups_rejected_composition.png)")
        summary_lines.append("")
    summary_lines.append("### Train position bias (means)")
    summary_lines.append("")
    summary_lines.append(f"![train bias vs n](plots/train_position_bias_by_n_considered.png)")
    summary_lines.append("")
    summary_lines.append("### Train position distribution")
    summary_lines.append("")
    summary_lines.append(f"![train abs distribution](plots/train_pred_rank_abs_distribution.png)")
    summary_lines.append(f"![train ratio distribution](plots/train_pred_rank_ratio_distribution.png)")
    summary_lines.append(f"![train abs heatmap](plots/train_pred_rank_abs_heatmap.png)")
    summary_lines.append(f"![train ratio heatmap](plots/train_pred_rank_ratio_heatmap.png)")
    summary_lines.append("")
    summary_lines.append("### Train position distributions (per variant)")
    summary_lines.append("")
    summary_lines.append(f"![train variant grid](plots/train_pred_rank_distribution_by_variant_grid.png)")
    summary_lines.append("")
    summary_lines.append("### Validation position bias (means by step)")
    summary_lines.append("")
    summary_lines.append(f"![val pred by step](plots/validation_pred_rank_by_step.png)")
    summary_lines.append(f"![val target by step](plots/validation_target_rank_by_step.png)")
    summary_lines.append("")
    summary_lines.append("### Validation position distributions (per dataset)")
    summary_lines.append("")
    for dataset_label in sorted(val_dist.keys()):
        summary_lines.append(f"#### {dataset_label}")
        summary_lines.append("")
        summary_lines.append(f"![val {dataset_label} pred abs](plots/validation_{dataset_label}_pred_rank_abs_distribution.png)")
        summary_lines.append(f"![val {dataset_label} pred ratio](plots/validation_{dataset_label}_pred_rank_ratio_distribution.png)")
        summary_lines.append(f"![val {dataset_label} target abs](plots/validation_{dataset_label}_target_rank_abs_distribution.png)")
        summary_lines.append(f"![val {dataset_label} target ratio](plots/validation_{dataset_label}_target_rank_ratio_distribution.png)")
        summary_lines.append("")

    if not val_bias_df.empty:
        summary_lines.append("### Validation pass@1 vs target position K")
        summary_lines.append("")
        summary_lines.append(
            "Pass@1 vs K uses the **target move's 0-based index in allowed_moves** for each prompt. "
            "`k_pass1_slope` is a weighted least-squares slope of pass@1 vs K "
            "(negative means higher accuracy for earlier positions). "
            "`k_pass1_corr` is the point-biserial correlation between correctness and K."
        )
        summary_lines.append("")
        for dataset_label in sorted(val_bias_df["dataset"].unique().tolist()):
            sub = val_bias_df[val_bias_df["dataset"] == dataset_label].sort_values("step")
            if sub.empty:
                continue
            rows = []
            for _, r in sub.iterrows():
                rows.append(
                    [
                        int(r["step"]),
                        int(r["n_records"]),
                        float(r["mean_pass_at1"]),
                        float(r["mean_k_pos"]),
                        float(r["k_pass1_slope"]),
                        float(r["k_pass1_corr"]),
                    ]
                )
            summary_lines.append(f"#### {dataset_label} bias by step")
            summary_lines.append("")
            summary_lines.append(
                _markdown_table(
                    ["step", "n_records", "mean_pass@1", "mean_k_pos", "k_pass1_slope", "k_pass1_corr"],
                    rows,
                )
            )
            summary_lines.append("")
        summary_lines.append(
            "Per-step plots are written to `plots/validation_pass1_by_k_step_<step>.png` "
            "(one plot per validation step)."
        )
        summary_lines.append("")

    if args.compare_evidence_root is not None:
        summary_lines.append("## Baseline comparison")
        summary_lines.append("")
        if (plots_dir / "compare_validation_pred_rank_by_step.png").exists():
            summary_lines.append("### Validation pred-rank comparison")
            summary_lines.append("")
            summary_lines.append("![compare validation pred](plots/compare_validation_pred_rank_by_step.png)")
            summary_lines.append("")
        if (plots_dir / "compare_train_full_legal.png").exists():
            summary_lines.append("### Train full-legal comparison")
            summary_lines.append("")
            summary_lines.append("![compare train](plots/compare_train_full_legal.png)")
            summary_lines.append("")

    summary_path = out_root / "summary_analysis.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
