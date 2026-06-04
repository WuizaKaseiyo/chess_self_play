#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None


LOGISTIC_K = math.log(10.0) / 400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one-on-one round-robin infer summaries, validate pair coverage, "
            "fit global Elo, and export CSV/JSON/PNG artifacts."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help=(
            "Input directory containing either summary_infer_shard_*.json files "
            "or shard_*/summary_infer.json files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated artifacts (default: input directory).",
    )
    parser.add_argument(
        "--input-layout",
        choices=["auto", "flat", "sharded"],
        default="auto",
        help=(
            "Input discovery mode: auto (prefer flat if both exist), "
            "flat (summary_infer_shard_*.json), or sharded (shard_*/summary_infer.json)."
        ),
    )
    parser.add_argument(
        "--expected-model-count",
        type=int,
        default=None,
        help="Optional expected number of models for completeness checks.",
    )
    parser.add_argument(
        "--expected-games-per-pair",
        type=int,
        default=None,
        help="Optional expected games per pair for consistency checks.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Continue even if completeness checks fail; checks are still recorded in elo_summary.json.",
    )
    parser.add_argument(
        "--elo-anchor",
        type=float,
        default=1500.0,
        help="Mean Elo anchor used during fitting (default: 1500).",
    )
    parser.add_argument(
        "--elo-max-iter",
        type=int,
        default=200,
        help="Maximum Newton iterations for Elo fitting.",
    )
    parser.add_argument(
        "--elo-tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance for max absolute rating update.",
    )
    parser.add_argument(
        "--elo-damping",
        type=float,
        default=1e-3,
        help="Diagonal damping for Newton system stability.",
    )
    parser.add_argument(
        "--implied-elo-epsilon",
        type=float,
        default=1e-4,
        help="Clipping epsilon for implied pairwise Elo from winrate.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG output DPI.",
    )
    return parser.parse_args()


def discover_summary_paths(input_path: Path, layout: str) -> Tuple[List[Path], str]:
    if input_path.is_file():
        return [input_path.resolve()], "single-file"

    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist or is not a directory: {input_path}")

    flat = sorted(input_path.glob("summary_infer_shard_*.json"))
    sharded = sorted(input_path.glob("shard_*/summary_infer.json"))

    if layout == "flat":
        if not flat:
            raise FileNotFoundError(f"No files matched summary_infer_shard_*.json in {input_path}")
        return [p.resolve() for p in flat], "flat"
    if layout == "sharded":
        if not sharded:
            raise FileNotFoundError(f"No files matched shard_*/summary_infer.json in {input_path}")
        return [p.resolve() for p in sharded], "sharded"

    if flat and sharded:
        return [p.resolve() for p in flat], "flat-preferred"
    if flat:
        return [p.resolve() for p in flat], "flat"
    if sharded:
        return [p.resolve() for p in sharded], "sharded"

    raise FileNotFoundError(
        "No infer summaries found. Expected either summary_infer_shard_*.json or shard_*/summary_infer.json"
    )


def _as_int(value: Any, *, field_name: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"Could not parse integer field {field_name!r}: {value!r}") from exc
    if out < 0:
        raise ValueError(f"Integer field {field_name!r} must be non-negative, got {out}")
    return out


def _extract_run_ids(pair_id: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    run_id_a = str(payload.get("run_id_a", "")).strip()
    run_id_b = str(payload.get("run_id_b", "")).strip()
    if run_id_a and run_id_b:
        if run_id_a == run_id_b:
            raise ValueError(f"Invalid pair with identical run ids in {pair_id!r}")
        return run_id_a, run_id_b

    if "_vs_" not in pair_id:
        raise ValueError(f"Could not infer run ids from pair key {pair_id!r}")
    a, b = pair_id.split("_vs_", 1)
    a = a.strip()
    b = b.strip()
    if not a or not b or a == b:
        raise ValueError(f"Invalid pair key format {pair_id!r}")
    return a, b


def normalize_pair_record(pair_id: str, payload: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    run_id_a, run_id_b = _extract_run_ids(pair_id, payload)

    model_a = payload.get("model_a") or {}
    model_b = payload.get("model_b") or {}

    a_wins = _as_int(model_a.get("wins", 0), field_name=f"{pair_id}.model_a.wins")
    a_losses = _as_int(model_a.get("losses", 0), field_name=f"{pair_id}.model_a.losses")
    a_draws = _as_int(model_a.get("draws", 0), field_name=f"{pair_id}.model_a.draws")

    b_wins = _as_int(model_b.get("wins", 0), field_name=f"{pair_id}.model_b.wins")
    b_losses = _as_int(model_b.get("losses", 0), field_name=f"{pair_id}.model_b.losses")
    b_draws = _as_int(model_b.get("draws", 0), field_name=f"{pair_id}.model_b.draws")

    games_total = _as_int(payload.get("games_total", a_wins + a_losses + a_draws), field_name=f"{pair_id}.games_total")

    if not (a_wins == b_losses and a_losses == b_wins and a_draws == b_draws):
        raise ValueError(
            f"Pair {pair_id!r} in {source_path} has asymmetric model_a/model_b counts: "
            f"a=({a_wins},{a_losses},{a_draws}) b=({b_wins},{b_losses},{b_draws})"
        )

    if a_wins + a_losses + a_draws != games_total:
        raise ValueError(
            f"Pair {pair_id!r} in {source_path} has inconsistent model_a totals: "
            f"{a_wins}+{a_losses}+{a_draws} != games_total={games_total}"
        )
    if b_wins + b_losses + b_draws != games_total:
        raise ValueError(
            f"Pair {pair_id!r} in {source_path} has inconsistent model_b totals: "
            f"{b_wins}+{b_losses}+{b_draws} != games_total={games_total}"
        )

    # Canonical key to dedupe independent of input orientation.
    if run_id_a < run_id_b:
        model_1, model_2 = run_id_a, run_id_b
        wins_1 = a_wins
        losses_1 = a_losses
        draws = a_draws
    else:
        model_1, model_2 = run_id_b, run_id_a
        wins_1 = b_wins
        losses_1 = b_losses
        draws = b_draws

    score_1 = float(wins_1) + 0.5 * float(draws)
    winrate_1 = score_1 / float(games_total) if games_total > 0 else float("nan")

    return {
        "model_1": model_1,
        "model_2": model_2,
        "games_total": games_total,
        "wins_1": wins_1,
        "losses_1": losses_1,
        "draws": draws,
        "score_1": score_1,
        "winrate_1": winrate_1,
        "source_files": [str(source_path)],
        "source_pair_ids": [pair_id],
    }


def load_and_aggregate(paths: Sequence[Path]) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Any]]:
    pair_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    config_run_ids: set[str] = set()
    config_games_per_pair_values: List[int] = []
    duplicate_identical_pairs: List[Tuple[str, str]] = []
    configs_seen = 0

    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cfg = raw.get("config") or {}
        configs_seen += 1

        run_ids = cfg.get("run_ids")
        if isinstance(run_ids, list):
            for rid in run_ids:
                s = str(rid).strip()
                if s:
                    config_run_ids.add(s)

        if cfg.get("games_per_pair") is not None:
            config_games_per_pair_values.append(_as_int(cfg["games_per_pair"], field_name=f"{path}:config.games_per_pair"))

        by_pair = ((raw.get("results") or {}).get("by_pair")) or {}
        if not isinstance(by_pair, dict):
            raise ValueError(f"{path} has invalid results.by_pair payload (expected object)")

        for pair_id, payload_any in by_pair.items():
            if not isinstance(payload_any, dict):
                raise ValueError(f"{path} pair {pair_id!r} must be an object")
            rec = normalize_pair_record(str(pair_id), payload_any, path)
            key = (rec["model_1"], rec["model_2"])

            if key not in pair_map:
                pair_map[key] = rec
                continue

            existing = pair_map[key]
            same_counts = (
                existing["games_total"] == rec["games_total"]
                and existing["wins_1"] == rec["wins_1"]
                and existing["losses_1"] == rec["losses_1"]
                and existing["draws"] == rec["draws"]
            )
            if not same_counts:
                raise ValueError(
                    f"Conflicting duplicate pair {key[0]} vs {key[1]} across files.\n"
                    f"Existing: games={existing['games_total']}, "
                    f"wins_1={existing['wins_1']}, losses_1={existing['losses_1']}, draws={existing['draws']}\n"
                    f"New: games={rec['games_total']}, "
                    f"wins_1={rec['wins_1']}, losses_1={rec['losses_1']}, draws={rec['draws']}\n"
                    f"New source: {path}"
                )
            existing["source_files"].extend(rec["source_files"])
            existing["source_pair_ids"].extend(rec["source_pair_ids"])
            duplicate_identical_pairs.append(key)

    meta = {
        "config_run_ids": sorted(config_run_ids),
        "config_games_per_pair_values": sorted(config_games_per_pair_values),
        "duplicate_identical_pairs": sorted(set(duplicate_identical_pairs)),
        "configs_seen": int(configs_seen),
    }
    return pair_map, meta


def build_checks(
    *,
    pair_map: Dict[Tuple[str, str], Dict[str, Any]],
    aggregate_meta: Dict[str, Any],
    expected_model_count_arg: int | None,
    expected_games_per_pair_arg: int | None,
) -> Dict[str, Any]:
    observed_models = sorted({m for pair in pair_map.keys() for m in pair})
    run_ids_from_config = list(aggregate_meta.get("config_run_ids", []))
    run_ids_set = set(run_ids_from_config)

    if run_ids_from_config:
        model_universe = sorted(run_ids_set)
    else:
        model_universe = observed_models

    observed_set = set(observed_models)
    missing_models_from_config = sorted(set(model_universe) - observed_set)
    extra_models_not_in_config = sorted(observed_set - set(model_universe)) if run_ids_from_config else []

    inferred_model_count = len(model_universe)
    expected_model_count = expected_model_count_arg if expected_model_count_arg is not None else inferred_model_count
    model_count_ok = len(observed_models) == expected_model_count

    # Build expected pair universe when model ids are known.
    expected_pair_set: set[Tuple[str, str]] = set()
    missing_pairs: List[str] = []
    extra_pairs: List[str] = []
    if model_universe:
        expected_pair_set = {(a, b) for a, b in combinations(sorted(model_universe), 2)}
        observed_pair_set = set(pair_map.keys())
        missing_pairs = [f"{a}_vs_{b}" for a, b in sorted(expected_pair_set - observed_pair_set)]
        extra_pairs = [f"{a}_vs_{b}" for a, b in sorted(observed_pair_set - expected_pair_set)]
    else:
        observed_pair_set = set(pair_map.keys())

    expected_pair_count = expected_model_count * (expected_model_count - 1) // 2
    observed_pair_count = len(pair_map)
    pair_count_ok = observed_pair_count == expected_pair_count

    config_games_per_pair_values = list(aggregate_meta.get("config_games_per_pair_values", []))
    expected_games_per_pair = expected_games_per_pair_arg
    if expected_games_per_pair is None and config_games_per_pair_values:
        uniq = sorted(set(config_games_per_pair_values))
        if len(uniq) == 1:
            expected_games_per_pair = uniq[0]

    observed_games_per_pair_values = sorted({int(rec["games_total"]) for rec in pair_map.values()})
    games_per_pair_ok = True
    pairs_with_unexpected_games: List[str] = []
    if expected_games_per_pair is not None:
        for (a, b), rec in sorted(pair_map.items()):
            if int(rec["games_total"]) != int(expected_games_per_pair):
                games_per_pair_ok = False
                pairs_with_unexpected_games.append(f"{a}_vs_{b}:{rec['games_total']}")

    config_games_per_pair_consistent = len(set(config_games_per_pair_values)) <= 1 if config_games_per_pair_values else True

    hard_checks_ok = (
        model_count_ok
        and pair_count_ok
        and games_per_pair_ok
        and (len(missing_pairs) == 0)
        and (len(extra_pairs) == 0)
        and (len(missing_models_from_config) == 0)
        and (len(extra_models_not_in_config) == 0)
        and config_games_per_pair_consistent
    )

    return {
        "observed_models": observed_models,
        "model_universe": sorted(model_universe),
        "expected_model_count": int(expected_model_count),
        "observed_model_count": int(len(observed_models)),
        "missing_models_from_config": missing_models_from_config,
        "extra_models_not_in_config": extra_models_not_in_config,
        "expected_pair_count": int(expected_pair_count),
        "observed_pair_count": int(observed_pair_count),
        "missing_pairs": missing_pairs,
        "extra_pairs": extra_pairs,
        "expected_games_per_pair": expected_games_per_pair,
        "observed_games_per_pair_values": observed_games_per_pair_values,
        "pairs_with_unexpected_games": pairs_with_unexpected_games,
        "config_games_per_pair_values": sorted(config_games_per_pair_values),
        "config_games_per_pair_consistent": bool(config_games_per_pair_consistent),
        "model_count_ok": bool(model_count_ok),
        "pair_count_ok": bool(pair_count_ok),
        "games_per_pair_ok": bool(games_per_pair_ok),
        "hard_checks_ok": bool(hard_checks_ok),
        "observed_pair_set_size": int(len(observed_pair_set)),
        "expected_pair_set_size": int(len(expected_pair_set)),
    }


def _logistic_expectation(delta_elo: np.ndarray) -> np.ndarray:
    x = np.clip(LOGISTIC_K * delta_elo, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _elo_nll(ratings: np.ndarray, pair_rows: Sequence[Tuple[int, int, float, float]]) -> float:
    # pair row format: (i, j, games, score_i)
    if not pair_rows:
        return 0.0
    nll = 0.0
    for i, j, games, score_i in pair_rows:
        p = float(_logistic_expectation(np.array([ratings[i] - ratings[j]]))[0])
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        nll -= score_i * math.log(p) + (games - score_i) * math.log(1.0 - p)
    return float(nll)


def fit_global_elo(
    *,
    model_ids: Sequence[str],
    pair_map: Dict[Tuple[str, str], Dict[str, Any]],
    anchor: float,
    max_iter: int,
    tol: float,
    damping: float,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, np.ndarray]:
    m = len(model_ids)
    ratings = np.full(m, float(anchor), dtype=np.float64)
    idx = {model: i for i, model in enumerate(model_ids)}

    pair_rows: List[Tuple[int, int, float, float]] = []
    games_by_model = np.zeros(m, dtype=np.float64)
    score_by_model = np.zeros(m, dtype=np.float64)

    for (model_1, model_2), rec in pair_map.items():
        i = idx[model_1]
        j = idx[model_2]
        games = float(rec["games_total"])
        score_i = float(rec["score_1"])
        pair_rows.append((i, j, games, score_i))
        games_by_model[i] += games
        games_by_model[j] += games
        score_by_model[i] += score_i
        score_by_model[j] += (games - score_i)

    if not pair_rows:
        meta = {
            "iterations": 0,
            "converged": True,
            "objective_nll": 0.0,
            "max_abs_update": 0.0,
            "line_search_steps": 0,
        }
        return ratings, meta, games_by_model, score_by_model

    converged = False
    max_abs_update = float("inf")
    line_search_steps_total = 0
    objective = _elo_nll(ratings, pair_rows)

    for it in range(1, int(max_iter) + 1):
        g = np.zeros(m, dtype=np.float64)
        h = np.zeros((m, m), dtype=np.float64)

        for i, j, games, score_i in pair_rows:
            p = float(_logistic_expectation(np.array([ratings[i] - ratings[j]]))[0])
            p = min(max(p, 1e-12), 1.0 - 1e-12)
            grad = LOGISTIC_K * (games * p - score_i)
            w = (LOGISTIC_K * LOGISTIC_K) * games * p * (1.0 - p)

            g[i] += grad
            g[j] -= grad

            h[i, i] += w
            h[j, j] += w
            h[i, j] -= w
            h[j, i] -= w

        h.flat[:: m + 1] += float(damping)

        try:
            delta = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(h, g, rcond=None)[0]

        step = 1.0
        accepted = False
        trial_updates = 0
        while step >= 1e-6:
            cand = ratings - step * delta
            cand += float(anchor) - float(cand.mean())
            cand_obj = _elo_nll(cand, pair_rows)
            trial_updates += 1
            if cand_obj <= objective + 1e-12:
                accepted = True
                break
            step *= 0.5

        line_search_steps_total += trial_updates

        if not accepted:
            cand = ratings - 1e-3 * delta
            cand += float(anchor) - float(cand.mean())
            cand_obj = _elo_nll(cand, pair_rows)

        max_abs_update = float(np.max(np.abs(cand - ratings)))
        ratings = cand
        objective = cand_obj

        if max_abs_update <= float(tol):
            converged = True
            break

    meta = {
        "iterations": int(it),
        "converged": bool(converged),
        "objective_nll": float(objective),
        "max_abs_update": float(max_abs_update),
        "line_search_steps": int(line_search_steps_total),
    }
    return ratings, meta, games_by_model, score_by_model


def build_pairwise_results_df(
    *,
    pair_map: Dict[Tuple[str, str], Dict[str, Any]],
    elo_by_model: Dict[str, float],
    rank_by_model: Dict[str, int],
    implied_elo_epsilon: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (model_1, model_2), rec in sorted(pair_map.items()):
        games = int(rec["games_total"])
        score_1 = float(rec["score_1"])
        score_2 = float(games) - score_1
        winrate_1 = score_1 / float(games) if games > 0 else float("nan")
        winrate_2 = score_2 / float(games) if games > 0 else float("nan")
        p_clip = float(np.clip(winrate_1, implied_elo_epsilon, 1.0 - implied_elo_epsilon))
        implied_diff = 400.0 * math.log10(p_clip / (1.0 - p_clip))

        rows.append(
            {
                "model_1": model_1,
                "model_2": model_2,
                "rank_1": int(rank_by_model.get(model_1, -1)),
                "rank_2": int(rank_by_model.get(model_2, -1)),
                "games_total": games,
                "wins_1": int(rec["wins_1"]),
                "draws": int(rec["draws"]),
                "losses_1": int(rec["losses_1"]),
                "score_1": score_1,
                "score_2": score_2,
                "winrate_1": winrate_1,
                "winrate_2": winrate_2,
                "implied_elo_diff_1_minus_2": implied_diff,
                "global_elo_1": float(elo_by_model.get(model_1, float("nan"))),
                "global_elo_2": float(elo_by_model.get(model_2, float("nan"))),
                "global_elo_diff_1_minus_2": float(elo_by_model.get(model_1, 0.0) - elo_by_model.get(model_2, 0.0)),
                "source_files": ";".join(sorted(set(rec.get("source_files", [])))),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["rank_1", "rank_2", "model_1", "model_2"], kind="mergesort").reset_index(drop=True)
    return out


def build_winrate_matrix(
    *,
    model_order: Sequence[str],
    pair_map: Dict[Tuple[str, str], Dict[str, Any]],
) -> pd.DataFrame:
    m = len(model_order)
    matrix = np.full((m, m), np.nan, dtype=np.float64)
    idx = {model: i for i, model in enumerate(model_order)}
    for i in range(m):
        matrix[i, i] = 0.5

    for (model_1, model_2), rec in pair_map.items():
        i = idx.get(model_1)
        j = idx.get(model_2)
        if i is None or j is None:
            continue
        games = float(rec["games_total"])
        if games <= 0:
            continue
        score_1 = float(rec["score_1"])
        winrate_1 = score_1 / games
        matrix[i, j] = winrate_1
        matrix[j, i] = 1.0 - winrate_1

    return pd.DataFrame(matrix, index=list(model_order), columns=list(model_order))


def build_global_elo_diff_matrix(
    *,
    model_order: Sequence[str],
    elo_by_model: Dict[str, float],
) -> pd.DataFrame:
    m = len(model_order)
    out = np.full((m, m), np.nan, dtype=np.float64)
    for i, model_i in enumerate(model_order):
        elo_i = float(elo_by_model[model_i])
        for j, model_j in enumerate(model_order):
            elo_j = float(elo_by_model[model_j])
            out[i, j] = elo_i - elo_j
    return pd.DataFrame(out, index=list(model_order), columns=list(model_order))


def save_heatmap(
    matrix_df: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    cmap: str,
    cbar_label: str,
    fmt: str,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    dpi: int = 220,
) -> None:
    n = matrix_df.shape[0]
    figsize = (max(10.0, 0.62 * n + 4.0), max(8.0, 0.56 * n + 3.0))
    fig, ax = plt.subplots(figsize=figsize)
    data = matrix_df.to_numpy(dtype=np.float64)
    mask = np.isnan(data)
    annot = n <= 16

    if sns is not None:
        sns.set_theme(style="ticks", context="notebook")
        sns.heatmap(
            matrix_df,
            ax=ax,
            cmap=cmap,
            mask=mask,
            vmin=vmin,
            vmax=vmax,
            center=center,
            annot=annot,
            fmt=fmt if annot else "",
            annot_kws={"fontsize": 8},
            linewidths=0.5,
            linecolor="#f2f2f2",
            cbar_kws={"label": cbar_label},
        )
        if annot:
            valid = data[~mask]
            lo = float(vmin) if vmin is not None else float(np.nanmin(valid))
            hi = float(vmax) if vmax is not None else float(np.nanmax(valid))
            span = max(hi - lo, 1e-12)
            text_k = 0
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if mask[i, j]:
                        continue
                    val = float(data[i, j])
                    norm = (val - lo) / span
                    txt = ax.texts[text_k]
                    text_k += 1
                    if center is None:
                        txt.set_color("white" if norm > 0.62 else "#0f172a")
                    else:
                        c_norm = (float(center) - lo) / span
                        txt.set_color("white" if abs(norm - c_norm) > 0.22 else "#0f172a")
    else:
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(matrix_df.columns.tolist())
        ax.set_yticklabels(matrix_df.index.tolist())
        if annot:
            for i in range(n):
                for j in range(n):
                    value = data[i, j]
                    if np.isnan(value):
                        continue
                    text_color = "white" if (vmax is not None and vmin is not None and value > (vmin + vmax) / 2.0) else "#0f172a"
                    ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8, color=text_color)

    ax.set_facecolor("#fbfbfc")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Opponent (column)")
    ax.set_ylabel("Model (row)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi))
    plt.close(fig)


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def main() -> None:
    args = parse_args()

    input_path = args.input_path.resolve()
    summary_paths, discovered_layout = discover_summary_paths(input_path, args.input_layout)

    output_dir = (args.output_dir.resolve() if args.output_dir is not None else (input_path if input_path.is_dir() else input_path.parent))
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_map, aggregate_meta = load_and_aggregate(summary_paths)
    checks = build_checks(
        pair_map=pair_map,
        aggregate_meta=aggregate_meta,
        expected_model_count_arg=args.expected_model_count,
        expected_games_per_pair_arg=args.expected_games_per_pair,
    )

    if not pair_map:
        raise RuntimeError("No pairs found after parsing input summaries.")

    if (not checks["hard_checks_ok"]) and (not args.allow_incomplete):
        raise RuntimeError(
            "Completeness checks failed. Re-run with --allow-incomplete to continue.\n"
            f"Missing pairs: {len(checks['missing_pairs'])}, extra pairs: {len(checks['extra_pairs'])}, "
            f"model_count_ok={checks['model_count_ok']}, games_per_pair_ok={checks['games_per_pair_ok']}, "
            f"pair_count_ok={checks['pair_count_ok']}"
        )

    model_ids = checks["model_universe"] if checks["model_universe"] else checks["observed_models"]
    ratings, elo_fit_meta, games_by_model, score_by_model = fit_global_elo(
        model_ids=model_ids,
        pair_map=pair_map,
        anchor=float(args.elo_anchor),
        max_iter=int(args.elo_max_iter),
        tol=float(args.elo_tol),
        damping=float(args.elo_damping),
    )

    elo_df = pd.DataFrame(
        {
            "model": model_ids,
            "elo": ratings.astype(float),
            "games": games_by_model.astype(int),
            "score": score_by_model.astype(float),
        }
    )
    elo_df["winrate"] = np.where(elo_df["games"] > 0, elo_df["score"] / elo_df["games"], np.nan)
    elo_df = elo_df.sort_values(["elo", "model"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    elo_df.insert(0, "rank", np.arange(1, len(elo_df) + 1, dtype=int))

    sorted_models = elo_df["model"].tolist()
    elo_by_model = {r["model"]: float(r["elo"]) for _, r in elo_df.iterrows()}
    rank_by_model = {r["model"]: int(r["rank"]) for _, r in elo_df.iterrows()}

    pairwise_df = build_pairwise_results_df(
        pair_map=pair_map,
        elo_by_model=elo_by_model,
        rank_by_model=rank_by_model,
        implied_elo_epsilon=float(args.implied_elo_epsilon),
    )
    winrate_df = build_winrate_matrix(model_order=sorted_models, pair_map=pair_map)
    elo_diff_df = build_global_elo_diff_matrix(model_order=sorted_models, elo_by_model=elo_by_model)

    total_games = int(sum(int(rec["games_total"]) for rec in pair_map.values()))
    total_score_points = float(sum(float(rec["score_1"]) for rec in pair_map.values()))
    # Each pair contributes games_total points in total across both players.
    total_points_all_players = float(total_games)

    artifacts = {
        "elo_ratings_csv": output_dir / "elo_ratings.csv",
        "pairwise_results_csv": output_dir / "pairwise_results.csv",
        "winrate_matrix_csv": output_dir / "winrate_matrix.csv",
        "elo_diff_matrix_csv": output_dir / "elo_diff_matrix.csv",
        "elo_summary_json": output_dir / "elo_summary.json",
        "winrate_matrix_png": output_dir / "winrate_matrix.png",
        "elo_diff_matrix_png": output_dir / "elo_diff_matrix.png",
    }

    elo_df.to_csv(artifacts["elo_ratings_csv"], index=False, float_format="%.6f")
    pairwise_df.to_csv(artifacts["pairwise_results_csv"], index=False, float_format="%.6f")
    winrate_df.to_csv(artifacts["winrate_matrix_csv"], index=True, index_label="model", float_format="%.6f")
    elo_diff_df.to_csv(artifacts["elo_diff_matrix_csv"], index=True, index_label="model", float_format="%.6f")

    max_abs_diff = float(np.nanmax(np.abs(elo_diff_df.to_numpy(dtype=np.float64)))) if not elo_diff_df.empty else 0.0
    diff_lim = max(100.0, math.ceil(max_abs_diff / 50.0) * 50.0)

    display_labels = [f"{i:02d} {m}" for i, m in enumerate(sorted_models, start=1)]
    plot_winrate_df = winrate_df.copy()
    plot_winrate_df.index = display_labels
    plot_winrate_df.columns = display_labels
    plot_elo_diff_df = elo_diff_df.copy()
    plot_elo_diff_df.index = display_labels
    plot_elo_diff_df.columns = display_labels

    save_heatmap(
        plot_winrate_df,
        title="Pairwise Winrate Matrix (Row Score vs Column)",
        output_path=artifacts["winrate_matrix_png"],
        cmap="RdYlGn",
        cbar_label="Score Rate",
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        center=0.5,
        dpi=int(args.dpi),
    )
    save_heatmap(
        plot_elo_diff_df,
        title="Global Elo Difference Matrix (Row Elo - Column Elo)",
        output_path=artifacts["elo_diff_matrix_png"],
        cmap="RdBu_r",
        cbar_label="Elo Difference",
        fmt=".0f",
        vmin=-diff_lim,
        vmax=diff_lim,
        center=0.0,
        dpi=int(args.dpi),
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "input": {
            "input_path": str(input_path),
            "discovered_layout": discovered_layout,
            "summary_files": [str(p) for p in summary_paths],
            "summary_files_count": len(summary_paths),
        },
        "analysis_config": {
            "expected_model_count_arg": args.expected_model_count,
            "expected_games_per_pair_arg": args.expected_games_per_pair,
            "allow_incomplete": bool(args.allow_incomplete),
            "elo_anchor": float(args.elo_anchor),
            "elo_max_iter": int(args.elo_max_iter),
            "elo_tol": float(args.elo_tol),
            "elo_damping": float(args.elo_damping),
            "implied_elo_epsilon": float(args.implied_elo_epsilon),
        },
        "aggregate_meta": aggregate_meta,
        "checks": checks,
        "global_stats": {
            "model_count": int(len(model_ids)),
            "pair_count": int(len(pair_map)),
            "total_games": int(total_games),
            "total_pair_score_points_model_1_side": float(total_score_points),
            "total_points_all_players": float(total_points_all_players),
            "matrix_shape": [int(winrate_df.shape[0]), int(winrate_df.shape[1])],
        },
        "elo_fit": elo_fit_meta,
        "elo_matrix_definition": "global_elo_row_minus_column",
        "elo_ranking": elo_df.to_dict(orient="records"),
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }
    artifacts["elo_summary_json"].write_text(json.dumps(_to_serializable(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Loaded {len(summary_paths)} summaries from {input_path}")
    print(f"Models: {len(model_ids)} | Pairs: {len(pair_map)} | Games: {total_games}")
    print("Top Elo ranking:")
    for _, row in elo_df.iterrows():
        print(
            f"  {int(row['rank']):2d}. {row['model']}  "
            f"Elo={float(row['elo']):.2f}  Games={int(row['games'])}  Winrate={float(row['winrate']):.4f}"
        )
    print("Wrote artifacts:")
    for key in [
        "elo_ratings_csv",
        "pairwise_results_csv",
        "winrate_matrix_csv",
        "elo_diff_matrix_csv",
        "elo_summary_json",
        "winrate_matrix_png",
        "elo_diff_matrix_png",
    ]:
        print(f"  - {artifacts[key]}")


if __name__ == "__main__":
    main()
