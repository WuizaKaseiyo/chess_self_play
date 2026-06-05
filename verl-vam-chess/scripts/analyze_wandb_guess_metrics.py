#!/usr/bin/env python3
"""
Analyze "guess-first" behavior for a W&B chess RL run.

This script is intentionally self-contained:
- Downloads the run config + rollout/validation JSONLs via the W&B API (with a local cache)
- Parses `<guess>...</guess>` and `<uci_move>...</uci_move>` from raw model outputs
- Joins samples with per-position `move_values_json` from the dataset to score guess/final moves
- Produces step-wise plots and logs them back to W&B (optional)

Example:
  python3 scripts/analyze_wandb_guess_metrics.py \\
    --entity gabr1e11 --project chess_rl --run x2futx5p \\
    --evidence-root analysis/wandb_evidence \\
    --step-bin-size 5 \\
    --log-to-wandb
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import wandb
import yaml

# Ensure local imports resolve when the script is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipe.chess.reward_fn import _extract_guess_then_uci_move, _parse_move_values_json, _to_uci


_FEN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Current FEN string:\s*(?P<fen>[^\r\n]+)", flags=re.IGNORECASE),
    re.compile(r"FEN string:\s*(?P<fen>[^\r\n]+)", flags=re.IGNORECASE),
    re.compile(r"\bFEN:\s*(?P<fen>[^\r\n]+)", flags=re.IGNORECASE),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _yaml_dump(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(obj, sort_keys=True, allow_unicode=True), encoding="utf-8")


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _parse_fen_from_prompt_text(prompt_text: str) -> Optional[str]:
    if not prompt_text:
        return None
    for pat in _FEN_PATTERNS:
        m = pat.search(prompt_text)
        if m:
            fen = str(m.group("fen") or "").strip()
            return fen or None
    return None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSONL {path} at line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise TypeError(f"Expected dict JSONL row in {path} line {line_no}, got {type(obj)}")
            yield obj


def _sorted_jsonl_files(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    files = [p for p in dir_path.glob("*.jsonl") if p.is_file()]

    def _key(p: Path) -> tuple[int, str]:
        m = re.search(r"(?P<num>\d+)\.jsonl$", p.name)
        if m:
            return int(m.group("num")), p.name
        return (1 << 60), p.name

    return sorted(files, key=_key)


def _mean_ci95(values: np.ndarray) -> tuple[float, float, float, int]:
    """Return (mean, lo, hi, n) for finite values, using a normal-approx CI."""
    finite = values[np.isfinite(values)]
    n = int(finite.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(finite.mean())
    if n == 1:
        return mean, float("nan"), float("nan"), 1
    std = float(finite.std(ddof=1))
    se = std / math.sqrt(n)
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se
    return mean, lo, hi, n


def _maybe_download_wandb_file(
    *,
    run: Any,
    file_name: str,
    dst_root: Path,
    expected_size: int | None,
    max_retries: int,
    sleep_s: float,
) -> Path:
    local_path = dst_root / file_name
    if local_path.exists() and local_path.is_file():
        if expected_size is None or local_path.stat().st_size == int(expected_size):
            return local_path

    _ensure_dir(local_path.parent)

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            run.file(file_name).download(root=str(dst_root), replace=True)
            if expected_size is not None and local_path.exists():
                got = local_path.stat().st_size
                if got != int(expected_size):
                    raise RuntimeError(f"Downloaded size mismatch for {file_name}: got={got} expected={expected_size}")
            return local_path
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                break
            time.sleep(sleep_s * (2 ** (attempt - 1)))
    raise RuntimeError(f"Failed to download W&B file {file_name} after {max_retries} attempts: {last_err}") from last_err


@dataclass(frozen=True)
class FenMoveMaps:
    move_values_by_fen: dict[str, dict[str, float]]


def _load_move_values_by_fen(parquet_path: str) -> FenMoveMaps:
    dataset = ds.dataset(parquet_path, format="parquet")
    table = dataset.to_table(columns=["reward_model"])
    rows = table.to_pylist()
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        rm = row.get("reward_model") or {}
        fen = str(rm.get("fen") or "").strip()
        if not fen:
            continue
        out[fen] = _parse_move_values_json(rm.get("move_values_json"))
    return FenMoveMaps(move_values_by_fen=out)


def _bin_step(step: int, bin_size: int) -> int:
    if bin_size <= 1:
        return int(step)
    return int(((int(step) - 1) // int(bin_size)) * int(bin_size) + 1)


def _plot_save(path: Path, title: str) -> None:
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=12)
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--run", required=True, help="W&B run id (e.g., x2futx5p)")
    ap.add_argument("--evidence-root", default="analysis/wandb_evidence")
    ap.add_argument("--step-bin-size", type=int, default=5)
    ap.add_argument("--max-download-retries", type=int, default=6)
    ap.add_argument("--download-sleep-s", type=float, default=2.0)
    ap.add_argument(
        "--limit-steps",
        type=int,
        default=None,
        help="Optional: only analyze the first N rollout steps (for smoke tests).",
    )
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--log-to-wandb", action="store_true")
    args = ap.parse_args()

    if args.step_bin_size <= 0:
        raise ValueError("--step-bin-size must be > 0")

    random.seed(int(args.random_seed))
    np.random.seed(int(args.random_seed))

    run_path = f"{args.entity}/{args.project}/{args.run}"
    evidence_root = Path(args.evidence_root)
    ev_dir = evidence_root / args.run
    files_dir = ev_dir / "files"
    out_dir = ev_dir / "guess_analysis"
    plots_dir = out_dir / "plots"
    _ensure_dir(out_dir)
    _ensure_dir(plots_dir)

    api = wandb.Api()
    run = api.run(run_path)

    meta = {
        "downloaded_at": _now_iso(),
        "run_path": run_path,
        "run_id": run.id,
        "run_name": run.name,
        "run_state": run.state,
        "created_at": getattr(run, "created_at", None),
        "url": getattr(run, "url", None),
        "commit": getattr(run, "commit", None),
        "wandb_library": getattr(wandb, "__version__", None),
    }
    _json_dump(ev_dir / "run_meta.json", meta)

    config_api = dict(run.config or {})
    _json_dump(ev_dir / "config_api.json", config_api)
    _yaml_dump(ev_dir / "config_api.yaml", config_api)

    # Small root files: always download (they're tiny and useful for provenance).
    root_files = [
        "config.yaml",
        "config_hash.json",
        "output.log",
        "requirements.txt",
        "wandb-metadata.json",
        "wandb-summary.json",
    ]
    all_files = list(run.files())
    size_by_name = {f.name: getattr(f, "size", None) for f in all_files}
    for name in root_files:
        if name not in size_by_name:
            continue
        _maybe_download_wandb_file(
            run=run,
            file_name=name,
            dst_root=files_dir,
            expected_size=size_by_name.get(name),
            max_retries=int(args.max_download_retries),
            sleep_s=float(args.download_sleep_s),
        )

    # Download rollout + validation logs (needed for analysis).
    wanted_prefixes = ("rollout_logs/", "validation_logs/")
    wanted_files = [f for f in all_files if any(f.name.startswith(p) for p in wanted_prefixes)]
    wanted_manifest = [
        {"name": f.name, "size": getattr(f, "size", None), "md5": getattr(f, "md5", None)} for f in wanted_files
    ]
    _json_dump(ev_dir / "files_manifest_guess_analysis.json", {"downloaded_at": _now_iso(), "files": wanted_manifest})

    # Deterministic ordering so partial downloads are reproducible.
    for idx, wf in enumerate(sorted(wanted_files, key=lambda x: x.name), start=1):
        # If limit-steps is set, do not download beyond that rollout step (validation logs still downloaded).
        if args.limit_steps is not None and wf.name.startswith("rollout_logs/"):
            m = re.search(r"rollout_logs/(?P<num>\d+)\.jsonl$", wf.name)
            if m and int(m.group("num")) > int(args.limit_steps):
                continue

        _maybe_download_wandb_file(
            run=run,
            file_name=wf.name,
            dst_root=files_dir,
            expected_size=size_by_name.get(wf.name),
            max_retries=int(args.max_download_retries),
            sleep_s=float(args.download_sleep_s),
        )

        if idx % 50 == 0:
            print(f"[download] {idx}/{len(wanted_files)} files")

    # Download history (scan_history is cheap and helps validate the step axis).
    history_jsonl = ev_dir / "history.jsonl"
    if not history_jsonl.exists():
        n_rows = 0
        with history_jsonl.open("w", encoding="utf-8") as f:
            for row in run.scan_history(page_size=10000):
                n_rows += 1
                f.write(json.dumps(row, default=str) + "\n")
        _json_dump(ev_dir / "history_meta.json", {"rows": n_rows, "downloaded_at": _now_iso()})
        df_hist = pd.read_json(history_jsonl, lines=True)
        df_hist.to_parquet(ev_dir / "history.parquet", index=False)

    # Load dataset move maps for scoring guess/final moves.
    train_parquet = (
        str(((config_api.get("data") or {}).get("train_files") or "")).strip() if isinstance(config_api.get("data"), dict) else ""
    )
    val_parquet = (
        str(((config_api.get("data") or {}).get("val_files") or "")).strip() if isinstance(config_api.get("data"), dict) else ""
    )
    if not train_parquet:
        raise ValueError("Run config missing data.train_files; cannot join move_values_json.")
    if not Path(train_parquet).exists():
        raise FileNotFoundError(f"train_files not found on disk: {train_parquet}")

    train_maps = _load_move_values_by_fen(train_parquet)
    val_maps = _load_move_values_by_fen(val_parquet) if val_parquet and Path(val_parquet).exists() else FenMoveMaps({})

    # Parse rollout logs.
    rollout_dir = files_dir / "rollout_logs"
    rollout_files = _sorted_jsonl_files(rollout_dir)
    if args.limit_steps is not None:
        rollout_files = [p for p in rollout_files if int(re.sub(r"\D", "", p.stem) or 0) <= int(args.limit_steps)]
    if not rollout_files:
        raise FileNotFoundError(f"No rollout logs found under {rollout_dir}")

    records: list[dict[str, Any]] = []
    for fp in rollout_files:
        for obj in _iter_jsonl(fp):
            step = int(obj.get("step") or 0)
            if step <= 0:
                # Fall back to file stem if needed.
                m = re.search(r"(?P<num>\d+)$", fp.stem)
                step = int(m.group("num")) if m else 0

            uid = str(obj.get("uid") or "")
            output_text = str(obj.get("output") or "")
            input_text = str(obj.get("input") or "")
            forced = bool(obj.get("forced_prefix_is_forced") or False)

            fen = _parse_fen_from_prompt_text(input_text)
            move_map = None
            if fen is not None:
                move_map = train_maps.move_values_by_fen.get(fen) or val_maps.move_values_by_fen.get(fen)

            guess_payload, final_payload, strict_format_reward = _extract_guess_then_uci_move(output_text)
            guess_payload_str = guess_payload if guess_payload is not None else None
            final_payload_str = final_payload if final_payload is not None else None

            guess_uci = _to_uci(guess_payload_str or "") if guess_payload_str is not None else None
            final_uci = _to_uci(final_payload_str or "") if final_payload_str is not None else None
            guess_tag_present = guess_payload_str is not None
            uci_tag_present = final_payload_str is not None
            guess_uci_valid = guess_uci is not None
            final_uci_valid = final_uci is not None

            guess_value = float("nan")
            final_value = float("nan")
            if isinstance(move_map, dict):
                if guess_uci is not None:
                    guess_value = _safe_float(move_map.get(guess_uci))
                if final_uci is not None:
                    final_value = _safe_float(move_map.get(final_uci))

            # Parse failure taxonomy (best-effort, for analysis only).
            parse_reason = "ok"
            if not guess_tag_present:
                parse_reason = "missing_guess_tag"
            elif not uci_tag_present:
                parse_reason = "missing_uci_move_tag"
            elif not guess_uci_valid:
                parse_reason = "bad_guess_uci"
            elif final_uci is None:
                parse_reason = "bad_final_uci"

            rec = {
                "step": step,
                "step_bin": _bin_step(step, int(args.step_bin_size)),
                "uid": uid,
                "forced_prefix_is_forced": forced,
                "fen": fen or "",
                "has_move_values_map": bool(isinstance(move_map, dict) and len(move_map) > 0),
                "strict_format_reward": float(strict_format_reward),
                "guess_tag_present": bool(guess_tag_present),
                "uci_tag_present": bool(uci_tag_present),
                "guess_uci_valid": bool(guess_uci_valid),
                "final_uci_valid": bool(final_uci_valid),
                "guess_payload": guess_payload_str or "",
                "final_payload": final_payload_str or "",
                "guess_uci": guess_uci or "",
                "final_uci": final_uci or "",
                "follow_guess": bool(guess_uci is not None and final_uci is not None and guess_uci == final_uci),
                "guess_value": guess_value,
                "final_value": final_value,
                "guess_in_mapping": bool(isinstance(move_map, dict) and guess_uci is not None and guess_uci in move_map),
                "final_in_mapping": bool(isinstance(move_map, dict) and final_uci is not None and final_uci in move_map),
                "parse_reason": parse_reason,
                # Logged reward fields (for cross-checks).
                "logged_pred_move": str(obj.get("pred_move") or ""),
                "logged_guess_uci": str(obj.get("guess_uci") or ""),
                "logged_move_value": _safe_float(obj.get("move_value")),
                "logged_total_reward": _safe_float(obj.get("total_reward")),
                "logged_score": _safe_float(obj.get("score")),
                "logged_format_reward": _safe_float(obj.get("format_reward")),
            }
            records.append(rec)

    df = pd.DataFrame.from_records(records)
    df.to_parquet(out_dir / "rollout_records.parquet", index=False)
    df.to_csv(out_dir / "rollout_records.csv.gz", index=False, compression="gzip")

    # Sanity checks: compare our parser to logged fields.
    mismatch_guess = (df["guess_uci"].fillna("") != df["logged_guess_uci"].fillna("")).mean()
    mismatch_final = (df["final_uci"].fillna("") != df["logged_pred_move"].fillna("")).mean()
    _json_dump(
        out_dir / "parser_consistency.json",
        {
            "mismatch_guess_rate": float(mismatch_guess),
            "mismatch_final_rate": float(mismatch_final),
            "rows": int(len(df)),
        },
    )

    # Step-wise metrics (binned).
    def _metric_group_key(extra: str) -> list[str]:
        return ["step_bin"] + ([extra] if extra else [])

    def _agg_follow(g: pd.DataFrame) -> dict[str, Any]:
        usable = g[(g["guess_uci"] != "") & (g["final_uci"] != "")]
        n = int(len(usable))
        rate = float(usable["follow_guess"].mean()) if n else float("nan")
        return {"follow_guess_rate": rate, "follow_guess_n": n}

    def _agg_quality(g: pd.DataFrame) -> dict[str, Any]:
        gv = g["guess_value"].to_numpy(dtype=np.float64)
        fv = g["final_value"].to_numpy(dtype=np.float64)
        g_mean, g_lo, g_hi, g_n = _mean_ci95(gv)
        f_mean, f_lo, f_hi, f_n = _mean_ci95(fv)
        return {
            "guess_value_mean": g_mean,
            "guess_value_ci95_lo": g_lo,
            "guess_value_ci95_hi": g_hi,
            "guess_value_n": g_n,
            "final_value_mean": f_mean,
            "final_value_ci95_lo": f_lo,
            "final_value_ci95_hi": f_hi,
            "final_value_n": f_n,
        }

    def _agg_parse(g: pd.DataFrame) -> dict[str, Any]:
        n = int(len(g))
        strict_ok = float((g["strict_format_reward"] >= 1.0).mean()) if n else float("nan")
        missing_guess = float((g["parse_reason"] == "missing_guess_tag").mean()) if n else float("nan")
        missing_uci = float((g["parse_reason"] == "missing_uci_move_tag").mean()) if n else float("nan")
        bad_guess_uci = float((g["parse_reason"] == "bad_guess_uci").mean()) if n else float("nan")
        bad_final_uci = float((g["parse_reason"] == "bad_final_uci").mean()) if n else float("nan")
        map_missing = float((~g["has_move_values_map"]).mean()) if n else float("nan")
        guess_uci_valid_rate = float(g["guess_uci_valid"].mean()) if n else float("nan")
        final_uci_valid_rate = float(g["final_uci_valid"].mean()) if n else float("nan")
        return {
            "rows": n,
            "strict_format_ok_rate": strict_ok,
            "missing_guess_tag_rate": missing_guess,
            "missing_uci_move_tag_rate": missing_uci,
            "bad_guess_uci_rate": bad_guess_uci,
            "bad_final_uci_rate": bad_final_uci,
            "guess_uci_valid_rate": guess_uci_valid_rate,
            "final_uci_valid_rate": final_uci_valid_rate,
            "missing_move_values_map_rate": map_missing,
            "guess_missing_in_mapping_rate": float((~g["guess_in_mapping"]).mean()) if n else float("nan"),
            "final_missing_in_mapping_rate": float((~g["final_in_mapping"]).mean()) if n else float("nan"),
        }

    metric_rows: list[dict[str, Any]] = []
    for (step_bin, forced), g in df.groupby(["step_bin", "forced_prefix_is_forced"], sort=True):
        row = {"step_bin": int(step_bin), "forced_prefix_is_forced": bool(forced)}
        row.update(_agg_follow(g))
        row.update(_agg_quality(g))
        row.update(_agg_parse(g))
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows).sort_values(["step_bin", "forced_prefix_is_forced"])
    metrics_df.to_csv(out_dir / "metrics_by_step_bin.csv", index=False)
    metrics_df.to_parquet(out_dir / "metrics_by_step_bin.parquet", index=False)

    # Guess diversity (per uid group).
    gdf = (
        df.groupby(["step", "step_bin", "uid"], sort=False)
        .agg(
            rollouts=("uid", "size"),
            unique_guess_uci=("guess_uci", lambda s: int(pd.Series([x for x in s.tolist() if x]).nunique())),
        )
        .reset_index()
    )
    div_rows: list[dict[str, Any]] = []
    for step_bin, g in gdf.groupby("step_bin", sort=True):
        vals = g["unique_guess_uci"].to_numpy(dtype=np.float64)
        mean_u = float(np.mean(vals)) if len(vals) else float("nan")
        med_u = float(np.median(vals)) if len(vals) else float("nan")
        div_rows.append(
            {
                "step_bin": int(step_bin),
                "groups": int(len(g)),
                "rollouts_per_group_mean": float(g["rollouts"].mean()) if len(g) else float("nan"),
                "unique_guess_mean": mean_u,
                "unique_guess_median": med_u,
            }
        )
    div_df = pd.DataFrame(div_rows).sort_values("step_bin")
    div_df.to_csv(out_dir / "guess_diversity_by_step_bin.csv", index=False)
    div_df.to_parquet(out_dir / "guess_diversity_by_step_bin.parquet", index=False)

    # Save a small random sample for manual inspection (early/mid/late).
    by_step = sorted(df["step"].unique().tolist())
    if by_step:
        early = [s for s in by_step if s <= by_step[0] + 5]
        late = [s for s in by_step if s >= by_step[-1] - 5]
        mid_center = by_step[len(by_step) // 2]
        mid = [s for s in by_step if abs(s - mid_center) <= 2]

        def _sample_steps(steps: list[int], k: int) -> pd.DataFrame:
            sub = df[df["step"].isin(steps)]
            if len(sub) <= k:
                return sub
            return sub.sample(n=k, random_state=int(args.random_seed))

        sample_df = pd.concat(
            [
                _sample_steps(early, 8),
                _sample_steps(mid, 8),
                _sample_steps(late, 8),
            ],
            ignore_index=True,
        )
        sample_df.to_csv(out_dir / "samples_for_manual_check.csv", index=False)

    # ----------------
    # Plots
    # ----------------
    step_label = (
        "train step = RayPPOTrainer.training/global_step (1-indexed), from rollout_logs/*.jsonl field `step`"
    )

    # Follow-guess rate.
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    for forced in [False, True]:
        sub = metrics_df[metrics_df["forced_prefix_is_forced"] == forced]
        if sub.empty:
            continue
        ax.plot(
            sub["step_bin"],
            sub["follow_guess_rate"],
            label=("forced" if forced else "free"),
            linewidth=1.8,
        )
    ax.set_xlabel(f"step_bin (bin_size={int(args.step_bin_size)})")
    ax.set_ylabel("P(final_move == guess_move)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    follow_path = plots_dir / "follow_guess_rate.png"
    _plot_save(follow_path, f"Follow-guess rate vs step\n({step_label})")

    # Move quality (move_values_json): guess vs final.
    # Use the "free" rows by default if forced rows exist; else fall back to whatever exists.
    base = metrics_df[metrics_df["forced_prefix_is_forced"] == False]  # noqa: E712
    if base.empty:
        base = metrics_df

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    x = base["step_bin"].to_numpy()

    y_g = base["guess_value_mean"].to_numpy(dtype=np.float64)
    y_g_lo = base["guess_value_ci95_lo"].to_numpy(dtype=np.float64)
    y_g_hi = base["guess_value_ci95_hi"].to_numpy(dtype=np.float64)
    ax.plot(x, y_g, label="guess move value", linewidth=1.8)
    if np.isfinite(y_g_lo).any() and np.isfinite(y_g_hi).any():
        ax.fill_between(x, y_g_lo, y_g_hi, alpha=0.15)

    y_f = base["final_value_mean"].to_numpy(dtype=np.float64)
    y_f_lo = base["final_value_ci95_lo"].to_numpy(dtype=np.float64)
    y_f_hi = base["final_value_ci95_hi"].to_numpy(dtype=np.float64)
    ax.plot(x, y_f, label="final move value", linewidth=1.8)
    if np.isfinite(y_f_lo).any() and np.isfinite(y_f_hi).any():
        ax.fill_between(x, y_f_lo, y_f_hi, alpha=0.15)

    ax.set_xlabel(f"step_bin (bin_size={int(args.step_bin_size)})")
    ax.set_ylabel("move value in [0,1] (from move_values_json)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    quality_path = plots_dir / "move_quality_guess_vs_final.png"
    _plot_save(quality_path, f"Move quality vs step\n({step_label})")

    # Guess diversity.
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.plot(div_df["step_bin"], div_df["unique_guess_mean"], label="mean #unique(guess_uci) per uid", linewidth=1.8)
    ax.plot(
        div_df["step_bin"],
        div_df["unique_guess_median"],
        label="median #unique(guess_uci) per uid",
        linewidth=1.2,
    )
    ax.set_xlabel(f"step_bin (bin_size={int(args.step_bin_size)})")
    ax.set_ylabel("#unique guess moves across rollouts")
    ax.set_ylim(0, 8)
    ax.grid(True, alpha=0.3)
    ax.legend()
    diversity_path = plots_dir / "guess_diversity.png"
    _plot_save(diversity_path, f"Guess diversity vs step\n({step_label}; rollout_n inferred from uid groups)")

    # Parse/format rates.
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.plot(base["step_bin"], base["strict_format_ok_rate"], label="strict format valid rate", linewidth=1.8)
    ax.plot(base["step_bin"], base["guess_uci_valid_rate"], label="guess payload is valid UCI", linewidth=1.2)
    ax.plot(base["step_bin"], base["missing_uci_move_tag_rate"], label="missing <uci_move> tag rate", linewidth=1.2)
    ax.plot(base["step_bin"], base["missing_guess_tag_rate"], label="missing <guess> tag rate", linewidth=1.2)
    ax.set_xlabel(f"step_bin (bin_size={int(args.step_bin_size)})")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    parse_path = plots_dir / "format_and_parse_rates.png"
    _plot_save(parse_path, f"Format/parsing rates vs step\n({step_label})")

    # Histogram at selected steps (optional but useful).
    if by_step:
        pick_steps = [by_step[0], by_step[len(by_step) // 2], by_step[-1]]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
        for ax, s in zip(axes, pick_steps):
            vals = gdf[gdf["step"] == s]["unique_guess_uci"].to_numpy()
            ax.hist(vals, bins=np.arange(-0.5, 8.6, 1.0), edgecolor="black")
            ax.set_title(f"step {s}")
            ax.set_xlabel("#unique guess")
            ax.grid(True, alpha=0.2)
        axes[0].set_ylabel("#prompt groups (uid)")
        hist_path = plots_dir / "guess_diversity_hist_selected_steps.png"
        _plot_save(hist_path, f"Guess diversity histograms\n({step_label})")

    # Write report (markdown).
    forced_any = bool(df["forced_prefix_is_forced"].any())
    last_bin = int(metrics_df["step_bin"].max()) if not metrics_df.empty else None
    free_last = metrics_df[(metrics_df["forced_prefix_is_forced"] == False) & (metrics_df["step_bin"] == last_bin)]  # noqa: E712
    follow_last = float(free_last["follow_guess_rate"].iloc[0]) if len(free_last) else float("nan")

    report_lines: list[str] = []
    report_lines.append(f"# Guess-first analysis report — run `{args.run}`\n")
    report_lines.append(f"- generated_at: `{_now_iso()}`\n")
    report_lines.append(f"- run_path: `{run_path}`\n")
    report_lines.append(f"- commit: `{meta.get('commit')}`\n")
    report_lines.append(f"- step_bin_size: `{int(args.step_bin_size)}`\n")
    report_lines.append(f"- step_definition: {step_label}\n")
    report_lines.append("\n")

    report_lines.append("## Key plot files\n")
    report_lines.append(f"- follow-guess rate: `{follow_path}`\n")
    report_lines.append(f"- move quality: `{quality_path}`\n")
    report_lines.append(f"- guess diversity: `{diversity_path}`\n")
    report_lines.append(f"- format/parsing rates: `{parse_path}`\n")
    report_lines.append("\n")

    report_lines.append("## Headline numbers (free rollouts)\n")
    report_lines.append(f"- any forced rollouts present: `{forced_any}`\n")
    report_lines.append(f"- follow-guess rate at last bin (free): `{follow_last}`\n")
    report_lines.append("\n")

    report_lines.append("## Data products\n")
    report_lines.append(f"- parsed rollout records: `{out_dir / 'rollout_records.parquet'}`\n")
    report_lines.append(f"- metrics by step bin: `{out_dir / 'metrics_by_step_bin.csv'}`\n")
    report_lines.append(f"- diversity by step bin: `{out_dir / 'guess_diversity_by_step_bin.csv'}`\n")
    report_lines.append(f"- parser consistency vs logged fields: `{out_dir / 'parser_consistency.json'}`\n")
    report_lines.append(f"- samples for manual inspection: `{out_dir / 'samples_for_manual_check.csv'}`\n")
    report_lines.append("\n")

    # Minimal "vanilla GRPO?" determination from config.
    algo = config_api.get("algorithm") if isinstance(config_api.get("algorithm"), dict) else {}
    adv = (algo or {}).get("adv_estimator", None)
    rollout_is = (algo or {}).get("rollout_is", None)
    use_pf_ppo = (algo or {}).get("use_pf_ppo", None)
    report_lines.append("## Vanilla GRPO (config-level)\n")
    report_lines.append(f"- algorithm.adv_estimator: `{adv}`\n")
    report_lines.append(f"- algorithm.rollout_is: `{rollout_is}`\n")
    report_lines.append(f"- algorithm.use_pf_ppo: `{use_pf_ppo}`\n")
    report_lines.append("\n")

    (out_dir / "report.md").write_text("".join(report_lines), encoding="utf-8")

    # Log plots back to W&B (optional).
    if args.log_to_wandb:
        wb_run = wandb.init(entity=args.entity, project=args.project, id=args.run, resume="allow", job_type="analysis")
        try:
            wb_run.log(
                {
                    "analysis/guess_follow_rate": wandb.Image(str(follow_path)),
                    "analysis/move_quality_guess_vs_final": wandb.Image(str(quality_path)),
                    "analysis/guess_diversity": wandb.Image(str(diversity_path)),
                    "analysis/format_and_parse_rates": wandb.Image(str(parse_path)),
                }
            )
            if (plots_dir / "guess_diversity_hist_selected_steps.png").exists():
                wb_run.log(
                    {
                        "analysis/guess_diversity_hist_selected_steps": wandb.Image(
                            str(plots_dir / "guess_diversity_hist_selected_steps.png")
                        )
                    }
                )

            # Also log an artifact with the analysis outputs for convenient browsing.
            art = wandb.Artifact(name=f"guess-analysis-{args.run}", type="analysis")
            art.add_file(str(out_dir / "report.md"))
            art.add_file(str(out_dir / "metrics_by_step_bin.csv"))
            art.add_file(str(out_dir / "guess_diversity_by_step_bin.csv"))
            for p in plots_dir.glob("*.png"):
                art.add_file(str(p))
            wb_run.log_artifact(art)
        finally:
            wandb.finish()

    print(f"[done] Wrote report: {out_dir / 'report.md'}")
    print(f"[done] Plots: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
