#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from io import StringIO
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chess.engine
import chess.pgn

# Make repo-root imports work when invoked as `python scripts/xxx.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.chess.full_game_eval import (
    FullGameEvalConfig,
    StockfishConfig,
    _analyze_game_with_engine,
    _configure_stockfish,
    run_full_game_eval,
)


def _load_eval_harness_module():
    """Reuse the existing vLLM backend + model resolution logic without duplicating it."""
    eval_path = REPO_ROOT / "scripts" / "eval_chess_fullgame.py"
    spec = importlib.util.spec_from_file_location("_eval_chess_fullgame_mod", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _delta_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0.0}
    vals = sorted(values)
    n = len(vals)

    def pct(p: float) -> float:
        if n == 1:
            return float(vals[0])
        i = int(round((n - 1) * p))
        i = max(0, min(n - 1, i))
        return float(vals[i])

    mean = float(sum(vals) / n)
    return {
        "count": float(n),
        "mean": mean,
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": float(vals[-1]),
    }


def _pgn_hash(row: Dict[str, Any]) -> str:
    s = str(row.get("pgn") or "").encode("utf-8", errors="ignore")
    return md5(s).hexdigest()


def _pgn_alignment(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare per-game PGN traces to detect inference divergence.

    This is intentionally lightweight: it does not attempt move-by-move diffs, just
    a full-PGN hash match. This is enough to detect whether two runs played the
    same games under the same seeding.
    """
    a_by_id = {r.get("game_id"): r for r in rows_a if r.get("game_id")}
    b_by_id = {r.get("game_id"): r for r in rows_b if r.get("game_id")}
    shared = sorted(set(a_by_id) & set(b_by_id))
    mismatched: List[str] = []
    for gid in shared:
        if _pgn_hash(a_by_id[gid]) != _pgn_hash(b_by_id[gid]):
            mismatched.append(str(gid))
    return {
        "shared_games": int(len(shared)),
        "pgn_mismatch_count": int(len(mismatched)),
        "pgn_match_frac": float((len(shared) - len(mismatched)) / len(shared)) if shared else float("nan"),
        "mismatch_examples": mismatched[:10],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GH200 full-game eval benchmark (baseline vs optimized).")
    p.add_argument("--model", type=str, required=True, help="HF model id/path, or VERL FSDP actor checkpoint dir.")
    p.add_argument(
        "--out-root",
        type=str,
        default="outputs/gh200_fullgame_bench",
        help="Output root directory (creates one subdir per mode).",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run identifier (used in output folder + results filename). Recommended: pass SLURM_JOB_ID.",
    )

    # Evaluation sizing (competition-style).
    p.add_argument("--opponent-depths", type=int, nargs="+", default=[1, 5])
    # Optimized run size (training-aligned): 5 rounds × 50 games = 250 games per depth.
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--games-per-round", type=int, default=50)
    # Baseline/batched-serial run size (defaults to the old 50-game-per-depth setup so it finishes quickly).
    p.add_argument("--baseline-rounds", type=int, default=1)
    p.add_argument("--baseline-games-per-round", type=int, default=50)

    # vLLM runtime knobs (match eval harness defaults, but allow override).
    # Training uses DP/FSDP rather than tensor parallel, so keep TP=1 by default for alignment.
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=5120)
    p.add_argument("--max-num-seqs", type=int, default=2048)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")

    # Decoding + game loop knobs.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-response-tokens", type=int, default=512)
    p.add_argument("--max-retries-per-turn", type=int, default=3)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--opponent-movetime-ms", type=int, default=100)
    p.add_argument(
        "--prompt-template-path",
        type=str,
        default="recipe/chess/prompt_templates/chess_rl_chessr1_prompt.jinja",
    )

    # Stockfish (opponent + ACPL).
    p.add_argument("--stockfish-path", type=str, default="/usr/local/bin/stockfish")
    p.add_argument("--opponent-threads", type=int, default=1)
    # Align with training full_eval defaults (trainer.full_eval.stockfish_hash_mb=128).
    p.add_argument("--stockfish-hash-mb", type=int, default=128)
    p.add_argument("--opponent-skill-level", type=int, default=0)
    p.add_argument("--acpl-skill-level", type=int, default=20)

    p.add_argument("--acpl-depth", type=int, default=20)
    p.add_argument("--acpl-movetime-ms", type=int, default=1000)
    # Align with starter-kit + training defaults (cp cap = 1000, mate_score_cp = 1000).
    p.add_argument("--acpl-cp-cap", type=int, default=1000)
    p.add_argument("--mate-score-cp", type=int, default=1000)

    # Baseline vs optimized toggles.
    # NOTE: serial ACPL over 250 games × 2 depths is very slow with 1s/move limits.
    # The default baseline skips ACPL and we instead compute a serial ACPL *sample* for alignment.
    p.add_argument("--baseline-acpl-workers", type=int, default=0)
    p.add_argument("--optimized-acpl-workers", type=int, default=72)
    p.add_argument("--acpl-threads", type=int, default=2)
    p.add_argument(
        "--baseline-batched-inference",
        action="store_true",
        help="Enable batching even for baseline (useful to isolate ACPL-only changes).",
    )
    p.add_argument(
        "--skip-batched-serial",
        action="store_true",
        help="Skip the batched-inference + serial-ACPL run (normally used for ACPL alignment).",
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline run (only run optimized).",
    )
    p.add_argument(
        "--skip-optimized",
        action="store_true",
        help="Skip optimized run (only run baseline).",
    )

    # Serial ACPL alignment check (sampled games).
    p.add_argument(
        "--acpl-serial-sample-per-depth",
        type=int,
        default=20,
        help="How many games per opponent depth to recompute ACPL serially for alignment.",
    )

    return p.parse_args()


def _run_one(
    *,
    mode: str,
    cfg: FullGameEvalConfig,
    backend,
) -> Tuple[Dict[str, Any], float]:
    t0 = time.time()
    summary = run_full_game_eval(cfg=cfg, backend=backend)
    t1 = time.time()
    wall_s = float(t1 - t0)

    total_moves = 0
    total_cpl = 0.0
    for depth in cfg.opponent_depths:
        row = summary.get("summary_by_depth", {}).get(f"depth_{depth}", {})
        total_moves += int(row.get("acpl_moves", 0) or 0)
        total_cpl += float(row.get("acpl_sum_per_move", 0.0) or 0.0)

    thr_moves_s = (total_moves / wall_s) if wall_s > 0 else float("inf")
    print(
        f"[bench] mode={mode} wall_s={wall_s:.2f} model_moves={total_moves} thr_model_moves_s={thr_moves_s:.2f} total_cpl={total_cpl:.2f}",
        flush=True,
    )
    return summary, wall_s


def _parse_pgn_to_moves_uci(pgn_text: str) -> List[str]:
    if not pgn_text:
        return []
    game = chess.pgn.read_game(StringIO(pgn_text))
    if game is None:
        return []
    return [m.uci() for m in game.mainline_moves()]


def _compute_acpl_serial_sample(
    *,
    rows: List[Dict[str, Any]],
    stockfish_cfg: StockfishConfig,
    depth: int,
    movetime_ms: int,
    cp_cap: int,
    mate_score_cp: int,
    sample_per_depth: int,
    seed: int,
) -> Dict[str, Any]:
    """Recompute ACPL for a small subset of games and compare to the existing run's ACPL."""
    sample_per_depth = int(sample_per_depth)
    if sample_per_depth <= 0:
        return {"enabled": False, "reason": "sample_per_depth<=0"}

    # Group by opponent depth (1/5).
    by_depth: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        try:
            d = int(r.get("opponent_depth"))
        except Exception:
            continue
        by_depth.setdefault(d, []).append(r)

    # Deterministic sampling: take the first K games by game_id per depth.
    sample_rows: List[Dict[str, Any]] = []
    for d in sorted(by_depth):
        candidates = [r for r in by_depth[d] if r.get("game_id") and r.get("pgn")]
        candidates.sort(key=lambda x: str(x.get("game_id")))
        sample_rows.extend(candidates[:sample_per_depth])

    if not sample_rows:
        return {"enabled": False, "reason": "no_sample_rows"}

    # Compute serial ACPL for sample.
    deltas: List[float] = []
    abs_deltas: List[float] = []
    plies_full = sum(int(r.get("num_plies", 0) or 0) for r in rows)
    plies_sample = sum(int(r.get("num_plies", 0) or 0) for r in sample_rows)

    t0 = time.time()
    with chess.engine.SimpleEngine.popen_uci(stockfish_cfg.path) as engine:
        _configure_stockfish(engine, stockfish_cfg)
        for r in sample_rows:
            moves_uci = _parse_pgn_to_moves_uci(str(r.get("pgn") or ""))
            if not moves_uci:
                continue
            analysis = _analyze_game_with_engine(
                engine=engine,
                moves_uci=moves_uci,
                depth=int(depth),
                movetime_ms=int(movetime_ms),
                cp_cap=int(cp_cap),
                mate_score_cp=int(mate_score_cp),
            )
            model_color = str(r.get("model_color") or "").lower()
            serial_model_acpl = float(analysis["white_acpl"] if model_color == "white" else analysis["black_acpl"])
            parallel_model_acpl = float(r.get("model_acpl", 0.0) or 0.0)
            d = float(serial_model_acpl - parallel_model_acpl)
            deltas.append(d)
            abs_deltas.append(abs(d))

    t1 = time.time()
    wall_s = float(t1 - t0)

    # Estimate full serial time by scaling on plies (proxy for number of engine calls).
    serial_time_per_ply = (wall_s / plies_sample) if plies_sample > 0 else float("nan")
    serial_time_full_est_s = float(serial_time_per_ply * plies_full) if plies_full > 0 else float("nan")

    return {
        "enabled": True,
        "sample_per_depth": int(sample_per_depth),
        "sample_games": int(len(sample_rows)),
        "sample_plies": int(plies_sample),
        "full_plies": int(plies_full),
        "serial_wall_s": float(wall_s),
        "serial_time_per_ply_s": float(serial_time_per_ply),
        "serial_full_est_s": float(serial_time_full_est_s),
        "delta_serial_minus_parallel": _delta_summary(deltas),
        "abs_delta": _delta_summary(abs_deltas),
    }


def main() -> None:
    args = parse_args()

    eval_mod = _load_eval_harness_module()
    VllmChatBackend = eval_mod.VllmChatBackend
    _resolve_model_for_vllm = eval_mod._resolve_model_for_vllm
    _sanitize_model_name = eval_mod._sanitize_model_name

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    safe_model = _sanitize_model_name(args.model)
    run_id = str(args.run_id).strip() or time.strftime("%Y%m%d_%H%M%S")

    # Resolve model path once (supports passing a VERL FSDP actor checkpoint dir).
    model_for_vllm = _resolve_model_for_vllm(
        model=args.model,
        out_dir=out_root / f"{safe_model}_artifacts",
        trust_remote_code=bool(args.trust_remote_code),
        use_cpu_initialization=False,
    )

    backend = VllmChatBackend(
        model=model_for_vllm,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        max_num_seqs=int(args.max_num_seqs),
        enforce_eager=bool(args.enforce_eager),
        seed=int(args.seed),
        trust_remote_code=bool(args.trust_remote_code),
    )

    baseline_games_per_depth = int(args.baseline_rounds) * int(args.baseline_games_per_round)
    if baseline_games_per_depth <= 0:
        raise ValueError(f"Invalid baseline_games_per_depth={baseline_games_per_depth}")

    optimized_games_per_depth = int(args.rounds) * int(args.games_per_round)
    if optimized_games_per_depth <= 0:
        raise ValueError(f"Invalid optimized_games_per_depth={optimized_games_per_depth}")

    prompt_template_path = str(args.prompt_template_path).strip() or None

    stockfish_opponent = StockfishConfig(
        path=str(args.stockfish_path),
        threads=int(args.opponent_threads),
        hash_mb=int(args.stockfish_hash_mb),
        skill_level=int(args.opponent_skill_level),
    )
    stockfish_eval = StockfishConfig(
        path=str(args.stockfish_path),
        threads=int(args.acpl_threads),
        hash_mb=int(args.stockfish_hash_mb),
        skill_level=int(args.acpl_skill_level),
    )

    run_cfg_common = dict(
        opponent_depths=list(args.opponent_depths),
        seed=int(args.seed),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_response_tokens=int(args.max_response_tokens),
        max_retries_per_turn=int(args.max_retries_per_turn),
        opponent_movetime_ms=int(args.opponent_movetime_ms),
        acpl_eval_depth=int(args.acpl_depth),
        acpl_eval_movetime_ms=int(args.acpl_movetime_ms),
        acpl_eval_cp_cap=int(args.acpl_cp_cap),
        mate_score_cp=int(args.mate_score_cp),
        max_plies=(int(args.max_plies) if int(args.max_plies) > 0 else None),
        stockfish_opponent=stockfish_opponent,
        stockfish_eval=stockfish_eval,
        prompt_template_path=prompt_template_path,
    )

    summaries: Dict[str, Dict[str, Any]] = {}
    walls: Dict[str, float] = {}
    throughputs: Dict[str, float] = {}

    if not args.skip_baseline:
        out_dir = out_root / (
            f"{safe_model}_baseline_run{run_id}_r{args.baseline_rounds}_gpr{args.baseline_games_per_round}_seed{args.seed}"
        )
        cfg = FullGameEvalConfig(
            **run_cfg_common,
            games_per_depth=int(baseline_games_per_depth),
            rounds=int(args.baseline_rounds),
            games_per_round=int(args.baseline_games_per_round),
            out_dir=out_dir,
            batched_inference=bool(args.baseline_batched_inference),
            acpl_workers=int(args.baseline_acpl_workers),
        )
        print(
            f"[bench] RUN baseline out_dir={out_dir} batched_inference={cfg.batched_inference} acpl_workers={cfg.acpl_workers} acpl_threads={cfg.stockfish_eval.threads}",
            flush=True,
        )
        summaries["baseline"], walls["baseline"] = _run_one(mode="baseline", cfg=cfg, backend=backend)

    # Run a batched-inference + serial-ACPL variant so we can:
    #  - isolate inference batching speedup (baseline -> batched_serial),
    #  - isolate ACPL parallel speedup (batched_serial -> optimized),
    #  - measure ACPL alignment with identical inference traces (batched_serial vs optimized).
    if not args.skip_batched_serial:
        out_dir = out_root / (
            f"{safe_model}_batched_serial_run{run_id}_r{args.baseline_rounds}_gpr{args.baseline_games_per_round}_seed{args.seed}"
        )
        cfg = FullGameEvalConfig(
            **run_cfg_common,
            games_per_depth=int(baseline_games_per_depth),
            rounds=int(args.baseline_rounds),
            games_per_round=int(args.baseline_games_per_round),
            out_dir=out_dir,
            batched_inference=True,
            acpl_workers=int(args.baseline_acpl_workers),
        )
        print(
            f"[bench] RUN batched_serial out_dir={out_dir} batched_inference={cfg.batched_inference} acpl_workers={cfg.acpl_workers} acpl_threads={cfg.stockfish_eval.threads}",
            flush=True,
        )
        summaries["batched_serial"], walls["batched_serial"] = _run_one(mode="batched_serial", cfg=cfg, backend=backend)

    if not args.skip_optimized:
        out_dir = out_root / f"{safe_model}_optimized_run{run_id}_r{args.rounds}_gpr{args.games_per_round}_seed{args.seed}"
        cfg = FullGameEvalConfig(
            **run_cfg_common,
            games_per_depth=int(optimized_games_per_depth),
            rounds=int(args.rounds),
            games_per_round=int(args.games_per_round),
            out_dir=out_dir,
            batched_inference=True,
            acpl_workers=int(args.optimized_acpl_workers),
        )
        print(
            f"[bench] RUN optimized out_dir={out_dir} batched_inference={cfg.batched_inference} acpl_workers={cfg.acpl_workers} acpl_threads={cfg.stockfish_eval.threads}",
            flush=True,
        )
        summaries["optimized"], walls["optimized"] = _run_one(mode="optimized", cfg=cfg, backend=backend)

    # Build a single JSON payload for grep-free consumption (e.g., by the sbatch wrapper).
    results: Dict[str, Any] = {
        "run_id": run_id,
        "ts_unix": time.time(),
        "args": vars(args),
        "model": {"requested": args.model, "resolved_for_vllm": model_for_vllm},
    }

    for mode in ("baseline", "batched_serial", "optimized"):
        if mode not in summaries:
            continue
        s = summaries[mode]
        results[mode] = {
            "wall_s": float(walls.get(mode, float("nan"))),
            "paths": s.get("paths", {}),
            "summary_by_depth": s.get("summary_by_depth", {}),
        }

    # Speedups.
    # NOTE: baseline/batched_serial typically run with ACPL skipped (acpl_workers=0), while optimized
    # runs with parallel ACPL enabled. We therefore compute:
    # - `speedup_infer`: baseline vs batched_serial on measured inference time
    # - `speedup_acpl_est`: estimated serial-ACPL time vs measured parallel-ACPL time
    # - `speedup_total_est`: (baseline wall + est serial ACPL) / optimized wall

    # Prefer explicit inference timing stats if present.
    def _sum_infer_time(mode: str) -> float:
        total = 0.0
        for _, row in (summaries[mode].get("summary_by_depth", {}) or {}).items():
            total += float(row.get("infer_time_s", 0.0) or 0.0)
        return total

    def _sum_infer_positions(mode: str) -> int:
        total = 0
        for _, row in (summaries[mode].get("summary_by_depth", {}) or {}).items():
            total += int(row.get("infer_positions", 0) or 0)
        return int(total)

    if "baseline" in summaries and "batched_serial" in summaries:
        base_infer_s = _sum_infer_time("baseline")
        batched_infer_s = _sum_infer_time("batched_serial")
        base_pos = _sum_infer_positions("baseline")
        batched_pos = _sum_infer_positions("batched_serial")

        results["infer_time_s_baseline"] = float(base_infer_s)
        results["infer_time_s_batched_serial"] = float(batched_infer_s)
        results["infer_positions_baseline"] = int(base_pos)
        results["infer_positions_batched_serial"] = int(batched_pos)

        # Speedup on the *same* baseline sizing (default 50 games/depth).
        if base_infer_s > 0 and batched_infer_s > 0:
            speedup_infer = float(base_infer_s / batched_infer_s)
            results["speedup_infer"] = speedup_infer
            print(
                f"[bench][speedup_infer] baseline_infer_s={base_infer_s:.2f} batched_infer_s={batched_infer_s:.2f} speedup={speedup_infer:.3f}",
                flush=True,
            )

        # Also emit normalized per-position costs (useful when run sizes differ).
        if base_infer_s > 0 and base_pos > 0:
            results["infer_s_per_pos_baseline"] = float(base_infer_s / base_pos)
        if batched_infer_s > 0 and batched_pos > 0:
            results["infer_s_per_pos_batched_serial"] = float(batched_infer_s / batched_pos)

    if "baseline" in summaries and "optimized" in summaries:
        base_infer_s = _sum_infer_time("baseline")
        base_pos = _sum_infer_positions("baseline")
        opt_infer_s = _sum_infer_time("optimized")
        opt_pos = _sum_infer_positions("optimized")

        results["infer_time_s_optimized"] = float(opt_infer_s)
        results["infer_positions_optimized"] = int(opt_pos)

        if opt_infer_s > 0 and opt_pos > 0:
            results["infer_s_per_pos_optimized"] = float(opt_infer_s / opt_pos)

        # Estimate "what serial inference would have cost" for the full optimized run
        # using the measured per-position cost from the small serial baseline.
        if base_infer_s > 0 and base_pos > 0 and opt_pos > 0:
            serial_infer_est_s_for_optimized = float((base_infer_s / base_pos) * opt_pos)
            results["infer_serial_est_s_for_optimized"] = serial_infer_est_s_for_optimized
            if opt_infer_s > 0:
                speedup_infer_est_full = float(serial_infer_est_s_for_optimized / opt_infer_s)
                results["speedup_infer_est_full"] = speedup_infer_est_full
                print(
                    f"[bench][speedup_infer_est_full] serial_est_s={serial_infer_est_s_for_optimized:.2f} "
                    f"optimized_infer_s={opt_infer_s:.2f} speedup={speedup_infer_est_full:.3f}",
                    flush=True,
                )

    # Alignment checks.
    def _load_games(mode: str) -> List[Dict[str, Any]]:
        return _read_jsonl(Path(summaries[mode]["paths"]["games_jsonl"]))

    if "baseline" in summaries and "batched_serial" in summaries:
        base_rows = _load_games("baseline")
        mid_rows = _load_games("batched_serial")
        align = _pgn_alignment(base_rows, mid_rows)
        results["pgn_alignment_baseline_vs_batched_serial"] = align
        print(
            f"[bench][pgn_alignment] baseline_vs_batched_serial shared_games={align['shared_games']} "
            f"pgn_mismatch_count={align['pgn_mismatch_count']} pgn_match_frac={align['pgn_match_frac']:.4f}",
            flush=True,
        )

    if "batched_serial" in summaries and "optimized" in summaries:
        mid_rows = _load_games("batched_serial")
        opt_rows = _load_games("optimized")
        pgn_align = _pgn_alignment(mid_rows, opt_rows)
        results["pgn_alignment_batched_serial_vs_optimized"] = pgn_align
        print(
            f"[bench][pgn_alignment] batched_serial_vs_optimized shared_games={pgn_align['shared_games']} "
            f"pgn_mismatch_count={pgn_align['pgn_mismatch_count']} pgn_match_frac={pgn_align['pgn_match_frac']:.4f}",
            flush=True,
        )

        # ACPL alignment is performed via a serial Stockfish recompute over a small sample of games.
        # This keeps the Slurm job within the 1-day QOS walltime limit.
        acpl_serial_sample = _compute_acpl_serial_sample(
            rows=opt_rows,
            stockfish_cfg=stockfish_eval,
            depth=int(args.acpl_depth),
            movetime_ms=int(args.acpl_movetime_ms),
            cp_cap=int(args.acpl_cp_cap),
            mate_score_cp=int(args.mate_score_cp),
            sample_per_depth=int(args.acpl_serial_sample_per_depth),
            seed=int(args.seed),
        )
        results["acpl_alignment_serial_sample_vs_parallel"] = acpl_serial_sample
        if acpl_serial_sample.get("enabled"):
            d = acpl_serial_sample.get("delta_serial_minus_parallel") or {}
            ad = acpl_serial_sample.get("abs_delta") or {}
            print(
                "[bench][acpl_alignment_serial_sample] "
                f"games={acpl_serial_sample.get('sample_games')} "
                f"plies={acpl_serial_sample.get('sample_plies')} "
                f"serial_wall_s={acpl_serial_sample.get('serial_wall_s'):.2f} "
                f"delta_mean={d.get('mean', float('nan')):.6g} abs_delta_mean={ad.get('mean', float('nan')):.6g}",
                flush=True,
            )

        # ACPL speedup estimate: compare estimated full-serial time to measured parallel time.
        parallel_acpl_time_s = 0.0
        for _, row in (summaries["optimized"].get("summary_by_depth", {}) or {}).items():
            parallel_acpl_time_s += float(row.get("acpl_time_s", 0.0) or 0.0)
        results["acpl_parallel_time_s"] = float(parallel_acpl_time_s)

        serial_full_est_s = float(acpl_serial_sample.get("serial_full_est_s", float("nan")))
        if parallel_acpl_time_s > 0 and serial_full_est_s > 0:
            speedup_acpl_est = float(serial_full_est_s / parallel_acpl_time_s)
            results["speedup_acpl_est"] = speedup_acpl_est
            print(
                f"[bench][speedup_acpl_est] serial_full_est_s={serial_full_est_s:.2f} parallel_acpl_s={parallel_acpl_time_s:.2f} speedup={speedup_acpl_est:.3f}",
                flush=True,
            )

        # Total speedup estimate (old ~= serial inference + serial ACPL, new ~= batched inference + parallel ACPL).
        #
        # When the baseline run is intentionally small (default 50 games/depth), we estimate what the
        # full serial inference time would have been for the *optimized* run's number of positions.
        if "baseline" in summaries and walls.get("optimized"):
            opt_infer_s = _sum_infer_time("optimized")
            opt_pos = _sum_infer_positions("optimized")
            base_infer_s = _sum_infer_time("baseline")
            base_pos = _sum_infer_positions("baseline")

            serial_infer_est_s = (
                float((base_infer_s / base_pos) * opt_pos) if base_infer_s > 0 and base_pos > 0 and opt_pos > 0 else float("nan")
            )
            results["infer_serial_est_s_for_optimized"] = float(serial_infer_est_s)

            # Reuse optimized wall time to estimate the non-optimized remainder (opponent moves, prompt build, I/O).
            optimized_total = float(walls["optimized"])
            other_time_s = optimized_total - float(opt_infer_s) - float(parallel_acpl_time_s)
            results["other_time_s_optimized"] = float(other_time_s)

            serial_acpl_est_s = float(acpl_serial_sample.get("serial_full_est_s", float("nan")))
            results["acpl_serial_full_est_s_for_optimized"] = float(serial_acpl_est_s)

            baseline_total_est = float(serial_infer_est_s + other_time_s + serial_acpl_est_s)
            results["baseline_total_est_s_for_optimized"] = float(baseline_total_est)

            if optimized_total > 0 and baseline_total_est > 0:
                speedup_total_est = float(baseline_total_est / optimized_total)
                results["speedup_total_est"] = speedup_total_est
                print(
                    f"[bench][speedup_total_est] serial_infer_est_s={serial_infer_est_s:.2f} + other_s={other_time_s:.2f} "
                    f"+ serial_acpl_est_s={serial_acpl_est_s:.2f} -> {baseline_total_est:.2f} "
                    f"optimized_wall_s={optimized_total:.2f} speedup={speedup_total_est:.3f}",
                    flush=True,
                )

    results_path = out_root / f"bench_results_{run_id}.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[bench] Wrote results: {results_path}", flush=True)


if __name__ == "__main__":
    # Avoid tokenizer parallelism oversubscription (esp. in container).
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
