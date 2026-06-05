#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
import chess.pgn
from jinja2 import Template, meta

# Make repo-root imports work when invoked as `python scripts/xxx.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import recipe.chess.full_game_eval as fge
from scripts.eval_chess_fullgame_model_vs_model import (
    OpenAIChatBackend,
    _kill_process_tree,
    _read_checkpoints_by_run,
    _resolve_model_for_vllm_two_stage,
    _sanitize_run_id,
    _start_vllm_server,
    _unordered_pairs,
    _wait_for_server_ready,
)


def _comma_list(s: str) -> List[str]:
    return [p.strip() for p in str(s or "").split(",") if p.strip()]


def _load_prompt_template(path: str) -> Tuple[Template, set[str]]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    src = p.read_text(encoding="utf-8")
    tmpl = Template(src)
    vars_ = meta.find_undeclared_variables(tmpl.environment.parse(src))
    return tmpl, vars_


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _set_forfeit_for_color(game: fge._GameState, *, reason: str, failed_color: chess.Color) -> None:
    game.is_over = True
    game.forfeit = True
    game.forfeit_reason = reason
    game.termination = "resignation"
    game.result = "0-1" if failed_color == chess.WHITE else "1-0"
    game.result_str = "Black wins (White resigned)" if failed_color == chess.WHITE else "White wins (Black resigned)"


def _step_actor_moves(
    *,
    cfg: fge.FullGameEvalConfig,
    backend: Any,
    actor_run_id: str,
    games: List[fge._GameState],
    moves_fp,
    prompt_template: Template,
    prompt_template_vars: set[str],
    forfeit_color_by_game: Dict[str, chess.Color],
) -> Dict[str, Any]:
    infer_time_s = 0.0
    infer_positions = 0
    infer_batches = 0

    pending: Dict[str, Dict[str, Any]] = {
        g.game_id: {"game": g, "last_output": "", "last_error": "", "last_prompt_text": ""} for g in games
    }

    for retry_idx in range(int(cfg.max_retries_per_turn)):
        todo = [st["game"] for st in pending.values() if not st["game"].is_over]
        if not todo:
            return {
                "infer_time_s": float(infer_time_s),
                "infer_positions": int(infer_positions),
                "infer_batches": int(infer_batches),
            }

        prompts: List[List[Dict[str, str]]] = []
        prompt_texts: List[str] = []
        meta_rows: List[Tuple[fge._GameState, str, str]] = []
        for g in todo:
            white_run_id = getattr(g, "white_run_id")
            black_run_id = getattr(g, "black_run_id")
            actor_color = chess.WHITE if actor_run_id == white_run_id else chess.BLACK
            if g.board.turn != actor_color:
                continue

            legal_moves = list(g.board.legal_moves)
            side_to_move = "White" if g.board.turn else "Black"
            move_history = [m.uci() for m in g.board.move_stack]
            ctx = fge._build_prompt_context(
                board=g.board,
                legal_moves=legal_moves,
                move_history=move_history,
                side_to_move=side_to_move,
                needed_vars=prompt_template_vars,
            )
            prompt_text = prompt_template.render(**ctx)
            prompts.append([{"role": "user", "content": prompt_text}])
            prompt_texts.append(prompt_text)
            meta_rows.append((g, white_run_id, black_run_id))

        if not prompts:
            return {
                "infer_time_s": float(infer_time_s),
                "infer_positions": int(infer_positions),
                "infer_batches": int(infer_batches),
            }

        seeds = [
            fge._safe_int_seed(
                fge._mix_seed(cfg.seed, salt=f"rrd|{actor_run_id}|{g.game_id}|ply={len(g.board.move_stack)}|try={retry_idx}")
            )
            for (g, _, _) in meta_rows
        ]

        t0 = time.time()
        outputs = backend.generate(
            prompts,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_response_tokens,
            seeds=seeds,
        )
        t1 = time.time()

        infer_time_s += float(t1 - t0)
        infer_positions += int(len(outputs))
        infer_batches += 1

        if len(outputs) != len(meta_rows):
            raise RuntimeError(f"[rrd][{actor_run_id}] backend returned {len(outputs)} outputs for {len(meta_rows)} prompts")

        for (g, white_run_id, black_run_id), prompt_text, output_text in zip(meta_rows, prompt_texts, outputs):
            ts = time.time()
            actor_color = chess.WHITE if actor_run_id == white_run_id else chess.BLACK
            opponent_run_id = black_run_id if actor_color == chess.WHITE else white_run_id

            parsed_uci, parse_err = fge._parse_model_move(output_text)
            legal_moves = list(g.board.legal_moves)
            legal_moves_uci = [m.uci().lower() for m in legal_moves]
            legal_moves_set = set(legal_moves_uci)

            is_legal = False
            played_move: Optional[chess.Move] = None
            if parsed_uci and parsed_uci in legal_moves_set:
                is_legal = True
                try:
                    played_move = chess.Move.from_uci(parsed_uci)
                except Exception:
                    played_move = None

            error_reason = ""
            if parse_err:
                error_reason = parse_err
            elif parsed_uci is None:
                error_reason = "bad_move"
            elif not is_legal:
                error_reason = "illegal_move"

            record: Dict[str, Any] = {
                "ts": ts,
                "game_id": g.game_id,
                "pair_id": getattr(g, "pair_id"),
                "ply": int(len(g.board.move_stack)),
                "fen": g.board.fen(),
                "actor_run_id": actor_run_id,
                "opponent_run_id": opponent_run_id,
                "side_to_move": "white" if g.board.turn == chess.WHITE else "black",
                "white_run_id": white_run_id,
                "black_run_id": black_run_id,
                "legal_moves_uci": legal_moves_uci,
                "prompt_text": prompt_text,
                "raw_output_text": output_text,
                "format_ok": bool(parse_err == ""),
                "parsed_move_uci": parsed_uci or "",
                "is_legal": bool(is_legal),
                "retry_idx": int(retry_idx),
                "error_reason": error_reason,
            }

            if error_reason:
                fge._jsonl_write(moves_fp, record)
                pending[g.game_id]["last_output"] = output_text
                pending[g.game_id]["last_error"] = error_reason
                pending[g.game_id]["last_prompt_text"] = prompt_text
                continue

            if played_move is None:
                record["error_reason"] = "bad_move"
                fge._jsonl_write(moves_fp, record)
                pending[g.game_id]["last_output"] = output_text
                pending[g.game_id]["last_error"] = "bad_move"
                pending[g.game_id]["last_prompt_text"] = prompt_text
                continue

            record["accepted_move_uci"] = played_move.uci().lower()
            fge._jsonl_write(moves_fp, record)

            g.board.push(played_move)
            g.pgn_node = g.pgn_node.add_variation(played_move)
            if g.board.is_game_over(claim_draw=False):
                fge._apply_board_outcome(g)

            pending.pop(g.game_id, None)

    for st in pending.values():
        g = st["game"]
        if g.is_over:
            continue
        reason = st.get("last_error") or "invalid_output"
        failed_color = g.board.turn
        forfeit_color_by_game[g.game_id] = failed_color
        _set_forfeit_for_color(g, reason=reason, failed_color=failed_color)
        fge._jsonl_write(
            moves_fp,
            {
                "ts": time.time(),
                "game_id": g.game_id,
                "pair_id": getattr(g, "pair_id"),
                "ply": int(len(g.board.move_stack)),
                "fen": g.board.fen(),
                "actor_run_id": actor_run_id,
                "opponent_run_id": "<unknown>",
                "side_to_move": "white" if failed_color == chess.WHITE else "black",
                "white_run_id": getattr(g, "white_run_id"),
                "black_run_id": getattr(g, "black_run_id"),
                "legal_moves_uci": [m.uci().lower() for m in g.board.legal_moves],
                "prompt_text": st.get("last_prompt_text") or "<prompt_text_unavailable>",
                "raw_output_text": st.get("last_output") or "",
                "format_ok": False,
                "parsed_move_uci": "",
                "is_legal": False,
                "retry_idx": int(cfg.max_retries_per_turn),
                "error_reason": f"forfeit:{reason}",
                "forfeit": True,
            },
        )

    return {
        "infer_time_s": float(infer_time_s),
        "infer_positions": int(infer_positions),
        "infer_batches": int(infer_batches),
    }


def _init_pair_games(
    *,
    run_id_a: str,
    run_id_b: str,
    games_per_pair: int,
    seed: int,
) -> List[fge._GameState]:
    n = int(games_per_pair)
    if n <= 0:
        raise ValueError("--games-per-pair must be > 0")
    if n % 2 != 0:
        raise ValueError("--games-per-pair must be even for strict color balancing")

    pair_id = f"{run_id_a}_vs_{run_id_b}"

    colors: List[chess.Color] = [chess.WHITE] * (n // 2) + [chess.BLACK] * (n // 2)
    rng = fge._DeterministicRng(seed, salt=f"rrd_pair_color_shuffle|{pair_id}")
    rng.shuffle(colors)

    games: List[fge._GameState] = []
    for i, a_color in enumerate(colors):
        board = chess.Board()
        pgn = chess.pgn.Game()
        pgn.headers["Event"] = "chess-rl full-game eval (round-robin distributed, infer)"
        pgn.headers["Site"] = "isambard"
        pgn.headers["Date"] = time.strftime("%Y.%m.%d")

        if a_color == chess.WHITE:
            white_run_id = run_id_a
            black_run_id = run_id_b
        else:
            white_run_id = run_id_b
            black_run_id = run_id_a

        pgn.headers["White"] = white_run_id
        pgn.headers["Black"] = black_run_id

        game_id = f"{pair_id}_g{i:03d}"
        pgn.headers["GameId"] = game_id
        pgn.headers["PairId"] = pair_id

        g = fge._GameState(
            game_id=game_id,
            opponent_depth=0,
            model_color=a_color,
            board=board,
            pgn=pgn,
            pgn_node=pgn,
            round_idx=0,
            game_idx_in_round=i,
        )
        setattr(g, "pair_id", pair_id)
        setattr(g, "white_run_id", white_run_id)
        setattr(g, "black_run_id", black_run_id)
        setattr(g, "run_id_a", run_id_a)
        setattr(g, "run_id_b", run_id_b)
        games.append(g)

    return games


@dataclass(frozen=True)
class _ServerInfo:
    run_id: str
    host: str
    port: int
    base_url: str
    api_key: str
    model: str


def _resolve_node_context(args: argparse.Namespace) -> Tuple[int, int]:
    if int(args.node_count) > 0:
        if int(args.node_idx) < 0:
            raise ValueError("--node-idx is required when --node-count is provided")
        return int(args.node_idx), int(args.node_count)

    env_node_id = os.environ.get("SLURM_NODEID", "")
    env_nnodes = os.environ.get("SLURM_NNODES", "")
    if env_node_id.strip() and env_nnodes.strip():
        return int(env_node_id), int(env_nnodes)

    return 0, 1


def _resolve_run_ids(args: argparse.Namespace, ckpts: Dict[str, str]) -> List[str]:
    if args.run_ids:
        return _comma_list(args.run_ids)
    return sorted(ckpts.keys())


def _assign_run_ids_to_nodes(run_ids: Sequence[str], *, node_count: int) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    if node_count <= 0:
        raise ValueError(f"node_count must be >= 1, got {node_count}")
    if len(run_ids) == 0:
        raise ValueError("No run ids provided")
    if len(run_ids) % node_count != 0:
        raise ValueError(f"run_ids={len(run_ids)} must be divisible by node_count={node_count}")

    models_per_node = len(run_ids) // node_count
    run_id_to_node: Dict[str, int] = {}
    run_id_to_local_idx: Dict[str, int] = {}
    for i, rid in enumerate(run_ids):
        node_idx = i // models_per_node
        local_idx = i % models_per_node
        run_id_to_node[rid] = int(node_idx)
        run_id_to_local_idx[rid] = int(local_idx)
    return int(models_per_node), run_id_to_node, run_id_to_local_idx


def _assign_pairs_evenly(
    pairs: Sequence[Tuple[str, str]],
    *,
    run_id_to_node: Dict[str, int],
) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}

    by_node_pair: Dict[Tuple[int, int], List[Tuple[str, str]]] = {}
    for a, b in pairs:
        na = int(run_id_to_node[a])
        nb = int(run_id_to_node[b])
        if na == nb:
            out[(a, b)] = na
            continue
        i, j = (na, nb) if na < nb else (nb, na)
        by_node_pair.setdefault((i, j), []).append((a, b))

    for (i, j), ps in by_node_pair.items():
        ps_sorted = sorted(ps, key=lambda t: (str(t[0]), str(t[1])))
        for idx, (a, b) in enumerate(ps_sorted):
            out[(a, b)] = i if (idx % 2 == 0) else j

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Round-robin (distributed, inference only): start one vLLM server per model across multiple nodes "
            "and distribute match coordination evenly across nodes."
        )
    )
    p.add_argument("--checkpoints-json", type=str, default="checkpoints_by_run.json")
    p.add_argument(
        "--run-ids",
        type=str,
        default="",
        help="Optional comma-separated run ids from checkpoints_by_run.json. Default: all keys (sorted).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output root directory (default: outputs/full_game_eval_rr_distributed/<job>/).",
    )

    p.add_argument("--node-idx", type=int, default=-1, help="Override SLURM_NODEID for local debugging (0-indexed).")
    p.add_argument("--node-count", type=int, default=-1, help="Override SLURM_NNODES for local debugging.")
    p.add_argument("--cluster-dir", type=str, default="", help="Optional shared directory for server map barrier.")

    p.add_argument("--games-per-pair", type=int, default=100, help="Games per unordered pair (must be even).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-response-tokens", type=int, default=2000)
    p.add_argument("--max-retries-per-turn", type=int, default=1)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument(
        "--prompt-template-path",
        type=str,
        default="recipe/chess/prompt_templates/select_prompt.jinja",
        help="Jinja prompt template path.",
    )

    p.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--server-port-base", type=int, default=8000)
    p.add_argument("--server-ready-timeout-s", type=float, default=900.0)
    p.add_argument("--backend-max-workers", type=int, default=0, help="Max in-flight OpenAI requests per actor (0 = no cap).")

    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--merge-use-cpu-initialization", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    node_idx, node_count = _resolve_node_context(args)
    print(f"[rrd] node_idx={node_idx} node_count={node_count}", flush=True)

    ckpt_path = Path(args.checkpoints_json)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    ckpts = _read_checkpoints_by_run(ckpt_path)
    run_ids = _resolve_run_ids(args, ckpts)
    if len(run_ids) < 2:
        raise ValueError("Need at least 2 run ids for a round robin.")
    if any(rid not in ckpts for rid in run_ids):
        missing = [rid for rid in run_ids if rid not in ckpts]
        raise KeyError(f"run id(s) missing from checkpoints json: {missing}")

    run_ids = sorted(run_ids)
    models_per_node, run_id_to_node, run_id_to_local_idx = _assign_run_ids_to_nodes(run_ids, node_count=node_count)

    gpus_on_node_env = os.environ.get("SLURM_GPUS_ON_NODE", "").strip()
    if gpus_on_node_env.isdigit():
        gpus_on_node = int(gpus_on_node_env)
        if models_per_node > gpus_on_node:
            raise RuntimeError(f"models_per_node={models_per_node} exceeds SLURM_GPUS_ON_NODE={gpus_on_node}")

    local_run_ids = [rid for rid in run_ids if run_id_to_node[rid] == node_idx]
    if len(local_run_ids) != models_per_node:
        raise RuntimeError(f"[rrd] expected {models_per_node} local run ids for node {node_idx}, got {len(local_run_ids)}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"outputs/full_game_eval_rr_distributed/job_{os.getpid()}_{int(time.time())}")
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dir = out_dir / f"shard_{node_idx}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    cluster_dir = Path(args.cluster_dir) if args.cluster_dir else (out_dir / "cluster")
    if not cluster_dir.is_absolute():
        cluster_dir = out_dir / cluster_dir
    cluster_dir.mkdir(parents=True, exist_ok=True)

    run_args_path = shard_dir / "run_args.json"
    run_args_path.write_text(
        json.dumps(
            {"argv": sys.argv, "parsed_args": vars(args), "node_idx": node_idx, "node_count": node_count}, indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[rrd] wrote run args: {run_args_path}", flush=True)

    prompt_template, prompt_template_vars = _load_prompt_template(args.prompt_template_path)
    print(f"[rrd] prompt_template_vars={sorted(prompt_template_vars)}", flush=True)

    model_dirs: Dict[str, Path] = {}
    resolved_models: Dict[str, str] = {}
    for rid in local_run_ids:
        safe = _sanitize_run_id(rid)
        d = out_dir / "models" / safe
        d.mkdir(parents=True, exist_ok=True)
        model_dirs[rid] = d
        resolved = _resolve_model_for_vllm_two_stage(
            model=ckpts[rid],
            out_dir=d,
            trust_remote_code=bool(args.trust_remote_code),
            use_cpu_initialization=bool(args.merge_use_cpu_initialization),
        )
        resolved_models[rid] = resolved
        print(f"[rrd] resolved model {rid} -> {resolved}", flush=True)

    api_key = "dummy"
    servers = []
    for rid in local_run_ids:
        local_idx = int(run_id_to_local_idx[rid])
        port = int(args.server_port_base) + int(local_idx)
        gpu = str(local_idx)
        log_path = out_dir / "logs" / f"node{node_idx}_vllm_server_{_sanitize_run_id(rid)}.log"
        s = _start_vllm_server(
            name=f"node{node_idx}:{rid}",
            model=resolved_models[rid],
            served_model_name=f"run-{_sanitize_run_id(rid)}",
            port=port,
            cuda_visible_devices=gpu,
            tensor_parallel_size=1,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=int(args.max_model_len),
            enforce_eager=bool(args.enforce_eager),
            trust_remote_code=bool(args.trust_remote_code),
            api_key=api_key,
            log_path=log_path,
        )
        servers.append((rid, s))

    def _shutdown() -> None:
        for _, s in servers:
            try:
                _kill_process_tree(s.proc, name=s.name)
            except Exception:
                pass
            try:
                s._log_fp.close()
            except Exception:
                pass

    atexit.register(_shutdown)

    def _handle_sigterm(signum, frame) -> None:  # noqa: ANN001
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    for _, s in servers:
        _wait_for_server_ready(s, timeout_s=float(args.server_ready_timeout_s))

    host = (
        os.environ.get("SLURMD_NODENAME")
        or os.environ.get("HOSTNAME")
        or socket.gethostname()
        or f"node{node_idx}"
    )
    local_server_infos: List[Dict[str, Any]] = []
    for rid, s in servers:
        local_idx = int(run_id_to_local_idx[rid])
        port = int(args.server_port_base) + int(local_idx)
        local_server_infos.append(
            {
                "run_id": rid,
                "host": host,
                "port": int(port),
                "base_url": f"http://{host}:{int(port)}/v1",
                "api_key": str(s.api_key),
                "model": str(s.served_model_name),
                "cuda_visible_devices": str(s.cuda_visible_devices),
            }
        )

    servers_path = cluster_dir / f"servers_node{node_idx}.json"
    _atomic_write_json(
        servers_path,
        {
            "ts": time.time(),
            "node_idx": int(node_idx),
            "node_count": int(node_count),
            "host": str(host),
            "models_per_node": int(models_per_node),
            "run_ids": list(run_ids),
            "local_run_ids": list(local_run_ids),
            "servers": local_server_infos,
        },
    )
    print(f"[rrd] wrote local servers: {servers_path}", flush=True)

    t_barrier0 = time.time()
    while True:
        missing = []
        for i in range(int(node_count)):
            if not (cluster_dir / f"servers_node{i}.json").exists():
                missing.append(i)
        if not missing:
            break
        if (time.time() - t_barrier0) > float(args.server_ready_timeout_s):
            raise TimeoutError(f"[rrd] timeout waiting for servers files from nodes {missing} under {cluster_dir}")
        time.sleep(2.0)

    server_map: Dict[str, _ServerInfo] = {}
    for i in range(int(node_count)):
        obj = json.loads((cluster_dir / f"servers_node{i}.json").read_text(encoding="utf-8"))
        for s in obj.get("servers", []) or []:
            rid = str(s["run_id"])
            server_map[rid] = _ServerInfo(
                run_id=rid,
                host=str(s["host"]),
                port=int(s["port"]),
                base_url=str(s["base_url"]),
                api_key=str(s["api_key"]),
                model=str(s["model"]),
            )
    if set(server_map.keys()) != set(run_ids):
        missing = sorted(set(run_ids) - set(server_map.keys()))
        extra = sorted(set(server_map.keys()) - set(run_ids))
        raise RuntimeError(f"[rrd] server_map mismatch missing={missing} extra={extra}")

    if node_idx == 0:
        _atomic_write_json(
            cluster_dir / "server_map.json",
            {
                "ts": time.time(),
                "node_count": int(node_count),
                "models_per_node": int(models_per_node),
                "run_ids": list(run_ids),
                "servers": {rid: server_map[rid].__dict__ for rid in run_ids},
            },
        )

    backends: Dict[str, OpenAIChatBackend] = {}
    for rid in run_ids:
        si = server_map[rid]
        backends[rid] = OpenAIChatBackend(
            name=rid,
            base_url=si.base_url,
            api_key=si.api_key,
            model=si.model,
            max_workers=int(args.backend_max_workers),
        )

    cfg = fge.FullGameEvalConfig(
        opponent_depths=[0],
        games_per_depth=int(args.games_per_pair),
        seed=int(args.seed),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_response_tokens=int(args.max_response_tokens),
        max_retries_per_turn=int(args.max_retries_per_turn),
        resignation_cpl=0,
        acpl_workers=0,
        max_plies=(int(args.max_plies) if int(args.max_plies) > 0 else None),
        prompt_template_path=str(args.prompt_template_path),
        out_dir=shard_dir,
    )

    pairs = _unordered_pairs(run_ids)
    pair_assignment = _assign_pairs_evenly(pairs, run_id_to_node=run_id_to_node)
    pairs_for_node = [p for p in pairs if int(pair_assignment[p]) == int(node_idx)]
    print(f"[rrd] runs={len(run_ids)} pairs_total={len(pairs)} pairs_for_node={len(pairs_for_node)}", flush=True)

    all_games: List[fge._GameState] = []
    for run_id_a, run_id_b in pairs_for_node:
        all_games.extend(
            _init_pair_games(
                run_id_a=run_id_a,
                run_id_b=run_id_b,
                games_per_pair=int(args.games_per_pair),
                seed=int(args.seed),
            )
        )

    moves_path = shard_dir / "moves.jsonl"
    games_path = shard_dir / "games.jsonl"
    pgn_path = shard_dir / "games.pgn"
    summary_path = shard_dir / "summary_infer.json"

    forfeit_color_by_game: Dict[str, chess.Color] = {}

    infer_time_s = 0.0
    infer_positions = 0
    infer_batches = 0

    t0 = time.time()
    with open(moves_path, "w", encoding="utf-8") as moves_fp:
        moves_lock = Lock()

        class _LockedTextIO:
            def __init__(self, fp, lock: Lock):
                self._fp = fp
                self._lock = lock

            def write(self, s: str) -> int:
                with self._lock:
                    return self._fp.write(s)

            def flush(self) -> None:
                with self._lock:
                    self._fp.flush()

        moves_fp_locked = _LockedTextIO(moves_fp, moves_lock)

        active = [g for g in all_games if not g.is_over]
        while active:
            fge._enforce_max_plies(active, max_plies=cfg.max_plies)

            by_actor: Dict[str, List[fge._GameState]] = {}
            for g in active:
                if g.is_over:
                    continue
                actor = getattr(g, "white_run_id") if g.board.turn == chess.WHITE else getattr(g, "black_run_id")
                by_actor.setdefault(actor, []).append(g)

            import concurrent.futures as cf

            with cf.ThreadPoolExecutor(max_workers=len(by_actor)) as ex:
                futs = []
                for actor, gs in by_actor.items():
                    futs.append(
                        ex.submit(
                            _step_actor_moves,
                            cfg=cfg,
                            backend=backends[actor],
                            actor_run_id=actor,
                            games=gs,
                            moves_fp=moves_fp_locked,
                            prompt_template=prompt_template,
                            prompt_template_vars=prompt_template_vars,
                            forfeit_color_by_game=forfeit_color_by_game,
                        )
                    )
                for fut in futs:
                    stats = fut.result()
                    infer_time_s += float(stats.get("infer_time_s", 0.0) or 0.0)
                    infer_positions += int(stats.get("infer_positions", 0) or 0)
                    infer_batches += int(stats.get("infer_batches", 0) or 0)

            fge._enforce_max_plies(active, max_plies=cfg.max_plies)
            active = [g for g in active if not g.is_over]

    t1 = time.time()

    with (
        open(games_path, "w", encoding="utf-8") as games_fp,
        open(pgn_path, "w", encoding="utf-8") as pgn_fp,
    ):
        for g in all_games:
            if g.pgn.headers.get("Result", "*") in ("", "*"):
                g.pgn.headers["Result"] = g.result
            if not g.result_str:
                g.result_str = fge._pgn_result_to_result_str(g.result)

            pgn_text = str(g.pgn).strip()
            if pgn_text:
                if pgn_fp.tell() > 0:
                    pgn_fp.write("\n\n")
                pgn_fp.write(pgn_text)

            forfeited_color = forfeit_color_by_game.get(g.game_id, None)
            forfeited_color_str = None
            if forfeited_color == chess.WHITE:
                forfeited_color_str = "white"
            elif forfeited_color == chess.BLACK:
                forfeited_color_str = "black"

            fge._jsonl_write(
                games_fp,
                {
                    "ts": time.time(),
                    "stage": "infer",
                    "node_idx": int(node_idx),
                    "node_count": int(node_count),
                    "game_id": g.game_id,
                    "pair_id": getattr(g, "pair_id"),
                    "run_id_a": getattr(g, "run_id_a"),
                    "run_id_b": getattr(g, "run_id_b"),
                    "white_run_id": getattr(g, "white_run_id"),
                    "black_run_id": getattr(g, "black_run_id"),
                    "result": g.result_str or g.result,
                    "pgn_result": g.result,
                    "termination": g.termination,
                    "engine_error": g.engine_error,
                    "forfeit": bool(g.forfeit),
                    "forfeit_reason": g.forfeit_reason,
                    "forfeit_color": forfeited_color_str,
                    "num_plies": int(len(g.board.move_stack)),
                    "pgn": pgn_text,
                },
            )

    per_pair: Dict[str, Dict[str, Any]] = {}
    for g in all_games:
        pid = getattr(g, "pair_id")
        run_id_a = getattr(g, "run_id_a")
        run_id_b = getattr(g, "run_id_b")
        a_is_white = g.model_color == chess.WHITE
        if pid not in per_pair:
            per_pair[pid] = {
                "run_id_a": run_id_a,
                "run_id_b": run_id_b,
                "games_total": 0,
                "model_a": {"wins": 0, "losses": 0, "draws": 0},
                "model_b": {"wins": 0, "losses": 0, "draws": 0},
            }
        per_pair[pid]["games_total"] += 1
        if g.result == "1/2-1/2":
            per_pair[pid]["model_a"]["draws"] += 1
            per_pair[pid]["model_b"]["draws"] += 1
        else:
            a_won = (g.result == "1-0" and a_is_white) or (g.result == "0-1" and (not a_is_white))
            if a_won:
                per_pair[pid]["model_a"]["wins"] += 1
                per_pair[pid]["model_b"]["losses"] += 1
            else:
                per_pair[pid]["model_a"]["losses"] += 1
                per_pair[pid]["model_b"]["wins"] += 1

    summary = {
        "config": {
            "stage": "infer",
            "node_idx": int(node_idx),
            "node_count": int(node_count),
            "run_ids": run_ids,
            "models_per_node": int(models_per_node),
            "pairs_for_node": pairs_for_node,
            "games_per_pair": int(args.games_per_pair),
            "seed": int(args.seed),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_response_tokens": int(args.max_response_tokens),
            "max_retries_per_turn": int(args.max_retries_per_turn),
            "max_plies": int(args.max_plies),
            "prompt_template_path": str(args.prompt_template_path),
            "server_port_base": int(args.server_port_base),
            "server_map_path": str(cluster_dir / "server_map.json"),
        },
        "timing": {
            "wall_time_s": float(t1 - t0),
            "infer_time_s": float(infer_time_s),
            "infer_positions": int(infer_positions),
            "infer_batches": int(infer_batches),
        },
        "results": {"by_pair": per_pair},
        "paths": {
            "moves_jsonl": str(moves_path),
            "games_jsonl": str(games_path),
            "games_pgn": str(pgn_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[rrd] wrote: {summary_path}", flush=True)
    print(f"[rrd] outputs under: {shard_dir}", flush=True)


if __name__ == "__main__":
    main()
