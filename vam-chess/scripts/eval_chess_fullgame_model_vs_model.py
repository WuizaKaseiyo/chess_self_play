#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
import chess.pgn
import httpx
from jinja2 import Template, meta
from openai import AsyncOpenAI, OpenAI

# Make repo-root imports work when invoked as `python scripts/xxx.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import recipe.chess.full_game_eval as fge


def _sanitize_run_id(s: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in (s or ""))
    return safe or "run"


def _read_checkpoints_by_run(path: Path) -> Dict[str, str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    out: Dict[str, str] = {}
    for k, v in obj.items():
        if not isinstance(k, str) or not k.strip():
            continue
        if not isinstance(v, str) or not v.strip():
            continue
        out[k.strip()] = v.strip()
    return out


def _unordered_pairs(items: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pairs.append((items[i], items[j]))
    return pairs


@dataclass
class _VllmServer:
    name: str
    base_url: str
    api_key: str
    served_model_name: str
    port: int
    cuda_visible_devices: str
    cmd: List[str]
    proc: subprocess.Popen[bytes]
    log_path: Path
    _log_fp: Any


def _parse_cuda_visible_devices(s: str) -> List[str]:
    raw = str(s or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts


def _kill_process_tree(proc: subprocess.Popen[Any], *, name: str, timeout_s: float = 60.0) -> None:
    if proc.poll() is not None:
        return

    # vLLM frequently spawns child processes (TP/DP workers). Ensure we terminate the whole
    # process group so jobs don't hang.
    pgid: Optional[int] = None
    try:
        pgid = os.getpgid(int(proc.pid))
    except Exception:
        pgid = None

    def _send(sig: int) -> None:
        if proc.poll() is not None:
            return
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except Exception:
            return

    _send(signal.SIGTERM)
    t0 = time.time()
    while proc.poll() is None and (time.time() - t0) < float(timeout_s):
        time.sleep(0.5)
    if proc.poll() is not None:
        return
    _send(signal.SIGKILL)


def _start_vllm_server(
    *,
    name: str,
    model: str,
    served_model_name: str,
    port: int,
    cuda_visible_devices: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
    trust_remote_code: bool,
    api_key: str,
    log_path: Path,
) -> _VllmServer:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "wb")
    base_url = f"http://127.0.0.1:{int(port)}/v1"

    tp = int(tensor_parallel_size)
    if tp <= 0:
        raise ValueError(f"tensor_parallel_size must be >= 1, got {tp}")

    visible = _parse_cuda_visible_devices(cuda_visible_devices)
    if not visible:
        raise ValueError(f"cuda_visible_devices is empty for server {name}")
    want_gpus = int(tp)
    if len(visible) != int(want_gpus):
        raise ValueError(
            f"server {name}: CUDA_VISIBLE_DEVICES has {len(visible)} GPUs ({cuda_visible_devices}), "
            f"but tp={tp} requires exactly {want_gpus} GPUs."
        )

    # Use `python -m ...` so we don't rely on `vllm` being on PATH.
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(int(port)),
        "--api-key",
        str(api_key),
        "--model",
        str(model),
        "--served-model-name",
        str(served_model_name),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(int(tensor_parallel_size)),
        "--gpu-memory-utilization",
        str(float(gpu_memory_utilization)),
        "--max-model-len",
        str(int(max_model_len)),
        "--disable-log-stats",
    ]
    cmd.append("--trust-remote-code" if trust_remote_code else "--no-trust-remote-code")
    cmd.append("--enforce-eager" if enforce_eager else "--no-enforce-eager")

    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(
        f"[mvm][server:{name}] starting port={port} cuda_visible_devices={cuda_visible_devices} served_model_name={served_model_name} tp={tp}",
        flush=True,
    )
    print(f"[mvm][server:{name}] cmd={' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, env=env, start_new_session=True)

    return _VllmServer(
        name=name,
        base_url=base_url,
        api_key=str(api_key),
        served_model_name=str(served_model_name),
        port=int(port),
        cuda_visible_devices=str(cuda_visible_devices),
        cmd=cmd,
        proc=proc,
        log_path=log_path,
        _log_fp=log_fp,
    )


def _wait_for_server_ready(server: _VllmServer, *, timeout_s: float = 600.0) -> None:
    client = OpenAI(base_url=server.base_url, api_key=server.api_key)
    t0 = time.time()
    last_err = ""
    while (time.time() - t0) < float(timeout_s):
        if server.proc.poll() is not None:
            raise RuntimeError(
                f"[mvm][server:{server.name}] exited early with code {server.proc.returncode}. Log: {server.log_path}"
            )
        try:
            models = client.models.list()
            model_ids = [m.id for m in (models.data or []) if getattr(m, "id", None)]
            if server.served_model_name in model_ids:
                print(f"[mvm][server:{server.name}] ready models={model_ids}", flush=True)
                return
            last_err = f"served_model_name {server.served_model_name} not in {model_ids}"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0)
    raise TimeoutError(f"[mvm][server:{server.name}] not ready after {timeout_s}s: {last_err}. Log: {server.log_path}")


class _AsyncRunner:
    def __init__(self, *, max_connections: int):
        self.max_connections = int(max_connections)
        if self.max_connections <= 0:
            raise ValueError(f"max_connections must be > 0, got {self.max_connections}")
        self._ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._thread = threading.Thread(target=self._thread_main, name="chess-rl-openai-async", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=30.0)
        if self._loop is None or self._http_client is None:
            raise RuntimeError("Async runner failed to start (loop/http_client not ready).")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        limits = httpx.Limits(
            max_connections=int(self.max_connections),
            max_keepalive_connections=int(self.max_connections),
        )
        self._http_client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(600.0))
        self._loop = loop
        self._ready.set()
        loop.run_forever()

    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            raise RuntimeError("Async runner not initialized (http_client missing).")
        return self._http_client

    def run(self, coro):  # noqa: ANN001
        if self._loop is None:
            raise RuntimeError("Async runner not initialized (loop missing).")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def shutdown(self, *, timeout_s: float = 10.0) -> None:
        loop = self._loop
        client = self._http_client
        if loop is None or client is None:
            return

        async def _close() -> None:
            await client.aclose()

        try:
            asyncio.run_coroutine_threadsafe(_close(), loop).result(timeout=float(timeout_s))
        except Exception:
            pass
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass


_ASYNC_RUNNER_LOCK = threading.Lock()
_ASYNC_RUNNER: Optional[_AsyncRunner] = None
_OPENAI_HTTP_MAX_CONNECTIONS = int(os.environ.get("CHESS_RL_OPENAI_HTTP_MAX_CONNECTIONS", "16384"))


def _get_async_runner(*, max_connections: int) -> _AsyncRunner:
    global _ASYNC_RUNNER  # noqa: PLW0603
    with _ASYNC_RUNNER_LOCK:
        if _ASYNC_RUNNER is None:
            _ASYNC_RUNNER = _AsyncRunner(max_connections=int(max_connections))

            def _shutdown() -> None:
                r = _ASYNC_RUNNER
                if r is not None:
                    r.shutdown(timeout_s=5.0)

            atexit.register(_shutdown)
        return _ASYNC_RUNNER


class OpenAIChatBackend:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        max_workers: int,
    ):
        self.name = str(name)
        self.base_url = str(base_url)
        self.api_key = str(api_key)
        self.model = str(model)
        self.max_workers = int(max_workers)
        self._seed_supported: Optional[bool] = None
        self._async_client: Optional[AsyncOpenAI] = None

    async def _one(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: Optional[int],
    ) -> str:
        def _err_text(exc: Exception) -> str:
            msg = str(exc).replace("\n", " ").strip()
            if len(msg) > 800:
                msg = msg[:800] + "…"
            return f"<error>{type(exc).__name__}: {msg}</error>"

        runner = _get_async_runner(max_connections=int(_OPENAI_HTTP_MAX_CONNECTIONS))
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                max_retries=2,
                http_client=runner.http_client,
            )

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }

        # vLLM's OpenAI server supports `seed` in some versions/configs; the official OpenAI API
        # accepts it as well. Probe once and fall back gracefully if unsupported.
        if seed is not None and self._seed_supported is not False:
            kwargs["seed"] = int(seed)

        try:
            resp = await self._async_client.chat.completions.create(**kwargs)
            self._seed_supported = True if ("seed" in kwargs) else self._seed_supported
        except Exception as exc:
            msg = str(exc).lower()
            if seed is not None and ("seed" in kwargs) and ("seed" in msg or "unrecognized" in msg or "unknown" in msg):
                # Retry without seed and remember.
                self._seed_supported = False
                kwargs.pop("seed", None)
                try:
                    resp = await self._async_client.chat.completions.create(**kwargs)
                except Exception as exc2:
                    return _err_text(exc2)
            else:
                return _err_text(exc)

        if not getattr(resp, "choices", None):
            return ""
        choice0 = resp.choices[0]
        msg0 = getattr(choice0, "message", None)
        content = getattr(msg0, "content", None)
        return content if isinstance(content, str) else ""

    def generate(
        self,
        prompts: List[List[Dict[str, str]]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seeds: Optional[List[int]] = None,
    ) -> List[str]:
        if seeds is not None and len(seeds) != len(prompts):
            raise ValueError(f"seeds length {len(seeds)} != batch size {len(prompts)}")

        n = len(prompts)
        if n == 0:
            return []

        max_in_flight = int(self.max_workers)
        if max_in_flight <= 0:
            max_in_flight = n
        max_in_flight = min(max_in_flight, n)
        runner = _get_async_runner(max_connections=int(_OPENAI_HTTP_MAX_CONNECTIONS))

        async def _run() -> List[str]:
            sem = asyncio.Semaphore(int(max_in_flight))

            async def _one_i(i: int, p: List[Dict[str, str]]) -> str:
                async with sem:
                    return await self._one(
                        p,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        seed=(seeds[i] if seeds is not None else None),
                    )

            tasks = [asyncio.create_task(_one_i(i, p)) for i, p in enumerate(prompts)]
            return list(await asyncio.gather(*tasks))

        return runner.run(_run())


class MultiServerChatBackend:
    def __init__(self, *, name: str, backends: List[OpenAIChatBackend]):
        if not backends:
            raise ValueError("MultiServerChatBackend requires at least one backend")
        self.name = str(name)
        self.backends = list(backends)

    def generate(
        self,
        prompts: List[List[Dict[str, str]]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seeds: Optional[List[int]] = None,
    ) -> List[str]:
        if seeds is not None and len(seeds) != len(prompts):
            raise ValueError(f"seeds length {len(seeds)} != batch size {len(prompts)}")

        n = len(prompts)
        if n == 0:
            return []

        # Simple deterministic load balancing: round-robin across replicas.
        groups: List[List[int]] = [[] for _ in range(len(self.backends))]
        for i in range(n):
            groups[i % len(self.backends)].append(i)

        out: List[str] = [""] * n

        import concurrent.futures as cf

        with cf.ThreadPoolExecutor(max_workers=len(self.backends)) as ex:
            futs = []
            for b_idx, idxs in enumerate(groups):
                if not idxs:
                    continue
                b = self.backends[b_idx]
                sub_prompts = [prompts[i] for i in idxs]
                sub_seeds = [seeds[i] for i in idxs] if seeds is not None else None
                futs.append((idxs, ex.submit(b.generate, sub_prompts, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seeds=sub_seeds)))

            for idxs, fut in futs:
                sub_out = fut.result()
                if len(sub_out) != len(idxs):
                    raise RuntimeError(f"[mvm][{self.name}] replica returned {len(sub_out)} outputs for {len(idxs)} prompts")
                for i, text in zip(idxs, sub_out):
                    out[i] = text

        return out


def _resolve_model_for_vllm_two_stage(
    *,
    model: str,
    out_dir: Path,
    trust_remote_code: bool,
    use_cpu_initialization: bool,
) -> str:
    """Resolve a model path/repo id for vLLM, tolerating global_step dirs.

    Reuses `scripts/eval_chess_fullgame.py::_resolve_model_for_vllm` and extends it by
    auto-trying `<path>/actor` when the provided path looks like a VERL global_step folder.
    """
    # Import lazily so `--enumerate-pairs` stays lightweight and doesn't import vLLM.
    from scripts.eval_chess_fullgame import _resolve_model_for_vllm

    try:
        return _resolve_model_for_vllm(
            model=model,
            out_dir=out_dir,
            trust_remote_code=trust_remote_code,
            use_cpu_initialization=use_cpu_initialization,
        )
    except ValueError:
        p = Path(model)
        if p.exists() and p.is_dir() and (p / "actor").is_dir():
            return _resolve_model_for_vllm(
                model=str(p / "actor"),
                out_dir=out_dir,
                trust_remote_code=trust_remote_code,
                use_cpu_initialization=use_cpu_initialization,
            )
        raise


def _set_forfeit_for_color(game: fge._GameState, *, reason: str, failed_color: chess.Color) -> None:
    game.is_over = True
    game.forfeit = True
    game.forfeit_reason = reason
    game.termination = "resignation"
    game.result = "0-1" if failed_color == chess.WHITE else "1-0"
    game.result_str = "Black wins (White resigned)" if failed_color == chess.WHITE else "White wins (Black resigned)"


def _step_agent_moves(
    *,
    cfg: fge.FullGameEvalConfig,
    backend: Any,
    games: List[fge._GameState],
    moves_fp,
    prompt_template: Optional[Template],
    prompt_template_vars: Optional[set[str]],
    actor_label: str,
    actor_run_id: str,
    opponent_run_id: str,
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
        for g in todo:
            legal_moves = list(g.board.legal_moves)
            if prompt_template is None:
                raise ValueError("prompt_template is required for model-vs-model eval (keep prompts aligned).")

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

        seeds = [
            fge._safe_int_seed(fge._mix_seed(cfg.seed, salt=f"{actor_label}|{g.game_id}|ply={len(g.board.move_stack)}|try={retry_idx}"))
            for g in todo
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

        dt = float(t1 - t0)
        infer_time_s += dt
        infer_positions += int(len(todo))
        infer_batches += 1

        if len(outputs) != len(todo):
            raise RuntimeError(f"[mvm][{actor_label}] backend returned {len(outputs)} outputs for {len(todo)} prompts")

        for g, prompt_text, output_text in zip(todo, prompt_texts, outputs):
            ts = time.time()
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
                "ply": int(len(g.board.move_stack)),
                "fen": g.board.fen(),
                "actor": actor_label,
                "actor_run_id": actor_run_id,
                "opponent_run_id": opponent_run_id,
                "side_to_move": "white" if g.board.turn == chess.WHITE else "black",
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

    # Any remaining pending games forfeit (resignation).
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
                "ply": int(len(g.board.move_stack)),
                "fen": g.board.fen(),
                "actor": actor_label,
                "actor_run_id": actor_run_id,
                "opponent_run_id": opponent_run_id,
                "side_to_move": "white" if failed_color == chess.WHITE else "black",
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


def _init_mvm_games(
    *,
    run_id_a: str,
    run_id_b: str,
    games_total: int,
    seed: int,
) -> List[fge._GameState]:
    n = int(games_total)
    if n <= 0:
        raise ValueError("--games-total must be > 0")
    if n % 2 != 0:
        raise ValueError("--games-total must be even for strict color-balancing")

    # Model A plays White in exactly half the games, Black in the other half.
    colors: List[chess.Color] = [chess.WHITE] * (n // 2) + [chess.BLACK] * (n // 2)
    rng = fge._DeterministicRng(seed, salt="mvm_color_shuffle")
    rng.shuffle(colors)

    games: List[fge._GameState] = []
    for i, a_color in enumerate(colors):
        board = chess.Board()
        pgn = chess.pgn.Game()
        pgn.headers["Event"] = "chess-rl full-game eval (model-vs-model)"
        pgn.headers["Site"] = "local"
        pgn.headers["Date"] = time.strftime("%Y.%m.%d")

        if a_color == chess.WHITE:
            pgn.headers["White"] = run_id_a
            pgn.headers["Black"] = run_id_b
        else:
            pgn.headers["White"] = run_id_b
            pgn.headers["Black"] = run_id_a

        game_id = f"g{i:03d}"
        games.append(
            fge._GameState(
                game_id=game_id,
                opponent_depth=0,
                model_color=a_color,
                board=board,
                pgn=pgn,
                pgn_node=pgn,
                round_idx=0,
                game_idx_in_round=i,
            )
        )

    return games


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model-vs-model full-game chess evaluation (vLLM servers + Stockfish ACPL).")

    p.add_argument(
        "--checkpoints-json",
        type=str,
        default="checkpoints_by_run.json",
        help="Path to checkpoints_by_run.json (run_id -> checkpoint path).",
    )
    p.add_argument("--enumerate-pairs", action="store_true", help="Print all unordered run-id pairs and exit.")
    p.add_argument(
        "--emit-sbatch-commands",
        action="store_true",
        help="Print one sbatch command per unordered pair (does not run them).",
    )
    p.add_argument(
        "--emit-sbatch-script",
        type=str,
        default="./sbatch_eval_chess_fullgame_model_vs_model_gh200.slurm",
        help="Slurm script to reference when emitting sbatch commands.",
    )
    p.add_argument(
        "--emit-out-root",
        type=str,
        default="",
        help="Optional OUT_ROOT override to include in emitted sbatch commands (avoid commas).",
    )
    p.add_argument(
        "--emit-games-total",
        type=int,
        default=100,
        help="GAMES_TOTAL to include in emitted sbatch commands (default: 100).",
    )

    g = p.add_argument_group("Match selection")
    g.add_argument("--run-id-a", type=str, default="", help="Run id for model A (key in checkpoints_by_run.json).")
    g.add_argument("--run-id-b", type=str, default="", help="Run id for model B (key in checkpoints_by_run.json).")
    g.add_argument("--model-a", type=str, default="", help="Override: HF repo id or local path for model A.")
    g.add_argument("--model-b", type=str, default="", help="Override: HF repo id or local path for model B.")

    p.add_argument("--out-dir", type=str, default="", help="Output directory (default: outputs/full_game_eval_mvm/<run>/)")
    p.add_argument(
        "--prompt-template-path",
        type=str,
        default="recipe/chess/prompt_templates/select_prompt.jinja",
        help="Jinja prompt template path (must match training prompt contract).",
    )

    # Match / sampling.
    p.add_argument("--games-total", type=int, default=10, help="Total games (must be even for strict color balance).")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-response-tokens", type=int, default=2000)
    p.add_argument("--max-retries-per-turn", type=int, default=1)
    p.add_argument("--max-plies", type=int, default=200, help="Max game length in plies (0 disables).")

    # ACPL / Stockfish (mirror training-time full-eval defaults on GH200).
    p.add_argument("--stockfish-path", type=str, default="/usr/local/bin/stockfish")
    p.add_argument("--stockfish-hash-mb", type=int, default=128)
    p.add_argument("--acpl-depth", type=int, default=20)
    p.add_argument("--acpl-movetime-ms", type=int, default=1000)
    p.add_argument("--acpl-cp-cap", type=int, default=1000)
    p.add_argument("--mate-score-cp", type=int, default=1000)
    p.add_argument("--resignation-cpl", type=int, default=1000)
    p.add_argument("--acpl-workers", type=int, default=72)
    p.add_argument("--acpl-threads", type=int, default=2)

    # vLLM server knobs (2 GPUs per model).
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Number of vLLM server replicas per model (data parallelism). Total GPUs per model = tensor_parallel_size * data_parallel_size.",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--enforce-eager", action="store_true", help="Force eager mode (disable CUDA graphs).")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--merge-use-cpu-initialization", action="store_true")

    p.add_argument("--server-port-a", type=int, default=8000)
    p.add_argument("--server-port-b", type=int, default=8001)
    p.add_argument("--server-gpus-a", type=str, default="0,1", help="CUDA_VISIBLE_DEVICES for model A server.")
    p.add_argument("--server-gpus-b", type=str, default="2,3", help="CUDA_VISIBLE_DEVICES for model B server.")
    p.add_argument("--server-ready-timeout-s", type=float, default=900.0)
    p.add_argument(
        "--backend-max-workers",
        type=int,
        default=0,
        help="Max in-flight OpenAI requests per server (0 = no explicit client-side cap).",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    checkpoints_path = Path(args.checkpoints_json)
    ckpts = _read_checkpoints_by_run(checkpoints_path)
    run_ids_sorted = sorted(ckpts.keys())

    if args.enumerate_pairs:
        pairs = _unordered_pairs(run_ids_sorted)
        print(json.dumps({"num_runs": len(run_ids_sorted), "num_pairs": len(pairs), "pairs": pairs}, indent=2))
        return
    if args.emit_sbatch_commands:
        pairs = _unordered_pairs(run_ids_sorted)
        export_kvs = [
            f"TEMPERATURE={float(args.temperature)}",
            f"TOP_P={float(args.top_p)}",
            f"MAX_RESPONSE_TOKENS={int(args.max_response_tokens)}",
            f"MAX_RETRIES_PER_TURN={int(args.max_retries_per_turn)}",
            f"TENSOR_PARALLEL_SIZE={int(args.tensor_parallel_size)}",
            f"DATA_PARALLEL_SIZE={int(args.data_parallel_size)}",
            f"GAMES_TOTAL={int(args.emit_games_total)}",
        ]
        if str(args.emit_out_root).strip():
            if "," in str(args.emit_out_root):
                raise ValueError("--emit-out-root must not contain commas (sbatch --export uses commas).")
            export_kvs.append(f"OUT_ROOT={str(args.emit_out_root).strip()}")

        sbatch_script = str(args.emit_sbatch_script).strip() or "./sbatch_eval_chess_fullgame_model_vs_model_gh200.slurm"

        print(f"# num_runs={len(run_ids_sorted)} num_pairs={len(pairs)}")
        print("# NOTE: This prints commands only (no submission).")
        print("# NOTE: Avoid passing values containing commas via sbatch --export (e.g., SERVER_GPUS_A/B).")
        for run_a, run_b in pairs:
            kvs = [f"RUN_ID_A={run_a}", f"RUN_ID_B={run_b}", *export_kvs]
            export_arg = "ALL," + ",".join(kvs)
            print(f"sbatch --wait --export={export_arg} {sbatch_script}")
        return

    # Resolve models.
    run_id_a = str(args.run_id_a).strip()
    run_id_b = str(args.run_id_b).strip()

    if (not run_id_a) != (not run_id_b):
        raise ValueError("Provide both --run-id-a and --run-id-b, or neither (use --model-a/--model-b).")

    model_a_in = str(args.model_a).strip()
    model_b_in = str(args.model_b).strip()

    if run_id_a and run_id_b and (model_a_in or model_b_in):
        raise ValueError("Use either --run-id-{a,b} or --model-{a,b}, not both.")

    if run_id_a and run_id_b:
        if run_id_a not in ckpts:
            raise KeyError(f"Unknown run-id-a={run_id_a}. Known: {run_ids_sorted}")
        if run_id_b not in ckpts:
            raise KeyError(f"Unknown run-id-b={run_id_b}. Known: {run_ids_sorted}")
        model_a_in = ckpts[run_id_a]
        model_b_in = ckpts[run_id_b]
    else:
        if not model_a_in or not model_b_in:
            raise ValueError("Provide either --run-id-a/--run-id-b or --model-a/--model-b.")
        if not run_id_a:
            run_id_a = _sanitize_run_id(Path(model_a_in).name or "model_a")
        if not run_id_b:
            run_id_b = _sanitize_run_id(Path(model_b_in).name or "model_b")

    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_a = _sanitize_run_id(run_id_a)
    safe_b = _sanitize_run_id(run_id_b)
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/full_game_eval_mvm") / f"{safe_a}_vs_{safe_b}_{ts}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist invocation for reproducibility.
    run_args_path = out_dir / "run_args.json"
    run_args_path.write_text(
        json.dumps(
            {
                "argv": list(sys.argv),
                "args": vars(args),
                "resolved": {
                    "run_id_a": run_id_a,
                    "run_id_b": run_id_b,
                    "model_a_in": model_a_in,
                    "model_b_in": model_b_in,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[mvm] Wrote run args: {run_args_path}", flush=True)

    # Prompt template (required for aligned evaluation).
    template_text = Path(args.prompt_template_path).read_text(encoding="utf-8")
    prompt_template = Template(template_text)
    try:
        prompt_template_vars = set(meta.find_undeclared_variables(prompt_template.environment.parse(template_text)))
    except Exception:
        prompt_template_vars = None

    # Resolve both models for vLLM (merge FSDP -> HF when needed).
    model_a_dir = out_dir / "model_a"
    model_b_dir = out_dir / "model_b"
    model_a_dir.mkdir(parents=True, exist_ok=True)
    model_b_dir.mkdir(parents=True, exist_ok=True)

    model_a_for_vllm = _resolve_model_for_vllm_two_stage(
        model=model_a_in,
        out_dir=model_a_dir,
        trust_remote_code=bool(args.trust_remote_code),
        use_cpu_initialization=bool(args.merge_use_cpu_initialization),
    )
    model_b_for_vllm = _resolve_model_for_vllm_two_stage(
        model=model_b_in,
        out_dir=model_b_dir,
        trust_remote_code=bool(args.trust_remote_code),
        use_cpu_initialization=bool(args.merge_use_cpu_initialization),
    )
    print(f"[mvm] model_a_for_vllm={model_a_for_vllm}", flush=True)
    print(f"[mvm] model_b_for_vllm={model_b_for_vllm}", flush=True)

    # Start vLLM servers.
    api_key = "dummy"
    data_parallel_size = int(args.data_parallel_size)
    tensor_parallel_size = int(args.tensor_parallel_size)
    if data_parallel_size <= 0:
        raise ValueError("--data-parallel-size must be >= 1")
    if tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be >= 1")

    # We implement DP by running multiple independent vLLM servers per model, each pinned
    # to its own GPU slice. This avoids relying on vLLM's in-process DP (which has been
    # unstable for some workloads/models on GH200).
    gpus_a = _parse_cuda_visible_devices(str(args.server_gpus_a))
    gpus_b = _parse_cuda_visible_devices(str(args.server_gpus_b))

    want_gpus_per_model = int(tensor_parallel_size) * int(data_parallel_size)
    if len(gpus_a) != want_gpus_per_model:
        raise ValueError(
            f"server-gpus-a has {len(gpus_a)} GPUs ({args.server_gpus_a}), but tp={tensor_parallel_size} "
            f"dp={data_parallel_size} requires exactly {want_gpus_per_model} GPUs."
        )
    if len(gpus_b) != want_gpus_per_model:
        raise ValueError(
            f"server-gpus-b has {len(gpus_b)} GPUs ({args.server_gpus_b}), but tp={tensor_parallel_size} "
            f"dp={data_parallel_size} requires exactly {want_gpus_per_model} GPUs."
        )

    def _chunks(gpus: List[str]) -> List[str]:
        out: List[str] = []
        for i in range(int(data_parallel_size)):
            chunk = gpus[i * int(tensor_parallel_size) : (i + 1) * int(tensor_parallel_size)]
            out.append(",".join(chunk))
        return out

    gpu_chunks_a = _chunks(gpus_a)
    gpu_chunks_b = _chunks(gpus_b)
    ports_a = [int(args.server_port_a) + i for i in range(int(data_parallel_size))]
    ports_b = [int(args.server_port_b) + i for i in range(int(data_parallel_size))]

    if set(ports_a).intersection(set(ports_b)):
        raise ValueError(
            f"Port ranges overlap for dp={data_parallel_size}: A ports={ports_a} B ports={ports_b}. "
            f"Set --server-port-b to a non-overlapping base (e.g., 8100)."
        )

    servers: List[_VllmServer] = []
    servers_a: List[_VllmServer] = []
    servers_b: List[_VllmServer] = []

    for r in range(int(data_parallel_size)):
        s = _start_vllm_server(
            name=f"A{r}",
            model=model_a_for_vllm,
            served_model_name=f"model-a-r{r}",
            port=int(ports_a[r]),
            cuda_visible_devices=str(gpu_chunks_a[r]),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=int(args.max_model_len),
            enforce_eager=bool(args.enforce_eager),
            trust_remote_code=bool(args.trust_remote_code),
            api_key=api_key,
            log_path=out_dir / "logs" / f"vllm_server_a_r{r}.log",
        )
        servers.append(s)
        servers_a.append(s)

    for r in range(int(data_parallel_size)):
        s = _start_vllm_server(
            name=f"B{r}",
            model=model_b_for_vllm,
            served_model_name=f"model-b-r{r}",
            port=int(ports_b[r]),
            cuda_visible_devices=str(gpu_chunks_b[r]),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=int(args.max_model_len),
            enforce_eager=bool(args.enforce_eager),
            trust_remote_code=bool(args.trust_remote_code),
            api_key=api_key,
            log_path=out_dir / "logs" / f"vllm_server_b_r{r}.log",
        )
        servers.append(s)
        servers_b.append(s)

    def _shutdown_servers() -> None:
        for s in servers:
            try:
                _kill_process_tree(s.proc, name=s.name)
            except Exception:
                pass
            try:
                s._log_fp.close()
            except Exception:
                pass

    atexit.register(_shutdown_servers)

    def _handle_sigterm(signum, frame) -> None:  # noqa: ANN001
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    try:
        for s in servers:
            _wait_for_server_ready(s, timeout_s=float(args.server_ready_timeout_s))

        per_replica_workers = max(1, int(args.backend_max_workers) // max(1, int(data_parallel_size)))

        backend_a = MultiServerChatBackend(
            name="A",
            backends=[
                OpenAIChatBackend(
                    name=f"A{r}",
                    base_url=s.base_url,
                    api_key=s.api_key,
                    model=s.served_model_name,
                    max_workers=per_replica_workers,
                )
                for r, s in enumerate(servers_a)
            ],
        )
        backend_b = MultiServerChatBackend(
            name="B",
            backends=[
                OpenAIChatBackend(
                    name=f"B{r}",
                    base_url=s.base_url,
                    api_key=s.api_key,
                    model=s.served_model_name,
                    max_workers=per_replica_workers,
                )
                for r, s in enumerate(servers_b)
            ],
        )

        # Config shared with existing ACPL helpers.
        cfg = fge.FullGameEvalConfig(
            opponent_depths=[0],
            games_per_depth=int(args.games_total),
            seed=int(args.seed),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_response_tokens=int(args.max_response_tokens),
            max_retries_per_turn=int(args.max_retries_per_turn),
            resignation_cpl=int(args.resignation_cpl),
            acpl_eval_depth=int(args.acpl_depth),
            acpl_eval_movetime_ms=int(args.acpl_movetime_ms),
            acpl_eval_cp_cap=int(args.acpl_cp_cap),
            mate_score_cp=int(args.mate_score_cp),
            max_plies=(int(args.max_plies) if int(args.max_plies) > 0 else None),
            stockfish_opponent=fge.StockfishConfig(
                path=str(args.stockfish_path),
                threads=int(args.acpl_threads),
                hash_mb=int(args.stockfish_hash_mb),
                skill_level=0,
            ),
            stockfish_eval=fge.StockfishConfig(
                path=str(args.stockfish_path),
                threads=int(args.acpl_threads),
                hash_mb=int(args.stockfish_hash_mb),
                skill_level=20,
            ),
            acpl_workers=int(args.acpl_workers),
            prompt_template_path=str(args.prompt_template_path),
            out_dir=out_dir,
        )

        moves_path = out_dir / "moves.jsonl"
        games_path = out_dir / "games.jsonl"
        summary_path = out_dir / "summary.json"
        pgn_path = out_dir / "games.pgn"

        games = _init_mvm_games(
            run_id_a=run_id_a,
            run_id_b=run_id_b,
            games_total=int(args.games_total),
            seed=int(args.seed),
        )

        forfeit_color_by_game: Dict[str, chess.Color] = {}

        t_eval0 = time.time()
        infer_time_s = 0.0
        infer_positions = 0
        infer_batches = 0
        with (
            open(moves_path, "w", encoding="utf-8") as moves_fp,
            open(games_path, "w", encoding="utf-8") as games_fp,
            open(pgn_path, "w", encoding="utf-8") as pgn_fp,
        ):
            # Thread-safe writer for moves.jsonl so we can overlap A/B move generation.
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

            active = [g for g in games if not g.is_over]
            while active:
                fge._enforce_max_plies(active, max_plies=cfg.max_plies)

                a_turn_games = [g for g in active if (not g.is_over) and g.board.turn == g.model_color]
                b_turn_games = [g for g in active if (not g.is_over) and g.board.turn != g.model_color]
                if a_turn_games and b_turn_games:
                    # Crucial speedup: overlap model A and model B inference since they run on disjoint GPUs.
                    import concurrent.futures as cf

                    with cf.ThreadPoolExecutor(max_workers=2) as ex:
                        fut_a = ex.submit(
                            _step_agent_moves,
                            cfg=cfg,
                            backend=backend_a,
                            games=a_turn_games,
                            moves_fp=moves_fp_locked,
                            prompt_template=prompt_template,
                            prompt_template_vars=prompt_template_vars,
                            actor_label="A",
                            actor_run_id=run_id_a,
                            opponent_run_id=run_id_b,
                            forfeit_color_by_game=forfeit_color_by_game,
                        )
                        fut_b = ex.submit(
                            _step_agent_moves,
                            cfg=cfg,
                            backend=backend_b,
                            games=b_turn_games,
                            moves_fp=moves_fp_locked,
                            prompt_template=prompt_template,
                            prompt_template_vars=prompt_template_vars,
                            actor_label="B",
                            actor_run_id=run_id_b,
                            opponent_run_id=run_id_a,
                            forfeit_color_by_game=forfeit_color_by_game,
                        )
                        stats_a = fut_a.result()
                        stats_b = fut_b.result()

                    for stats in (stats_a, stats_b):
                        infer_time_s += float(stats.get("infer_time_s", 0.0) or 0.0)
                        infer_positions += int(stats.get("infer_positions", 0) or 0)
                        infer_batches += int(stats.get("infer_batches", 0) or 0)
                else:
                    if a_turn_games:
                        stats = _step_agent_moves(
                            cfg=cfg,
                            backend=backend_a,
                            games=a_turn_games,
                            moves_fp=moves_fp_locked,
                            prompt_template=prompt_template,
                            prompt_template_vars=prompt_template_vars,
                            actor_label="A",
                            actor_run_id=run_id_a,
                            opponent_run_id=run_id_b,
                            forfeit_color_by_game=forfeit_color_by_game,
                        )
                        infer_time_s += float(stats.get("infer_time_s", 0.0) or 0.0)
                        infer_positions += int(stats.get("infer_positions", 0) or 0)
                        infer_batches += int(stats.get("infer_batches", 0) or 0)

                    if b_turn_games:
                        stats = _step_agent_moves(
                            cfg=cfg,
                            backend=backend_b,
                            games=b_turn_games,
                            moves_fp=moves_fp_locked,
                            prompt_template=prompt_template,
                            prompt_template_vars=prompt_template_vars,
                            actor_label="B",
                            actor_run_id=run_id_b,
                            opponent_run_id=run_id_a,
                            forfeit_color_by_game=forfeit_color_by_game,
                        )
                        infer_time_s += float(stats.get("infer_time_s", 0.0) or 0.0)
                        infer_positions += int(stats.get("infer_positions", 0) or 0)
                        infer_batches += int(stats.get("infer_batches", 0) or 0)

                # Mirror starter-kit-style ply cap enforcement between turns.
                fge._enforce_max_plies(active, max_plies=cfg.max_plies)

                active = [g for g in active if not g.is_over]

            # Engine analysis (ACPL / accuracy).
            t_acpl0 = time.time()
            analyses = fge._analyze_games_with_engine(cfg=cfg, eval_engine=None, games=games)
            t_acpl1 = time.time()
            acpl_time_s = float(t_acpl1 - t_acpl0)

            # Flush games to games.jsonl and combined PGN.
            model_a_wins = 0
            model_a_losses = 0
            model_a_draws = 0

            model_a_acpl_sum_per_game = 0.0
            model_a_acpl_games = 0
            model_a_cpl_sum = 0.0
            model_a_moves = 0

            model_b_acpl_sum_per_game = 0.0
            model_b_acpl_games = 0
            model_b_cpl_sum = 0.0
            model_b_moves = 0

            resignation_penalty = float(cfg.resignation_cpl or 0)

            for g in games:
                if g.pgn.headers.get("Result", "*") in ("", "*"):
                    g.pgn.headers["Result"] = g.result
                if not g.result_str:
                    g.result_str = fge._pgn_result_to_result_str(g.result)

                analysis = analyses.get(g.game_id, None) or {}
                white_acpl = float(analysis.get("white_acpl", float(cfg.acpl_eval_cp_cap)))
                black_acpl = float(analysis.get("black_acpl", float(cfg.acpl_eval_cp_cap)))
                white_accuracy_pct = float(analysis.get("white_accuracy_pct", 0.0))
                black_accuracy_pct = float(analysis.get("black_accuracy_pct", 0.0))
                white_moves = int(analysis.get("white_moves", 0))
                black_moves = int(analysis.get("black_moves", 0))
                white_cpl_sum = float(analysis.get("white_cpl_sum", 0.0))
                black_cpl_sum = float(analysis.get("black_cpl_sum", 0.0))

                resignation_penalty_white = 0.0
                resignation_penalty_black = 0.0
                if g.forfeit and resignation_penalty > 0:
                    forfeited_color = forfeit_color_by_game.get(g.game_id, None)
                    if forfeited_color == chess.WHITE:
                        resignation_penalty_white = float(resignation_penalty)
                        white_cpl_sum += resignation_penalty
                        if white_moves <= 0:
                            white_moves = 1
                        white_acpl = float(white_cpl_sum / white_moves)
                    elif forfeited_color == chess.BLACK:
                        resignation_penalty_black = float(resignation_penalty)
                        black_cpl_sum += resignation_penalty
                        if black_moves <= 0:
                            black_moves = 1
                        black_acpl = float(black_cpl_sum / black_moves)

                # Persist per-side metrics.
                g.white_acpl = float(white_acpl)
                g.black_acpl = float(black_acpl)
                g.white_accuracy_pct = float(white_accuracy_pct)
                g.black_accuracy_pct = float(black_accuracy_pct)

                # Attribute to model A/B based on color assignment.
                a_is_white = g.model_color == chess.WHITE
                model_a_color = "white" if a_is_white else "black"
                model_b_color = "black" if a_is_white else "white"

                model_a_game_acpl = float(white_acpl if a_is_white else black_acpl)
                model_a_game_moves = int(white_moves if a_is_white else black_moves)
                model_a_game_cpl_sum = float(white_cpl_sum if a_is_white else black_cpl_sum)
                model_a_game_acc = float(white_accuracy_pct if a_is_white else black_accuracy_pct)

                model_b_game_acpl = float(black_acpl if a_is_white else white_acpl)
                model_b_game_moves = int(black_moves if a_is_white else white_moves)
                model_b_game_cpl_sum = float(black_cpl_sum if a_is_white else white_cpl_sum)
                model_b_game_acc = float(black_accuracy_pct if a_is_white else white_accuracy_pct)

                # W/D/L for model A.
                if g.result == "1/2-1/2":
                    model_a_draws += 1
                else:
                    a_won = (g.result == "1-0" and a_is_white) or (g.result == "0-1" and (not a_is_white))
                    if a_won:
                        model_a_wins += 1
                    else:
                        model_a_losses += 1

                model_a_acpl_sum_per_game += float(model_a_game_acpl)
                model_a_acpl_games += 1
                model_a_cpl_sum += float(model_a_game_cpl_sum)
                model_a_moves += int(model_a_game_moves)

                model_b_acpl_sum_per_game += float(model_b_game_acpl)
                model_b_acpl_games += 1
                model_b_cpl_sum += float(model_b_game_cpl_sum)
                model_b_moves += int(model_b_game_moves)

                pgn_text = str(g.pgn).strip()
                if pgn_text:
                    if pgn_fp.tell() > 0:
                        pgn_fp.write("\n\n")
                    pgn_fp.write(pgn_text)

                fge._jsonl_write(
                    games_fp,
                    {
                        "ts": time.time(),
                        "game_id": g.game_id,
                        "run_id_a": run_id_a,
                        "run_id_b": run_id_b,
                        "model_a_color": model_a_color,
                        "model_b_color": model_b_color,
                        "result": g.result_str or g.result,
                        "pgn_result": g.result,
                        "termination": g.termination,
                        "engine_error": g.engine_error,
                        "forfeit": bool(g.forfeit),
                        "forfeit_reason": g.forfeit_reason,
                        "num_plies": int(len(g.board.move_stack)),
                        "white_acpl": float(white_acpl),
                        "black_acpl": float(black_acpl),
                        "white_accuracy_pct": float(white_accuracy_pct),
                        "black_accuracy_pct": float(black_accuracy_pct),
                        "resignation_penalty_white": float(resignation_penalty_white),
                        "resignation_penalty_black": float(resignation_penalty_black),
                        "model_a_acpl": float(model_a_game_acpl),
                        "model_a_accuracy_pct": float(model_a_game_acc),
                        "model_a_cpl_sum": float(model_a_game_cpl_sum),
                        "model_a_moves": int(model_a_game_moves),
                        "model_b_acpl": float(model_b_game_acpl),
                        "model_b_accuracy_pct": float(model_b_game_acc),
                        "model_b_cpl_sum": float(model_b_game_cpl_sum),
                        "model_b_moves": int(model_b_game_moves),
                        "pgn": pgn_text,
                    },
                )

        t_eval1 = time.time()

        summary: Dict[str, Any] = {
            "config": {
                "run_id_a": run_id_a,
                "run_id_b": run_id_b,
                "model_a": {"requested": model_a_in, "resolved_for_vllm": model_a_for_vllm},
                "model_b": {"requested": model_b_in, "resolved_for_vllm": model_b_for_vllm},
                "prompt_template_path": str(args.prompt_template_path),
                "games_total": int(args.games_total),
                "seed": int(args.seed),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "max_response_tokens": int(args.max_response_tokens),
                "max_retries_per_turn": int(args.max_retries_per_turn),
                "max_plies": int(args.max_plies),
                "server_a": {
                    "base_ports": ports_a,
                    "cuda_visible_devices": str(args.server_gpus_a),
                    "tensor_parallel_size": int(args.tensor_parallel_size),
                    "data_parallel_size": int(args.data_parallel_size),
                    "replicas": [
                        {
                            "name": s.name,
                            "base_url": s.base_url,
                            "port": s.port,
                            "cuda_visible_devices": s.cuda_visible_devices,
                            "served_model_name": s.served_model_name,
                            "log_path": str(s.log_path),
                        }
                        for s in servers_a
                    ],
                },
                "server_b": {
                    "base_ports": ports_b,
                    "cuda_visible_devices": str(args.server_gpus_b),
                    "tensor_parallel_size": int(args.tensor_parallel_size),
                    "data_parallel_size": int(args.data_parallel_size),
                    "replicas": [
                        {
                            "name": s.name,
                            "base_url": s.base_url,
                            "port": s.port,
                            "cuda_visible_devices": s.cuda_visible_devices,
                            "served_model_name": s.served_model_name,
                            "log_path": str(s.log_path),
                        }
                        for s in servers_b
                    ],
                },
                "stockfish": {
                    "path": str(args.stockfish_path),
                    "hash_mb": int(args.stockfish_hash_mb),
                    "acpl_eval_depth": int(args.acpl_depth),
                    "acpl_eval_movetime_ms": int(args.acpl_movetime_ms),
                    "acpl_eval_cp_cap": int(args.acpl_cp_cap),
                    "mate_score_cp": int(args.mate_score_cp),
                    "resignation_cpl": int(args.resignation_cpl),
                    "acpl_workers": int(args.acpl_workers),
                    "acpl_threads": int(args.acpl_threads),
                },
            },
            "paths": {
                "moves_jsonl": str(moves_path),
                "games_jsonl": str(games_path),
                "games_pgn": str(pgn_path),
                "summary_json": str(summary_path),
                "run_args_json": str(run_args_path),
                "vllm_server_a_logs": [str(s.log_path) for s in servers_a],
                "vllm_server_b_logs": [str(s.log_path) for s in servers_b],
            },
            "timing": {
                "wall_time_s": float(t_eval1 - t_eval0),
                "infer_time_s": float(infer_time_s),
                "infer_positions": int(infer_positions),
                "infer_batches": int(infer_batches),
                "acpl_time_s": float(acpl_time_s),
            },
            "results": {
                "model_a": {
                    "wins": int(model_a_wins),
                    "losses": int(model_a_losses),
                    "draws": int(model_a_draws),
                    "win_rate": float(model_a_wins / int(args.games_total)) if int(args.games_total) > 0 else float("nan"),
                    "acpl_mean": float(model_a_acpl_sum_per_game / model_a_acpl_games)
                    if model_a_acpl_games > 0
                    else float("nan"),
                    "acpl_mean_per_move": float(model_a_cpl_sum / model_a_moves) if model_a_moves > 0 else float("nan"),
                    "acpl_moves": int(model_a_moves),
                },
                "model_b": {
                    "wins": int(model_a_losses),
                    "losses": int(model_a_wins),
                    "draws": int(model_a_draws),
                    "win_rate": float(model_a_losses / int(args.games_total)) if int(args.games_total) > 0 else float("nan"),
                    "acpl_mean": float(model_b_acpl_sum_per_game / model_b_acpl_games)
                    if model_b_acpl_games > 0
                    else float("nan"),
                    "acpl_mean_per_move": float(model_b_cpl_sum / model_b_moves) if model_b_moves > 0 else float("nan"),
                    "acpl_moves": int(model_b_moves),
                },
            },
        }

        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[mvm] Wrote summary: {summary_path}", flush=True)
        print(f"[mvm] Results model_a={summary['results']['model_a']}", flush=True)
        print(f"[mvm] Results model_b={summary['results']['model_b']}", flush=True)
    finally:
        _shutdown_servers()


if __name__ == "__main__":
    main()
