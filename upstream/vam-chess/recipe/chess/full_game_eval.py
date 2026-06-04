from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import chess
import chess.engine
import chess.pgn
from jinja2 import Template, meta

from recipe.chess.reward_fn import UCI_MOVE_ONLY_RE


DEFAULT_SYSTEM_PROMPT_UCI_MOVE = (
    "Chess move-prediction task.\n"
    "\n"
    "You will be given:\n"
    "- A chess position as a FEN string, and\n"
    "- The list of legal moves in strict UCI.\n"
    "\n"
    "Your task: choose the single best move from the provided legal move list.\n"
    "\n"
    "Output contract (required):\n"
    "- Output exactly <think>...</think><uci_move>...</uci_move> and no other text.\n"
    "- The <uci_move> payload must be exactly one move from the provided legal move list.\n"
    "- Use strict UCI: from-square + to-square (+ optional promotion letter q/r/b/n).\n"
)


class ChatBackend(Protocol):
    """Abstraction for high-throughput batched prompt generation.

    Contract:
      - input: list of message lists whose content is the rendered prompt
      - output: list of decoded response strings (one per prompt)
    """

    def generate(
        self,
        prompts: List[List[Dict[str, str]]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seeds: Optional[List[int]] = None,
    ) -> List[str]: ...


@dataclass(frozen=True)
class StockfishConfig:
    path: str
    # Match starter-kit defaults. Override for speed if desired.
    threads: int = 1
    hash_mb: int = 128
    skill_level: int = 0


@dataclass(frozen=True)
class FullGameEvalConfig:
    opponent_depths: List[int]
    games_per_depth: int
    seed: int
    # "Rounds" are the competition-style way to describe repeating the evaluation
    # with independent game IDs (still starting from the initial position).
    # In this repo we still schedule all games in-flight concurrently; rounds are
    # mainly for bookkeeping and default sizing (5 rounds * 50 games = 250 games).
    rounds: Optional[int] = None
    games_per_round: Optional[int] = None

    temperature: float = 0.6
    top_p: float = 0.95
    max_response_tokens: int = 512
    max_retries_per_turn: int = 3
    opponent_movetime_ms: int = 100
    # Kept for backward-compat with older training configs. The starter-kit ACPL
    # computation does not add an explicit resignation penalty term; resignation
    # impacts results primarily via win-rate and (if no moves were played) the
    # `ACPL=cp_cap` fallback.
    resignation_cpl: int = 1000

    # ACPL reference engine settings.
    acpl_eval_depth: int = 20
    acpl_eval_movetime_ms: int = 1000
    acpl_eval_cp_cap: int = 1000
    mate_score_cp: int = 1000
    # Parallelize ACPL over completed games using multiple Stockfish processes.
    # - Each worker spawns its own Stockfish engine configured with `stockfish_eval` options.
    # - Use spawn() to avoid forking GPU/vLLM state (safe for large in-process inference evals).
    acpl_workers: int = 1
    # Retry failed ACPL worker shards before failing the evaluation.
    acpl_worker_retries: int = 3
    acpl_worker_retry_backoff_s: float = 2.0

    # Game termination (starter-kit uses `max_moves=200` plies by default).
    max_plies: Optional[int] = 200

    stockfish_opponent: StockfishConfig = StockfishConfig(path=".third_party_cache/stockfish/src/stockfish")
    stockfish_eval: StockfishConfig = StockfishConfig(path=".third_party_cache/stockfish/src/stockfish", skill_level=20)

    # Inference behavior:
    # - batched_inference=True runs one backend.generate() call per ply over all active games.
    # - Set to False only for benchmarking / debugging (slower, should be identical given per-prompt seeds).
    batched_inference: bool = True
    # Log per-batch inference timing to stdout (grep-friendly).
    log_batch_stats: bool = True

    # Prompting:
    # - If set, render a starter-kit-compatible Jinja prompt template each turn.
    # - Else, fall back to the legacy system+user prompt builder below.
    prompt_template_path: Optional[str] = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT_UCI_MOVE
    out_dir: Path = Path("outputs/full_game_eval")


@dataclass
class _GameState:
    game_id: str
    opponent_depth: int
    model_color: chess.Color
    board: chess.Board
    pgn: chess.pgn.Game
    pgn_node: chess.pgn.GameNode
    # Optional bookkeeping (for 5-round evaluation runs).
    round_idx: int = 0
    game_idx_in_round: int = 0

    # Engine-based metrics (computed once per finished game; starter-kit style).
    # `model_*` refers to the evaluated model side (white or black depending on assignment).
    model_acpl: float = 0.0
    model_moves: int = 0
    model_cpl_sum: float = 0.0
    model_accuracy_pct: float = 0.0
    white_acpl: float = 0.0
    black_acpl: float = 0.0
    white_accuracy_pct: float = 0.0
    black_accuracy_pct: float = 0.0

    # Terminal bookkeeping.
    is_over: bool = False
    termination: str = ""
    result: str = ""  # PGN result string: "1-0", "0-1", "1/2-1/2"
    forfeit: bool = False
    forfeit_reason: str = ""
    result_str: str = ""
    engine_error: str = ""

    def model_color_str(self) -> str:
        return "white" if self.model_color == chess.WHITE else "black"


def _jsonl_write(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


def _build_user_prompt(fen: str, legal_moves_uci: List[str]) -> str:
    return (
        f"Current FEN string: {fen}\n"
        f"Legal moves (UCI): {', '.join(legal_moves_uci)}\n\n"
        "Choose the single best move from the legal moves list.\n"
        "Output format (required):\n"
        "<guess> GUESS_UCI </guess>\n"
        "<think>...</think><uci_move>...</uci_move>\n"
        "- Put exactly one strict UCI move in <guess>.\n"
        "- Use exactly one <think> block.\n"
        "- Put exactly one legal UCI move in <uci_move>.\n"
        "- Do not write any other text outside the tags."
    )


def _build_retry_suffix(*, error_reason: str, retry_idx: int) -> str:
    # Keep this short; we want to preserve alignment with the training prompt style,
    # but still nudge the model into the strict `<think>...</think><uci_move>...</uci_move>` format.
    return (
        "\n\n"
        f"Your previous response was invalid or illegal (reason: {error_reason}, retry {retry_idx + 1}). "
        "Reply with exactly:\n"
        "<guess> GUESS_UCI </guess>\n"
        "<think>...</think><uci_move>...</uci_move>\n"
        "- Put exactly one strict UCI move in <guess>.\n"
        "- Use exactly one <think> block.\n"
        "- Put exactly one legal UCI move in <uci_move>.\n"
        "- Do not write any other text outside the tags."
    )


def _build_chat_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    # Mirror the dataset contract: a single user message containing system+user text.
    prompt_text = f"{(system_prompt or '').strip()}\n\n{(user_prompt or '').strip()}".strip()
    return [{"role": "user", "content": prompt_text}]


def _score_to_cp_pov(score: Any, *, pov: chess.Color, mate_score_cp: int) -> Optional[int]:
    if score is None:
        return None
    try:
        cp = score.pov(pov).score(mate_score=mate_score_cp)
    except Exception:
        return None
    if cp is None:
        return None
    return int(cp)


def _compute_cpl(
    *,
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    played_move: chess.Move,
    depth: int,
    mate_score_cp: int,
) -> Tuple[str, Optional[int], Optional[int], float]:
    """Compute centipawn loss for a played move.

    Spec alignment:
      loss = max(0, ModelMoveEval - BestMoveEval)
    where Eval is taken from the POV of the side to move *after* the move
    (i.e., the opponent to the player who just moved).
    """
    info_best = engine.analyse(board, chess.engine.Limit(depth=depth))
    pv = info_best.get("pv") or []
    if not pv:
        return "", None, None, 0.0
    best_move = pv[0]

    board_best = board.copy(stack=False)
    board_best.push(best_move)
    info_best_after = engine.analyse(board_best, chess.engine.Limit(depth=depth))
    best_cp_pov = _score_to_cp_pov(info_best_after.get("score"), pov=board_best.turn, mate_score_cp=mate_score_cp)

    board_played = board.copy(stack=False)
    board_played.push(played_move)
    info_played_after = engine.analyse(board_played, chess.engine.Limit(depth=depth))
    played_cp_pov = _score_to_cp_pov(info_played_after.get("score"), pov=board_played.turn, mate_score_cp=mate_score_cp)

    if best_cp_pov is None or played_cp_pov is None:
        return best_move.uci(), best_cp_pov, played_cp_pov, 0.0

    loss = float(max(0, played_cp_pov - best_cp_pov))
    return best_move.uci(), int(best_cp_pov), int(played_cp_pov), loss


def _apply_board_outcome(game: _GameState) -> None:
    # Starter-kit environment does *not* auto-claim 50-move / threefold draws.
    outcome = game.board.outcome(claim_draw=False)
    if outcome is None:
        game.is_over = False
        return

    game.is_over = True
    # Match starter-kit naming (human-readable, stable).
    game.termination = str(getattr(outcome.termination, "name", str(outcome.termination))).lower()
    game.result = game.board.result(claim_draw=False)
    if not game.result_str:
        game.result_str = _pgn_result_to_result_str(game.result)


def _set_forfeit(game: _GameState, reason: str, resignation_cpl: int) -> None:
    game.is_over = True
    game.forfeit = True
    game.forfeit_reason = reason
    # Starter-kit treats this as a resignation / failure-to-move.
    game.termination = "resignation"
    # Model resigns immediately on its turn.
    game.result = "0-1" if game.model_color == chess.WHITE else "1-0"
    game.result_str = "Black wins (White resigned)" if game.model_color == chess.WHITE else "White wins (Black resigned)"


def _parse_model_move(output_text: str) -> tuple[Optional[str], str]:
    """Starter-kit compatible move parsing.

    - Requires `<uci_move>...</uci_move>` tags (case-insensitive).
    - Does not require a `<think>` tag.
    - Does not normalize non-UCI strings (beyond `.strip()`).

    Returns: (parsed_uci_or_none, error_reason)
    """
    s = output_text or ""
    m = UCI_MOVE_ONLY_RE.search(s)
    if not m:
        return None, "format_missing"

    raw = (m.group("ans") or "").strip()
    if not raw:
        return None, "empty_uci_move"

    # NOTE: Keep parsing strict (match starter kit): do not lowercase/normalize before
    # legality/parsing checks. This ensures outputs like `<uci_move>E2E4</uci_move>`
    # behave like invalid moves (python-chess rejects them).
    move_str = raw.strip()
    if move_str.lower() == "resign":
        # Starter-kit treats explicit resign as a non-move and retries until attempts exhausted.
        return None, "resign"

    return move_str, ""


def _safe_int_seed(seed: int) -> int:
    # vLLM expects 32-bit seeds.
    return int(seed) % 0x7FFFFFFF


def run_full_game_eval(
    *,
    cfg: FullGameEvalConfig,
    backend: ChatBackend,
) -> Dict[str, Any]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    moves_path = cfg.out_dir / "moves.jsonl"
    games_path = cfg.out_dir / "games.jsonl"
    summary_path = cfg.out_dir / "summary.json"
    pgn_path = cfg.out_dir / "games.pgn"

    prompt_template: Optional[Template] = None
    prompt_template_vars: Optional[set[str]] = None
    if cfg.prompt_template_path:
        template_text = Path(cfg.prompt_template_path).read_text(encoding="utf-8")
        prompt_template = Template(template_text)
        # Performance: only compute the template context keys that are actually used.
        # Prompt templates vary by training mode:
        # - Chess-R1-style prompts typically use `FEN` and `legal_moves_uci_list`.
        # - Selection prompts additionally require `considered_moves_uci_list`.
        # The starter-kit template uses many more fields.
        try:
            prompt_template_vars = set(meta.find_undeclared_variables(prompt_template.environment.parse(template_text)))
        except Exception:
            prompt_template_vars = None

    eval_engine_cm = (
        chess.engine.SimpleEngine.popen_uci(cfg.stockfish_eval.path) if int(cfg.acpl_workers) == 1 else nullcontext(None)
    )

    # Open in text mode; jsonl is append-friendly.
    with (
        open(moves_path, "w", encoding="utf-8") as moves_fp,
        open(games_path, "w", encoding="utf-8") as games_fp,
        open(pgn_path, "w", encoding="utf-8") as pgn_fp,
        chess.engine.SimpleEngine.popen_uci(cfg.stockfish_opponent.path) as opp_engine,
        eval_engine_cm as eval_engine,
    ):
        _configure_stockfish(opp_engine, cfg.stockfish_opponent)
        if eval_engine is not None:
            _configure_stockfish(eval_engine, cfg.stockfish_eval)

        summary_by_depth: Dict[str, Any] = {}
        games_by_depth: Dict[int, List[_GameState]] = {}
        for depth in cfg.opponent_depths:
            games_by_depth[int(depth)] = _init_games_for_depth(cfg, opponent_depth=int(depth))

        all_games: List[_GameState] = []
        for depth in cfg.opponent_depths:
            all_games.extend(games_by_depth[int(depth)])

        # Mixed-depth scheduling: advance all games concurrently so each model inference batch can
        # include positions from multiple opponent depths (e.g., depth=1 and depth=5).
        run_stats = _run_depth_games(
            cfg=cfg,
            backend=backend,
            opp_engine=opp_engine,
            eval_engine=eval_engine,
            games=all_games,
            moves_fp=moves_fp,
            games_fp=games_fp,
            prompt_template=prompt_template,
            prompt_template_vars=prompt_template_vars,
        )

        # Append PGN to a combined file (starter-kit convenience). Preserve the previous ordering
        # (group by depth in the user-provided order) even though gameplay was mixed.
        for depth in cfg.opponent_depths:
            for g in games_by_depth[int(depth)]:
                pgn_text = str(g.pgn).strip()
                if not pgn_text:
                    continue
                if pgn_fp.tell() > 0:
                    pgn_fp.write("\n\n")
                pgn_fp.write(pgn_text)

        run_stats_by_depth: Dict[int, Dict[str, Any]] = {}
        if isinstance(run_stats, dict):
            by_depth = run_stats.get("by_depth", None)
            if isinstance(by_depth, dict):
                # JSON serialization converts int keys to strings; normalize to int for lookup.
                for k, v in by_depth.items():
                    try:
                        depth_k = int(k)
                    except Exception:
                        continue
                    if isinstance(v, dict):
                        run_stats_by_depth[depth_k] = v

        overall_row = (run_stats.get("overall", None) if isinstance(run_stats, dict) else None) or {}

        for depth in cfg.opponent_depths:
            games = games_by_depth[int(depth)]
            depth_stats = run_stats_by_depth.get(int(depth), {})

            # Aggregate.
            wins = 0
            losses = 0
            draws = 0
            # Starter-kit style: average per-game ACPL for the model side.
            acpl_sum_per_game = 0.0
            acpl_games = 0
            # Extra: move-weighted ACPL over all model moves.
            acpl_sum_per_move = 0.0
            acpl_moves = 0
            for g in games:
                if g.result == "1/2-1/2":
                    draws += 1
                else:
                    model_won = (g.result == "1-0" and g.model_color == chess.WHITE) or (
                        g.result == "0-1" and g.model_color == chess.BLACK
                    )
                    if model_won:
                        wins += 1
                    else:
                        losses += 1
                acpl_sum_per_game += float(g.model_acpl)
                acpl_games += 1
                acpl_sum_per_move += float(g.model_cpl_sum)
                acpl_moves += int(g.model_moves)

            summary_by_depth[f"depth_{depth}"] = {
                "opponent_depth": int(depth),
                "num_games": int(len(games)),
                "wins": int(wins),
                "losses": int(losses),
                "draws": int(draws),
                "acpl_sum": float(acpl_sum_per_game),
                "acpl_games": int(acpl_games),
                "acpl_mean": float(acpl_sum_per_game / acpl_games) if acpl_games > 0 else float("nan"),
                "acpl_sum_per_move": float(acpl_sum_per_move),
                "acpl_moves": int(acpl_moves),
                "acpl_mean_per_move": float(acpl_sum_per_move / acpl_moves) if acpl_moves > 0 else float("nan"),
                **(depth_stats or {}),
            }

        # Convenience: overall summary (kept separate from per-depth stats).
        total_games = 0
        total_wins = 0
        total_losses = 0
        total_draws = 0
        total_acpl_sum_per_game = 0.0
        total_acpl_games = 0
        total_acpl_sum_per_move = 0.0
        total_acpl_moves = 0
        for row in summary_by_depth.values():
            total_games += int(row.get("num_games", 0) or 0)
            total_wins += int(row.get("wins", 0) or 0)
            total_losses += int(row.get("losses", 0) or 0)
            total_draws += int(row.get("draws", 0) or 0)
            total_acpl_sum_per_game += float(row.get("acpl_sum", 0.0) or 0.0)
            total_acpl_games += int(row.get("acpl_games", 0) or 0)
            total_acpl_sum_per_move += float(row.get("acpl_sum_per_move", 0.0) or 0.0)
            total_acpl_moves += int(row.get("acpl_moves", 0) or 0)

        summary_overall: Dict[str, Any] = {
            "num_games": int(total_games),
            "wins": int(total_wins),
            "losses": int(total_losses),
            "draws": int(total_draws),
            "win_rate": float(total_wins / total_games) if total_games > 0 else float("nan"),
            "acpl_sum": float(total_acpl_sum_per_game),
            "acpl_games": int(total_acpl_games),
            "acpl_mean": float(total_acpl_sum_per_game / total_acpl_games) if total_acpl_games > 0 else float("nan"),
            "acpl_sum_per_move": float(total_acpl_sum_per_move),
            "acpl_moves": int(total_acpl_moves),
            "acpl_mean_per_move": float(total_acpl_sum_per_move / total_acpl_moves)
            if total_acpl_moves > 0
            else float("nan"),
            **(overall_row or {}),
        }

        summary: Dict[str, Any] = {
            "config": {
                "opponent_depths": cfg.opponent_depths,
                "games_per_depth": cfg.games_per_depth,
                "rounds": cfg.rounds,
                "games_per_round": cfg.games_per_round,
                "max_retries_per_turn": cfg.max_retries_per_turn,
                "seed": cfg.seed,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_response_tokens": cfg.max_response_tokens,
                "batched_inference": bool(cfg.batched_inference),
                "opponent_movetime_ms": cfg.opponent_movetime_ms,
                "acpl_eval_depth": cfg.acpl_eval_depth,
                "acpl_eval_movetime_ms": cfg.acpl_eval_movetime_ms,
                "acpl_eval_cp_cap": cfg.acpl_eval_cp_cap,
                "acpl_workers": int(cfg.acpl_workers),
                "resignation_cpl": cfg.resignation_cpl,
                "mate_score_cp": cfg.mate_score_cp,
                "max_plies": cfg.max_plies,
                "stockfish_opponent": {
                    "path": cfg.stockfish_opponent.path,
                    "threads": cfg.stockfish_opponent.threads,
                    "hash_mb": cfg.stockfish_opponent.hash_mb,
                    "skill_level": cfg.stockfish_opponent.skill_level,
                },
                "stockfish_eval": {
                    "path": cfg.stockfish_eval.path,
                    "threads": cfg.stockfish_eval.threads,
                    "hash_mb": cfg.stockfish_eval.hash_mb,
                    "skill_level": cfg.stockfish_eval.skill_level,
                },
                "prompt_template_path": cfg.prompt_template_path,
                # If a prompt template is provided, `system_prompt` is not used at inference time.
                # Keep it empty here to avoid misleading output-contract conflicts in summaries.
                "system_prompt": cfg.system_prompt if not cfg.prompt_template_path else "",
                "out_dir": str(cfg.out_dir),
            },
            "summary_by_depth": summary_by_depth,
            "paths": {
                "moves_jsonl": str(moves_path),
                "games_jsonl": str(games_path),
                "summary_json": str(summary_path),
                "games_pgn": str(pgn_path),
            },
            "summary_overall": summary_overall,
        }

        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return summary


def _configure_stockfish(engine: chess.engine.SimpleEngine, cfg: StockfishConfig) -> None:
    opts: Dict[str, Any] = {}
    if cfg.threads is not None:
        opts["Threads"] = int(cfg.threads)
    if cfg.hash_mb is not None:
        opts["Hash"] = int(cfg.hash_mb)
    if cfg.skill_level is not None:
        opts["Skill Level"] = int(cfg.skill_level)
    if opts:
        try:
            engine.configure(opts)
        except Exception:
            # Some images/binaries may not support all options; best-effort only.
            pass


def _init_games_for_depth(cfg: FullGameEvalConfig, *, opponent_depth: int) -> List[_GameState]:
    n = int(cfg.games_per_depth)
    if n <= 0:
        raise ValueError("games_per_depth must be > 0")

    games_per_round = int(cfg.games_per_round) if cfg.games_per_round else n
    if games_per_round <= 0:
        games_per_round = n

    games: List[_GameState] = []
    # Competition-style: rounds of 50 games, each round is color-balanced when possible
    # (e.g., 25 white + 25 black). This keeps the "round" concept meaningful while still
    # scheduling all games in-flight concurrently.
    created = 0
    round_idx = 0
    while created < n:
        round_size = min(int(games_per_round), n - created)
        round_white = round_size // 2
        round_black = round_size - round_white
        colors: List[chess.Color] = [chess.WHITE] * round_white + [chess.BLACK] * round_black

        # Deterministic shuffle to avoid ordering bias, but keep per-round balance.
        rng = _DeterministicRng(cfg.seed, salt=f"depth={opponent_depth}|round={round_idx}")
        rng.shuffle(colors)

        for game_idx_in_round, color in enumerate(colors):
            i = created + int(game_idx_in_round)
            game_id = f"d{opponent_depth}_g{i:03d}"
            board = chess.Board()
            pgn = chess.pgn.Game()
            pgn.headers["Event"] = "chess-rl full-game eval"
            pgn.headers["Site"] = "local"
            pgn.headers["Date"] = time.strftime("%Y.%m.%d")
            if color == chess.WHITE:
                pgn.headers["White"] = "model"
                pgn.headers["Black"] = f"stockfish_depth{opponent_depth}_skill{cfg.stockfish_opponent.skill_level}"
            else:
                pgn.headers["White"] = f"stockfish_depth{opponent_depth}_skill{cfg.stockfish_opponent.skill_level}"
                pgn.headers["Black"] = "model"
            games.append(
                _GameState(
                    game_id=game_id,
                    opponent_depth=int(opponent_depth),
                    model_color=color,
                    board=board,
                    pgn=pgn,
                    pgn_node=pgn,
                    round_idx=int(round_idx),
                    game_idx_in_round=int(game_idx_in_round),
                )
            )

        created += int(round_size)
        round_idx += 1
    return games


def _enforce_max_plies(games: List[_GameState], *, max_plies: Optional[int]) -> None:
    """Enforce the starter-kit-style max-moves cap (max plies).

    Starter kit semantics:
      - The cap is checked *before* attempting the next move.
      - If the board is already terminal (checkmate/stalemate/etc.), keep the terminal result.
      - Else, declare a draw with termination_reason="max_moves".
    """
    if max_plies is None:
        return
    cap = int(max_plies)
    if cap <= 0:
        return
    for g in games:
        if g.is_over:
            continue
        if len(g.board.move_stack) < cap:
            continue
        if g.board.is_game_over(claim_draw=False):
            _apply_board_outcome(g)
            continue
        g.is_over = True
        g.termination = "max_moves"
        g.result = "1/2-1/2"
        g.result_str = "Draw (max moves reached)"


def _run_depth_games(
    *,
    cfg: FullGameEvalConfig,
    backend: ChatBackend,
    opp_engine: chess.engine.SimpleEngine,
    eval_engine: Optional[chess.engine.SimpleEngine],
    games: List[_GameState],
    moves_fp,
    games_fp,
    prompt_template: Optional[Template],
    prompt_template_vars: Optional[set[str]],
) -> Dict[str, Any]:
    infer_time_s = 0.0
    infer_positions = 0
    infer_batches = 0
    infer_time_s_by_depth: Dict[int, float] = {}
    infer_positions_by_depth: Dict[int, int] = {}
    infer_batches_by_depth: Dict[int, float] = {}

    active = [g for g in games if not g.is_over]
    while active:
        # Starter-kit semantics: enforce the ply cap before attempting the next move.
        _enforce_max_plies(active, max_plies=cfg.max_plies)

        # First, let the model move on all games where it's the model's turn.
        model_games = [g for g in active if (not g.is_over) and g.board.turn == g.model_color]
        if model_games:
            infer_stats = _step_model_moves(
                cfg=cfg,
                backend=backend,
                eval_engine=eval_engine,
                games=model_games,
                moves_fp=moves_fp,
                prompt_template=prompt_template,
                prompt_template_vars=prompt_template_vars,
            )
            if isinstance(infer_stats, dict):
                infer_time_s += float(infer_stats.get("infer_time_s", 0.0) or 0.0)
                infer_positions += int(infer_stats.get("infer_positions", 0) or 0)
                infer_batches += int(infer_stats.get("infer_batches", 0) or 0)
                by_depth_time = infer_stats.get("infer_time_s_by_depth", None)
                if isinstance(by_depth_time, dict):
                    for depth, v in by_depth_time.items():
                        try:
                            depth_k = int(depth)
                        except Exception:
                            continue
                        infer_time_s_by_depth[depth_k] = infer_time_s_by_depth.get(depth_k, 0.0) + float(v or 0.0)

                by_depth_pos = infer_stats.get("infer_positions_by_depth", None)
                if isinstance(by_depth_pos, dict):
                    for depth, v in by_depth_pos.items():
                        try:
                            depth_k = int(depth)
                        except Exception:
                            continue
                        infer_positions_by_depth[depth_k] = infer_positions_by_depth.get(depth_k, 0) + int(v or 0)

                by_depth_batches = infer_stats.get("infer_batches_by_depth", None)
                if isinstance(by_depth_batches, dict):
                    for depth, v in by_depth_batches.items():
                        try:
                            depth_k = int(depth)
                        except Exception:
                            continue
                        infer_batches_by_depth[depth_k] = infer_batches_by_depth.get(depth_k, 0.0) + float(v or 0.0)

        # Enforce max-plies after the model move (a game may hit the cap exactly and should
        # not allow the opponent to play an extra move beyond it).
        _enforce_max_plies(active, max_plies=cfg.max_plies)

        # Then play Stockfish responses.
        opp_games = [g for g in active if (not g.is_over) and g.board.turn != g.model_color]
        for g in opp_games:
            # Cap check before opponent move (mirrors starter kit's per-move loop condition).
            _enforce_max_plies([g], max_plies=cfg.max_plies)
            if g.is_over:
                continue
            if g.board.is_game_over(claim_draw=False):
                _apply_board_outcome(g)
                continue

            try:
                # Match starter-kit local evaluation behavior: depth-limited opponent with a small
                # time budget (100ms). The competition spec names "Depth 1/5"; the starter kit's
                # local evaluator also sets a movetime budget.
                limit_kwargs: Dict[str, Any] = {"depth": int(g.opponent_depth)}
                if int(cfg.opponent_movetime_ms) > 0:
                    limit_kwargs["time"] = float(cfg.opponent_movetime_ms) / 1000.0
                result = opp_engine.play(g.board, chess.engine.Limit(**limit_kwargs))
                move = result.move
            except Exception as exc:
                # Treat engine failure as a draw to avoid killing the whole eval.
                g.is_over = True
                g.termination = "engine_error"
                g.result = "1/2-1/2"
                g.result_str = "Draw (engine error)"
                g.engine_error = str(exc)
                continue

            g.board.push(move)
            g.pgn_node = g.pgn_node.add_variation(move)

            if g.board.is_game_over(claim_draw=False):
                _apply_board_outcome(g)
            else:
                # Cap check after opponent move (if we hit the cap exactly, end as draw).
                _enforce_max_plies([g], max_plies=cfg.max_plies)

        # Drop finished games so:
        # - the loop terminates,
        # - subsequent inference batches only include active games,
        # - early terminations don't keep paying per-iteration overhead.
        active = [g for g in active if not g.is_over]

    # Compute ACPL/accuracy after *all* games are finished.
    # This keeps the main game loop throughput high and enables CPU parallelism.
    t_acpl0 = time.time()
    analyses = _analyze_games_with_engine(
        cfg=cfg,
        eval_engine=eval_engine,
        games=games,
    )
    t_acpl1 = time.time()
    acpl_time_s = float(t_acpl1 - t_acpl0)
    depth_counts: Dict[int, int] = {}
    for g in games:
        depth_counts[int(g.opponent_depth)] = depth_counts.get(int(g.opponent_depth), 0) + 1
    total_games = sum(depth_counts.values())
    acpl_time_s_by_depth: Dict[int, float] = {}
    if total_games > 0:
        for depth, count in depth_counts.items():
            acpl_time_s_by_depth[int(depth)] = float(acpl_time_s) * float(count / total_games)

    # Flush all games (with ACPL) to games.jsonl.
    for g in games:
        if g.pgn.headers.get("Result", "*") in ("", "*"):
            g.pgn.headers["Result"] = g.result
        if not g.result_str:
            g.result_str = _pgn_result_to_result_str(g.result)

        analysis = analyses.get(g.game_id, None) or {}
        white_acpl = float(analysis.get("white_acpl", float(cfg.acpl_eval_cp_cap)))
        black_acpl = float(analysis.get("black_acpl", float(cfg.acpl_eval_cp_cap)))
        white_accuracy_pct = float(analysis.get("white_accuracy_pct", 0.0))
        black_accuracy_pct = float(analysis.get("black_accuracy_pct", 0.0))
        white_moves = int(analysis.get("white_moves", 0))
        black_moves = int(analysis.get("black_moves", 0))
        white_cpl_sum = float(analysis.get("white_cpl_sum", 0.0))
        black_cpl_sum = float(analysis.get("black_cpl_sum", 0.0))

        # Resignation / invalid-move penalty:
        # - If the model forfeits after exhausting retries (or outputs explicit resign),
        #   add a constant CPL penalty to the model side.
        # - Apply after engine analysis so we don't attribute the penalty to a specific ply.
        resignation_penalty = float(cfg.resignation_cpl or 0)
        resignation_cpl_penalty = 0.0
        if g.forfeit and resignation_penalty > 0:
            resignation_cpl_penalty = float(resignation_penalty)
            if g.model_color == chess.WHITE:
                white_cpl_sum += resignation_penalty
                # Ensure the penalty contributes to move-weighted ACPL even if the model made 0 moves.
                if white_moves <= 0:
                    white_moves = 1
                white_acpl = float(white_cpl_sum / white_moves)
            else:
                black_cpl_sum += resignation_penalty
                if black_moves <= 0:
                    black_moves = 1
                black_acpl = float(black_cpl_sum / black_moves)

        g.white_acpl = float(white_acpl)
        g.black_acpl = float(black_acpl)
        g.white_accuracy_pct = float(white_accuracy_pct)
        g.black_accuracy_pct = float(black_accuracy_pct)

        if g.model_color == chess.WHITE:
            g.model_acpl = float(g.white_acpl)
            g.model_accuracy_pct = float(g.white_accuracy_pct)
            g.model_moves = int(white_moves)
            g.model_cpl_sum = float(white_cpl_sum)
        else:
            g.model_acpl = float(g.black_acpl)
            g.model_accuracy_pct = float(g.black_accuracy_pct)
            g.model_moves = int(black_moves)
            g.model_cpl_sum = float(black_cpl_sum)

        pgn_text = str(g.pgn).strip()
        _jsonl_write(
            games_fp,
            {
                "ts": time.time(),
                "game_id": g.game_id,
                "opponent_depth": g.opponent_depth,
                "model_color": g.model_color_str(),
                "round": int(getattr(g, "round_idx", 0)),
                "game_idx_in_round": int(getattr(g, "game_idx_in_round", 0)),
                "result": g.result_str or g.result,
                "pgn_result": g.result,
                "termination": g.termination,
                "engine_error": g.engine_error,
                "forfeit": bool(g.forfeit),
                "forfeit_reason": g.forfeit_reason,
                "num_plies": int(len(g.board.move_stack)),
                "white_acpl": float(g.white_acpl),
                "black_acpl": float(g.black_acpl),
                "white_accuracy_pct": float(g.white_accuracy_pct),
                "black_accuracy_pct": float(g.black_accuracy_pct),
                "model_acpl": float(g.model_acpl),
                "model_accuracy_pct": float(g.model_accuracy_pct),
                "model_cpl_sum": float(g.model_cpl_sum),
                "model_moves": int(g.model_moves),
                "resignation_cpl_penalty": float(resignation_cpl_penalty),
                "pgn": pgn_text,
            },
        )

    # Return both overall stats and a per-depth attribution. Inference/ACPL are shared across
    # depths in the mixed-depth schedule, so per-depth times are attributed proportionally.
    depths_present = sorted(depth_counts.keys())
    by_depth: Dict[int, Dict[str, Any]] = {}
    for depth in depths_present:
        by_depth[int(depth)] = {
            "infer_time_s": float(infer_time_s_by_depth.get(int(depth), 0.0)),
            "infer_positions": int(infer_positions_by_depth.get(int(depth), 0)),
            "infer_batches": float(infer_batches_by_depth.get(int(depth), 0.0)),
            "acpl_time_s": float(acpl_time_s_by_depth.get(int(depth), 0.0)),
        }

    return {
        "overall": {
            "infer_time_s": float(infer_time_s),
            "infer_positions": int(infer_positions),
            "infer_batches": int(infer_batches),
            "acpl_time_s": float(acpl_time_s),
        },
        "by_depth": by_depth,
    }


def _step_model_moves(
    *,
    cfg: FullGameEvalConfig,
    backend: ChatBackend,
    eval_engine: Optional[chess.engine.SimpleEngine],
    games: List[_GameState],
    moves_fp,
    prompt_template: Optional[Template],
    prompt_template_vars: Optional[set[str]],
) -> Dict[str, Any]:
    infer_time_s = 0.0
    infer_positions = 0
    infer_batches = 0
    infer_time_s_by_depth: Dict[int, float] = {}
    infer_positions_by_depth: Dict[int, int] = {}
    infer_batches_by_depth: Dict[int, float] = {}

    # Track per-game retry state.
    # NOTE: the full-game move trace (`moves.jsonl`) is our primary debugging artifact. Keep it
    # self-contained: even terminal forfeit rows should include the last prompt text so that a
    # reader can reconstruct the model call that led to resignation.
    pending: Dict[str, Dict[str, Any]] = {
        g.game_id: {"game": g, "last_output": "", "last_error": "", "last_prompt_text": ""} for g in games
    }

    for retry_idx in range(cfg.max_retries_per_turn):
        todo = [st["game"] for st in pending.values() if not st["game"].is_over]
        if not todo:
            return {
                "infer_time_s": float(infer_time_s),
                "infer_positions": int(infer_positions),
                "infer_batches": int(infer_batches),
                "infer_time_s_by_depth": infer_time_s_by_depth,
                "infer_positions_by_depth": infer_positions_by_depth,
                "infer_batches_by_depth": infer_batches_by_depth,
            }

        counts_by_depth: Dict[int, int] = {}
        for g in todo:
            counts_by_depth[int(g.opponent_depth)] = counts_by_depth.get(int(g.opponent_depth), 0) + 1

        prompts: List[List[Dict[str, str]]] = []
        prompt_texts: List[str] = []
        for g in todo:
            legal_moves = list(g.board.legal_moves)

            if prompt_template is not None:
                side_to_move = "White" if g.board.turn else "Black"
                move_history: List[str] = [m.uci() for m in g.board.move_stack]
                ctx = _build_prompt_context(
                    board=g.board,
                    legal_moves=legal_moves,
                    move_history=move_history,
                    side_to_move=side_to_move,
                    needed_vars=prompt_template_vars,
                )
                prompt_text = prompt_template.render(**ctx)
                messages = [{"role": "user", "content": prompt_text}]
            else:
                # Legacy prompt builder: a single user message containing system+user text.
                fen = g.board.fen()
                legal_moves_uci = [m.uci().lower() for m in legal_moves]
                user_prompt = _build_user_prompt(fen, legal_moves_uci)
                messages = _build_chat_messages(cfg.system_prompt, user_prompt)
                prompt_text = messages[0]["content"]

            prompts.append(messages)
            prompt_texts.append(prompt_text)

        # Deterministic per-game seeds (optional; backend may ignore).
        # We mix by game_id + ply + retry to keep retries exploring while remaining reproducible.
        seeds = [
            _safe_int_seed(_mix_seed(cfg.seed, salt=f"{g.game_id}|ply={len(g.board.move_stack)}|try={retry_idx}"))
            for g in todo
        ]

        t_gen0 = time.time()
        if cfg.batched_inference:
            outputs = backend.generate(
                prompts,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_response_tokens,
                seeds=seeds,
            )
        else:
            # Slow path for benchmarking/debugging: submit each prompt individually.
            outputs = []
            for prompt, seed in zip(prompts, seeds):
                out = backend.generate(
                    [prompt],
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_response_tokens,
                    seeds=[int(seed)],
                )
                outputs.append(out[0] if out else "")
        t_gen1 = time.time()
        gen_dt_s = float(t_gen1 - t_gen0)
        infer_time_s += float(gen_dt_s)
        infer_positions += int(len(todo))
        infer_batches += 1
        # Attribute per-depth inference timing proportional to the number of prompts from each depth
        # in this batch. This keeps the per-depth totals additive even under mixed-depth batching.
        if len(todo) > 0:
            for depth, count in counts_by_depth.items():
                share = float(count / len(todo))
                infer_time_s_by_depth[int(depth)] = infer_time_s_by_depth.get(int(depth), 0.0) + float(gen_dt_s) * share
                infer_positions_by_depth[int(depth)] = infer_positions_by_depth.get(int(depth), 0) + int(count)
                infer_batches_by_depth[int(depth)] = infer_batches_by_depth.get(int(depth), 0.0) + share

        if cfg.log_batch_stats:
            mode = "batched" if cfg.batched_inference else "serial"
            thr = (len(todo) / gen_dt_s) if gen_dt_s > 0 else float("inf")
            depths_str = ",".join(f"{d}x{counts_by_depth[d]}" for d in sorted(counts_by_depth.keys()))
            print(
                f"[fullgame][infer] mode={mode} retry={retry_idx} batch={len(todo)} dt_s={gen_dt_s:.3f} thr_pos_s={thr:.2f} depths={depths_str}",
                flush=True,
            )
        if len(outputs) != len(todo):
            raise RuntimeError(f"Backend returned {len(outputs)} outputs for {len(todo)} prompts")

        for g, prompt_text, output_text in zip(todo, prompt_texts, outputs):
            ts = time.time()
            parsed_uci, parse_err = _parse_model_move(output_text)

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
                "opponent_depth": g.opponent_depth,
                "model_color": g.model_color_str(),
                "round": int(getattr(g, "round_idx", 0)),
                "game_idx_in_round": int(getattr(g, "game_idx_in_round", 0)),
                "ply": int(len(g.board.move_stack)),
                "fen": g.board.fen(),
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
                _jsonl_write(moves_fp, record)
                pending[g.game_id]["last_output"] = output_text
                pending[g.game_id]["last_error"] = error_reason
                pending[g.game_id]["last_prompt_text"] = prompt_text
                continue

            if played_move is None:
                # UCI was in the legal move list but couldn't be parsed by python-chess.
                # Treat as a hard error (retry/forfeit path).
                record["error_reason"] = "bad_move"
                _jsonl_write(moves_fp, record)
                pending[g.game_id]["last_output"] = output_text
                pending[g.game_id]["last_error"] = "bad_move"
                pending[g.game_id]["last_prompt_text"] = prompt_text
                continue

            record.update(
                {
                    "accepted_move_uci": played_move.uci().lower(),
                }
            )
            _jsonl_write(moves_fp, record)

            # Apply move.
            g.board.push(played_move)
            g.pgn_node = g.pgn_node.add_variation(played_move)

            if g.board.is_game_over(claim_draw=False):
                _apply_board_outcome(g)

            # Resolved; remove from pending.
            pending.pop(g.game_id, None)

        # Next retry loop for remaining pending.

    # Any remaining pending games forfeit.
    for st in pending.values():
        g = st["game"]
        if g.is_over:
            continue
        reason = st.get("last_error") or "invalid_output"
        _set_forfeit(g, reason=reason, resignation_cpl=cfg.resignation_cpl)
        _jsonl_write(
            moves_fp,
            {
                "ts": time.time(),
                "game_id": g.game_id,
                "opponent_depth": g.opponent_depth,
                "model_color": g.model_color_str(),
                "round": int(getattr(g, "round_idx", 0)),
                "game_idx_in_round": int(getattr(g, "game_idx_in_round", 0)),
                "ply": int(len(g.board.move_stack)),
                "fen": g.board.fen(),
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
        "infer_time_s_by_depth": infer_time_s_by_depth,
        "infer_positions_by_depth": infer_positions_by_depth,
        "infer_batches_by_depth": infer_batches_by_depth,
    }


def _acpl_worker_main(
    *,
    worker_id: int,
    tasks: List[Dict[str, Any]],
    output_path: str,
    engine_path: str,
    engine_cfg: StockfishConfig,
    depth: int,
    movetime_ms: int,
    cp_cap: int,
    mate_score_cp: int,
) -> None:
    """Worker entrypoint for parallel ACPL computation (spawn-safe)."""
    print(
        f"[fullgame][acpl] worker_start id={worker_id} tasks={len(tasks)} threads={engine_cfg.threads} hash_mb={engine_cfg.hash_mb}",
        flush=True,
    )
    with (
        open(output_path, "w", encoding="utf-8") as fp,
        chess.engine.SimpleEngine.popen_uci(engine_path) as engine,
    ):
        _configure_stockfish(engine, engine_cfg)

        for t in tasks:
            game_id = str(t.get("game_id", ""))
            moves_uci = t.get("moves_uci", None)
            if not isinstance(moves_uci, list):
                moves_uci = []
            moves_uci = [str(x) for x in moves_uci]

            try:
                analysis = _analyze_game_with_engine(
                    engine=engine,
                    moves_uci=moves_uci,
                    depth=int(depth),
                    movetime_ms=int(movetime_ms),
                    cp_cap=int(cp_cap),
                    mate_score_cp=int(mate_score_cp),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"ACPL worker failed game_id={game_id}: {type(exc).__name__}: {exc}"
                ) from exc

            fp.write(
                json.dumps(
                    {
                        "game_id": game_id,
                        "analysis": analysis,
                        "error": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fp.flush()

    print(f"[fullgame][acpl] worker_done id={worker_id}", flush=True)


def _analyze_games_with_engine(
    *,
    cfg: FullGameEvalConfig,
    eval_engine: Optional[chess.engine.SimpleEngine],
    games: List[_GameState],
) -> Dict[str, Dict[str, Any]]:
    """Compute ACPL/accuracy for each game, optionally parallelized across CPU workers."""
    if int(cfg.acpl_workers) <= 0:
        print("[fullgame][acpl] mode=skip", flush=True)
        return {}

    tasks: List[Dict[str, Any]] = []
    for g in games:
        tasks.append(
            {
                "game_id": g.game_id,
                "moves_uci": [m.uci() for m in g.board.move_stack],
            }
        )

    if not tasks:
        return {}

    # Serial path: reuse the already-open eval engine when available.
    if int(cfg.acpl_workers) <= 1:
        if eval_engine is None:
            print("[fullgame][acpl] mode=serial open_engine=1", flush=True)
            with chess.engine.SimpleEngine.popen_uci(cfg.stockfish_eval.path) as engine:
                _configure_stockfish(engine, cfg.stockfish_eval)
                return {
                    t["game_id"]: _analyze_game_with_engine(
                        engine=engine,
                        moves_uci=t["moves_uci"],
                        depth=int(cfg.acpl_eval_depth),
                        movetime_ms=int(cfg.acpl_eval_movetime_ms),
                        cp_cap=int(cfg.acpl_eval_cp_cap),
                        mate_score_cp=int(cfg.mate_score_cp),
                    )
                    for t in tasks
                }

        print("[fullgame][acpl] mode=serial open_engine=0", flush=True)
        out: Dict[str, Dict[str, Any]] = {}
        for t in tasks:
            out[t["game_id"]] = _analyze_game_with_engine(
                engine=eval_engine,
                moves_uci=t["moves_uci"],
                depth=int(cfg.acpl_eval_depth),
                movetime_ms=int(cfg.acpl_eval_movetime_ms),
                cp_cap=int(cfg.acpl_eval_cp_cap),
                mate_score_cp=int(cfg.mate_score_cp),
            )
        return out

    # Parallel path: shard tasks across N workers (spawn-safe).
    workers = min(int(cfg.acpl_workers), len(tasks))
    workers = max(1, workers)
    print(
        f"[fullgame][acpl] mode=parallel workers={workers} threads_per_worker={cfg.stockfish_eval.threads} games={len(tasks)}",
        flush=True,
    )

    shards: List[List[Dict[str, Any]]] = [[] for _ in range(workers)]
    for i, t in enumerate(tasks):
        shards[i % workers].append(t)

    tmp_dir = cfg.out_dir / "_tmp" / "acpl"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out_paths_by_worker: Dict[int, Path] = {}
    for wid in range(workers):
        out_paths_by_worker[int(wid)] = tmp_dir / f"acpl_worker{wid:03d}.jsonl"

    def _launch_worker_attempt(worker_ids: List[int], *, attempt: int) -> List[int]:
        procs: List[Tuple[int, mp.Process]] = []
        failed: List[int] = []
        for wid in worker_ids:
            out_path = out_paths_by_worker[int(wid)]
            # Avoid stale rows from previous failed attempts.
            try:
                if out_path.exists():
                    out_path.unlink()
            except Exception:
                pass
            p = ctx.Process(
                target=_acpl_worker_main,
                kwargs=dict(
                    worker_id=int(wid),
                    tasks=shards[wid],
                    output_path=str(out_path),
                    engine_path=str(cfg.stockfish_eval.path),
                    engine_cfg=cfg.stockfish_eval,
                    depth=int(cfg.acpl_eval_depth),
                    movetime_ms=int(cfg.acpl_eval_movetime_ms),
                    cp_cap=int(cfg.acpl_eval_cp_cap),
                    mate_score_cp=int(cfg.mate_score_cp),
                ),
            )
            try:
                p.start()
            except Exception as exc:
                print(
                    f"[fullgame][acpl][warn] failed to start worker id={wid} attempt={attempt}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                failed.append(int(wid))
                continue
            procs.append((int(wid), p))

        for wid, p in procs:
            p.join()
            if p.exitcode != 0:
                failed.append(int(wid))
                print(
                    f"[fullgame][acpl][warn] worker id={wid} exited with code {p.exitcode} attempt={attempt}",
                    flush=True,
                )
        return sorted(set(failed))

    max_worker_retries = max(0, int(cfg.acpl_worker_retries))
    backoff_s = max(0.0, float(cfg.acpl_worker_retry_backoff_s))
    failed_worker_ids = _launch_worker_attempt(list(range(workers)), attempt=0)
    if failed_worker_ids:
        print(
            f"[fullgame][acpl][warn] initial failed workers={failed_worker_ids}; "
            f"max_retries={max_worker_retries}",
            flush=True,
        )
    for attempt in range(1, max_worker_retries + 1):
        if not failed_worker_ids:
            break
        if backoff_s > 0:
            time.sleep(backoff_s)
        print(
            f"[fullgame][acpl] retry attempt={attempt} workers={failed_worker_ids}",
            flush=True,
        )
        failed_worker_ids = _launch_worker_attempt(failed_worker_ids, attempt=attempt)

    analyses: Dict[str, Dict[str, Any]] = {}
    err_count = 0
    for _, out_path in sorted(out_paths_by_worker.items(), key=lambda kv: kv[0]):
        if not out_path.exists():
            continue
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # Preserve forward progress even if a crashed worker left a partial JSON line.
                    continue
                game_id = str(obj.get("game_id", ""))
                analysis = obj.get("analysis", None)
                if isinstance(analysis, dict) and game_id:
                    analyses[game_id] = analysis
                if obj.get("error"):
                    err_count += 1

    if failed_worker_ids:
        raise RuntimeError(
            "ACPL workers failed after retries; aborting eval "
            f"(failed_workers={sorted(failed_worker_ids)}, retries={max_worker_retries})"
        )

    # Validate that every scheduled game has ACPL output from workers.
    task_by_game_id: Dict[str, Dict[str, Any]] = {}
    for shard in shards:
        for t in shard:
            gid = str(t.get("game_id", ""))
            if gid:
                task_by_game_id[gid] = t

    missing_game_ids = sorted(gid for gid in task_by_game_id.keys() if gid not in analyses)
    if missing_game_ids:
        preview = ", ".join(missing_game_ids[:10])
        if len(missing_game_ids) > 10:
            preview += ", ..."
        raise RuntimeError(
            "ACPL output incomplete after worker execution; aborting eval "
            f"(missing_games={len(missing_game_ids)} [{preview}])"
        )

    print(
        f"[fullgame][acpl] merged games={len(analyses)}/{len(tasks)} "
        f"worker_errors={err_count} failed_workers={len(failed_worker_ids)} "
        "fallback_errors=0",
        flush=True,
    )
    return analyses


def _pgn_result_to_result_str(pgn_result: str) -> str:
    if pgn_result == "1-0":
        return "White wins"
    if pgn_result == "0-1":
        return "Black wins"
    if pgn_result == "1/2-1/2":
        return "Draw"
    return pgn_result or ""


_UNICODE_PIECES: Dict[str, str] = {
    "P": "♙",
    "R": "♖",
    "N": "♘",
    "B": "♗",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "r": "♜",
    "n": "♞",
    "b": "♝",
    "q": "♛",
    "k": "♚",
}


def _render_board_unicode(board: chess.Board) -> str:
    """Render the chess board using Unicode pieces and coordinates (starter-kit style)."""
    lines: List[str] = []
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ranks = ["8", "7", "6", "5", "4", "3", "2", "1"]

    coord_parts = [f" {file} " for file in files]
    coord_line = "   " + "".join(coord_parts) + "  "
    lines.append(coord_line)
    border_width = len(files) * 3
    lines.append("   +" + "-" * border_width + "+")

    for rank in ranks:
        line_parts: List[str] = [f"{rank} |"]
        for file in files:
            square = chess.parse_square(file + rank)
            piece = board.piece_at(square)
            if piece is None:
                piece_char = "·"
            else:
                piece_char = _UNICODE_PIECES[piece.symbol()]
            line_parts.append(f" {piece_char} ")
        line_parts.append(f"| {rank}")
        lines.append("".join(line_parts))

    lines.append("   +" + "-" * border_width + "+")
    lines.append(coord_line)
    return "\n".join(lines)


def _build_prompt_context(
    *,
    board: chess.Board,
    legal_moves: List[chess.Move],
    move_history: List[str],
    side_to_move: str,
    needed_vars: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Build the starter-kit prompt template context."""
    # Default: preserve the full context keys (starter-kit template).
    # For performance, callers may pass `needed_vars` (from Jinja meta analysis) so we only
    # compute keys that are actually rendered by the prompt template.
    needed = needed_vars
    fen = board.fen() if (needed is None or "FEN" in needed) else ""
    board_utf = _render_board_unicode(board) if (needed is None or "board_utf" in needed) else ""
    board_ascii = board.unicode() if (needed is None or "board_ascii" in needed) else ""

    if needed is None or "last_move" in needed:
        if board.move_stack:
            last_move = board.move_stack[-1]
            temp_board = chess.Board()
            for mv in board.move_stack[:-1]:
                temp_board.push(mv)
            last_move_san = temp_board.san(last_move)
            last_side = "Black" if board.turn else "White"
            last_move_desc = f"{last_side} played {last_move_san}"
        else:
            last_move_desc = "(start of game)"
    else:
        last_move_desc = ""

    want_legal_moves_uci_list = (
        needed is None
        or "legal_moves_uci_list" in needed
        or "legal_moves_uci" in needed
        or "first_legal_move" in needed
        # Selection templates use `considered_moves_uci_list` (a subset of legal moves).
        or "considered_moves_uci_list" in needed
    )
    legal_moves_uci_list = [m.uci() for m in legal_moves] if want_legal_moves_uci_list else []
    legal_moves_san_list = [board.san(m) for m in legal_moves] if (needed is None or "legal_moves_san_list" in needed or "legal_moves_san" in needed) else []
    legal_moves_uci_str = " ".join(legal_moves_uci_list) if (needed is None or "legal_moves_uci" in needed) else ""
    legal_moves_san_str = " ".join(legal_moves_san_list) if (needed is None or "legal_moves_san" in needed) else ""

    want_hist_uci_list = needed is None or "move_history_uci_list" in needed or "move_history_uci" in needed
    want_hist_san_list = needed is None or "move_history_san_list" in needed or "move_history_san" in needed
    if move_history and (want_hist_uci_list or want_hist_san_list):
        move_history_uci_list = list(move_history) if want_hist_uci_list else []
        move_history_uci_str = " ".join(move_history_uci_list) if (needed is None or "move_history_uci" in needed) else ""

        if want_hist_san_list:
            try:
                history_board = chess.Board()
                move_history_san_list = []
                for uci_move in move_history:
                    try:
                        mv = chess.Move.from_uci(uci_move)
                        san = history_board.san(mv)
                        move_history_san_list.append(san)
                        history_board.push(mv)
                    except Exception:
                        move_history_san_list.append(uci_move)
                move_history_san_str = " ".join(move_history_san_list) if (needed is None or "move_history_san" in needed) else ""
            except Exception:
                move_history_san_list = list(move_history)
                move_history_san_str = " ".join(move_history) if (needed is None or "move_history_san" in needed) else ""
        else:
            move_history_san_list = []
            move_history_san_str = ""
    else:
        move_history_uci_list = []
        move_history_san_list = []
        move_history_uci_str = "(no moves yet)" if (needed is None or "move_history_uci" in needed) else ""
        move_history_san_str = "(no moves yet)" if (needed is None or "move_history_san" in needed) else ""

    first_legal_move = legal_moves_uci_list[0] if legal_moves_uci_list else ""
    if needed is not None and "first_legal_move" not in needed:
        first_legal_move = ""

    # Full-game eval does not use restricted candidate sets. For selection templates,
    # interpret "considered moves" as "all legal moves" for the current position.
    considered_moves_uci_list: List[str] = list(legal_moves_uci_list) if (needed is None or "considered_moves_uci_list" in needed) else []

    return {
        "board_utf": board_utf,
        "board_ascii": board_ascii,
        "FEN": fen,
        "side_to_move": side_to_move if (needed is None or "side_to_move" in needed) else "",
        "last_move": last_move_desc,
        "legal_moves_uci": legal_moves_uci_str,
        "legal_moves_san": legal_moves_san_str,
        "move_history_uci": move_history_uci_str,
        "move_history_san": move_history_san_str,
        "legal_moves_uci_list": legal_moves_uci_list,
        "considered_moves_uci_list": considered_moves_uci_list,
        "legal_moves_san_list": legal_moves_san_list,
        "move_history_uci_list": move_history_uci_list,
        "move_history_san_list": move_history_san_list,
        "first_legal_move": first_legal_move,
    }


def _analyze_game_with_engine(
    *,
    engine: chess.engine.SimpleEngine,
    moves_uci: List[str],
    depth: int,
    movetime_ms: int,
    cp_cap: int,
    mate_score_cp: int,
    initial_fen: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute starter-kit-style per-side ACPL and engine-match accuracy.

    This matches `competition/starter-kit/chess-env/run_game.py::_StockfishAnalyzer`
    and `competition/starter-kit/local_evaluation.py` (depth=20, movetime=1000ms, cp capped to ±1000).
    """

    def _limit() -> chess.engine.Limit:
        return chess.engine.Limit(
            time=float(movetime_ms) / 1000.0,
            depth=int(depth) if int(depth) > 0 else None,
        )

    def _analyse_position(b: chess.Board) -> tuple[int, Optional[str]]:
        info = engine.analyse(b, _limit())

        score = info.get("score")
        rel = score.relative if score is not None else None
        if rel is None:
            eval_cp = int(cp_cap)
        else:
            cp = rel.score(mate_score=int(mate_score_cp))
            if cp is None:
                eval_cp = int(cp_cap)
            else:
                eval_cp = int(max(-int(cp_cap), min(int(cp_cap), int(cp))))

        best_move = None
        pv = info.get("pv") or []
        if pv:
            try:
                best_move = pv[0].uci()
            except Exception:
                best_move = None

        return eval_cp, best_move

    board = chess.Board(initial_fen) if initial_fen else chess.Board()
    white_moves = 0
    black_moves = 0
    white_matches = 0
    black_matches = 0
    white_cpl_sum = 0.0
    black_cpl_sum = 0.0

    for uci in moves_uci:
        eval_before, best_move = _analyse_position(board)
        is_white = bool(board.turn)

        if best_move and best_move == uci:
            if is_white:
                white_matches += 1
            else:
                black_matches += 1

        move = chess.Move.from_uci(uci)
        board.push(move)

        eval_after, _ = _analyse_position(board)
        eval_after_mover_pov = -eval_after

        cpl = float(max(0, int(eval_before) - int(eval_after_mover_pov)))

        if is_white:
            white_moves += 1
            white_cpl_sum += cpl
        else:
            black_moves += 1
            black_cpl_sum += cpl

    white_accuracy = (white_matches / white_moves * 100.0) if white_moves else 0.0
    black_accuracy = (black_matches / black_moves * 100.0) if black_moves else 0.0
    white_acpl = (white_cpl_sum / white_moves) if white_moves else float(cp_cap)
    black_acpl = (black_cpl_sum / black_moves) if black_moves else float(cp_cap)

    return {
        "white_accuracy_pct": float(white_accuracy),
        "black_accuracy_pct": float(black_accuracy),
        "white_acpl": float(white_acpl),
        "black_acpl": float(black_acpl),
        "white_moves": int(white_moves),
        "black_moves": int(black_moves),
        "white_cpl_sum": float(white_cpl_sum),
        "black_cpl_sum": float(black_cpl_sum),
    }


class _DeterministicRng:
    """Tiny deterministic RNG wrapper (stable across Python versions)."""

    def __init__(self, seed: int, *, salt: str):
        self._state = _mix_seed(seed, salt=salt)

    def shuffle(self, items: List[Any]) -> None:
        # Fisher-Yates with xorshift32.
        n = len(items)
        for i in range(n - 1, 0, -1):
            j = self._randbelow(i + 1)
            items[i], items[j] = items[j], items[i]

    def _randbelow(self, n: int) -> int:
        if n <= 0:
            return 0
        self._state = _xorshift32(self._state)
        return int(self._state % n)


def _xorshift32(x: int) -> int:
    x &= 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


def _mix_seed(seed: int, *, salt: str) -> int:
    # Simple FNV-1a-ish mixing into 32 bits.
    h = 2166136261
    h ^= int(seed) & 0xFFFFFFFF
    h = (h * 16777619) & 0xFFFFFFFF
    for ch in salt.encode("utf-8"):
        h ^= int(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h & 0xFFFFFFFF
