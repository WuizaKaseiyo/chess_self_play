#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Make repo-root imports work when invoked as `python scripts/xxx.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.chess.full_game_eval import FullGameEvalConfig, StockfishConfig, run_full_game_eval
from verl.utils.prompt import encode_prompt_from_messages, infer_use_chat_template_from_model_name, is_qwen3_base_model


def _sanitize_model_name(model: str) -> str:
    safe = model.replace("/", "__").replace(":", "_")
    safe = "".join(ch if (ch.isalnum() or ch in "._-__") else "_" for ch in safe)
    return safe


def _looks_like_hf_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    cfg = path / "config.json"
    if not cfg.exists():
        return False
    # Weights may be sharded, but at least one weight file should exist.
    weight_globs = [
        "pytorch_model*.bin",
        "model*.safetensors",
        "*.safetensors",
    ]
    for pat in weight_globs:
        if list(path.glob(pat)):
            return True
    return False


def _looks_like_fsdp_actor_ckpt(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "fsdp_config.json").exists():
        return False
    if not list(path.glob("model_world_size_*_rank_*.pt")):
        return False
    if not (path / "huggingface" / "config.json").exists():
        return False
    return True


def _merge_fsdp_actor_to_hf(
    *,
    actor_dir: Path,
    target_dir: Path,
    trust_remote_code: bool,
    use_cpu_initialization: bool,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(target_dir),
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if use_cpu_initialization:
        cmd.append("--use_cpu_initialization")

    print(f"[fullgame] Merging FSDP checkpoint -> HF: {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def _resolve_model_for_vllm(
    *,
    model: str,
    out_dir: Path,
    trust_remote_code: bool,
    use_cpu_initialization: bool,
) -> str:
    # HF repo id (not a local dir) -> pass through.
    p = Path(model)
    if not p.exists():
        return model

    if p.is_dir() and _looks_like_hf_model_dir(p):
        return str(p)

    # Support passing ".../actor/huggingface" by promoting to the actor dir.
    if p.name == "huggingface" and _looks_like_fsdp_actor_ckpt(p.parent):
        p = p.parent

    if p.is_dir() and _looks_like_fsdp_actor_ckpt(p):
        merged_dir = out_dir / "merged_hf_model"
        if not _looks_like_hf_model_dir(merged_dir):
            _merge_fsdp_actor_to_hf(
                actor_dir=p,
                target_dir=merged_dir,
                trust_remote_code=trust_remote_code,
                use_cpu_initialization=use_cpu_initialization,
            )
        return str(merged_dir)

    raise ValueError(
        f"Unrecognized model path: {model}\n"
        f"- If this is a Hugging Face repo id, pass it as-is (and ensure network access).\n"
        f"- If this is an FSDP actor checkpoint, pass the actor dir (contains fsdp_config.json + model_world_size_*).\n"
        f"- If this is a merged HF dir, it must contain config.json and weight files."
    )


class VllmChatBackend:
    def __init__(
        self,
        *,
        model: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_num_seqs: int,
        max_num_batched_tokens: int | None,
        enforce_eager: bool,
        seed: int,
        trust_remote_code: bool,
        use_chat_template: bool,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat_template = bool(use_chat_template)

        llm_kwargs = dict(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
            disable_log_stats=True,
            seed=seed,
            max_num_seqs=max_num_seqs,
        )
        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
        self.llm = LLM(**llm_kwargs)

    def generate(
        self,
        prompts: List[List[Dict[str, str]]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seeds: Optional[List[int]] = None,
    ) -> List[str]:
        prompt_token_ids: List[List[int]] = []
        for messages in prompts:
            _, ids = encode_prompt_from_messages(
                self.tokenizer,
                messages,
                use_chat_template=self.use_chat_template,
                add_generation_prompt=True,
            )
            prompt_token_ids.append([int(x) for x in ids])

        vllm_inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
        if seeds is None:
            sampling_params = SamplingParams(
                n=1,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=-1,
                min_p=0.0,
                max_tokens=int(max_tokens),
                repetition_penalty=1.0,
            )
            outputs = self.llm.generate(prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
        else:
            if len(seeds) != len(vllm_inputs):
                raise ValueError(f"seeds length {len(seeds)} != batch size {len(vllm_inputs)}")
            sampling_params_batch = [
                SamplingParams(
                    n=1,
                    seed=int(s),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=-1,
                    min_p=0.0,
                    max_tokens=int(max_tokens),
                    repetition_penalty=1.0,
                )
                for s in seeds
            ]
            outputs = self.llm.generate(prompts=vllm_inputs, sampling_params=sampling_params_batch, use_tqdm=False)

        decoded: List[str] = []
        for out in outputs:
            if not out.outputs:
                decoded.append("")
                continue
            # We keep detokenize=True for stop-string support.
            text = getattr(out.outputs[0], "text", None)
            if isinstance(text, str):
                decoded.append(text)
                continue
            token_ids = out.outputs[0].token_ids
            decoded.append(self.tokenizer.decode(token_ids, skip_special_tokens=True))
        return decoded


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-game chess evaluation harness (vLLM + Stockfish).")
    p.add_argument("--model", type=str, required=True, help="HF model id/path, or VERL FSDP actor checkpoint dir.")
    p.add_argument("--out-dir", type=str, default="", help="Output directory. Default: outputs/full_game_eval/<run>/")
    p.add_argument(
        "--prompt-template-path",
        type=str,
        default="recipe/chess/prompt_templates/chess_rl_chessr1_prompt.jinja",
        help=(
            "Jinja prompt template path (should match the training prompt). "
            "Default: canonical training template under recipe/chess/prompt_templates/ "
            "(the submission template is copied from this file). "
            "Set to empty to use the legacy prompt builder."
        ),
    )

    p.add_argument("--opponent-depths", type=int, nargs="+", default=[1, 5], help="Stockfish depths to evaluate against.")
    p.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of 50-game rounds per depth (default=5 => 250 games).",
    )
    p.add_argument(
        "--games-per-round",
        type=int,
        default=50,
        help="Games per round (color-balanced across the total games).",
    )
    p.add_argument(
        "--games-per-depth",
        type=int,
        default=None,
        help="Total games per depth (overrides --rounds/--games-per-round).",
    )

    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-response-tokens", type=int, default=512)
    p.add_argument("--max-retries-per-turn", type=int, default=3)
    p.add_argument("--opponent-movetime-ms", type=int, default=100, help="Opponent Stockfish movetime budget (0 disables).")
    p.add_argument("--resignation-cpl", type=int, default=1000)
    p.add_argument("--acpl-depth", type=int, default=20)
    p.add_argument("--acpl-movetime-ms", type=int, default=1000, help="ACPL analysis movetime budget.")
    p.add_argument("--acpl-cp-cap", type=int, default=1000, help="ACPL analysis CP cap (also used as mate score).")
    p.add_argument("--mate-score-cp", type=int, default=1000, help="Mate score used for ACPL analysis.")
    p.add_argument("--max-plies", type=int, default=200, help="Max game length in plies (0 disables).")

    p.add_argument("--stockfish-path", type=str, default=".third_party_cache/stockfish/src/stockfish")
    p.add_argument("--stockfish-threads", type=int, default=1)
    p.add_argument("--stockfish-hash-mb", type=int, default=128)
    p.add_argument("--stockfish-skill-level", type=int, default=0, help="Opponent Stockfish skill (must be 0).")
    p.add_argument("--eval-skill-level", type=int, default=20, help="Reference Stockfish skill (depth-20 ACPL).")
    p.add_argument(
        "--acpl-workers",
        type=int,
        default=1,
        help="Number of CPU workers for ACPL computation (1 = serial).",
    )
    p.add_argument(
        "--acpl-threads",
        type=int,
        default=None,
        help="Stockfish Threads per ACPL worker (default: --stockfish-threads).",
    )

    # vLLM knobs (defaults chosen for this repo's known-good local setup).
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=5120)
    p.add_argument("--max-num-seqs", type=int, default=2048)
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Optional vLLM scheduler token cap. Set to a large value (e.g., 65536) for higher throughput.",
    )
    p.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graph (safer, often slower).")
    p.add_argument("--trust-remote-code", action="store_true", help="Required for some HF models (e.g., Qwen).")
    p.add_argument("--use-chat-template", dest="use_chat_template", action="store_true", default=None)
    p.add_argument("--no-use-chat-template", dest="use_chat_template", action="store_false")

    p.add_argument(
        "--no-batched-inference",
        action="store_true",
        help="Disable inference batching (slow; for benchmarking/debugging).",
    )

    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--merge-use-cpu-initialization",
        action="store_true",
        help="When merging FSDP checkpoints, init model on CPU to reduce peak memory.",
    )

    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny 2-game eval (1 white, 1 black) vs Stockfish depth 1.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.stockfish_skill_level != 0:
        raise ValueError("--stockfish-skill-level must be 0 per spec.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_model = _sanitize_model_name(args.model)
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/full_game_eval") / f"{safe_model}_{ts}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_for_vllm = _resolve_model_for_vllm(
        model=args.model,
        out_dir=out_dir,
        trust_remote_code=bool(args.trust_remote_code),
        use_cpu_initialization=bool(args.merge_use_cpu_initialization),
    )
    if args.use_chat_template is None:
        use_chat_template = infer_use_chat_template_from_model_name(model_for_vllm, default=True)
    else:
        use_chat_template = bool(args.use_chat_template)
    if is_qwen3_base_model(str(model_for_vllm)) and use_chat_template:
        raise ValueError("Qwen3 base full-game eval must use --no-use-chat-template.")

    # Persist invocation for reproducibility (helpful when runs happen inside docker).
    run_args_path = out_dir / "run_args.json"
    run_args_path.write_text(
        json.dumps(
            {
                "argv": list(sys.argv),
                "args": vars(args),
                "resolved_model_for_vllm": model_for_vllm,
                "use_chat_template": bool(use_chat_template),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("[fullgame] Wrote run args:", str(run_args_path), flush=True)

    if args.smoke_test:
        opponent_depths = [1]
        rounds = 1
        games_per_round = 2
        games_per_depth = 2
    else:
        opponent_depths = list(args.opponent_depths)
        if args.games_per_depth is not None:
            games_per_depth = int(args.games_per_depth)
            rounds = None
            games_per_round = None
        else:
            rounds = int(args.rounds)
            games_per_round = int(args.games_per_round)
            games_per_depth = rounds * games_per_round

    if int(games_per_depth) <= 0:
        raise ValueError(f"Invalid games_per_depth={games_per_depth}")

    backend = VllmChatBackend(
        model=model_for_vllm,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        max_num_seqs=int(args.max_num_seqs),
        max_num_batched_tokens=(
            int(args.max_num_batched_tokens)
            if (args.max_num_batched_tokens is not None and int(args.max_num_batched_tokens) > 0)
            else None
        ),
        enforce_eager=bool(args.enforce_eager),
        seed=int(args.seed),
        trust_remote_code=bool(args.trust_remote_code),
        use_chat_template=bool(use_chat_template),
    )

    acpl_threads = int(args.acpl_threads) if args.acpl_threads is not None else int(args.stockfish_threads)

    cfg = FullGameEvalConfig(
        opponent_depths=opponent_depths,
        games_per_depth=games_per_depth,
        seed=int(args.seed),
        rounds=rounds,
        games_per_round=games_per_round,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_response_tokens=int(args.max_response_tokens),
        max_retries_per_turn=int(args.max_retries_per_turn),
        opponent_movetime_ms=int(args.opponent_movetime_ms),
        resignation_cpl=int(args.resignation_cpl),
        acpl_eval_depth=int(args.acpl_depth),
        acpl_eval_movetime_ms=int(args.acpl_movetime_ms),
        acpl_eval_cp_cap=int(args.acpl_cp_cap),
        acpl_workers=int(args.acpl_workers),
        mate_score_cp=int(args.mate_score_cp),
        max_plies=(int(args.max_plies) if int(args.max_plies) > 0 else None),
        batched_inference=(not bool(args.no_batched_inference)),
        stockfish_opponent=StockfishConfig(
            path=str(args.stockfish_path),
            threads=int(args.stockfish_threads),
            hash_mb=int(args.stockfish_hash_mb),
            skill_level=int(args.stockfish_skill_level),
        ),
        stockfish_eval=StockfishConfig(
            path=str(args.stockfish_path),
            threads=int(acpl_threads),
            hash_mb=int(args.stockfish_hash_mb),
            skill_level=int(args.eval_skill_level),
        ),
        prompt_template_path=(str(args.prompt_template_path) if str(args.prompt_template_path).strip() else None),
        out_dir=out_dir,
    )

    summary = run_full_game_eval(cfg=cfg, backend=backend)
    print("[fullgame] Wrote summary:", summary["paths"]["summary_json"])
    for depth_key, row in summary["summary_by_depth"].items():
        print(f"[fullgame] {depth_key}: {row}")


if __name__ == "__main__":
    # Avoid HF tokenizer parallelism oversubscription (can slow down startup in docker).
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
