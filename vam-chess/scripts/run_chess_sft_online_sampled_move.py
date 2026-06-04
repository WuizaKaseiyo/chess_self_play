#!/usr/bin/env python3
from __future__ import annotations

"""
Online chess SFT (standalone) with sampled `{move}` forced-prefix variants (A/B/C/E).

High-level loop (per global step):
  1) Take a batch of 128 prompts (global, drop_last).
  2) For each prompt, sample 8 `{move}` candidates from the RL forced-move distribution.
  3) Generate continuations from the *current* model with the forced-prefix injected.
  4) Score generations with the chess reward fn (format + scorable move + expected-score fields).
  5) Do exactly one SFT optimization step on this batch (weighted by sft_weighting).

This is intentionally research-oriented and favors auditability over efficiency.

Notes on generation engine:
  - We use HuggingFace `generate()` so the next step's generations come from the updated in-memory model.
  - vLLM is not used for in-loop rollouts because hot-updating vLLM weights batch-to-batch is non-trivial.
"""

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe.chess.reward_fn import compute_score_batch
from scripts.build_chess_sft_prefix_dataset import VARIANTS, _sample_forced_moves, _strip_boilerplate, _stable_u32_seed


def _init_dist() -> tuple[int, int, int, torch.device]:
    """Return (rank, world_size, local_rank, device)."""
    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    x = np.where(np.isfinite(x), x, -np.inf)
    m = np.max(x)
    if not np.isfinite(m):
        return np.full_like(x, 1.0 / float(len(x)), dtype=np.float64)
    e = np.exp(x - m)
    s = float(np.sum(e))
    if not np.isfinite(s) or s <= 0.0:
        return np.full_like(x, 1.0 / float(len(x)), dtype=np.float64)
    return e / s


_UCI_TOKEN_RE = re.compile(r"\b([a-h][1-8][a-h][1-8](?:=[qrbnQRBN]|[qrbnQRBN])?)\b")
_TAG_STRIP_RE = re.compile(r"</?think>|</?uci_move>", re.IGNORECASE)


def _normalize_uci_token(token: str) -> str | None:
    if not token:
        return None
    t = token.strip()
    t = t.strip("`'\"")
    t = t.rstrip(".!?;,:")
    if re.fullmatch(r"[a-h][1-8][a-h][1-8]=[QRBNqrbn]", t):
        return t.replace("=", "").lower()
    if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbnQRBN]?", t):
        return t.lower()
    return None


def _first_uci_in_text(text: str) -> str | None:
    if not text:
        return None
    for m in _UCI_TOKEN_RE.finditer(text):
        tok = _normalize_uci_token(m.group(1) or "")
        if tok:
            return tok
    return None


def _canonicalize_strict_response(*, text: str, forced_move: str) -> str:
    """Best-effort convert a decoded response into strict `<think>...</think><uci_move>...</uci_move>`.

    This is used for online SFT to avoid "format_error" collapse early on when the
    base model tends to forget closing tags.
    """
    s = (text or "").strip()
    low = s.lower()

    think_open = "<think>"
    think_close = "</think>"
    move_open = "<uci_move>"
    move_close = "</uci_move>"

    # Extract a think span from the original text when possible.
    i_to = low.find(think_open)
    if i_to == -1:
        think_body = s
    else:
        i_think_start = i_to + len(think_open)
        i_move_open = low.find(move_open, i_think_start)
        i_tc = low.find(think_close, i_think_start)
        if i_tc != -1 and (i_move_open == -1 or i_tc < i_move_open):
            think_body = s[i_think_start:i_tc]
        elif i_move_open != -1:
            think_body = s[i_think_start:i_move_open]
        else:
            think_body = s[i_think_start:]

    # Avoid introducing extra tag occurrences inside the think block (reward fn counts tags globally).
    think_body = _TAG_STRIP_RE.sub("", think_body)

    # Extract a move from an existing <uci_move> block if present; else fall back.
    mv: str | None = None
    i_mo = low.find(move_open)
    if i_mo != -1:
        i_move_start = i_mo + len(move_open)
        i_mc = low.find(move_close, i_move_start)
        move_body = s[i_move_start:i_mc] if i_mc != -1 else s[i_move_start:]
        mv = _first_uci_in_text(move_body)

    if mv is None:
        # If the model didn't provide a parsable move, fall back to the forced move to keep examples scorable.
        mv = _normalize_uci_token(forced_move) if forced_move else None

    mv = mv or ""
    return f"<think>{think_body}</think><uci_move> {mv} </uci_move>"


def _build_train_tensors(
    *,
    tokenizer,
    prompt_msgs: list[dict[str, Any]],
    response_text: str,
    max_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (input_ids, attention_mask, loss_mask) padded to max_length.

    Uses the same masking rule as `verl/utils/dataset/sft_dataset.py`:
      - loss_mask is 0 on prompt (except the last prompt token, which predicts the first response token),
      - and 0 on the last response token.
    """
    prompt_str = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    resp_str = (response_text or "") + (tokenizer.eos_token or "")
    resp_ids = tokenizer.encode(resp_str, add_special_tokens=False)

    input_ids = prompt_ids + resp_ids
    attention_mask = [1] * len(input_ids)
    loss_mask = attention_mask.copy()

    prompt_len = len(prompt_ids)
    resp_len = len(resp_ids)
    if prompt_len > 1:
        loss_mask[: prompt_len - 1] = [0] * (prompt_len - 1)
    # Mask last token in response.
    last_idx = min(prompt_len + resp_len, len(loss_mask)) - 1
    if last_idx >= 0:
        loss_mask[last_idx] = 0

    # Truncate/pad (right).
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        loss_mask = loss_mask[:max_length]
    elif len(input_ids) < max_length:
        pad = max_length - len(input_ids)
        input_ids = input_ids + [pad_token_id] * pad
        attention_mask = attention_mask + [0] * pad
        loss_mask = loss_mask + [0] * pad

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(loss_mask, dtype=torch.long),
    )


def _compute_weighted_token_mean_loss(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Token-mean CE, weighted by per-example sample_weight."""
    # input_ids: (B, S)
    # loss_mask: (B, S) with 0/1, aligned with input_ids.
    bsz, seqlen = input_ids.shape
    labels = input_ids[:, 1:].contiguous()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels.view(-1).to(shift_logits.device)
    token_losses = nn.functional.cross_entropy(shift_logits, shift_labels, reduction="none").view(bsz, seqlen - 1)

    lm = loss_mask[:, 1:].to(token_losses.device).to(torch.float32)
    w = sample_weight.to(token_losses.device).to(torch.float32).view(bsz, 1)
    lm = lm * w

    numer = torch.sum(token_losses * lm)
    denom = torch.sum(lm) + 1e-8
    return numer / denom


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["A", "B", "C", "E"], required=True)
    p.add_argument("--model_path", required=True, help="HF id or local path (must include tokenizer).")
    p.add_argument("--train_parquet", default="data/chess_puzzles/train_hard.parquet")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)

    # Online loop sizing.
    p.add_argument("--prompts_per_step", type=int, default=128)
    p.add_argument("--moves_per_prompt", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--total_steps", type=int, default=None, help="Optional override; default is epochs * floor(N/128).")

    # Forced-move sampling (match RL).
    p.add_argument("--move_temperature", type=float, default=2.0)

    # Weighting schemes (match RL aux-trace weighting semantics; per-prompt group of 8).
    p.add_argument("--sft_weighting", choices=["awr", "uniform", "best_only"], default="uniform")
    p.add_argument("--awr_beta", type=float, default=2.0)

    # Generation params.
    p.add_argument("--gen_max_new_tokens", type=int, default=256)
    p.add_argument("--gen_do_sample", action="store_true", default=False)
    p.add_argument("--gen_temperature", type=float, default=0.6)
    p.add_argument("--gen_top_p", type=float, default=0.95)
    p.add_argument("--gen_batch_size", type=int, default=32)
    p.add_argument(
        "--debug_dump_examples",
        type=int,
        default=5,
        help="Rank0 only: write a small JSON of (forced_move, response, score, weight) for manual inspection.",
    )

    # Train params.
    p.add_argument("--train_max_length", type=int, default=1024)
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true", default=True)
    args = p.parse_args()

    rank, world_size, local_rank, device = _init_dist()
    if args.prompts_per_step % world_size != 0:
        raise ValueError(f"--prompts_per_step must be divisible by world_size={world_size}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "online_sft_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        cfg_path = out_dir / "online_config.json"
        cfg_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # Load train rows (small enough to keep in memory).
    table = pq.read_table(args.train_parquet, columns=["prompt", "reward_model", "extra_info"])
    rows = table.to_pylist()
    train_rows = len(rows)
    steps_per_epoch = train_rows // int(args.prompts_per_step)
    if steps_per_epoch <= 0:
        raise ValueError(
            f"train_rows={train_rows} too small for prompts_per_step={args.prompts_per_step} (drop_last semantics)."
        )
    total_steps = int(args.total_steps) if args.total_steps is not None else int(args.epochs) * int(steps_per_epoch)
    if rank == 0:
        print(f"[online_sft] train_rows={train_rows} steps_per_epoch={steps_per_epoch} total_steps={total_steps}")

    # Tokenizer + model.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        # Decoder-only models often have no pad token; reuse EOS for padding.
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if bool(args.bf16) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch_dtype)
    model.to(device)
    # Online SFT uses plain DDP (not FSDP), so activation memory can be large on 7B+ models.
    # Enable gradient checkpointing to reduce peak memory and avoid OOM in the CE/logsoftmax.
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    optimizer = torch.optim.AdamW(
        [p for p in ddp.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )

    variant = VARIANTS[str(args.variant)]
    if not variant.forced_prefix_template:
        raise ValueError(f"variant={args.variant} must have a non-empty forced_prefix_template for online SFT")

    prompts_per_rank = int(args.prompts_per_step) // world_size
    moves_per_prompt = int(args.moves_per_prompt)
    examples_per_rank = prompts_per_rank * moves_per_prompt

    epoch_perm_cache: dict[int, np.ndarray] = {}

    for global_step in range(total_steps):
        t0 = time.time()
        epoch_idx = global_step // steps_per_epoch
        step_in_epoch = global_step % steps_per_epoch

        if epoch_idx not in epoch_perm_cache:
            rng = np.random.default_rng(int(args.seed) + int(epoch_idx))
            perm = rng.permutation(train_rows)
            # Drop last incomplete batch for determinism.
            perm = perm[: steps_per_epoch * int(args.prompts_per_step)]
            epoch_perm_cache[epoch_idx] = perm

        perm = epoch_perm_cache[epoch_idx]
        batch_start = step_in_epoch * int(args.prompts_per_step)
        batch_idxs = perm[batch_start : batch_start + int(args.prompts_per_step)]
        local_prompt_idxs = batch_idxs[rank * prompts_per_rank : (rank + 1) * prompts_per_rank].tolist()

        # Build generation inputs (one example per (prompt, sampled_move)).
        gen_inputs: list[list[int]] = []
        base_prompt_lens: list[int] = []
        full_input_lens: list[int] = []
        prompt_msgs_list: list[list[dict[str, Any]]] = []
        reward_models: list[dict[str, Any]] = []
        forced_moves: list[str] = []
        group_ids: list[int] = []
        move_sample_idxs: list[int] = []

        for prompt_pos, row_idx in enumerate(local_prompt_idxs):
            row = rows[int(row_idx)]
            prompt_msgs = row.get("prompt") or []
            reward_model = dict(row.get("reward_model") or {})
            extra_info = row.get("extra_info") or {}
            group_idx = int(extra_info.get("index", -1) or -1)

            if not isinstance(prompt_msgs, list) or not prompt_msgs:
                continue

            fen = str(reward_model.get("fen", "") or "")
            idx = str(group_idx)
            seed_u32 = _stable_u32_seed(str(args.seed), fen, idx, f"step{global_step}", "forced_moves")
            sampled_moves, _vals, _src = _sample_forced_moves(
                reward_model,
                seed=int(seed_u32),
                temperature=float(args.move_temperature),
                count=moves_per_prompt,
            )

            prompt_text = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False)
            base_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            base_len = len(base_ids)

            for j, mv in enumerate(sampled_moves):
                prefix_text = variant.forced_prefix_template.format(move=str(mv).strip().lower())
                prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
                full_ids = base_ids + prefix_ids

                gen_inputs.append(full_ids)
                base_prompt_lens.append(base_len)
                full_input_lens.append(len(full_ids))
                prompt_msgs_list.append(prompt_msgs)
                reward_models.append(reward_model)
                forced_moves.append(str(mv).strip().lower())
                group_ids.append(group_idx)
                move_sample_idxs.append(int(j))

        # Pad to fixed size so each rank participates in DDP collectives.
        # If we dropped any malformed prompts above, pad with duplicates (weight=0 later).
        if len(gen_inputs) == 0:
            # Degenerate: no valid prompts on this rank.
            # Create a dummy single-token prompt to keep code paths alive.
            gen_inputs = [[tokenizer.eos_token_id]]
            base_prompt_lens = [0]
            full_input_lens = [1]
            prompt_msgs_list = [[{"role": "user", "content": ""}]]
            reward_models = [{}]
            forced_moves = [""]
            group_ids = [-1]
            move_sample_idxs = [0]

        while len(gen_inputs) < examples_per_rank:
            gen_inputs.append(gen_inputs[-1])
            base_prompt_lens.append(base_prompt_lens[-1])
            full_input_lens.append(full_input_lens[-1])
            prompt_msgs_list.append(prompt_msgs_list[-1])
            reward_models.append(reward_models[-1])
            forced_moves.append(forced_moves[-1])
            group_ids.append(group_ids[-1])
            move_sample_idxs.append(move_sample_idxs[-1])

        if len(gen_inputs) > examples_per_rank:
            gen_inputs = gen_inputs[:examples_per_rank]
            base_prompt_lens = base_prompt_lens[:examples_per_rank]
            full_input_lens = full_input_lens[:examples_per_rank]
            prompt_msgs_list = prompt_msgs_list[:examples_per_rank]
            reward_models = reward_models[:examples_per_rank]
            forced_moves = forced_moves[:examples_per_rank]
            group_ids = group_ids[:examples_per_rank]
            move_sample_idxs = move_sample_idxs[:examples_per_rank]

        # Generation.
        ddp.module.eval()
        responses: list[str] = [""] * examples_per_rank
        with torch.no_grad():
            for start in range(0, examples_per_rank, int(args.gen_batch_size)):
                end = min(examples_per_rank, start + int(args.gen_batch_size))
                sub_inputs = gen_inputs[start:end]
                sub_base_lens = base_prompt_lens[start:end]
                sub_full_lens = full_input_lens[start:end]

                max_in = max(len(x) for x in sub_inputs)
                pad_id = int(tokenizer.pad_token_id or 0)
                input_ids = torch.full((len(sub_inputs), max_in), pad_id, dtype=torch.long, device=device)
                attn = torch.zeros((len(sub_inputs), max_in), dtype=torch.long, device=device)
                for i, ids in enumerate(sub_inputs):
                    ids_t = torch.tensor(ids, dtype=torch.long, device=device)
                    input_ids[i, max_in - len(ids) :] = ids_t
                    attn[i, max_in - len(ids) :] = 1

                gen = ddp.module.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=int(args.gen_max_new_tokens),
                    do_sample=bool(args.gen_do_sample),
                    temperature=float(args.gen_temperature),
                    top_p=float(args.gen_top_p),
                    pad_token_id=pad_id,
                    eos_token_id=int(tokenizer.eos_token_id or pad_id),
                    use_cache=True,
                )

                # Extract assistant response (forced prefix + generated continuation).
                gen = gen.detach().cpu()
                for i in range(gen.size(0)):
                    full_len = int(sub_full_lens[i])
                    base_len = int(sub_base_lens[i])
                    pad_len = int(max_in - full_len)
                    start_idx = pad_len + base_len
                    resp_ids = gen[i].tolist()[start_idx:]
                    txt = tokenizer.decode(resp_ids, skip_special_tokens=True)
                    responses[start + i] = txt

        # Apply variant stripping rule to the *training target*.
        train_targets: list[str] = []
        for mv, txt in zip(forced_moves, responses):
            stripped = _strip_boilerplate(full_response=txt, forced_move=mv, variant=variant)
            train_targets.append(_canonicalize_strict_response(text=stripped, forced_move=mv))

        # Generation can allocate large KV caches (especially with long prompts). Explicitly clear
        # cached blocks before the SFT forward to reduce OOM risk on large models.
        gc.collect()
        torch.cuda.empty_cache()

        # Score + compute weights (per-prompt group of size moves_per_prompt).
        scored = compute_score_batch(reward_models, train_targets, chess_reward_fn="winrate", logit_eps=1e-6)
        penalty = [str((s or {}).get("penalty_reason", "") or "") for s in scored]
        expected = np.asarray([float((s or {}).get("move_expected_score", float("nan"))) for s in scored], dtype=np.float64)
        best_expected = np.asarray([float((s or {}).get("best_expected_score", float("nan"))) for s in scored], dtype=np.float64)
        pred_move = [str((s or {}).get("pred_move", "") or "") for s in scored]

        weights = np.zeros(examples_per_rank, dtype=np.float64)
        valid = np.asarray([p == "" for p in penalty], dtype=bool)

        for i0 in range(0, examples_per_rank, moves_per_prompt):
            idxs = np.arange(i0, i0 + moves_per_prompt)
            if idxs[-1] >= examples_per_rank:
                break
            group_valid = valid[idxs]

            if args.sft_weighting == "uniform":
                weights[idxs[group_valid]] = 1.0
                continue

            if args.sft_weighting == "best_only":
                ok = group_valid & np.isfinite(expected[idxs]) & np.isfinite(best_expected[idxs])
                is_best = np.zeros_like(group_valid, dtype=bool)
                is_best[ok] = np.abs(expected[idxs][ok] - best_expected[idxs][ok]) <= 1e-12
                weights[idxs[is_best]] = 1.0
                continue

            if args.sft_weighting == "awr":
                ok = group_valid & np.isfinite(expected[idxs])
                if not np.any(ok):
                    continue
                r = expected[idxs][ok] / max(float(args.awr_beta), 1e-6)
                p_sm = _softmax(r)
                w = float(moves_per_prompt) * p_sm
                weights[idxs[ok]] = w
                continue

            raise ValueError(f"Unknown sft_weighting: {args.sft_weighting!r}")

        # Log one JSONL per-rank per-step (small and grep-friendly).
        log_path = logs_dir / f"rank{rank}_step{global_step:06d}.jsonl"
        with log_path.open("w", encoding="utf-8") as f:
            for i in range(examples_per_rank):
                f.write(
                    json.dumps(
                        {
                            "global_step": int(global_step),
                            "epoch_idx": int(epoch_idx),
                            "step_in_epoch": int(step_in_epoch),
                            "group_index": int(group_ids[i]),
                            "move_sample_idx": int(move_sample_idxs[i]),
                            "forced_move": forced_moves[i],
                            "pred_move": pred_move[i],
                            "penalty_reason": penalty[i],
                            "move_expected_score": float(expected[i]) if np.isfinite(expected[i]) else None,
                            "best_expected_score": float(best_expected[i]) if np.isfinite(best_expected[i]) else None,
                            "sample_weight": float(weights[i]),
                        }
                    )
                    + "\n"
                )

        # Small human-readable dump for quick manual inspection.
        if rank == 0 and int(args.debug_dump_examples or 0) > 0:
            n = int(min(int(args.debug_dump_examples), examples_per_rank))
            dump = []
            for i in range(n):
                dump.append(
                    {
                        "global_step": int(global_step),
                        "group_index": int(group_ids[i]),
                        "move_sample_idx": int(move_sample_idxs[i]),
                        "forced_move": forced_moves[i],
                        "raw_response": responses[i],
                        "train_target": train_targets[i],
                        "pred_move": pred_move[i],
                        "penalty_reason": penalty[i],
                        "move_expected_score": float(expected[i]) if np.isfinite(expected[i]) else None,
                        "best_expected_score": float(best_expected[i]) if np.isfinite(best_expected[i]) else None,
                        "sample_weight": float(weights[i]),
                    }
                )
            (out_dir / f"debug_examples_step{global_step:06d}.json").write_text(
                json.dumps(dump, indent=2), encoding="utf-8"
            )

        # SFT update: one optimizer step per outer loop.
        ddp.train()
        optimizer.zero_grad(set_to_none=True)

        # Build all tensors on CPU then move per micro-batch.
        pad_id = int(tokenizer.pad_token_id or 0)
        input_ids_all = []
        attn_all = []
        loss_mask_all = []
        for pm, resp in zip(prompt_msgs_list, train_targets):
            ids, am, lm = _build_train_tensors(
                tokenizer=tokenizer,
                prompt_msgs=pm,
                response_text=resp,
                max_length=int(args.train_max_length),
                pad_token_id=pad_id,
            )
            input_ids_all.append(ids)
            attn_all.append(am)
            loss_mask_all.append(lm)

        input_ids_all = torch.stack(input_ids_all, dim=0)
        attn_all = torch.stack(attn_all, dim=0)
        loss_mask_all = torch.stack(loss_mask_all, dim=0)
        sample_weight_all = torch.tensor(weights, dtype=torch.float32)

        micro = int(args.micro_batch_size)
        if micro <= 0:
            micro = examples_per_rank
        n_micro = int((examples_per_rank + micro - 1) // micro)
        step_loss = 0.0
        for mb in range(n_micro):
            s = mb * micro
            e = min(examples_per_rank, (mb + 1) * micro)
            with torch.autocast(device_type="cuda", dtype=torch_dtype):
                loss = _compute_weighted_token_mean_loss(
                    model=ddp,
                    input_ids=input_ids_all[s:e].to(device),
                    attention_mask=attn_all[s:e].to(device),
                    loss_mask=loss_mask_all[s:e].to(device),
                    sample_weight=sample_weight_all[s:e].to(device),
                )
                loss = loss / float(n_micro)
            loss.backward()
            step_loss += float(loss.detach().cpu().item())

        torch.nn.utils.clip_grad_norm_(ddp.parameters(), max_norm=float(args.grad_clip))
        optimizer.step()

        # Simple cross-rank logging.
        loss_t = torch.tensor(step_loss, device=device, dtype=torch.float32)
        dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
        valid_frac = float(np.mean(valid))
        valid_t = torch.tensor(valid_frac, device=device, dtype=torch.float32)
        dist.all_reduce(valid_t, op=dist.ReduceOp.AVG)

        if rank == 0:
            dt = time.time() - t0
            print(
                f"[online_sft] step={global_step+1}/{total_steps} "
                f"loss={float(loss_t.item()):.4f} valid_frac={float(valid_t.item()):.3f} "
                f"dt={dt:.1f}s"
            )

        dist.barrier()

    # Save final HF model for downstream vLLM pass@k evaluation.
    dist.barrier()
    if rank == 0:
        ckpt_dir = out_dir / "checkpoints" / f"global_step_{total_steps}" / "huggingface"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ddp.module.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        (out_dir / "checkpoints" / "latest_checkpointed_iteration.txt").write_text(str(total_steps), encoding="utf-8")
        print(f"[online_sft] Saved HF model to {ckpt_dir}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
