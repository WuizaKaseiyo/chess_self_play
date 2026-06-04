from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Ensure local namespace packages (e.g., `recipe/`) resolve under pytest.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipe.chess.reward_fn import compute_score
from verl.utils.prompt import infer_use_chat_template_from_model_name, is_qwen3_base_model
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty


def test_reward_parsing_rejects_guess_prefix_and_requires_strict_contract() -> None:
    # Minimal reward payload (enough for compute_score to score a selection target by μ).
    rm = {
        "style": "rule",
        "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
        "ground_truth": "e2e4",
        "legal_moves_uci": ["e2e4", "d2d4"],
        # Use move_values_json as a μ-map fallback (0..1).
        "move_values_json": json.dumps({"e2e4": 0.9, "d2d4": 0.1}),
    }

    # Historical `<guess>` prefixes are outside the current strict selection contract.
    response = "<guess> d2d4 </guess>\n<reason>t</reason><uci_move>e2e4</uci_move>"
    res = compute_score(data_source=rm, solution_str=response, ground_truth="e2e4", extra_info=None)
    assert isinstance(res, dict)
    assert res["format_reward"] == 0.0
    assert res["penalty_reason"] == "format_error"
    assert res["score"] == -1.0

    # Missing `<uci_move>` should still be a format error (this is the failure mode we care about in training).
    bad = compute_score(
        data_source=rm,
        solution_str="<guess> e2e4 </guess>\n<reason>t</reason>",
        ground_truth="e2e4",
        extra_info=None,
    )
    assert isinstance(bad, dict)
    assert bad["format_reward"] == 0.0
    assert bad["penalty_reason"] == "format_error"
    assert bad["score"] == -1.0


def test_qwen3_base_model_names_default_to_plain_prompts() -> None:
    for model_name in (
        "Qwen/Qwen3-4B-Base",
        "/projects/a5l/ziyan/models/Qwen/Qwen3-1.7B-Base",
    ):
        assert is_qwen3_base_model(model_name)
        assert infer_use_chat_template_from_model_name(model_name, default=True) is False


def test_forced_token_mask_excludes_forced_tokens_from_pg_and_kl_losses() -> None:
    # Tiny, deterministic tensors:
    # - First 2 tokens are "forced" (should not contribute to policy gradient nor KL-to-ref loss).
    # - Last 2 tokens are "free" (should contribute if advantages/logprobs are non-zero there).
    response_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    forced_token_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    loss_mask = response_mask & ~forced_token_mask

    # Make ONLY the forced tokens carry signal.
    advantages = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    log_prob = torch.zeros((1, 4), dtype=torch.float32)
    old_log_prob = torch.zeros((1, 4), dtype=torch.float32)

    # PPO policy loss (vanilla) should be exactly 0 when masking out the forced tokens.
    actor_cfg = OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
        }
    )
    policy_loss_fn = get_policy_loss_fn("vanilla")
    pg_loss_full, _, _, _ = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=actor_cfg,
    )
    pg_loss_masked, _, _, _ = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=loss_mask,
        loss_agg_mode="token-mean",
        config=actor_cfg,
    )
    assert pg_loss_full.item() != 0.0
    assert pg_loss_masked.item() == 0.0

    # KL-to-ref loss: make ONLY forced tokens have KL signal (non-zero logprob vs ref).
    ref_log_prob = torch.zeros((1, 4), dtype=torch.float32)
    log_prob_for_kl = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    kld = kl_penalty(logprob=log_prob_for_kl, ref_logprob=ref_log_prob, kl_penalty="low_var_kl")
    kl_loss_full = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode="token-mean")
    kl_loss_masked = agg_loss(loss_mat=kld, loss_mask=loss_mask, loss_agg_mode="token-mean")

    assert kl_loss_full.item() > 0.0
    assert kl_loss_masked.item() == 0.0
