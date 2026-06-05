#!/usr/bin/env python3
"""Manual sanity checks for Pass@k analytic GRPO advantages.

This is intentionally print-based (research workflow), not a unit-test harness.
"""

import itertools
import math

import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, passk_advantages_max_subsets


def _make_token_rewards(rewards: list[float]) -> torch.Tensor:
    return torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)


def _make_response_mask(n: int) -> torch.Tensor:
    return torch.ones((n, 1), dtype=torch.float32)


def _make_single_group_index(n: int) -> np.ndarray:
    return np.array(["group0"] * n, dtype=object)


def _bruteforce_passk(rewards: list[float], k: int, eps: float = 1e-8) -> tuple[list[float], float, float]:
    n = len(rewards)
    subsets = list(itertools.combinations(range(n), k))
    group_max = [max(rewards[i] for i in subset) for subset in subsets]
    mu = float(np.mean(group_max))
    sig = float(np.std(group_max))

    if sig <= eps:
        return [0.0] * n, mu, sig

    e_i = []
    for i in range(n):
        cond = [group_max[j] for j, subset in enumerate(subsets) if i in subset]
        e_i.append(float(np.mean(cond)))

    adv = [(x - mu) / sig for x in e_i]
    return adv, mu, sig


def main() -> None:
    print("== Pass@k advantage manual checks ==")

    # 0) default-off behavior: config absent vs explicit False should match exactly.
    base_rewards = [-0.31, -0.44, -0.26, -0.71, -0.53, -0.12]
    base_token_rewards = _make_token_rewards(base_rewards)
    base_response_mask = _make_response_mask(len(base_rewards))
    base_index = _make_single_group_index(len(base_rewards))
    adv_default_off, _ = compute_grpo_outcome_advantage(
        token_level_rewards=base_token_rewards,
        response_mask=base_response_mask,
        index=base_index,
        config=None,
    )
    adv_explicit_off, _ = compute_grpo_outcome_advantage(
        token_level_rewards=base_token_rewards,
        response_mask=base_response_mask,
        index=base_index,
        config={"pass_k_training": False, "pass_k_k": 4},
    )
    diff_default_off = torch.max(torch.abs(adv_default_off.squeeze(-1) - adv_explicit_off.squeeze(-1))).item()
    print(f"[check default-off] max_abs_diff={diff_default_off:.10f}")
    if diff_default_off > 0.0:
        raise SystemExit("default-off behavior check failed")

    # 1) k=1 parity with the existing GRPO normalization path.
    rewards = [-0.35, -0.6, -0.25, -0.8, -0.42, -0.5]
    token_rewards = _make_token_rewards(rewards)
    response_mask = _make_response_mask(len(rewards))
    index = _make_single_group_index(len(rewards))
    adv_grpo, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=index,
        config={"pass_k_training": False},
    )
    adv_passk_k1, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=index,
        config={"pass_k_training": True, "pass_k_k": 1},
    )
    diff_k1 = torch.max(torch.abs(adv_grpo.squeeze(-1) - adv_passk_k1.squeeze(-1))).item()
    print(f"[check k=1 parity] max_abs_diff={diff_k1:.10f}")
    if diff_k1 > 2e-7:
        raise SystemExit("k=1 parity check failed")

    # 2) sigma <= eps behavior.
    flat_rewards = [-0.4, -0.4, -0.4, -0.4, -0.4, -0.4]
    adv_flat, mu_flat, sig_flat = passk_advantages_max_subsets(flat_rewards, k=4, eps=1e-8)
    print(f"[check sigma<=eps] mu={mu_flat:.6f}, sig={sig_flat:.6e}, adv={adv_flat}")
    if any(abs(x) > 1e-12 for x in adv_flat):
        raise SystemExit("sigma<=eps check failed: expected all-zero advantages")

    # 3) small-N brute-force subset agreement.
    brute_rewards = [-0.1, -0.55, -0.32, -0.8, -0.27, -0.43]
    k = 3
    adv_analytic, mu_analytic, sig_analytic = passk_advantages_max_subsets(brute_rewards, k=k, eps=1e-8)
    adv_bruteforce, mu_bruteforce, sig_bruteforce = _bruteforce_passk(brute_rewards, k=k, eps=1e-8)
    mu_diff = abs(mu_analytic - mu_bruteforce)
    sig_diff = abs(sig_analytic - sig_bruteforce)
    adv_diff = max(abs(a - b) for a, b in zip(adv_analytic, adv_bruteforce, strict=True))
    print(
        "[check brute-force] "
        f"mu_diff={mu_diff:.10e}, sig_diff={sig_diff:.10e}, adv_max_diff={adv_diff:.10e}"
    )
    if mu_diff > 1e-10 or sig_diff > 1e-10 or adv_diff > 1e-9:
        raise SystemExit("brute-force agreement check failed")

    # 4) stable mapping back to original rollout order.
    map_rewards = [-0.61, -0.23, -0.78, -0.39, -0.14, -0.52]
    map_k = 4
    adv_orig, _, _ = passk_advantages_max_subsets(map_rewards, k=map_k, eps=1e-8)
    perm = [2, 5, 0, 4, 1, 3]
    perm_rewards = [map_rewards[i] for i in perm]
    adv_perm, _, _ = passk_advantages_max_subsets(perm_rewards, k=map_k, eps=1e-8)
    adv_perm_back = [0.0] * len(map_rewards)
    for perm_idx, orig_idx in enumerate(perm):
        adv_perm_back[orig_idx] = adv_perm[perm_idx]
    map_diff = max(abs(a - b) for a, b in zip(adv_orig, adv_perm_back, strict=True))
    print(f"[check order-mapping] max_abs_diff={map_diff:.10e}")
    if map_diff > 1e-9:
        raise SystemExit("mapping-back check failed")

    print("All pass@k manual checks passed.")


if __name__ == "__main__":
    torch.set_printoptions(precision=8, sci_mode=True)
    np.set_printoptions(precision=8, suppress=False)
    main()
