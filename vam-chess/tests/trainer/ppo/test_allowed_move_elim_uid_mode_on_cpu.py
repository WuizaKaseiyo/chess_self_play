# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections import defaultdict

import numpy as np

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _simulate(*, num_prompts: int, r_max: int, rollout_n: int, uid_mode: str):
    uid_mode = str(uid_mode).strip().lower()
    assert uid_mode in {"per_round", "per_prompt"}

    base_uids = [f"uid_prompt{p}" for p in range(num_prompts)] if uid_mode == "per_prompt" else []

    prompt_idx: list[int] = []
    round_idx: list[int] = []
    uids: list[str] = []
    for r in range(1, r_max + 1):
        for p in range(num_prompts):
            uid = f"uid_prompt{p}_round{r}" if uid_mode == "per_round" else base_uids[p]
            for _ in range(rollout_n):
                prompt_idx.append(p)
                round_idx.append(r)
                uids.append(uid)

    return (
        np.asarray(prompt_idx, dtype=np.int64),
        np.asarray(round_idx, dtype=np.int64),
        np.asarray(uids, dtype=object),
    )


def _total_weight_by_prompt(prompt_idx_arr: np.ndarray, denom_by_prompt: dict[int, int]) -> dict[int, float]:
    out: dict[int, float] = defaultdict(float)
    for pidx in prompt_idx_arr:
        p = int(pidx)
        out[p] += 1.0 / float(denom_by_prompt.get(p, 1))
    return out


def test_allowed_move_elim_uid_mode_per_round_counts():
    num_prompts = 3
    r_max = 4
    rollout_n = 5
    prompt_idx_arr, round_idx_arr, uid_arr = _simulate(
        num_prompts=num_prompts, r_max=r_max, rollout_n=rollout_n, uid_mode="per_round"
    )

    uid_counts = RayPPOTrainer._allowed_move_elim_count_unique_uids_by_prompt(
        prompt_idx_arr=prompt_idx_arr,
        uid_arr=uid_arr,
    )
    round_counts = RayPPOTrainer._allowed_move_elim_count_unique_rounds_by_prompt(
        prompt_idx_arr=prompt_idx_arr,
        round_arr=round_idx_arr,
    )

    assert all(uid_counts[i] == r_max for i in range(num_prompts))
    assert all(round_counts[i] == r_max for i in range(num_prompts))

    # Default loss denom uses uid group counts.
    totals = _total_weight_by_prompt(prompt_idx_arr, uid_counts)
    assert all(abs(totals[i] - float(rollout_n)) < 1e-6 for i in range(num_prompts))


def test_allowed_move_elim_uid_mode_per_prompt_counts():
    num_prompts = 3
    r_max = 4
    rollout_n = 5
    prompt_idx_arr, round_idx_arr, uid_arr = _simulate(
        num_prompts=num_prompts, r_max=r_max, rollout_n=rollout_n, uid_mode="per_prompt"
    )

    uid_counts = RayPPOTrainer._allowed_move_elim_count_unique_uids_by_prompt(
        prompt_idx_arr=prompt_idx_arr,
        uid_arr=uid_arr,
    )
    round_counts = RayPPOTrainer._allowed_move_elim_count_unique_rounds_by_prompt(
        prompt_idx_arr=prompt_idx_arr,
        round_arr=round_idx_arr,
    )

    assert all(uid_counts[i] == 1 for i in range(num_prompts))
    assert all(round_counts[i] == r_max for i in range(num_prompts))

    # New uid_mode still normalizes by rounds-per-prompt (so total weight stays ~rollout_n).
    totals = _total_weight_by_prompt(prompt_idx_arr, round_counts)
    assert all(abs(totals[i] - float(rollout_n)) < 1e-6 for i in range(num_prompts))
