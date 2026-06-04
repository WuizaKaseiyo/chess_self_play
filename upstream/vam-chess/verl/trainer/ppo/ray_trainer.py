# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from jinja2 import Environment
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean, postprocess_data
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.model import compute_position_id_with_mask
from verl.utils.prompt import as_bool, encode_prompt_from_messages, render_prompt_from_messages


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    def _get_grpo_group_index(d: DataProto) -> np.ndarray:
        """Return a GRPO grouping index.

        Grouping key: `uid` (one group per prompt).

        Forced-prefix injection may be applied to a subset of rollouts, but optimization
        always normalizes/centers within the full per-uid rollout group (no forced/free
        stratification).
        """
        uids = d.non_tensor_batch.get("uid", None)
        if uids is None:
            return np.arange(len(d), dtype=np.int64)
        uids = np.asarray(uids).reshape(-1)
        if uids.shape[0] != len(d):
            uids = np.resize(uids, (len(d),))
        return uids

    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        bsz = len(data)
        pred_move_arr = data.non_tensor_batch.get("pred_move", None)
        penalty_applied_arr = data.non_tensor_batch.get("penalty_applied", None)
        in_subset_arr = data.non_tensor_batch.get("in_subset", None)
        n_considered_moves_arr = data.non_tensor_batch.get("n_considered_moves", None)
        reward_model_arr = data.non_tensor_batch.get("reward_model", None)

        # Canonical answer IDs for diversity methods:
        # use predicted UCI when compliant; otherwise map to deterministic invalid sentinel.
        canonical_answer_ids = np.array(["__invalid__"] * bsz, dtype=object)
        sample_valid = np.zeros((bsz,), dtype=np.bool_)
        sample_candidate_sizes = np.ones((bsz,), dtype=np.int64)

        if pred_move_arr is not None:
            pred_move_arr = np.asarray(pred_move_arr).reshape(-1)
            if pred_move_arr.shape[0] != bsz:
                pred_move_arr = np.resize(pred_move_arr, (bsz,))
        else:
            pred_move_arr = np.array([""] * bsz, dtype=object)

        if penalty_applied_arr is not None:
            penalty_applied_arr = np.asarray(penalty_applied_arr).reshape(-1)
            if penalty_applied_arr.shape[0] != bsz:
                penalty_applied_arr = np.resize(penalty_applied_arr, (bsz,))
        else:
            penalty_applied_arr = np.ones((bsz,), dtype=np.bool_)

        if in_subset_arr is not None:
            in_subset_arr = np.asarray(in_subset_arr).reshape(-1)
            if in_subset_arr.shape[0] != bsz:
                in_subset_arr = np.resize(in_subset_arr, (bsz,))
        else:
            in_subset_arr = np.ones((bsz,), dtype=np.bool_)

        if n_considered_moves_arr is not None:
            n_considered_moves_arr = np.asarray(n_considered_moves_arr).reshape(-1)
            if n_considered_moves_arr.shape[0] != bsz:
                n_considered_moves_arr = np.resize(n_considered_moves_arr, (bsz,))

        if reward_model_arr is not None:
            reward_model_arr = np.asarray(reward_model_arr).reshape(-1)
            if reward_model_arr.shape[0] != bsz:
                reward_model_arr = np.resize(reward_model_arr, (bsz,))

        for i in range(bsz):
            penalty_applied = bool(penalty_applied_arr[i])
            in_subset = bool(in_subset_arr[i])
            pm = str(pred_move_arr[i] or "").strip().lower()

            if (not penalty_applied) and in_subset and pm:
                canonical_answer_ids[i] = pm
                sample_valid[i] = True
            else:
                canonical_answer_ids[i] = "__invalid__"
                sample_valid[i] = False

            n_considered = None
            if n_considered_moves_arr is not None:
                try:
                    n_considered = int(n_considered_moves_arr[i])
                except Exception:
                    n_considered = None
            if not n_considered or n_considered <= 0:
                rm = reward_model_arr[i] if reward_model_arr is not None else None
                if isinstance(rm, dict):
                    considered = rm.get("considered_moves_uci") or rm.get("considered_moves_uci_list")
                    legal = rm.get("legal_moves_uci")
                    if isinstance(considered, list) and considered:
                        n_considered = len(considered)
                    elif isinstance(legal, list) and legal:
                        n_considered = len(legal)
            if not n_considered or n_considered <= 0:
                n_considered = 1
            sample_candidate_sizes[i] = int(n_considered)

        sample_is_optimal = None
        if config is not None:
            allowed_move_elim_cfg = config.get("allowed_move_elim", None) or {}
            allowed_move_elim_enable = bool(allowed_move_elim_cfg.get("enable", False))
            need_sample_is_optimal = bool(
                allowed_move_elim_cfg.get("pass_k_when_no_optimal", False)
            ) or bool(allowed_move_elim_cfg.get("diversity_when_no_optimal", False))
            if allowed_move_elim_enable and need_sample_is_optimal:
                gt_uci_arr = data.non_tensor_batch.get("gt_uci", None)
                if pred_move_arr is not None and gt_uci_arr is not None and penalty_applied_arr is not None:
                    pred_move_arr = np.asarray(pred_move_arr).reshape(-1)
                    gt_uci_arr = np.asarray(gt_uci_arr).reshape(-1)
                    penalty_applied_arr = np.asarray(penalty_applied_arr).reshape(-1)
                    if pred_move_arr.shape[0] != len(data):
                        pred_move_arr = np.resize(pred_move_arr, (len(data),))
                    if gt_uci_arr.shape[0] != len(data):
                        gt_uci_arr = np.resize(gt_uci_arr, (len(data),))
                    if penalty_applied_arr.shape[0] != len(data):
                        penalty_applied_arr = np.resize(penalty_applied_arr, (len(data),))
                    if in_subset_arr is not None:
                        in_subset_arr = np.asarray(in_subset_arr).reshape(-1)
                        if in_subset_arr.shape[0] != len(data):
                            in_subset_arr = np.resize(in_subset_arr, (len(data),))

                    sample_is_optimal = np.zeros((len(data),), dtype=np.bool_)
                    for i in range(len(data)):
                        if bool(penalty_applied_arr[i]):
                            continue
                        pm = str(pred_move_arr[i] or "").strip().lower()
                        gt = str(gt_uci_arr[i] or "").strip().lower()
                        if not pm or not gt:
                            continue
                        if in_subset_arr is not None and (not bool(in_subset_arr[i])):
                            continue
                        if pm == gt:
                            sample_is_optimal[i] = True

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns, diversity_aux = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=_get_grpo_group_index(data),
            sample_is_optimal=sample_is_optimal,
            answer_ids=canonical_answer_ids,
            sample_valid=sample_valid,
            candidate_sizes=sample_candidate_sizes,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
            return_aux=True,
        )
        if diversity_aux:
            data.non_tensor_batch.update(diversity_aux)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_name = adv_estimator.value if hasattr(adv_estimator, "value") else str(adv_estimator)
            if str(adv_name).startswith("grpo"):
                adv_kwargs["index"] = _get_grpo_group_index(data)
            else:
                adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self._hf_config_hash: str | None = None
        self._hf_dataset_fingerprint: dict[str, Any] | None = None
        self._hf_upload_executor = None
        self._hf_upload_futures: list[Any] = []

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        self._dump_generations_to_file(
            inputs=inputs,
            outputs=outputs,
            gts=gts,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            filename=filename,
        )

    def _dump_generations_to_file(self, inputs, outputs, gts, scores, reward_extra_infos_dict, filename: str) -> None:
        """Dump rollout/validation samples as JSONL to an explicit filename."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")
        if "wandb" in self.config.trainer.logger:
            import wandb

            if wandb.run is not None:
                wandb.save(filename, policy="now")

    def _dump_filter_groups_rejected_group_summaries(
        self,
        batch: DataProto,
        rejected_prompt_uids: list[str],
        metric_name: str,
        filename: str,
        gen_batch_index: int,
    ) -> int:
        """Dump one JSONL record per rejected prompt-group (uid) for filter_groups diagnostics."""
        if not rejected_prompt_uids:
            return 0
        if "uid" not in batch.non_tensor_batch:
            return 0

        uids = batch.non_tensor_batch["uid"]
        uid2idxs: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            uid2idxs[str(uid)].append(idx)

        metric_arr = batch.non_tensor_batch.get(metric_name, None)
        pred_move_arr = batch.non_tensor_batch.get("pred_move", None)
        acc_arr = batch.non_tensor_batch.get("acc", None)
        penalty_reason_arr = batch.non_tensor_batch.get("penalty_reason", None)
        in_subset_arr = batch.non_tensor_batch.get("in_subset", None)
        format_reward_arr = batch.non_tensor_batch.get("format_reward", None)
        mu_pred_arr = batch.non_tensor_batch.get("mu_pred", None)
        mu_target_arr = batch.non_tensor_batch.get("mu_target", None)

        def _to_py(v: Any) -> Any:
            if isinstance(v, np.generic):
                return v.item()
            return v

        entries: list[str] = []
        num_groups_written = 0
        for uid in rejected_prompt_uids:
            idxs = uid2idxs.get(str(uid))
            if not idxs:
                continue

            # Prompt-level metadata is identical across the rollouts in a group; take the first.
            fen = None
            ground_truth = None
            legal_moves_uci = None
            considered_moves_uci = None
            row_id = None

            rm_arr = batch.non_tensor_batch.get("reward_model", None)
            if rm_arr is not None:
                rm = rm_arr[idxs[0]]
                if isinstance(rm, dict):
                    fen = rm.get("fen", None)
                    ground_truth = rm.get("ground_truth", None)
                    legal_moves_uci = rm.get("legal_moves_uci", None)
                    considered_moves_uci = rm.get("considered_moves_uci", None)

            extra_arr = batch.non_tensor_batch.get("extra_info", None)
            if extra_arr is not None:
                extra = extra_arr[idxs[0]]
                if isinstance(extra, dict):
                    row_id = extra.get("index", None)

            metric_vals = []
            if metric_arr is not None:
                metric_vals = [_to_py(metric_arr[i]) for i in idxs]

            pred_moves = []
            if pred_move_arr is not None:
                pred_moves = [_to_py(pred_move_arr[i]) for i in idxs]

            accs = []
            if acc_arr is not None:
                accs = [_to_py(acc_arr[i]) for i in idxs]

            penalty_reasons = []
            if penalty_reason_arr is not None:
                penalty_reasons = [_to_py(penalty_reason_arr[i]) for i in idxs]

            in_subset = []
            if in_subset_arr is not None:
                in_subset = [_to_py(in_subset_arr[i]) for i in idxs]

            format_rewards = []
            if format_reward_arr is not None:
                format_rewards = [_to_py(format_reward_arr[i]) for i in idxs]

            mu_preds = []
            if mu_pred_arr is not None:
                mu_preds = [_to_py(mu_pred_arr[i]) for i in idxs]

            mu_targets = []
            if mu_target_arr is not None:
                mu_targets = [_to_py(mu_target_arr[i]) for i in idxs]

            # Derived group-level indicators.
            metric_std = None
            try:
                metric_std = float(np.std(np.asarray(metric_vals, dtype=np.float32)))
            except Exception:
                metric_std = None

            pred_moves_str = [str(m) for m in pred_moves if m is not None]
            pred_move_unique = len(set(pred_moves_str)) if pred_moves_str else 0

            penalty_reasons_str = [str(p) for p in penalty_reasons if p is not None]
            all_valid = bool(penalty_reasons_str) and set(penalty_reasons_str) == {""}

            all_best_move = False
            all_suboptimal_move = False
            if all_valid and accs:
                accs_f = [float(a) for a in accs]
                all_best_move = all(a == 1.0 for a in accs_f)
                all_suboptimal_move = all(a == 0.0 for a in accs_f)

            entry = {
                "step": int(self.global_steps),
                "gen_batch_index": int(gen_batch_index),
                "uid": str(uid),
                "row_id": row_id,
                "metric_name": str(metric_name),
                "metric_vals": metric_vals,
                "metric_std": metric_std,
                "pred_moves": pred_moves,
                "pred_move_unique": int(pred_move_unique),
                "accs": accs,
                "penalty_reasons": penalty_reasons,
                "in_subset": in_subset,
                "format_rewards": format_rewards,
                "mu_preds": mu_preds,
                "mu_targets": mu_targets,
                "all_valid": bool(all_valid),
                "all_best_move": bool(all_best_move),
                "all_suboptimal_move": bool(all_suboptimal_move),
                "fen": fen,
                "ground_truth": ground_truth,
                "legal_moves_uci": legal_moves_uci,
                "considered_moves_uci": considered_moves_uci,
            }
            entries.append(json.dumps(entry, ensure_ascii=False))
            num_groups_written += 1

        if not entries:
            return 0

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(entries) + "\n")
        print(f"Dumped filter_groups rejected-group summaries to {filename}")
        if "wandb" in self.config.trainer.logger:
            import wandb

            if wandb.run is not None:
                wandb.save(filename, policy="now")

        return num_groups_written

    def _dump_filter_groups_rejected_rollout_samples(
        self,
        batch: DataProto,
        rejected_prompt_uids: list[str],
        metric_name: str,
        filename: str,
        gen_batch_index: int,
    ) -> int:
        """Dump rollout-level JSONL samples (prompt+output) for a subset of rejected prompt-groups."""
        if not rejected_prompt_uids:
            return 0
        if "uid" not in batch.non_tensor_batch:
            return 0

        uid_set = set(str(u) for u in rejected_prompt_uids)
        traj_idxs = [i for i, uid in enumerate(batch.non_tensor_batch["uid"]) if str(uid) in uid_set]
        if not traj_idxs:
            return 0
        sample_batch = batch[traj_idxs]

        inputs = self.tokenizer.batch_decode(sample_batch.batch["prompts"], skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(sample_batch.batch["responses"], skip_special_tokens=True)
        scores = sample_batch.batch["token_level_scores"].sum(-1).detach().cpu().tolist()
        sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in sample_batch]

        # Keep this allowlist small; dumping raw `reward_model` for each rollout is redundant
        # and can bloat JSONLs significantly.
        extra_keys = [
            "uid",
            str(metric_name),
            "score",
            "acc",
            "pred_move",
            "target_move",
            "gt_uci",
            "mu_pred",
            "mu_target",
            "in_subset",
            "format_reward",
            "penalty",
            "penalty_reason",
            "penalty_applied",
            "reward_reason",
        ]
        reward_extra_infos_to_dump: dict[str, list[Any]] = {}
        for k in extra_keys:
            if k in sample_batch.non_tensor_batch:
                v = sample_batch.non_tensor_batch[k]
                try:
                    reward_extra_infos_to_dump[k] = v.tolist()
                except Exception:
                    # Fallback for non-ndarray values
                    reward_extra_infos_to_dump[k] = list(v)

        n = len(inputs)
        reward_extra_infos_to_dump.setdefault("filter_groups/rejected", [True] * n)
        reward_extra_infos_to_dump.setdefault("filter_groups/metric_name", [str(metric_name)] * n)
        reward_extra_infos_to_dump.setdefault("filter_groups/gen_batch_index", [int(gen_batch_index)] * n)

        self._dump_generations_to_file(
            inputs=inputs,
            outputs=outputs,
            gts=sample_gts,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_to_dump,
            filename=filename,
        )

        # Count prompt-groups (uids), not trajectories.
        unique_uids = set(str(u) for u in sample_batch.non_tensor_batch["uid"].tolist())
        return len(unique_uids)

    def _compute_grpo_effective_batch(self, batch: DataProto) -> tuple[int, int]:
        """Return (effective_groups, total_groups) for GRPO based on reward std != 0."""
        uids = batch.non_tensor_batch.get("uid", None)
        if uids is None or "token_level_rewards" not in batch.batch:
            return 0, 0
        seq_rewards = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
        uid2vals: dict[str, list[float]] = defaultdict(list)
        for uid, reward_val in zip(uids, seq_rewards, strict=True):
            uid2vals[str(uid)].append(float(reward_val))
        total = len(uid2vals)
        effective = 0
        for vals in uid2vals.values():
            if np.std(np.asarray(vals, dtype=np.float32)) > 0:
                effective += 1
        return effective, total

    def _compute_diversity_metrics(self, batch: DataProto) -> dict[str, Any]:
        required_keys = {
            "diversity_group_T",
            "diversity_group_top_freq",
            "diversity_group_collision_count",
            "diversity_a_base",
            "diversity_a_div",
            "uid",
        }
        if not required_keys.issubset(set(batch.non_tensor_batch.keys())):
            return {}

        uids = np.asarray(batch.non_tensor_batch["uid"]).reshape(-1)
        group_t = np.asarray(batch.non_tensor_batch["diversity_group_T"], dtype=np.float32).reshape(-1)
        group_top = np.asarray(batch.non_tensor_batch["diversity_group_top_freq"], dtype=np.float32).reshape(-1)
        group_collision = np.asarray(batch.non_tensor_batch["diversity_group_collision_count"], dtype=np.float32).reshape(-1)
        base_vals = np.asarray(batch.non_tensor_batch["diversity_a_base"], dtype=np.float32).reshape(-1)
        div_vals = np.asarray(batch.non_tensor_batch["diversity_a_div"], dtype=np.float32).reshape(-1)

        if len(uids) == 0:
            return {}

        # Ignore padded trajectories (no valid response tokens).
        keep_mask = np.ones((len(uids),), dtype=np.bool_)
        if "response_mask" in batch.batch:
            with torch.no_grad():
                resp_len = batch.batch["response_mask"].sum(dim=-1).detach().cpu().numpy()
            keep_mask = np.asarray(resp_len > 0, dtype=np.bool_)

        uid2idxs: dict[str, list[int]] = defaultdict(list)
        for i, uid in enumerate(uids):
            if not bool(keep_mask[i]):
                continue
            uid2idxs[str(uid)].append(i)
        if not uid2idxs:
            return {}

        group_t_vals: list[float] = []
        group_top_vals: list[float] = []
        group_collision_vals: list[float] = []
        group_base_mean_vals: list[float] = []
        group_base_std_vals: list[float] = []
        group_div_mean_vals: list[float] = []
        group_div_std_vals: list[float] = []

        for idxs in uid2idxs.values():
            i0 = idxs[0]
            group_t_vals.append(float(group_t[i0]))
            group_top_vals.append(float(group_top[i0]))
            group_collision_vals.append(float(group_collision[i0]))
            base_group = base_vals[idxs]
            div_group = div_vals[idxs]
            group_base_mean_vals.append(float(np.mean(base_group)))
            group_base_std_vals.append(float(np.std(base_group, ddof=1)) if len(base_group) > 1 else 0.0)
            group_div_mean_vals.append(float(np.mean(div_group)))
            group_div_std_vals.append(float(np.std(div_group, ddof=1)) if len(div_group) > 1 else 0.0)

        kept_indices = [i for i, keep in enumerate(keep_mask.tolist()) if keep]
        kept_base_vals = base_vals[kept_indices]
        kept_div_vals = div_vals[kept_indices]

        metrics: dict[str, Any] = {
            "diversity/group_count": int(len(uid2idxs)),
            "diversity/group_T_mean": float(np.mean(group_t_vals)),
            "diversity/group_top_freq_mean": float(np.mean(group_top_vals)),
            "diversity/group_collision_mean": float(np.mean(group_collision_vals)),
            "diversity/group_A_base_mean_mean": float(np.mean(group_base_mean_vals)),
            "diversity/group_A_base_std_mean": float(np.mean(group_base_std_vals)),
            "diversity/group_A_div_mean_mean": float(np.mean(group_div_mean_vals)),
            "diversity/group_A_div_std_mean": float(np.mean(group_div_std_vals)),
            "diversity/A_base_mean": float(np.mean(kept_base_vals)),
            "diversity/A_base_std": float(np.std(kept_base_vals, ddof=1)) if len(kept_base_vals) > 1 else 0.0,
            "diversity/A_div_mean": float(np.mean(kept_div_vals)),
            "diversity/A_div_std": float(np.std(kept_div_vals, ddof=1)) if len(kept_div_vals) > 1 else 0.0,
        }

        if "diversity_lambda_coeff" in batch.non_tensor_batch:
            lam_arr = np.asarray(batch.non_tensor_batch["diversity_lambda_coeff"], dtype=np.float32).reshape(-1)
            if lam_arr.size > 0:
                metrics["diversity/lambda_coeff"] = float(lam_arr[0])
        if "diversity_include_base_advantage" in batch.non_tensor_batch:
            include_arr = np.asarray(batch.non_tensor_batch["diversity_include_base_advantage"]).reshape(-1)
            if include_arr.size > 0:
                metrics["diversity/include_base_advantage"] = float(bool(include_arr[0]))
        if "diversity_enabled" in batch.non_tensor_batch:
            enabled_arr = np.asarray(batch.non_tensor_batch["diversity_enabled"]).reshape(-1)
            if enabled_arr.size > 0:
                metrics["diversity/enabled"] = float(bool(enabled_arr[0]))

        return metrics

    @staticmethod
    def _allowed_move_elim_count_unique_uids_by_prompt(
        *, prompt_idx_arr: np.ndarray, uid_arr: np.ndarray
    ) -> dict[int, int]:
        prompt_to_uids: dict[int, set[str]] = defaultdict(set)
        for uid, pidx in zip(uid_arr, prompt_idx_arr, strict=True):
            prompt_to_uids[int(pidx)].add(str(uid))
        return {k: max(1, len(v)) for k, v in prompt_to_uids.items()}

    @staticmethod
    def _allowed_move_elim_count_unique_rounds_by_prompt(
        *, prompt_idx_arr: np.ndarray, round_arr: np.ndarray
    ) -> dict[int, int]:
        prompt_to_rounds: dict[int, set[int]] = defaultdict(set)
        for r, pidx in zip(round_arr, prompt_idx_arr, strict=True):
            prompt_to_rounds[int(pidx)].add(int(r))
        return {k: max(1, len(v)) for k, v in prompt_to_rounds.items()}

    def _maybe_save_config_hash_json(self) -> None:
        """Write a small JSON file containing the deterministic config hash and upload it to W&B.

        This makes it easy to recover the exact `config_hash` (used for HF resume/upload) given only
        a W&B run ID, without relying on local filesystem paths.
        """
        if "wandb" not in self.config.trainer.logger:
            return
        try:
            import wandb
        except Exception:
            return
        if wandb.run is None:
            return

        config_hash = self._get_or_compute_config_hash()
        dataset_fp = self._compute_dataset_fingerprint().get("combined_sha256")

        # Derive an output root from the checkpoint dir (expected layout: <run_root>/checkpoints).
        from pathlib import Path

        ckpt_dir = str(self.config.trainer.default_local_dir)
        ckpt_path = Path(ckpt_dir)
        if not ckpt_path.is_absolute():
            ckpt_path = (Path.cwd() / ckpt_path).resolve()
        run_root = ckpt_path.parent
        run_root.mkdir(parents=True, exist_ok=True)

        payload = {
            "config_hash": config_hash,
            "dataset_fingerprint": dataset_fp,
            "paths": {
                "run_root": str(run_root),
                "default_local_dir": str(ckpt_path),
                "rollout_data_dir": str(self.config.trainer.get("rollout_data_dir", "")),
                "validation_data_dir": str(self.config.trainer.get("validation_data_dir", "")),
            },
        }

        out_path = run_root / "config_hash.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.write("\n")

        # Also surface it in the run summary for quick filtering/debugging.
        try:
            wandb.summary["config_hash"] = config_hash
        except Exception:
            pass

        wandb.save(str(out_path), base_path=str(run_root), policy="now")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            # Add forced-prefix diagnostics so rollout JSONLs can be used to sanity-check
            # per-rollout forced vs free behavior (mixed groups are expected).
            if "uid" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault("uid", batch.non_tensor_batch["uid"].tolist())
            if "forced_prefix_is_forced" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "forced_prefix_is_forced", batch.non_tensor_batch["forced_prefix_is_forced"].tolist()
                )
            elif "forced_token_mask" in batch.batch:
                forced_seq = torch.any(batch.batch["forced_token_mask"].to(torch.bool), dim=-1)
                reward_extra_infos_to_dump.setdefault("forced_prefix_is_forced", forced_seq.cpu().tolist())
            for k in ("forced_prefix_move", "forced_prefix_value"):
                if k in batch.non_tensor_batch:
                    reward_extra_infos_to_dump.setdefault(k, batch.non_tensor_batch[k].tolist())
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault("request_id", batch.non_tensor_batch["request_id"].tolist())

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    @staticmethod
    def _normalize_uci_moves(moves: Any) -> list[str]:
        if moves is None:
            return []
        if isinstance(moves, str):
            s = moves.strip().lower()
            return [s] if s else []
        out: list[str] = []
        try:
            for m in moves:
                s = str(m).strip().lower()
                if s:
                    out.append(s)
        except Exception:
            return []
        return out

    @staticmethod
    def _compute_allowed_move_elim_r_max(
        *, step: int, total_steps: int, r_start: int, r_end: int, anneal_frac: float
    ) -> int:
        step = max(int(step), 1)
        total_steps = max(int(total_steps), 1)
        r_start = int(r_start)
        r_end = int(r_end)
        anneal_frac = float(anneal_frac)
        if anneal_frac <= 0:
            return max(1, r_end)
        anneal_steps = max(1, int(np.ceil(total_steps * anneal_frac)))
        if anneal_steps <= 1:
            return max(1, r_end)
        progress = min(1.0, max(0.0, float(step - 1) / float(anneal_steps - 1)))
        value = float(r_start) + float(r_end - r_start) * progress
        if r_end <= r_start:
            r_max = int(np.floor(value + 1e-8))
        else:
            r_max = int(np.ceil(value - 1e-8))
        r_max = max(min(r_max, max(r_start, r_end)), min(r_start, r_end))
        return max(1, int(r_max))

    def _get_allowed_move_elim_template(self, template_path: str):
        template_path = str(template_path or "").strip()
        if not template_path:
            raise ValueError("allowed_move_elim.template_path is empty")
        cache_path = getattr(self, "_allowed_move_elim_template_path", None)
        if getattr(self, "_allowed_move_elim_template", None) is None or cache_path != template_path:
            path = Path(template_path)
            if not path.exists():
                raise FileNotFoundError(f"Selection template not found: {template_path}")
            env = Environment(autoescape=False)
            template = env.from_string(path.read_text(encoding="utf-8"))
            self._allowed_move_elim_template = template
            self._allowed_move_elim_template_path = template_path
        return self._allowed_move_elim_template

    def _build_allowed_move_elim_batch(
        self,
        *,
        base_batch: DataProto,
        indices: list[int],
        allowed_moves: list[list[str]],
        legal_moves: list[list[str]],
        template,
    ) -> DataProto:
        if not indices:
            raise ValueError("indices is empty in _build_allowed_move_elim_batch")
        if len(indices) != len(allowed_moves):
            raise ValueError("indices and allowed_moves length mismatch")

        subset = base_batch[indices]
        apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}) or {})
        use_chat_template = as_bool(self.config.data.get("use_chat_template", True), default=True)
        max_prompt_length = int(self.config.data.max_prompt_length)
        truncation = str(self.config.data.truncation)
        pad_token_id = int(self.tokenizer.pad_token_id)
        return_raw_chat = bool(self.config.data.get("return_raw_chat", False)) or (
            "raw_prompt" in subset.non_tensor_batch
        )

        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        raw_prompt_ids_list = []
        raw_prompt_list = []
        reward_models = []

        for idx, allowed in zip(indices, allowed_moves, strict=True):
            base_rm = base_batch.non_tensor_batch.get("reward_model", [None])[idx]
            rm = dict(base_rm) if isinstance(base_rm, dict) else {}

            legal = legal_moves[idx] if idx < len(legal_moves) else self._normalize_uci_moves(rm.get("legal_moves_uci"))
            if not legal:
                raise ValueError("Empty legal_moves_uci encountered during allowed-move elimination.")
            rm["legal_moves_uci"] = legal
            rm["considered_moves_uci"] = list(allowed)
            reward_models.append(rm)

            fen = str(rm.get("fen") or "").strip()
            if not fen:
                raise ValueError("Empty FEN encountered during allowed-move elimination.")

            prompt_text = str(
                template.render(
                    FEN=fen,
                    legal_moves_uci_list=legal,
                    considered_moves_uci_list=allowed,
                )
            )
            messages = [{"role": "user", "content": prompt_text}]
            raw_prompt = render_prompt_from_messages(
                self.tokenizer,
                messages,
                use_chat_template=use_chat_template,
                add_generation_prompt=True,
                apply_chat_template_kwargs=apply_kwargs,
            )

            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]

            input_ids, attention_mask = postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_prompt_length,
                pad_token_id=pad_token_id,
                left_pad=True,
                truncation=truncation,
            )

            input_ids = input_ids[0]
            attention_mask = attention_mask[0]
            position_ids = compute_position_id_with_mask(attention_mask)

            raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > max_prompt_length:
                if truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]
                elif truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
                elif truncation == "middle":
                    left_half = max_prompt_length // 2
                    right_half = max_prompt_length - left_half
                    raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                elif truncation == "error":
                    raise RuntimeError(
                        f"Prompt length {len(raw_prompt_ids)} is longer than {max_prompt_length}."
                    )

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
            raw_prompt_ids_list.append(raw_prompt_ids)
            if return_raw_chat:
                raw_prompt_list.append(messages)

        subset.batch["input_ids"] = torch.stack(input_ids_list, dim=0)
        subset.batch["attention_mask"] = torch.stack(attention_mask_list, dim=0)
        subset.batch["position_ids"] = torch.stack(position_ids_list, dim=0)
        subset.non_tensor_batch["raw_prompt_ids"] = np.array(raw_prompt_ids_list, dtype=object)
        subset.non_tensor_batch["reward_model"] = np.array(reward_models, dtype=object)
        if return_raw_chat:
            subset.non_tensor_batch["raw_prompt"] = np.array(raw_prompt_list, dtype=object)
        return subset

    @staticmethod
    def _force_use_considered_moves_uci(batch: DataProto) -> None:
        """Force reward_fn subset gating via extra_info override.

        This keeps subset semantics correct even when prompt wording omits the
        literal `allowed_moves` token (e.g., small-legal templates).
        """
        extras = batch.non_tensor_batch.get("extra_info", None)
        n_rows = len(batch)
        updated_extras: list[dict[str, Any]] = []

        if extras is None:
            updated_extras = [{"use_considered_moves_uci": True} for _ in range(n_rows)]
        else:
            extra_len = len(extras)
            for i in range(n_rows):
                extra_i = extras[i] if i < extra_len else {}
                if not isinstance(extra_i, dict):
                    extra_i = {}
                else:
                    extra_i = dict(extra_i)
                extra_i["use_considered_moves_uci"] = True
                updated_extras.append(extra_i)

        batch.non_tensor_batch["extra_info"] = np.array(updated_extras, dtype=object)

    @staticmethod
    def _stitch_allowed_move_elim_round0_prompt_context(
        *,
        round_batch: DataProto,
        round0_prompt_input_ids: torch.Tensor,
        round0_prompt_attention_mask: torch.Tensor,
        round0_prompt_position_ids: torch.Tensor,
    ) -> None:
        """Replace round prompts with per-prompt round-0 (full-legal) prompt context.

        Responses remain unchanged. This affects policy/reference log-prob
        context used by optimization while preserving reward parsing metadata
        already computed from per-round prompts.
        """
        prompt_idx_arr = round_batch.non_tensor_batch.get("allowed_move_elim_prompt_idx", None)
        if prompt_idx_arr is None:
            raise ValueError("allowed_move_elim missing allowed_move_elim_prompt_idx for round-0 prompt stitching.")

        prompt_idx_np = np.asarray(prompt_idx_arr, dtype=np.int64).reshape(-1)
        if prompt_idx_np.shape[0] != len(round_batch):
            raise ValueError(
                "allowed_move_elim round-0 prompt stitching got mismatched prompt_idx length "
                f"(prompt_idx={prompt_idx_np.shape[0]}, batch={len(round_batch)})."
            )
        if prompt_idx_np.size == 0:
            return

        max_prompt_idx = int(round0_prompt_input_ids.shape[0]) - 1
        if int(prompt_idx_np.min()) < 0 or int(prompt_idx_np.max()) > max_prompt_idx:
            raise ValueError(
                "allowed_move_elim round-0 prompt stitching found out-of-range prompt indices "
                f"(min={int(prompt_idx_np.min())}, max={int(prompt_idx_np.max())}, allowed_max={max_prompt_idx})."
            )

        index = torch.from_numpy(prompt_idx_np).to(device=round0_prompt_input_ids.device, dtype=torch.long)
        stitched_prompts = round0_prompt_input_ids.index_select(0, index)
        stitched_prompt_attention = round0_prompt_attention_mask.index_select(0, index)
        stitched_prompt_position = round0_prompt_position_ids.index_select(0, index)

        responses = round_batch.batch["responses"]
        target_device = responses.device

        stitched_prompts = stitched_prompts.to(device=target_device, dtype=round_batch.batch["prompts"].dtype)
        stitched_prompt_attention = stitched_prompt_attention.to(
            device=target_device, dtype=round_batch.batch["attention_mask"].dtype
        )
        stitched_prompt_position = stitched_prompt_position.to(
            device=target_device, dtype=round_batch.batch["position_ids"].dtype
        )

        response_mask = round_batch.batch.get("response_mask", None)
        if response_mask is None:
            response_mask = compute_response_mask(round_batch)
        response_mask = response_mask.to(device=target_device, dtype=stitched_prompt_attention.dtype)

        response_length = responses.size(1)
        delta = torch.arange(1, response_length + 1, device=target_device, dtype=stitched_prompt_position.dtype)
        if stitched_prompt_position.dim() == 3:  # qwen2vl mrope
            delta = delta.view(1, 1, -1).expand(
                stitched_prompt_position.size(0), stitched_prompt_position.size(1), -1
            )
        else:
            delta = delta.unsqueeze(0).expand(stitched_prompt_position.size(0), -1)
        stitched_response_position = stitched_prompt_position[..., -1:] + delta

        round_batch.batch["prompts"] = stitched_prompts
        round_batch.batch["input_ids"] = torch.cat((stitched_prompts, responses), dim=-1)
        round_batch.batch["attention_mask"] = torch.cat((stitched_prompt_attention, response_mask), dim=-1)
        round_batch.batch["position_ids"] = torch.cat((stitched_prompt_position, stitched_response_position), dim=-1)
        round_batch.batch["response_mask"] = response_mask

    @staticmethod
    def _clone_allowed_move_elim_logprob_batch(batch: DataProto) -> DataProto:
        required_keys = ["prompts", "responses", "input_ids", "attention_mask", "position_ids"]
        missing_keys = [k for k in required_keys if k not in batch.batch.keys()]
        if missing_keys:
            raise ValueError(
                "allowed_move_elim gain filter missing required tensor keys for logprob scoring: "
                f"{missing_keys}"
            )

        clone_tensors = {k: batch.batch[k].clone() for k in required_keys}
        if "response_mask" in batch.batch.keys():
            clone_tensors["response_mask"] = batch.batch["response_mask"].clone()

        clone_non_tensors: dict[str, np.ndarray] = {}
        if "allowed_move_elim_prompt_idx" in batch.non_tensor_batch:
            clone_non_tensors["allowed_move_elim_prompt_idx"] = np.asarray(
                batch.non_tensor_batch["allowed_move_elim_prompt_idx"]
            ).copy()

        return DataProto.from_dict(
            tensors=clone_tensors,
            non_tensors=clone_non_tensors,
            meta_info=deepcopy(batch.meta_info),
        )

    @staticmethod
    def _sum_masked_token_log_probs(*, token_log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        if token_log_probs.shape != response_mask.shape:
            raise ValueError(
                "Expected token log probs and response mask to have identical shape, got "
                f"{tuple(token_log_probs.shape)} vs {tuple(response_mask.shape)}"
            )
        mask = response_mask.to(dtype=token_log_probs.dtype, device=token_log_probs.device)
        return (token_log_probs.to(torch.float32) * mask).sum(dim=-1)

    def _self_play_cfg(self) -> dict[str, Any]:
        cfg_raw = getattr(self.config.data, "self_play", None)
        if cfg_raw is None:
            return {}
        if isinstance(cfg_raw, dict):
            return cfg_raw
        try:
            cfg = OmegaConf.to_container(cfg_raw, resolve=True)
        except Exception:
            return {}
        return cfg if isinstance(cfg, dict) else {}

    def _self_play_enabled(self) -> bool:
        cfg = self._self_play_cfg()
        return bool(cfg.get("enable", False))

    def _get_self_play_engine(self):
        import chess.engine

        cfg = self._self_play_cfg()
        engine_path = str(cfg.get("stockfish_path", ".third_party_cache/stockfish/src/stockfish"))
        threads = int(cfg.get("stockfish_threads", 1) or 1)
        hash_mb = int(cfg.get("stockfish_hash_mb", 128) or 128)
        skill = int(cfg.get("stockfish_skill_level", 0) or 0)

        key = (engine_path, threads, hash_mb, skill)
        engine = getattr(self, "_self_play_engine", None)
        if engine is not None and getattr(self, "_self_play_engine_key", None) == key:
            return engine

        # Recreate on config changes.
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass

        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        opts: dict[str, Any] = {
            "Threads": int(threads),
            "Hash": int(hash_mb),
            "Skill Level": int(skill),
            # Prefer WDL so we can populate `move_expected_scores_json` for online/self-play rows
            # (offline datasets include it and some reward modes require it).
            "UCI_ShowWDL": True,
        }
        try:
            engine.configure(opts)
        except Exception:
            # Some builds/images may not support all options; best-effort only.
            pass

        self._self_play_engine = engine
        self._self_play_engine_key = key
        return engine

    def _self_play_decode_and_validate_move(
        self,
        *,
        output_text: str,
        legal_moves_uci: list[str],
    ) -> tuple[Optional[str], str]:
        """Parse and validate a model move (starter-kit style; strict UCI within legal list)."""
        import chess

        from recipe.chess.reward_fn import UCI_MOVE_ONLY_RE

        s = output_text or ""
        m = UCI_MOVE_ONLY_RE.search(s)
        if not m:
            return None, "format_missing"

        raw = (m.group("ans") or "").strip()
        if not raw:
            return None, "empty_uci_move"

        move_str = raw.strip()
        if move_str.lower() == "resign":
            # Match starter-kit semantics: treat as invalid and retry until attempts exhausted.
            return None, "resign"

        legal_set = set(legal_moves_uci)
        if move_str not in legal_set:
            return None, "illegal_move"

        try:
            _ = chess.Move.from_uci(move_str)
        except Exception:
            return None, "bad_move"

        return move_str, ""

    def _self_play_generate(
        self,
        *,
        prompts: list[list[dict[str, str]]],
        sampling_kwargs: dict[str, Any],
    ) -> list[str]:
        """Generate one decoded response per prompt using the in-training rollout engine."""
        if self.async_rollout_mode:
            raise NotImplementedError("self_play is only supported in sync rollout mode for now.")

        apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}) or {})
        use_chat_template = as_bool(self.config.data.get("use_chat_template", True), default=True)
        max_prompt_length = int(self.config.data.max_prompt_length)
        truncation = str(self.config.data.truncation)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has no pad_token_id or eos_token_id; cannot build prompts.")

        raw_prompt_ids: list[list[int]] = []
        for messages in prompts:
            _, ids = encode_prompt_from_messages(
                self.tokenizer,
                messages,
                use_chat_template=use_chat_template,
                add_generation_prompt=True,
                apply_chat_template_kwargs=apply_kwargs,
            )
            if len(ids) > max_prompt_length:
                if truncation == "left":
                    ids = ids[-max_prompt_length:]
                elif truncation == "right":
                    ids = ids[:max_prompt_length]
                elif truncation == "middle":
                    left_half = max_prompt_length // 2
                    right_half = max_prompt_length - left_half
                    ids = ids[:left_half] + ids[-right_half:]
                elif truncation == "error":
                    raise RuntimeError(
                        f"Prompt length {len(ids)} is longer than max_prompt_length={max_prompt_length}."
                    )
            raw_prompt_ids.append([int(x) for x in ids])

        if not raw_prompt_ids:
            return []

        max_len = max(len(x) for x in raw_prompt_ids)
        bsz = len(raw_prompt_ids)

        input_ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        for i, ids in enumerate(raw_prompt_ids):
            start = max_len - len(ids)
            input_ids[i, start:] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, start:] = 1

        position_ids = compute_position_id_with_mask(attention_mask)

        gen_batch = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            non_tensors={},
            meta_info={
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": int(pad_id),
                "recompute_log_prob": False,
                "do_sample": True,
                # Treat self-play generation as validation-like by default (no forced-prefix, stable sampling).
                "validate": True,
                "sampling_kwargs": dict(sampling_kwargs or {}),
                "global_steps": int(self.global_steps),
            },
        )

        size_divisor = self.actor_rollout_wg.world_size
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
        out_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)

        output_ids = out.batch["responses"]
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

    def _build_self_play_train_batch(self) -> DataProto:
        """Generate a fresh batch of chess positions online (model vs Stockfish depth=1).

        Semantics:
        - Maintain a persistent pool of `B = data.train_batch_size` parallel games across training steps.
        - On each trainer step:
          1) Record exactly one position per game (both model-to-move and engine-to-move).
          2) Advance each game by exactly one ply:
             - model plays when it's the model's turn,
             - Stockfish plays when it's the opponent's turn.
          3) Restart any ended/forfeited games immediately, toggling which side the model plays for
             that slot (to keep long-run "who moves first" ~50/50 even if game lengths differ).
        - Games are never truncated to a fixed length; they continue across steps until naturally ending
          (mate/stalemate/insufficient material/...) or being forfeited due to invalid model output / engine errors.
        """

        import chess
        import chess.engine

        from recipe.chess.stockfish_scoring import dumps_compact_sorted, score_position_all_legal_moves

        cfg = self._self_play_cfg()

        opponent_depth = int(cfg.get("opponent_depth", 1) or 1)
        analysis_depth = int(cfg.get("analysis_depth", opponent_depth) or opponent_depth)
        max_retries_per_turn = int(cfg.get("max_retries_per_turn", 3) or 3)
        template_path = str(cfg.get("template_path", "recipe/chess/prompt_templates/select_prompt.jinja"))
        sampling_kwargs = cfg.get("sampling_kwargs", {}) or {}
        sp_timing: dict[str, float] = defaultdict(float)

        train_bsz = int(self.config.data.train_batch_size)
        if train_bsz <= 0:
            raise ValueError("data.train_batch_size must be > 0 for self_play")

        # New semantics: tie the persistent pool size to the train batch size.
        num_parallel_games = int(train_bsz)

        # Back-compat / config hygiene: older configs may still set `data.self_play.num_parallel_games`
        # (or `expected_length`) but these are now ignored in favor of `train_batch_size`.
        cfg_num_parallel_games_raw = cfg.get("num_parallel_games", cfg.get("expected_length", None))
        if cfg_num_parallel_games_raw is not None:
            try:
                cfg_num_parallel_games = int(cfg_num_parallel_games_raw)
            except Exception:
                cfg_num_parallel_games = None
            if cfg_num_parallel_games is not None and cfg_num_parallel_games != num_parallel_games:
                print(
                    f"[SELF_PLAY] ignoring data.self_play.num_parallel_games={cfg_num_parallel_games} "
                    f"(pool size is tied to data.train_batch_size={num_parallel_games})",
                    flush=True,
                )

        # Fixed defaults for this task (make violations loud to prevent silent drift).
        if opponent_depth != 1:
            raise ValueError(f"Self-play opponent_depth is fixed to 1 for now (got {opponent_depth}).")

        template = self._get_allowed_move_elim_template(template_path)
        engine = self._get_self_play_engine()

        @dataclass
        class _SPGame:
            game_id: str
            board: chess.Board
            model_color: chess.Color
            turn_idx: int = 0
            is_over: bool = False
            forfeit_reason: str = ""

        def _get_or_init_games(*, n_games: int) -> list[_SPGame]:
            """Return the persistent self-play games pool (initializing if needed)."""
            key = (int(n_games), int(opponent_depth))
            games = getattr(self, "_self_play_games", None)
            if (
                games is None
                or not isinstance(games, list)
                or len(games) != int(n_games)
                or getattr(self, "_self_play_games_key", None) != key
            ):
                # Initialize to ~50/50 "model moves first" vs "engine moves first".
                # In chess, White moves first, so this corresponds to:
                #   model_color=WHITE -> model moves first
                #   model_color=BLACK -> engine moves first
                n_model_first = int(n_games) // 2
                n_engine_first = int(n_games) - int(n_model_first)
                colors: list[chess.Color] = [chess.WHITE] * int(n_model_first) + [chess.BLACK] * int(n_engine_first)
                rng = np.random.default_rng(0)
                rng.shuffle(colors)

                setattr(self, "_self_play_games_key", key)
                setattr(self, "_self_play_game_counter", 0)

                fresh: list[_SPGame] = []
                for color in colors:
                    fresh.append(_new_game(model_color=color))
                setattr(self, "_self_play_games", fresh)
                games = fresh
                print(
                    f"[SELF_PLAY] init games pool: n_games={int(n_games)} "
                    f"(model_first={int(n_model_first)}, engine_first={int(n_engine_first)})",
                    flush=True,
                )
            return games

        def _new_game(*, model_color: chess.Color) -> _SPGame:
            counter = int(getattr(self, "_self_play_game_counter", 0) or 0)
            game_id = f"sp_g{counter:08d}"
            setattr(self, "_self_play_game_counter", counter + 1)
            return _SPGame(game_id=game_id, board=chess.Board(), model_color=model_color)

        def _restart_game(*, prev_game: _SPGame) -> _SPGame:
            """Restart a finished game, toggling who moves first for long-run balance."""
            next_color = chess.BLACK if prev_game.model_color == chess.WHITE else chess.WHITE
            return _new_game(model_color=next_color)

        def _step_model_moves(games: list[_SPGame]) -> None:
            # Retry loop (starter-kit semantics: resend same prompt).
            pending: dict[str, dict[str, Any]] = {g.game_id: {"game": g, "last_error": ""} for g in games}
            for _retry_idx in range(max_retries_per_turn):
                todo: list[_SPGame] = []
                prompts: list[list[dict[str, str]]] = []
                legal_lists: list[list[str]] = []

                # Build prompts only for games where it's actually the model's turn.
                for st in pending.values():
                    g = st["game"]
                    if g.is_over:
                        continue
                    if g.board.is_game_over(claim_draw=False):
                        g.is_over = True
                        continue
                    if g.board.turn != g.model_color:
                        st["last_error"] = "wrong_turn"
                        continue

                    legal_moves_uci = [m.uci().lower() for m in g.board.legal_moves]
                    if not legal_moves_uci:
                        g.is_over = True
                        continue

                    prompt_text = str(
                        template.render(
                            FEN=g.board.fen(),
                            legal_moves_uci_list=legal_moves_uci,
                            considered_moves_uci_list=legal_moves_uci,
                        )
                    )
                    todo.append(g)
                    legal_lists.append(legal_moves_uci)
                    prompts.append([{"role": "user", "content": prompt_text}])

                if not todo:
                    return

                outputs = self._self_play_generate(prompts=prompts, sampling_kwargs=sampling_kwargs)
                if len(outputs) != len(todo):
                    raise RuntimeError(f"self_play backend returned {len(outputs)} outputs for {len(todo)} prompts")

                for g, out_text, legal_moves_uci in zip(todo, outputs, legal_lists, strict=True):
                    parsed, err = self._self_play_decode_and_validate_move(
                        output_text=out_text, legal_moves_uci=legal_moves_uci
                    )
                    if err:
                        pending[g.game_id]["last_error"] = err
                        continue

                    try:
                        mv = chess.Move.from_uci(str(parsed))
                    except Exception:
                        pending[g.game_id]["last_error"] = "bad_move"
                        continue

                    g.board.push(mv)
                    if g.board.is_game_over(claim_draw=False):
                        g.is_over = True
                    pending.pop(g.game_id, None)

            # Any remaining pending games forfeit.
            for st in pending.values():
                g = st["game"]
                if g.is_over:
                    continue
                g.is_over = True
                g.forfeit_reason = str(st.get("last_error") or "invalid_output")

        def _step_opponent_moves(games: list[_SPGame]) -> None:
            for g in games:
                if g.is_over:
                    continue
                if g.board.is_game_over(claim_draw=False):
                    g.is_over = True
                    continue
                try:
                    result = engine.play(g.board, chess.engine.Limit(depth=int(opponent_depth)))
                    mv = result.move
                except Exception as exc:
                    g.is_over = True
                    g.forfeit_reason = f"engine_error:{exc}"
                    continue
                g.board.push(mv)
                if g.board.is_game_over(claim_draw=False):
                    g.is_over = True

        with marked_timer("self_play/init_games", sp_timing):
            games = _get_or_init_games(n_games=num_parallel_games)

        # Ensure we never record terminal positions: restart games that ended on a previous step.
        with marked_timer("self_play/record_positions", sp_timing):
            restarted_before = 0
            for idx, g in enumerate(games):
                if (not g.is_over) and g.board.is_game_over(claim_draw=False):
                    g.is_over = True
                if g.is_over:
                    games[idx] = _restart_game(prev_game=g)
                    restarted_before += 1

            # Record exactly one position per game (both model-to-move and opponent-to-move).
            positions: list[dict[str, Any]] = []
            model_to_move: list[_SPGame] = []
            opp_to_move: list[_SPGame] = []
            model_to_move_count = 0
            opp_to_move_count = 0
            for slot_id, g in enumerate(games):
                if g.board.is_game_over(claim_draw=False):
                    # Extremely defensive: should not happen because we restarted above.
                    games[slot_id] = _restart_game(prev_game=g)
                    g = games[slot_id]
                    restarted_before += 1

                is_model_turn = bool(g.board.turn == g.model_color)
                if is_model_turn:
                    model_to_move.append(g)
                    model_to_move_count += 1
                else:
                    opp_to_move.append(g)
                    opp_to_move_count += 1

                positions.append(
                    {
                        "fen": g.board.fen(),
                        "game_id": g.game_id,
                        "slot_id": int(slot_id),
                        "ply": int(len(g.board.move_stack)),
                        "model_color": "white" if g.model_color == chess.WHITE else "black",
                        "to_move_color": "white" if g.board.turn == chess.WHITE else "black",
                        "is_model_turn": bool(is_model_turn),
                        "turn_idx": int(g.turn_idx),
                        "forfeit_reason": "",
                    }
                )
                g.turn_idx += 1

        if len(positions) != int(train_bsz):
            raise RuntimeError(
                f"self_play: internal error: expected {int(train_bsz)} positions but got {len(positions)}"
            )

        # Advance each game by exactly one ply.
        if model_to_move:
            with marked_timer("self_play/model_moves", sp_timing):
                _step_model_moves(model_to_move)
        if opp_to_move:
            with marked_timer("self_play/opponent_moves", sp_timing):
                _step_opponent_moves(opp_to_move)

        # Restart any games that ended this step (toggle which side the model plays for that slot).
        with marked_timer("self_play/restart_finished", sp_timing):
            restarted_after = 0
            for idx, g in enumerate(games):
                if (not g.is_over) and g.board.is_game_over(claim_draw=False):
                    g.is_over = True
                if g.is_over:
                    games[idx] = _restart_game(prev_game=g)
                    restarted_after += 1

        print(
            f"[SELF_PLAY] step={int(self.global_steps)} pool_games={int(num_parallel_games)} "
            f"rows={len(positions)} model_to_move={int(model_to_move_count)} opp_to_move={int(opp_to_move_count)} "
            f"restarted_before={int(restarted_before)} restarted_after={int(restarted_after)}",
            flush=True,
        )

        data_source = f"self_play_depth{opponent_depth}_games{num_parallel_games}"
        reward_models: list[dict[str, Any]] = []
        extra_infos: list[dict[str, Any]] = []

        with marked_timer("self_play/score_and_build_rows", sp_timing):
            for i, pos in enumerate(positions):
                fen = str(pos.get("fen") or "").strip()
                if not fen:
                    raise ValueError("self_play produced an empty FEN")

                board = chess.Board(fen)
                with marked_timer("self_play/score_position_all_legal_moves", sp_timing):
                    legal_moves_uci, move_cps, move_winprobs, move_expected = score_position_all_legal_moves(
                        engine, board, depth=int(analysis_depth)
                    )
                if not legal_moves_uci:
                    raise ValueError(f"self_play: no legal moves for fen={fen!r}")

                # `score_position_all_legal_moves` guarantees `move_expected` is present and aligned to
                # the legal move list (WDL expected score when available; deterministic fallback otherwise).
                mu_map = move_expected
                best_move = ""
                best_mu = -float("inf")
                for mv in legal_moves_uci:
                    mu = float(mu_map.get(mv, -float("inf")))
                    if (mu > best_mu) or (mu == best_mu and (not best_move or mv < best_move)):
                        best_move = mv
                        best_mu = mu
                if not best_move:
                    raise ValueError(f"self_play: failed to select μ-best move for fen={fen!r}")

                import math

                # Offline-compatible baselines for delta-style rewards.
                best_cp = max(move_cps.values()) if move_cps else 0
                best_cp_moves = sorted([m for m, cp in move_cps.items() if cp == best_cp]) if move_cps else []
                best_move_uci = best_cp_moves[0] if best_cp_moves else best_move

                position_win_prob = float(move_winprobs.get(best_move_uci, 0.0))
                position_expected_score = float(max(move_expected.values())) if move_expected else float(position_win_prob)

                # Reward payload sanity checks (fail fast instead of propagating NaNs into GRPO stats).
                missing_keys = [
                    mv for mv in legal_moves_uci if mv not in move_cps or mv not in move_winprobs or mv not in move_expected
                ]
                if missing_keys:
                    raise ValueError(
                        f"self_play: incomplete move maps (missing {len(missing_keys)} moves; e.g. {missing_keys[:5]}) "
                        f"for fen={fen!r}"
                    )

                for mv in legal_moves_uci:
                    cp_v = float(move_cps[mv])
                    wp_v = float(move_winprobs[mv])
                    ex_v = float(move_expected[mv])
                    if not math.isfinite(cp_v):
                        raise ValueError(f"self_play: non-finite cp for move={mv} fen={fen!r}")
                    if not (math.isfinite(wp_v) and 0.0 <= wp_v <= 1.0):
                        raise ValueError(f"self_play: invalid winprob={wp_v} for move={mv} fen={fen!r}")
                    if not (math.isfinite(ex_v) and 0.0 <= ex_v <= 1.0):
                        raise ValueError(f"self_play: invalid expected_score={ex_v} for move={mv} fen={fen!r}")

                if best_move not in set(legal_moves_uci):
                    raise ValueError(f"self_play: ground_truth not in legal move list: gt={best_move} fen={fen!r}")
                if not math.isfinite(position_win_prob):
                    raise ValueError(f"self_play: non-finite position_win_prob for fen={fen!r}")
                if not math.isfinite(position_expected_score):
                    raise ValueError(f"self_play: non-finite position_expected_score for fen={fen!r}")

                rm: dict[str, Any] = {
                    "style": "rule",
                    "fen": fen,
                    "ground_truth": best_move,
                    "best_move_uci": best_move_uci,
                    "legal_moves_uci": list(legal_moves_uci),
                    "considered_moves_uci": list(legal_moves_uci),
                    "position_cp": int(best_cp),
                    "position_win_prob": float(position_win_prob),
                    "position_expected_score": float(position_expected_score),
                    "move_values_json": dumps_compact_sorted(move_winprobs),
                    "move_cps_json": dumps_compact_sorted(move_cps),
                    "move_expected_scores_json": dumps_compact_sorted(move_expected),
                }
                reward_models.append(rm)

                extra_infos.append(
                    {
                        "index": int(self.global_steps) * int(train_bsz) + int(i),
                        "self_play_step": int(self.global_steps),
                        "self_play_game_id": str(pos.get("game_id") or ""),
                        "self_play_slot_id": int(pos.get("slot_id") or 0),
                        "self_play_ply": int(pos.get("ply") or 0),
                        "self_play_model_color": str(pos.get("model_color") or ""),
                        "self_play_to_move_color": str(pos.get("to_move_color") or ""),
                        "self_play_is_model_to_move": bool(pos.get("is_model_turn") or False),
                        "self_play_turn_idx": int(pos.get("turn_idx") or 0),
                        "self_play_expected_score_source": "wdl" if move_expected != move_winprobs else "cp_winprob",
                    }
                )

        # Optional debug dump.
        dump_dir_raw = str(cfg.get("dump_dir", "") or "").strip()
        dump_steps = cfg.get("dump_steps", []) or []
        should_dump = bool(dump_dir_raw) and int(self.global_steps) in {int(x) for x in dump_steps if str(x).strip()}
        if should_dump:
            with marked_timer("self_play/dump_batch", sp_timing):
                out_dir = Path(dump_dir_raw)
                out_dir.mkdir(parents=True, exist_ok=True)
                dump_path = out_dir / f"step{int(self.global_steps):06d}.jsonl"
                with dump_path.open("w", encoding="utf-8") as fp:
                    for rm, extra in zip(reward_models, extra_infos, strict=True):
                        prompt_text = str(
                            template.render(
                                FEN=rm["fen"],
                                legal_moves_uci_list=rm["legal_moves_uci"],
                                considered_moves_uci_list=rm["considered_moves_uci"],
                            )
                        )
                        row = {
                            "data_source": data_source,
                            "ability": "chess",
                            "prompt": [{"role": "user", "content": prompt_text}],
                            "reward_model": rm,
                            "extra_info": extra,
                        }
                        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[SELF_PLAY] dumped step batch to {dump_path}", flush=True)

        dummy = torch.zeros((train_bsz, 1), dtype=torch.long)
        dummy_mask = torch.ones((train_bsz, 1), dtype=torch.long)
        dummy_pos = torch.zeros((train_bsz, 1), dtype=torch.long)

        return DataProto.from_dict(
            tensors={
                "input_ids": dummy,
                "attention_mask": dummy_mask,
                "position_ids": dummy_pos,
            },
            non_tensors={
                "data_source": np.array([data_source] * train_bsz, dtype=object),
                "reward_model": np.array(reward_models, dtype=object),
                "extra_info": np.array(extra_infos, dtype=object),
            },
            meta_info={"self_play_timing": dict(sp_timing)},
        )

    def _sample_forced_move(self, reward_payload: Any, temperature: float, rng: np.random.Generator) -> tuple[str, float]:
        """Sample a move for forced-prefix exploration using a temperature-adjusted distribution.

        Source signal (chess):
          - Prefer `reward_model.move_expected_scores_json` (bounded WDL expected score in [0, 1]).
          - Fall back to `reward_model.move_values_json` (winrate-like in [0, 1]) if expected scores are absent.

        We drop non-positive entries (0 or negative expectations) so that exploration
        never forces moves that were explicitly scored as zero.
        """
        move_map = {}
        gt = ""
        gt_val = 0.0
        if isinstance(reward_payload, dict):
            gt = (reward_payload.get("ground_truth") or "").strip().lower()
            raw_map = reward_payload.get("move_expected_scores_json") or reward_payload.get("move_values_json")
            if raw_map:
                try:
                    obj = json.loads(raw_map)
                    if isinstance(obj, dict):
                        move_map = {str(k).lower(): float(v) for k, v in obj.items()}
                except Exception:
                    move_map = {}
        if gt and gt in move_map:
            gt_val = float(move_map.get(gt, 0.0))

        if not move_map:
            return gt, gt_val

        # Remove zero/negative valued moves; fall back to GT if the filter empties the list.
        filtered = [(m, v) for m, v in move_map.items() if v > 0]
        if filtered:
            moves, vals = zip(*filtered)
        else:
            if gt:
                return gt, gt_val
            moves, vals = zip(*move_map.items())

        moves = np.array(moves)
        vals = np.array(vals, dtype=np.float64)
        orig_vals = vals.copy()
        temp = max(float(temperature), 1e-6)
        vals = np.clip(vals, 1e-6, None)
        weights = np.power(vals, 1.0 / temp)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            probs = np.full_like(weights, 1.0 / len(weights), dtype=np.float64)
        else:
            probs = weights / weights.sum()
        idx = int(rng.choice(len(moves), p=probs))
        return str(moves[idx]), float(orig_vals[idx])

    def _sample_forced_moves_for_group(
        self, reward_payload: Any, temperature: float, rng: np.random.Generator, count: int
    ) -> tuple[list[str], list[float]]:
        """Sample `count` forced moves for one uid group.

        Best-effort uniqueness:
          - If there are at least `count` eligible moves, sample without replacement.
          - Otherwise, sample with replacement.
        """
        count = int(count or 0)
        if count <= 0:
            return [], []

        # Reuse the same parsing and weighting semantics as `_sample_forced_move`,
        # but sample `count` moves in one call.
        move_map: dict[str, float] = {}
        gt = ""
        gt_val = 0.0
        if isinstance(reward_payload, dict):
            gt = (reward_payload.get("ground_truth") or "").strip().lower()
            raw_map = reward_payload.get("move_expected_scores_json") or reward_payload.get("move_values_json")
            if raw_map:
                try:
                    obj = json.loads(raw_map)
                    if isinstance(obj, dict):
                        move_map = {str(k).lower(): float(v) for k, v in obj.items()}
                except Exception:
                    move_map = {}
        if gt and gt in move_map:
            gt_val = float(move_map.get(gt, 0.0))

        if not move_map:
            # Degenerate case: no distribution → force GT (duplicates unavoidable).
            if gt:
                return [gt] * count, [gt_val] * count
            return [""] * count, [0.0] * count

        # Remove zero/negative valued moves; fall back to GT if the filter empties the list.
        filtered = [(m, v) for m, v in move_map.items() if v > 0]
        if filtered:
            moves, vals = zip(*filtered)
        else:
            if gt:
                return [gt] * count, [gt_val] * count
            moves, vals = zip(*move_map.items())

        moves_arr = np.array(moves)
        vals_arr = np.array(vals, dtype=np.float64)
        orig_vals = vals_arr.copy()
        temp = max(float(temperature), 1e-6)
        vals_arr = np.clip(vals_arr, 1e-6, None)
        weights = np.power(vals_arr, 1.0 / temp)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            probs = np.full_like(weights, 1.0 / len(weights), dtype=np.float64)
        else:
            probs = weights / weights.sum()

        replace = bool(count > len(moves_arr))
        idxs = rng.choice(len(moves_arr), size=count, replace=replace, p=probs)
        out_moves = [str(moves_arr[int(i)]) for i in np.asarray(idxs).tolist()]
        out_vals = [float(orig_vals[int(i)]) for i in np.asarray(idxs).tolist()]
        return out_moves, out_vals

    def _format_forced_move(self, move: str) -> str:
        """Normalize a forced-prefix move token for display."""
        if not move:
            return ""
        mv = str(move).strip().strip("`'\"")
        mv = re.sub(r"^(move|moving)\s+", "", mv, flags=re.IGNORECASE)
        mv = mv.replace("...", " ").replace("=", "")
        mv = mv.split()[0] if mv.split() else mv
        return mv.lower()

    def _tokenize_forced_prefix(self, template: str, move: str, response_cap: int | None) -> list[int]:
        """Tokenize a forced-prefix template into response-side token IDs."""
        template = template or ""
        move = (move or "").strip()

        if not template:
            return []

        token_ids: list[int]
        if "{move}" in template:
            before, after = template.split("{move}", 1)
            before_ids = self.tokenizer.encode(before, add_special_tokens=False)
            move_ids = self.tokenizer.encode(move, add_special_tokens=False) if move else []
            after_ids = self.tokenizer.encode(after, add_special_tokens=False)
            token_ids = before_ids + move_ids + after_ids
        else:
            token_ids = self.tokenizer.encode(template, add_special_tokens=False)

        if response_cap is not None and response_cap > 0:
            token_ids = token_ids[:response_cap]
        return token_ids

    def _compute_forced_prefix_apply_prob(self, cfg: dict) -> float:
        """Compute the per-rollout Bernoulli probability for applying forced-prefix injection.

        Schedule:
          - p(t) starts at `apply_prob_start` (default 1.0)
          - linearly anneals to `apply_prob_end` (default 0.2)
          - over the first `annealing_frac` portion of total training steps
        """
        annealing_frac_raw = cfg.get("annealing_frac", 0.5)
        if annealing_frac_raw is None:
            annealing_frac_raw = 0.5
        annealing_frac = float(annealing_frac_raw)
        annealing_frac = max(0.0, min(annealing_frac, 1.0))

        p_start_raw = cfg.get("apply_prob_start", 1.0)
        if p_start_raw is None:
            p_start_raw = 1.0
        p_end_raw = cfg.get("apply_prob_end", 0.2)
        if p_end_raw is None:
            p_end_raw = 0.2
        p_start = float(p_start_raw)
        p_end = float(p_end_raw)
        p_start = max(0.0, min(p_start, 1.0))
        p_end = max(0.0, min(p_end, 1.0))

        total_steps = int(getattr(self, "total_training_steps", 0) or 0)
        # `RayPPOTrainer.train()` treats `global_steps` as 1-indexed (it increments before the loop),
        # so use a 0-indexed step counter for schedules.
        step = max(int(self.global_steps or 0) - 1, 0)

        # When total steps are unknown/unset (or annealing_frac == 0), fall back to the final prob.
        anneal_horizon = float(total_steps) * annealing_frac
        if anneal_horizon <= 0.0:
            p = p_end
        else:
            progress = min(step / anneal_horizon, 1.0)
            p = p_start + (p_end - p_start) * progress

        lo = min(p_start, p_end)
        hi = max(p_start, p_end)
        return float(max(lo, min(p, hi)))

    def _apply_forced_prefix(self, gen_batch: DataProto, batch: DataProto) -> DataProto:
        """Annotate gen_batch with forced prefixes for exploration slots."""
        cfg = self.config.get("forced_prefix", None)
        if cfg is None or not cfg.get("enable", True):
            return gen_batch

        if gen_batch.meta_info.get("validate", False):
            return gen_batch

        apply_prob = self._compute_forced_prefix_apply_prob(cfg)
        temperature = float(cfg.get("move_temperature", 2.0) or 2.0)
        template_raw = cfg.get("prefix_template", "")
        template = str(template_raw) if template_raw is not None else ""
        # Hydra CLI overrides sometimes pass an extra layer of quotes, e.g.
        #   forced_prefix.prefix_template="\"<guess> {move} </guess>\""
        # which can survive into the resolved config as a *literal* leading/trailing quote.
        if len(template) >= 2 and template[0] == template[-1] and template[0] in ("\"", "'"):
            template = template[1:-1]
        # Interpret common escape sequences so CLI configs can use "\n" for newlines.
        template = template.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        total_size = len(gen_batch)
        if total_size == 0:
            return gen_batch

        rng = np.random.default_rng(self.global_steps or None)

        reward_payloads = batch.non_tensor_batch.get("reward_model", [])
        prompt_batch_size = len(reward_payloads)
        if prompt_batch_size == 0:
            prompt_batch_size = len(batch)
        if prompt_batch_size <= 0:
            prompt_batch_size = total_size

        group_size = 1
        if prompt_batch_size > 0 and total_size % prompt_batch_size == 0:
            group_size = max(1, total_size // prompt_batch_size)
        else:
            # Fallback: treat each row as its own "prompt group".
            prompt_batch_size = total_size
            group_size = 1

        response_cap = getattr(self.config.actor_rollout_ref.rollout, "response_length", None)
        forced_prefixes: list[list[int]] = []
        forced_moves: list[str] = []
        forced_move_values: list[float] = []
        forced_is_forced: list[bool] = []

        for prompt_idx in range(prompt_batch_size):
            # Per-rollout Bernoulli decision (annealed): apply forced prefix independently per rollout.
            forced_flags = rng.random(group_size) < apply_prob
            forced_slots = [i for i, f in enumerate(forced_flags.tolist()) if bool(f)]

            # Sample one forced move per forced rollout.
            group_moves_raw: list[str] = []
            group_move_vals: list[float] = []
            if forced_slots and prompt_idx < len(reward_payloads):
                group_moves_raw, group_move_vals = self._sample_forced_moves_for_group(
                    reward_payloads[prompt_idx], temperature, rng, count=len(forced_slots)
                )

            forced_j = 0
            for slot in range(group_size):
                token_ids: list[int] = []
                move = ""
                move_value = 0.0
                is_forced = bool(forced_flags[slot])
                if is_forced and prompt_idx < len(reward_payloads):
                    move_raw = group_moves_raw[forced_j] if forced_j < len(group_moves_raw) else ""
                    move_value = group_move_vals[forced_j] if forced_j < len(group_move_vals) else 0.0
                    forced_j += 1
                    move = self._format_forced_move(move_raw)
                    if move and template:
                        token_ids = self._tokenize_forced_prefix(template=template, move=move, response_cap=response_cap)

                forced_prefixes.append(token_ids)
                forced_moves.append(move)
                forced_move_values.append(float(move_value))
                forced_is_forced.append(bool(token_ids))

        gen_batch.non_tensor_batch["forced_prefix_token_ids"] = np.array(forced_prefixes, dtype=object)
        gen_batch.non_tensor_batch["forced_prefix_move"] = np.array(forced_moves, dtype=object)
        gen_batch.non_tensor_batch["forced_prefix_value"] = np.array(forced_move_values, dtype=np.float32)
        gen_batch.non_tensor_batch["forced_prefix_is_forced"] = np.array(forced_is_forced, dtype=np.bool_)
        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            test_gen_batch = self._apply_forced_prefix(test_gen_batch, test_batch)
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _run_full_game_eval(self) -> dict:
        """Optional chess-only full-game evaluation (vLLM + Stockfish).

        This is intentionally decoupled from `test_freq` (one-step puzzle validation) via the
        dedicated `trainer.full_eval_freq` knob.
        """
        freq = int(self.config.trainer.get("full_eval_freq", -1) or -1)
        if freq <= 0:
            return {}

        full_cfg = self.config.trainer.get("full_eval", None) or {}

        # Import lazily so non-chess recipes don't pay import costs or dependency requirements.
        from pathlib import Path

        import torch

        from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto

        from recipe.chess.full_game_eval import FullGameEvalConfig, StockfishConfig, run_full_game_eval

        # Derive a stable run root: `${VERL_BASE_DIR}/full_game_eval/global_step_{step}`.
        ckpt_dir = str(self.config.trainer.default_local_dir)
        ckpt_path = Path(ckpt_dir)
        if not ckpt_path.is_absolute():
            ckpt_path = (Path.cwd() / ckpt_path).resolve()
        run_root = ckpt_path.parent
        out_dir = run_root / "full_game_eval" / f"global_step_{self.global_steps}"

        def _as_int_list(x) -> list[int]:
            if x is None:
                return []
            try:
                return [int(v) for v in list(x)]
            except Exception:
                return [int(x)]

        opponent_depths = _as_int_list(full_cfg.get("opponent_depths", [1, 5]))
        games_per_depth = int(full_cfg.get("games_per_depth", 250) or 250)
        rounds_raw = full_cfg.get("rounds", None)
        games_per_round_raw = full_cfg.get("games_per_round", None)
        rounds = None
        games_per_round = None
        if rounds_raw is not None:
            try:
                rounds = int(rounds_raw)
            except Exception:
                rounds = None
        if games_per_round_raw is not None:
            try:
                games_per_round = int(games_per_round_raw)
            except Exception:
                games_per_round = None

        temperature = float(full_cfg.get("temperature", 0.6) or 0.6)
        top_p = float(full_cfg.get("top_p", 0.95) or 0.95)
        max_response_tokens = int(full_cfg.get("max_response_tokens", 512) or 512)

        max_retries_per_turn = int(full_cfg["max_retries_per_turn"])
        opponent_movetime_ms = int(full_cfg.get("opponent_movetime_ms", 100) or 100)
        resignation_cpl = int(full_cfg.get("resignation_cpl", 1000) or 1000)
        acpl_eval_depth = int(full_cfg.get("acpl_eval_depth", 20) or 20)
        acpl_eval_movetime_ms = int(full_cfg.get("acpl_eval_movetime_ms", 1000) or 1000)
        acpl_eval_cp_cap = int(full_cfg.get("acpl_eval_cp_cap", 1000) or 1000)
        mate_score_cp = int(full_cfg.get("mate_score_cp", 1000) or 1000)

        # Match starter-kit default: `max_moves=200` (plies). Allow disabling by setting <= 0.
        max_plies_raw = full_cfg.get("max_plies", 200)
        if max_plies_raw is None:
            max_plies = None
        else:
            try:
                max_plies_int = int(max_plies_raw)
            except Exception:
                max_plies_int = 200
            max_plies = max_plies_int if max_plies_int > 0 else None

        stockfish_path = str(full_cfg.get("stockfish_path", ".third_party_cache/stockfish/src/stockfish"))
        stockfish_threads = int(full_cfg.get("stockfish_threads", 1) or 1)
        stockfish_hash_mb = int(full_cfg.get("stockfish_hash_mb", 128) or 128)
        opponent_skill_level = int(full_cfg.get("opponent_skill_level", 0) or 0)
        eval_skill_level = int(full_cfg.get("eval_skill_level", 20) or 20)
        acpl_workers = int(full_cfg.get("acpl_workers", 1) or 1)
        acpl_threads_raw = full_cfg.get("acpl_threads", None)
        if acpl_threads_raw is None:
            acpl_threads = int(stockfish_threads)
        else:
            try:
                acpl_threads = int(acpl_threads_raw)
            except Exception:
                acpl_threads = int(stockfish_threads)

        prompt_template_path_raw = full_cfg.get("prompt_template_path", None)
        prompt_template_path = None
        if prompt_template_path_raw is not None:
            s = str(prompt_template_path_raw).strip()
            prompt_template_path = s if s else None

        # Chat backend backed by the in-training rollout vLLM engine.
        # This keeps evaluation high-throughput without instantiating a second vLLM engine.
        class _RolloutChatBackend:
            def __init__(self, trainer: "RayPPOTrainer"):
                self.trainer = trainer

            def generate(
                self,
                prompts: list[list[dict[str, str]]],
                *,
                temperature: float,
                top_p: float,
                max_tokens: int,
                seeds: list[int] | None = None,
            ) -> list[str]:
                # NOTE: We ignore per-prompt seeds; vLLM rollout in sync mode doesn't expose per-request seeds here.
                tokenizer = self.trainer.tokenizer
                apply_kwargs = dict(self.trainer.config.data.get("apply_chat_template_kwargs", {}) or {})
                use_chat_template = as_bool(
                    self.trainer.config.data.get("use_chat_template", True),
                    default=True,
                )

                prompt_token_ids: list[list[int]] = []
                for messages in prompts:
                    _, ids = encode_prompt_from_messages(
                        tokenizer,
                        messages,
                        use_chat_template=use_chat_template,
                        add_generation_prompt=True,
                        apply_chat_template_kwargs=apply_kwargs,
                    )
                    prompt_token_ids.append([int(x) for x in ids])

                if not prompt_token_ids:
                    return []

                pad_id = tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = tokenizer.eos_token_id
                if pad_id is None:
                    raise ValueError("Tokenizer has no pad_token_id or eos_token_id; cannot build prompts.")

                max_len = max(len(x) for x in prompt_token_ids)
                bsz = len(prompt_token_ids)

                input_ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long)
                attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
                for i, ids in enumerate(prompt_token_ids):
                    start = max_len - len(ids)
                    input_ids[i, start:] = torch.tensor(ids, dtype=torch.long)
                    attention_mask[i, start:] = 1

                position_ids = torch.cumsum(attention_mask, dim=1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)

                gen_batch = DataProto.from_dict(
                    tensors={
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                    },
                    non_tensors={},
                    meta_info={
                        "eos_token_id": tokenizer.eos_token_id,
                        "pad_token_id": int(pad_id),
                        "recompute_log_prob": False,
                        "do_sample": True,
                        # Treat as validation-like generation so forced-prefix exploration is disabled.
                        "validate": True,
                         "sampling_kwargs": {
                             "temperature": float(temperature),
                             "top_p": float(top_p),
                             "max_tokens": int(max_tokens),
                             "detokenize": True,
                         },
                        "global_steps": int(self.trainer.global_steps),
                    },
                )

                if self.trainer.async_rollout_mode:
                    raise NotImplementedError("full_game_eval is only supported in sync rollout mode for now.")

                # Pad to be divisible by dp_size, mirroring validation behavior.
                size_divisor = self.trainer.actor_rollout_wg.world_size
                gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
                out_padded = self.trainer.actor_rollout_wg.generate_sequences(gen_batch_padded)
                out = unpad_dataproto(out_padded, pad_size=pad_size)

                output_ids = out.batch["responses"]
                return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

        backend = _RolloutChatBackend(self)

        cfg = FullGameEvalConfig(
            opponent_depths=opponent_depths,
            games_per_depth=games_per_depth,
            seed=int(self.global_steps),
            rounds=rounds,
            games_per_round=games_per_round,
            temperature=temperature,
            top_p=top_p,
            max_response_tokens=max_response_tokens,
            max_retries_per_turn=max_retries_per_turn,
            opponent_movetime_ms=opponent_movetime_ms,
            resignation_cpl=resignation_cpl,
            acpl_eval_depth=acpl_eval_depth,
            acpl_eval_movetime_ms=acpl_eval_movetime_ms,
            acpl_eval_cp_cap=acpl_eval_cp_cap,
            acpl_workers=acpl_workers,
            mate_score_cp=mate_score_cp,
            max_plies=max_plies,
            stockfish_opponent=StockfishConfig(
                path=stockfish_path,
                threads=stockfish_threads,
                hash_mb=stockfish_hash_mb,
                skill_level=opponent_skill_level,
            ),
            stockfish_eval=StockfishConfig(
                path=stockfish_path,
                threads=acpl_threads,
                hash_mb=stockfish_hash_mb,
                skill_level=eval_skill_level,
            ),
            prompt_template_path=prompt_template_path,
            out_dir=out_dir,
        )

        summary = run_full_game_eval(cfg=cfg, backend=backend)

        # Persist full-game-eval artifacts to W&B (mirrors rollout/validation JSONL saving).
        if "wandb" in self.config.trainer.logger:
            import wandb

            if wandb.run is not None:
                for k in ("moves_jsonl", "games_jsonl", "summary_json", "games_pgn"):
                    p = summary.get("paths", {}).get(k, None)
                    if isinstance(p, str) and p:
                        wandb.save(p, policy="now")

        # Return metrics for W&B/console logging.
        metrics: dict[str, float] = {}
        total_games = 0
        total_wins = 0
        total_losses = 0
        total_draws = 0
        # ACPL aggregates:
        # - Starter-kit style: mean of per-game ACPL values (equal weight per game).
        total_acpl_sum_per_game = 0.0
        total_acpl_games = 0
        # - Move-weighted: sum(CPL) / num_moves (useful for debugging; not what starter-kit prints).
        total_acpl_sum_per_move = 0.0
        total_acpl_moves = 0

        for depth in opponent_depths:
            row = summary.get("summary_by_depth", {}).get(f"depth_{depth}", {})
            num_games = int(row.get("num_games", 0) or 0)
            wins = int(row.get("wins", 0) or 0)
            losses = int(row.get("losses", 0) or 0)
            draws = int(row.get("draws", 0) or 0)
            acpl_mean_per_game = float(row.get("acpl_mean", float("nan")))
            acpl_mean_per_move = float(row.get("acpl_mean_per_move", float("nan")))

            total_games += num_games
            total_wins += wins
            total_losses += losses
            total_draws += draws
            total_acpl_sum_per_game += float(row.get("acpl_sum", 0.0) or 0.0)
            total_acpl_games += int(row.get("acpl_games", num_games) or num_games)
            total_acpl_sum_per_move += float(row.get("acpl_sum_per_move", 0.0) or 0.0)
            total_acpl_moves += int(row.get("acpl_moves", 0) or 0)

            depth_prefix = f"full_game_eval/depth_{depth}"
            metrics[f"{depth_prefix}/wins"] = float(wins)
            metrics[f"{depth_prefix}/losses"] = float(losses)
            metrics[f"{depth_prefix}/draws"] = float(draws)
            metrics[f"{depth_prefix}/win_rate"] = float(wins / num_games) if num_games > 0 else float("nan")
            # Competition-facing number (matches starter-kit printing).
            metrics[f"{depth_prefix}/acpl"] = float(acpl_mean_per_game)
            metrics[f"{depth_prefix}/acpl_per_move"] = float(acpl_mean_per_move)

        metrics["full_game_eval/overall/num_games"] = float(total_games)
        metrics["full_game_eval/overall/wins"] = float(total_wins)
        metrics["full_game_eval/overall/losses"] = float(total_losses)
        metrics["full_game_eval/overall/draws"] = float(total_draws)
        metrics["full_game_eval/overall/win_rate"] = float(total_wins / total_games) if total_games > 0 else float("nan")
        metrics["full_game_eval/overall/acpl"] = (
            float(total_acpl_sum_per_game / total_acpl_games) if total_acpl_games > 0 else float("nan")
        )
        metrics["full_game_eval/overall/acpl_per_move"] = (
            float(total_acpl_sum_per_move / total_acpl_moves) if total_acpl_moves > 0 else float("nan")
        )

        return metrics

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _get_hf_checkpoint_repo_id(self) -> str:
        # Allow overrides without changing launcher configs.
        return str(os.environ.get("HF_CKPT_REPO_ID") or "Gabr1e11/a_lot_of_models")

    def _get_hf_token(self) -> str | None:
        # Never print this token. Prefer environment variables that are already
        # commonly used by `huggingface_hub`.
        for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            v = os.environ.get(k)
            if v:
                return str(v)
        return None

    def _normalize_data_files(self, data_files: Any) -> list[str]:
        """Normalize Hydra values like `data.train_files` into a stable list of file paths."""
        if data_files is None:
            return []
        if isinstance(data_files, str):
            return [data_files]
        try:
            return [str(x) for x in list(data_files)]
        except Exception:
            return [str(data_files)]

    def _sha256_file(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(8 * 1024 * 1024)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    def _compute_dataset_fingerprint(self) -> dict[str, Any]:
        """Compute a deterministic dataset fingerprint derived from file contents."""
        if self._hf_dataset_fingerprint is not None:
            return self._hf_dataset_fingerprint

        train_files = self._normalize_data_files(getattr(self.config.data, "train_files", None))
        val_files = self._normalize_data_files(getattr(self.config.data, "val_files", None))

        def _fingerprint_files(files: list[str]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            cwd = os.getcwd()
            for p in files:
                p2 = os.path.expanduser(str(p))
                if not os.path.isabs(p2):
                    p2 = os.path.abspath(p2)
                # Use a stable identifier (prefer repo-relative path) so the same dataset
                # contents hash identically across machines/containers.
                try:
                    file_id = os.path.relpath(p2, cwd) if os.path.commonpath([cwd, p2]) == cwd else os.path.basename(p2)
                except Exception:
                    file_id = os.path.basename(p2)
                if not os.path.exists(p2):
                    out.append({"id": file_id, "missing": True})
                    continue
                out.append(
                    {
                        "id": file_id,
                        "size": int(os.path.getsize(p2)),
                        "sha256": self._sha256_file(p2),
                    }
                )
            # Sort for determinism and to reduce sensitivity to path ordering.
            out.sort(key=lambda x: (x.get("id", ""), x.get("sha256", "")))
            return out

        train_fp = _fingerprint_files(train_files)
        val_fp = _fingerprint_files(val_files)
        payload = {"train": train_fp, "val": val_fp}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        combined = hashlib.sha256(blob).hexdigest()

        self._hf_dataset_fingerprint = {**payload, "combined_sha256": combined}
        return self._hf_dataset_fingerprint

    def _compute_config_hash(self) -> str:
        """Compute a deterministic run hash for HF checkpoint naming/resume."""
        cfg = OmegaConf.to_container(self.config, resolve=True)
        if not isinstance(cfg, dict):
            cfg = {"config": cfg}

        # Remove machine-specific / logging-only fields so the same *training* config
        # (plus dataset contents) maps to the same hash across restarts/machines.
        def _del_path(d: dict[str, Any], path: list[str]) -> None:
            cur: Any = d
            for key in path[:-1]:
                if not isinstance(cur, dict) or key not in cur:
                    return
                cur = cur[key]
            if isinstance(cur, dict):
                cur.pop(path[-1], None)

        for p in (
            ["data", "train_files"],
            ["data", "val_files"],
            ["trainer", "default_local_dir"],
            ["trainer", "default_hdfs_dir"],
            ["trainer", "rollout_data_dir"],
            ["trainer", "validation_data_dir"],
            ["trainer", "config_hash"],
            ["trainer", "experiment_name"],
            ["trainer", "project_name"],
            ["trainer", "logger"],
            ["ray_kwargs"],
        ):
            _del_path(cfg, p)

        dataset_fp = self._compute_dataset_fingerprint()
        identity = {"config": cfg, "dataset_fingerprint": dataset_fp.get("combined_sha256"), "dataset": dataset_fp}
        blob = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def _get_or_compute_config_hash(self) -> str:
        if self._hf_config_hash is None:
            self._hf_config_hash = self._compute_config_hash()
        return self._hf_config_hash

    def _maybe_init_hf_upload_executor(self) -> None:
        if self._hf_upload_executor is not None:
            return
        from concurrent.futures import ThreadPoolExecutor

        max_workers = int(self.config.trainer.get("hf_upload_max_workers", 1) or 1)
        max_workers = max(1, max_workers)
        self._hf_upload_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hf_upload")

        # Best-effort final flush on clean interpreter shutdown.
        import atexit

        def _flush_on_exit() -> None:
            try:
                self._flush_hf_uploads(block=True)
            except Exception:
                pass

        atexit.register(_flush_on_exit)

    def _prune_hf_upload_futures(self) -> None:
        """Drop completed futures and surface any async upload errors in logs."""
        futs: list[Any] = []
        for fut in list(self._hf_upload_futures):
            if fut is None:
                continue
            if not hasattr(fut, "done") or not hasattr(fut, "result"):
                continue
            if fut.done():
                try:
                    fut.result()
                except Exception as e:
                    print(f"[hf] Warning: async upload failed: {type(e).__name__}: {e}")
                continue
            futs.append(fut)
        self._hf_upload_futures = futs

    def _schedule_hf_upload(self, local_global_step_folder: str, *, global_step: int) -> None:
        """Start a non-blocking HF upload for a just-written checkpoint folder."""
        # Keep the existing "token is mandatory when saving is enabled" contract,
        # but do the network I/O asynchronously.
        token = self._get_hf_token()
        if not token:
            raise RuntimeError("HF checkpoint upload requested but no token found in env (HF_TOKEN/HUGGINGFACE_HUB_TOKEN).")

        self._maybe_init_hf_upload_executor()
        self._prune_hf_upload_futures()

        # Note: do *not* rely on `self.global_steps` inside the background task.
        # The driver continues training and will mutate it.
        assert self._hf_upload_executor is not None
        fut = self._hf_upload_executor.submit(
            self._upload_checkpoint_to_hf,
            local_global_step_folder,
            global_step=int(global_step),
        )
        self._hf_upload_futures.append(fut)
        print(f"[hf] Scheduled async upload: global_step={int(global_step)}", flush=True)

    def _flush_hf_uploads(self, *, block: bool) -> None:
        """Wait for any queued HF uploads to complete (called on clean shutdown)."""
        if self._hf_upload_executor is None:
            return
        self._prune_hf_upload_futures()
        if block:
            for fut in list(self._hf_upload_futures):
                if fut is None:
                    continue
                try:
                    fut.result()
                except Exception as e:
                    print(f"[hf] Warning: async upload failed: {type(e).__name__}: {e}")
            self._hf_upload_futures = []
            try:
                self._hf_upload_executor.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass
            self._hf_upload_executor = None

    def _upload_checkpoint_to_hf(self, local_global_step_folder: str, *, global_step: int) -> None:
        """Upload a fully materialized checkpoint folder to Hugging Face Hub.

        Uses a two-phase commit marker so partially uploaded checkpoints are ignored on resume.
        """
        token = self._get_hf_token()
        if not token:
            raise RuntimeError(
                "HF checkpoint upload requested but no token found in env (HF_TOKEN/HUGGINGFACE_HUB_TOKEN)."
            )

        repo_id = self._get_hf_checkpoint_repo_id()
        config_hash = self._get_or_compute_config_hash()
        # Hugging Face layout: `{config_hash}/step000010/...`
        remote_folder = f"{config_hash}/step{int(global_step):06d}"

        # Build a completeness manifest (paths + sizes) for robust resume.
        files: dict[str, int] = {}
        for root, _, filenames in os.walk(local_global_step_folder):
            for name in filenames:
                p = os.path.join(root, name)
                rel = os.path.relpath(p, local_global_step_folder)
                try:
                    files[rel] = int(os.path.getsize(p))
                except Exception:
                    files[rel] = -1

        complete_payload = {
            "config_hash": config_hash,
            "global_step": int(global_step),
            "dataset_fingerprint": self._compute_dataset_fingerprint().get("combined_sha256"),
            "files": files,
        }

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        # Upload the checkpoint folder.
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=local_global_step_folder,
            path_in_repo=remote_folder,
            commit_message=f"Add checkpoint {remote_folder}",
        )

        # Upload completion marker last (separate commit) so resume can ignore partial folders.
        marker_name = "_UPLOAD_COMPLETE.json"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=marker_name) as f:
            json.dump(complete_payload, f, ensure_ascii=False, sort_keys=True)
            marker_path = f.name
        try:
            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=marker_path,
                path_in_repo=f"{remote_folder}/{marker_name}",
                commit_message=f"Mark checkpoint complete {remote_folder}",
            )
        finally:
            try:
                os.remove(marker_path)
            except Exception:
                pass

        print(f"[hf] Uploaded checkpoint to {repo_id}:{remote_folder}", flush=True)
        # Optional: delete the local checkpoint folder after a successful upload so HF is the
        # source-of-truth (useful on clusters where persistent storage is scarce).
        #
        # Note: we also remove the checkpoint tracker file so `find_latest_ckpt_path(...)` does not
        # point at a now-missing local folder. With `resume_mode=auto`, the trainer will restore the
        # newest remote checkpoint from HF instead.
        delete_local = os.environ.get("VERL_HF_UPLOAD_DELETE_LOCAL", "")
        if str(delete_local).strip().lower() in ("1", "true", "yes", "y", "on"):
            try:
                import shutil

                shutil.rmtree(local_global_step_folder, ignore_errors=True)
                tracker = os.path.join(os.path.dirname(local_global_step_folder), "latest_checkpointed_iteration.txt")
                try:
                    if os.path.exists(tracker):
                        os.remove(tracker)
                except Exception:
                    pass
                print(f"[hf] Deleted local checkpoint folder after upload: {local_global_step_folder}", flush=True)
            except Exception as e:
                print(
                    f"[hf] Warning: failed to delete local checkpoint folder after upload: {type(e).__name__}: {e}",
                    flush=True,
                )

    def _maybe_sync_latest_hf_checkpoint(self, checkpoint_folder: str) -> None:
        """If a newer HF checkpoint exists for this run config, download it into `checkpoint_folder`."""
        token = self._get_hf_token()
        if not token:
            return

        repo_id = self._get_hf_checkpoint_repo_id()
        config_hash = self._get_or_compute_config_hash()

        from huggingface_hub import HfApi, hf_hub_download, snapshot_download

        api = HfApi(token=token)
        try:
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        except Exception as e:
            print(f"[hf] Warning: failed to list repo files for resume: {type(e).__name__}: {e}")
            return

        marker_suffix = "/_UPLOAD_COMPLETE.json"
        marker_paths: list[tuple[int, str, str]] = []
        for f in repo_files:
            if not f.endswith(marker_suffix):
                continue
            # New layout: `{config_hash}/step000010/_UPLOAD_COMPLETE.json`
            if f.startswith(f"{config_hash}/step"):
                parts = f.split("/")
                if len(parts) < 3:
                    continue
                step_part = parts[1]
                if not step_part.startswith("step"):
                    continue
                try:
                    step = int(step_part[len("step") :])
                except Exception:
                    continue
                remote_folder = "/".join(parts[:2])
                marker_paths.append((step, remote_folder, f))
                continue

        if not marker_paths:
            return

        # Local latest step (if any).
        local_latest = find_latest_ckpt_path(checkpoint_folder)
        local_step = -1
        if local_latest and "global_step_" in local_latest:
            try:
                local_step = int(local_latest.split("global_step_")[-1])
            except Exception:
                local_step = -1

        best_step = -1
        best_remote_folder = None
        for step, remote_folder, marker in sorted(marker_paths, key=lambda x: x[0]):
            if step <= best_step:
                continue

            # Verify completeness: marker file lists all expected file paths.
            try:
                marker_local = hf_hub_download(repo_id=repo_id, repo_type="model", filename=marker, token=token)
                with open(marker_local, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                expected = payload.get("files", {})
                if not isinstance(expected, dict) or not expected:
                    continue
                ok = True
                for rel in expected.keys():
                    if f"{remote_folder}/{rel}" not in repo_files:
                        ok = False
                        break
                if not ok:
                    continue
            except Exception:
                continue

            best_step = step
            best_remote_folder = remote_folder

        if best_remote_folder is None or best_step <= local_step:
            return

        # Download the newest remote checkpoint into a temp dir, then atomically place it
        # under `checkpoint_folder/global_step_{step}`.
        target_global_step_dir = os.path.join(checkpoint_folder, f"global_step_{best_step}")
        if os.path.exists(target_global_step_dir):
            return

        os.makedirs(checkpoint_folder, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="hf_ckpt_", dir=checkpoint_folder)
        try:
            # NOTE: `snapshot_download(...)` can fail for transient reasons on HPC
            # (rate limiting, flaky outbound network, filesystem hiccups, etc.). In those cases, it's better
            # to warn loudly and continue training from scratch than to crash the whole job at startup.
            #
            # This function is called only when `resume_mode=auto`, so "best effort" is the right default.
            # A successful download will create `checkpoint_folder/global_step_{step}` and update the local
            # `latest_checkpointed_iteration.txt` tracker, enabling the existing loader path.
            print(
                f"[hf] Downloading checkpoint {best_remote_folder} -> {target_global_step_dir} (tmp={tmp_dir})",
                flush=True,
            )
            try:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    allow_patterns=[f"{best_remote_folder}/*"],
                    local_dir=tmp_dir,
                    local_dir_use_symlinks=False,
                    token=token,
                )
            except Exception as e:
                import traceback

                print(
                    f"[hf] Warning: snapshot_download failed for {repo_id}:{best_remote_folder}: {type(e).__name__}: {e}",
                    flush=True,
                )
                print(traceback.format_exc(), flush=True)
                return
            downloaded = os.path.join(tmp_dir, best_remote_folder)
            if not os.path.isdir(downloaded):
                print(f"[hf] Warning: downloaded folder missing: {downloaded}")
                return
            os.makedirs(checkpoint_folder, exist_ok=True)
            os.rename(downloaded, target_global_step_dir)

            # Keep the local latest-step marker consistent for the existing loader.
            local_latest_checkpointed_iteration = os.path.join(checkpoint_folder, "latest_checkpointed_iteration.txt")
            with open(local_latest_checkpointed_iteration, "w") as f:
                f.write(str(best_step))

            print(f"[hf] Downloaded checkpoint {best_remote_folder} -> {target_global_step_dir}")
        finally:
            # Clean up the temp directory if it still exists.
            try:
                if os.path.isdir(tmp_dir):
                    # Best-effort cleanup; ignore errors (e.g., if we renamed out the folder).
                    for root, dirs, files in os.walk(tmp_dir, topdown=False):
                        for name in files:
                            try:
                                os.remove(os.path.join(root, name))
                            except Exception:
                                pass
                        for name in dirs:
                            try:
                                os.rmdir(os.path.join(root, name))
                            except Exception:
                                pass
                    try:
                        os.rmdir(tmp_dir)
                    except Exception:
                        pass
            except Exception:
                pass

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}", flush=True)
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # IMPORTANT: checkpoint save is a common source of "silent" job failures on HPC
        # (filesystem hiccups, quota, slow I/O, occasional worker crashes). Make it loud and
        # best-effort so training doesn't die without a traceback.
        try:
            print(f"[ckpt] Saving actor checkpoint -> {actor_local_path}", flush=True)
            self.actor_rollout_wg.save_checkpoint(
                actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
            )
            print(f"[ckpt] Saved actor checkpoint -> {actor_local_path}", flush=True)

            if self.use_critic:
                critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
                critic_remote_path = (
                    None
                    if self.config.trainer.default_hdfs_dir is None
                    else os.path.join(
                        self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                    )
                )
                print(f"[ckpt] Saving critic checkpoint -> {critic_local_path}", flush=True)
                self.critic_wg.save_checkpoint(
                    critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
                )
                print(f"[ckpt] Saved critic checkpoint -> {critic_local_path}", flush=True)

            # Save dataloader state.
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
            print(f"[ckpt] Saving dataloader state -> {dataloader_local_path}", flush=True)
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)
            print(f"[ckpt] Saved dataloader state -> {dataloader_local_path}", flush=True)

            # Latest checkpointed iteration tracker (for the existing loader path).
            local_latest_checkpointed_iteration = os.path.join(
                self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
            )
            print(f"[ckpt] Updating latest checkpoint marker -> {local_latest_checkpointed_iteration}", flush=True)
            with open(local_latest_checkpointed_iteration, "w") as f:
                f.write(str(self.global_steps))
            print(f"[ckpt] Updated latest checkpoint marker (step={self.global_steps})", flush=True)
        except Exception as e:
            import shutil
            import traceback

            print(
                f"[ckpt] ERROR: checkpoint save failed at step={self.global_steps}: {type(e).__name__}: {e}",
                flush=True,
            )
            print(traceback.format_exc(), flush=True)
            # Best-effort cleanup of the partially-written checkpoint folder to avoid filling disk.
            try:
                shutil.rmtree(local_global_step_folder, ignore_errors=True)
            except Exception:
                pass
            return

        # Hugging Face upload (step-scoped, resume-by-config-hash).
        # - Only run when a token is present.
        # - Allow an explicit opt-out (useful on HPC when large uploads trigger memory pressure).
        hf_upload_enable = os.environ.get("VERL_HF_UPLOAD_ENABLE", "")
        if str(hf_upload_enable).strip().lower() in ("0", "false", "no", "off"):
            return
        if not self._get_hf_token():
            return
        try:
            self._schedule_hf_upload(local_global_step_folder, global_step=int(self.global_steps))
        except Exception as e:
            import traceback

            print(
                f"[hf] Warning: failed to schedule HF upload for step={self.global_steps}: {type(e).__name__}: {e}",
                flush=True,
            )
            print(traceback.format_exc(), flush=True)
            return

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # Be explicit in logs: this is often the first thing users check when a job "finishes early".
            print("[RESUME FROM STEP 0] (resume_mode=disable)")
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            # HF auto-resume: if a newer checkpoint exists for this config hash, pull it locally
            # before resolving `find_latest_ckpt_path`.
            if self.config.trainer.resume_mode == "auto":
                self._maybe_sync_latest_hf_checkpoint(checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("[RESUME FROM STEP 0] (no checkpoint found; training from scratch)")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        # Single, easy-to-grep line for Slurm logs.
        print(f"[RESUME FROM STEP {self.global_steps}]")
        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute IS weights and apply rejection sampling for rollout-training mismatch.

        Computes importance sampling weights to correct for distribution mismatch between
        rollout and training policies. Applies rejection sampling (mask mode/veto) by
        modifying response_mask. Always updates response_mask; conditionally adds IS weights.

        Key behavior:
        - response_mask: ALWAYS updated with rejection (mask mode + veto excluded from training)
        - rollout_is_weights: Added to batch ONLY if config.algorithm.rollout_is=True

        This separation ensures:
        - Rejection works even when IS weights are disabled (rollout_is=False)
        - Metrics can be monitored before enabling IS weight application

        Args:
            batch: DataProto with old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics):
                updated_batch: Batch with modified response_mask (always) and rollout_is_weights (if rollout_is=True)
                metrics: Dict of IS and mismatch metrics, all with "mismatch/" prefix
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch (None = disabled, float = enabled)
        rollout_is_threshold = self.config.algorithm.get("rollout_is_threshold", None)
        if rollout_is_threshold is not None and rollout_is_threshold > 0 and "rollout_log_probs" in batch.batch:
            # Compute IS weights and get modified response_mask
            rollout_is_weights, modified_response_mask, rollout_is_metrics = compute_rollout_importance_weights(
                old_log_prob=batch.batch["old_log_probs"],
                rollout_log_prob=batch.batch["rollout_log_probs"],
                response_mask=batch.batch["response_mask"],
                rollout_is_level=self.config.algorithm.rollout_is_level,
                rollout_is_mode=self.config.algorithm.rollout_is_mode,
                rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
                rollout_is_threshold_lower=self.config.algorithm.get("rollout_is_threshold_lower", None),
                rollout_is_veto_threshold=self.config.algorithm.get("rollout_is_veto_threshold", None),
            )

            # ALWAYS update response_mask with rejection (even if rollout_is=False)
            # - Mask mode: tokens with outlier IS ratios excluded
            # - Veto: sequences with catastrophic tokens excluded
            # This ensures correct loss normalization (rejected samples not in denominator)
            batch.batch["response_mask"] = modified_response_mask

            # Conditionally add IS weights based on rollout_is config flag
            # - rollout_is=True: Enable IS weight correction in policy loss
            # - rollout_is=False: Metrics-only mode (rejection still applied via mask)
            apply_weights = self.config.algorithm.get("rollout_is", False)

            if apply_weights:
                # Add IS weights (safety-bounded, mode-processed) to enable weight correction
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        return batch, {}

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # Upload a small run file that records the deterministic config hash used for HF checkpointing.
        self._maybe_save_config_hash_json()

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # Optional full-game chess evaluation before training starts.
        # This mirrors the periodic in-training full-game eval, but schedules one extra run
        # at the initial step (fresh run: global_steps=0; resume: global_steps=checkpoint step)
        # when `val_before_train` is enabled.
        if self.config.trainer.get("val_before_train", True):
            full_eval_freq = int(self.config.trainer.get("full_eval_freq", -1) or -1)
            if full_eval_freq > 0:
                pprint(f"Initial full-game eval start (step={self.global_steps})")
                full_eval_metrics = self._run_full_game_eval()
                if full_eval_metrics:
                    pprint(f"Initial full-game eval metrics (step={self.global_steps}): {full_eval_metrics}")
                    logger.log(data=full_eval_metrics, step=self.global_steps)

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # DAPO-style dynamic sampling via group filtering.
        # When enabled, we keep generating `data.gen_batch_size` prompt batches until we have
        # enough *qualified* prompt groups for `data.train_batch_size`, or we hit
        # `algorithm.filter_groups.max_num_gen_batches`.
        filter_cfg = self.config.algorithm.get("filter_groups", None) or {}
        filter_groups_enabled = bool(filter_cfg.get("enable", False))
        filter_metric_name = filter_cfg.get("metric", None)
        max_num_gen_batches = int(filter_cfg.get("max_num_gen_batches", 0) or 0)
        if filter_groups_enabled:
            if not filter_metric_name:
                raise ValueError("algorithm.filter_groups.enable=True requires algorithm.filter_groups.metric to be set.")
            if self.config.algorithm.use_kl_in_reward:
                raise NotImplementedError("filter_groups is not supported when algorithm.use_kl_in_reward=True.")
            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                raise NotImplementedError("filter_groups is not supported with AdvantageEstimator.REMAX.")
            print(
                f"[FILTER_GROUPS] enable=True metric={filter_metric_name!r} "
                f"train_batch_size={int(self.config.data.train_batch_size)} "
                f"gen_batch_size={int(self.config.data.get('gen_batch_size', self.config.data.train_batch_size))} "
                f"max_num_gen_batches={max_num_gen_batches}"
            )

        # On-the-fly allowed-move elimination (selection sampling).
        allowed_move_elim_cfg = self.config.algorithm.get("allowed_move_elim", None) or {}
        allowed_move_elim_enabled = bool(allowed_move_elim_cfg.get("enable", False))
        allowed_move_elim_template_path = str(allowed_move_elim_cfg.get("template_path", "") or "")
        allowed_move_elim_uid_mode = (
            str(allowed_move_elim_cfg.get("uid_mode", "per_round") or "per_round").strip().lower()
        )
        if allowed_move_elim_uid_mode in {"round", "per_round"}:
            allowed_move_elim_uid_mode = "per_round"
        elif allowed_move_elim_uid_mode in {"prompt", "per_prompt"}:
            allowed_move_elim_uid_mode = "per_prompt"
        allowed_move_elim_no_success_policy = str(
            allowed_move_elim_cfg.get("no_success_policy", "accept_last") or "accept_last"
        )
        allowed_move_elim_r_start = int(allowed_move_elim_cfg.get("r_max_start", 4) or 4)
        allowed_move_elim_r_end = int(allowed_move_elim_cfg.get("r_max_end", 1) or 1)
        allowed_move_elim_anneal_frac_raw = allowed_move_elim_cfg.get("anneal_frac", 0.5)
        allowed_move_elim_anneal_frac = (
            0.5 if allowed_move_elim_anneal_frac_raw is None else float(allowed_move_elim_anneal_frac_raw)
        )
        allowed_move_elim_group_reward_range_min = float(allowed_move_elim_cfg.get("group_reward_range_min", 0.0) or 0.0)
        allowed_move_elim_group_reward_range_dump_max_groups = int(
            allowed_move_elim_cfg.get("group_reward_range_dump_max_groups", 16) or 16
        )
        allowed_move_elim_stitch_round0_prompt_for_logprob = bool(
            allowed_move_elim_cfg.get("stitch_round0_prompt_for_logprob", False)
        )
        allowed_move_elim_gain_threshold = float(allowed_move_elim_cfg.get("gain_threshold", np.log(10.0)))
        if np.isnan(allowed_move_elim_gain_threshold):
            raise ValueError("allowed_move_elim.gain_threshold must be a finite float or +/-inf (got NaN).")
        allowed_move_elim_gain_filter_enabled = bool(np.isfinite(allowed_move_elim_gain_threshold))
        allowed_move_elim_stitch_per_round_for_logprob = bool(
            allowed_move_elim_stitch_round0_prompt_for_logprob and (not allowed_move_elim_gain_filter_enabled)
        )
        allowed_move_elim_force_use_considered_moves_uci = bool(
            allowed_move_elim_cfg.get("force_use_considered_moves_uci", False)
        )
        if allowed_move_elim_enabled:
            if allowed_move_elim_uid_mode not in {"per_round", "per_prompt"}:
                raise ValueError(
                    "allowed_move_elim.uid_mode must be one of {'per_round','per_prompt'} "
                    f"(got {allowed_move_elim_uid_mode!r})."
                )
            if filter_groups_enabled:
                raise ValueError("allowed_move_elim.enable=True is incompatible with filter_groups.enable=True.")
            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                raise NotImplementedError("allowed_move_elim is not supported with AdvantageEstimator.REMAX.")
            gen_bsz = int(self.config.data.get("gen_batch_size", self.config.data.train_batch_size))
            train_bsz = int(self.config.data.train_batch_size)
            if gen_bsz != train_bsz:
                raise ValueError(
                    "allowed_move_elim requires data.gen_batch_size == data.train_batch_size "
                    f"(got gen_batch_size={gen_bsz}, train_batch_size={train_bsz})."
                )
            if allowed_move_elim_no_success_policy not in {"accept_last"}:
                raise ValueError(
                    "allowed_move_elim.no_success_policy must be 'accept_last' for now "
                    f"(got {allowed_move_elim_no_success_policy!r})."
                )
            print(
                "[ALLOWED_MOVE_ELIM] enable=True "
                f"template_path={allowed_move_elim_template_path!r} "
                f"uid_mode={allowed_move_elim_uid_mode!r} "
                f"policy={allowed_move_elim_no_success_policy!r} "
                f"r_start={allowed_move_elim_r_start} r_end={allowed_move_elim_r_end} "
                f"anneal_frac={allowed_move_elim_anneal_frac} "
                f"group_reward_range_min={allowed_move_elim_group_reward_range_min} "
                f"gain_threshold={allowed_move_elim_gain_threshold} "
                f"gain_filter_enabled={allowed_move_elim_gain_filter_enabled!r} "
                f"stitch_round0_prompt_for_logprob={allowed_move_elim_stitch_round0_prompt_for_logprob!r} "
                f"force_use_considered_moves_uci={allowed_move_elim_force_use_considered_moves_uci!r}"
            )

        self_play_enabled = self._self_play_enabled()
        if self_play_enabled and not allowed_move_elim_enabled:
            raise ValueError("data.self_play.enable=True currently requires algorithm.allowed_move_elim.enable=True.")

        # Accumulators for filter_groups mode (persist across gen batches until one update step completes).
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        timing_raw: dict[str, float] = defaultdict(float) if filter_groups_enabled else {}
        filter_rejected_groups_total = 0
        filter_rejected_groups_by_penalty: dict[str, int] = defaultdict(int)
        filter_logged_rejected_group_summaries = 0
        filter_logged_rejected_rollout_samples = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                if allowed_move_elim_enabled:
                    timing_raw = defaultdict(float)
                elif not filter_groups_enabled:
                    timing_raw = {}
                elif batch is None and num_prompt_in_batch == 0 and num_gen_batches == 0:
                    # Start of a new update step.
                    timing_raw = defaultdict(float)
                    filter_rejected_groups_total = 0
                    filter_rejected_groups_by_penalty = defaultdict(int)
                    filter_logged_rejected_group_summaries = 0
                    filter_logged_rejected_rollout_samples = 0

                iteration_start_time = time.perf_counter()
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                if self_play_enabled:
                    with marked_timer("self_play_build_batch", timing_raw, color="yellow"):
                        new_batch = self._build_self_play_train_batch()
                    self_play_timing = new_batch.meta_info.pop("self_play_timing", {}) or {}
                    for name, value in self_play_timing.items():
                        timing_raw[str(name)] += float(value)
                is_last_step = self.global_steps >= self.total_training_steps
                reward_extra_infos_dict = {}
                selection_metrics: dict[str, Any] = {}

                if allowed_move_elim_enabled:
                    base_batch = new_batch
                    template = self._get_allowed_move_elim_template(allowed_move_elim_template_path)
                    round0_prompt_input_ids = base_batch.batch["input_ids"]
                    round0_prompt_attention_mask = base_batch.batch["attention_mask"]
                    round0_prompt_position_ids = base_batch.batch["position_ids"]
                    base_reward_models = base_batch.non_tensor_batch.get("reward_model", [])
                    legal_moves_all: list[list[str]] = []
                    for rm in base_reward_models:
                        if isinstance(rm, dict):
                            legal = self._normalize_uci_moves(rm.get("legal_moves_uci"))
                        else:
                            legal = []
                        if not legal:
                            raise ValueError("Empty legal_moves_uci encountered in allowed_move_elim batch.")
                        legal_moves_all.append(legal)

                    allowed_moves_current = [list(moves) for moves in legal_moves_all]
                    num_prompts = len(allowed_moves_current)
                    unresolved = list(range(num_prompts))
                    rounds_used = [0] * num_prompts
                    success_flags = [False] * num_prompts
                    forced_accept_flags = [False] * num_prompts
                    all_round_batches = []
                    groups_per_prompt = [0] * num_prompts
                    base_uids = (
                        [str(uuid.uuid4()) for _ in range(num_prompts)]
                        if allowed_move_elim_uid_mode == "per_prompt"
                        else []
                    )

                    r_max = self._compute_allowed_move_elim_r_max(
                        step=self.global_steps,
                        total_steps=self.total_training_steps,
                        r_start=allowed_move_elim_r_start,
                        r_end=allowed_move_elim_r_end,
                        anneal_frac=allowed_move_elim_anneal_frac,
                    )
                    selection_metrics["selection_sampler/r_max"] = int(r_max)

                    with marked_timer("step", timing_raw):
                        for round_idx in range(1, r_max + 1):
                            if not unresolved:
                                break

                            for prompt_idx in unresolved:
                                groups_per_prompt[prompt_idx] += 1

                            b_sizes_by_prompt = {i: len(allowed_moves_current[i]) for i in unresolved}
                            round_b_sizes = [b_sizes_by_prompt[i] for i in unresolved]
                            if round_b_sizes:
                                selection_metrics[f"selection_sampler/round{round_idx}_avg_b"] = float(
                                    np.mean(round_b_sizes)
                                )
                            selection_metrics[f"selection_sampler/round{round_idx}_prompt_count"] = int(len(unresolved))

                            round_allowed = [allowed_moves_current[i] for i in unresolved]
                            with marked_timer("allowed_move_elim/build_round_batch", timing_raw):
                                with marked_timer(
                                    f"allowed_move_elim/round{round_idx}_build_round_batch", timing_raw
                                ):
                                    round_batch = self._build_allowed_move_elim_batch(
                                        base_batch=base_batch,
                                        indices=unresolved,
                                        allowed_moves=round_allowed,
                                        legal_moves=legal_moves_all,
                                        template=template,
                                    )
                            round_batch.non_tensor_batch["allowed_move_elim_prompt_idx"] = np.array(
                                unresolved, dtype=np.int64
                            )
                            round_batch.non_tensor_batch["allowed_move_elim_round"] = np.array(
                                [round_idx] * len(unresolved), dtype=np.int64
                            )

                            if allowed_move_elim_uid_mode == "per_prompt":
                                round_uids = [base_uids[i] for i in unresolved]
                            else:
                                round_uids = [str(uuid.uuid4()) for _ in range(len(round_batch.batch))]
                            round_batch.non_tensor_batch["uid"] = np.array(round_uids, dtype=object)

                            gen_batch = self._get_gen_batch(round_batch)
                            gen_batch.meta_info["global_steps"] = self.global_steps
                            gen_batch_output = gen_batch.repeat(
                                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                            )
                            gen_batch_output = self._apply_forced_prefix(gen_batch_output, round_batch)

                            with marked_timer("gen", timing_raw, color="red"):
                                with marked_timer(f"allowed_move_elim/round{round_idx}_gen", timing_raw, color="red"):
                                    size_divisor = (
                                        self.actor_rollout_wg.world_size
                                        if not self.async_rollout_mode
                                        else self.config.actor_rollout_ref.rollout.agent.num_workers
                                    )
                                    gen_batch_output_padded, pad_size = pad_dataproto_to_divisor(
                                        gen_batch_output, size_divisor
                                    )
                                    if not self.async_rollout_mode:
                                        gen_batch_output_padded = self.actor_rollout_wg.generate_sequences(
                                            gen_batch_output_padded
                                        )
                                    else:
                                        gen_batch_output_padded = self.async_rollout_manager.generate_sequences(
                                            gen_batch_output_padded
                                        )
                                    gen_batch_output = unpad_dataproto(gen_batch_output_padded, pad_size=pad_size)

                                    worker_timing = dict(gen_batch_output.meta_info.get("timing", {}) or {})
                                    for timing_name, timing_value in worker_timing.items():
                                        timing_raw[f"allowed_move_elim/round{round_idx}_{timing_name}"] += float(
                                            timing_value
                                        )
                                        timing_raw[f"allowed_move_elim/{timing_name}_sum"] += float(timing_value)
                                    timing_raw.update(worker_timing)
                                    gen_batch_output.meta_info.pop("timing", None)

                            round_batch = round_batch.repeat(
                                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                            )
                            round_batch = round_batch.union(gen_batch_output)

                            if "response_mask" not in round_batch.batch.keys():
                                round_batch.batch["response_mask"] = compute_response_mask(round_batch)

                            if allowed_move_elim_force_use_considered_moves_uci:
                                self._force_use_considered_moves_uci(round_batch)

                            with marked_timer("reward", timing_raw, color="yellow"):
                                with marked_timer(f"allowed_move_elim/round{round_idx}_reward", timing_raw):
                                    if self.use_rm and "rm_scores" not in round_batch.batch.keys():
                                        reward_tensor = self.rm_wg.compute_rm_score(round_batch)
                                        round_batch = round_batch.union(reward_tensor)

                                    reward_tensor, reward_extra_infos_dict = compute_reward(round_batch, self.reward_fn)
                                    round_batch.batch["token_level_scores"] = reward_tensor
                                    round_batch.batch["token_level_rewards"] = round_batch.batch["token_level_scores"]
                                    if reward_extra_infos_dict:
                                        round_batch.non_tensor_batch.update(
                                            {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                        )

                            round_postprocess_start_time = time.perf_counter()
                            uids = round_batch.non_tensor_batch["uid"]
                            pred_moves = round_batch.non_tensor_batch.get("pred_move", [""] * len(round_batch))
                            gt_uci = round_batch.non_tensor_batch.get("gt_uci", [""] * len(round_batch))
                            penalty_applied = round_batch.non_tensor_batch.get("penalty_applied", [False] * len(round_batch))
                            in_subset = round_batch.non_tensor_batch.get("in_subset", [False] * len(round_batch))

                            uid_to_indices: dict[str, list[int]] = defaultdict(list)
                            for i, uid in enumerate(uids):
                                uid_to_indices[str(uid)].append(i)

                            uid_to_prompt = {
                                uid: prompt_idx for uid, prompt_idx in zip(round_uids, unresolved, strict=True)
                            }
                            success_uids: list[str] = []

                            for uid, idxs in uid_to_indices.items():
                                gt = str(gt_uci[idxs[0]] or "").strip().lower()
                                success = False
                                remove_set: set[str] = set()
                                for j in idxs:
                                    if not bool(penalty_applied[j]):
                                        pm = str(pred_moves[j] or "").strip().lower()
                                        if pm:
                                            if in_subset is None or bool(in_subset[j]):
                                                remove_set.add(pm)
                                        if pm and gt and pm == gt:
                                            success = True

                                prompt_idx = uid_to_prompt.get(uid)
                                if success:
                                    success_uids.append(uid)
                                    if prompt_idx is not None:
                                        success_flags[prompt_idx] = True
                                        rounds_used[prompt_idx] = round_idx
                                else:
                                    if prompt_idx is not None and remove_set:
                                        current = allowed_moves_current[prompt_idx]
                                        new_list = [m for m in current if m not in remove_set]
                                        if new_list:
                                            allowed_moves_current[prompt_idx] = new_list

                            forced_uids: list[str] = []
                            if round_idx == r_max and allowed_move_elim_no_success_policy == "accept_last":
                                success_set = set(success_uids)
                                forced_uids = [uid for uid in round_uids if uid not in success_set]
                                for uid in forced_uids:
                                    prompt_idx = uid_to_prompt.get(uid)
                                    if prompt_idx is not None and rounds_used[prompt_idx] == 0:
                                        rounds_used[prompt_idx] = round_idx
                                        forced_accept_flags[prompt_idx] = True

                            accept_uids = list(success_uids) + list(forced_uids)
                            accept_uid_set = set(accept_uids) if accept_uids else set()

                            # Log all round rollouts (including rejected) when a rollout log dir is enabled.
                            round_log_root = self.config.trainer.get("rejected_rollout_data_dir", None)
                            if not round_log_root:
                                round_log_root = self.config.trainer.get("rollout_data_dir", None)
                            round_log_root = str(round_log_root).strip() if round_log_root is not None else ""
                            if round_log_root:
                                round_dump_start_time = time.perf_counter()
                                round_log_dir = os.path.join(round_log_root, "allowed_move_elim_rounds")
                                uid_list = [str(uid) for uid in uids]
                                success_uid_set = set(success_uids)
                                forced_uid_set = set(forced_uids)
                                prompt_idx_list = [uid_to_prompt.get(uid, -1) for uid in uid_list]
                                round_extra = dict(reward_extra_infos_dict)
                                round_extra.update(
                                    {
                                        "allowed_move_elim_round": [round_idx] * len(uid_list),
                                        "allowed_move_elim_r_max": [r_max] * len(uid_list),
                                        "allowed_move_elim_prompt_idx": prompt_idx_list,
                                        "allowed_move_elim_b_size": [
                                            b_sizes_by_prompt.get(pidx, -1) for pidx in prompt_idx_list
                                        ],
                                        "allowed_move_elim_success": [uid in success_uid_set for uid in uid_list],
                                        "allowed_move_elim_forced_accept": [uid in forced_uid_set for uid in uid_list],
                                        "allowed_move_elim_accepted": [uid in accept_uid_set for uid in uid_list],
                                    }
                                )
                                inputs = self.tokenizer.batch_decode(round_batch.batch["prompts"], skip_special_tokens=True)
                                outputs = self.tokenizer.batch_decode(round_batch.batch["responses"], skip_special_tokens=True)
                                scores = round_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                                sample_gts = [
                                    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                    for item in round_batch
                                ]
                                round_filename = os.path.join(
                                    round_log_dir, f"{self.global_steps}_round{round_idx}.jsonl"
                                )
                                self._dump_generations_to_file(
                                    inputs=inputs,
                                    outputs=outputs,
                                    gts=sample_gts,
                                    scores=scores,
                                    reward_extra_infos_dict=round_extra,
                                    filename=round_filename,
                                )
                                round_dump_elapsed = time.perf_counter() - round_dump_start_time
                                timing_raw["allowed_move_elim/dump_round_rollouts"] += round_dump_elapsed
                                timing_raw[f"allowed_move_elim/round{round_idx}_dump_rollouts"] += round_dump_elapsed

                            if allowed_move_elim_stitch_per_round_for_logprob:
                                self._stitch_allowed_move_elim_round0_prompt_context(
                                    round_batch=round_batch,
                                    round0_prompt_input_ids=round0_prompt_input_ids,
                                    round0_prompt_attention_mask=round0_prompt_attention_mask,
                                    round0_prompt_position_ids=round0_prompt_position_ids,
                                )

                            all_round_batches.append(round_batch)
                            accepted_prompt_set = {uid_to_prompt[uid] for uid in accept_uids if uid in uid_to_prompt}
                            unresolved = [i for i in unresolved if i not in accepted_prompt_set]

                            if round_uids:
                                selection_metrics[f"selection_sampler/round{round_idx}_success_frac"] = float(
                                    len(success_uids) / len(round_uids)
                                )
                                selection_metrics[f"selection_sampler/round{round_idx}_success_count"] = int(
                                    len(success_uids)
                                )
                                if forced_uids:
                                    selection_metrics[f"selection_sampler/round{round_idx}_forced_accept_count"] = int(
                                        len(forced_uids)
                                    )

                            if unresolved:
                                selection_metrics[f"selection_sampler/round{round_idx}_avg_b_after"] = float(
                                    np.mean([len(allowed_moves_current[i]) for i in unresolved])
                                )
                            else:
                                selection_metrics[f"selection_sampler/round{round_idx}_avg_b_after"] = 0.0
                            timing_raw[f"allowed_move_elim/round{round_idx}_postprocess"] += (
                                time.perf_counter() - round_postprocess_start_time
                            )

                        if not all_round_batches:
                            raise ValueError("allowed_move_elim produced an empty batch.")
                        with marked_timer("allowed_move_elim/concat_round_batches", timing_raw):
                            batch = DataProto.concat(all_round_batches)
                        batch_before_range_filter = batch
                        reward_extra_infos_dict = {}

                        rounds_used_vals = [r for r in rounds_used if r > 0]
                        selection_metrics["selection_sampler/avg_rounds_used"] = float(
                            np.mean(rounds_used_vals) if rounds_used_vals else 0.0
                        )
                        selection_metrics["selection_sampler/avg_groups_per_prompt"] = float(
                            np.mean(groups_per_prompt) if groups_per_prompt else 0.0
                        )
                        selection_metrics["selection_sampler/min_groups_per_prompt"] = int(
                            min(groups_per_prompt) if groups_per_prompt else 0
                        )
                        selection_metrics["selection_sampler/max_groups_per_prompt"] = int(
                            max(groups_per_prompt) if groups_per_prompt else 0
                        )
                        selection_metrics["selection_sampler/total_groups"] = int(sum(groups_per_prompt))
                        selection_metrics["selection_sampler/success_rate"] = float(
                            sum(1 for x in success_flags if x) / float(max(1, num_prompts))
                        )
                        selection_metrics["selection_sampler/forced_accept_frac"] = float(
                            sum(1 for x in forced_accept_flags if x) / float(max(1, num_prompts))
                        )
                        selection_metrics["selection_sampler/accept_rate"] = float(
                            len(rounds_used_vals) / float(max(1, num_prompts))
                        )

                        # Optional: reject GRPO prompt-groups (uids) with too-small within-group reward range.
                        # This is designed to be compatible with allowed_move_elim (unlike algorithm.filter_groups).
                        if allowed_move_elim_group_reward_range_min > 0:
                            uids = batch_before_range_filter.non_tensor_batch.get("uid", None)
                            scores = batch_before_range_filter.non_tensor_batch.get("score", None)
                            if uids is None or scores is None:
                                raise ValueError(
                                    "allowed_move_elim group_reward_range_min requires non_tensor_batch['uid'] "
                                    "and non_tensor_batch['score'] to be present."
                                )

                            penalty_applied = batch_before_range_filter.non_tensor_batch.get("penalty_applied", None)
                            in_subset = batch_before_range_filter.non_tensor_batch.get("in_subset", None)

                            uid2idxs: dict[str, list[int]] = defaultdict(list)
                            for i, uid in enumerate(uids):
                                uid2idxs[str(uid)].append(int(i))

                            uid2range_all: dict[str, float] = {}
                            uid2range_valid: dict[str, float] = {}
                            for uid, idxs in uid2idxs.items():
                                all_scores = [float(scores[i]) for i in idxs]
                                uid2range_all[uid] = float(max(all_scores) - min(all_scores)) if all_scores else 0.0

                                valid_scores: list[float] = []
                                for i in idxs:
                                    if penalty_applied is not None and bool(penalty_applied[i]):
                                        continue
                                    if in_subset is not None and (not bool(in_subset[i])):
                                        continue
                                    valid_scores.append(float(scores[i]))
                                uid2range_valid[uid] = (
                                    float(max(valid_scores) - min(valid_scores))
                                    if valid_scores
                                    else float("nan")
                                )

                            # Attach per-sample group metrics for analysis / dumps (constant within each group).
                            batch_before_range_filter.non_tensor_batch["group_reward_range_all"] = np.asarray(
                                [uid2range_all[str(uid)] for uid in uids], dtype=np.float32
                            )
                            batch_before_range_filter.non_tensor_batch["group_reward_range_valid"] = np.asarray(
                                [uid2range_valid[str(uid)] for uid in uids], dtype=np.float32
                            )

                            rejected_uids = [
                                uid for uid, r in uid2range_all.items() if float(r) < float(allowed_move_elim_group_reward_range_min)
                            ]
                            rejected_uid_set = set(rejected_uids)

                            total_groups = len(uid2idxs)
                            rejected_groups = len(rejected_uids)
                            kept_groups = total_groups - rejected_groups
                            selection_metrics["selection_sampler/group_reward_range_min"] = float(
                                allowed_move_elim_group_reward_range_min
                            )
                            selection_metrics["selection_sampler/reward_range_groups_total"] = int(total_groups)
                            selection_metrics["selection_sampler/reward_range_groups_rejected"] = int(rejected_groups)
                            selection_metrics["selection_sampler/reward_range_groups_kept"] = int(kept_groups)
                            if total_groups > 0:
                                selection_metrics["selection_sampler/reward_range_rejected_frac"] = float(
                                    rejected_groups / total_groups
                                )
                                selection_metrics["selection_sampler/reward_range_kept_frac"] = float(
                                    kept_groups / total_groups
                                )

                            if rejected_uids:
                                keep_traj_idxs = [
                                    i for i, uid in enumerate(uids) if str(uid) not in rejected_uid_set
                                ]
                                if not keep_traj_idxs:
                                    raise ValueError(
                                        "allowed_move_elim group_reward_range_min rejected all groups for this step "
                                        f"(threshold={allowed_move_elim_group_reward_range_min}). "
                                        "Lower the threshold or increase batch size."
                                    )
                                batch = batch_before_range_filter[keep_traj_idxs]

                                print(
                                    f"[ALLOWED_MOVE_ELIM] group_reward_range_min={allowed_move_elim_group_reward_range_min} "
                                    f"rejected_groups={rejected_groups}/{total_groups} "
                                    f"(kept_groups={kept_groups})",
                                    flush=True,
                                )

                                # Optional debug dumps for rejected groups/samples.
                                reject_log_root = self.config.trainer.get("rejected_rollout_data_dir", None)
                                if not reject_log_root:
                                    reject_log_root = self.config.trainer.get("rollout_data_dir", None)
                                reject_log_root = (
                                    str(reject_log_root).strip() if reject_log_root is not None else ""
                                )
                                if reject_log_root:
                                    # 1) Per-group summaries (one record per uid).
                                    try:
                                        summary_dir = os.path.join(
                                            reject_log_root, "allowed_move_elim_group_reward_range_rejections"
                                        )
                                        os.makedirs(summary_dir, exist_ok=True)
                                        summary_path = os.path.join(summary_dir, f"{self.global_steps}.jsonl")

                                        prompt_idx_arr_full = batch_before_range_filter.non_tensor_batch.get(
                                            "allowed_move_elim_prompt_idx", None
                                        )
                                        round_arr_full = batch_before_range_filter.non_tensor_batch.get(
                                            "allowed_move_elim_round", None
                                        )
                                        b_size_arr_full = batch_before_range_filter.non_tensor_batch.get(
                                            "allowed_move_elim_b_size", None
                                        )
                                        rm_arr = batch_before_range_filter.non_tensor_batch.get("reward_model", None)
                                        penalty_reason_arr = batch_before_range_filter.non_tensor_batch.get(
                                            "penalty_reason", None
                                        )

                                        summary_lines: list[str] = []
                                        for uid in rejected_uids:
                                            idxs = uid2idxs.get(uid) or []
                                            if not idxs:
                                                continue
                                            i0 = int(idxs[0])
                                            rm = rm_arr[i0] if rm_arr is not None else {}
                                            fen = rm.get("fen") if isinstance(rm, dict) else None
                                            gt = rm.get("ground_truth") if isinstance(rm, dict) else None
                                            legal = rm.get("legal_moves_uci") if isinstance(rm, dict) else None
                                            entry = {
                                                "step": int(self.global_steps),
                                                "uid": str(uid),
                                                "allowed_move_elim_round": int(round_arr_full[i0])
                                                if round_arr_full is not None
                                                else None,
                                                "allowed_move_elim_prompt_idx": int(prompt_idx_arr_full[i0])
                                                if prompt_idx_arr_full is not None
                                                else None,
                                                "allowed_move_elim_b_size": int(b_size_arr_full[i0])
                                                if b_size_arr_full is not None
                                                else None,
                                                "group_reward_range_threshold": float(
                                                    allowed_move_elim_group_reward_range_min
                                                ),
                                                "group_reward_range_all": float(uid2range_all.get(uid, 0.0)),
                                                "group_reward_range_valid": float(uid2range_valid.get(uid, float("nan"))),
                                                "scores": [float(scores[i]) for i in idxs],
                                                "penalty_reasons": [
                                                    str(penalty_reason_arr[i])
                                                    for i in idxs
                                                    if penalty_reason_arr is not None
                                                ],
                                                "fen": fen,
                                                "ground_truth": gt,
                                                "n_legal_moves": int(len(legal)) if isinstance(legal, list) else None,
                                            }
                                            summary_lines.append(json.dumps(entry, ensure_ascii=False))
                                        if summary_lines:
                                            with open(summary_path, "w", encoding="utf-8") as f:
                                                f.write("\n".join(summary_lines) + "\n")
                                            print(
                                                f"[ALLOWED_MOVE_ELIM] dumped reward-range rejection summaries to {summary_path}",
                                                flush=True,
                                            )
                                    except Exception as exc:
                                        print(
                                            f"[ALLOWED_MOVE_ELIM] warning: failed to dump reward-range rejection summaries: {exc}",
                                            flush=True,
                                        )

                                    # 2) Rollout-level samples for a capped subset of rejected groups.
                                    try:
                                        max_groups = int(allowed_move_elim_group_reward_range_dump_max_groups)
                                        if max_groups != 0:
                                            if max_groups < 0:
                                                dump_uids = rejected_uids
                                            else:
                                                dump_uids = rejected_uids[:max_groups]
                                            dump_uid_set = set(str(u) for u in dump_uids)
                                            traj_idxs = [
                                                i
                                                for i, uid in enumerate(uids)
                                                if str(uid) in dump_uid_set
                                            ]
                                            if traj_idxs:
                                                sample_batch = batch_before_range_filter[traj_idxs]
                                                inputs = self.tokenizer.batch_decode(
                                                    sample_batch.batch["prompts"], skip_special_tokens=True
                                                )
                                                outputs = self.tokenizer.batch_decode(
                                                    sample_batch.batch["responses"], skip_special_tokens=True
                                                )
                                                dump_scores = (
                                                    sample_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                                                )
                                                sample_gts = [
                                                    item.non_tensor_batch.get("reward_model", {}).get(
                                                        "ground_truth", None
                                                    )
                                                    for item in sample_batch
                                                ]

                                                extra_keys = [
                                                    "uid",
                                                    "score",
                                                    "group_reward_range_all",
                                                    "group_reward_range_valid",
                                                    "allowed_move_elim_round",
                                                    "allowed_move_elim_prompt_idx",
                                                    "allowed_move_elim_b_size",
                                                    "pred_move",
                                                    "target_move",
                                                    "gt_uci",
                                                    "mu_pred",
                                                    "mu_target",
                                                    "in_subset",
                                                    "format_reward",
                                                    "penalty",
                                                    "penalty_reason",
                                                    "penalty_applied",
                                                    "reward_reason",
                                                ]
                                                extra_to_dump: dict[str, list[Any]] = {}
                                                for k in extra_keys:
                                                    if k in sample_batch.non_tensor_batch:
                                                        v = sample_batch.non_tensor_batch[k]
                                                        try:
                                                            extra_to_dump[k] = v.tolist()
                                                        except Exception:
                                                            extra_to_dump[k] = list(v)
                                                n = len(inputs)
                                                extra_to_dump["allowed_move_elim/rejected_by_reward_range"] = [
                                                    True
                                                ] * n
                                                extra_to_dump["allowed_move_elim/group_reward_range_threshold"] = [
                                                    float(allowed_move_elim_group_reward_range_min)
                                                ] * n
                                                sample_dir = os.path.join(
                                                    reject_log_root,
                                                    "allowed_move_elim_group_reward_range_rejected_samples",
                                                )
                                                os.makedirs(sample_dir, exist_ok=True)
                                                sample_path = os.path.join(sample_dir, f"{self.global_steps}.jsonl")
                                                self._dump_generations_to_file(
                                                    inputs=inputs,
                                                    outputs=outputs,
                                                    gts=sample_gts,
                                                    scores=dump_scores,
                                                    reward_extra_infos_dict=extra_to_dump,
                                                    filename=sample_path,
                                                )
                                    except Exception as exc:
                                        print(
                                            f"[ALLOWED_MOVE_ELIM] warning: failed to dump reward-range rejection samples: {exc}",
                                            flush=True,
                                        )
                            else:
                                batch = batch_before_range_filter

                        gain_filter_start_time = time.perf_counter()
                        if allowed_move_elim_gain_filter_enabled:
                            if "response_mask" not in batch.batch.keys():
                                batch.batch["response_mask"] = compute_response_mask(batch)

                            with marked_timer("allowed_move_elim_gain_logprob_pi", timing_raw, color="blue"):
                                gain_logprob_pi = self.actor_rollout_wg.compute_log_prob(batch)
                            logprob_pi_seq = self._sum_masked_token_log_probs(
                                token_log_probs=gain_logprob_pi.batch["old_log_probs"],
                                response_mask=batch.batch["response_mask"],
                            )

                            gain_batch_p0 = self._clone_allowed_move_elim_logprob_batch(batch)
                            self._stitch_allowed_move_elim_round0_prompt_context(
                                round_batch=gain_batch_p0,
                                round0_prompt_input_ids=round0_prompt_input_ids,
                                round0_prompt_attention_mask=round0_prompt_attention_mask,
                                round0_prompt_position_ids=round0_prompt_position_ids,
                            )
                            if "response_mask" not in gain_batch_p0.batch.keys():
                                gain_batch_p0.batch["response_mask"] = compute_response_mask(gain_batch_p0)
                            with marked_timer("allowed_move_elim_gain_logprob_p0", timing_raw, color="blue"):
                                gain_logprob_p0 = self.actor_rollout_wg.compute_log_prob(gain_batch_p0)
                            logprob_p0_seq = self._sum_masked_token_log_probs(
                                token_log_probs=gain_logprob_p0.batch["old_log_probs"],
                                response_mask=gain_batch_p0.batch["response_mask"],
                            )

                            gain_seq = logprob_pi_seq - logprob_p0_seq
                            gain_np = gain_seq.detach().cpu().numpy().astype(np.float32)
                            logprob_pi_np = logprob_pi_seq.detach().cpu().numpy().astype(np.float32)
                            logprob_p0_np = logprob_p0_seq.detach().cpu().numpy().astype(np.float32)

                            batch.non_tensor_batch["allowed_move_elim_logprob_pi"] = logprob_pi_np
                            batch.non_tensor_batch["allowed_move_elim_logprob_p0"] = logprob_p0_np
                            batch.non_tensor_batch["allowed_move_elim_gain"] = gain_np

                            gain_keep_mask = gain_np <= float(allowed_move_elim_gain_threshold)
                            gain_total = int(gain_keep_mask.shape[0])
                            gain_kept = int(np.count_nonzero(gain_keep_mask))
                            gain_filtered = int(gain_total - gain_kept)

                            selection_metrics["selection_sampler/gain_threshold"] = float(allowed_move_elim_gain_threshold)
                            selection_metrics["selection_sampler/gain_filter_enabled"] = 1.0
                            selection_metrics["selection_sampler/gain_total_samples"] = int(gain_total)
                            selection_metrics["selection_sampler/gain_kept_samples"] = int(gain_kept)
                            selection_metrics["selection_sampler/gain_filtered_samples"] = int(gain_filtered)
                            selection_metrics["selection_sampler/gain_kept_frac"] = float(
                                gain_kept / float(max(1, gain_total))
                            )
                            selection_metrics["selection_sampler/gain_filtered_frac"] = float(
                                gain_filtered / float(max(1, gain_total))
                            )
                            if gain_total > 0:
                                selection_metrics["selection_sampler/gain_mean"] = float(np.mean(gain_np))
                                selection_metrics["selection_sampler/gain_std"] = float(np.std(gain_np))
                                selection_metrics["selection_sampler/gain_min"] = float(np.min(gain_np))
                                selection_metrics["selection_sampler/gain_p50"] = float(np.percentile(gain_np, 50))
                                selection_metrics["selection_sampler/gain_p90"] = float(np.percentile(gain_np, 90))
                                selection_metrics["selection_sampler/gain_p99"] = float(np.percentile(gain_np, 99))
                                selection_metrics["selection_sampler/gain_max"] = float(np.max(gain_np))

                            uid_arr_for_gain = batch.non_tensor_batch.get("uid", None)
                            if uid_arr_for_gain is not None and len(uid_arr_for_gain) == gain_total:
                                uid2idxs_gain: dict[str, list[int]] = defaultdict(list)
                                for i, uid in enumerate(uid_arr_for_gain):
                                    uid2idxs_gain[str(uid)].append(i)
                                kept_uid_set = {
                                    uid
                                    for uid, idxs in uid2idxs_gain.items()
                                    if any(bool(gain_keep_mask[i]) for i in idxs)
                                }
                                partially_filtered_uids = {
                                    uid
                                    for uid, idxs in uid2idxs_gain.items()
                                    if any(bool(gain_keep_mask[i]) for i in idxs)
                                    and any((not bool(gain_keep_mask[i])) for i in idxs)
                                }
                                selection_metrics["selection_sampler/gain_uid_groups_total"] = int(len(uid2idxs_gain))
                                selection_metrics["selection_sampler/gain_uid_groups_kept"] = int(len(kept_uid_set))
                                selection_metrics["selection_sampler/gain_uid_groups_all_filtered"] = int(
                                    len(uid2idxs_gain) - len(kept_uid_set)
                                )
                                selection_metrics["selection_sampler/gain_uid_groups_partially_filtered"] = int(
                                    len(partially_filtered_uids)
                                )

                            if gain_filtered > 0:
                                keep_traj_idxs = np.flatnonzero(gain_keep_mask).tolist()
                                if not keep_traj_idxs:
                                    raise ValueError(
                                        "allowed_move_elim gain filter rejected all samples for this step "
                                        f"(threshold={allowed_move_elim_gain_threshold}). "
                                        "Increase threshold or disable via allowed_move_elim.gain_threshold=inf."
                                    )
                                batch = batch[keep_traj_idxs]
                                print(
                                    f"[ALLOWED_MOVE_ELIM] gain_threshold={allowed_move_elim_gain_threshold} "
                                    f"filtered_samples={gain_filtered}/{gain_total} "
                                    f"(kept_samples={gain_kept})",
                                    flush=True,
                                )
                        else:
                            selection_metrics["selection_sampler/gain_filter_enabled"] = 0.0
                        timing_raw["allowed_move_elim/gain_filter_total"] += (
                            time.perf_counter() - gain_filter_start_time
                        )

                        if (
                            allowed_move_elim_stitch_round0_prompt_for_logprob
                            and (not allowed_move_elim_stitch_per_round_for_logprob)
                        ):
                            self._stitch_allowed_move_elim_round0_prompt_context(
                                round_batch=batch,
                                round0_prompt_input_ids=round0_prompt_input_ids,
                                round0_prompt_attention_mask=round0_prompt_attention_mask,
                                round0_prompt_position_ids=round0_prompt_position_ids,
                            )

                        loss_weight_padding_start_time = time.perf_counter()
                        prompt_idx_arr = batch.non_tensor_batch.get("allowed_move_elim_prompt_idx", None)
                        if prompt_idx_arr is None:
                            raise ValueError("allowed_move_elim missing allowed_move_elim_prompt_idx for loss weighting.")
                        uid_arr = batch.non_tensor_batch.get("uid", None)
                        if uid_arr is None:
                            raise ValueError("allowed_move_elim missing uid for loss weighting.")

                        uid_group_counts_by_prompt = self._allowed_move_elim_count_unique_uids_by_prompt(
                            prompt_idx_arr=prompt_idx_arr,
                            uid_arr=uid_arr,
                        )

                        # Loss normalization: keep per-original-prompt total loss weight ~constant even though
                        # allowed_move_elim expands a single prompt into multiple rounds (and thus multiple
                        # trajectories).
                        #
                        # - Default (uid_mode=per_round): one uid per (prompt, round), so unique(uid) per prompt is
                        #   the number of rounds used/kept, matching historical behavior.
                        # - New (uid_mode=per_prompt): one uid per prompt across all rounds, so we must normalize by
                        #   the number of rounds (unique allowed_move_elim_round) instead of unique(uid).
                        if allowed_move_elim_uid_mode == "per_round":
                            loss_denom_by_prompt = uid_group_counts_by_prompt
                        elif allowed_move_elim_uid_mode == "per_prompt":
                            round_arr = batch.non_tensor_batch.get("allowed_move_elim_round", None)
                            if round_arr is None:
                                raise ValueError(
                                    "allowed_move_elim missing allowed_move_elim_round for loss weighting "
                                    "when uid_mode='per_prompt'."
                                )
                            loss_denom_by_prompt = self._allowed_move_elim_count_unique_rounds_by_prompt(
                                prompt_idx_arr=prompt_idx_arr,
                                round_arr=round_arr,
                            )
                        else:
                            raise ValueError(
                                "allowed_move_elim.uid_mode must be one of {'per_round','per_prompt'} "
                                f"(got {allowed_move_elim_uid_mode!r})."
                            )

                        seq_weights = np.array(
                            [1.0 / float(loss_denom_by_prompt.get(int(i), 1)) for i in prompt_idx_arr],
                            dtype=np.float32,
                        )
                        batch.batch["loss_weights"] = torch.from_numpy(seq_weights).unsqueeze(-1)
                        if seq_weights.size:
                            selection_metrics["selection_sampler/avg_loss_weight"] = float(np.mean(seq_weights))
                            selection_metrics["selection_sampler/min_loss_weight"] = float(np.min(seq_weights))
                            selection_metrics["selection_sampler/max_loss_weight"] = float(np.max(seq_weights))

                        # Keep post-filter group-count stats for debugging.
                        if loss_denom_by_prompt:
                            kept_counts = np.asarray(list(loss_denom_by_prompt.values()), dtype=np.float32)
                            selection_metrics["selection_sampler/avg_groups_per_prompt_kept"] = float(
                                np.mean(kept_counts)
                            )
                            selection_metrics["selection_sampler/min_groups_per_prompt_kept"] = int(
                                np.min(kept_counts)
                            )
                            selection_metrics["selection_sampler/max_groups_per_prompt_kept"] = int(
                                np.max(kept_counts)
                            )
                            selection_metrics["selection_sampler/total_groups_kept"] = int(len(set(str(u) for u in uid_arr)))

                        # Also log the *actual* uid-group counts per prompt (1 when uid_mode='per_prompt').
                        if uid_group_counts_by_prompt:
                            uid_counts = np.asarray(list(uid_group_counts_by_prompt.values()), dtype=np.float32)
                            selection_metrics["selection_sampler/avg_uid_groups_per_prompt_kept"] = float(
                                np.mean(uid_counts)
                            )
                            selection_metrics["selection_sampler/min_uid_groups_per_prompt_kept"] = int(
                                np.min(uid_counts)
                            )
                            selection_metrics["selection_sampler/max_uid_groups_per_prompt_kept"] = int(
                                np.max(uid_counts)
                            )

                        # GRPO effective batch size (allowed_move_elim path). Use existing rewards if present,
                        # otherwise fall back to raw token_level_scores from per-round reward computation.
                    adv_name = self.config.algorithm.adv_estimator
                    is_grpo = (
                        adv_name == AdvantageEstimator.GRPO
                        if isinstance(adv_name, AdvantageEstimator)
                        else str(adv_name).lower() == "grpo"
                    )
                    if is_grpo:
                        if "token_level_rewards" not in batch.batch and "token_level_scores" in batch.batch:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        effective_groups, total_groups = self._compute_grpo_effective_batch(batch)
                        if total_groups > 0:
                            metrics["grpo/effective_batch_size"] = int(effective_groups)
                            metrics["grpo/effective_batch_frac"] = float(effective_groups / total_groups)
                            metrics["grpo/group_count"] = int(total_groups)

                    # NOTE: In allowed_move_elim, each prompt can contribute a variable number of GRPO groups.
                    # With rollout.n=8 and multi-node runs (e.g., 4 nodes × 4 GPUs = world_size=16), the total
                    # number of trajectories is always divisible by rollout.n but may be *not* divisible by the
                    # DP world size. This breaks downstream equal-size partitioning (e.g., balance_batch).
                    #
                    # Pad to lcm(world_size, rollout.n) and zero-out loss weights for the padded portion so it
                    # has no effect on optimization.
                    world_size = int(getattr(self.actor_rollout_wg, "world_size", 1) or 1)
                    rollout_n = int(getattr(self.config.actor_rollout_ref.rollout, "n", 1) or 1)
                    pad_divisor = int(np.lcm(max(1, world_size), max(1, rollout_n)))
                    if pad_divisor > 1:
                        batch_padded, pad_size = pad_dataproto_to_divisor(batch, pad_divisor)
                        if pad_size:
                            # Assign fresh uids for the padded portion so GRPO group stats for real groups
                            # are unchanged (avoids changing group-size-dependent torch.std behavior).
                            try:
                                uids = np.asarray(batch_padded.non_tensor_batch.get("uid")).copy()
                                pad_start = len(batch_padded) - pad_size
                                if rollout_n > 0:
                                    for i in range(pad_start, len(batch_padded), rollout_n):
                                        uids[i : i + rollout_n] = str(uuid.uuid4())
                                else:
                                    for i in range(pad_start, len(batch_padded)):
                                        uids[i] = str(uuid.uuid4())
                                batch_padded.non_tensor_batch["uid"] = uids
                            except Exception:
                                pass

                            if "loss_weights" in batch_padded.batch:
                                batch_padded.batch["loss_weights"][-pad_size:] = 0.0

                            selection_metrics["selection_sampler/pad_size"] = int(pad_size)
                            selection_metrics["selection_sampler/pad_divisor"] = int(pad_divisor)
                            selection_metrics["selection_sampler/pad_world_size"] = int(world_size)
                            selection_metrics["selection_sampler/pad_rollout_n"] = int(rollout_n)
                        batch = batch_padded
                    timing_raw["allowed_move_elim/loss_weight_and_padding"] += (
                        time.perf_counter() - loss_weight_padding_start_time
                    )

                    # Forced-prefix diagnostics on final accepted batch.
                    if "forced_token_mask" in batch.batch.keys():
                        forced_token_mask = batch.batch["forced_token_mask"].to(torch.bool)
                        forced_seq = torch.any(forced_token_mask, dim=-1)
                        metrics["forced_prefix/forced_seq_frac"] = forced_seq.float().mean().item()
                        denom = batch.batch["response_mask"].to(torch.float32).sum().clamp(min=1.0)
                        metrics["forced_prefix/forced_token_frac"] = (
                            forced_token_mask.to(torch.float32).sum() / denom
                        ).item()

                        forced_cfg = self.config.get("forced_prefix", None) or {}
                        if forced_cfg.get("enable", True):
                            metrics["forced_prefix/apply_prob"] = float(self._compute_forced_prefix_apply_prob(forced_cfg))

                        template = forced_cfg.get("prefix_template", None)
                        if template:
                            tpl = str(template).strip()
                            if (tpl.startswith('"') and tpl.endswith('"')) or (tpl.startswith("'") and tpl.endswith("'")):
                                tpl = tpl[1:-1]
                            prefix_start = tpl.split("{move}")[0] if "{move}" in tpl else tpl
                            prefix_start = prefix_start.strip()
                            if prefix_start:
                                prefix_ids = self.tokenizer.encode(prefix_start, add_special_tokens=False)
                            else:
                                prefix_ids = []
                            if prefix_ids:
                                responses = batch.batch["responses"]
                                if responses.size(1) >= len(prefix_ids):
                                    prefix_tensor = torch.tensor(
                                        prefix_ids, device=responses.device, dtype=responses.dtype
                                    )
                                    starts_with_prefix = torch.all(
                                        responses[:, : len(prefix_ids)] == prefix_tensor, dim=-1
                                    )
                                    valid_seq = torch.any(batch.batch["response_mask"].to(torch.bool), dim=-1)
                                    free_valid = (~forced_seq) & valid_seq
                                    denom = free_valid.to(torch.float32).sum().clamp(min=1.0)
                                    metrics["forced_prefix/free_self_prefix_frac"] = (
                                        (starts_with_prefix & free_valid).to(torch.float32).sum() / denom
                                    ).item()

                        if "uid" in batch.non_tensor_batch:
                            uids = batch.non_tensor_batch["uid"]
                            _, inv = np.unique(uids, return_inverse=True)
                            counts = np.bincount(inv)
                            forced_counts = np.bincount(inv, weights=forced_seq.cpu().numpy().astype(np.int32))
                            mixed = (forced_counts > 0) & (forced_counts < counts)
                            metrics["forced_prefix/mixed_group_frac"] = float(mixed.mean()) if mixed.size else 0.0
                            metrics["forced_prefix/forced_per_group_mean"] = (
                                float(forced_counts.mean()) if forced_counts.size else 0.0
                            )

                        if "forced_prefix_value" in batch.non_tensor_batch:
                            vals = batch.non_tensor_batch["forced_prefix_value"].astype(np.float32, copy=False)
                            forced_np = forced_seq.cpu().numpy()
                            if forced_np.any():
                                metrics["forced_prefix/value_mean"] = float(vals[forced_np].mean())

                    metrics.update(selection_metrics)
                else:
                    # add uid to batch
                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )

                    gen_batch = self._get_gen_batch(new_batch)

                    # pass global_steps to trace
                    gen_batch.meta_info["global_steps"] = self.global_steps
                    gen_batch_output = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )
                    gen_batch_output = self._apply_forced_prefix(gen_batch_output, new_batch)

                    with marked_timer("step", timing_raw):
                        # generate a batch
                        with marked_timer("gen", timing_raw, color="red"):
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            if self.reward_fn is None:
                                raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                if not self.async_rollout_mode:
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                                else:
                                    gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                new_batch = new_batch.union(gen_baseline_output)
                                # compute reward model score on new_batch
                                rm_scores = None
                                if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                    rm_scores = self.rm_wg.compute_rm_score(new_batch)
                                    new_batch = new_batch.union(rm_scores)
                                reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                keys_to_pop = set(gen_baseline_output.batch.keys())
                                if rm_scores is not None:
                                    keys_to_pop.update(rm_scores.batch.keys())
                                new_batch.pop(batch_keys=list(keys_to_pop))

                                new_batch.batch["reward_baselines"] = reward_baseline_tensor

                                del rm_scores, gen_baseline_batch, gen_baseline_output
                        # repeat to align with repeated responses in rollout
                        new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        new_batch = new_batch.union(gen_batch_output)

                        if "response_mask" not in new_batch.batch.keys():
                            new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                        # Forced-prefix diagnostics (cheap; helps verify ratio/temp wiring).
                        if "forced_token_mask" in new_batch.batch.keys():
                            forced_token_mask = new_batch.batch["forced_token_mask"].to(torch.bool)
                            forced_seq = torch.any(forced_token_mask, dim=-1)
                            metrics["forced_prefix/forced_seq_frac"] = forced_seq.float().mean().item()
                            denom = new_batch.batch["response_mask"].to(torch.float32).sum().clamp(min=1.0)
                            metrics["forced_prefix/forced_token_frac"] = (
                                forced_token_mask.to(torch.float32).sum() / denom
                            ).item()

                            forced_cfg = self.config.get("forced_prefix", None) or {}
                            if forced_cfg.get("enable", True):
                                metrics["forced_prefix/apply_prob"] = float(self._compute_forced_prefix_apply_prob(forced_cfg))

                            # Self-prefixing diagnostics: among *free* rollouts, how often does the model
                            # naturally start with the forced-prefix template?
                            template = forced_cfg.get("prefix_template", None)
                            if template:
                                tpl = str(template).strip()
                                if (tpl.startswith('"') and tpl.endswith('"')) or (tpl.startswith("'") and tpl.endswith("'")):
                                    tpl = tpl[1:-1]
                                prefix_start = tpl.split("{move}")[0] if "{move}" in tpl else tpl
                                prefix_start = prefix_start.strip()
                                if prefix_start:
                                    prefix_ids = self.tokenizer.encode(prefix_start, add_special_tokens=False)
                                else:
                                    prefix_ids = []
                                if prefix_ids:
                                    responses = new_batch.batch["responses"]
                                    if responses.size(1) >= len(prefix_ids):
                                        prefix_tensor = torch.tensor(
                                            prefix_ids, device=responses.device, dtype=responses.dtype
                                        )
                                        starts_with_prefix = torch.all(
                                            responses[:, : len(prefix_ids)] == prefix_tensor, dim=-1
                                        )
                                        valid_seq = torch.any(new_batch.batch["response_mask"].to(torch.bool), dim=-1)
                                        free_valid = (~forced_seq) & valid_seq
                                        denom = free_valid.to(torch.float32).sum().clamp(min=1.0)
                                        metrics["forced_prefix/free_self_prefix_frac"] = (
                                            (starts_with_prefix & free_valid).to(torch.float32).sum() / denom
                                        ).item()

                            if "uid" in new_batch.non_tensor_batch:
                                uids = new_batch.non_tensor_batch["uid"]
                                _, inv = np.unique(uids, return_inverse=True)
                                counts = np.bincount(inv)
                                forced_counts = np.bincount(inv, weights=forced_seq.cpu().numpy().astype(np.int32))
                                mixed = (forced_counts > 0) & (forced_counts < counts)
                                metrics["forced_prefix/mixed_group_frac"] = float(mixed.mean()) if mixed.size else 0.0
                                metrics["forced_prefix/forced_per_group_mean"] = (
                                    float(forced_counts.mean()) if forced_counts.size else 0.0
                                )

                            if "forced_prefix_value" in new_batch.non_tensor_batch:
                                vals = new_batch.non_tensor_batch["forced_prefix_value"].astype(np.float32, copy=False)
                                forced_np = forced_seq.cpu().numpy()
                                if forced_np.any():
                                    metrics["forced_prefix/value_mean"] = float(vals[forced_np].mean())
                        # NOTE: For non-filtered training, reward computation is performed after optional
                        # DP balancing so async reward (if enabled) matches the final ordering.
                        if filter_groups_enabled:
                            with marked_timer("reward", timing_raw, color="yellow"):
                                # compute reward model score
                                if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                    reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                                    new_batch = new_batch.union(reward_tensor)

                                # Group filtering requires reward values immediately; force sync.
                                reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)
                                new_batch.batch["token_level_scores"] = reward_tensor
                                new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
                                if reward_extra_infos_dict:
                                    new_batch.non_tensor_batch.update(
                                        {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                    )

                        if filter_groups_enabled:
                            # Filter groups whose per-rollout metric has zero variance (unqualified).
                            metric_name = str(filter_metric_name)
                            if metric_name == "seq_final_reward":
                                new_batch.non_tensor_batch["seq_final_reward"] = (
                                    new_batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
                                )
                            elif metric_name == "seq_reward":
                                new_batch.non_tensor_batch["seq_reward"] = (
                                    new_batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()
                                )

                            if metric_name not in new_batch.non_tensor_batch:
                                raise KeyError(
                                    f"filter_groups.metric={metric_name!r} not found in reward_extra_info/non_tensor_batch. "
                                    "Ensure the reward function returns this key, or use seq_reward/seq_final_reward."
                                )

                            prompt_uid2metric_vals = defaultdict(list)
                            for uid, metric_val in zip(
                                new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                            ):
                                prompt_uid2metric_vals[uid].append(metric_val)

                            kept_prompt_uids = [
                                uid
                                for uid, vals in prompt_uid2metric_vals.items()
                                if (np.std(vals) > 0) or (len(vals) == 1)
                            ]
                            rejected_prompt_uids = [
                                uid for uid in prompt_uid2metric_vals.keys() if uid not in set(kept_prompt_uids)
                            ]
                            num_prompt_in_batch += len(kept_prompt_uids)
                            num_gen_batches += 1

                            # Rejection diagnostics (group-level): why were groups unqualified?
                            # Use `penalty_reason` (if present) as a coarse attribution signal.
                            if "penalty_reason" in new_batch.non_tensor_batch:
                                uid2penalties = defaultdict(list)
                                for uid, pr in zip(
                                    new_batch.non_tensor_batch["uid"],
                                    new_batch.non_tensor_batch["penalty_reason"],
                                    strict=True,
                                ):
                                    uid2penalties[uid].append(str(pr))
                                for uid in rejected_prompt_uids:
                                    prs = uid2penalties.get(uid, [])
                                    # Empty string means "no penalty" (valid in-subset sample).
                                    uniq = set(prs)
                                    if uniq == {""}:
                                        filter_rejected_groups_by_penalty["all_valid"] += 1
                                    elif len(uniq) == 1:
                                        filter_rejected_groups_by_penalty[next(iter(uniq)) or "<empty>"] += 1
                                    else:
                                        filter_rejected_groups_by_penalty["mixed"] += 1
                            filter_rejected_groups_total += int(len(rejected_prompt_uids))

                            # Optional: dump rejected prompt-groups (zero-variance) for offline analysis.
                            rejected_log_dir = self.config.trainer.get("rejected_rollout_data_dir", None)
                            rejected_log_dir = str(rejected_log_dir).strip() if rejected_log_dir is not None else ""
                            if rejected_log_dir and rejected_prompt_uids:
                                # 1) Group-level summaries (one JSONL record per rejected uid).
                                max_summary_groups = int(
                                    self.config.trainer.get("rejected_group_summary_max_groups_per_step", 0) or 0
                                )
                                if max_summary_groups != 0:
                                    if max_summary_groups < 0:
                                        summary_uids = list(rejected_prompt_uids)
                                    else:
                                        remaining = max(0, max_summary_groups - filter_logged_rejected_group_summaries)
                                        summary_uids = list(rejected_prompt_uids)[:remaining]
                                    if summary_uids:
                                        summary_path = os.path.join(
                                            rejected_log_dir,
                                            "rejected_group_summaries",
                                            f"{self.global_steps}_gen{num_gen_batches}.jsonl",
                                        )
                                        filter_logged_rejected_group_summaries += self._dump_filter_groups_rejected_group_summaries(
                                            batch=new_batch,
                                            rejected_prompt_uids=summary_uids,
                                            metric_name=metric_name,
                                            filename=summary_path,
                                            gen_batch_index=int(num_gen_batches),
                                        )

                                # 2) Rollout-level samples (prompt+output) for a bounded number of rejected uids.
                                max_sample_groups = int(
                                    self.config.trainer.get("rejected_rollout_max_groups_per_step", 0) or 0
                                )
                                if max_sample_groups != 0:
                                    if max_sample_groups < 0:
                                        sample_uids = list(rejected_prompt_uids)
                                    else:
                                        remaining = max(0, max_sample_groups - filter_logged_rejected_rollout_samples)
                                        sample_uids = list(rejected_prompt_uids)[:remaining]
                                    if sample_uids:
                                        sample_path = os.path.join(
                                            rejected_log_dir,
                                            "rejected_rollout_samples",
                                            f"{self.global_steps}_gen{num_gen_batches}.jsonl",
                                        )
                                        filter_logged_rejected_rollout_samples += self._dump_filter_groups_rejected_rollout_samples(
                                            batch=new_batch,
                                            rejected_prompt_uids=sample_uids,
                                            metric_name=metric_name,
                                            filename=sample_path,
                                            gen_batch_index=int(num_gen_batches),
                                        )

                            kept_traj_idxs = [
                                idx
                                for idx, traj_uid in enumerate(new_batch.non_tensor_batch["uid"])
                                if traj_uid in set(kept_prompt_uids)
                            ]
                            new_batch = new_batch[kept_traj_idxs]
                            batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                            prompt_bsz = int(self.config.data.train_batch_size)
                            if num_prompt_in_batch < prompt_bsz:
                                print(
                                    f"[FILTER_GROUPS] kept_prompts={num_prompt_in_batch} < train_batch_size={prompt_bsz}"
                                )
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f"[FILTER_GROUPS] num_gen_batches={num_gen_batches} keep generating...")
                                    continue
                                raise ValueError(
                                    f"filter_groups: num_gen_batches={num_gen_batches} >= max_num_gen_batches={max_num_gen_batches}. "
                                    "Generated too many batches without enough qualified groups."
                                )

                            # Align to an exact number of prompt groups for GRPO (train_batch_size).
                            traj_bsz = int(self.config.data.train_batch_size) * int(
                                self.config.actor_rollout_ref.rollout.n
                            )
                            batch = batch[:traj_bsz]
                        else:
                            batch = new_batch

                # Balance tokens across DP ranks once we have the final batch (filter_groups may concatenate).
                if self.config.trainer.balance_batch:
                    with marked_timer("balance_batch", timing_raw):
                        self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # Reward (may be async) — after balancing for consistent ordering.
                reward_extra_infos_dict = {}
                if not filter_groups_enabled and not allowed_move_elim_enabled:
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            batch.batch["token_level_scores"] = reward_tensor
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                # recompute old_log_probs
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                    metrics.update(old_log_prob_metrics)
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)

                    if "rollout_log_probs" in batch.batch.keys():
                        # TODO: we may want to add diff of probs too.
                        from verl.utils.debug.metrics import calculate_debug_metrics

                        metrics.update(calculate_debug_metrics(batch))

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw, color="brown"):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    if (
                        self.config.reward_model.launch_reward_fn_async
                        and not filter_groups_enabled
                        and not allowed_move_elim_enabled
                    ):
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                    if filter_groups_enabled:
                        # Per-gen-batch reward_extra_infos_dict lengths do not match concatenated/sliced batches.
                        reward_extra_infos_dict = {}

                    # Stratified training "accuracy" (mean reward) by forced vs free rollouts.
                    # Note: `token_level_scores` is the *raw* reward function output (±1 for chess),
                    # i.e. before any optional in-reward KL penalty.
                    if "forced_token_mask" in batch.batch:
                        forced_seq = torch.any(batch.batch["forced_token_mask"].to(torch.bool), dim=-1)
                        free_seq = ~forced_seq
                        seq_scores = batch.batch["token_level_scores"].sum(dim=-1).detach()
                        if torch.any(free_seq):
                            metrics["forced_prefix/train_reward_mean_free"] = seq_scores[free_seq].mean().float().item()
                        else:
                            metrics["forced_prefix/train_reward_mean_free"] = 0.0
                        if torch.any(forced_seq):
                            metrics["forced_prefix/train_reward_mean_forced"] = (
                                seq_scores[forced_seq].mean().float().item()
                            )
                        else:
                            metrics["forced_prefix/train_reward_mean_forced"] = 0.0

                    # compute rewards. apply_kl_penalty if available
                    _adv_name_for_reward = self.config.algorithm.adv_estimator
                    _is_distill = (
                        _adv_name_for_reward == AdvantageEstimator.DISTILL
                        if isinstance(_adv_name_for_reward, AdvantageEstimator)
                        else str(_adv_name_for_reward).lower() == "distill"
                    )
                    if _is_distill:
                        # On-policy distillation (Thinking Machines blog):
                        # per-token reward = log p_teacher - log p_student.
                        # `ref_log_prob` holds teacher logprobs because the ref policy
                        # is configured (via actor_rollout_ref.ref.model.path) to load
                        # the teacher checkpoint; `old_log_probs` is the student's
                        # rollout-time logprob.
                        response_mask = batch.batch["response_mask"]
                        teacher_logp = batch.batch["ref_log_prob"]
                        student_logp = batch.batch["old_log_probs"]
                        distill_signal = (teacher_logp - student_logp) * response_mask
                        batch.batch["token_level_rewards"] = distill_signal
                        # token_level_scores is consumed by metrics/logging downstream;
                        # mirror the distill signal so dashboards remain populated even
                        # when there's no separate task reward.
                        if "token_level_scores" not in batch.batch:
                            batch.batch["token_level_scores"] = distill_signal
                        # Surface a few headline metrics:
                        seq_signal = distill_signal.sum(dim=-1).detach()
                        seq_len = response_mask.sum(dim=-1).clamp_min(1).detach()
                        per_token = seq_signal / seq_len
                        metrics["distill/per_token_logp_gap/mean"] = float(per_token.mean().item())
                        metrics["distill/per_token_logp_gap/min"] = float(per_token.min().item())
                        metrics["distill/per_token_logp_gap/max"] = float(per_token.max().item())
                    elif self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # GRPO effective batch size: groups where std(reward) != 0.
                    adv_name = self.config.algorithm.adv_estimator
                    is_grpo = (
                        adv_name == AdvantageEstimator.GRPO
                        if isinstance(adv_name, AdvantageEstimator)
                        else str(adv_name).lower() == "grpo"
                    )
                    if is_grpo:
                        effective_groups, total_groups = self._compute_grpo_effective_batch(batch)
                        if total_groups > 0:
                            metrics["grpo/effective_batch_size"] = int(effective_groups)
                            metrics["grpo/effective_batch_frac"] = float(effective_groups / total_groups)
                            metrics["grpo/group_count"] = int(total_groups)

                    # Compute rollout importance sampling weights centrally (once per batch)
                    # This corrects for mismatch between rollout policy and training policy
                    # Also computes mismatch metrics (KL, PPL, etc.)
                    batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                    # IS and mismatch metrics already have mismatch/ prefix
                    metrics.update(is_metrics)

                    # compute advantages, executed on the driver process
                    norm_adv_by_std_in_grpo = self.config.algorithm.get(
                        "norm_adv_by_std_in_grpo", True
                    )  # GRPO adv normalization factor

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )
                    if is_grpo:
                        metrics.update(self._compute_diversity_metrics(batch))

                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        batch.meta_info["global_steps"] = int(self.global_steps)
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    diversity_dump_keys = (
                        "diversity_group_T",
                        "diversity_group_top_freq",
                        "diversity_group_collision_count",
                        "diversity_a_base",
                        "diversity_a_div",
                        "diversity_group_size",
                        "diversity_group_valid_count",
                        "diversity_group_L_min",
                        "diversity_group_L_max",
                        "diversity_method",
                        "diversity_enabled",
                        "diversity_lambda_coeff",
                        "diversity_include_base_advantage",
                    )
                    for k in diversity_dump_keys:
                        if k in batch.non_tensor_batch and k not in reward_extra_infos_dict:
                            v = batch.non_tensor_batch[k]
                            try:
                                reward_extra_infos_dict[k] = v.tolist()
                            except Exception:
                                reward_extra_infos_dict[k] = list(v)
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Optional full-game chess evaluation (separate knob from `test_freq`).
                full_eval_freq = int(self.config.trainer.get("full_eval_freq", -1) or -1)
                if full_eval_freq > 0 and (is_last_step or self.global_steps % full_eval_freq == 0):
                    with marked_timer("full_game_eval", timing_raw, color="green"):
                        full_eval_metrics = self._run_full_game_eval()
                    metrics.update(full_eval_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                timing_raw["iteration"] = time.perf_counter() - iteration_start_time
                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                if "advantages" not in batch.batch or "returns" not in batch.batch:
                    # Safety net: ensure advantages/returns exist before metrics/logging.
                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                if "global_token_num" not in batch.meta_info:
                    batch.meta_info["global_token_num"] = (
                        torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    )
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                if filter_groups_enabled:
                    metrics["train/num_gen_batches"] = int(num_gen_batches)
                    metrics["filter_groups/rejected_groups_total"] = int(filter_rejected_groups_total)
                    metrics["filter_groups/kept_groups_total"] = int(num_prompt_in_batch)
                    metrics["filter_groups/rejected_group_summaries_logged"] = int(filter_logged_rejected_group_summaries)
                    metrics["filter_groups/rejected_rollout_samples_logged"] = int(filter_logged_rejected_rollout_samples)
                    for k, v in sorted(filter_rejected_groups_by_penalty.items()):
                        metrics[f"filter_groups/rejected_by_penalty/{k}"] = int(v)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if filter_groups_enabled:
                    # Reset dynamic sampling state for the next update step.
                    batch = None
                    num_prompt_in_batch = 0
                    num_gen_batches = 0

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    # Ensure any queued HF uploads (started after local checkpoint saves) are finished
                    # before a clean training exit. This keeps auto-resume consistent.
                    self._flush_hf_uploads(block=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
