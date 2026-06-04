# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        from verl.trainer.ppo.ray_trainer import Role

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # TODO Here you can use the new registration method to support dynamic registration of roles
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import RewardModelWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss / KL reward / on-policy distill needs it.

        For on-policy distillation the ref slot is repurposed to host the teacher,
        so the worker must be created even when both kl flags are False.
        """
        from verl.trainer.ppo.ray_trainer import Role
        from verl.trainer.ppo.core_algos import AdvantageEstimator

        adv_est = config.algorithm.adv_estimator
        is_distill = (
            adv_est == AdvantageEstimator.DISTILL
            if isinstance(adv_est, AdvantageEstimator)
            else str(adv_est).lower() == "distill"
        )

        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
            or is_distill
        ):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Deterministic output directories (default): `outputs/{config_hash}/...`
        # - Prevents accidental resume across different configs that share a long/unstable experiment name.
        # - Aligns local output naming with HF resume-by-config-hash (same `config_hash` computation).
        #
        # Override:
        # - Set `VERL_BASE_DIR=/abs/or/rel/path` to pin outputs elsewhere.
        #
        # Note: use `HYDRA_ORIGINAL_CWD` to remain stable even when Hydra changes the working directory.
        from pathlib import Path

        from omegaconf import open_dict

        dummy = RayPPOTrainer.__new__(RayPPOTrainer)
        dummy.config = config
        dummy._hf_config_hash = None
        dummy._hf_dataset_fingerprint = None
        config_hash = dummy._get_or_compute_config_hash()

        base_dir_env = os.environ.get("VERL_BASE_DIR", "").strip()
        base_root = Path(os.environ.get("HYDRA_ORIGINAL_CWD", os.getcwd())).resolve()
        # Decide how to set output dirs:
        # - If `VERL_BASE_DIR` is set, treat it as an explicit output-root override and derive the
        #   standard subdirs under it.
        # - Otherwise, if the config still uses Hydra's default checkpoint layout
        #   (`checkpoints/{project}/{experiment}`), override it to the hash-based default:
        #     outputs/{config_hash}/...
        # - Otherwise, keep user-provided dirs; if rollout/validation dump dirs are unset, fill them
        #   relative to the checkpoint dir so logging remains enabled.
        project_name = str(config.trainer.get("project_name", "verl_examples"))
        experiment_name = str(config.trainer.get("experiment_name", "run"))
        default_ckpt_dir = os.path.normpath(f"checkpoints/{project_name}/{experiment_name}")
        current_ckpt_dir = os.path.normpath(str(config.trainer.get("default_local_dir", "")))
        current_rollout_dir = config.trainer.get("rollout_data_dir", None)
        current_val_dir = config.trainer.get("validation_data_dir", None)
        current_rejected_dir = config.trainer.get("rejected_rollout_data_dir", None)
        try:
            rejected_summary_max = int(config.trainer.get("rejected_group_summary_max_groups_per_step", 0) or 0)
            rejected_samples_max = int(config.trainer.get("rejected_rollout_max_groups_per_step", 0) or 0)
            want_rejected_logs = (rejected_summary_max != 0) or (rejected_samples_max != 0)
        except Exception:
            want_rejected_logs = False
        override_all_dirs = bool(base_dir_env) or (current_ckpt_dir == default_ckpt_dir)

        if override_all_dirs:
            if base_dir_env:
                base_dir = Path(os.path.expanduser(base_dir_env))
                if not base_dir.is_absolute():
                    base_dir = (base_root / base_dir).resolve()
            else:
                base_dir = (base_root / "outputs" / config_hash).resolve()

            rollout_dir = base_dir / "rollout" / "rollout_logs"
            validation_dir = base_dir / "rollout" / "validation_logs"
            rejected_dir = base_dir / "rollout" / "rejected_rollout_logs"
            ckpt_dir = base_dir / "checkpoints"

            rollout_dir.mkdir(parents=True, exist_ok=True)
            validation_dir.mkdir(parents=True, exist_ok=True)
            if want_rejected_logs and current_rejected_dir is None:
                rejected_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        with open_dict(config):
            config.trainer.config_hash = config_hash
            if override_all_dirs:
                config.trainer.rollout_data_dir = str(rollout_dir)
                config.trainer.validation_data_dir = str(validation_dir)
                config.trainer.default_local_dir = str(ckpt_dir)
                if want_rejected_logs and current_rejected_dir is None:
                    config.trainer.rejected_rollout_data_dir = str(rejected_dir)
            else:
                # If the user pinned a checkpoint dir but didn't enable JSONL dumps, keep their ckpt dir
                # and default the dump dirs relative to it so runs remain debuggable.
                ckpt_dir_str = str(config.trainer.get("default_local_dir", "") or "")
                if ckpt_dir_str:
                    ckpt_path = Path(os.path.expanduser(ckpt_dir_str))
                    if not ckpt_path.is_absolute():
                        ckpt_path = (base_root / ckpt_path).resolve()
                    run_root = ckpt_path.parent
                else:
                    run_root = base_root

                if current_rollout_dir is None:
                    rollout_dir2 = run_root / "rollout" / "rollout_logs"
                    rollout_dir2.mkdir(parents=True, exist_ok=True)
                    config.trainer.rollout_data_dir = str(rollout_dir2)
                if current_val_dir is None:
                    val_dir2 = run_root / "rollout" / "validation_logs"
                    val_dir2.mkdir(parents=True, exist_ok=True)
                    config.trainer.validation_data_dir = str(val_dir2)
                if want_rejected_logs and current_rejected_dir is None:
                    rej_dir2 = run_root / "rollout" / "rejected_rollout_logs"
                    rej_dir2.mkdir(parents=True, exist_ok=True)
                    config.trainer.rejected_rollout_data_dir = str(rej_dir2)

        print(f"[paths] config_hash={config_hash}")
        if override_all_dirs:
            print(f"[paths] base_dir={base_dir}")
        else:
            print("[paths] Using user-configured checkpoint dir (rollout/val dirs may be auto-filled).")
        print(f"[paths] trainer.rollout_data_dir={config.trainer.get('rollout_data_dir', None)}")
        print(f"[paths] trainer.validation_data_dir={config.trainer.get('validation_data_dir', None)}")
        print(f"[paths] trainer.rejected_rollout_data_dir={config.trainer.get('rejected_rollout_data_dir', None)}")
        print(f"[paths] trainer.default_local_dir={config.trainer.get('default_local_dir', None)}")

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        model_path = config.actor_rollout_ref.model.path
        local_path = copy_to_local(model_path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer_path = config.actor_rollout_ref.model.get("tokenizer_path", None) or model_path
        if str(tokenizer_path) == str(model_path):
            local_tokenizer_path = local_path
        else:
            local_tokenizer_path = copy_to_local(
                tokenizer_path,
                use_shm=config.actor_rollout_ref.model.get("use_shm", False),
            )
        tokenizer = hf_tokenizer(local_tokenizer_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_tokenizer_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    self_play_cfg = getattr(data_config, "self_play", None)
    self_play_enabled = bool(getattr(self_play_cfg, "enable", False)) if self_play_cfg is not None else False

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    # NOTE: when `data.self_play.enable=True`, the trainer ignores the dataloader payload for training
    # steps, but validation still needs a real RLHF dataset (parquet-backed, with tokenized prompts).
    # Treat `data.custom_cls` as *train-only* in this mode so `val_dataset` remains valid.
    if (not is_train) and self_play_enabled:
        dataset_cls = RLHFDataset
    elif "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
