from __future__ import annotations

from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def _trainer_config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "verl" / "trainer" / "config"


def train_chess(overrides: Sequence[str] | None = None) -> None:
    """
    Start PPO training on the chess environment using Hydra config ``chess_agent``.

    Example::

        train_chess(["actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct", "trainer.n_gpus_per_node=1"])

    CLI equivalent::

        python -m verl.trainer.main_chess_agent actor_rollout_ref.model.path=...
    """
    GlobalHydra.instance().clear()
    config_dir = _trainer_config_dir()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="chess_agent", overrides=list(overrides or ()))
    from verl.trainer.main_ppo import run_ppo

    run_ppo(cfg)


if __name__ == "__main__":
    import sys

    train_chess(overrides=sys.argv[1:])
