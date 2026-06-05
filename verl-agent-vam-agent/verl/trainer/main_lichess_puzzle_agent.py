"""Entrypoint for HGPO / GRPO training over Chess-R1 / SF μ-graded puzzle data
(chess_self_play's `lichess_puzzle/Curriculum` env), with optional VAM.

Same wiring as main_chesslesson_agent / main_chess_agent — defers to
`run_ppo` and lets the env_manager dispatch to `lichess_puzzle` based on
`env.env_name`.
"""
import hydra

from verl.trainer.main_ppo import run_ppo


@hydra.main(config_path="config", config_name="lichess_puzzle_agent", version_base=None)
def main(config):
    run_ppo(config)


if __name__ == "__main__":
    main()
