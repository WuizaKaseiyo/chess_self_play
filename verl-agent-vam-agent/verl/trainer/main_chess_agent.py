import hydra

from verl.trainer.main_ppo import run_ppo


@hydra.main(config_path="config", config_name="chess_agent", version_base=None)
def main(config):
    run_ppo(config)


if __name__ == "__main__":
    main()
