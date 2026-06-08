import hydra

# Route to the HGPO trainer (recipe/hgpo/main_hgpo.py) which has hgpo support
# in its RayPPOTrainer. The vanilla verl.trainer.main_ppo's RayPPOTrainer
# raises NotImplementedError when adv_estimator=hgpo.
from recipe.hgpo.main_hgpo import run_ppo


@hydra.main(config_path="config", config_name="chesslesson_agent", version_base=None)
def main(config):
    run_ppo(config)


if __name__ == "__main__":
    main()
