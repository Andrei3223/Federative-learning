import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import wandb

# from src.datasets.data_utils import get_dataloaders
from src.trainer import (
    setup_basic_training,
    setup_federative_training_common_users,
    setup_federative_training,
)

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    # set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # prepare data for basic or federative experiment
    if config.trainer.get("federative", False):
        if config.trainer.get("only_common_users", False):
            trainer = setup_federative_training_common_users(config, device)
        else:
            trainer = setup_federative_training(config, device)
    else:
        trainer = setup_basic_training(config, device)

    #load model
    if "domain_A" in config.trainer and "checkpoint" in config.trainer.domain_A:
        trainer._resume_checkpoint(config.trainer.domain_A.checkpoint, domain="A")
        trainer._resume_checkpoint(config.trainer.domain_B.checkpoint, domain="B")
    # wandb
    run = wandb.init(
        project=config.wandb.get("project"),
        entity=config.wandb.get("entity"),
        name=config.wandb.get("run_name"),
        config=project_config
    )
    trainer.writer = run
    trainer.train()
    run.finish()


if __name__ == "__main__":
    main()