import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import wandb

# from src.datasets.data_utils import get_dataloaders
from src.trainer import BaseTrainer
# from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.datasets import DataProcessor
from src.datasets import AmazonDataset

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
    # logger = setup_saving_and_logging(config)
    # writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device

    if config.train.federative:
        pass
        # TODO 2 dataproc, two sets of users, common set of users => third dataproc and loader 
    data_proc = DataProcessor("data/Office_Products_5.json", "", min_hist_len=4)

    user_train, user_valid, user_test, usernum, itemnum = data_proc.preprocess_dataset("data/preprocessed", "office_4")

    dataset = AmazonDataset(user_train, usernum, itemnum, 50)
    train_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True,
        batch_size=config.trainer.batch_size, num_workers=2, pin_memory=True)

    # build model architecture, then print to console
    OmegaConf.set_struct(config, False)  
    config.model.user_num = usernum
    config.model.item_num = itemnum
    model = instantiate(config.model).to(device)
    # logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # wandb
    run = wandb.init(
        project=config.wandb.get("project"),
        entity=config.wandb.get("entity"),
        name=config.wandb.get("run_name"),
        # mode=config.wandb.get("mode"),
        # config=config.wandb.config 
    )

    # TODO new trainer: new params, 3 loaders, new logging 
    trainer = BaseTrainer(
        model=model,
        criterion=loss_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders={"train": train_loader},
        dataset=[user_train, user_valid, user_test, usernum, itemnum],
        train_dataset=dataset,
        # logger=logger,
        writer=run,
        # skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()

    run.finish()


if __name__ == "__main__":
    main()