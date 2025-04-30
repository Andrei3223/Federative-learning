import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import wandb

# from src.datasets.data_utils import get_dataloaders
from src.trainer import BaseTrainer, FederativeTrainer
# from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.datasets import DataProcessor
from src.datasets import AmazonDataset, FederativeDataset

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
        # A
        data_proc = DataProcessor(
            config.trainer.domain_A.dataset["data_path"], "",
            min_hist_len=config.trainer.domain_A.dataset["min_hist_len"],
            use_file=False,
        )
        user_train, user_valid, user_test, usernum, itemnum = data_proc.preprocess_dataset(
            "data/preprocessed",
            config.trainer.domain_A.dataset["name"]
        )
        dataset = AmazonDataset(user_train, usernum, itemnum, 50)
        train_loader = torch.utils.data.DataLoader(
            dataset, shuffle=True,
            batch_size=config.trainer.domain_A.batch_size, num_workers=2, pin_memory=True
        )
        OmegaConf.set_struct(config, False)  
        config.model.user_num = usernum
        config.model.item_num = itemnum
        model = instantiate(config.model).to(device)

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = instantiate(config.trainer.domain_A.optimizer, params=trainable_params)
        lr_scheduler = instantiate(config.trainer.domain_A.lr_scheduler, optimizer=optimizer)

        # B
        data_proc_B = DataProcessor(
            config.trainer.domain_B.dataset["data_path"], "",
            min_hist_len=config.trainer.domain_B.dataset["min_hist_len"],
            use_file=False,
        )
        user_train_B, user_valid_B, user_test_B, usernum_B, itemnum_B = data_proc_B.preprocess_dataset(
            "data/preprocessed",
           config.trainer.domain_B.dataset["name"],
        )
        dataset_B = AmazonDataset(user_train_B, usernum_B, itemnum_B, 50)
        train_loader_B = torch.utils.data.DataLoader(
            dataset_B, shuffle=True,
            batch_size=config.trainer.domain_B.batch_size, num_workers=2, pin_memory=True
        )
        OmegaConf.set_struct(config, False)  
        config.model.user_num = usernum_B
        config.model.item_num = itemnum_B
        model_B = instantiate(config.model).to(device)

        trainable_params_B = filter(lambda p: p.requires_grad, model_B.parameters())
        optimizer_B = instantiate(config.trainer.domain_B.optimizer, params=trainable_params_B)
        lr_scheduler_B = instantiate(config.trainer.domain_B.lr_scheduler, optimizer=optimizer_B)

        # common
        federative_dataset = FederativeDataset(
            user_train, user_train_B,
            data_proc.user_map, data_proc_B.user_map,
            config.trainer.domain_A.max_len, config.trainer.domain_B.max_len)
        common_loader = torch.utils.data.DataLoader(
            federative_dataset, shuffle=True,
            batch_size=config.trainer.common_data_bs, num_workers=2, pin_memory=True
        )

        trainable_params_common = [
            {'params': trainable_params},
            {'params': trainable_params_B}
        ]
        optimizer_constructor = instantiate(
            config.trainer.approx.optimizer,
            _partial_=True
        )
        optimizer_frob = optimizer_constructor(trainable_params_common)
        lr_scheduler_frob = instantiate(config.trainer.approx.lr_scheduler, optimizer=optimizer_frob)
        
    else:  # basic model
        data_proc = DataProcessor(
            config.dataset["data_path"], "",
            min_hist_len=config.dataset["min_hist_len"],
        )
        user_train, user_valid, user_test, usernum, itemnum = data_proc.preprocess_dataset(
            "data/preprocessed",
            config.dataset["name"]
        )
        dataset = AmazonDataset(user_train, usernum, itemnum, 50)
        train_loader = torch.utils.data.DataLoader(
            dataset, shuffle=True,
            batch_size=config.trainer.batch_size, num_workers=2, pin_memory=True
        )

        # build model architecture, then print to console
        OmegaConf.set_struct(config, False)  
        config.model.user_num = usernum
        config.model.item_num = itemnum
        model = instantiate(config.model).to(device)

        # build optimizer, learning rate scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = instantiate(config.optimizer, params=trainable_params)
        lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)

    # wandb
    run = wandb.init(
        project=config.wandb.get("project"),
        entity=config.wandb.get("entity"),
        name=config.wandb.get("run_name"),
        # config=config.wandb.config 
    )

    if config.trainer.get("federative", False):
        trainer = FederativeTrainer(
            model,
            optimizer,
            lr_scheduler,
            dataset_A=[user_train, user_valid, user_test, usernum, itemnum],

            model_B=model_B,
            optimizer_B=optimizer_B,
            lr_scheduler_B=lr_scheduler_B,
            dataset_B=[user_train_B, user_valid_B, user_test_B, usernum_B, itemnum_B],

            config=config,
            criterion=loss_function,
            dataloaders={"train_A": train_loader, "train_B": train_loader_B, "common": common_loader},

            optimizer_frob=optimizer_frob,
            lr_scheduler_frob=lr_scheduler_frob,

            device=device,
            writer=run,
        )
    else:
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
            writer=run,
            # skip_oom=config.trainer.get("skip_oom", True),
        )

    trainer.train()

    run.finish()


if __name__ == "__main__":
    main()