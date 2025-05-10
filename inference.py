import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import wandb

from src.trainer import BaseTrainer, FederativeTrainer
from src.datasets import DataProcessor
from src.datasets import AmazonDataset, FederativeDataset

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference.
    Same datasets as in train are used.
    Args:
        config (DictConfig): hydra inference config.
    """
    project_config = OmegaConf.to_container(config)
    OmegaConf.set_struct(config, False)  

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
        config.model.user_num = usernum
        config.model.item_num = itemnum
        model = instantiate(config.model).to(device)
        checkpoint = torch.load(config.trainer.domain_A.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

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
        config.model.user_num = usernum_B
        config.model.item_num = itemnum_B

        model_B = instantiate(config.model).to(device)
        checkpoint = torch.load(config.trainer.domain_B.model_path, map_location=device, weights_only=False)
        model_B.load_state_dict(checkpoint['state_dict'])

        # common
        federative_dataset = FederativeDataset(
            user_train, user_train_B,
            data_proc.user_map, data_proc_B.user_map,
            config.trainer.domain_A.max_len, config.trainer.domain_B.max_len
        )
        idxs_common_A, idxs_common_B = federative_dataset.idxs_A, federative_dataset.idxs_B 
        common_loader = torch.utils.data.DataLoader(
            federative_dataset, shuffle=True,
            batch_size=config.trainer.common_data_bs, num_workers=2, pin_memory=True
        )
        
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
        config.model.user_num = usernum
        config.model.item_num = itemnum
        model = instantiate(config.model).to(device)
        checkpoint = torch.load(config.trainer.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])


    if config.trainer.get("federative", False):
        trainer = FederativeTrainer(
            model,
            None, None,
            dataset_A=[user_train, user_valid, user_test, usernum, itemnum],
            model_B=model_B,
            optimizer_B=None,lr_scheduler_B=None,
            dataset_B=[user_train_B, user_valid_B, user_test_B, usernum_B, itemnum_B],

            config=config,
            criterion=None,
            dataloaders={"train_A": train_loader, "train_B": train_loader_B, "common": common_loader},
            optimizer_frob=None, lr_scheduler_frob=None,

            idxs_common_A=idxs_common_A,
            idxs_common_B=idxs_common_B,

            device=device,
            writer=None,
        )
    else:
        # TODO base inference
        trainer = BaseTrainer(
            model=model,
            criterion=None,
            optimizer=None,
            lr_scheduler=None,
            config=config,
            device=device,
            dataloaders={"train": train_loader},
            dataset=[user_train, user_valid, user_test, usernum, itemnum],
            train_dataset=dataset,
            writer=None,

        )

    trainer.inference(dataset_common=federative_dataset)


if __name__ == "__main__":
    main()