import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import wandb

from src.trainer import BaseTrainer, FederativeTrainer
from src.datasets import DataProcessor
from src.datasets import AmazonDataset, FederativeDataset
from src.trainer.utils import setup_domain_data, setup_common_data

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
        if config.trainer.get("only_common_users", False):
            temp_data_proc_A = DataProcessor(
                config.trainer.domain_A.dataset["data_path"], "",
                min_hist_len=config.trainer.domain_A.dataset["min_hist_len"],
                use_file=False,
            )
            temp_user_train_A, _, _, _, _ = temp_data_proc_A.preprocess_dataset(
                "data/preprocessed",
                config.trainer.domain_A.dataset["name"]
            )
            
            temp_data_proc_B = DataProcessor(
                config.trainer.domain_B.dataset["data_path"], "",
                min_hist_len=config.trainer.domain_B.dataset["min_hist_len"],
                use_file=False,
            )
            temp_user_train_B, _, _, _, _ = temp_data_proc_B.preprocess_dataset(
                "data/preprocessed",
                config.trainer.domain_B.dataset["name"]
            )
            
            # Find common users
            temp_federative_dataset = FederativeDataset(
                temp_user_train_A, temp_user_train_B,
                temp_data_proc_A.user_map, temp_data_proc_B.user_map,
                config.trainer.domain_A.max_len, config.trainer.domain_B.max_len
            )
            common_user_ids = temp_federative_dataset.common_ids
            print(f"Number of common users: {len(common_user_ids)}")
            
            # Setup domain A with common users only
            domain_A_data = setup_domain_data(
                config, 
                device,
                data_path=config.trainer.domain_A.dataset["data_path"],
                min_hist_len=config.trainer.domain_A.dataset["min_hist_len"],
                dataset_name=config.trainer.domain_A.dataset["name"],
                batch_size=config.trainer.domain_A.batch_size,
                user_idxs=common_user_ids,
                use_optimizer=False,
            )
            
            # Setup domain B with common users only and shared user embeddings
            domain_B_data = setup_domain_data(
                config, 
                device,
                data_path=config.trainer.domain_B.dataset["data_path"],
                min_hist_len=config.trainer.domain_B.dataset["min_hist_len"],
                dataset_name=config.trainer.domain_B.dataset["name"],
                batch_size=config.trainer.domain_B.batch_size,
                user_idxs=common_user_ids,
                base_model=domain_A_data["model"],
                 use_optimizer=False,
            )
            
            # Setup common data
            common_data = setup_common_data(
                config,
                domain_A_data["user_train"], 
                domain_B_data["user_train"],
                domain_A_data["data_proc"].user_map, 
                domain_B_data["data_proc"].user_map,
                config.trainer.domain_A.max_len, 
                config.trainer.domain_B.max_len,
                config.trainer.common_data_bs
            )
            federative_dataset, common_loader = common_data["dataset"], common_data["loader"]
            idxs_common_A, idxs_common_B = common_data["idxs_A"], common_data["idxs_B"]

            model, train_loader = domain_A_data["model"], domain_A_data["loader"]
            user_train, user_valid, user_test, usernum, itemnum = domain_A_data["user_train"], domain_A_data["user_valid"], domain_A_data["user_test"], domain_A_data["usernum"], domain_A_data["itemnum"]
            checkpoint = torch.load(config.trainer.domain_A.model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
    
            model_B, train_loader_B = domain_B_data["model"], domain_B_data["loader"]
            user_train_B, user_valid_B, user_test_B, usernum_B, itemnum_B = domain_B_data["user_train"], domain_B_data["user_valid"], domain_B_data["user_test"], domain_B_data["usernum"], domain_B_data["itemnum"]
            checkpoint = torch.load(config.trainer.domain_B.model_path, map_location=device, weights_only=False)
            model_B.load_state_dict(checkpoint['state_dict'])

        else:
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
            # dataset = AmazonDataset(user_train, usernum, itemnum, 50)
            dataset = instantiate(
                config.dataset, 
                user_train=user_train,
                usernum=usernum, 
                itemnum=itemnum,
            )
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
            # dataset_B = AmazonDataset(user_train_B, usernum_B, itemnum_B, 50)
            dataset_B = instantiate(
                config.dataset, 
                user_train=user_train_B,
                usernum=usernum_B, 
                itemnum=itemnum_B,
            )
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
        trainer_partial = instantiate(
            config.trainer,
            _partial_=True  # This tells Hydra not to instantiate the object yet
        )
        trainer = trainer_partial(
            model_A=model,
            optimizer_A=None,
            lr_scheduler_A=None,
            dataset_A=[user_train, user_valid, user_test, usernum, itemnum],
            model_B=model_B,
            optimizer_B=None,lr_scheduler_B=None,
            dataset_B=[user_train_B, user_valid_B, user_test_B, usernum_B, itemnum_B],
            config_json=config,
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