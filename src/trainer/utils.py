import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.datasets import DataProcessor
from src.datasets import (
    FederativeDataset,
)


def setup_basic_training(config, device):
    """Set up training environment for basic (non-federative) training."""
    OmegaConf.set_struct(config, False)
    # Process dataset
    data_proc = DataProcessor(
        config.dataset["data_path"], "",
        min_hist_len=config.dataset["min_hist_len"],
    )
    user_train, user_valid, user_test, usernum, itemnum = data_proc.preprocess_dataset(
        "data/preprocessed",
        config.dataset["name"]
    )
    dataset = instantiate(
        config.dataset, 
        user_train=user_train,
        usernum=usernum, 
        itemnum=itemnum,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True,
        batch_size=config.trainer.batch_size, num_workers=2, pin_memory=True
    )

    # Build model 
    config.model.user_num = usernum
    config.model.item_num = itemnum
    model = instantiate(config.model).to(device)

    # Configure optimizer and scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    
    # Initialize loss function
    loss_function = instantiate(config.loss_function).to(device)

    # Create trainer
    trainer_partial = instantiate(
        config.trainer,
        _partial_=True  # This tells Hydra not to instantiate the object yet
    )
    trainer = trainer_partial(
        model=model,
        criterion=loss_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config_json=config,
        device=device,
        dataloaders={"train": train_loader},
        dataset=[user_train, user_valid, user_test, usernum, itemnum],
        train_dataset=dataset,
        writer=None
    )
    
    print(model)
    return trainer


def setup_federative_training(config, device):
    """Set up training environment for federative learning."""
    # Setup domain A
    domain_A_data = setup_domain_data(
        config, 
        device,
        data_path=config.trainer.domain_A.dataset["data_path"],
        min_hist_len=config.trainer.domain_A.dataset["min_hist_len"],
        dataset_name=config.trainer.domain_A.dataset["name"],
        batch_size=config.trainer.domain_A.batch_size,
        user_idxs=None
    )
    
    # Setup domain B with shared user embeddings
    domain_B_data = setup_domain_data(
        config, 
        device,
        data_path=config.trainer.domain_B.dataset["data_path"],
        min_hist_len=config.trainer.domain_B.dataset["min_hist_len"],
        dataset_name=config.trainer.domain_B.dataset["name"],
        batch_size=config.trainer.domain_B.batch_size,
        user_idxs=None,
        base_model=domain_A_data["model"]
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
    
    # Setup Frobenius optimizer for joint training
    optimizer_frob, lr_scheduler_frob = setup_frobenius_optimizer(
        config,
        domain_A_data["trainable_params"],
        domain_B_data["trainable_params"]
    )
    
    # Initialize loss function
    loss_function = instantiate(config.loss_function).to(device)
    
    # Create federative trainer
    trainer_partial = instantiate(
        config.trainer,
        _partial_=True  # This tells Hydra not to instantiate the object yet
    )
    trainer = trainer_partial(
        model_A=domain_A_data["model"],
        optimizer_A=domain_A_data["optimizer"],
        lr_scheduler_A=domain_A_data["lr_scheduler"],
        dataset_A=[
            domain_A_data["user_train"], 
            domain_A_data["user_valid"], 
            domain_A_data["user_test"], 
            domain_A_data["usernum"], 
            domain_A_data["itemnum"]
        ],
        model_B=domain_B_data["model"],
        optimizer_B=domain_B_data["optimizer"],
        lr_scheduler_B=domain_B_data["lr_scheduler"],
        dataset_B=[
            domain_B_data["user_train"], 
            domain_B_data["user_valid"], 
            domain_B_data["user_test"], 
            domain_B_data["usernum"], 
            domain_B_data["itemnum"]
        ],
        config_json=config,
        criterion=loss_function,
        dataloaders={
            "train_A": domain_A_data["loader"], 
            "train_B": domain_B_data["loader"], 
            "common": common_data["loader"]
        },
        optimizer_frob=optimizer_frob,
        lr_scheduler_frob=lr_scheduler_frob,
        idxs_common_A=common_data["idxs_A"],
        idxs_common_B=common_data["idxs_B"],
        device=device,
        writer=None,  # Will be set in main
    )

    return trainer


def setup_federative_training_common_users(config, device):
    """Set up federative training with only common users."""
    # Get common users first
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
        user_idxs=common_user_ids
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
        base_model=domain_A_data["model"]
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
    
    # Setup Frobenius optimizer for joint training
    optimizer_frob, lr_scheduler_frob = setup_frobenius_optimizer(
        config,
        domain_A_data["trainable_params"],
        domain_B_data["trainable_params"]
    )
    
    # Initialize loss function
    loss_function = instantiate(config.loss_function).to(device)
    
    # Create federative trainer
    trainer_partial = instantiate(
        config.trainer,
        _partial_=True  # This tells Hydra not to instantiate the object yet
    )
    trainer = trainer_partial(
        model_A=domain_A_data["model"],
        optimizer_A=domain_A_data["optimizer"],
        lr_scheduler_A=domain_A_data["lr_scheduler"],
        dataset_A=[
            domain_A_data["user_train"], 
            domain_A_data["user_valid"], 
            domain_A_data["user_test"], 
            domain_A_data["usernum"], 
            domain_A_data["itemnum"]
        ],
        model_B=domain_B_data["model"],
        optimizer_B=domain_B_data["optimizer"],
        lr_scheduler_B=domain_B_data["lr_scheduler"],
        dataset_B=[
            domain_B_data["user_train"], 
            domain_B_data["user_valid"], 
            domain_B_data["user_test"], 
            domain_B_data["usernum"], 
            domain_B_data["itemnum"]
        ],
        config_json=config,
        criterion=loss_function,
        dataloaders={
            "train_A": domain_A_data["loader"], 
            "train_B": domain_B_data["loader"], 
            "common": common_data["loader"]
        },
        optimizer_frob=optimizer_frob,
        lr_scheduler_frob=lr_scheduler_frob,
        idxs_common_A=common_data["idxs_A"],
        idxs_common_B=common_data["idxs_B"],
        device=device,
        writer=None,  # Will be set in main
    )
    
    return trainer


def setup_domain_data(config, device, data_path, min_hist_len, dataset_name, batch_size, user_idxs=None, base_model=None, use_optimizer=True):
    """Set up data, model, optimizer, and scheduler for a domain."""
    data_proc = DataProcessor(
        data_path, "",
        min_hist_len=min_hist_len,
        use_file=False,
    )
    
    user_train, user_valid, user_test, usernum, itemnum = data_proc.preprocess_dataset(
        "data/preprocessed",
        dataset_name,
        user_idxs=user_idxs,
    )
    dataset = instantiate(
        config.dataset, 
        user_train=user_train,
        usernum=usernum, 
        itemnum=itemnum,
    )
    loader = torch.utils.data.DataLoader(
        dataset, shuffle=True,
        batch_size=batch_size, num_workers=2, pin_memory=True
    )
    
    # Build model
    OmegaConf.set_struct(config, False)  
    config.model.user_num = usernum
    config.model.item_num = itemnum
    
    if base_model is None:
        model = instantiate(config.model).to(device)
    else:
        model = instantiate(config.model).to(device)
        # Copy user embeddings from base model, but not item embeddings
        filtered_state_dict = {k: v for k, v in base_model.state_dict().items() 
                               if ('item_emb' not in k) and ('embedding' not in k) and ('out' not in k)}
        model.load_state_dict(filtered_state_dict, strict=False)
    
    # Configure optimizer and scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    if use_optimizer:
        if "domain_A" in dataset_name:
            optimizer_partial = instantiate(config.trainer.domain_A.optimizer, _partial_=True)
            optimizer = optimizer_partial(params=trainable_params)
            
            lr_scheduler_partial = instantiate(config.trainer.domain_A.lr_scheduler, _partial_=True)
            lr_scheduler = lr_scheduler_partial(optimizer=optimizer)
        else:
            optimizer_partial = instantiate(config.trainer.domain_B.optimizer, _partial_=True)
            optimizer = optimizer_partial(params=trainable_params)
            
            lr_scheduler_partial = instantiate(config.trainer.domain_B.lr_scheduler, _partial_=True)
            lr_scheduler = lr_scheduler_partial(optimizer=optimizer)
    else:
        lr_scheduler, optimizer = None, None
    return {
        "data_proc": data_proc,
        "user_train": user_train,
        "user_valid": user_valid,
        "user_test": user_test,
        "usernum": usernum,
        "itemnum": itemnum,
        "dataset": dataset,
        "loader": loader,
        "model": model,
        "trainable_params": trainable_params,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler
    }


def setup_common_data(config, user_train_A, user_train_B, user_map_A, user_map_B, max_len_A, max_len_B, batch_size):
    """Set up data for common users between domains."""
    federative_dataset = FederativeDataset(
        user_train_A, user_train_B,
        user_map_A, user_map_B,
        max_len_A, max_len_B
    )
    
    print(f"Number of common users: {len(federative_dataset)}")
    
    common_loader = torch.utils.data.DataLoader(
        federative_dataset, shuffle=True,
        batch_size=batch_size, num_workers=2, pin_memory=True
    )
    
    return {
        "dataset": federative_dataset,
        "loader": common_loader,
        "idxs_A": federative_dataset.idxs_A,
        "idxs_B": federative_dataset.idxs_B,
    }


def setup_frobenius_optimizer(config, trainable_params_A, trainable_params_B):
    """Set up optimizer and scheduler for joint training."""
    trainable_params_common = [
        {'params': trainable_params_A},
        {'params': trainable_params_B}
    ]
    optimizer_partial = instantiate(
        config.trainer.approx.optimizer,
        _partial_=True
    )
    optimizer_frob = optimizer_partial(params=trainable_params_common)
    
    lr_scheduler_partial = instantiate(
        config.trainer.approx.lr_scheduler,
        _partial_=True
    )
    lr_scheduler_frob = lr_scheduler_partial(optimizer=optimizer_frob)
    
    return optimizer_frob, lr_scheduler_frob
