from src.trainer import FederativeTrainer
from src.loss import calculate_2_wasserstein_dist

from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import os
import random
import copy


class BERTFederativeTrainer(FederativeTrainer):
    """
    Class for federative training of two models: A and B.
    """

    def __init__(
        self,
        model_A,
        optimizer_A,
        lr_scheduler_A,
        dataset_A,  #  = [train, valid, test, usernum, itemnum]

        model_B,
        optimizer_B,
        lr_scheduler_B,
        dataset_B,  #  = [train, valid, test, usernum, itemnum]

        config_json,
        criterion,
        dataloaders,

        optimizer_frob,
        lr_scheduler_frob,

        device,
        writer,

        idxs_common_A=None,  # user idxs from domain A which active on domain B
        idxs_common_B=None,
        **kwargs,
    ):
        super().__init__(
            model_A,
            optimizer_A,
            lr_scheduler_A,
            dataset_A,  #  = [train, valid, test, usernum, itemnum]

            model_B,
            optimizer_B,
            lr_scheduler_B,
            dataset_B,  #  = [train, valid, test, usernum, itemnum]

            config_json,
            criterion,
            dataloaders,

            optimizer_frob,
            lr_scheduler_frob,

            device,
            writer,

            idxs_common_A=None,  # user idxs from domain A which active on domain B
            idxs_common_B=None,
            **kwargs,
        )
        self.bert_evaluation = True
    
    def train_epoch(self, train_config, model, train_dataloader, optimizer, lr_scheduler=None, name="A"):
        model.train()
        for batch in train_dataloader:
            seq, labels = batch["seq"].to(self.device), batch["labels"].to(self.device)
            logits, _ = model(seq) # (bs, t, vocab)
            logits = logits.view(-1, logits.size(-1)) # (bs * t, vocab)
            labels = labels.view(-1) 

            optimizer.zero_grad()
            loss = self.criterion(logits, labels)

            l2_coef = train_config["l2_emb"]
            for param in model.parameters():
                if param.requires_grad:
                    loss += l2_coef * torch.norm(param)
            loss.backward()
            # self._clip_grad_norm()
            optimizer.step()
            
            self.writer.log({f"train_loss_{name}":  loss.item()})
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        self.writer.log({f"learning_rate_{name}": lr})
        # self.writer.log({}, commit=True)

        optimizer.zero_grad()
    
    def approximate_epoch(self, emb_type="out_embed"):
        self.optimizer_A.zero_grad()
        self.optimizer_B.zero_grad()
    
        self.model_A.train()
        self.model_B.train()
        loss_function = self.config_trainer.approx.get("loss_function", "Frobenius")
        for batch in self.common_loader:
            self.optimizer_frob.zero_grad()
            
            seq_a, seq_b = batch["seq_A"], batch["seq_B"]
            values_a = torch.full((seq_a.size(0), 1), self.dataset_A[-1] + 1)
            values_b = torch.full((seq_b.size(0), 1), self.dataset_B[-1] + 1)
            seq_a = torch.cat([seq_a, values_a], dim=1)[..., -self.max_len_A:].to(self.device)
            seq_b = torch.cat([seq_b, values_b], dim=1)[..., -self.max_len_B:].to(self.device)
            # print(f"{seq_a.shape=}")
            
            _, emb_a = self.model_A(seq_a)
            _, emb_b = self.model_B(seq_b)
            emb_a = emb_a[:, -1, :]
            emb_b = emb_b[:, -1, :]
            # print(f"{emb_a.shape=}")
            

            if loss_function == "Frobenius":
                approx_loss = torch.norm(emb_a - emb_b, p='fro')
            elif loss_function == "Wasserstein":
                approx_loss = calculate_2_wasserstein_dist(emb_a, emb_b, device=self.device)
            else:
                raise NameError(f"Loss should be Frobenius or Wasserstein")
            approx_loss.backward()
            self.optimizer_frob.step()

            self.writer.log({f"loss_{loss_function}_{emb_type}":  approx_loss.item()})
        
        if self.lr_scheduler_frob is not None:
            self.lr_scheduler_frob.step()
        lr = self.optimizer_frob.param_groups[0]['lr']
        self.writer.log({"learning_rate_approx": lr})
        # self.writer.log({}, commit=True)

        self.optimizer_frob.zero_grad()
