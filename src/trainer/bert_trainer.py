import copy
import numpy as np
import torch
import random
from tqdm import tqdm
import sys
from pathlib import Path
import os
from src.trainer import BaseTrainer


class BERTTrainer(BaseTrainer):
    """
    Bert training class.
    """
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        config_json,
        device,
        dataloaders,
        dataset,  #  = [train, valid, test, usernum, itemnum]
        writer,
        **kwargs,
    ):
        super().__init__(
            model,
            criterion,
            optimizer,
            lr_scheduler,
            config_json,
            device,
            dataloaders,
            dataset,
            writer,
            **kwargs,
        )
    
    def train(self):
        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            self.model.train()
            for batch_idx, batch in enumerate(self.train_dataloader):
                seq, labels = batch["seq"].to(self.device), batch["labels"].to(self.device)
                logits = self.model(seq) # (bs, t, vocab)
                logits = logits.view(-1, logits.size(-1)) # (bs * t, vocab)
                labels = labels.view(-1) 

                self.optimizer.zero_grad()
                loss = self.criterion(logits, labels)

                l2_coef = self.cfg_trainer["l2_emb"]
                for param in self.model.parameters():
                    if param.requires_grad:
                        loss += l2_coef * torch.norm(param)
                loss.backward()
                # self._clip_grad_norm()
                self.optimizer.step()
                
                self.writer.log({"train_loss":  loss.item()})
            if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.log({"learning_rate": lr})

            if epoch % self.cfg_trainer.get("val_freq", 10) == 0:
                self.model.eval()

                NDCG, HT = self.evaluate_valid(self.model, self.dataset, self.max_len, bert_evaluation=True)

                self.writer.log({"NDCG": NDCG, "HT": HT})
                print(f"validation on {epoch=}: {NDCG=}, {HT=}")
            
            if epoch % self.cfg_trainer.get("save_freq", 20) == 0:
                self._save_checkpoint(epoch=epoch)

