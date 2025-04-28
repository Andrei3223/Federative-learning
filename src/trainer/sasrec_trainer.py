import copy
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import random
from tqdm import tqdm
import sys
from pathlib import Path


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        # metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        dataset,
        # logger,
        writer,
        **kwargs,
    ):
        self.config = config
        self.cfg_trainer = self.config["trainer"]
        self.max_len = self.cfg_trainer["max_len"]

        self.device = device

        # self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.writer = writer

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_dataloader = dataloaders["train"]
        # self.val_dataloader = dataloaders["val"]

        self.dataset = dataset

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # define metrics
        # self.metrics = metrics

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "models"))
    
    def train(self):
        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            for batch_idx, batch in enumerate(
                # tqdm(self.train_dataloader, desc="train")
                self.train_dataloader
            ):
                u, seq, pos, neg = batch["user"], batch["seq"], batch["pos"], batch["neg"]
                # print(batch)q
                pos_logits, neg_logits = self.model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape, device=self.device)
                # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0


                self.optimizer.zero_grad()

                indices = np.where(pos != 0)

                loss = self.criterion(pos_logits[indices], pos_labels[indices])
                loss += self.criterion(neg_logits[indices], neg_labels[indices])

                l2_coef = self.cfg_trainer["l2_emb"]
                for param in self.model.parameters():
                    if param.requires_grad:
                        loss += l2_coef * torch.norm(param) 
                # for param in self.model.item_emb.parameters(): loss += l2_coef * torch.norm(param)
                # for param in self.model.abs_pos_K_emb.parameters(): loss += l2_coef * torch.norm(param)
                # for param in self.model.abs_pos_V_emb.parameters(): loss += l2_coef * torch.norm(param)
                # for param in self.model.time_matrix_K_emb.parameters(): loss += l2_coef * torch.norm(param)
                # for param in self.model.time_matrix_V_emb.parameters(): loss += l2_coef * torch.norm(param)

                loss.backward()
                
                # self._clip_grad_norm()
                self.optimizer.step()
                
                self.writer.log({"train_loss":  loss.item()})

                # print("loss in epoch {} iteration {}: {}".format(epoch, batch_idx, loss.item())) # expected 0.4~0.6 after init few epochs

            if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.log({"learning_rate": lr})

            # print(epoch, self.cfg_trainer.get("val_freq", 10))
            if epoch % self.cfg_trainer.get("val_freq", 10) == 0:
                self.model.eval()

                NDCG, HT = self.evaluate_valid(self.model, self.dataset, self.max_len)

                self.writer.log({"NDCG": NDCG, "HT": HT})
                print(f"validation on {epoch=}: {NDCG=}, {HT=}")
            
            if epoch % self.cfg_trainer.get("save_freq", 20) == 0:
                self._save_checkpoint(epoch=epoch)


    
    def evaluate_valid(self, model, dataset, maxlen):
        """
        Evaluates a PyTorch recommendation model on validation data
        
        Args:
            model: PyTorch model
            dataset: Dataset containing train, valid, test sets and user/item counts
            maxlen: maxlen 
        
        Returns:
            NDCG@10 and HR@10 metrics
        """
        print("evaluating")
        model.eval()
        
        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

        NDCG = 0.0
        valid_user = 0.0
        HT = 0.0
        
        # Sample users if there are too many
        if usernum > 10000:
            users = random.sample(range(1, usernum + 1), 10000)
        else:
            users = range(1, usernum + 1)
        
        for u in tqdm(users, desc="Evaluating"):
            if len(train[u]) < 1 or len(valid[u]) < 1: 
                continue

            # Create sequence from user history
            seq = np.zeros(maxlen, dtype=np.int64)
            idx = maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: 
                    break

            # Create set of items user has already interacted with
            rated = set(train[u])
            rated.add(0)
            
            # Add positive item (from validation) with 100 negative samples
            item_idx = [valid[u][0]]  # check index
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: 
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            # print(len(item_idx))
            # seq_tensor = torch.LongTensor([seq]).to(self.device)
            # item_tensor = torch.LongTensor(item_idx).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                # predictions = model.predict(u, seq_tensor, item_idx)
                # predictions = model.predict(u, seq, item_idx)
                predictions = -model.predict(*[np.array(l) for l in [[u], [seq],item_idx]])

                # print(predictions.shape)
                predictions = predictions[0]

                rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            # Calculate metrics
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

        # print(predictions)
        # Compute final metrics
        if valid_user > 0:
            NDCG = NDCG / valid_user
            HT = HT / valid_user
        else:
            NDCG = 0
            HT = 0

        return NDCG, HT
    
    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"{self.config.wandb.run_name}checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.wandb.log_checkpoints:
                self.writer.save(filename, str(self.checkpoint_dir.parent))
            # self.logger.info(f"Saving checkpoint: {filename} ...")
            print(f"Saving checkpoint: {filename} ...")
        # if save_best:
        #     best_path = str(self.checkpoint_dir / "model_best.pth")
        #     torch.save(state, best_path)
        #     if self.config.writer.log_checkpoints:
        #         self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
        #     self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
    
    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()
